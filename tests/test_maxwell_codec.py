import ctypes
import os
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from textwrap import dedent

import h5py
import numpy as np
import pytest
from numcodecs.registry import get_codec

from lindi import LindiH5pyFile, LindiRangeReadOptions
from lindi import maxwell_codec
from lindi.maxwell_codec import MaxwellMxwDataCodec, register_maxwell_codec


@pytest.fixture
def decoder_paths(monkeypatch):
    paths = []
    samples = (ctypes.c_uint16 * 2)(42, 256)

    def decompress(source, length, output):
        pointer = ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)))
        pointer[0] = ctypes.cast(samples, ctypes.POINTER(ctypes.c_uint16))
        return len(samples)

    def native_decoder(path):
        paths.append(path)
        return None, decompress, lambda pointer: None

    monkeypatch.setattr(maxwell_codec, "_native_decoder", native_decoder)
    return paths


def test_plugin_path_resolved_once_across_threads_and_codecs(tmp_path, monkeypatch, decoder_paths):
    plugin = tmp_path / "libcompression.so"
    plugin.touch()
    monkeypatch.setenv("HDF5_PLUGIN_PATH", str(tmp_path))
    expected_path = str(plugin.resolve())
    checked_paths = []
    is_file = Path.is_file

    def count_plugin_checks(path):
        if path == plugin:
            checked_paths.append(path)
        return is_file(path)

    monkeypatch.setattr(Path, "is_file", count_plugin_checks)
    start = threading.Barrier(8, timeout=5)

    def decode_chunks():
        codec = MaxwellMxwDataCodec()
        start.wait()
        for _ in range(50):
            assert codec.decode(b"chunk") == b"\x2a\x00\x00\x01"

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda _: decode_chunks(), range(8)))

    assert checked_paths == [plugin]
    assert decoder_paths == [expected_path] * 400


@pytest.mark.parametrize("use_directory", [False, True])
def test_plugin_override_and_environment_change(tmp_path, monkeypatch, decoder_paths, use_directory):
    first = tmp_path / "first.so"
    second = tmp_path / "second.so"
    override_dir = tmp_path / "override"
    override_dir.mkdir()
    override = override_dir / "libcompression.so"
    for path in (first, second, override):
        path.touch()

    monkeypatch.setenv("HDF5_PLUGIN_PATH", os.pathsep.join([str(tmp_path / "missing"), str(first)]))
    default_codec = MaxwellMxwDataCodec()
    override_codec = MaxwellMxwDataCodec(plugin_path=str(override_dir if use_directory else override))
    default_codec.decode(b"chunk")
    override_codec.decode(b"chunk")
    monkeypatch.setenv("HDF5_PLUGIN_PATH", str(second))
    default_codec.decode(b"chunk")
    override_codec.decode(b"chunk")
    assert decoder_paths == [str(path.resolve()) for path in (first, override, second, override)]


def test_missing_plugin_error_and_retry(tmp_path, monkeypatch, decoder_paths):
    plugin = tmp_path / "libcompression.so"
    monkeypatch.setenv("HDF5_PLUGIN_PATH", str(tmp_path))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    codec = MaxwellMxwDataCodec()
    message = r"^Maxwell decoder unavailable; set HDF5_PLUGIN_PATH or plugin_path$"
    with pytest.raises(RuntimeError, match=message):
        codec.decode(b"chunk")

    plugin.touch()
    codec.decode(b"chunk")
    assert decoder_paths == [str(plugin.resolve())]
    with pytest.raises(RuntimeError, match=message):
        MaxwellMxwDataCodec(plugin_path=str(tmp_path / "missing")).decode(b"chunk")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_child_resolves_plugin_with_its_own_cache_and_lock(tmp_path):
    script = dedent("""
        import os
        import signal
        import sys
        from pathlib import Path

        from lindi import maxwell_codec

        directory = Path(sys.argv[1])
        first = directory / "first.so"
        second = directory / "second.so"
        first.touch()
        second.touch()
        plugin = directory / "libcompression.so"
        plugin.symlink_to(first)
        assert maxwell_codec._resolve_plugin_path(str(plugin)) == str(first.resolve())
        plugin.unlink()
        plugin.symlink_to(second)

        with maxwell_codec._plugin_paths_lock:
            pid = os.fork()
            if pid == 0:
                signal.alarm(5)
                resolved = maxwell_codec._resolve_plugin_path(str(plugin))
                os._exit(0 if resolved == str(second.resolve()) else 1)
            _, status = os.waitpid(pid, 0)
        assert status == 0
        assert maxwell_codec._resolve_plugin_path(str(plugin)) == str(first.resolve())
    """)
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr


def test_versioned_codec_registration():
    register_maxwell_codec()
    config = {"id": "maxwell-mxw-data", "hdf5_filter_id": 401, "codec_version": 1}
    assert get_codec(config).get_config() == config
    assert MaxwellMxwDataCodec(plugin_path="/machine-specific").get_config() == config
    with pytest.raises(ValueError, match="Unsupported"):
        MaxwellMxwDataCodec(codec_version=2)
    with pytest.raises(ValueError, match="Unsupported"):
        MaxwellMxwDataCodec(hdf5_filter_id=1)


def test_registration_is_explicit_in_a_clean_process():
    result = subprocess.run([
        sys.executable, "-c",
        "import lindi; from numcodecs.registry import get_codec; "
        "get_codec({'id': 'maxwell-mxw-data', 'hdf5_filter_id': 401, 'codec_version': 1})",
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert "not available" in result.stderr


@pytest.mark.skipif(not os.environ.get("HDF5_PLUGIN_PATH"), reason="native plugin not installed")
@pytest.mark.parametrize("enabled", [True, False])
def test_native_decoding_matches_hdf5_including_tail(tmp_path, enabled):
    register_maxwell_codec()
    assert h5py.h5z.filter_avail(401)
    path = tmp_path / "maxwell.nwb"
    values = np.broadcast_to(
        (512 + 15 * np.sin(np.arange(4003) / 12)).astype("<u2")[:, None], (4003, 3),
    ).copy()
    rfs = {"refs": {
        ".zgroup": {"zarr_format": 2}, "data/.zattrs": {},
        "data/.zarray": {
            "zarr_format": 2, "shape": list(values.shape), "chunks": [2000, 1],
            "dtype": "<u2", "order": "C", "fill_value": 0,
            "compressor": None, "filters": [MaxwellMxwDataCodec().get_config()],
        },
    }}
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("data", data=values, chunks=(2000, 1), compression=401)
        for time in range(0, 4003, 2000):
            for channel in range(3):
                info = dataset.id.get_chunk_info_by_coord((time, channel))
                assert info.filter_mask == 0
                rfs["refs"][f"data/{time // 2000}.{channel}"] = [
                    str(path), info.byte_offset, info.size,
                ]
    with LindiH5pyFile.from_reference_file_system(
        rfs, range_read_options=LindiRangeReadOptions(enabled=enabled),
    ) as reader, h5py.File(path) as direct:
        for start, end in [(0, 1), (1995, 2012), (3999, 4003), (0, 4003)]:
            np.testing.assert_array_equal(reader["data"][start:end], direct["data"][start:end])

