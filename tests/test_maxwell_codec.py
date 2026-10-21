import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from numcodecs.registry import get_codec

from lindi import LindiH5pyFile, LindiRangeReadOptions
from lindi.maxwell_codec import MaxwellMxwDataCodec, register_maxwell_codec


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

