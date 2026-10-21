import base64

import numpy as np
import pytest
from numcodecs.abc import Codec
from numcodecs.registry import register_codec
from numcodecs import Zlib

from lindi import LindiH5pyFile, LindiRangeReadOptions
from lindi.LindiH5pyFile.LindiReferenceFileSystemStore import LindiReferenceFileSystemStore


class _ExactFilter(Codec):
    codec_id = "test-exact-length"

    def encode(self, buf):
        return b"encoded"

    def decode(self, buf, out=None):
        assert bytes(buf) == b"encoded", "encoded payload was padded before decode"
        return np.arange(16, dtype="<i2").tobytes()


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("inline", [True, False])
def test_filter_payload_is_not_padded_before_decode(tmp_path, enabled, inline):
    register_codec(_ExactFilter)
    path = tmp_path / "chunk.bin"
    path.write_bytes(b"encoded")
    reference = (
        "base64:" + base64.b64encode(b"encoded").decode()
        if inline else [str(path), 0, 7]
    )
    rfs = {"refs": {
        ".zgroup": {"zarr_format": 2},
        "data/.zattrs": {},
        "data/.zarray": {
            "zarr_format": 2, "shape": [16], "chunks": [16],
            "dtype": "<i2", "order": "C", "fill_value": 0,
            "compressor": None, "filters": [{"id": _ExactFilter.codec_id}],
        },
        "data/0": reference,
    }}
    with LindiH5pyFile.from_reference_file_system(
        rfs, range_read_options=LindiRangeReadOptions(enabled=enabled),
    ) as reader:
        np.testing.assert_array_equal(reader["data"][:], np.arange(16))


def test_compressor_payload_is_returned_byte_exact(tmp_path):
    encoded = Zlib().encode(np.zeros(1000, dtype="<i2"))
    path = tmp_path / "compressed.bin"
    path.write_bytes(encoded)
    store = LindiReferenceFileSystemStore({"refs": {
        "data/.zarray": {
            "shape": [1000], "chunks": [1000], "dtype": "<i2",
            "compressor": Zlib().get_config(), "filters": None,
        },
        "data/0": [str(path), 0, len(encoded)],
    }})
    try:
        assert store["data/0"] == encoded
        assert store.getitems(["data/0"], contexts={})["data/0"] == encoded
    finally:
        store.close()

