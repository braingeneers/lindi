"""Explicit NumCodecs support for Maxwell's native HDF5 filter 401."""

from __future__ import annotations

import ctypes
import os
import sys
import threading
from functools import lru_cache
from pathlib import Path

import numpy as np
from numcodecs.abc import Codec
from numcodecs.registry import register_codec


_plugin_paths: dict[tuple[str | None, str], str] = {}
_plugin_paths_lock = threading.Lock()


def _reset_plugin_paths_after_fork() -> None:
    global _plugin_paths, _plugin_paths_lock
    # The parent may have forked while another thread held the lock.
    _plugin_paths_lock = threading.Lock()
    _plugin_paths = {}


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_plugin_paths_after_fork)


def _resolve_plugin_path(plugin_path: str | None) -> str:
    plugin_path = plugin_path or None
    search_path = "" if plugin_path else os.environ.get("HDF5_PLUGIN_PATH", "")
    key = (plugin_path, search_path)
    with _plugin_paths_lock:
        resolved = _plugin_paths.get(key)
        if resolved is not None:
            return resolved

        if plugin_path:
            candidates = [Path(plugin_path).expanduser()]
        else:
            candidates = [Path(directory) for directory in search_path.split(os.pathsep) if directory]
            candidates.append(Path.home() / "hdf5_plugin_path_maxwell")
        for candidate in candidates:
            if candidate.is_dir():
                candidate = candidate / "libcompression.so"
            if candidate.is_file():
                resolved = str(candidate.resolve())
                _plugin_paths[key] = resolved
                return resolved

        # A later call can succeed if the plugin is installed in the meantime.
        raise RuntimeError("Maxwell decoder unavailable; set HDF5_PLUGIN_PATH or plugin_path")


@lru_cache(maxsize=4)
def _native_decoder(plugin_path: str) -> tuple:
    """
    Load the native decoder and its matching memory release function

    Parameters
    -----------
    plugin_path: shared library path

    Returns
    --------
    loaded library, decoder and release function
    """
    library = ctypes.CDLL(plugin_path)
    decompress = library.decompress_samples
    decompress.argtypes = [
        ctypes.POINTER(ctypes.c_uint8), ctypes.c_size_t,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_uint16)),
    ]
    decompress.restype = ctypes.c_size_t
    release = ctypes.CDLL(None).free
    release.argtypes = [ctypes.c_void_p]
    release.restype = None
    return library, decompress, release


class MaxwellMxwDataCodec(Codec):
    codec_id = "maxwell-mxw-data"

    def __init__(
        self, hdf5_filter_id: int = 401, codec_version: int = 1,
        *, plugin_path: str | None = None,
    ):
        """
        Decode the workflow's versioned Maxwell compressed chunk format

        Parameters
        -----------
        hdf5_filter_id: expected maxwell hdf5 filter id
        codec_version: supported wire format version
        plugin_path: explicit shared library path or directory

        Each explicit path or HDF5_PLUGIN_PATH value is resolved on its first
        successful lookup in the process. Relative paths use the working
        directory at that lookup.
        """
        if hdf5_filter_id != 401 or codec_version != 1:
            raise ValueError("Unsupported Maxwell filter or codec version")
        self.plugin_path = plugin_path

    def get_config(self) -> dict:
        """
        Describe the portable wire format without machine-specific paths

        Returns
        --------
        versioned numcodecs configuration
        """
        return {"id": self.codec_id, "hdf5_filter_id": 401, "codec_version": 1}

    def encode(self, buf):
        """
        Reject encoding, which belongs to the native HDF5 writer

        Parameters
        -----------
        buf: decoded samples
        """
        raise NotImplementedError("Maxwell encoding requires the HDF5 plugin")

    def decode(self, buf, out=None):
        """
        Decode an exact compressed payload to little-endian uint16 bytes

        Parameters
        -----------
        buf: encoded chunk
        out: optional destination buffer

        Returns
        --------
        decoded bytes or the supplied destination
        """
        plugin = _resolve_plugin_path(self.plugin_path)
        compressed = memoryview(buf).cast("B").tobytes()
        if not compressed:
            raise ValueError("Cannot decode an empty Maxwell chunk")
        _library, decompress, release = _native_decoder(plugin)
        source = (ctypes.c_uint8 * len(compressed)).from_buffer_copy(compressed)
        pointer = ctypes.POINTER(ctypes.c_uint16)()
        count = int(decompress(source, len(compressed), ctypes.byref(pointer)))
        if not pointer:
            raise RuntimeError("Maxwell decoder returned no samples")
        try:
            if count <= 0:
                raise RuntimeError("Maxwell decoder returned an invalid sample count")
            values = np.ctypeslib.as_array(pointer, shape=(count,)).copy()
        finally:
            release(pointer)
        if sys.byteorder != "little":
            values = values.astype("<u2")
        decoded = values.tobytes()
        if out is None:
            return decoded
        destination = memoryview(out).cast("B")
        if len(destination) != len(decoded):
            raise ValueError("Maxwell destination size does not match decoded chunk")
        destination[:] = decoded
        return out


def register_maxwell_codec() -> None:
    """
    Register the optional native Maxwell decoder explicitly
    """
    register_codec(MaxwellMxwDataCodec)
