from dataclasses import dataclass
from typing import Any, Dict, Literal, Mapping, Sequence, Union
import os
import json
import time
import base64
import numpy as np
from zarr.storage import Store as ZarrStore

from ..LocalCache.LocalCache import ChunkTooLargeError, LocalCache
from ..range_reading import (
    ByteRange,
    LindiRangeReadOptions,
    LindiRangeReadMetrics,
    PlannedByteRangeRequest,
    RangeTransportFactory,
    _RangeTransportManager,
    fetch_byte_range_requests,
    plan_byte_range_requests,
)
from ..range_diagnostics import capture_range_read_metrics
from ..tar.lindi_tar import LindiTarFile


@dataclass(frozen=True)
class _ResolvedReference:
    key: str
    data: Union[bytes, None] = None
    url_or_path: Union[str, None] = None
    offset: int = 0
    length: int = 0

    @property
    def is_inline(self) -> bool:
        return self.data is not None


class LindiReferenceFileSystemStore(ZarrStore):
    """
    A Zarr store that reads data from a reference file system.

    The reference file system is based on the ReferenceFileSystem of fsspec, but
    this is a custom implementation that serves some LINDI-specific needs. In
    particular, it handles reading data from DANDI URLs, even when the file is
    part of an embargoed dataset. This requires some special handling as the
    DANDI API URL must be exchanged for a pre-signed S3 bucket URL by
    authenticating with a DANDI API token. This presigned URL expires after a
    period of time, so this Zarr store handles the renewal of the presigned URL.
    It also does the exchange once the first time and caches the redirected URL
    for a period so that the redirect doesn't need to be done every time a
    segment of a file is read.

    To read from a file in an embargoed DANDI dataset, you will need to set the
    DANDI_API_KEY environment variable to your DANDI API token. Or, if this is a
    Dandiset on the staging server, you will need to set the
    DANDI_STAGING_API_KEY.

    Following the fsspec convention
    (https://fsspec.github.io/kerchunk/spec.html), the reference file system is
    specified as a dictionary with a "refs" key. The value of "refs" is a
    dictionary where the keys are the names of the files and each value is
    either a string, a list, or a dict. If the value is a string, it is assumed
    to be the data of the file, which may be base64 encoded (see below). If the
    value is a list, it is assumed to have three elements: the URL of the file
    (or path of a local file), the byte offset of the data within the file, and
    the byte length of the data. Note that we do not permit the case of a list
    of a single (url) element supported by fsspec, because it is good to be able
    to know the size of the chunks without making a request to the file. If the
    value is a dict, it represents a json file, and the content of the file is
    the json representation of the dict.

    If the value for a file is a string, it may be prefixed with "base64:". If
    it is, the string is assumed to be base64 encoded and is decoded before
    being returned. Otherwise, the string is utf-8 encoded and returned as is.
    Note that a file that actually begins with "base64:" should be represented
    by a base64 encoded string, to avoid ambiguity.

    We also support the use of templates as in fsspec, but do not support the
    full jinja2 templating. There may be an optional "templates" key in the
    dictionary, which is a dictionary of template strings. For example, {
        "templates": {"u1": "https://some/url", "u2": "https://some/other/url"},
        "refs": {
            ... "/some/key/0": [
                "{{u1}}" 0, 100
            ],
            ...
        }
    } In this case, the "{{u1}}" will be replaced with the value of the "u1"
    template string.

    Optionally, the reference file system may contain a "generationMetadata"
    key, which is a dictionary of metadata about the generation of the reference
    file system. This metadata is not used by this class, but could be by other
    software. See LindiH5pyFile.write_lindi_file(...)

    It is okay for rfs to be modified outside of this class, and the changes
    will be reflected immediately in the store.
    """
    def __init__(
            self,
            rfs: dict,
            *,
            mode: Literal["r", "r+"] = "r+",
            local_cache: Union[LocalCache, None] = None,
            range_read_options: Union[LindiRangeReadOptions, None] = None,
            range_transports: Union[
                Mapping[str, RangeTransportFactory], None
            ] = None,
            _source_url_or_path: Union[str, None] = None,
            _source_tar_file: Union[LindiTarFile, None] = None
    ):
        """
        Create a LindiReferenceFileSystemStore.

        Parameters
        ----------
        rfs : dict
            The reference file system (see class docstring for details).
        mode : str
            The mode to open the store in. Only "r" is supported at this time.
        local_cache : LocalCache, optional
            The local cache to use for caching data chunks read from the
            remote URLs. If None, no caching is done.
        range_read_options : LindiRangeReadOptions, optional
            Coalescing, concurrency, retry, and instrumentation options for
            remote chunk references.
        """
        if "refs" not in rfs:
            raise Exception("rfs must contain a 'refs' key")

        # validate rfs['refs']
        for k, v in rfs["refs"].items():
            if isinstance(v, str):
                pass
            elif isinstance(v, dict):
                # the content of the file is the json representation of the dict
                pass
            elif isinstance(v, list):
                if len(v) != 3:
                    raise Exception(f"Problem with {k}: list must have 3 elements")
                if not isinstance(v[0], str):
                    raise Exception(f"Problem with {k}: first element must be a string")
                if not isinstance(v[1], int):
                    raise Exception(f"Problem with {k}: second element must be an int")
                if not isinstance(v[2], int):
                    raise Exception(f"Problem with {k}: third element must be an int")
            else:
                raise Exception(f"Problem with {k}: value must be a string or a list")

        # validate templates
        if "templates" in rfs:
            for k, v in rfs["templates"].items():
                if not isinstance(v, str):
                    raise Exception(f"Problem with templates: value for {k} must be a string")

        self.rfs = rfs
        self.mode = mode
        self.local_cache = local_cache
        self.range_read_options = range_read_options or LindiRangeReadOptions()
        self._range_transport_manager = _RangeTransportManager(range_transports)
        self._source_url_or_path = _source_url_or_path
        self._source_tar_file = _source_tar_file

    # These methods are overridden from MutableMapping
    def __contains__(self, key: object):
        if not isinstance(key, str):
            return False
        return key in self.rfs["refs"]

    def __getitem__(self, key: str):
        val = self._get_helper(key)

        if val is not None:
            padded_size = _get_padded_size(self, key, val)
            if padded_size is not None:
                val = _pad_chunk(val, padded_size)

        return val

    def _get_helper(self, key: str):
        reference = self._resolve_reference(key)
        if reference.is_inline:
            return reference.data
        return self._read_resolved_reference(reference)

    def _resolve_reference(self, key: str) -> _ResolvedReference:
        if key not in self.rfs["refs"]:
            raise KeyError(key)
        x = self.rfs["refs"][key]
        if isinstance(x, str):
            if x.startswith("base64:"):
                data = base64.b64decode(x[len("base64:"):])
            else:
                data = x.encode("utf-8")
            return _ResolvedReference(key=key, data=data)
        elif isinstance(x, dict):
            return _ResolvedReference(key=key, data=json.dumps(x).encode("utf-8"))
        elif isinstance(x, list):
            if len(x) != 3:
                raise Exception("list must have 3 elements")  # pragma: no cover
            url_or_path = x[0]
            offset = x[1]
            length = x[2]
            if '{{' in url_or_path and '}}' in url_or_path and 'templates' in self.rfs:
                for k, v in self.rfs["templates"].items():
                    url_or_path = url_or_path.replace("{{" + k + "}}", v)
            if url_or_path.startswith('./'):
                if self._source_url_or_path is None:
                    raise Exception(f"Cannot resolve relative path {url_or_path} without source file path")
                if self._source_tar_file is None:
                    raise Exception(f"Cannot resolve relative path {url_or_path} without source file type")
                if self._source_tar_file and (not self._source_tar_file._dir_representation):
                    start_byte, end_byte = self._source_tar_file.get_file_byte_range(file_name=url_or_path[2:])
                    if start_byte + offset + length > end_byte:
                        raise Exception(f"Chunk {key} is out of bounds in tar file {url_or_path}")
                    url_or_path = self._source_url_or_path
                    offset = offset + start_byte
                elif self._source_tar_file and self._source_tar_file._dir_representation:
                    fname = self._source_tar_file._tar_path_or_url + '/' + url_or_path[2:]
                    if not os.path.exists(fname):
                        raise Exception(f"File does not exist: {fname}")
                    file_size = os.path.getsize(fname)
                    if offset + length > file_size:
                        raise Exception(f"Chunk {key} is out of bounds in tar file {url_or_path}: {fname}")
                    url_or_path = fname
            return _ResolvedReference(
                key=key,
                url_or_path=url_or_path,
                offset=offset,
                length=length,
            )
        else:
            # should not happen given checks in __init__, but self.rfs is mutable
            # and contains mutable lists
            raise Exception(f"Problem with {key}: value {x} must be a string or a list")

    def _read_resolved_reference(self, reference: _ResolvedReference) -> bytes:
        if reference.url_or_path is None:
            raise ValueError("Resolved byte reference has no URL or path")
        transport = self._range_transport_manager.resolve(reference.url_or_path)
        cacheable = transport.capabilities.cacheable
        if cacheable and self.local_cache is not None:
            cached = self.local_cache.get_remote_chunk(
                url=reference.url_or_path,
                offset=reference.offset,
                size=reference.length,
            )
            if cached is not None:
                return cached
        if reference.length == 0:
            return b""
        byte_range = ByteRange(
            key=reference.key,
            source=reference.url_or_path,
            offset=reference.offset,
            length=reference.length,
        )
        request = PlannedByteRangeRequest(
            source=byte_range.source,
            offset=byte_range.offset,
            length=byte_range.length,
            ranges=(byte_range,),
            requested_bytes=byte_range.length,
            overread_bytes=0,
        )
        fetched = fetch_byte_range_requests(
            [request],
            self.range_read_options,
            _transport_manager=self._range_transport_manager,
        )
        data = fetched[0].data
        if cacheable:
            self._put_in_local_cache(reference, data)
        return data

    def _put_in_local_cache(
        self, reference: _ResolvedReference, data: bytes
    ) -> None:
        if self.local_cache is None or reference.url_or_path is None:
            return
        transport = self._range_transport_manager.resolve(reference.url_or_path)
        if not transport.capabilities.cacheable:
            return
        try:
            self.local_cache.put_remote_chunk(
                url=reference.url_or_path,
                offset=reference.offset,
                size=reference.length,
                data=data,
            )
        except ChunkTooLargeError:
            print(
                f"Warning: unable to cache chunk of size {reference.length} "
                f"on LocalCache (key: {reference.key})"
            )

    def getitems(
        self,
        keys: Sequence[str],
        *,
        contexts: Mapping[str, Any],
    ) -> Mapping[str, bytes]:
        """Retrieve all referenced chunks for one Zarr selection efficiently."""
        del contexts  # Zarr store contexts are not needed for bytes output.
        started = time.perf_counter()
        unique_keys = list(dict.fromkeys(keys))
        results: Dict[str, bytes] = {}
        references_by_key: Dict[str, _ResolvedReference] = {}
        source_ranges = []
        inline_value_count = 0
        range_reference_count = 0
        high_latency_reference_count = 0
        cache_hit_count = 0
        referenced_bytes = 0
        cache_hit_bytes = 0
        high_latency_keys = set()

        for key in unique_keys:
            try:
                reference = self._resolve_reference(key)
            except KeyError:
                continue
            if reference.is_inline:
                assert reference.data is not None
                results[key] = reference.data
                inline_value_count += 1
                continue
            if reference.url_or_path is None:
                raise ValueError("Resolved byte reference has no URL or path")
            transport = self._range_transport_manager.resolve(reference.url_or_path)
            capabilities = transport.capabilities
            range_reference_count += 1
            referenced_bytes += reference.length
            if capabilities.high_latency:
                high_latency_reference_count += 1
                high_latency_keys.add(key)
            if capabilities.cacheable and self.local_cache is not None:
                cached = self.local_cache.get_remote_chunk(
                    url=reference.url_or_path,
                    offset=reference.offset,
                    size=reference.length,
                )
                if cached is not None:
                    results[key] = cached
                    cache_hit_count += 1
                    cache_hit_bytes += reference.length
                    continue
            if reference.length == 0:
                results[key] = b""
                continue
            references_by_key[key] = reference
            source_ranges.append(
                ByteRange(
                    key=key,
                    source=reference.url_or_path,
                    offset=reference.offset,
                    length=reference.length,
                )
            )

        planned_requests = plan_byte_range_requests(
            source_ranges, self.range_read_options
        )
        fetched_requests = fetch_byte_range_requests(
            planned_requests,
            self.range_read_options,
            _transport_manager=self._range_transport_manager,
        )
        for fetched in fetched_requests:
            request = fetched.request
            for byte_range in request.ranges:
                relative_start = byte_range.offset - request.offset
                relative_end = relative_start + byte_range.length
                chunk_data = fetched.data[relative_start:relative_end]
                if len(chunk_data) != byte_range.length:
                    raise ValueError(
                        f"Unable to reconstruct {byte_range.key}: "
                        f"received {len(chunk_data)} bytes, expected {byte_range.length}"
                    )
                results[byte_range.key] = chunk_data
                self._put_in_local_cache(
                    references_by_key[byte_range.key], chunk_data
                )

        for key, value in list(results.items()):
            padded_size = _get_padded_size(self, key, value)
            if padded_size is not None:
                results[key] = _pad_chunk(value, padded_size)

        transport_attempt_counts: Dict[str, int] = {}
        transport_fetched_bytes: Dict[str, int] = {}
        for fetched in fetched_requests:
            transport_attempt_counts[fetched.transport_name] = (
                transport_attempt_counts.get(fetched.transport_name, 0)
                + fetched.attempt_count
            )
            transport_fetched_bytes[fetched.transport_name] = (
                transport_fetched_bytes.get(fetched.transport_name, 0)
                + fetched.fetched_bytes
            )
        metrics = LindiRangeReadMetrics(
            optimized=self.range_read_options.enabled,
            requested_key_count=len(unique_keys),
            returned_key_count=len(results),
            inline_value_count=inline_value_count,
            range_reference_count=range_reference_count,
            high_latency_reference_count=high_latency_reference_count,
            cache_hit_count=cache_hit_count,
            source_range_count=len(source_ranges),
            planned_request_count=len(planned_requests),
            read_attempt_count=sum(x.attempt_count for x in fetched_requests),
            requested_bytes=sum(x.requested_bytes for x in planned_requests),
            fetched_bytes=sum(x.fetched_bytes for x in fetched_requests),
            overread_bytes=sum(x.overread_bytes for x in planned_requests),
            elapsed_seconds=time.perf_counter() - started,
            referenced_bytes=referenced_bytes,
            cache_hit_bytes=cache_hit_bytes,
            transport_attempt_counts=tuple(sorted(transport_attempt_counts.items())),
            transport_fetched_bytes=tuple(sorted(transport_fetched_bytes.items())),
        )
        captured = capture_range_read_metrics(
            metrics,
            options=self.range_read_options,
            range_keys=tuple(high_latency_keys.intersection(results)),
        )
        if not captured and self.range_read_options.metrics_callback is not None:
            self.range_read_options.metrics_callback(metrics)
        return results

    def close(self) -> None:
        self._range_transport_manager.close()

    def __setitem__(self, key: str, value: bytes):
        # We intentionally do not allow value to be a dict here! When the rfs is
        # written to a .json file elsewhere in the codebase of lindi, the value
        # will automatically be converted to a json object if it is json
        # serializable.
        if not isinstance(value, bytes):
            raise ValueError("value must be bytes")
        try:
            # try to ascii encode the value
            value2 = value.decode("ascii")
        except UnicodeDecodeError:
            # if that fails, base64 encode it
            value2 = "base64:" + base64.b64encode(value).decode("ascii")
        self.rfs["refs"][key] = value2

    def __delitem__(self, key: str):
        del self.rfs["refs"][key]

    def __iter__(self):
        return iter(self.rfs["refs"])

    def __len__(self):
        return len(self.rfs["refs"])

    # These methods are overridden from BaseStore
    def is_readable(self):
        return self.mode in ["r", "r+"]

    def is_writeable(self):
        return self.mode in ["r+"]

    def is_listable(self):
        return True

    def is_erasable(self):
        return False

    @staticmethod
    def replace_meta_file_contents_with_dicts_in_rfs(rfs: dict) -> None:
        """
        Utility function for replacing the contents of the .zattrs, .zgroup, and
        .zarray files in an rfs with the json representation of the contents.
        """
        # important to use the LindiReferenceFileSystemStore here because then we
        # can resolve any base64 encoded values, etc when converting them to dicts
        store = LindiReferenceFileSystemStore(rfs)
        for k, v in rfs['refs'].items():
            if k.endswith('.zattrs') or k.endswith('.zgroup') or k.endswith('.zarray') or k.endswith('zarr.json'):  # note: zarr.json is for zarr v3
                rfs['refs'][k] = json.loads(store[k].decode('utf-8'))

    @staticmethod
    def use_templates_in_rfs(rfs: dict) -> None:
        """
        Utility for replacing URLs in an rfs with template strings. Only URLs
        that occur 5 or more times are replaced with template strings. The
        templates are added to the "templates" key of the rfs. The template
        strings are of the form "{{u1}}", "{{u2}}", etc.
        """
        url_counts: Dict[str, int] = {}
        for k, v in rfs['refs'].items():
            if isinstance(v, list):
                url = v[0]
                if '{{' not in url:
                    url_counts[url] = url_counts.get(url, 0) + 1
        urls_with_many_occurrences = sorted([url for url, count in url_counts.items() if count >= 5])
        new_templates = rfs.get('templates', {})
        template_names_for_urls: Dict[str, str] = {}
        for template_name, url in new_templates.items():
            template_names_for_urls[url] = template_name
        for url in urls_with_many_occurrences:
            if url in template_names_for_urls:
                continue
            i = 1
            while f'u{i}' in new_templates:
                i += 1
            new_templates[f'u{i}'] = url
            template_names_for_urls[url] = f'u{i}'
        if new_templates:
            rfs['templates'] = new_templates
        for k, v in rfs['refs'].items():
            if isinstance(v, list):
                url = v[0]
                if url in template_names_for_urls:
                    v[0] = '{{' + template_names_for_urls[url] + '}}'

    @staticmethod
    def remove_templates_in_rfs(rfs: dict) -> None:
        """
        Utility for removing templates from an rfs. This is the opposite of
        use_templates_in_rfs.
        """
        templates0 = rfs.get('templates', {})
        for k, v in rfs['refs'].items():
            if isinstance(v, list):
                url = v[0]
                if '{{' in url and '}}' in url:
                    template_name = url[2:-2].strip()
                    if template_name in templates0:
                        v[0] = templates0[template_name]
        rfs['templates'] = {}


def _is_chunk_base_key(base_key: str) -> bool:
    a = base_key.split('.')
    if len(a) == 0:
        return False
    for x in a:
        # check if integer
        try:
            int(x)
        except ValueError:
            return False
    return True


def _get_itemsize(dtype: str) -> int:
    d = np.dtype(dtype)
    return d.itemsize


def _pad_chunk(data: bytes, expected_chunk_size: int) -> bytes:
    return data + b'\0' * (expected_chunk_size - len(data))


def _get_padded_size(store, key: str, val: bytes):
    # If the key is a chunk and it's smaller than the expected size, then we
    # need to pad it with zeros. This can happen if this is the final chunk
    # in a contiguous hdf5 dataset. See
    # https://github.com/NeurodataWithoutBorders/lindi/pull/84
    base_key = key.split('/')[-1]
    if val and _is_chunk_base_key(base_key):
        parent_key = key.split('/')[:-1]
        zarray_key = '/'.join(parent_key) + '/.zarray'
        if zarray_key in store:
            zarray_json = store.__getitem__(zarray_key)
            assert isinstance(zarray_json, bytes)
            zarray = json.loads(zarray_json)
            # Encoded chunks have a variable byte length. Padding before the
            # compressor/filter pipeline changes its input (notably filter 401).
            if zarray.get('compressor') or zarray.get('filters'):
                return None
            chunk_shape = zarray['chunks']
            dtype = zarray['dtype']
            if np.dtype(dtype).kind in ['i', 'u', 'f']:
                expected_chunk_size = int(np.prod(chunk_shape)) * _get_itemsize(dtype)
                if len(val) < expected_chunk_size:
                    return expected_chunk_size

    return None
