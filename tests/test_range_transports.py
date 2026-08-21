import os
import threading
import time

import numpy as np
import pytest

import lindi
from lindi.range_reading import (
    ByteRange,
    PlannedByteRangeRequest,
    RangeReadError,
    RetryableRangeTransportError,
    fetch_byte_range_requests,
)


def _reference_file_system(source, *, second_offset=4):
    return {
        "version": 1,
        "refs": {
            ".zgroup": {"zarr_format": 2},
            "data/.zarray": {
                "chunks": [4],
                "compressor": None,
                "dtype": "|u1",
                "fill_value": 0,
                "filters": None,
                "order": "C",
                "shape": [8],
                "zarr_format": 2,
            },
            "data/.zattrs": {},
            "data/0": [source, 0, 4],
            "data/1": [source, second_offset, 4],
        },
    }


class _MemoryTransport(lindi.RangeTransport):
    def __init__(
        self,
        data,
        *,
        failures=0,
        identities=None,
        short_reads=False,
        supports_concurrent_reads=True,
    ):
        self._data = data
        self._failures = failures
        self._identities = list(identities or ["memory-v1"])
        self._short_reads = short_reads
        self._capabilities = lindi.RangeTransportCapabilities(
            name="memory",
            cacheable=False,
            high_latency=True,
            supports_concurrent_reads=supports_concurrent_reads,
            supports_source_identity=True,
        )
        self.calls = []
        self.closed = False
        self.active_reads = 0
        self.max_active_reads = 0
        self._lock = threading.Lock()

    @property
    def capabilities(self):
        return self._capabilities

    def read_range(self, source, offset, length, *, timeout_seconds):
        del timeout_seconds
        started = time.perf_counter()
        with self._lock:
            self.calls.append((source, offset, length))
            self.active_reads += 1
            self.max_active_reads = max(self.max_active_reads, self.active_reads)
            call_index = len(self.calls) - 1
            should_fail = self._failures > 0
            if should_fail:
                self._failures -= 1
        try:
            if should_fail:
                raise RetryableRangeTransportError("imposed transient failure")
            value = self._data[offset:offset + length]
            if self._short_reads and value:
                value = value[:-1]
            identity = self._identities[min(call_index, len(self._identities) - 1)]
            return lindi.RangeTransportResult(
                data=value,
                fetched_bytes=len(value),
                elapsed_seconds=time.perf_counter() - started,
                source_identity=identity,
            )
        finally:
            with self._lock:
                self.active_reads -= 1

    def close(self):
        self.closed = True


def _planned_request(source, offset, length, key):
    byte_range = ByteRange(key, source, offset, length)
    return PlannedByteRangeRequest(
        source=source,
        offset=offset,
        length=length,
        ranges=(byte_range,),
        requested_bytes=length,
        overread_bytes=0,
    )


def test_local_references_use_shared_planning_pipeline(tmp_path):
    source = tmp_path / "data.bin"
    source.write_bytes(bytes(range(8)))
    metrics = []
    options = lindi.LindiRangeReadOptions(metrics_callback=metrics.append)
    with lindi.LindiH5pyFile.from_reference_file_system(
        _reference_file_system(str(source)),
        range_read_options=options,
    ) as client:
        assert np.array_equal(client["data"][:], np.arange(8, dtype=np.uint8))
    assert metrics[-1].range_reference_count == 2
    assert metrics[-1].high_latency_reference_count == 0
    assert metrics[-1].planned_request_count == 1
    assert metrics[-1].read_attempt_count == 1
    assert metrics[-1].transport_attempt_counts == (("file", 1),)


def test_local_file_uri_and_descriptor_cleanup(tmp_path):
    source = tmp_path / "data.bin"
    source.write_bytes(b"abcdefgh")
    transport = lindi.LocalFileRangeTransport()
    result = transport.read_range(
        source.as_uri(), 2, 4, timeout_seconds=None
    )
    assert result.data == b"cdef"
    descriptors = tuple(transport._descriptors.values())
    assert descriptors
    transport.close()
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)


@pytest.mark.parametrize(
    "source",
    [
        "file://remote.example/tmp/data.bin",
        "file:///tmp/data.bin?version=1",
        "file:///tmp/data.bin#fragment",
    ],
)
def test_unsafe_file_uris_are_rejected(source):
    transport = lindi.LocalFileRangeTransport()
    with pytest.raises(lindi.PermanentRangeTransportError):
        transport.read_range(source, 0, 1, timeout_seconds=None)


def test_custom_transport_injection_retry_metrics_and_cleanup():
    transport = _MemoryTransport(bytes(range(8)), failures=1)
    metrics = []
    options = lindi.LindiRangeReadOptions(
        max_retries=1,
        retry_backoff_seconds=0,
        metrics_callback=metrics.append,
    )
    with lindi.LindiH5pyFile.from_reference_file_system(
        _reference_file_system("memory://object"),
        range_read_options=options,
        range_transports={"memory": lambda: transport},
    ) as client:
        assert np.array_equal(client["data"][:], np.arange(8, dtype=np.uint8))
    assert transport.closed
    assert len(transport.calls) == 2
    assert metrics[-1].read_attempt_count == 2
    assert metrics[-1].transport_attempt_counts == (("memory", 2),)


def test_transport_identity_change_is_rejected():
    transport = _MemoryTransport(
        b"abcdefgh",
        identities=["version-1", "version-2"],
    )
    requests = [
        _planned_request("memory://object", 0, 4, "a"),
        _planned_request("memory://object", 4, 4, "b"),
    ]
    with pytest.raises(RangeReadError, match="identity changed"):
        fetch_byte_range_requests(
            requests,
            lindi.LindiRangeReadOptions(max_concurrency=1, max_retries=0),
            range_transports={"memory": lambda: transport},
        )
    assert transport.closed


def test_transport_short_read_and_unsupported_scheme_are_rejected():
    short_transport = _MemoryTransport(b"abcd", short_reads=True)
    with pytest.raises(RangeReadError, match="Unexpected byte count"):
        fetch_byte_range_requests(
            [_planned_request("memory://object", 0, 4, "a")],
            lindi.LindiRangeReadOptions(max_retries=0),
            range_transports={"memory": lambda: short_transport},
        )
    with pytest.raises(RangeReadError, match="Unsupported range source scheme"):
        fetch_byte_range_requests(
            [_planned_request("unknown://object", 0, 4, "a")],
            lindi.LindiRangeReadOptions(max_retries=0),
        )


def test_nonconcurrent_transport_is_serialized():
    class SlowTransport(_MemoryTransport):
        def read_range(self, source, offset, length, *, timeout_seconds):
            time.sleep(0.02)
            return super().read_range(
                source, offset, length, timeout_seconds=timeout_seconds
            )

    transport = SlowTransport(b"abcdefgh", supports_concurrent_reads=False)
    fetched = fetch_byte_range_requests(
        [
            _planned_request("memory://object", 0, 4, "a"),
            _planned_request("memory://object", 4, 4, "b"),
        ],
        lindi.LindiRangeReadOptions(max_concurrency=2, max_retries=0),
        range_transports={"memory": lambda: transport},
    )
    assert len(fetched) == 2
    assert transport.max_active_reads == 1


def test_transport_factory_must_return_range_transport():
    with pytest.raises(TypeError, match="did not return a RangeTransport"):
        fetch_byte_range_requests(
            [_planned_request("broken://object", 0, 1, "a")],
            lindi.LindiRangeReadOptions(max_retries=0),
            range_transports={"broken": lambda: object()},
        )


def test_transport_cannot_hide_attempts_beyond_budget():
    class HiddenRetryTransport(_MemoryTransport):
        def read_range(self, source, offset, length, *, timeout_seconds):
            result = super().read_range(
                source, offset, length, timeout_seconds=timeout_seconds
            )
            return lindi.RangeTransportResult(
                data=result.data,
                attempt_count=2,
                fetched_bytes=result.fetched_bytes,
                source_identity=result.source_identity,
            )

    transport = HiddenRetryTransport(b"abcd")
    with pytest.raises(RangeReadError, match="physical attempt budget"):
        fetch_byte_range_requests(
            [_planned_request("memory://object", 0, 4, "a")],
            lindi.LindiRangeReadOptions(max_retries=0),
            range_transports={"memory": lambda: transport},
        )
