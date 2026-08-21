import math
import os
import re
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import (
    Callable,
    Dict,
    List,
    Literal,
    Mapping,
    Sequence,
    Tuple,
    Union,
)
from urllib.parse import unquote, urlsplit

import requests
from tenacity import Retrying, retry_if_exception, wait_random_exponential
from tenacity.stop import stop_after_attempt, stop_before_delay


DEFAULT_RANGE_REQUEST_TIMEOUT_SECONDS = 60.0


class LindiRangeLayoutWarning(UserWarning):
    """An NWB signal is not laid out efficiently for high-latency reads."""


class LindiRangeLayoutError(RuntimeError):
    """An NWB signal failed the requested range-layout check."""


class LindiRangeEfficiencyWarning(UserWarning):
    """A logical selection caused severe range-read amplification."""


@dataclass(frozen=True)
class LindiRangeEfficiencyThresholds:
    """Thresholds for query-level efficiency warnings."""

    chunk_amplification_ratio: float = 8.0
    chunk_excess_bytes: int = 1024 * 1024
    transport_overread_ratio: float = 2.0
    transport_overread_bytes: int = 1024 * 1024
    fragmented_request_count: int = 8

    def __post_init__(self):
        if (
            not math.isfinite(self.chunk_amplification_ratio)
            or self.chunk_amplification_ratio <= 1
        ):
            raise ValueError(
                "chunk_amplification_ratio must be finite and greater than 1"
            )
        if self.chunk_excess_bytes < 0:
            raise ValueError("chunk_excess_bytes must be non-negative")
        if (
            not math.isfinite(self.transport_overread_ratio)
            or self.transport_overread_ratio <= 1
        ):
            raise ValueError(
                "transport_overread_ratio must be finite and greater than 1"
            )
        if self.transport_overread_bytes < 0:
            raise ValueError("transport_overread_bytes must be non-negative")
        if self.fragmented_request_count < 2:
            raise ValueError("fragmented_request_count must be at least 2")


@dataclass(frozen=True)
class LindiRangeReadMetrics:
    """Metrics for one Zarr bulk-read operation."""

    optimized: bool
    requested_key_count: int
    returned_key_count: int
    inline_value_count: int
    range_reference_count: int
    high_latency_reference_count: int
    cache_hit_count: int
    source_range_count: int
    planned_request_count: int
    read_attempt_count: int
    requested_bytes: int
    fetched_bytes: int
    overread_bytes: int
    elapsed_seconds: float
    referenced_bytes: int = 0
    cache_hit_bytes: int = 0
    transport_attempt_counts: Tuple[Tuple[str, int], ...] = ()
    transport_fetched_bytes: Tuple[Tuple[str, int], ...] = ()
    dataset_path: Union[str, None] = None
    logical_bytes: Union[int, None] = None
    decoded_chunk_bytes: Union[int, None] = None
    chunk_amplification_ratio: Union[float, None] = None
    transport_overread_ratio: Union[float, None] = None
    query_elapsed_seconds: Union[float, None] = None
    dataset_logical_bytes: Union[int, None] = None


RangeReadMetricsCallback = Callable[[LindiRangeReadMetrics], None]


@dataclass(frozen=True)
class LindiRangeReadOptions:
    """Controls coalescing, concurrency, retries, and instrumentation."""

    enabled: bool = True
    max_gap_bytes: int = 0
    max_overread_bytes: int = 0
    max_request_bytes: Union[int, None] = 512 * 1024 * 1024
    max_concurrency: int = 8
    max_retries: int = 8
    retry_backoff_seconds: float = 0.1
    retry_max_backoff_seconds: float = 10.0
    max_retry_elapsed_seconds: Union[float, None] = 120.0
    respect_retry_after_header: bool = True
    request_timeout_seconds: Union[float, None] = DEFAULT_RANGE_REQUEST_TIMEOUT_SECONDS
    layout_check: Literal["warn", "error", "ignore"] = "warn"
    runtime_efficiency_check: Literal["warn", "ignore"] = "warn"
    efficiency_warning_thresholds: LindiRangeEfficiencyThresholds = field(
        default_factory=LindiRangeEfficiencyThresholds
    )
    metrics_callback: Union[RangeReadMetricsCallback, None] = field(
        default=None, repr=False, compare=False
    )

    def __post_init__(self):
        if self.max_gap_bytes < 0:
            raise ValueError("max_gap_bytes must be non-negative")
        if self.max_overread_bytes < 0:
            raise ValueError("max_overread_bytes must be non-negative")
        if self.max_request_bytes is not None and self.max_request_bytes <= 0:
            raise ValueError("max_request_bytes must be positive or None")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be at least 1")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if (
            not math.isfinite(self.retry_backoff_seconds)
            or self.retry_backoff_seconds < 0
        ):
            raise ValueError("retry_backoff_seconds must be finite and non-negative")
        if (
            not math.isfinite(self.retry_max_backoff_seconds)
            or self.retry_max_backoff_seconds < self.retry_backoff_seconds
        ):
            raise ValueError(
                "retry_max_backoff_seconds must be finite and no smaller than "
                "retry_backoff_seconds"
            )
        if self.max_retry_elapsed_seconds is not None and (
            not math.isfinite(self.max_retry_elapsed_seconds)
            or self.max_retry_elapsed_seconds <= 0
        ):
            raise ValueError("max_retry_elapsed_seconds must be positive or None")
        if self.request_timeout_seconds is not None and (
            not math.isfinite(self.request_timeout_seconds)
            or self.request_timeout_seconds <= 0
        ):
            raise ValueError("request_timeout_seconds must be positive or None")
        if self.layout_check not in ("warn", "error", "ignore"):
            raise ValueError("layout_check must be 'warn', 'error', or 'ignore'")
        if self.runtime_efficiency_check not in ("warn", "ignore"):
            raise ValueError("runtime_efficiency_check must be 'warn' or 'ignore'")


@dataclass(frozen=True)
class ByteRange:
    """An exact referenced byte range required for one Zarr key."""

    key: str
    source: str
    offset: int
    length: int

    def __post_init__(self):
        if self.offset < 0:
            raise ValueError("Byte range offset must be non-negative")
        if self.length <= 0:
            raise ValueError("Byte range length must be positive")

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class PlannedByteRangeRequest:
    """One physical read and the exact ranges reconstructed from its payload."""

    source: str
    offset: int
    length: int
    ranges: Tuple[ByteRange, ...]
    requested_bytes: int
    overread_bytes: int

    @property
    def end(self) -> int:
        return self.offset + self.length


@dataclass(frozen=True)
class FetchedByteRangeRequest:
    request: PlannedByteRangeRequest
    data: bytes
    transport_name: str
    attempt_count: int
    fetched_bytes: int
    elapsed_seconds: float
    source_identity: Union[str, None] = None


@dataclass(frozen=True)
class RangeTransportCapabilities:
    name: str
    cacheable: bool
    high_latency: bool
    supports_concurrent_reads: bool
    supports_source_identity: bool

    def __post_init__(self):
        if not self.name:
            raise ValueError("transport capability name must not be empty")


@dataclass(frozen=True)
class RangeTransportResult:
    data: bytes
    attempt_count: int = 1
    fetched_bytes: Union[int, None] = None
    elapsed_seconds: float = 0.0
    source_identity: Union[str, None] = None

    def __post_init__(self):
        if self.attempt_count < 1:
            raise ValueError("attempt_count must be at least 1")
        if self.fetched_bytes is not None and self.fetched_bytes < len(self.data):
            raise ValueError("fetched_bytes cannot be smaller than returned data")


class RangeTransportError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        attempt_count: int = 1,
        fetched_bytes: int = 0,
        retry_after_seconds: Union[float, None] = None,
    ):
        super().__init__(message)
        self.attempt_count = attempt_count
        self.fetched_bytes = fetched_bytes
        self.retry_after_seconds = retry_after_seconds


class RetryableRangeTransportError(RangeTransportError):
    pass


class PermanentRangeTransportError(RangeTransportError):
    pass


class RangeReadError(RuntimeError):
    pass


class RangeTransport(ABC):
    """One-attempt byte-range transport independent of Zarr and NWB."""

    @property
    @abstractmethod
    def capabilities(self) -> RangeTransportCapabilities:
        raise NotImplementedError

    @abstractmethod
    def read_range(
        self,
        source: str,
        offset: int,
        length: int,
        *,
        timeout_seconds: Union[float, None],
    ) -> RangeTransportResult:
        raise NotImplementedError

    def close(self) -> None:
        pass


RangeTransportFactory = Callable[[], RangeTransport]


_RETRYABLE_HTTP_STATUS_CODES = frozenset((408, 429, 500, 502, 503, 504))


def _parse_retry_after_seconds(
    value: Union[str, None], *, now: Union[datetime, None] = None
) -> Union[float, None]:
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    if value.isdigit():
        return float(value)
    try:
        retry_at = parsedate_to_datetime(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if retry_at.tzinfo is None:
        retry_at = retry_at.replace(tzinfo=timezone.utc)
    current_time = now or datetime.now(timezone.utc)
    return max(0.0, (retry_at - current_time).total_seconds())


class HttpRangeTransport(RangeTransport):
    """HTTP(S) transport with bounded ranges and per-thread sessions."""

    _capabilities = RangeTransportCapabilities(
        name="http",
        cacheable=True,
        high_latency=True,
        supports_concurrent_reads=True,
        supports_source_identity=True,
    )

    def __init__(self):
        self._worker_local = threading.local()
        self._sessions: List[requests.Session] = []
        self._sessions_lock = threading.Lock()
        self._identities: Dict[str, str] = {}
        self._identities_lock = threading.Lock()

    @property
    def capabilities(self) -> RangeTransportCapabilities:
        return self._capabilities

    def _session(self) -> requests.Session:
        session = getattr(self._worker_local, "session", None)
        if session is None:
            session = requests.Session()
            self._worker_local.session = session
            with self._sessions_lock:
                self._sessions.append(session)
        return session

    def read_range(
        self,
        source: str,
        offset: int,
        length: int,
        *,
        timeout_seconds: Union[float, None],
    ) -> RangeTransportResult:
        from .LindiRemfile.LindiRemfile import _resolve_url

        started = time.perf_counter()
        with self._identities_lock:
            expected_identity = self._identities.get(source)
        headers = {
            "User-Agent": "LINDI range reader",
            "Accept-Encoding": "identity",
            "Range": f"bytes={offset}-{offset + length - 1}",
        }
        if expected_identity is not None:
            headers["If-Match"] = expected_identity
        try:
            response = self._session().get(
                _resolve_url(source),
                headers=headers,
                timeout=timeout_seconds,
                stream=True,
            )
            with response:
                retry_after_seconds = _parse_retry_after_seconds(
                    response.headers.get("Retry-After")
                )
                if response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
                    raise RetryableRangeTransportError(
                        f"Retryable status {response.status_code} for {source}",
                        retry_after_seconds=retry_after_seconds,
                    )
                if response.status_code == 412:
                    raise PermanentRangeTransportError(
                        f"Source changed while reading {source}"
                    )
                try:
                    response.raise_for_status()
                except requests.RequestException as error:
                    raise PermanentRangeTransportError(
                        f"HTTP status {response.status_code} for {source}"
                    ) from error
                if response.status_code == 200 and offset != 0:
                    raise RetryableRangeTransportError(
                        f"Server ignored byte range {offset}-{offset + length - 1} "
                        f"for {source}: status {response.status_code}"
                    )
                if response.status_code not in (200, 206):
                    raise PermanentRangeTransportError(
                        f"Unexpected status {response.status_code} for {source}"
                    )
                content_range = response.headers.get("Content-Range")
                if content_range is not None:
                    match = re.fullmatch(
                        r"bytes (\d+)-(\d+)/(?:\d+|\*)", content_range
                    )
                    if match is None or (
                        int(match.group(1)) != offset
                        or int(match.group(2)) != offset + length - 1
                    ):
                        raise RetryableRangeTransportError(
                            f"Unexpected Content-Range for {source}: "
                            f"{content_range!r}"
                        )
                content_length = response.headers.get("Content-Length")
                if content_length is not None:
                    try:
                        content_length_value = int(content_length)
                    except ValueError as error:
                        raise RetryableRangeTransportError(
                            f"Invalid Content-Length for {source}: "
                            f"{content_length!r}"
                        ) from error
                    if content_length_value != length:
                        raise RetryableRangeTransportError(
                            f"Unexpected Content-Length for {source}: received "
                            f"{content_length_value}, expected {length}"
                        )
                data_buffer = bytearray()
                for chunk in response.iter_content(
                    chunk_size=min(1024 * 1024, length + 1)
                ):
                    if not chunk:
                        continue
                    if len(data_buffer) + len(chunk) > length:
                        raise RetryableRangeTransportError(
                            f"Response exceeded expected byte count for {source}: "
                            f"expected {length}",
                            fetched_bytes=len(data_buffer) + len(chunk),
                        )
                    data_buffer.extend(chunk)
                data = bytes(data_buffer)
                if len(data) != length:
                    raise RetryableRangeTransportError(
                        f"Unexpected byte count for {source} at {offset}: received "
                        f"{len(data)}, expected {length}",
                        fetched_bytes=len(data),
                    )
                response_identity = response.headers.get("ETag")
        except RangeTransportError:
            raise
        except requests.exceptions.SSLError as error:
            raise PermanentRangeTransportError(
                f"TLS validation failed for {source}: {error}"
            ) from error
        except (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ContentDecodingError,
        ) as error:
            raise RetryableRangeTransportError(str(error)) from error
        except requests.RequestException as error:
            raise PermanentRangeTransportError(str(error)) from error

        if response_identity is None:
            response_identity = expected_identity
        elif expected_identity is not None and response_identity != expected_identity:
            raise PermanentRangeTransportError(
                f"Source changed while reading {source}"
            )
        elif expected_identity is None:
            with self._identities_lock:
                recorded = self._identities.setdefault(source, response_identity)
            if recorded != response_identity:
                raise PermanentRangeTransportError(
                    f"Source changed while reading {source}"
                )
        return RangeTransportResult(
            data=data,
            fetched_bytes=len(data),
            elapsed_seconds=time.perf_counter() - started,
            source_identity=response_identity,
        )

    def close(self) -> None:
        with self._sessions_lock:
            sessions = self._sessions
            self._sessions = []
        for session in sessions:
            session.close()


class LocalFileRangeTransport(RangeTransport):
    """Local transport using reusable descriptors and positional reads."""

    _capabilities = RangeTransportCapabilities(
        name="file",
        cacheable=False,
        high_latency=False,
        supports_concurrent_reads=True,
        supports_source_identity=True,
    )

    def __init__(self):
        self._descriptors: Dict[str, int] = {}
        self._descriptor_locks: Dict[str, threading.Lock] = {}
        self._lock = threading.Lock()

    @property
    def capabilities(self) -> RangeTransportCapabilities:
        return self._capabilities

    @staticmethod
    def _path(source: str) -> str:
        parts = urlsplit(source)
        if parts.scheme.lower() != "file":
            return source
        if parts.netloc not in ("", "localhost"):
            raise PermanentRangeTransportError(
                f"file URI host must be empty or localhost: {source!r}"
            )
        if parts.query or parts.fragment:
            raise PermanentRangeTransportError(
                f"file URI cannot contain a query or fragment: {source!r}"
            )
        path = unquote(parts.path)
        if os.name == "nt" and re.match(r"^/[A-Za-z]:", path):
            path = path[1:]
        return path

    def _descriptor(self, path: str) -> Tuple[int, threading.Lock]:
        with self._lock:
            descriptor = self._descriptors.get(path)
            if descriptor is None:
                descriptor = os.open(path, os.O_RDONLY)
                self._descriptors[path] = descriptor
                self._descriptor_locks[path] = threading.Lock()
            return descriptor, self._descriptor_locks[path]

    @staticmethod
    def _identity(descriptor: int) -> str:
        stat = os.fstat(descriptor)
        return ":".join(
            str(value)
            for value in (
                stat.st_dev,
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
            )
        )

    def read_range(
        self,
        source: str,
        offset: int,
        length: int,
        *,
        timeout_seconds: Union[float, None],
    ) -> RangeTransportResult:
        del timeout_seconds
        started = time.perf_counter()
        path = self._path(source)
        try:
            descriptor, descriptor_lock = self._descriptor(path)
            identity_before = self._identity(descriptor)
            if hasattr(os, "pread"):
                data = os.pread(descriptor, length, offset)
            else:  # pragma: no cover - os.pread is available on supported CI
                with descriptor_lock:
                    os.lseek(descriptor, offset, os.SEEK_SET)
                    data = os.read(descriptor, length)
            identity_after = self._identity(descriptor)
        except OSError as error:
            raise PermanentRangeTransportError(
                f"Unable to read local range from {path}: {error}"
            ) from error
        if identity_before != identity_after:
            raise PermanentRangeTransportError(
                f"Local source changed while reading {path}"
            )
        return RangeTransportResult(
            data=data,
            fetched_bytes=len(data),
            elapsed_seconds=time.perf_counter() - started,
            source_identity=identity_after,
        )

    def close(self) -> None:
        with self._lock:
            descriptors = list(self._descriptors.values())
            self._descriptors = {}
            self._descriptor_locks = {}
        for descriptor in descriptors:
            os.close(descriptor)


def _http_transport_factory() -> RangeTransport:
    return HttpRangeTransport()


def _local_transport_factory() -> RangeTransport:
    return LocalFileRangeTransport()


_DEFAULT_RANGE_TRANSPORTS: Mapping[str, RangeTransportFactory] = {
    "http": _http_transport_factory,
    "https": _http_transport_factory,
    "file": _local_transport_factory,
}


class _RangeTransportManager:
    def __init__(
        self,
        factories: Union[Mapping[str, RangeTransportFactory], None] = None,
    ):
        merged = dict(_DEFAULT_RANGE_TRANSPORTS)
        if factories is not None:
            for scheme, factory in factories.items():
                normalized = scheme.lower()
                if not normalized or normalized != scheme:
                    raise ValueError(
                        "range transport schemes must be non-empty lowercase strings"
                    )
                if not callable(factory):
                    raise TypeError(f"range transport factory for {scheme!r} is not callable")
                merged[normalized] = factory
        self._factories = merged
        self._instances: Dict[int, RangeTransport] = {}
        self._instance_lock = threading.Lock()
        self._execution_locks: Dict[int, threading.Lock] = {}
        self._source_identities: Dict[Tuple[int, str], str] = {}
        self._identity_lock = threading.Lock()
        self._closed = False

    @staticmethod
    def _scheme(source: str) -> str:
        scheme = urlsplit(source).scheme.lower()
        if os.name == "nt" and len(scheme) == 1 and source[1:3] in (":\\", ":/"):
            return "file"
        return scheme or "file"

    def resolve(self, source: str) -> RangeTransport:
        scheme = self._scheme(source)
        factory = self._factories.get(scheme)
        if factory is None:
            raise PermanentRangeTransportError(
                f"Unsupported range source scheme {scheme!r} for {source!r}"
            )
        factory_key = id(factory)
        with self._instance_lock:
            if self._closed:
                raise RuntimeError("range transport manager is closed")
            transport = self._instances.get(factory_key)
            if transport is None:
                transport = factory()
                if not isinstance(transport, RangeTransport):
                    raise TypeError(
                        f"range transport factory for {scheme!r} did not return "
                        "a RangeTransport"
                    )
                self._instances[factory_key] = transport
                self._execution_locks[factory_key] = threading.Lock()
        return transport

    def execution_context(self, transport: RangeTransport):
        if transport.capabilities.supports_concurrent_reads:
            return nullcontext()
        return self._execution_locks[id(self._factories[self._scheme_for_transport(transport)])]

    def _scheme_for_transport(self, transport: RangeTransport) -> str:
        for scheme, factory in self._factories.items():
            if self._instances.get(id(factory)) is transport:
                return scheme
        raise RuntimeError("range transport is not managed by this manager")

    def validate_identity(
        self,
        transport: RangeTransport,
        source: str,
        identity: Union[str, None],
    ) -> None:
        if identity is None:
            return
        key = (id(transport), source)
        with self._identity_lock:
            previous = self._source_identities.setdefault(key, identity)
        if previous != identity:
            raise PermanentRangeTransportError(
                f"Source identity changed while reading {source}"
            )

    def close(self) -> None:
        with self._instance_lock:
            if self._closed:
                return
            self._closed = True
            transports = list(self._instances.values())
            self._instances = {}
        first_error = None
        for transport in transports:
            try:
                transport.close()
            except Exception as error:  # noqa: BLE001 - close all owned transports
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error


def _make_planned_request(
    source: str,
    ranges: Sequence[ByteRange],
    start: int,
    end: int,
    covered_bytes: int,
) -> PlannedByteRangeRequest:
    span = end - start
    return PlannedByteRangeRequest(
        source=source,
        offset=start,
        length=span,
        ranges=tuple(ranges),
        requested_bytes=covered_bytes,
        overread_bytes=span - covered_bytes,
    )


def plan_byte_range_requests(
    ranges: Sequence[ByteRange], options: LindiRangeReadOptions
) -> List[PlannedByteRangeRequest]:
    """Plan bounded coalesced reads for exact byte ranges."""

    ranges_by_source: Dict[str, List[ByteRange]] = {}
    for byte_range in ranges:
        ranges_by_source.setdefault(byte_range.source, []).append(byte_range)

    requests_to_make: List[PlannedByteRangeRequest] = []
    for source in sorted(ranges_by_source):
        source_ranges = sorted(
            ranges_by_source[source], key=lambda item: (item.offset, item.end, item.key)
        )
        if not options.enabled:
            requests_to_make.extend(
                PlannedByteRangeRequest(
                    source=source,
                    offset=item.offset,
                    length=item.length,
                    ranges=(item,),
                    requested_bytes=item.length,
                    overread_bytes=0,
                )
                for item in source_ranges
            )
            continue

        current_ranges: List[ByteRange] = []
        current_start = 0
        current_end = 0
        current_covered_bytes = 0
        for byte_range in source_ranges:
            if not current_ranges:
                current_ranges = [byte_range]
                current_start = byte_range.offset
                current_end = byte_range.end
                current_covered_bytes = byte_range.length
                continue
            gap = max(0, byte_range.offset - current_end)
            proposed_end = max(current_end, byte_range.end)
            added_covered_bytes = max(
                0, byte_range.end - max(byte_range.offset, current_end)
            )
            proposed_covered_bytes = current_covered_bytes + added_covered_bytes
            proposed_span = proposed_end - current_start
            proposed_overread = proposed_span - proposed_covered_bytes
            within_request_size = (
                options.max_request_bytes is None
                or proposed_span <= options.max_request_bytes
            )
            if (
                gap <= options.max_gap_bytes
                and proposed_overread <= options.max_overread_bytes
                and within_request_size
            ):
                current_ranges.append(byte_range)
                current_end = proposed_end
                current_covered_bytes = proposed_covered_bytes
            else:
                requests_to_make.append(
                    _make_planned_request(
                        source,
                        current_ranges,
                        current_start,
                        current_end,
                        current_covered_bytes,
                    )
                )
                current_ranges = [byte_range]
                current_start = byte_range.offset
                current_end = byte_range.end
                current_covered_bytes = byte_range.length
        if current_ranges:
            requests_to_make.append(
                _make_planned_request(
                    source,
                    current_ranges,
                    current_start,
                    current_end,
                    current_covered_bytes,
                )
            )
    return requests_to_make


class _RangeRetryWait:
    def __init__(self, options: LindiRangeReadOptions):
        self._options = options
        self._jitter_wait = wait_random_exponential(
            multiplier=options.retry_backoff_seconds,
            max=options.retry_max_backoff_seconds,
        )

    def __call__(self, retry_state) -> float:
        jitter_seconds = float(self._jitter_wait(retry_state))
        if not self._options.respect_retry_after_header:
            return jitter_seconds
        error = retry_state.outcome.exception() if retry_state.outcome else None
        retry_after_seconds = getattr(error, "retry_after_seconds", None)
        if retry_after_seconds is None:
            return jitter_seconds
        return max(jitter_seconds, retry_after_seconds)


def _make_retry_stop(options: LindiRangeReadOptions):
    attempts_stop = stop_after_attempt(options.max_retries + 1)
    elapsed_stop = (
        stop_before_delay(options.max_retry_elapsed_seconds)
        if options.max_retry_elapsed_seconds is not None
        else None
    )

    def should_stop(retry_state) -> bool:
        if attempts_stop(retry_state):
            return True
        return elapsed_stop is not None and elapsed_stop(retry_state)

    return should_stop


def _is_retryable_range_error(error: BaseException) -> bool:
    return isinstance(error, RetryableRangeTransportError)


def _fetch_one_request(
    request: PlannedByteRangeRequest,
    manager: _RangeTransportManager,
    options: LindiRangeReadOptions,
) -> FetchedByteRangeRequest:
    started = time.perf_counter()
    attempt_count = 0
    fetched_bytes = 0
    transport = manager.resolve(request.source)
    retrying = Retrying(
        retry=retry_if_exception(_is_retryable_range_error),
        wait=_RangeRetryWait(options),
        stop=_make_retry_stop(options),
        reraise=True,
    )
    try:
        for attempt in retrying:
            with attempt:
                try:
                    with manager.execution_context(transport):
                        result = transport.read_range(
                            request.source,
                            request.offset,
                            request.length,
                            timeout_seconds=options.request_timeout_seconds,
                        )
                except RangeTransportError as error:
                    attempt_count += error.attempt_count
                    fetched_bytes += error.fetched_bytes
                    raise
                except Exception as error:
                    attempt_count += 1
                    raise PermanentRangeTransportError(str(error)) from error
                attempt_count += result.attempt_count
                fetched_bytes += (
                    len(result.data)
                    if result.fetched_bytes is None
                    else result.fetched_bytes
                )
                if attempt_count > options.max_retries + 1:
                    raise PermanentRangeTransportError(
                        f"Transport exceeded the physical attempt budget for "
                        f"{request.source}"
                    )
                if len(result.data) != request.length:
                    raise RetryableRangeTransportError(
                        f"Unexpected byte count for {request.source} at "
                        f"{request.offset}: received {len(result.data)}, expected "
                        f"{request.length}",
                        attempt_count=0,
                    )
                manager.validate_identity(
                    transport, request.source, result.source_identity
                )
        return FetchedByteRangeRequest(
            request=request,
            data=result.data,
            transport_name=transport.capabilities.name,
            attempt_count=attempt_count,
            fetched_bytes=fetched_bytes,
            elapsed_seconds=time.perf_counter() - started,
            source_identity=result.source_identity,
        )
    except Exception as error:
        raise RangeReadError(
            f"Failed to read bytes {request.offset}-{request.end - 1} from "
            f"{request.source} after {attempt_count} attempts: {error}"
        ) from error


def fetch_byte_range_requests(
    requests_to_make: Sequence[PlannedByteRangeRequest],
    options: LindiRangeReadOptions,
    *,
    range_transports: Union[Mapping[str, RangeTransportFactory], None] = None,
    _transport_manager: Union[_RangeTransportManager, None] = None,
) -> List[FetchedByteRangeRequest]:
    """Fetch planned reads with bounded, transport-neutral concurrency."""

    if not requests_to_make:
        return []
    if range_transports is not None and _transport_manager is not None:
        raise ValueError("Specify range_transports or _transport_manager, not both")
    owns_manager = _transport_manager is None
    manager = _transport_manager or _RangeTransportManager(range_transports)
    worker_count = 1 if not options.enabled else min(
        options.max_concurrency, len(requests_to_make)
    )
    try:
        if worker_count == 1:
            return [
                _fetch_one_request(request, manager, options)
                for request in requests_to_make
            ]
        futures = []
        try:
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(_fetch_one_request, request, manager, options)
                    for request in requests_to_make
                ]
                return [future.result() for future in as_completed(futures)]
        except Exception:
            for future in futures:
                future.cancel()
            raise
    except RangeReadError:
        raise
    except RangeTransportError as error:
        raise RangeReadError(str(error)) from error
    finally:
        if owns_manager:
            manager.close()
