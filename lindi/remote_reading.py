import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Dict, List, Literal, Sequence, Tuple, Union

import requests
from tenacity import Retrying, retry_if_exception, wait_random_exponential
from tenacity.stop import stop_after_attempt, stop_before_delay


DEFAULT_REMOTE_REQUEST_TIMEOUT_SECONDS = 60.0


class LindiRemoteLayoutWarning(UserWarning):
    """An NWB signal is not laid out efficiently for remote reads."""


class LindiRemoteLayoutError(RuntimeError):
    """An NWB signal failed the requested remote-layout check."""


class LindiRemoteEfficiencyWarning(UserWarning):
    """A logical selection caused severe remote-read amplification."""


@dataclass(frozen=True)
class LindiRemoteEfficiencyThresholds:
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
class LindiRemoteReadMetrics:
    """Metrics for one Zarr bulk-read operation.

    Byte counters cover cache misses handled by the transport. ``range_get_count``
    counts physical attempts, including retries that precede a successful read.
    """

    optimized: bool
    requested_key_count: int
    returned_key_count: int
    inline_value_count: int
    local_file_read_count: int
    remote_reference_count: int
    cache_hit_count: int
    source_range_count: int
    planned_request_count: int
    range_get_count: int
    requested_bytes: int
    fetched_bytes: int
    overread_bytes: int
    elapsed_seconds: float
    referenced_bytes: int = 0
    cache_hit_bytes: int = 0
    dataset_path: Union[str, None] = None
    logical_bytes: Union[int, None] = None
    decoded_chunk_bytes: Union[int, None] = None
    chunk_amplification_ratio: Union[float, None] = None
    transport_overread_ratio: Union[float, None] = None
    query_elapsed_seconds: Union[float, None] = None
    dataset_logical_bytes: Union[int, None] = None


RemoteReadMetricsCallback = Callable[[LindiRemoteReadMetrics], None]


@dataclass(frozen=True)
class LindiRemoteReadOptions:
    """Controls coalescing and concurrent reads from remote reference objects.

    ``max_overread_bytes`` is applied independently to every coalesced HTTP
    request. ``max_request_bytes`` only limits merging; a single referenced
    chunk larger than the limit is still fetched as one atomic range.
    """

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
    request_timeout_seconds: Union[float, None] = DEFAULT_REMOTE_REQUEST_TIMEOUT_SECONDS
    layout_check: Literal["warn", "error", "ignore"] = "warn"
    runtime_efficiency_check: Literal["warn", "ignore"] = "warn"
    efficiency_warning_thresholds: LindiRemoteEfficiencyThresholds = field(
        default_factory=LindiRemoteEfficiencyThresholds
    )
    metrics_callback: Union[RemoteReadMetricsCallback, None] = field(
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
    """One HTTP request and the exact ranges reconstructed from its payload."""

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
    attempt_count: int
    elapsed_seconds: float


class RemoteRangeReadError(RuntimeError):
    pass


class _RetryableRemoteRangeReadError(RemoteRangeReadError):
    def __init__(
        self, message: str, *, retry_after_seconds: Union[float, None] = None
    ):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


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


def _is_retryable_remote_error(error: BaseException) -> bool:
    if isinstance(error, _RetryableRemoteRangeReadError):
        return True
    if isinstance(error, requests.exceptions.SSLError):
        return False
    return isinstance(
        error,
        (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.exceptions.ContentDecodingError,
        ),
    )


class _RemoteRetryWait:
    def __init__(self, options: LindiRemoteReadOptions):
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


def _make_retry_stop(options: LindiRemoteReadOptions):
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
    ranges: Sequence[ByteRange], options: LindiRemoteReadOptions
) -> List[PlannedByteRangeRequest]:
    """Plan bounded coalesced requests for exact byte ranges."""

    ranges_by_source: Dict[str, List[ByteRange]] = {}
    for byte_range in ranges:
        ranges_by_source.setdefault(byte_range.source, []).append(byte_range)

    requests_to_make: List[PlannedByteRangeRequest] = []
    for source in sorted(ranges_by_source):
        source_ranges = sorted(
            ranges_by_source[source], key=lambda x: (x.offset, x.end, x.key)
        )
        if not options.enabled:
            requests_to_make.extend(
                PlannedByteRangeRequest(
                    source=source,
                    offset=x.offset,
                    length=x.length,
                    ranges=(x,),
                    requested_bytes=x.length,
                    overread_bytes=0,
                )
                for x in source_ranges
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
            added_covered_bytes = max(0, byte_range.end - max(byte_range.offset, current_end))
            proposed_covered_bytes = current_covered_bytes + added_covered_bytes
            proposed_span = proposed_end - current_start
            proposed_overread = proposed_span - proposed_covered_bytes
            within_request_size = (
                options.max_request_bytes is None
                or proposed_span <= options.max_request_bytes
            )
            should_merge = (
                gap <= options.max_gap_bytes
                and proposed_overread <= options.max_overread_bytes
                and within_request_size
            )
            if should_merge:
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


def fetch_byte_range_requests(
    requests_to_make: Sequence[PlannedByteRangeRequest],
    options: LindiRemoteReadOptions,
) -> List[FetchedByteRangeRequest]:
    """Fetch planned requests using bounded concurrency and per-worker sessions."""

    if not requests_to_make:
        return []

    worker_count = 1 if not options.enabled else min(
        options.max_concurrency, len(requests_to_make)
    )
    if worker_count == 1:
        with requests.Session() as session:
            return [_fetch_one_request(x, session, options) for x in requests_to_make]

    worker_local = threading.local()
    sessions: List[requests.Session] = []
    sessions_lock = threading.Lock()

    def fetch_in_worker(request: PlannedByteRangeRequest) -> FetchedByteRangeRequest:
        session = getattr(worker_local, "session", None)
        if session is None:
            session = requests.Session()
            worker_local.session = session
            with sessions_lock:
                sessions.append(session)
        return _fetch_one_request(request, session, options)

    futures = []
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(fetch_in_worker, x) for x in requests_to_make]
            return [future.result() for future in as_completed(futures)]
    except Exception:
        for future in futures:
            future.cancel()
        raise
    finally:
        for session in sessions:
            session.close()


def _fetch_one_request(
    request: PlannedByteRangeRequest,
    session: requests.Session,
    options: LindiRemoteReadOptions,
) -> FetchedByteRangeRequest:
    started = time.perf_counter()
    attempt_count = 0
    retrying = Retrying(
        retry=retry_if_exception(_is_retryable_remote_error),
        wait=_RemoteRetryWait(options),
        stop=_make_retry_stop(options),
        reraise=True,
    )
    try:
        for attempt in retrying:
            attempt_count = attempt.retry_state.attempt_number
            with attempt:
                data = _fetch_and_validate_request(request, session, options)
        return FetchedByteRangeRequest(
            request=request,
            data=data,
            attempt_count=attempt_count,
            elapsed_seconds=time.perf_counter() - started,
        )
    except Exception as error:
        raise RemoteRangeReadError(
            f"Failed to read bytes {request.offset}-{request.end - 1} "
            f"from {request.source} after {attempt_count} attempts: {error}"
        ) from error


def _fetch_and_validate_request(
    request: PlannedByteRangeRequest,
    session: requests.Session,
    options: LindiRemoteReadOptions,
) -> bytes:
    from .LindiRemfile.LindiRemfile import _resolve_url

    resolved_url = _resolve_url(request.source)
    headers = {
        "User-Agent": "LINDI remote range reader",
        "Accept-Encoding": "identity",
        "Range": f"bytes={request.offset}-{request.end - 1}",
    }
    response = session.get(
        resolved_url,
        headers=headers,
        timeout=options.request_timeout_seconds,
        stream=True,
    )
    with response:
        retry_after_seconds = _parse_retry_after_seconds(
            response.headers.get("Retry-After")
        )
        if response.status_code in _RETRYABLE_HTTP_STATUS_CODES:
            raise _RetryableRemoteRangeReadError(
                f"Retryable status {response.status_code} for {request.source}",
                retry_after_seconds=retry_after_seconds,
            )
        response.raise_for_status()
        if response.status_code == 200 and request.offset != 0:
            raise _RetryableRemoteRangeReadError(
                f"Server ignored byte range {request.offset}-{request.end - 1} "
                f"for {request.source}: status {response.status_code}"
            )
        if response.status_code not in (200, 206):
            raise RemoteRangeReadError(
                f"Unexpected status {response.status_code} for {request.source}"
            )
        content_range = response.headers.get("Content-Range")
        if content_range is not None:
            match = re.fullmatch(r"bytes (\d+)-(\d+)/(?:\d+|\*)", content_range)
            if match is None or (
                int(match.group(1)) != request.offset
                or int(match.group(2)) != request.end - 1
            ):
                raise _RetryableRemoteRangeReadError(
                    f"Unexpected Content-Range for {request.source}: {content_range!r}"
                )
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                content_length_value = int(content_length)
            except ValueError as error:
                raise _RetryableRemoteRangeReadError(
                    f"Invalid Content-Length for {request.source}: {content_length!r}"
                ) from error
            if content_length_value != request.length:
                raise _RetryableRemoteRangeReadError(
                    f"Unexpected Content-Length for {request.source}: received "
                    f"{content_length_value}, expected {request.length}"
                )
        data_buffer = bytearray()
        for chunk in response.iter_content(
            chunk_size=min(1024 * 1024, request.length + 1)
        ):
            if not chunk:
                continue
            if len(data_buffer) + len(chunk) > request.length:
                raise _RetryableRemoteRangeReadError(
                    f"Response exceeded expected byte count for {request.source}: "
                    f"expected {request.length}"
                )
            data_buffer.extend(chunk)
        data = bytes(data_buffer)
    if len(data) != request.length:
        raise _RetryableRemoteRangeReadError(
            f"Unexpected byte count for {request.source} at {request.offset}: "
            f"received {len(data)}, expected {request.length}"
        )
    return data
