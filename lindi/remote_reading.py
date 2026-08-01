import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence, Tuple, Union

import requests


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
    request_timeout_seconds: Union[float, None] = None
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
        if self.retry_backoff_seconds < 0:
            raise ValueError("retry_backoff_seconds must be non-negative")
        if self.request_timeout_seconds is not None and self.request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive or None")


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
    from .LindiRemfile.LindiRemfile import _resolve_url

    started = time.perf_counter()
    attempt_count = 0
    last_error: Union[Exception, None] = None
    for attempt in range(options.max_retries + 1):
        attempt_count += 1
        try:
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
            )
            with response:
                response.raise_for_status()
                data = response.content
                if response.status_code != 206:
                    whole_object_response = (
                        response.status_code == 200
                        and request.offset == 0
                        and len(data) == request.length
                    )
                    if not whole_object_response:
                        raise RemoteRangeReadError(
                            f"Server ignored byte range {request.offset}-{request.end - 1} "
                            f"for {request.source}: status {response.status_code}"
                        )
                content_range = response.headers.get("Content-Range")
                if content_range is not None:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(?:\d+|\*)", content_range)
                    if match is None or (
                        int(match.group(1)) != request.offset
                        or int(match.group(2)) != request.end - 1
                    ):
                        raise RemoteRangeReadError(
                            f"Unexpected Content-Range for {request.source}: "
                            f"{content_range!r}"
                        )
            if len(data) != request.length:
                raise RemoteRangeReadError(
                    f"Unexpected byte count for {request.source} at {request.offset}: "
                    f"received {len(data)}, expected {request.length}"
                )
            return FetchedByteRangeRequest(
                request=request,
                data=data,
                attempt_count=attempt_count,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as error:  # noqa: BLE001 - URL resolvers may raise arbitrary exceptions
            last_error = error
            if attempt == options.max_retries:
                break
            time.sleep(options.retry_backoff_seconds * (2 ** attempt))

    error_detail = f": {last_error}" if last_error is not None else ""
    raise RemoteRangeReadError(
        f"Failed to read bytes {request.offset}-{request.end - 1} "
        f"from {request.source} after {attempt_count} attempts{error_detail}"
    ) from last_error
