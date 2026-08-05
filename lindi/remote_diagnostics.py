import contextvars
import inspect
import json
import os
import threading
import time
import warnings
from dataclasses import dataclass, field, replace
from typing import Any, List, Sequence, Set, Tuple, Union

import numpy as np

from .remote_reading import (
    ByteRange,
    LindiRemoteEfficiencyWarning,
    LindiRemoteLayoutError,
    LindiRemoteLayoutWarning,
    LindiRemoteReadMetrics,
    LindiRemoteReadOptions,
    plan_byte_range_requests,
)


@dataclass
class _RemoteReadQueryCapture:
    options: LindiRemoteReadOptions
    started: float
    metrics: List[LindiRemoteReadMetrics] = field(default_factory=list)
    remote_keys: Set[str] = field(default_factory=set)


_active_remote_read_query = contextvars.ContextVar(
    "lindi_active_remote_read_query", default=None
)
_lindi_package_directory = os.path.dirname(os.path.abspath(__file__))


def start_remote_read_query(options: LindiRemoteReadOptions):
    capture = _RemoteReadQueryCapture(options=options, started=time.perf_counter())
    token = _active_remote_read_query.set(capture)
    return token, capture


def abort_remote_read_query(token) -> None:
    _active_remote_read_query.reset(token)


def capture_remote_read_metrics(
    metrics: LindiRemoteReadMetrics,
    *,
    options: LindiRemoteReadOptions,
    remote_keys: Sequence[str],
) -> bool:
    capture = _active_remote_read_query.get()
    if capture is None or capture.options is not options:
        return False
    capture.metrics.append(metrics)
    capture.remote_keys.update(remote_keys)
    return True


def complete_remote_read_query(
    token,
    capture: _RemoteReadQueryCapture,
    *,
    dataset_path: str,
    zarr_array: Any,
    result: Any,
    warned_categories: Set[Tuple[str, str]],
    warning_lock: threading.Lock,
) -> None:
    _active_remote_read_query.reset(token)
    if not capture.metrics:
        return

    logical_bytes = _logical_result_bytes(result)
    dataset_logical_bytes = _logical_dataset_bytes(zarr_array)
    decoded_chunk_bytes = _decoded_remote_chunk_bytes(
        zarr_array, capture.remote_keys
    )
    chunk_amplification_ratio = None
    if logical_bytes and decoded_chunk_bytes is not None:
        chunk_amplification_ratio = decoded_chunk_bytes / logical_bytes

    requested_bytes = sum(x.requested_bytes for x in capture.metrics)
    fetched_bytes = sum(x.fetched_bytes for x in capture.metrics)
    transport_overread_ratio = None
    if requested_bytes:
        transport_overread_ratio = fetched_bytes / requested_bytes

    metrics = LindiRemoteReadMetrics(
        optimized=all(x.optimized for x in capture.metrics),
        requested_key_count=sum(x.requested_key_count for x in capture.metrics),
        returned_key_count=sum(x.returned_key_count for x in capture.metrics),
        inline_value_count=sum(x.inline_value_count for x in capture.metrics),
        local_file_read_count=sum(x.local_file_read_count for x in capture.metrics),
        remote_reference_count=sum(x.remote_reference_count for x in capture.metrics),
        cache_hit_count=sum(x.cache_hit_count for x in capture.metrics),
        source_range_count=sum(x.source_range_count for x in capture.metrics),
        planned_request_count=sum(x.planned_request_count for x in capture.metrics),
        range_get_count=sum(x.range_get_count for x in capture.metrics),
        requested_bytes=requested_bytes,
        fetched_bytes=fetched_bytes,
        overread_bytes=sum(x.overread_bytes for x in capture.metrics),
        elapsed_seconds=sum(x.elapsed_seconds for x in capture.metrics),
        referenced_bytes=sum(x.referenced_bytes for x in capture.metrics),
        cache_hit_bytes=sum(x.cache_hit_bytes for x in capture.metrics),
        dataset_path=dataset_path,
        logical_bytes=logical_bytes,
        decoded_chunk_bytes=decoded_chunk_bytes,
        chunk_amplification_ratio=chunk_amplification_ratio,
        transport_overread_ratio=transport_overread_ratio,
        query_elapsed_seconds=time.perf_counter() - capture.started,
        dataset_logical_bytes=dataset_logical_bytes,
    )
    if capture.options.metrics_callback is not None:
        capture.options.metrics_callback(metrics)
    _warn_for_query_inefficiency(
        metrics,
        capture.options,
        warned_categories=warned_categories,
        warning_lock=warning_lock,
    )


def _logical_result_bytes(result: Any) -> Union[int, None]:
    try:
        array = np.asarray(result)
    except Exception:  # noqa: BLE001 - custom reference objects may reject conversion
        return None
    if array.dtype.hasobject:
        return None
    return int(array.nbytes)


def _logical_dataset_bytes(zarr_array: Any) -> Union[int, None]:
    dtype = zarr_array.dtype
    if dtype.hasobject or dtype.itemsize <= 0:
        return None
    element_count = 1
    for size in zarr_array.shape:
        element_count *= size
    return int(element_count * dtype.itemsize)


def _decoded_remote_chunk_bytes(
    zarr_array: Any, remote_keys: Set[str]
) -> Union[int, None]:
    dtype = zarr_array.dtype
    if dtype.hasobject or dtype.itemsize <= 0 or zarr_array.ndim == 0:
        return None
    prefix = zarr_array.name.strip("/")
    chunk_keys = set()
    for key in remote_keys:
        if prefix:
            prefix_with_separator = prefix + "/"
            if not key.startswith(prefix_with_separator):
                continue
            relative_key = key[len(prefix_with_separator):]
        else:
            relative_key = key
        coordinates = (
            relative_key.split(".") if "." in relative_key else relative_key.split("/")
        )
        if len(coordinates) != zarr_array.ndim:
            continue
        try:
            tuple(int(x) for x in coordinates)
        except ValueError:
            continue
        chunk_keys.add(key)
    if not chunk_keys:
        return None
    chunk_elements = 1
    for size in zarr_array.chunks:
        chunk_elements *= size
    return int(len(chunk_keys) * chunk_elements * dtype.itemsize)


def _warn_for_query_inefficiency(
    metrics: LindiRemoteReadMetrics,
    options: LindiRemoteReadOptions,
    *,
    warned_categories: Set[Tuple[str, str]],
    warning_lock: threading.Lock,
) -> None:
    if options.runtime_efficiency_check == "ignore":
        return
    if metrics.remote_reference_count == 0 or metrics.dataset_path is None:
        return
    thresholds = options.efficiency_warning_thresholds
    problems: List[Tuple[str, str]] = []
    is_partial_selection = (
        metrics.dataset_logical_bytes is None
        or metrics.logical_bytes is None
        or metrics.logical_bytes < metrics.dataset_logical_bytes
    )
    if (
        is_partial_selection
        and metrics.chunk_amplification_ratio is not None
        and metrics.decoded_chunk_bytes is not None
        and metrics.logical_bytes is not None
        and metrics.chunk_amplification_ratio >= thresholds.chunk_amplification_ratio
        and metrics.decoded_chunk_bytes - metrics.logical_bytes
        >= thresholds.chunk_excess_bytes
    ):
        problems.append((
            "chunk_amplification",
            f"chunk amplification was {metrics.chunk_amplification_ratio:.1f}x "
            f"({metrics.decoded_chunk_bytes} decoded bytes for "
            f"{metrics.logical_bytes} returned bytes)",
        ))
    if (
        metrics.transport_overread_ratio is not None
        and metrics.transport_overread_ratio >= thresholds.transport_overread_ratio
        and metrics.overread_bytes >= thresholds.transport_overread_bytes
    ):
        problems.append((
            "transport_overread",
            f"transport over-read was {metrics.transport_overread_ratio:.1f}x "
            f"({metrics.overread_bytes} unnecessary fetched bytes)",
        ))
    if metrics.planned_request_count >= thresholds.fragmented_request_count:
        problems.append((
            "request_fragmentation",
            f"the selection required {metrics.planned_request_count} independent "
            "range requests",
        ))

    new_problems = []
    with warning_lock:
        for category, detail in problems:
            key = (metrics.dataset_path, category)
            if key not in warned_categories:
                warned_categories.add(key)
                new_problems.append(detail)
    if new_problems:
        _warn_at_user_callsite(
            f"Inefficient remote read from {metrics.dataset_path}: "
            + "; ".join(new_problems)
            + ". Consider changing the dataset chunk layout or selection pattern.",
            LindiRemoteEfficiencyWarning,
        )


def validate_nwb_electrical_series_layout(store: Any) -> None:
    """Validate remote NWB ElectricalSeries references without fetching data."""
    options = store.remote_read_options
    if not options.enabled or options.layout_check == "ignore":
        return
    refs = store.rfs["refs"]
    electrical_series_paths = []
    for key, value in refs.items():
        if not key.endswith("/.zattrs"):
            continue
        attrs = _metadata_dict(value)
        if attrs is not None and attrs.get("neurodata_type") == "ElectricalSeries":
            electrical_series_paths.append(key[:-len("/.zattrs")])

    issues = []
    remaining_reference_checks = len(refs)
    layout_options = replace(
        options,
        max_request_bytes=None,
        metrics_callback=None,
        runtime_efficiency_check="ignore",
    )
    for series_path in electrical_series_paths:
        data_path = f"{series_path}/data"
        zarray = _metadata_dict(refs.get(f"{data_path}/.zarray"))
        data_attrs = _metadata_dict(refs.get(f"{data_path}/.zattrs")) or {}
        external_link = data_attrs.get("_EXTERNAL_ARRAY_LINK")
        if isinstance(external_link, dict):
            url = external_link.get("url")
            if isinstance(url, str) and url.startswith(("http://", "https://")):
                issues.append(
                    f"{data_path} uses _EXTERNAL_ARRAY_LINK, so explicit range "
                    "planning and layout validation are unavailable"
                )
            continue
        if zarray is None:
            continue
        shape = zarray.get("shape")
        chunks = zarray.get("chunks")
        if not _valid_two_dimensional_layout(shape, chunks):
            continue
        if shape[1] == 0 or shape[0] == 0:
            continue
        time_chunk_count = (shape[0] + chunks[0] - 1) // chunks[0]
        channel_chunk_count = (shape[1] + chunks[1] - 1) // chunks[1]
        required_checks = time_chunk_count * channel_chunk_count
        if required_checks > remaining_reference_checks:
            issues.append(
                f"{data_path} could not be safely verified because its advertised "
                "chunk grid exceeds the available reference metadata"
            )
            continue
        remaining_reference_checks -= required_checks
        dimension_separator = zarray.get("dimension_separator", ".")
        remote_reference_count = 0
        fragmented_channel_count = 0
        maximum_run_count = 1
        for channel_chunk in range(channel_chunk_count):
            channel_ranges = []
            for time_chunk in range(time_chunk_count):
                coordinates = (
                    f"{time_chunk}.{channel_chunk}"
                    if dimension_separator == "."
                    else f"{time_chunk}/{channel_chunk}"
                )
                key = f"{data_path}/{coordinates}"
                try:
                    reference = store._resolve_reference(key)
                except KeyError:
                    continue
                if not reference.is_remote or reference.length == 0:
                    continue
                remote_reference_count += 1
                channel_ranges.append(ByteRange(
                    key=key,
                    source=reference.url_or_path,
                    offset=reference.offset,
                    length=reference.length,
                ))
            if chunks[1] == 1 and len(channel_ranges) > 1:
                run_count = len(plan_byte_range_requests(channel_ranges, layout_options))
                if run_count > 1:
                    fragmented_channel_count += 1
                    maximum_run_count = max(maximum_run_count, run_count)

        if remote_reference_count == 0:
            continue
        channels_per_chunk = min(chunks[1], shape[1])
        if channels_per_chunk != 1:
            issues.append(
                f"{data_path} has chunk shape {tuple(chunks)}, so reading one "
                f"channel fetches and decodes data for up to {channels_per_chunk} channels"
            )
        elif fragmented_channel_count:
            issues.append(
                f"{data_path} is chunked per channel but is not channel-major under "
                f"the configured coalescing policy; {fragmented_channel_count} channel "
                f"chunk sequences are split into as many as {maximum_run_count} byte runs"
            )

    if not issues:
        return
    message = (
        "Remote NWB layout check found inefficient ElectricalSeries data: "
        + "; ".join(issues)
        + ". Repack the source into time-by-one-channel chunks stored channel-major "
        "and regenerate the LINDI sidecar with explicit chunk references. Set "
        "layout_check='ignore' only when this behavior is intentional."
    )
    if options.layout_check == "error":
        raise LindiRemoteLayoutError(message)
    _warn_at_user_callsite(message, LindiRemoteLayoutWarning)


def _valid_two_dimensional_layout(shape: Any, chunks: Any) -> bool:
    return (
        isinstance(shape, (list, tuple))
        and isinstance(chunks, (list, tuple))
        and len(shape) == 2
        and len(chunks) == 2
        and all(type(x) is int and x >= 0 for x in shape)
        and all(type(x) is int and x > 0 for x in chunks)
    )


def _metadata_dict(value: Any) -> Union[dict, None]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and not value.startswith("base64:"):
        try:
            parsed = json.loads(value)
        except Exception:  # noqa: BLE001 - malformed metadata is handled elsewhere
            return None
        if isinstance(parsed, dict):
            return parsed
    return None


def _warn_at_user_callsite(message: str, category: type) -> None:
    frame = inspect.currentframe()
    stacklevel = 1
    while frame is not None:
        filename = os.path.abspath(frame.f_code.co_filename)
        if not filename.startswith(_lindi_package_directory + os.sep):
            break
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, category, stacklevel=stacklevel)
