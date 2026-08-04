#!/usr/bin/env python3
"""Benchmark random remote signal chunk reads through LINDI's range executor.

This intentionally operates on external chunk references rather than decoding
the Zarr array. It therefore isolates the remote range planning and transport
path and can byte-compare an optimized fetch with the one-GET-per-chunk
baseline even when a dataset uses a site-specific NumCodecs filter.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urlsplit

import requests

from lindi.remote_reading import (
    ByteRange,
    FetchedByteRangeRequest,
    LindiRemoteReadOptions,
    fetch_byte_range_requests,
    plan_byte_range_requests,
)


_TEMPLATE_PATTERN = re.compile(r"\{\{([^{}]+)\}\}")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lindi-url", required=True)
    parser.add_argument(
        "--dataset", default="/acquisition/ElectricalSeries/data"
    )
    parser.add_argument("--pattern", choices=("single", "all"), required=True)
    parser.add_argument("--window-samples", type=_positive_int, required=True)
    parser.add_argument("--iterations", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--channel", type=_nonnegative_int)
    parser.add_argument(
        "--strategy",
        choices=("baseline", "optimized", "paired"),
        default="optimized",
    )
    parser.add_argument("--time-axis", type=_nonnegative_int, default=0)
    parser.add_argument("--channel-axis", type=_nonnegative_int, default=1)
    parser.add_argument("--max-gap-bytes", type=_nonnegative_int, default=0)
    parser.add_argument(
        "--max-overread-bytes", type=_nonnegative_int, default=0
    )
    parser.add_argument(
        "--max-request-bytes", type=_positive_int, default=512 * 1024 * 1024
    )
    parser.add_argument("--max-concurrency", type=_positive_int, default=8)
    parser.add_argument("--max-retries", type=_nonnegative_int, default=3)
    parser.add_argument("--retry-backoff-seconds", type=float, default=0.1)
    parser.add_argument("--request-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--metadata-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--label", default="")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.time_axis == args.channel_axis:
        parser.error("--time-axis and --channel-axis must differ")
    if args.channel is not None and args.pattern != "single":
        parser.error("--channel is valid only with --pattern single")
    if args.retry_backoff_seconds < 0:
        parser.error("--retry-backoff-seconds must be nonnegative")
    if args.request_timeout_seconds <= 0 or args.metadata_timeout_seconds <= 0:
        parser.error("timeouts must be greater than zero")
    return args


def _load_sidecar(url: str, timeout_seconds: float) -> tuple[dict[str, Any], int, float]:
    if urlsplit(url).scheme not in {"http", "https"}:
        raise ValueError("the benchmark accepts only read-only HTTP(S) sidecars")
    started = time.perf_counter()
    response = requests.get(url, timeout=timeout_seconds)
    response.raise_for_status()
    content = response.content
    elapsed = time.perf_counter() - started
    document = json.loads(content)
    if not isinstance(document, dict) or not isinstance(document.get("refs"), dict):
        raise ValueError("invalid LINDI reference document")
    return document, len(content), elapsed


def _metadata_object(value: Any, key: str) -> dict[str, Any]:
    if isinstance(value, str):
        value = json.loads(value)
    if not isinstance(value, dict):
        raise ValueError(f"{key} must contain an object")
    return value


def _expand_template(value: str, templates: dict[str, Any]) -> str:
    expanded = value
    for _ in range(16):
        match = _TEMPLATE_PATTERN.search(expanded)
        if match is None:
            break
        name = match.group(1)
        replacement = templates.get(name)
        if not isinstance(replacement, str):
            raise ValueError(f"missing string template {name!r}")
        expanded = expanded[: match.start()] + replacement + expanded[match.end() :]
    if _TEMPLATE_PATTERN.search(expanded):
        raise ValueError(f"template expansion did not converge for {value!r}")
    if urlsplit(expanded).scheme not in {"http", "https"}:
        raise ValueError(f"chunk source is not read-only HTTP(S): {expanded!r}")
    return expanded


def _chunk_coordinates(
    *,
    shape: Sequence[int],
    chunks: Sequence[int],
    start: int,
    stop: int,
    pattern: str,
    channel: int | None,
    time_axis: int,
    channel_axis: int,
) -> Iterable[tuple[int, ...]]:
    if len(shape) != 2 or len(chunks) != 2:
        raise ValueError("the random signal benchmark currently requires a 2-D array")
    time_chunks = range(start // chunks[time_axis], (stop - 1) // chunks[time_axis] + 1)
    if pattern == "all":
        channel_chunks = range(
            (shape[channel_axis] + chunks[channel_axis] - 1) // chunks[channel_axis]
        )
    else:
        if channel is None:
            raise ValueError("single-channel selection requires a channel")
        channel_chunks = (channel // chunks[channel_axis],)
    for time_chunk, channel_chunk in itertools.product(time_chunks, channel_chunks):
        coordinates = [0, 0]
        coordinates[time_axis] = time_chunk
        coordinates[channel_axis] = channel_chunk
        yield tuple(coordinates)


def _selection_ranges(
    *,
    refs: dict[str, Any],
    templates: dict[str, Any],
    dataset_key: str,
    zarray: dict[str, Any],
    start: int,
    stop: int,
    pattern: str,
    channel: int | None,
    time_axis: int,
    channel_axis: int,
) -> list[ByteRange]:
    shape = tuple(int(value) for value in zarray["shape"])
    chunks = tuple(int(value) for value in zarray["chunks"])
    separator = zarray.get("dimension_separator", ".")
    if separator not in {".", "/"}:
        raise ValueError(f"unsupported Zarr dimension separator: {separator!r}")
    ranges = []
    for coordinates in _chunk_coordinates(
        shape=shape,
        chunks=chunks,
        start=start,
        stop=stop,
        pattern=pattern,
        channel=channel,
        time_axis=time_axis,
        channel_axis=channel_axis,
    ):
        chunk_id = separator.join(str(value) for value in coordinates)
        key = f"{dataset_key}/{chunk_id}"
        reference = refs.get(key)
        if not isinstance(reference, list) or len(reference) != 3:
            raise ValueError(f"expected an explicit three-part reference for {key!r}")
        source, offset, length = reference
        if not isinstance(source, str):
            raise ValueError(f"reference source for {key!r} must be a string")
        ranges.append(
            ByteRange(
                key=key,
                source=_expand_template(source, templates),
                offset=int(offset),
                length=int(length),
            )
        )
    return ranges


def _digest_fetched_ranges(
    requested_ranges: Sequence[ByteRange],
    fetched: Sequence[FetchedByteRangeRequest],
) -> str:
    values: dict[str, bytes] = {}
    for item in fetched:
        request = item.request
        for requested_range in request.ranges:
            relative_offset = requested_range.offset - request.offset
            value = item.data[
                relative_offset : relative_offset + requested_range.length
            ]
            if len(value) != requested_range.length:
                raise RuntimeError(f"short reconstructed range for {requested_range.key!r}")
            if requested_range.key in values:
                raise RuntimeError(f"duplicate reconstructed range {requested_range.key!r}")
            values[requested_range.key] = value
    digest = hashlib.sha256()
    for requested_range in requested_ranges:
        value = values.get(requested_range.key)
        if value is None:
            raise RuntimeError(f"missing reconstructed range {requested_range.key!r}")
        digest.update(requested_range.key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value)
    return digest.hexdigest()


def _run_strategy(
    ranges: Sequence[ByteRange],
    *,
    optimized: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    options = LindiRemoteReadOptions(
        enabled=optimized,
        max_gap_bytes=args.max_gap_bytes,
        max_overread_bytes=args.max_overread_bytes,
        max_request_bytes=args.max_request_bytes,
        max_concurrency=args.max_concurrency if optimized else 1,
        max_retries=args.max_retries,
        retry_backoff_seconds=args.retry_backoff_seconds,
        request_timeout_seconds=args.request_timeout_seconds,
    )
    planning_started = time.perf_counter()
    planned = plan_byte_range_requests(ranges, options)
    planning_seconds = time.perf_counter() - planning_started
    fetch_started = time.perf_counter()
    fetched = fetch_byte_range_requests(planned, options)
    fetch_seconds = time.perf_counter() - fetch_started
    requested_bytes = sum(item.length for item in ranges)
    fetched_bytes = sum(len(item.data) for item in fetched)
    request_count = sum(item.attempt_count for item in fetched)
    return {
        "strategy": "optimized" if optimized else "baseline",
        "source_range_count": len(ranges),
        "planned_request_count": len(planned),
        "range_get_count": request_count,
        "retry_count": request_count - len(planned),
        "requested_bytes": requested_bytes,
        "fetched_bytes": fetched_bytes,
        "overread_bytes": fetched_bytes - requested_bytes,
        "request_reduction_fraction": (
            1.0 - (len(planned) / len(ranges)) if ranges else 0.0
        ),
        "planning_seconds": planning_seconds,
        "fetch_seconds": fetch_seconds,
        "requested_mib_per_second": (
            requested_bytes / (1024 * 1024) / fetch_seconds if fetch_seconds else None
        ),
        "fetched_mib_per_second": (
            fetched_bytes / (1024 * 1024) / fetch_seconds if fetch_seconds else None
        ),
        "sha256": _digest_fetched_ranges(ranges, fetched),
    }


def _main() -> None:
    args = _parse_args()
    document, metadata_bytes, metadata_seconds = _load_sidecar(
        args.lindi_url, args.metadata_timeout_seconds
    )
    refs = document["refs"]
    templates = document.get("templates") or {}
    if not isinstance(templates, dict):
        raise ValueError("LINDI templates must be an object")
    dataset_key = args.dataset.strip("/")
    zarray_key = f"{dataset_key}/.zarray"
    if zarray_key not in refs:
        raise KeyError(f"dataset metadata not found: {zarray_key}")
    zarray = _metadata_object(refs[zarray_key], zarray_key)
    shape = tuple(int(value) for value in zarray["shape"])
    chunks = tuple(int(value) for value in zarray["chunks"])
    if max(args.time_axis, args.channel_axis) >= len(shape):
        raise ValueError("axis is outside the dataset rank")
    time_length = shape[args.time_axis]
    channel_count = shape[args.channel_axis]
    if args.window_samples > time_length:
        raise ValueError("window is longer than the time axis")
    if args.channel is not None and args.channel >= channel_count:
        raise ValueError("channel is outside the channel axis")

    print(
        json.dumps(
            {
                "event": "benchmark_start",
                "label": args.label,
                "pattern": args.pattern,
                "strategy": args.strategy,
                "iterations": args.iterations,
                "seed": args.seed,
                "shape": shape,
                "chunks": chunks,
                "metadata_bytes": metadata_bytes,
                "metadata_seconds": metadata_seconds,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    rng = random.Random(args.seed)
    measurements = []
    for iteration in range(args.iterations):
        start = rng.randrange(time_length - args.window_samples + 1)
        stop = start + args.window_samples
        channel = None
        if args.pattern == "single":
            channel = args.channel if args.channel is not None else rng.randrange(channel_count)
        ranges = _selection_ranges(
            refs=refs,
            templates=templates,
            dataset_key=dataset_key,
            zarray=zarray,
            start=start,
            stop=stop,
            pattern=args.pattern,
            channel=channel,
            time_axis=args.time_axis,
            channel_axis=args.channel_axis,
        )
        print(
            json.dumps(
                {
                    "event": "iteration_start",
                    "iteration": iteration,
                    "start": start,
                    "stop": stop,
                    "channel": channel,
                    "source_range_count": len(ranges),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        measurement: dict[str, Any] = {
            "iteration": iteration,
            "start": start,
            "stop": stop,
            "channel": channel,
            "selected_channel_count": 1 if args.pattern == "single" else channel_count,
        }
        if args.strategy in {"baseline", "paired"}:
            measurement["baseline"] = _run_strategy(
                ranges, optimized=False, args=args
            )
        if args.strategy in {"optimized", "paired"}:
            measurement["optimized"] = _run_strategy(
                ranges, optimized=True, args=args
            )
        if args.strategy == "paired":
            measurement["byte_exact"] = (
                measurement["baseline"]["sha256"]
                == measurement["optimized"]["sha256"]
            )
            if not measurement["byte_exact"]:
                raise RuntimeError("optimized bytes differ from the sequential baseline")
        measurements.append(measurement)
        print(
            json.dumps(
                {"event": "iteration_complete", **measurement}, sort_keys=True
            ),
            flush=True,
        )

    report = {
        "event": "benchmark_summary",
        "schema_version": 1,
        "label": args.label,
        "lindi_url": args.lindi_url,
        "dataset": "/" + dataset_key,
        "pattern": args.pattern,
        "strategy": args.strategy,
        "window_samples": args.window_samples,
        "iterations": args.iterations,
        "seed": args.seed,
        "shape": shape,
        "chunks": chunks,
        "metadata_bytes": metadata_bytes,
        "metadata_seconds": metadata_seconds,
        "max_gap_bytes": args.max_gap_bytes,
        "max_overread_bytes": args.max_overread_bytes,
        "max_request_bytes": args.max_request_bytes,
        "max_concurrency": args.max_concurrency,
        "measurements": measurements,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized, flush=True)


if __name__ == "__main__":
    _main()

