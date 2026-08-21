#!/usr/bin/env python3
"""Run reproducible LINDI remote-reader windows and emit JSON evidence."""

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from urllib.parse import urlsplit, urlunsplit

import numpy as np

import lindi


def _parse_channels(value):
    if value is None or value.lower() == "all":
        return slice(None)
    return [int(x) for x in value.split(",") if x.strip()]


def _redact_url(value):
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def _selection_for_dataset(dataset, args):
    if args.time_axis >= dataset.ndim or args.channel_axis >= dataset.ndim:
        raise ValueError(
            f"Axes ({args.time_axis}, {args.channel_axis}) do not fit shape {dataset.shape}"
        )
    if args.time_axis == args.channel_axis:
        raise ValueError("time-axis and channel-axis must be different")
    selection = [slice(None)] * dataset.ndim
    selection[args.time_axis] = slice(args.start, args.stop)
    selection[args.channel_axis] = _parse_channels(args.channels)
    return tuple(selection)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lindi-url", default=os.environ.get("LINDI_URL"), required=os.environ.get("LINDI_URL") is None)
    parser.add_argument("--dataset", default=os.environ.get("LINDI_DATASET"), required=os.environ.get("LINDI_DATASET") is None)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--stop", type=int, required=True)
    parser.add_argument("--channels", default="all", help="Comma-separated indices or 'all'")
    parser.add_argument("--time-axis", type=int, default=0)
    parser.add_argument("--channel-axis", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-gap-bytes", type=int, default=0)
    parser.add_argument("--max-overread-bytes", type=int, default=0)
    parser.add_argument("--max-request-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--request-timeout-seconds", type=float)
    parser.add_argument("--baseline", action="store_true", help="One sequential GET per referenced chunk")
    parser.add_argument("--cache-dir")
    parser.add_argument("--label", default="")
    parser.add_argument("--output", help="Write the JSON result to this path")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.stop <= args.start:
        parser.error("--stop must be greater than --start")

    metrics = []
    options = lindi.LindiRangeReadOptions(
        enabled=not args.baseline,
        max_gap_bytes=args.max_gap_bytes,
        max_overread_bytes=args.max_overread_bytes,
        max_request_bytes=args.max_request_bytes,
        max_concurrency=args.max_concurrency,
        max_retries=args.max_retries,
        request_timeout_seconds=args.request_timeout_seconds,
        metrics_callback=metrics.append,
    )
    cache = lindi.LocalCache(cache_dir=args.cache_dir) if args.cache_dir else None
    client = lindi.LindiH5pyFile.from_lindi_file(
        args.lindi_url,
        local_cache=cache,
        range_read_options=options,
    )
    try:
        dataset = client[args.dataset]
        external_array_link = dataset._zarr_array.attrs.get(
            "_EXTERNAL_ARRAY_LINK", None
        )
        if external_array_link is not None:
            raise RuntimeError(
                "Benchmark dataset uses _EXTERNAL_ARRAY_LINK; regenerate LINDI "
                "with force_explicit_chunk_references for this dataset"
            )
        selection = _selection_for_dataset(dataset, args)
        iterations = []
        for iteration in range(args.warmup + args.iterations):
            metrics_start = len(metrics)
            started = time.perf_counter()
            result = dataset[selection]
            elapsed = time.perf_counter() - started
            result_array = np.asarray(result)
            record = {
                "iteration": iteration - args.warmup,
                "warmup": iteration < args.warmup,
                "elapsed_seconds": elapsed,
                "result_bytes": int(result_array.nbytes),
                "result_shape": list(result_array.shape),
                "throughput_mib_per_second": (
                    result_array.nbytes / (1024 ** 2) / elapsed if elapsed else None
                ),
                "reads": [asdict(x) for x in metrics[metrics_start:]],
            }
            iterations.append(record)
        measured = [x for x in iterations if not x["warmup"]]
        elapsed_values = [x["elapsed_seconds"] for x in measured]
        output = {
            "schema_version": 1,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "label": args.label,
            "host": platform.node(),
            "python": platform.python_version(),
            "lindi_url": _redact_url(args.lindi_url),
            "dataset": args.dataset,
            "dataset_shape": list(dataset.shape),
            "dataset_chunks": list(dataset.chunks),
            "dataset_dtype": str(dataset.dtype),
            "selection": {
                "start": args.start,
                "stop": args.stop,
                "channels": args.channels,
                "time_axis": args.time_axis,
                "channel_axis": args.channel_axis,
            },
            "options": {
                key: value
                for key, value in asdict(options).items()
                if key != "metrics_callback"
            },
            "summary": {
                "iterations": len(measured),
                "mean_elapsed_seconds": statistics.mean(elapsed_values),
                "median_elapsed_seconds": statistics.median(elapsed_values),
                "min_elapsed_seconds": min(elapsed_values),
                "max_elapsed_seconds": max(elapsed_values),
                "total_read_attempts": sum(
                    read["read_attempt_count"]
                    for iteration in measured
                    for read in iteration["reads"]
                ),
                "total_fetched_bytes": sum(
                    read["fetched_bytes"]
                    for iteration in measured
                    for read in iteration["reads"]
                ),
                "total_overread_bytes": sum(
                    read["overread_bytes"]
                    for iteration in measured
                    for read in iteration["reads"]
                ),
            },
            "iterations": iterations,
        }
    finally:
        client.close()

    rendered = json.dumps(output, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        with open(args.output, "w") as file:
            file.write(rendered + "\n")


if __name__ == "__main__":
    main()
