import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import zlib
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import h5py
import numpy as np
import pytest

import lindi
from lindi.remote_reading import (
    ByteRange,
    PlannedByteRangeRequest,
    RemoteRangeReadError,
    fetch_byte_range_requests,
    plan_byte_range_requests,
)


class _RangeServer:
    def __init__(self, *, delay_seconds=0):
        self.data = b""
        self.delay_seconds = delay_seconds
        self.fail_requests = 0
        self.truncate_requests = 0
        self.response_status = 206
        self.omit_content_range = False
        self.content_range_offset_delta = 0
        self.requests = []
        self.active_requests = 0
        self.max_active_requests = 0
        self._lock = threading.Lock()
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                range_header = self.headers.get("Range", "")
                match = re.fullmatch(r"bytes=(\d+)-(\d+)", range_header)
                if match is None:
                    self.send_error(400, "A single byte range is required")
                    return
                start = int(match.group(1))
                end = int(match.group(2))
                with owner._lock:
                    owner.requests.append((start, end))
                    owner.active_requests += 1
                    owner.max_active_requests = max(
                        owner.max_active_requests, owner.active_requests
                    )
                    should_fail = owner.fail_requests > 0
                    if should_fail:
                        owner.fail_requests -= 1
                    should_truncate = owner.truncate_requests > 0
                    if should_truncate:
                        owner.truncate_requests -= 1
                try:
                    if owner.delay_seconds:
                        time.sleep(owner.delay_seconds)
                    if should_fail:
                        self.send_error(500, "Imposed test failure")
                        return
                    payload = owner.data[start:end + 1]
                    if should_truncate and payload:
                        payload = payload[:-1]
                    self.send_response(owner.response_status)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(len(payload)))
                    if not owner.omit_content_range and owner.response_status == 206:
                        content_start = start + owner.content_range_offset_delta
                        self.send_header(
                            "Content-Range",
                            f"bytes {content_start}-{content_start + len(payload) - 1}/{len(owner.data)}",
                        )
                    self.end_headers()
                    self.wfile.write(payload)
                finally:
                    with owner._lock:
                        owner.active_requests -= 1

            def log_message(self, format, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def url(self):
        host, port = self._server.server_address
        return f"http://{host}:{port}/object.bin"

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()

    def reset_observations(self):
        with self._lock:
            self.requests = []
            self.max_active_requests = self.active_requests


def _make_reference_file_system(
    data,
    url,
    *,
    chunks=(4, 2),
    gap_bytes=0,
    fill_value=0,
    omitted_chunks=(),
    compressor=None,
    truncate_edge_chunks=False,
):
    data = np.asarray(data)
    dtype = data.dtype
    refs = {
        ".zgroup": {"zarr_format": 2},
        "data/.zarray": {
            "chunks": list(chunks),
            "compressor": (
                {"id": "zlib", "level": 1} if compressor == "zlib" else None
            ),
            "dtype": dtype.str,
            "fill_value": fill_value,
            "filters": None,
            "order": "C",
            "shape": list(data.shape),
            "zarr_format": 2,
        },
        "data/.zattrs": {},
    }
    object_parts = [b"LINDI-TEST-OBJECT"]
    offset = len(object_parts[0])
    num_chunk_rows = (data.shape[0] + chunks[0] - 1) // chunks[0]
    num_chunk_columns = (data.shape[1] + chunks[1] - 1) // chunks[1]
    omitted_chunks = set(omitted_chunks)
    for chunk_row in range(num_chunk_rows):
        for chunk_column in range(num_chunk_columns):
            coords = (chunk_row, chunk_column)
            if coords in omitted_chunks:
                continue
            row_start = chunk_row * chunks[0]
            column_start = chunk_column * chunks[1]
            actual = data[
                row_start:min(row_start + chunks[0], data.shape[0]),
                column_start:min(column_start + chunks[1], data.shape[1]),
            ]
            if truncate_edge_chunks:
                chunk = np.ascontiguousarray(actual)
            else:
                chunk = np.full(chunks, fill_value, dtype=dtype)
                chunk[:actual.shape[0], :actual.shape[1]] = actual
            payload = chunk.tobytes(order="C")
            if compressor == "zlib":
                payload = zlib.compress(payload, level=1)
            key = f"data/{chunk_row}.{chunk_column}"
            refs[key] = [url, offset, len(payload)]
            object_parts.append(payload)
            offset += len(payload)
            if gap_bytes:
                gap = bytes([chunk_row + chunk_column + 1]) * gap_bytes
                object_parts.append(gap)
                offset += len(gap)
    return {"refs": refs, "version": 1}, b"".join(object_parts)


def _open_remote_dataset(server, data, *, options, **rfs_kwargs):
    rfs, object_data = _make_reference_file_system(
        data, server.url, **rfs_kwargs
    )
    server.data = object_data
    client = lindi.LindiH5pyFile.from_reference_file_system(
        rfs, remote_read_options=options
    )
    return client, client["data"]


def test_range_planner_coalesces_with_bounded_overread():
    ranges = [
        ByteRange("a", "u1", 0, 10),
        ByteRange("b", "u1", 15, 10),
        ByteRange("c", "u1", 32, 10),
        ByteRange("d", "u2", 0, 10),
    ]
    options = lindi.LindiRemoteReadOptions(
        max_gap_bytes=10,
        max_overread_bytes=10,
        max_request_bytes=None,
    )
    planned = plan_byte_range_requests(ranges, options)
    assert [(x.source, x.offset, x.length) for x in planned] == [
        ("u1", 0, 25),
        ("u1", 32, 10),
        ("u2", 0, 10),
    ]
    assert [x.overread_bytes for x in planned] == [5, 0, 0]


def test_range_planner_handles_overlap_size_limit_and_disabled_mode():
    ranges = [
        ByteRange("a", "u", 0, 10),
        ByteRange("b", "u", 5, 10),
        ByteRange("c", "u", 15, 10),
    ]
    planned = plan_byte_range_requests(
        ranges,
        lindi.LindiRemoteReadOptions(
            max_gap_bytes=0,
            max_overread_bytes=0,
            max_request_bytes=20,
        ),
    )
    assert [(x.offset, x.length, x.requested_bytes) for x in planned] == [
        (0, 15, 15),
        (15, 10, 10),
    ]
    baseline = plan_byte_range_requests(
        ranges, lindi.LindiRemoteReadOptions(enabled=False)
    )
    assert len(baseline) == 3
    assert all(len(x.ranges) == 1 for x in baseline)


def test_byte_range_validation():
    with pytest.raises(ValueError):
        ByteRange("a", "u", -1, 1)
    with pytest.raises(ValueError):
        ByteRange("a", "u", 0, 0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_gap_bytes": -1},
        {"max_overread_bytes": -1},
        {"max_request_bytes": 0},
        {"max_concurrency": 0},
        {"max_retries": -1},
        {"retry_backoff_seconds": -1},
        {"request_timeout_seconds": 0},
    ],
)
def test_remote_read_options_validation(kwargs):
    with pytest.raises(ValueError):
        lindi.LindiRemoteReadOptions(**kwargs)


def test_contiguous_selection_uses_one_range_get():
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    metrics = []
    options = lindi.LindiRemoteReadOptions(metrics_callback=metrics.append)
    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(server, data, options=options)
        server.reset_observations()
        actual = dataset[0:4, :]
        assert np.array_equal(actual, data[0:4, :])
        assert len(server.requests) == 1
        assert metrics[-1].planned_request_count == 1
        assert metrics[-1].range_get_count == 1
        assert metrics[-1].overread_bytes == 0
        assert metrics[-1].requested_bytes == metrics[-1].fetched_bytes
        assert asdict(metrics[-1])["returned_key_count"] == 4
        client.close()


def test_discontiguous_channel_runs_are_fetched_concurrently():
    data = np.arange(8 * 8, dtype=np.int16).reshape(8, 8)
    metrics = []
    options = lindi.LindiRemoteReadOptions(
        max_concurrency=4, metrics_callback=metrics.append
    )
    with _RangeServer(delay_seconds=0.1) as server:
        client, dataset = _open_remote_dataset(server, data, options=options)
        server.reset_observations()
        selection = (slice(0, 4), [0, 1, 6, 7])
        actual = dataset[selection]
        assert np.array_equal(actual, data[selection])
        assert len(server.requests) == 2
        assert server.max_active_requests >= 2
        assert metrics[-1].planned_request_count == 2
        client.close()


def test_concurrency_is_bounded_when_runs_exceed_worker_count():
    data = np.arange(4 * 8, dtype=np.int16).reshape(4, 8)
    metrics = []
    options = lindi.LindiRemoteReadOptions(
        max_concurrency=2, metrics_callback=metrics.append
    )
    with _RangeServer(delay_seconds=0.05) as server:
        client, dataset = _open_remote_dataset(
            server, data, options=options, chunks=(4, 1)
        )
        server.reset_observations()
        selection = (slice(None), [0, 2, 4, 6])
        assert np.array_equal(dataset[selection], data[selection])
        assert len(server.requests) == 4
        assert server.max_active_requests == 2
        assert metrics[-1].planned_request_count == 4
        client.close()


def test_nearby_ranges_respect_gap_and_cumulative_overread_limits():
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    metrics = []
    options = lindi.LindiRemoteReadOptions(
        max_gap_bytes=7,
        max_overread_bytes=13,
        max_request_bytes=None,
        metrics_callback=metrics.append,
    )
    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(
            server, data, options=options, gap_bytes=7
        )
        server.reset_observations()
        actual = dataset[0:4, :]
        assert np.array_equal(actual, data[0:4, :])
        assert len(server.requests) == 2
        assert metrics[-1].overread_bytes == 14
        assert metrics[-1].fetched_bytes == metrics[-1].requested_bytes + 14
        client.close()


def test_disabled_reader_provides_instrumented_sequential_baseline():
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    metrics = []
    options = lindi.LindiRemoteReadOptions(
        enabled=False, max_concurrency=8, metrics_callback=metrics.append
    )
    with _RangeServer(delay_seconds=0.02) as server:
        client, dataset = _open_remote_dataset(server, data, options=options)
        server.reset_observations()
        assert np.array_equal(dataset[0:4, :], data[0:4, :])
        assert len(server.requests) == 4
        assert server.max_active_requests == 1
        assert metrics[-1].optimized is False
        assert metrics[-1].planned_request_count == 4
        client.close()


def test_local_cache_stores_exact_chunks_after_coalesced_fetch():
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    metrics = []
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = lindi.LocalCache(cache_dir=tmpdir)
        options = lindi.LindiRemoteReadOptions(metrics_callback=metrics.append)
        with _RangeServer() as server:
            rfs, object_data = _make_reference_file_system(data, server.url)
            server.data = object_data
            client = lindi.LindiH5pyFile.from_reference_file_system(
                rfs, local_cache=cache, remote_read_options=options
            )
            dataset = client["data"]
            assert np.array_equal(dataset[0:4, :], data[0:4, :])
            assert len(server.requests) == 1
            server.reset_observations()
            assert np.array_equal(dataset[0:4, :], data[0:4, :])
            assert server.requests == []
            assert metrics[-1].cache_hit_count == 4
            assert metrics[-1].planned_request_count == 0
            client.close()


def test_partial_cache_hits_remove_only_cached_ranges_from_plan():
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    metrics = []
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = lindi.LocalCache(cache_dir=tmpdir)
        options = lindi.LindiRemoteReadOptions(metrics_callback=metrics.append)
        with _RangeServer() as server:
            rfs, object_data = _make_reference_file_system(data, server.url)
            server.data = object_data
            for key in ("data/0.0", "data/0.2"):
                url, offset, size = rfs["refs"][key]
                cache.put_remote_chunk(
                    url=url,
                    offset=offset,
                    size=size,
                    data=object_data[offset:offset + size],
                )
            client = lindi.LindiH5pyFile.from_reference_file_system(
                rfs, local_cache=cache, remote_read_options=options
            )
            dataset = client["data"]
            assert np.array_equal(dataset[0:4, :], data[0:4, :])
            assert len(server.requests) == 2
            assert metrics[-1].cache_hit_count == 2
            assert metrics[-1].source_range_count == 2
            assert metrics[-1].planned_request_count == 2
            client.close()


@pytest.mark.parametrize(
    "selection",
    [
        (slice(None), slice(None)),
        (2, slice(None)),
        (Ellipsis,),
        (slice(None, None, 2), slice(1, 7, 2)),
        (-1, slice(None)),
        (slice(0, 4), [0, 1, 6, 7]),
    ],
)
def test_selection_results_match_numpy(selection):
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(
            server, data, options=lindi.LindiRemoteReadOptions()
        )
        assert np.array_equal(dataset[selection], data[selection])
        client.close()


def test_compressed_and_big_endian_chunks_preserve_exact_data():
    data = np.arange(8 * 8, dtype=">i2").reshape(8, 8)
    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(
            server,
            data,
            options=lindi.LindiRemoteReadOptions(),
            compressor="zlib",
        )
        actual = dataset[1:7, 1:7]
        assert actual.dtype == data.dtype
        assert np.array_equal(actual, data[1:7, 1:7])
        assert np.array_equal(np.asarray(dataset), data)
        client.close()


def test_edge_chunk_padding_and_missing_chunk_fill_value():
    data = np.arange(6 * 4, dtype=np.int32).reshape(6, 4)
    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(
            server,
            data,
            options=lindi.LindiRemoteReadOptions(),
            chunks=(4, 2),
            truncate_edge_chunks=True,
        )
        assert np.array_equal(dataset[:], data)
        client.close()

    fill_value = -7
    expected = data.copy()
    expected[0:4, 2:4] = fill_value
    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(
            server,
            data,
            options=lindi.LindiRemoteReadOptions(),
            chunks=(4, 2),
            fill_value=fill_value,
            omitted_chunks={(0, 1)},
        )
        assert np.array_equal(dataset[:], expected)
        client.close()


def test_retry_metrics_and_short_response_failure():
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    metrics = []
    options = lindi.LindiRemoteReadOptions(
        max_retries=1,
        retry_backoff_seconds=0,
        metrics_callback=metrics.append,
    )
    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(server, data, options=options)
        server.fail_requests = 1
        assert np.array_equal(dataset[0:4, 0:2], data[0:4, 0:2])
        assert metrics[-1].range_get_count == 2
        client.close()

    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(server, data, options=options)
        server.truncate_requests = 2
        server.omit_content_range = True
        with pytest.raises(RemoteRangeReadError):
            dataset[0:4, 0:2]
        assert len(server.requests) == 2
        client.close()


def test_concurrent_worker_failure_and_ignored_range_are_rejected():
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    options = lindi.LindiRemoteReadOptions(
        max_concurrency=2,
        max_retries=0,
    )
    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(server, data, options=options)
        server.fail_requests = 10
        with pytest.raises(RemoteRangeReadError):
            dataset[0:4, [0, 1, 6, 7]]
        assert len(server.requests) >= 2
        client.close()

    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(server, data, options=options)
        server.response_status = 200
        with pytest.raises(RemoteRangeReadError, match="ignored byte range"):
            dataset[0:4, 0:2]
        client.close()

    with _RangeServer() as server:
        client, dataset = _open_remote_dataset(server, data, options=options)
        server.content_range_offset_delta = 1
        with pytest.raises(RemoteRangeReadError, match="Unexpected Content-Range"):
            dataset[0:4, 0:2]
        client.close()


def test_whole_object_200_response_is_accepted():
    with _RangeServer() as server:
        server.data = b"whole object"
        byte_range = ByteRange("key", server.url, 0, len(server.data))
        request = PlannedByteRangeRequest(
            source=server.url,
            offset=0,
            length=len(server.data),
            ranges=(byte_range,),
            requested_bytes=len(server.data),
            overread_bytes=0,
        )
        server.response_status = 200
        fetched = fetch_byte_range_requests(
            [request],
            lindi.LindiRemoteReadOptions(max_retries=0),
        )
        assert fetched[0].data == server.data


def test_force_explicit_chunk_references_for_selected_dataset():
    with tempfile.TemporaryDirectory() as tmpdir:
        filename = f"{tmpdir}/test.h5"
        data = np.arange(50 * 12, dtype=np.int16).reshape(50, 12)
        with h5py.File(filename, "w") as file:
            file.create_dataset("dataset1", data=data, chunks=(10, 6))
            file.create_dataset("dataset2", data=data, chunks=(10, 6))
        opts = lindi.LindiH5ZarrStoreOpts(
            num_dataset_chunks_threshold=4,
            force_explicit_chunk_references=("/dataset1",),
        )
        with lindi.LindiH5ZarrStore.from_file(
            filename, url=filename, opts=opts
        ) as store:
            rfs = store.to_reference_file_system()
        dataset1_attrs = rfs["refs"].get("dataset1/.zattrs", {})
        assert "_EXTERNAL_ARRAY_LINK" not in dataset1_attrs
        assert "dataset1/0.0" in rfs["refs"]
        assert "_EXTERNAL_ARRAY_LINK" in rfs["refs"]["dataset2/.zattrs"]
        client = lindi.LindiH5pyFile.from_reference_file_system(rfs)
        assert np.array_equal(client["dataset1"][:], data)
        assert np.array_equal(client["dataset2"][:], data)
        client.close()


def test_metrics_are_json_serializable():
    metrics = lindi.LindiRemoteReadMetrics(
        optimized=True,
        requested_key_count=1,
        returned_key_count=1,
        inline_value_count=0,
        local_file_read_count=0,
        remote_reference_count=1,
        cache_hit_count=0,
        source_range_count=1,
        planned_request_count=1,
        range_get_count=1,
        requested_bytes=10,
        fetched_bytes=10,
        overread_bytes=0,
        elapsed_seconds=0.1,
    )
    json.dumps(asdict(metrics))


def test_benchmark_cli_reports_optimized_and_baseline_requests():
    data = np.arange(8 * 8, dtype=np.int32).reshape(8, 8)
    repo_dir = Path(__file__).parent.parent
    script = repo_dir / "benchmarks" / "benchmark_remote_reader.py"
    with tempfile.TemporaryDirectory() as tmpdir, _RangeServer() as server:
        rfs, object_data = _make_reference_file_system(data, server.url)
        server.data = object_data
        lindi_path = Path(tmpdir) / "test.lindi.json"
        with open(lindi_path, "w") as file:
            json.dump(rfs, file)
        base_command = [
            sys.executable,
            str(script),
            "--lindi-url",
            str(lindi_path),
            "--dataset",
            "data",
            "--start",
            "0",
            "--stop",
            "4",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(repo_dir)
        optimized = subprocess.run(
            base_command,
            cwd=repo_dir,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        optimized_result = json.loads(optimized.stdout)
        assert optimized_result["summary"]["total_range_gets"] == 1
        assert optimized_result["summary"]["total_overread_bytes"] == 0

        baseline = subprocess.run(
            base_command + ["--baseline"],
            cwd=repo_dir,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        baseline_result = json.loads(baseline.stdout)
        assert baseline_result["summary"]["total_range_gets"] == 4
        assert baseline_result["options"]["enabled"] is False
