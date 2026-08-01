# Remote reader campaign harness

`benchmark_remote_reader.py` is the handoff point for the future NRP Ceph/S3
campaign. It performs repeatable NWB/LINDI window reads and emits machine-readable
JSON containing latency, throughput, physical Range GET count, fetched bytes,
over-read bytes, cache hits, and the full reader configuration.
Signed URL query parameters and fragments are removed from the output record.

The target `.lindi.json` must contain explicit chunk references for the measured
dataset. `_EXTERNAL_ARRAY_LINK` datasets use the compatibility HDF5 path and do
not pass through the bulk reference reader.

## Example

```bash
python benchmarks/benchmark_remote_reader.py \
  --lindi-url "$LINDI_URL" \
  --dataset acquisition/ElectricalSeries/data \
  --start 10000000 \
  --stop 10100000 \
  --channels 0,1,2,3,128,129,130,131 \
  --max-gap-bytes 1048576 \
  --max-overread-bytes 8388608 \
  --max-concurrency 8 \
  --iterations 10 \
  --label nrp-cold-cache-c8 \
  --output result.json
```

Use `--baseline` to issue one sequential GET per referenced chunk through the
same instrumented transport. Omit `--cache-dir` for cold object-store runs. Set a
fresh pod-local directory for controlled warm-cache runs.

## Campaign matrix to run on NRP

- Baseline and optimized modes with identical windows.
- Contiguous all-channel windows and discontiguous channel groups.
- Cold cache and warm cache.
- Concurrency values `1, 2, 4, 8, 16`.
- Gap/over-read policies including `0/0`, `64 KiB/1 MiB`, and `1 MiB/8 MiB`.
- Several window lengths and at least two physical dataset layouts.
- Multiple repetitions in fresh pods, retaining raw JSON output for analysis.

Record the Ceph endpoint, node, pod resources, object identity/version, network
conditions, and LINDI commit alongside every result. Do not use wall-clock-only
acceptance: compare returned data, Range GET count, fetched bytes, and throughput.
