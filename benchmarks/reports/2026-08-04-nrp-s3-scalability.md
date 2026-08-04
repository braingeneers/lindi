# LINDI remote reader live S3 scalability report

Date: 2026-08-04  
Cluster: NRP Nautilus, namespace `braingeneers`  
Campaign: `lindi-s3-20260804-a`  
Reader commit: `cae07e2`  
Campaign tooling commit: `337f068`  
Image: `davidparks21/lindi:20260804-s3-reader-337f068`  
Image digest: `sha256:618065125636f994c55835685f6e8c8deb03597e510370d69269889bd4101036`

## Executive result

The LINDI range planner and parallel range executor are byte-correct against live Ceph/S3 data and materially reduce latency for both target access patterns.

- Random single-channel 40,000-sample windows used three adjacent compressed chunks. Coalescing reduced three GETs to one with zero over-read. Across 20 paired random reads, internal median latency improved from `0.207 s` to `0.115 s`; public-route median latency improved from `1.142 s` to `0.754 s`.
- Random all-channel 1,024-sample windows used 1,008 physically discontiguous chunks. Coalescing correctly did not over-read or merge them. On one fixed NRP node, bounded concurrency reduced fetch time from `67.39 s` at concurrency 1 to `1.95 s` at concurrency 32.
- Four concurrent ML-style workers at reader concurrency 8 completed 12 random all-channel windows, totaling `12,096` range GETs, with zero retries. Mean per-window latency was `7.88 s`, compared with `7.77 s` for the post-load solo control on the same node.
- Every paired optimized fetch matched the baseline compressed chunk bytes exactly. The campaign compared 1,146 live chunk references across 47 paired queries.

The transport layer therefore meets its range-planning, bounded-concurrency, and byte-preservation goals. It is not yet an end-to-end production ML reader for these workflow outputs because the signal array declares the custom `maxwell-mxw-data` filter and the deployed LINDI/runtime image does not register or provide its native decoder. The live correctness result is exact compressed chunk bytes, not decoded NumPy sample arrays.

## Dataset selection and provenance

Recent workflow provenance came from successful Workflows run `57687039-0c67-4ada-8a88-d15c5826436f`:

- Workflow: `braingeneers-ephys-dataset-pipeline` version `0.3.13`
- UUID: `2026-07-29-e-24472J_H9SynGFP_MO_SCH`
- Workflow chunk parameters: `nwb_chunk_duration_ms=1000`, `nwb_chunk_channels=1`
- Four generated NWB objects ranged from 3.40 GB to 14.74 GB
- Selected pilot object: `24472J_H9SynGFP_MO_D38_sch_24hr_connectedconfig_07302026.nwb`

Selected object details:

| Property | Value |
|---|---:|
| NWB object bytes | 3,398,188,127 |
| LINDI sidecar bytes | 63,105,499 |
| LINDI references | 613,034 |
| Signal shape | `[12,159,800, 1,008]` |
| Signal chunks | `[20,000, 1]` |
| Signal dtype | `<u2` |
| Signal filter | `maxwell-mxw-data`, HDF5 filter 401, codec version 1 |
| NWB ETag | `61a498dccec8a6277f96485749ebafcb-406` |
| Sidecar ETag | `456665d638e24a07092452e1dc90fbae-8` |
| NWB schema validation | Passed, no PyNWB schema errors |

## Method

All workloads were Kubernetes `Job` resources. No standalone Pods, Deployments, or other workload controllers were created.

- Job retry policy: `backoffLimit: 0`, `restartPolicy: Never`
- Job resources: 1 CPU, 2 GiB memory, and 1 GiB ephemeral storage, with limits equal to requests
- Data access: unauthenticated public read or internal public Ceph read
- S3 credentials: not mounted or injected into benchmark Jobs
- S3 operations: HTTP `GET`, `HEAD`, and range `GET` only
- Baseline: one sequential GET per explicit chunk reference
- Optimized: LINDI `plan_byte_range_requests` plus `fetch_byte_range_requests`
- Coalescing policy: `max_gap_bytes=4096`, `max_overread_bytes=16384`
- Correctness: reconstruct each original chunk from fetched requests and compare ordered SHA-256 digests
- Randomness: seeded sample starts and channels, with identical seeds for paired baseline/optimized comparisons

The internal route was `http://rook-ceph-rgw-nautiluss3.rook`. The public route was `https://s3.braingeneers.gi.ucsc.edu`.

## Single-channel random windows

Each 40,000-sample random selection intersected three adjacent `[20,000, 1]` chunks. The planner merged them into one exact byte range with no over-read.

Twenty paired windows per route used seed `20260806`.

| Route | Metric | Sequential baseline | Coalesced | Speedup |
|---|---|---:|---:|---:|
| Internal | Mean | 0.221 s | 0.131 s | 1.70x |
| Internal | Median | 0.207 s | 0.115 s | 1.79x |
| Internal | p95 | 0.335 s | 0.226 s | 1.48x |
| Public | Mean | 1.999 s | 1.553 s | 1.29x |
| Public | Median | 1.142 s | 0.754 s | 1.51x |
| Public | p95 | 7.132 s | 5.293 s | 1.35x |

Request results:

| Route | Baseline GET attempts | Coalesced GET attempts | Retries | Byte exact |
|---|---:|---:|---:|---:|
| Internal | 60 | 20 | 0 | 20/20 |
| Public | 61 | 20 | 1 | 20/20 |

The public route had substantially higher variance and tail latency. Coalescing still reduced request count by two thirds and improved all reported latency statistics.

## All-channel random windows

Each 1,024-sample all-channel window selected one time chunk for every channel. The channel-major allocation produced 1,008 discontiguous references totaling 5,488,757 compressed bytes. No references were close enough to merge under the configured policy.

The paired internal correctness Job produced identical baseline and optimized digest `d879527a870ece1cc8743f7fe8907d4366d9d86c42f4e93007bb9b1684c4f587`.

### Same-node internal concurrency sweep

This primary sweep ran every concurrency setting inside one Job on node `k8s-a100-01.suncorridor.org`. The non-monotonic execution order was 8, 1, 32, 4, 16, 2.

| Concurrency | GETs | Fetch time | Throughput | Speedup vs 1 |
|---:|---:|---:|---:|---:|
| 1 | 1,008 | 67.388 s | 0.078 MiB/s | 1.00x |
| 2 | 1,008 | 21.993 s | 0.238 MiB/s | 3.06x |
| 4 | 1,008 | 12.800 s | 0.409 MiB/s | 5.26x |
| 8 | 1,008 | 5.624 s | 0.931 MiB/s | 11.98x |
| 16 | 1,008 | 3.121 s | 1.677 MiB/s | 21.59x |
| 32 | 1,008 | 1.947 s | 2.689 MiB/s | 34.62x |

Planning 1,008 ranges took approximately 3 ms. No request retried, and fetched bytes equaled requested bytes at every setting.

The greater-than-linear speedups are observed end-to-end results, not a claim about algorithmic scaling. The serial path pays connection, HTTP, endpoint, and tail-latency costs without overlap. These measurements should not be extrapolated beyond the tested endpoint and concurrency range.

### Public route

These three Jobs ran on the same node, `nrp-00.rcac.purdue.edu`, in increasing concurrency order.

| Concurrency | GETs | Fetch time | Throughput | Speedup vs 1 |
|---:|---:|---:|---:|---:|
| 1 | 1,008 | 507.078 s | 0.010 MiB/s | 1.00x |
| 8 | 1,008 | 18.826 s | 0.278 MiB/s | 26.94x |
| 32 | 1,008 | 5.659 s | 0.925 MiB/s | 89.61x |

The public result includes route and temporal cache effects. It demonstrates severe serial request cost and a large practical benefit from bounded parallelism, but it is not a controlled storage-server scaling curve.

## Four-worker load test

Four Jobs ran concurrently on `rci-nrp-gpu-03.sdsu.edu`. Each Job performed three random all-channel reads with reader concurrency 8 and a distinct seed.

| Run | Queries | GETs | Mean | Median | p95 | Retries |
|---|---:|---:|---:|---:|---:|---:|
| Solo post-load control | 3 | 3,024 | 7.767 s | 7.566 s | 8.510 s | 0 |
| Four concurrent workers | 12 | 12,096 | 7.878 s | 7.870 s | 8.734 s | 0 |

Mean per-query latency increased by approximately 1.4%. No material contention was observed at four workers by eight reader threads, approximately 32 simultaneous in-flight GETs.

## Issues found

### 1. Maxwell filter blocks decoded end-to-end reads

Opening `/acquisition/ElectricalSeries/data` through `LindiH5pyFile` fails with `UnknownCodecError: maxwell-mxw-data`. The workflow image contains a Python codec declaration but does not contain the native `libcompression.so` decoder at any declared search path. Workflow documentation also notes a compressed-filter padding incompatibility in the generic LINDI 0.4.6 path.

Impact: the range layer is live-tested and byte-correct, but an ML process cannot yet receive decoded signal arrays from these workflow-generated sidecars.

Required resolution: package and register a supported filter-401 decoder, correct the compressed-chunk decode contract, and repeat the paired live tests through `LindiH5pyFile` with decoded NumPy array equality.

### 2. Public DNS and route instability

The first paired campaign Job failed before metadata loading with a temporary DNS resolution error for `s3.braingeneers.gi.ucsc.edu`. A later diagnostic Job succeeded and measured:

| Endpoint | DNS | One-byte range GET |
|---|---:|---:|
| Braingeneers public | 0.774 s | 1.785 s |
| NRP S3-West | 0.612 s | 0.618 s |
| Internal Ceph | 0.086 s | 0.445 s |

Campaign tooling now retries sidecar metadata GETs with bounded exponential backoff. Product-side sidecar loading should receive equivalent retry behavior. NRP workloads should prefer the internal Ceph route when the sidecar and object are available there.

### 3. All-channel layout remains request-heavy

The reader cannot coalesce the 1,008 all-channel chunks because the workflow deliberately allocates `[20,000, 1]` chunks in channel-major order. Parallelism reduces latency but not S3 request count.

For workloads dominated by all-channel windows, evaluate a larger `nwb_chunk_channels` value or a separate layout tuned for that query shape. Keep one-channel chunks when single-channel access is the dominant requirement.

### 4. Workflows service token was stale

The Kubernetes `braingeneers-jwt-service-account-token` expired on 2026-07-11, so the authenticated Workflows API redirected to OAuth. Exact run identity and parameters were recovered read-only from the retained Workflows Job and PVC provenance. Secret mutation is operator-owned and was not attempted.

## Recommendations

1. Treat Maxwell codec integration and decoded-array equality as the production release gate.
2. Keep concurrency configurable. Concurrency 8 is a conservative default; 16 and 32 were faster in this campaign, but should be enabled deliberately until broader storage-fairness testing is complete.
3. Use the internal Ceph endpoint for NRP Jobs and retain bounded retry behavior for metadata and signal ranges.
4. Cache the 63 MB LINDI sidecar per long-lived ML worker so metadata loading is amortized.
5. Match `nwb_chunk_channels` to the expected training access pattern. Reader concurrency cannot eliminate request amplification caused by one-channel physical layout.
6. Repeat this campaign on the 14.74 GB sibling recording after decoded filter support is available.

## Acceptance assessment

| Criterion | Result |
|---|---|
| Fewer GETs for contiguous reads | Pass: 3 to 1, zero over-read |
| Parallel discontiguous reads | Pass: 1,008 GETs accelerated through bounded concurrency |
| Exact transport bytes | Pass: all paired live reads matched |
| Random single-channel workload | Pass: 46 paired random queries across both routes |
| Random all-channel workload | Pass at transport layer |
| Multi-worker scale | Pass at four workers, 12,096 GETs, zero retries |
| Exact decoded arrays | Blocked by missing `maxwell-mxw-data` decoder |
| S3 read-only guarantee | Pass: no credentials in Jobs and GET-only code path |
| Jobs-only deployment | Pass |

Overall: the new remote range reader is validated for production-like transport planning and fetching on live Ceph/S3. Production use on the selected NWB signal dataset remains blocked until the workflow's custom Maxwell filter can be decoded through the normal LINDI array API.

