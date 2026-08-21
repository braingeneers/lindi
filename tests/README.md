# Test tiers

The default `pytest` run is deterministic and does not contact external services.
`make test` builds the test image and runs this tier in Docker. Set
`PYTHON_VERSION=3.9` or `PYTHON_VERSION=3.12` to select the interpreter.

Run the public DANDI integration tests explicitly:

```bash
make test-network-docker
```

Run a smoke test against a stable LINDI manifest backed by Ceph/S3:

```bash
LINDI_LIVE_REFERENCE_URL=https://example/path/data.lindi.json \
LINDI_LIVE_DATASET_PATH=acquisition/ElectricalSeries/data \
make test-live-object-store-docker
```

The live fixture should be immutable. The smoke test reads a small selection through
the baseline and optimized paths and requires exact equality. Large-scale request,
throughput, and latency measurements remain in `benchmarks/`.

## Portability notes

- The image supplies writable `HOME` and `XDG_CACHE_HOME` locations so PyNWB works
  when the container runs with the host user's UID.
- The benchmark CLI is included because the deterministic regression suite invokes
  it directly.
- Remote TAR behavior is tested against an in-process HTTP range server. External
  DANDI assets remain isolated behind the `network` marker.
