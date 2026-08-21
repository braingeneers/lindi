from uuid import uuid4

import h5py
import numpy as np
from hypothesis import HealthCheck, given, settings, strategies as st

from lindi import LindiH5pyFile, LindiH5ZarrStoreOpts, LindiRangeReadOptions


@st.composite
def _array_cases(draw):
    rows = draw(st.integers(min_value=48, max_value=80))
    columns = draw(st.integers(min_value=24, max_value=40))
    chunk_rows = draw(st.integers(min_value=4, max_value=rows))
    chunk_columns = draw(st.integers(min_value=1, max_value=columns))
    row_start = draw(st.integers(min_value=0, max_value=rows - 1))
    row_stop = draw(st.integers(min_value=row_start + 1, max_value=rows))
    channels = draw(
        st.lists(
            st.integers(min_value=0, max_value=columns - 1),
            min_size=1,
            max_size=columns,
            unique=True,
        )
    )
    return (
        (rows, columns),
        (chunk_rows, chunk_columns),
        slice(row_start, row_stop),
        sorted(channels),
    )


@settings(
    max_examples=16,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(case=_array_cases())
def test_random_chunk_layouts_and_selections_match_hdf5(tmp_path, case):
    shape, chunks, time_selection, channel_selection = case
    values = np.arange(np.prod(shape), dtype=np.int32).reshape(shape)
    case_id = uuid4().hex
    hdf5_path = tmp_path / f"source-{case_id}.h5"
    lindi_path = tmp_path / f"source-{case_id}.lindi.json"

    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset("data", data=values, chunks=chunks)
    conversion_options = LindiH5ZarrStoreOpts(
        force_explicit_chunk_references=("data",)
    )
    with LindiH5pyFile.from_hdf5_file(
        str(hdf5_path),
        url=hdf5_path.resolve().as_uri(),
        zarr_store_opts=conversion_options,
    ) as source:
        source.write_lindi_file(str(lindi_path))

    expected = values[time_selection][:, channel_selection]
    metrics = []
    read_options = LindiRangeReadOptions(metrics_callback=metrics.append)
    with LindiH5pyFile.from_lindi_file(
        str(lindi_path), range_read_options=read_options
    ) as lindi_file:
        observed = np.asarray(lindi_file["data"][time_selection, channel_selection])
    np.testing.assert_array_equal(observed, expected)
    assert metrics[-1].range_reference_count > 0
