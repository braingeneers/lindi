import h5py
import numpy as np
import pytest

from lindi import LindiH5pyFile, LindiH5ZarrStoreOpts, LindiRangeReadOptions


@pytest.mark.parametrize(
    "chunks, expected_ranges, expected_requests, expected_fetched_bytes",
    [
        ((8, 1), 96, 96, 3_072),
        ((8, 32), 96, 2, 98_304),
        ((8, 1024), 3, 1, 98_304),
    ],
)
def test_sparse_1024_channel_selection_across_layouts(
    tmp_path,
    chunks,
    expected_ranges,
    expected_requests,
    expected_fetched_bytes,
):
    values = np.arange(24 * 1024, dtype=np.int32).reshape(24, 1024)
    channels = list(range(0, 1024, 32))
    hdf5_path = tmp_path / "wide.h5"
    lindi_path = tmp_path / "wide.lindi.json"
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

    metrics = []
    read_options = LindiRangeReadOptions(
        max_gap_bytes=0,
        max_overread_bytes=0,
        metrics_callback=metrics.append,
    )
    with LindiH5pyFile.from_lindi_file(
        str(lindi_path), range_read_options=read_options
    ) as lindi_file:
        observed = np.asarray(lindi_file["data"][5:21, channels])
    np.testing.assert_array_equal(observed, values[5:21][:, channels])
    assert metrics[-1].range_reference_count == expected_ranges
    assert metrics[-1].planned_request_count == expected_requests
    assert metrics[-1].fetched_bytes == expected_fetched_bytes
    assert metrics[-1].logical_bytes == 2_048


@pytest.mark.parametrize(
    "values",
    [
        np.asarray([[True, False], [False, True]], dtype=np.bool_),
        np.arange(12, dtype=">i2").reshape(4, 3),
    ],
    ids=["boolean", "big-endian"],
)
@pytest.mark.parametrize("suffix", [".lindi.json", ".lindi.tar", ".lindi.d"])
def test_supported_dtype_round_trip(tmp_path, values, suffix):
    hdf5_path = tmp_path / "types.h5"
    lindi_path = tmp_path / f"types{suffix}"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset("data", data=values, chunks=True)
    with LindiH5pyFile.from_hdf5_file(
        str(hdf5_path), url=hdf5_path.resolve().as_uri()
    ) as source:
        source.write_lindi_file(str(lindi_path))

    with LindiH5pyFile.from_lindi_file(str(lindi_path)) as lindi_file:
        observed = np.asarray(lindi_file["data"][:])
    np.testing.assert_array_equal(observed, values)


@pytest.mark.parametrize("suffix", [".lindi.json", ".lindi.tar", ".lindi.d"])
def test_compound_records_and_fields_round_trip(tmp_path, suffix):
    values = np.asarray(
        [(1, 1.5), (2, 2.5), (3, 3.5)],
        dtype=[("channel", "<i4"), ("value", "<f8")],
    )
    hdf5_path = tmp_path / "compound.h5"
    lindi_path = tmp_path / f"compound{suffix}"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset("data", data=values, chunks=True)
    with LindiH5pyFile.from_hdf5_file(
        str(hdf5_path), url=hdf5_path.resolve().as_uri()
    ) as source:
        source.write_lindi_file(str(lindi_path))

    with LindiH5pyFile.from_lindi_file(str(lindi_path)) as lindi_file:
        dataset = lindi_file["data"]
        assert dataset["channel"].dtype == np.dtype("<i4")
        np.testing.assert_array_equal(dataset[:], values)
        np.testing.assert_array_equal(dataset["channel"][:], values["channel"])
        np.testing.assert_array_equal(dataset["value"][:], values["value"])


def test_complex_conversion_fails_with_clear_error(tmp_path):
    values = (np.arange(12).reshape(4, 3) * (1 + 2j)).astype(np.complex64)
    hdf5_path = tmp_path / "complex.h5"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset("data", data=values, chunks=True)
    with LindiH5pyFile.from_hdf5_file(
        str(hdf5_path), url=hdf5_path.resolve().as_uri()
    ) as source:
        with pytest.raises(Exception, match="Complex datasets are not supported"):
            source.write_lindi_file(str(tmp_path / "complex.lindi.json"))


def test_external_array_link_accepts_local_file_uri_and_closes_client(tmp_path):
    values = np.arange(40 * 40, dtype=np.int32).reshape(40, 40)
    hdf5_path = tmp_path / "external.h5"
    lindi_path = tmp_path / "external.lindi.json"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset("data", data=values, chunks=(1, 1))
    with LindiH5pyFile.from_hdf5_file(
        str(hdf5_path), url=hdf5_path.resolve().as_uri()
    ) as source:
        source.write_lindi_file(str(lindi_path))

    lindi_file = LindiH5pyFile.from_lindi_file(str(lindi_path))
    observed = np.asarray(lindi_file["data"][3:7, 5:9])
    assert len(lindi_file._external_hdf5_clients) == 1
    lindi_file.close()
    assert lindi_file._external_hdf5_clients == {}
    lindi_file.close()
    np.testing.assert_array_equal(observed, values[3:7, 5:9])


def test_scalar_and_variable_length_strings_round_trip(tmp_path):
    hdf5_path = tmp_path / "scalar-strings.h5"
    lindi_path = tmp_path / "scalar-strings.lindi.json"
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset("scalar", data=np.float64(3.25))
        hdf5_file.create_dataset(
            "strings",
            data=np.asarray(["alpha", "beta", "gamma"], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
    with LindiH5pyFile.from_hdf5_file(
        str(hdf5_path), url=hdf5_path.resolve().as_uri()
    ) as source:
        source.write_lindi_file(str(lindi_path))

    with LindiH5pyFile.from_lindi_file(str(lindi_path)) as lindi_file:
        assert lindi_file["scalar"][()] == pytest.approx(3.25)
        observed_strings = tuple(lindi_file["strings"][:])
    assert observed_strings in (
        ("alpha", "beta", "gamma"),
        (b"alpha", b"beta", b"gamma"),
    )
