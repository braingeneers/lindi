from datetime import datetime, timezone

import numpy as np
import pytest
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries, LFP

from lindi import LindiH5pyFile, LindiRangeReadOptions


def _create_nwb_file(path, *, chunks=(8, 1)):
    raw_data = np.arange(128 * 12, dtype=np.int16).reshape(128, 12)
    lfp_data = (raw_data // 4).astype(np.int16)
    nwbfile = NWBFile(
        session_description="range-reader semantic test",
        identifier="range-reader-test",
        session_start_time=datetime(2024, 1, 2, tzinfo=timezone.utc),
    )
    device = nwbfile.create_device(name="probe")
    group = nwbfile.create_electrode_group(
        name="shank0",
        description="synthetic shank",
        location="CA1",
        device=device,
    )
    for channel in range(raw_data.shape[1]):
        nwbfile.add_electrode(
            id=channel,
            x=float(channel),
            y=float(channel % 3),
            z=0.0,
            imp=-1.0,
            location="CA1",
            filtering="none",
            group=group,
        )
    electrodes = nwbfile.create_electrode_table_region(
        region=list(range(raw_data.shape[1])),
        description="all channels",
    )
    nwbfile.add_acquisition(
        ElectricalSeries(
            name="raw",
            data=H5DataIO(raw_data, chunks=chunks),
            electrodes=electrodes,
            starting_time=0.25,
            rate=30_000.0,
            description="raw extracellular data",
        )
    )
    lfp_series = ElectricalSeries(
        name="lfp",
        data=H5DataIO(lfp_data, chunks=chunks),
        electrodes=electrodes,
        starting_time=0.25,
        rate=1_000.0,
    )
    ecephys = nwbfile.create_processing_module(
        name="ecephys",
        description="processed extracellular data",
    )
    ecephys.add(LFP(electrical_series=lfp_series))
    nwbfile.add_unit(id=3, spike_times=[0.1, 0.2, 0.35])
    nwbfile.add_unit(id=7, spike_times=[0.05, 0.4])

    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwbfile)


def _snapshot(nwbfile):
    raw = nwbfile.acquisition["raw"]
    lfp = nwbfile.processing["ecephys"]["LFP"].electrical_series["lfp"]
    sparse_channels = [0, 3, 7, 11]
    return {
        "identifier": nwbfile.identifier,
        "session_description": nwbfile.session_description,
        "electrode_ids": np.asarray(nwbfile.electrodes.id[:]),
        "electrode_locations": tuple(nwbfile.electrodes["location"][:]),
        "electrode_region": np.asarray(raw.electrodes.data[:]),
        "raw": np.asarray(raw.data[:]),
        "raw_sparse": np.asarray(raw.data[5:23, sparse_channels]),
        "raw_rate": raw.rate,
        "raw_starting_time": raw.starting_time,
        "lfp": np.asarray(lfp.data[3:19, :]),
        "lfp_rate": lfp.rate,
        "unit_ids": np.asarray(nwbfile.units.id[:]),
        "unit_spike_times": tuple(
            np.asarray(nwbfile.units["spike_times"][index])
            for index in range(len(nwbfile.units))
        ),
    }


def _read_hdf5_snapshot(path):
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        return _snapshot(io.read())


def _read_lindi_snapshot(path):
    metrics = []
    read_options = LindiRangeReadOptions(metrics_callback=metrics.append)
    with LindiH5pyFile.from_lindi_file(
        str(path), range_read_options=read_options
    ) as lindi_file:
        with NWBHDF5IO(file=lindi_file, mode="r", load_namespaces=True) as io:
            snapshot = _snapshot(io.read())
    if str(path).endswith(".lindi.json"):
        assert any(metric.range_reference_count > 0 for metric in metrics)
    return snapshot


def _assert_snapshots_equal(expected, observed):
    assert observed.keys() == expected.keys()
    for key in expected:
        expected_value = expected[key]
        observed_value = observed[key]
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(observed_value, expected_value)
        elif key == "unit_spike_times":
            assert len(observed_value) == len(expected_value)
            for observed_spikes, expected_spikes in zip(observed_value, expected_value):
                np.testing.assert_array_equal(observed_spikes, expected_spikes)
        else:
            assert observed_value == expected_value


@pytest.mark.parametrize("suffix", [".lindi.json", ".lindi.tar", ".lindi.d"])
@pytest.mark.parametrize("chunks", [(8, 1), (8, 4), (8, 12)])
def test_pynwb_semantics_are_preserved_across_lindi_containers(
    tmp_path, suffix, chunks
):
    hdf5_path = tmp_path / "source.nwb"
    lindi_path = tmp_path / f"converted{suffix}"
    _create_nwb_file(hdf5_path, chunks=chunks)
    expected = _read_hdf5_snapshot(hdf5_path)

    with LindiH5pyFile.from_hdf5_file(
        str(hdf5_path), url=hdf5_path.resolve().as_uri()
    ) as source:
        source.write_lindi_file(str(lindi_path))

    _assert_snapshots_equal(expected, _read_lindi_snapshot(lindi_path))
