import os

import numpy as np
import pytest

from lindi import LindiH5pyFile
from lindi.range_reading import LindiRangeReadOptions


def _live_configuration():
    reference_url = os.environ.get("LINDI_LIVE_REFERENCE_URL")
    dataset_path = os.environ.get("LINDI_LIVE_DATASET_PATH")
    if not reference_url or not dataset_path:
        pytest.skip(
            "set LINDI_LIVE_REFERENCE_URL and LINDI_LIVE_DATASET_PATH to run "
            "the live object-store smoke test"
        )
    return reference_url, dataset_path


def _small_selection(shape):
    if not shape:
        return ()
    return tuple(slice(0, min(length, 8)) for length in shape)


@pytest.mark.live_object_store
def test_live_object_store_optimized_read_matches_baseline():
    reference_url, dataset_path = _live_configuration()
    disabled = LindiRangeReadOptions(enabled=False)
    with LindiH5pyFile.from_lindi_file(
        reference_url, range_read_options=disabled
    ) as baseline_file:
        dataset = baseline_file[dataset_path]
        selection = _small_selection(dataset.shape)
        expected = np.asarray(dataset[selection])

    with LindiH5pyFile.from_lindi_file(reference_url) as optimized_file:
        observed = np.asarray(optimized_file[dataset_path][selection])
    np.testing.assert_array_equal(observed, expected)
