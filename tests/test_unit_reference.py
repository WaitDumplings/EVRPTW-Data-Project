"""Spatial/source holdouts keep shared evidence out of both sides of a split."""
import numpy as np
import pandas as pd
import pytest

from evrptw_cle.unit_reference import make_reference_splits


def test_spatial_and_source_links_stay_together_under_row_permutation():
    reference = pd.DataFrame({
        "entity_id": ["a", "b", "c", "d", "e", "e", "f", "g"],
        "source_group": ["source-1", "source-1", "source-2", "source-3",
                         "source-4", "source-5", None, ""],
        "x": [0., 4000., 4100., 8000., 12000., 16000., 20000., 24000.],
        "y": [0.] * 8,
    }, index=[11, 20, 30, 40, 50, 60, 70, 80])
    splits = make_reference_splits(reference, "fixture")
    assert splits.index.equals(reference.index)
    # a/b share an upstream footprint, while b/c occupy the same metric block.
    assert splits.loc[[11, 20, 30]].nunique() == 1
    # Repeated entity identity joins distant blocks as well.
    assert splits.loc[[50, 60]].nunique() == 1
    assert splits.attrs["spatial_blocks"] == 7
    assert splits.attrs["connected_components"] == 5
    permuted = make_reference_splits(reference.sample(frac=1, random_state=9), "fixture")
    pd.testing.assert_series_equal(splits.sort_index(), permuted.sort_index())


def test_missing_source_keys_do_not_merge_blocks_and_configured_split_is_respected():
    reference = pd.DataFrame({
        "entity_id": [f"building-{i}" for i in range(12)],
        "source_group": [None, "", np.nan] * 4,
        "x": np.arange(12, dtype=float) * 2000.,
        "y": np.zeros(12),
    })
    splits = make_reference_splits(reference, "fixture", train_pct=0, cal_pct=100, test_pct=0)
    assert splits.attrs["connected_components"] == 12
    assert splits.eq("calibration").all()
    reference.loc[0, "x"] = np.nan
    with pytest.raises(ValueError, match="finite local-metric coordinates"):
        make_reference_splits(reference, "fixture")
