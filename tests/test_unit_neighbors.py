import unittest

import numpy as np
import pandas as pd

from evrptw_cle.unit_neighbors import EmpiricalMetricNeighborModel


class EmpiricalMetricTests(unittest.TestCase):
    def reference(self, n=40):
        rng = np.random.default_rng(74)
        return pd.DataFrame({
            "entity_id": [f"r{i}" for i in range(n)],
            "source_group": [f"s{i}" for i in range(n)],
            "use_type": ["apartments"] * n,
            "x": np.arange(n) * 50.0,
            "y": rng.uniform(-20, 20, n),
            "area_m2": np.exp(rng.uniform(4, 7, n)),
            "levels": rng.integers(1, 10, n).astype(float),
            "height_m": np.exp(rng.uniform(1, 3, n)),
            "units": rng.integers(1, 80, n).astype(float),
        })

    def test_empirical_rms_equalizes_three_group_mean_contributions(self):
        reference = self.reference()
        model = EmpiricalMetricNeighborModel({"scale_probe_count": 2000}).fit(reference)
        row = model.scales_table().query("branch == 'levels'").iloc[0]
        self.assertEqual(row.active_groups, 3)
        self.assertEqual(row.geographic_pair_count, len(reference) * 8)
        for dimension in ("spatial", "area", "vertical"):
            self.assertEqual(row[f"{dimension}_scale_scheme"], "geographic_neighbor_pairs")
            self.assertAlmostEqual(row[f"new_geo_pair_contribution_{dimension}"], 1/3)
        self.assertFalse(np.isclose(row.old_expected_share_spatial, 1/3))
        pd.testing.assert_frame_equal(model.predict(reference), model.predict(reference))

    def test_self_source_exclusion_and_constant_vertical_support(self):
        reference = self.reference(12)
        reference["levels"] = 2.0
        # The two closest geometries share an upstream source and must not
        # become each other's independent evidence.
        reference.loc[1, ["source_group", "x", "y", "area_m2"]] = [
            reference.loc[0, "source_group"], reference.loc[0, "x"],
            reference.loc[0, "y"], reference.loc[0, "area_m2"]]
        model = EmpiricalMetricNeighborModel({"k": 1}).fit(reference)
        row = model.scales_table().query("branch == 'levels'").iloc[0]
        self.assertTrue(row.vertical_constant)
        self.assertEqual(row.log_vertical_rms, 0)
        result = model.predict(reference.iloc[[0]])
        self.assertTrue(result.neighbor_valid.iloc[0])
        self.assertNotIn(result.neighbor_reference_id.iloc[0], ["r0", "r1"])
        same_support = reference.iloc[[0]].copy()
        same_support["entity_id"] = "new"
        same_support["source_group"] = "new-source"
        self.assertTrue(model.predict(same_support).neighbor_valid.iloc[0])
        same_support["levels"] = 3.0
        outside = model.predict(same_support)
        self.assertFalse(outside.neighbor_valid.iloc[0])
        self.assertTrue(outside.metric_constant_out_of_support.iloc[0])

    def test_local_zero_vertical_differences_are_not_a_global_constant(self):
        reference = self.reference(20)
        reference["x"] = np.r_[np.arange(10), 10000 + np.arange(10)]
        reference["y"] = 0.0
        reference["levels"] = np.r_[np.full(10, 2.0), np.full(10, 5.0)]
        model = EmpiricalMetricNeighborModel().fit(reference)
        row = model.scales_table().query("branch == 'levels'").iloc[0]
        self.assertFalse(row.vertical_constant)
        self.assertEqual(row.vertical_scale_scheme, "pooled_random_pair_rms")
        self.assertAlmostEqual(row.log_vertical_rms, np.sqrt(2*np.var(np.log(reference.levels))))
        self.assertEqual(row.new_geo_pair_contribution_vertical, 0)
        target = reference.iloc[[0]].copy()
        target["entity_id"] = "new"
        target["source_group"] = "new-source"
        target["levels"] = 3.0
        self.assertTrue(model.predict(target).neighbor_valid.iloc[0])

    def test_missing_height_uses_fixed_original_source_independent_reference(self):
        reference = self.reference(10)
        reference["area_m2"] = 100.0
        reference["levels"] = np.nan
        reference["height_m"] = [900.0, 900.0] + [6.0] * 8
        reference.loc[1, "source_group"] = reference.loc[0, "source_group"]
        reference.loc[1, ["x", "y"]] = reference.loc[0, ["x", "y"]]
        # Other upstream/estimated height fields are not predictor inputs.
        reference["nsi_height_m"] = 9999.0
        reference["neighbor_height_p90"] = 9999.0
        model = EmpiricalMetricNeighborModel().fit(reference)
        target = reference.iloc[[0]].copy()
        target["height_m"] = np.nan
        original_target = target.copy(deep=True)
        reference["height_m"] = 9999.0
        ordinary = model.predict(target)
        estimated = model.predict(target, estimate_missing_height=True)
        self.assertEqual(estimated.neighbor_height_p90.iloc[0], 6.0)
        self.assertTrue(estimated.neighbor_height_estimated.iloc[0])
        self.assertTrue(np.isnan(ordinary.neighbor_height_p90.iloc[0]))
        ignore = ["neighbor_height_p90", "neighbor_height_estimated"]
        pd.testing.assert_frame_equal(ordinary.drop(columns=ignore), estimated.drop(columns=ignore))
        pd.testing.assert_frame_equal(target, original_target)


if __name__ == "__main__":
    unittest.main()
