"""Small mechanism checks for calibration retention, ties and fallback strata."""
import unittest

import numpy as np
import pandas as pd

from evrptw_cle.unit_quality import apply_quality, fit_quality_thresholds


def frame(cost, spread, city="A", kind="generic"):
    n = len(cost)
    return pd.DataFrame({"city": city, "use_type": kind, "neighbor_branch": "height_m",
                         "neighbor_cost": cost, "neighbor_log_mad": spread,
                         "neighbor_units": np.ones(n), "neighbor_count": np.full(n, 8),
                         "source_units": np.ones(n)})


class QualityTests(unittest.TestCase):
    def test_disjoint_metric_tails_use_joint_retention(self):
        # Separate marginal P95 limits retain only 90% for opposed tails.
        cal = frame(np.arange(100.), np.arange(99., -1., -1.))
        fitted = fit_quality_thresholds(cal)
        out = apply_quality(cal, fitted)
        self.assertGreaterEqual(out.quality_pass.mean(), .95)
        self.assertLess(out.quality_pass.mean(), 1.)
        # Unit outcomes cannot feed back into the selected support rectangle.
        changed = cal.copy()
        changed["source_units"] = np.arange(1., 101.)
        second = fit_quality_thresholds(changed)
        for a, b in zip(fitted["strata"], second["strata"]):
            self.assertEqual(a["cost_limit"], b["cost_limit"])
            self.assertEqual(a["dispersion_limit"], b["dispersion_limit"])

    def test_zero_mad_atom_passes_inclusive_limit(self):
        cal = frame(np.arange(100.)/100, np.zeros(100))
        fitted = fit_quality_thresholds(cal)
        out = apply_quality(cal, fitted)
        self.assertEqual(out.quality_dispersion_limit.max(), 0.)
        self.assertTrue(out.dispersion_pass.all())
        self.assertGreaterEqual(out.quality_pass.mean(), .95)
        self.assertFalse(bool(out.quality_pass.iloc[-1]))

    def test_sparse_and_unseen_strata_fall_back_without_losing_rows(self):
        cal = pd.concat([frame(np.linspace(.01, .2, 100), np.zeros(100), kind="house"),
                         frame([.1]*8, [0.]*8, city="B", kind="apartments")], ignore_index=True)
        fitted = fit_quality_thresholds(cal)
        target = pd.concat([frame([.1], [0.], city="C", kind="house"),
                            frame([.1], [0.], city="B", kind="apartments"),
                            frame([np.nan], [0.], city="Z", kind="generic")], ignore_index=True)
        target.index = [30, 10, 90]
        out = apply_quality(target, fitted)
        self.assertEqual(out.index.tolist(), [30, 10, 90])
        self.assertEqual(out.quality_stratum.iloc[0], "use_type/neighbor_branch:house|height_m")
        self.assertEqual(out.quality_stratum.iloc[1], "global")
        self.assertTrue(bool(out.quality_pass.iloc[0]))
        self.assertTrue(bool(out.quality_pass.iloc[1]))
        self.assertFalse(bool(out.quality_pass.iloc[2]))

    def test_log_spread_90_is_preferred_over_zero_mad(self):
        cal = frame(np.zeros(100), np.zeros(100))
        cal["neighbor_log_spread_90"] = np.arange(100.)/100
        fitted = fit_quality_thresholds(cal)
        out = apply_quality(cal, fitted)
        self.assertEqual(fitted["dispersion_column"], "neighbor_log_spread_90")
        self.assertTrue(out.quality_dispersion_metric.eq("neighbor_log_spread_90").all())
        self.assertFalse(bool(out.dispersion_pass.iloc[-1]))
        explicit_mad = fit_quality_thresholds(cal, {"quality_dispersion_column": "neighbor_log_mad"})
        self.assertTrue(apply_quality(cal, explicit_mad).dispersion_pass.all())


if __name__ == "__main__":
    unittest.main()
