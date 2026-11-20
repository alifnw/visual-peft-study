import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from expansion_analysis import prediction_metrics, paired_interval


class ExpansionMetricTests(unittest.TestCase):
    def test_extreme_scores_and_binary_probabilities(self):
        result = prediction_metrics(np.array([[10000., -10000.], [0., 0.]]), np.array([0, 1]))
        self.assertAlmostEqual(result["nll"], np.log(2) / 2)
        self.assertAlmostEqual(result["brier"], 0.25)
        self.assertEqual(result["accuracy_percent"], 50)

    def test_absent_classes_are_unavailable(self):
        result = prediction_metrics(np.array([[2., 0., -1.], [3., 0., -1.]]), np.array([0, 0]))
        self.assertEqual(result["classes_observed"], 1)
        self.assertEqual(result["macro_recall_observed"], 1)
        self.assertTrue(np.isnan(result["recall"][1:]).all())

    def test_paired_bootstrap_is_seeded_and_has_correct_sign(self):
        a = paired_interval(np.array([0.1, 0.2]), np.array([0.2, 0.3]))
        b = paired_interval(np.array([0.1, 0.2]), np.array([0.2, 0.3]))
        self.assertEqual(a, b)
        self.assertAlmostEqual(a["difference"], -0.1)
        self.assertAlmostEqual(a["ci95_high"], -0.1)


if __name__ == "__main__":
    unittest.main()
