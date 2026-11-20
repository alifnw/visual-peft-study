import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibration_routing_pilot import (
    _route, fit_temperature, load_dataset, make_folds, metrics_from_probabilities, softmax,
)


class CalibrationRoutingTests(unittest.TestCase):
    def test_fold_partitions_are_deterministic_disjoint_and_complete(self):
        first = make_folds(199, 42)
        second = make_folds(199, 42)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(first, second)))
        merged = np.concatenate(first)
        self.assertEqual(len(merged), 199)
        self.assertEqual(len(np.unique(merged)), 199)

    def test_temperature_preserves_predictions_and_can_reduce_nll(self):
        logits = np.array([[4.0, 0.0, -1.0], [0.0, 3.0, -2.0], [1.0, -1.0, 0.0]])
        targets = np.array([0, 1, 2])
        result = fit_temperature(logits, targets)
        before = metrics_from_probabilities(targets, softmax(logits))
        after = metrics_from_probabilities(targets, softmax(logits, result["temperature"]))
        self.assertEqual(before["accuracy"], after["accuracy"])
        self.assertLessEqual(after["nll"], before["nll"] + 1e-10)
        self.assertGreaterEqual(result["temperature"], 0.05)
        self.assertLessEqual(result["temperature"], 20.0)

    def test_routing_endpoints_match_linear_and_always_on_fusion(self):
        linear = np.array([[0.7, 0.3], [0.2, 0.8]])
        lora = np.array([[0.4, 0.6], [0.9, 0.1]])
        no_fallback = np.zeros(2, dtype=bool)
        full_fallback = np.ones(2, dtype=bool)
        self.assertTrue(np.array_equal(_route(linear, lora, no_fallback), linear))
        self.assertTrue(np.array_equal(_route(linear, lora, full_fallback), 0.5 * (linear + lora)))

    def test_multiclass_brier_uses_sum_over_classes(self):
        target = np.array([0])
        probabilities = np.array([[0.5, 0.25, 0.25]])
        metrics = metrics_from_probabilities(target, probabilities)
        self.assertAlmostEqual(metrics["brier"], 0.375)

    def test_saved_epoch_100_predictions_match_dataset_audits(self):
        root = Path(__file__).resolve().parents[1]
        dtd = load_dataset(root, "dtd")
        resisc = load_dataset(root, "resisc45")
        self.assertEqual(len(dtd["targets"]), 199)
        self.assertEqual(len(resisc["targets"]), 200)
        self.assertEqual(len(dtd["runs"]), 4)
        self.assertEqual(len(resisc["runs"]), 4)
        self.assertEqual(len({r["result"]["frozen_backbone_sha256"] for r in dtd["runs"].values()}), 1)
        self.assertEqual(len({r["result"]["frozen_backbone_sha256"] for r in resisc["runs"].values()}), 1)


if __name__ == "__main__":
    unittest.main()
