import importlib.util
from pathlib import Path
import unittest


source = Path(__file__).resolve().parents[1] / "scripts/profile_training.py"
spec = importlib.util.spec_from_file_location("profile_training", source)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)


class TrainingProfileTests(unittest.TestCase):
    def test_mean_accounts_for_unequal_sample_durations(self):
        rows = [{"elapsed_seconds": 0, "process_tree_rss_mib": 100},
                {"elapsed_seconds": 2, "process_tree_rss_mib": 400}]
        result = profile.summarize(rows, 10)["process_tree_rss_mib"]
        self.assertEqual(result["time_weighted_mean"], 340)
        self.assertEqual(result["observed_peak"], 400)
        self.assertEqual(result["covered_seconds"], 10)

    def test_missing_gpu_sample_is_not_zero(self):
        rows = [{"elapsed_seconds": 0, "process_tree_vram_mib": 100},
                {"elapsed_seconds": 5}]
        result = profile.summarize(rows, 10)["process_tree_vram_mib"]
        self.assertEqual(result["time_weighted_mean"], 100)
        self.assertEqual(result["covered_seconds"], 5)

    def test_no_observations_remain_unavailable(self):
        result = profile.summarize([], 10)["process_tree_vram_mib"]
        self.assertIsNone(result["time_weighted_mean"])
        self.assertIsNone(result["observed_peak"])


if __name__ == "__main__":
    unittest.main()
