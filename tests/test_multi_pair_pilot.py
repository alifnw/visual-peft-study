import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from calibration_routing_pilot import load_dataset, EXTENDED_DATASETS
from multi_pair_pilot import route, matched_random_mask, parse_pairs, calibrate, evaluate_pairs, run_multi_pair


class MultiPairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.data = load_dataset(cls.root, 'dtd')
        cls.cache = calibrate(cls.data)[0]

    def test_both_fallback_rules_have_exact_endpoints(self):
        a = np.array([[0.8, 0.2], [0.4, 0.6]])
        b = np.array([[0.2, 0.8], [0.9, 0.1]])
        for rule in ('replacement', 'averaging'):
            np.testing.assert_array_equal(route(a, b, np.zeros(2, dtype=bool), rule), a)
            expected = b if rule == 'replacement' else (a + b) / 2
            np.testing.assert_array_equal(route(a, b, np.ones(2, dtype=bool), rule), expected)

    def test_random_routing_exact_count_including_empty_and_full(self):
        for k in (0, 1, 13, 40):
            mask = np.arange(40) < k
            for seed in (42, 43, 44):
                random = matched_random_mask(mask, seed)
                self.assertEqual(int(random.sum()), k)
                np.testing.assert_array_equal(random, matched_random_mask(mask, seed))

    def test_directed_pair_validation(self):
        self.assertEqual(len(parse_pairs(['all'])), 12)
        self.assertEqual(parse_pairs(['ssf:lora']), (('ssf', 'lora'),))
        for values in (['lora:lora'], ['ssf:bad'], ['all', 'lora:ssf'], ['ssf:lora', 'ssf:lora']):
            with self.assertRaises(ValueError):
                parse_pairs(values)

    def test_single_pair_matches_batch_and_rules_share_gate(self):
        one = evaluate_pairs(self.data, self.cache, (('linear', 'ssf'),), ('replacement', 'averaging'))
        batch = evaluate_pairs(self.data, self.cache, (('linear', 'lora'), ('linear', 'ssf')), ('replacement', 'averaging'))
        filtered = [row for row in batch[0] if row['second_method'] == 'ssf']
        pd.testing.assert_frame_equal(pd.DataFrame(one[0]), pd.DataFrame(filtered))
        decisions = pd.DataFrame(one[1])
        keys = ['split_seed', 'target_fallback_rate', 'sample_index']
        a = decisions[decisions.fallback_rule == 'replacement'].sort_values(keys)
        b = decisions[decisions.fallback_rule == 'averaging'].sort_values(keys)
        np.testing.assert_array_equal(a.fallback, b.fallback)
        metrics = pd.DataFrame(one[0])
        for _, group in metrics.groupby(['split_seed', 'fallback_rule', 'target_fallback_rate']):
            gate = group[group.strategy == 'confidence_routing']
            random = group[group.strategy == 'random_routing']
            self.assertTrue((random.fallback_count == gate.fallback_count.iloc[0]).all())

    def test_four_datasets_and_binary_metrics(self):
        for ds in EXTENDED_DATASETS:
            data = load_dataset(self.root, ds)
            self.assertEqual(len(data['targets']), 199 if ds == 'dtd' else 200)
        binary = load_dataset(self.root, 'patch_camelyon')
        self.assertEqual(binary['runs']['ssf']['logits'].shape, (200, 2))

    def test_missing_inputs_fail_without_creating_output_or_training_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'output'
            with self.assertRaises(FileNotFoundError):
                run_multi_pair(Path(temp), output, ('svhn',), (('linear', 'lora'),))
            self.assertFalse(output.exists())
        for name in ('calibration_routing_pilot.py', 'multi_pair_pilot.py'):
            tree = ast.parse((self.root / 'scripts' / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [x.name.split('.')[0] for x in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [(node.module or '').split('.')[0]]
                else:
                    continue
                self.assertFalse(set(names) & {'torch', 'torchvision', 'subprocess', 'train_validation', 'run_expansion'})


if __name__ == '__main__':
    unittest.main()
