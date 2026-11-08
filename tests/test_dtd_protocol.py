"""Reject stale split audits and Flowers102 settings in DTD runs."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_dtd import audit, file_hash, validate_audit_manifest
from peft_methods import expected_method_parameters


class DTDProtocolTests(unittest.TestCase):
    def test_recipes_and_classifier_size(self):
        recipes = {"linear": (0.001, 0.0001, 0), "lora": (0.001, 0.0001, 36864),
                   "ssf": (0.01, 0.001, 205824), "convpass": (0.01, 0.001, 327552)}
        for method, (lr, wd, adapters) in recipes.items():
            with self.subTest(method=method):
                config = json.loads((ROOT / f"configs/dtd_{method}.json").read_text())
                self.assertEqual(config["dataset"], "dtd")
                self.assertEqual(config["methods"], [method])
                self.assertEqual(config["expected_class_count"], 47)
                self.assertEqual(config["epochs"], 100)
                self.assertEqual(config["seed"], 42)
                self.assertEqual(config["expected_duplicate_validation_rows"], 1)
                self.assertEqual(config["batch_size"] // config["microbatch_size"], 8)
                self.assertEqual((config["learning_rate"], config["weight_decay"]), (lr, wd))
                self.assertFalse(config["test_set_access"])
                self.assertEqual(expected_method_parameters(config, method), adapters)
                self.assertEqual(768 * config["expected_class_count"] + config["expected_class_count"], 36143)

    def fixture(self, root):
        data = root / "data/vtab_processed/dtd"
        data.mkdir(parents=True)
        audit = {"phase": "complete", "dataset": "dtd", "class_count": 47,
                 "path_label_split_overlap": False,
                 "train_val_identical_file_content_count": 0, "splits": {}}
        for name, rows in (("train800.txt", 800), ("val200.txt", 200)):
            path = data / name
            path.write_text("sample.png 0\n")
            audit["splits"][name] = {"rows": rows, "classes_observed": 47,
                                     "list_sha256": file_hash(path)}
        (root / "manifests").mkdir()
        path = root / "manifests/dtd_validation.json"
        path.write_text(json.dumps(audit))
        config = {"dataset": "dtd", "expected_class_count": 47,
                  "dataset_audit_manifest": "manifests/dtd_validation.json"}
        return config, audit, path

    def test_split_changed_after_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, _, _ = self.fixture(root)
            self.assertEqual(validate_audit_manifest(config, root)["dataset"], "dtd")
            (root / "data/vtab_processed/dtd/val200.txt").write_text("changed.png 0\n")
            with self.assertRaisesRegex(ValueError, "Split changed"):
                validate_audit_manifest(config, root)

    def test_incomplete_wrong_dataset_or_leaking_audit_is_rejected(self):
        changes = [{"phase": "failed"}, {"dataset": "oxford_flowers102"},
                   {"class_count": 102}, {"train_val_identical_file_content_count": 1},
                   {"path_label_split_overlap": True}]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                config, audit, path = self.fixture(root)
                audit.update(change)
                path.write_text(json.dumps(audit))
                with self.assertRaises(ValueError):
                    validate_audit_manifest(config, root)

    def test_cp1_configs_do_not_require_a_dtd_audit(self):
        self.assertIsNone(validate_audit_manifest({"dataset": "oxford_flowers102"}, ROOT))

    def test_audit_decodes_development_and_never_opens_test_images(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            data = root / "data/vtab_processed/dtd"
            data.mkdir(parents=True)
            rows = []
            for index in range(1000):
                name = f"image_{index}.png"
                Image.new("RGB", (2, 2), (index // 256, index % 256, 0)).save(data / name)
                rows.append(f"{name} {index % 47}\n")
            (data / "train800.txt").write_text("".join(rows[:800]))
            (data / "val200.txt").write_text("".join(rows[800:]))
            (data / "train800val200.txt").write_text("".join(rows))
            (data / "test.bin").write_bytes(b"deliberately not an image")
            (data / "test.txt").write_text("test.bin 0\n")
            result = audit(root)
            self.assertEqual(result["development_images_decoded"], {"train800.txt": 800, "val200.txt": 200})
            self.assertEqual(result["test_images_decoded"], 0)
            self.assertEqual(result["splits"]["test.txt"]["sampled_images_decoded"], 0)
            (data / "image_800.png").write_bytes((data / "image_1.png").read_bytes())
            with self.assertRaisesRegex(ValueError, "identical"):
                audit(root)
            result = audit(root, expected_duplicate_validation_rows=1)
            self.assertEqual(result["duplicate_validation_indices"], [0])
            self.assertEqual(len(result["validation_indices_without_training_content"]), 199)
            self.assertEqual(result["clean_validation_classes"], 47)

    def test_declared_duplicate_requires_a_valid_validation_partition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config, audit, path = self.fixture(root)
            config["expected_duplicate_validation_rows"] = 1
            audit.update(train_val_identical_file_content_count=1,
                         duplicate_validation_indices=[0],
                         validation_indices_without_training_content=list(range(1, 200)))
            path.write_text(json.dumps(audit))
            self.assertEqual(len(validate_audit_manifest(config, root)["validation_indices_without_training_content"]), 199)
            audit["validation_indices_without_training_content"][0] = 0
            path.write_text(json.dumps(audit))
            with self.assertRaises(ValueError):
                validate_audit_manifest(config, root)


if __name__ == "__main__":
    unittest.main()
