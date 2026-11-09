from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import prepare_vtab


class SelectedDatasetExtractionTests(unittest.TestCase):
    def test_extracts_only_requested_dataset_with_archive_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "vtab.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("vtab-1k/dtd/train800.txt", "images/a.jpg 0\n")
                z.writestr("vtab-1k/other/train800.txt", "images/b.jpg 0\n")
            with patch.object(prepare_vtab, "ROOT", root), patch.object(prepare_vtab, "phase"):
                output = prepare_vtab.extract_archive(archive, ["dtd"])
            self.assertTrue((output / "dtd/train800.txt").is_file())
            self.assertFalse((output / "other").exists())

    def test_rejects_unsafe_member_in_selected_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "vtab.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("vtab-1k/dtd/../../escape.txt", "invalid")
            with patch.object(prepare_vtab, "ROOT", root), patch.object(prepare_vtab, "phase"):
                with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                    prepare_vtab.extract_archive(archive, ["dtd"])
            self.assertFalse((root / "escape.txt").exists())

    def test_missing_dataset_fails_explicitly(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "vtab.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("vtab-1k/other/train800.txt", "images/b.jpg 0\n")
            with patch.object(prepare_vtab, "ROOT", root), patch.object(prepare_vtab, "phase"):
                with self.assertRaisesRegex(RuntimeError, "absent"):
                    prepare_vtab.extract_archive(archive, ["dtd"])


if __name__ == "__main__":
    unittest.main()
