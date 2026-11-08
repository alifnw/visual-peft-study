"""Audit existing DTD development images without decoding test images."""

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_audit_manifest(config, root):
    relative = config.get("dataset_audit_manifest")
    if relative is None:
        return None
    path = Path(root) / relative
    audit = json.loads(path.read_text())
    if audit.get("phase") != "complete" or audit.get("dataset") != config["dataset"]:
        raise ValueError("Dataset audit is incomplete or belongs to another dataset")
    classes = config["expected_class_count"]
    if audit.get("class_count") != classes:
        raise ValueError("Dataset audit class count differs from the configuration")
    if audit.get("path_label_split_overlap") is not False:
        raise ValueError("Dataset audit has not established disjoint splits")
    for name, count in (("train800.txt", 800), ("val200.txt", 200)):
        record = audit["splits"][name]
        if record["rows"] != count or record["classes_observed"] != classes:
            raise ValueError("Dataset audit has unexpected rows or missing classes")
        source = Path(root) / "data/vtab_processed" / config["dataset"] / name
        if file_hash(source) != record["list_sha256"]:
            raise ValueError("Split changed after the dataset audit")
    duplicates = audit.get("duplicate_validation_indices", [])
    expected = config.get("expected_duplicate_validation_rows", 0)
    if len(duplicates) != expected or audit.get("train_val_identical_file_content_count") != expected:
        raise ValueError("Unexpected identical image content across development splits")
    clean = audit.get("validation_indices_without_training_content", list(range(200)))
    if len(set(clean)) != len(clean) or sorted(clean + duplicates) != list(range(200)):
        raise ValueError("Validation overlap masks do not partition all 200 rows")
    if set(clean) & set(duplicates):
        raise ValueError("Clean and duplicate validation masks overlap")
    return {"path": relative, "sha256": file_hash(path), "dataset": config["dataset"],
            "duplicate_validation_indices": duplicates,
            "validation_indices_without_training_content": clean}


def audit(root, expected_duplicate_validation_rows=0):
    from PIL import Image
    from prepare_vtab import REVISION, validate_dataset

    root = Path(root)
    data_root = root / "data/vtab_processed"
    summary = validate_dataset(data_root, "dtd", 47, decode_test_images=False)
    decoded = {}
    records = {}
    for name in ("train800.txt", "val200.txt"):
        if summary["splits"][name]["classes_observed"] != 47:
            raise ValueError("All 47 DTD classes must occur in both development splits")
        rows = (data_root / "dtd" / name).read_text().splitlines()
        records[name] = []
        for index, row in enumerate(rows):
            image_path, label = row.rsplit(maxsplit=1)
            with Image.open(data_root / "dtd" / image_path) as image:
                image.load()
            records[name].append({"index": index, "path": image_path,
                                  "label": int(label), "sha256": file_hash(data_root / "dtd" / image_path)})
        decoded[name] = len(rows)
    training = {}
    for row in records["train800.txt"]:
        training.setdefault(row["sha256"], []).append(row)
    duplicates = [row for row in records["val200.txt"] if row["sha256"] in training]
    if len(duplicates) != expected_duplicate_validation_rows:
        raise ValueError("Unexpected identical image content across development splits")
    if any(set(item["label"] for item in training[row["sha256"]]) != {row["label"]} for row in duplicates):
        raise ValueError("Identical development images have conflicting labels")
    clean = [row for row in records["val200.txt"] if row["sha256"] not in training]
    if len({row["label"] for row in clean}) != 47:
        raise ValueError("Validation without training duplicates must still cover all 47 classes")
    result = {
        "phase": "complete", "dataset": "dtd", "class_count": 47,
        "mirror_revision": REVISION, **summary,
        "development_images_decoded": decoded,
        "duplicate_validation_indices": [row["index"] for row in duplicates],
        "cross_split_duplicates": [{"validation": row, "training": training[row["sha256"]]} for row in duplicates],
        "validation_indices_without_training_content": [row["index"] for row in clean],
        "clean_validation_classes": 47,
        "overlap_policy": "Preserve the original split and also report validation without training duplicates",
        "test_images_decoded": 0, "test_predictions_generated": 0,
        "scope": "Development image integrity and split metadata. Test images are not opened.",
    }
    output = root / "manifests/dtd_validation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--expected-duplicate-validation-rows", type=int, default=1)
    args = parser.parse_args()
    result = audit(args.root, args.expected_duplicate_validation_rows)
    print(json.dumps({"phase": result["phase"], "dataset": result["dataset"],
                      "development_images_decoded": result["development_images_decoded"],
                      "duplicate_validation_indices": result["duplicate_validation_indices"],
                      "validation_rows_without_training_content": len(result["validation_indices_without_training_content"]),
                      "test_predictions_generated": 0}, indent=2))
