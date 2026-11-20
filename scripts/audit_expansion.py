"""Audit development images and safely extract missing tasks from the pinned archive."""

import argparse
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

from audit_dtd import file_hash
from prepare_expansion import DATASETS
from prepare_vtab import REVISION, EXPECTED_SHA256, EXPECTED_BYTES, validate_dataset

ROOT = Path(__file__).resolve().parents[1]


def prepare_missing(root, datasets):
    root = Path(root)
    destination = root / "data/vtab_processed"
    destination.mkdir(parents=True, exist_ok=True)
    missing = [name for name in datasets if not (destination / name).exists()]
    if not missing:
        return
    archive = root / "data/archives/vtab-1k.zip"
    if not archive.is_file() or archive.stat().st_size != EXPECTED_BYTES or file_hash(archive) != EXPECTED_SHA256:
        raise ValueError("A complete, checksum-verified pinned VTAB archive is required")
    with zipfile.ZipFile(archive) as source:
        for dataset in missing:
            staging = destination / f".{dataset}.extracting"
            staging.mkdir(exist_ok=False)
            selected = []
            for entry in source.infolist():
                parts = PurePosixPath(entry.filename).parts
                if not parts or entry.is_dir() or parts[0] == "__MACOSX":
                    continue
                offset = 0 if parts[0] in DATASETS else 1
                if len(parts) <= offset or parts[offset] != dataset:
                    continue
                if (PurePosixPath(entry.filename).is_absolute() or ".." in parts
                        or "\\" in entry.filename or stat.S_ISLNK(entry.external_attr >> 16)):
                    raise ValueError("Unsafe archive entry")
                selected.append((entry, parts[offset + 1:]))
            if not selected or shutil.disk_usage(destination).free < sum(e.file_size for e, _ in selected) + 1024**3:
                raise ValueError("Dataset missing from archive or insufficient disk space")
            for entry, parts in selected:
                target = staging.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open(entry) as incoming, target.open("xb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
            staging.rename(destination / dataset)


def audit(root, dataset):
    from PIL import Image
    root = Path(root)
    data_root = root / "data/vtab_processed"
    folder = data_root / dataset
    classes = DATASETS[dataset]
    summary = validate_dataset(data_root, dataset, classes, decode_test_images=False)
    records = {}
    for name in ("train800.txt", "val200.txt"):
        records[name] = []
        for index, row in enumerate((folder / name).read_text().splitlines()):
            path, label = row.rsplit(maxsplit=1)
            with Image.open(folder / path) as photo:
                photo.load()
            records[name].append({"index": index, "path": path, "label": int(label),
                                  "sha256": file_hash(folder / path)})
    training = {}
    for item in records["train800.txt"]:
        training.setdefault(item["sha256"], []).append(item)
    duplicates = [item for item in records["val200.txt"] if item["sha256"] in training]
    expected = 1 if dataset == "dtd" else 0
    if len(duplicates) != expected:
        raise ValueError(f"{dataset}: unexpected duplicate validation rows: {len(duplicates)}")
    if any({r["label"] for r in training[item["sha256"]]} != {item["label"]} for item in duplicates):
        raise ValueError("Identical images have conflicting development labels")
    clean = [r["index"] for r in records["val200.txt"] if r["sha256"] not in training]
    metadata = [p.relative_to(folder).as_posix() for p in folder.rglob("*")
                if p.is_file() and p.suffix.lower() in (".csv", ".json", ".h5", ".hdf5")]
    payload = {**summary, "phase": "complete", "dataset": dataset, "class_count": classes,
               "mirror_revision": REVISION, "development_records": records,
               "development_images_decoded": {name: len(rows) for name, rows in records.items()},
               "duplicate_validation_indices": [item["index"] for item in duplicates],
               "validation_indices_without_training_content": clean,
               "patient_slide_metadata_candidates": metadata,
               "patient_slide_split_verified": False,
               "test_images_decoded": 0, "test_predictions_generated": 0,
               "scope": "Development images and test split metadata only"}
    output = root / "manifests" / f"{dataset}_validation.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--prepare-missing", action="store_true")
    args = parser.parse_args()
    if args.prepare_missing:
        prepare_missing(ROOT, args.datasets)
    for dataset in args.datasets:
        result = audit(ROOT, dataset)
        print(json.dumps({"dataset": dataset, "phase": result["phase"],
                          "duplicates": result["duplicate_validation_indices"]}), flush=True)
