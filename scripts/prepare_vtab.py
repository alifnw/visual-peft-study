import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import time
import zipfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
REVISION = "a1c257fbbeebd3260c0a201fe6a6c9916b56eee7"
EXPECTED_SHA256 = "87edd008ebd4a2921609eea4a1629fb3c5ddaece46938d7e1fc1224723667726"
EXPECTED_BYTES = 4659851071
REPO_ID = "XiN0919/VTAB-1k"
FILENAME = "vtab-1k.zip"
SPLITS = {"train800.txt": 800, "val200.txt": 200, "train800val200.txt": 1000, "test.txt": None}
STATUS = ROOT / "manifests" / "dataset_status.json"


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def phase(name, **details):
    payload = {"phase": name, "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **details}
    save_json(STATUS, payload)
    print(json.dumps(payload), flush=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_datasets():
    source = ROOT / "code" / "utils" / "global_var.py"
    for statement in ast.parse(source.read_text()).body:
        if isinstance(statement, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "VTAB_DATASETS" for target in statement.targets):
            return ast.literal_eval(statement.value)
    raise RuntimeError("Cannot locate VTAB_DATASETS in the pinned official code")


def extract_archive(archive, datasets=None):
    target = ROOT / "data" / "vtab_processed"
    target.mkdir(parents=True, exist_ok=True)
    if any(target.iterdir()):
        raise RuntimeError("Extraction target is nonempty. Preserve it and inspect before rerunning")
    with zipfile.ZipFile(archive) as zipped:
        entries = zipped.infolist()
        image_entries = [entry for entry in entries if not entry.is_dir() and not entry.filename.startswith("__MACOSX/")]
        first_parts = {PurePosixPath(entry.filename).parts[0] for entry in image_entries}
        prefix = next(iter(first_parts)) if len(first_parts) == 1 else None
        if prefix not in {"vtab-1k", "vtab_processed", "vtab-1k-combined"}:
            prefix = None
        if datasets:
            selected = set(datasets)
            image_entries = [entry for entry in image_entries
                             if PurePosixPath(entry.filename).parts[1 if prefix else 0] in selected]
            if not image_entries:
                raise RuntimeError("Requested datasets are absent from the pinned archive")
        total = sum(entry.file_size for entry in image_entries)
        free = shutil.disk_usage(target).free
        if free < total + 1024**3:
            raise RuntimeError("Insufficient disk space for safe extraction")
        phase("extracting", archive_bytes=archive.stat().st_size, uncompressed_bytes=total, files=len(image_entries))
        for index, entry in enumerate(image_entries, 1):
            relative = PurePosixPath(entry.filename)
            if relative.is_absolute() or ".." in relative.parts or "\\" in entry.filename:
                raise RuntimeError("Unsafe archive member path")
            if stat.S_ISLNK(entry.external_attr >> 16):
                raise RuntimeError("Archive symlinks are not accepted")
            parts = relative.parts[1:] if prefix else relative.parts
            if not parts:
                continue
            destination = target.joinpath(*parts)
            if not destination.resolve().is_relative_to(target.resolve()):
                raise RuntimeError("Archive member escapes the target")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(entry) as source, destination.open("xb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            if index % 25000 == 0:
                print("Extracted", index, "of", len(image_entries), flush=True)
    return target


def validate_dataset(root, name, classes, decode_test_images=True):
    directory = root / name
    summary = {"classes_expected": classes, "splits": {}}
    split_records = {}
    content_records = {}
    for filename, expected_count in SPLITS.items():
        source = directory / filename
        lines = [line.strip() for line in source.read_text().splitlines() if line.strip()]
        if expected_count is not None and len(lines) != expected_count:
            raise RuntimeError(f"{name}/{filename}: unexpected row count {len(lines)}")
        records = []
        class_counts = {}
        for line in lines:
            image_path, label_text = line.rsplit(maxsplit=1)
            label = int(label_text)
            image = (directory / image_path).resolve()
            if not image.is_relative_to(directory.resolve()) or not image.is_file():
                raise RuntimeError(f"Missing or unsafe image referenced in {name}/{filename}")
            if not 0 <= label < classes:
                raise RuntimeError(f"Out-of-range class label in {name}/{filename}")
            records.append((image_path, label))
            class_counts[label] = class_counts.get(label, 0) + 1
        if len({path for path, _ in records}) != len(records):
            raise RuntimeError(f"Repeated image path in {name}/{filename}")
        if not records:
            raise RuntimeError(f"Empty split {name}/{filename}")
        step = max(1, len(records) // 10)
        sampled = [] if filename == "test.txt" and not decode_test_images else records[::step][:10]
        for image_path, _ in sampled:
            with Image.open(directory / image_path) as image:
                image.load()
        summary["splits"][filename] = {
            "rows": len(records),
            "list_sha256": sha256(source),
            "classes_observed": len(class_counts),
            "class_counts": class_counts,
            "sampled_images_decoded": len(sampled),
        }
        split_records[filename] = set(records)
        if filename != "test.txt":
            content_records[filename] = Counter((sha256(directory / path), label) for path, label in records)
    train = split_records["train800.txt"]
    validation = split_records["val200.txt"]
    test = split_records["test.txt"]
    train_paths = {path for path, _ in train}
    validation_paths = {path for path, _ in validation}
    test_paths = {path for path, _ in test}
    if train_paths & validation_paths or train_paths & test_paths or validation_paths & test_paths:
        raise RuntimeError(f"Overlapping path-label records in {name}")
    if content_records["train800.txt"] + content_records["val200.txt"] != content_records["train800val200.txt"]:
        raise RuntimeError(f"Combined training image content/labels do not match the train/validation union for {name}")
    summary["path_label_split_overlap"] = False
    summary["combined_image_content_matches_train_val_union"] = True
    train_hashes = {digest for digest, _ in content_records["train800.txt"]}
    validation_hashes = {digest for digest, _ in content_records["val200.txt"]}
    summary["train_val_identical_file_content_count"] = len(train_hashes & validation_hashes)
    summary["split_membership_matches_authors_seed_9526"] = "unverified"
    print("Validated", name, {key: value["rows"] for key, value in summary["splits"].items()}, flush=True)
    return summary


def main(validate_only=False, datasets=None):
    archive = ROOT / "data" / "archives" / FILENAME
    choices = expected_datasets()
    selected = list(dict.fromkeys(datasets)) if datasets else list(choices)
    if any(name not in choices for name in selected):
        raise ValueError("Unknown VTAB dataset requested")
    if not validate_only and not archive.exists():
        from huggingface_hub import hf_hub_download

        phase("downloading", source_repo=REPO_ID, revision=REVISION, expected_bytes=EXPECTED_BYTES)
        archive = Path(hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            filename=FILENAME,
            revision=REVISION,
            local_dir=str(ROOT / "data" / "archives"),
            token=False,
        ))
    phase("hashing", archive_bytes=archive.stat().st_size)
    observed_sha = sha256(archive)
    if archive.stat().st_size != EXPECTED_BYTES or observed_sha != EXPECTED_SHA256:
        raise RuntimeError("Downloaded archive does not match its published size/SHA-256")
    root = ROOT / "data" / "vtab_processed" if validate_only else extract_archive(archive, selected)
    phase("validating")
    results = {}
    for name in selected:
        results[name] = validate_dataset(root, name, choices[name], decode_test_images=False)
    manifest = {
        "source_repo": REPO_ID,
        "source_url": "https://huggingface.co/datasets/" + REPO_ID,
        "revision": REVISION,
        "filename": FILENAME,
        "archive_bytes": archive.stat().st_size,
        "archive_sha256": observed_sha,
        "official_code_commit": subprocess.check_output(["git", "-C", str(ROOT / "code"), "rev-parse", "HEAD"], text=True).strip(),
        "data_path": str(root),
        "dataset_count": len(results),
        "datasets_requested": selected,
        "datasets": results,
        "provenance_note": "Community mirror. Author SharePoint archive returned HTTP 404. Split identity with the authors' seed-9526 archive is not verified. Do not claim an exact published-score reproduction on this evidence alone.",
        "validation_scope": "For the requested datasets, all referenced paths, label bounds, row counts and path overlaps checked. Combined train/validation file content and labels checked by SHA-256 multiset union. Training/validation identical-file counts recorded. Selected ZIP member CRC checked while extracting. Up to ten images per development split fully decoded. Test images are not decoded. Content-level train/test duplicate detection not performed.",
        "license_note": "Observe the licenses and access conditions of each original VTAB component dataset. The mirror has no dataset card establishing an umbrella license.",
    }
    save_json(ROOT / "manifests" / "vtab_manifest.json", manifest)
    phase("complete", dataset_count=len(results), data_path=str(root), archive_sha256=observed_sha, exact_author_split_verified=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--dataset", action="append", help="Extract and validate only this VTAB dataset. Repeat to select several.")
    arguments = parser.parse_args()
    try:
        main(validate_only=arguments.validate_only, datasets=arguments.dataset)
    except Exception as error:
        phase("failed", error_type=type(error).__name__, message=str(error))
        raise
