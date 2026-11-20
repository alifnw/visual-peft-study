"""Prepare and run the fixed SVHN seed-42 label-noise experiment."""

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

from prepare_weights import sha256 as sha
from resource_guard import snapshot, stop_reason


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("convpass", "ssf", "lora")
NOISE_SEED = 2026
SPLITS = ("train800.txt", "val200.txt")
SOURCES = {
    "peft_methods.py": "method_helpers_sha256",
    "gradient_accumulation.py": "gradient_accumulation_source_sha256",
    "initialization_reference.py": "initialization_helpers_sha256",
    "smoke_test.py": "shared_helpers_sha256",
    "resource_guard.py": "resource_guard_source_sha256",
}
INIT = ("classifier_initial_sha256", "frozen_backbone_sha256", "pretrained_backbone_initial_sha256")


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sample(gpu, config, minimum):
    raw = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"], text=True, timeout=4,
    )
    free, utilization = (int(part.strip()) for part in raw.strip().split(","))
    ram = snapshot()
    disk = shutil.disk_usage(ROOT).free // 2**20
    stats = {"free_gpu_mib": free, "utilization": utilization,
             "ram_headroom_gib": ram["headroom_gib"], "free_disk_mib": disk}
    active_other_gpus = sum(
        path.parent.name != f"gpu-{gpu}" and read(path).get("phase") == "training"
        for path in (ROOT / ".runtime/seed_matrix").glob("gpu-*/status.json")
    )
    floor = config["minimum_conservative_ram_headroom_gib"] * (active_other_gpus + 1)
    ready = (free >= minimum and utilization <= 20 and ram["headroom_gib"] >= floor
             and disk >= 4096 and not stop_reason(ram, ram["oom_kill_count"]))
    return ready, stats


def rows(path):
    return [(image, int(label)) for image, label in
            (line.rsplit(maxsplit=1) for line in path.read_text().splitlines())]


def corrupt(training):
    if len(training) != 800 or len({p for p, _ in training}) != 800 or {y for _, y in training} != set(range(10)):
        raise ValueError("Expected 800 unique SVHN training paths covering ten classes")
    rng = np.random.Generator(np.random.PCG64(NOISE_SEED))
    positions = sorted(rng.choice(len(training), 160, replace=False).tolist())
    flips = []
    perturbed = training.copy()
    for position, draw in zip(positions, rng.integers(0, 9, size=len(positions)).tolist()):
        path, label = training[position]
        replacement = draw + int(draw >= label)
        perturbed[position] = (path, replacement)
        flips.append({"index": position, "path": path, "original_label": label, "noisy_label": replacement})
    return perturbed, flips


def prepare(study, baseline):
    dataset = ROOT / "data/vtab_processed/svhn"
    train, val = (rows(dataset / name) for name in SPLITS)
    if len(val) != 200 or {y for _, y in val} != set(range(10)) or {p for p, _ in train} & {p for p, _ in val}:
        raise ValueError("Validation split is incomplete or overlaps training paths")
    noisy, flips = corrupt(train)
    expected = {name: sha(dataset / name) for name in SPLITS}
    weights_path = ROOT / "code/pretrained_weights/ViT-B_16_in21k.npz"
    weights_sha = sha(weights_path)
    weight_manifest = read(ROOT / "manifests/pretrained_weights.json")
    if weights_sha != weight_manifest["sha256"]:
        raise ValueError("Pretrained file differs from node manifest")
    native_commit = subprocess.check_output(
        ["git", "-C", str(ROOT / "code"), "rev-parse", "HEAD"], text=True).strip()
    trainer_sha = sha(ROOT / "scripts/train_validation.py")
    with tempfile.TemporaryDirectory(prefix=study.name + "-", dir=study.parent) as directory:
        stage = Path(directory) / "study"
        stage.mkdir()
        clean, configs = {}, {}
        for method in METHODS:
            result, provenance = (baseline[method][key] for key in ("result", "provenance"))
            config = result["config"]
            if ((result["dataset"], result["method"], result["seed"], result["epochs"], result["phase"])
                    != ("svhn", method, 42, 100, "complete") or result["rows"]["test_used"]
                    or not result["checkpoint_reload_exact"] or config["test_set_access"]):
                raise ValueError(f"Invalid clean control for {method}")
            if expected != provenance["split_sha256"] or weights_sha != provenance["weights_manifest"]["sha256"]:
                raise ValueError(f"Clean split or weights differ for {method}")
            if native_commit != provenance["source_commit"] or any(
                    sha(ROOT / path) != value for path, value in provenance["native_source_sha256"].items()):
                raise ValueError(f"Native model or training source differs for {method}")
            for source, field in SOURCES.items():
                if field in provenance and sha(ROOT / "scripts" / source) != provenance[field]:
                    raise ValueError(f"Clean helper differs for {method}: {source}")
            if any(version(name) != value for name, value in provenance["versions"].items()):
                raise ValueError(f"Clean Python package versions differ for {method}")
            save(stage / f"reference/{method}.json", result)
            configs[method] = config
            clean[method] = {"trainer_sha256": provenance["script_sha256"],
                             "reference_sha256": sha(stage / f"reference/{method}.json")}

        staged_dataset = stage / "data/vtab_processed/svhn"
        staged_dataset.mkdir(parents=True)
        for path, _ in train + val:
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts or not (dataset / relative).is_file():
                raise ValueError(f"Invalid SVHN image path: {path}")
            link = staged_dataset / relative
            link.parent.mkdir(parents=True, exist_ok=True)
            link.symlink_to(dataset / relative)
        noisy_split = staged_dataset / SPLITS[0]
        noisy_split.write_text("".join(f"{p} {y}\n" for p, y in noisy))
        (staged_dataset / SPLITS[1]).symlink_to(dataset / SPLITS[1])
        save(stage / "corruption.json", {
            "dataset": "svhn", "seed": NOISE_SEED, "rng": "NumPy PCG64",
            "kind": "symmetric_other_class", "train_rows": len(train), "changed_rows": len(flips),
            "clean_split_sha256": expected, "noisy_train_sha256": sha(noisy_split),
            "changes": flips,
        })
        for method, config in configs.items():
            save(stage / f"configs/{method}.json", {
                **config, "experiment": f"svhn-{method}-seed42-label-noise20",
                "purpose": "svhn_seed42_controlled_training_label_noise_pilot",
                "label_noise_manifest_sha256": sha(stage / "corruption.json"),
            })
        scripts = stage / "scripts"
        scripts.mkdir()
        for source in (*SOURCES, "train_validation.py", "audit_dtd.py"):
            shutil.copy2(ROOT / "scripts" / source, scripts / source)
        (stage / "code").symlink_to(ROOT / "code", target_is_directory=True)
        (stage / "manifests").symlink_to(ROOT / "manifests", target_is_directory=True)
        (stage / "runs").mkdir()
        # The isolated split has its own output tree and uses the existing project training lock.
        os.link(ROOT / "runs/.training.lock", stage / "runs/.training.lock")
        save(stage / "protocol.json", {"methods": METHODS, "training_seed": 42, "epochs": 100,
             "clean_controls": clean, "trainer_sha256": trainer_sha, "upstream_commit": native_commit,
             "pretrained_sha256": weights_sha, "corruption_manifest_sha256": sha(stage / "corruption.json"),
             "configs_sha256": {m: sha(stage / f"configs/{m}.json") for m in METHODS}})
        if study.exists():
            raise FileExistsError(study)
        stage.rename(study)


def verify(study, method):
    directory = study / f"runs/{method}"
    result, provenance = (read(directory / f"{name}.json") for name in ("result", "provenance"))
    control = read(study / f"reference/{method}.json")
    corruption = read(study / "corruption.json")
    if ((result["phase"], result["dataset"], result["method"], result["seed"], result["epochs"])
            != ("complete", "svhn", method, 42, 100) or not result["checkpoint_reload_exact"]
            or result["rows"]["test_used"] or not result["frozen_backbone_unchanged"]):
        raise ValueError(f"Incomplete run for {method}")
    if (provenance["split_sha256"] != {**corruption["clean_split_sha256"],
                                       SPLITS[0]: corruption["noisy_train_sha256"]}
            or result["config"] != read(study / f"configs/{method}.json")
            or not all(result[key] == control[key] for key in INIT)):
        raise ValueError(f"Run provenance or initialization mismatch for {method}")
    with (directory / "history.csv").open() as handle:
        if [int(row["epoch"]) for row in csv.DictReader(handle)] != list(range(1, 101)):
            raise ValueError(f"Incomplete history for {method}")
    validation = rows(study / "data/vtab_processed/svhn/val200.txt")
    with np.load(directory / "validation_last.npz") as a, np.load(directory / "validation_last_reloaded.npz") as b:
        if (a["logits"].shape != (200, 10) or not np.isfinite(a["logits"]).all()
                or not np.array_equal(a["targets"], [y for _, y in validation])
                or not np.array_equal(a["image_paths"], [p for p, _ in validation])
                or set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files)
                or abs(100 * (a["logits"].argmax(1) == a["targets"]).mean()
                       - result["last_epoch_validation_top1_percent"]) > 1e-6):
            raise ValueError(f"Validation artifacts differ for {method}")
    return result["last_epoch_validation_top1_percent"]


def run(study, gpu, launch_until):
    protocol = read(study / "protocol.json")
    if (not (study / "runs/.training.lock").samefile(ROOT / "runs/.training.lock")
            or sha(study / "corruption.json") != protocol["corruption_manifest_sha256"]
            or sha(study / "scripts/train_validation.py") != protocol["trainer_sha256"]):
        raise ValueError("Staged protocol or project training lock changed")
    manifest = read(study / "corruption.json")
    audited = {**manifest["clean_split_sha256"], SPLITS[0]: manifest["noisy_train_sha256"]}
    if any(sha(study / "data/vtab_processed/svhn" / name) != value for name, value in audited.items()):
        raise ValueError("Staged split changed")
    with (study / "queue.lock").open("a") as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for method in METHODS:
            directory = study / f"runs/{method}"
            if (directory / "result.json").is_file():
                verify(study, method)
                continue
            config_path = study / f"configs/{method}.json"
            if sha(config_path) != protocol["configs_sha256"][method]:
                raise ValueError(f"Configuration changed for {method}")
            config = read(config_path)
            minimum = config["minimum_free_gpu_mib"]
            ready_samples = 0
            while ready_samples < 2:
                if datetime.now(timezone.utc) >= launch_until:
                    save(study / "queue.json", {"phase": "launch_deadline", "pending_method": method})
                    return
                ready, stats = sample(gpu, config, minimum)
                with (ROOT / "runs/.training.lock").open("a") as project_lock:
                    try:
                        fcntl.flock(project_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        ready = False
                ready_samples = ready_samples + 1 if ready else 0
                save(study / "queue.json", {"phase": "waiting_resources", "method": method, **stats})
                if ready_samples < 2:
                    time.sleep(5)
            resumed = (directory / "last.pt").is_file()
            if directory.exists() and not resumed:
                if read(directory / "status.json")["phase"] != "blocked_resources":
                    raise ValueError(f"Incomplete attempt without checkpoint needs inspection: {directory}")
                directory.rename(directory.with_name(f"{method}-blocked-{time.time_ns()}"))
            profile = study / f"profiles/{method}-{time.time_ns()}"
            command = [
                sys.executable, str(ROOT / "scripts/profile_training.py"), "--gpu", gpu,
                "--output", str(profile), "--result", str(directory / "result.json"), "--",
                sys.executable, str(study / "scripts/train_validation.py"), "--method", method,
                "--config", str(config_path), "--output", str(directory),
                "--reference-result", str(study / f"reference/{method}.json"), "--trim-host-allocator",
            ]
            if resumed:
                command.append("--resume")
            save(study / "queue.json", {"phase": "training", "method": method, "profile": str(profile),
                 "started_utc": datetime.now(timezone.utc).isoformat(), "resume": resumed})
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                save(study / "queue.json", {"phase": "needs_review", "method": method,
                     "profile": str(profile), "returncode": result.returncode})
                return
            accuracy = verify(study, method)
            save(study / f"{method}-verified.json", {"accuracy_percent": accuracy, "profile": str(profile)})
        save(study / "queue.json", {"phase": "complete", "methods": METHODS})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "run"))
    parser.add_argument("--study", type=Path, default=ROOT / ".runtime/svhn-label-noise20")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--launch-until", type=datetime.fromisoformat)
    args = parser.parse_args()
    study = args.study.resolve()
    if args.action == "prepare":
        if args.baseline is None:
            parser.error("Preparation requires the clean result and provenance bundle")
        prepare(study, read(args.baseline))
    else:
        if args.launch_until is None or args.launch_until.tzinfo is None:
            parser.error("Supply the last new-run launch time with a timezone")
        run(study, args.gpu, args.launch_until)


if __name__ == "__main__":
    main()
