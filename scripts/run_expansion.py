"""Serialize existing pending jobs and published-recipe expansion with a persistent budget."""

import argparse
import csv
from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from prepare_expansion import ADDITIONAL_METHODS, DATASETS
from smoke_test import resource_state, save

ROOT = Path(__file__).resolve().parents[1]


def complete(path, method, dataset, epochs=100):
    try:
        result = json.loads((path / "result.json").read_text())
        valid = (result["phase"] == "complete" and result["method"] == method
                and result["dataset"] == dataset and result["epochs"] == epochs
                and result.get("checkpoint_reload_exact") is True
                and result.get("smoke_check", False) == (epochs == 2)
                and all((path / f).is_file() for f in ("history.csv", "validation_last.npz", "last.pt",
                                                      "validation_last_reloaded.npz")))
        if not valid:
            return False
        import numpy as np
        with (path / "history.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != epochs or [int(row["epoch"]) for row in rows] != list(range(1, epochs+1)):
            return False
        with np.load(path / "validation_last.npz", allow_pickle=False) as original:
            logits, targets = original["logits"], original["targets"]
            if logits.shape != (4 if epochs == 2 else 200, DATASETS[dataset]) or not np.isfinite(logits).all():
                return False
            if targets.shape != (len(logits),) or targets.min() < 0 or targets.max() >= DATASETS[dataset]:
                return False
            with np.load(path / "validation_last_reloaded.npz", allow_pickle=False) as reloaded:
                if not np.array_equal(logits, reloaded["logits"]) or not np.array_equal(targets, reloaded["targets"]):
                    return False
            score = 100 * (logits.argmax(axis=1) == targets).mean()
        return bool(abs(score - result["last_epoch_validation_top1_percent"]) < 1e-6)
    except (OSError, ValueError, KeyError):
        return False


def jobs(root=ROOT):
    result = []
    for dataset in ("svhn", "patch_camelyon"):
        for method in ("linear", "lora", "ssf", "convpass"):
            config = Path("configs") / f"{dataset}_{method}.json"
            if (root / config).is_file():
                result.append({"dataset": dataset, "method": method, "config": str(config),
                               "output": f"runs/{dataset}/{method}", "legacy": True})
    for method in ADDITIONAL_METHODS:
        for dataset in DATASETS:
            result.append({"dataset": dataset, "method": method,
                           "config": f"configs/{dataset}_{method}.json",
                           "output": f"runs/development/{dataset}/{method}/seed42", "legacy": False})
    return result


def lock_available(root=ROOT):
    (root / "runs").mkdir(exist_ok=True)
    with (root / "runs/.training.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True


def ready(state, config):
    return (state["conservative_headroom_gib"] >= config["minimum_conservative_ram_headroom_gib"]
            and state["gpu_total_mib"] - state["gpu_used_mib"] >= config["minimum_free_gpu_mib"]
            and state["gpu_utilization_percent"] <= 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-hours", type=float, default=24)
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    parser.add_argument("--minimum-free-gpu-mib", type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    queue = jobs()
    if args.inventory_only:
        print(json.dumps([{**job, "complete": complete(ROOT / job["output"], job["method"], job["dataset"])}
                          for job in queue], indent=2))
        return 0
    runtime = ROOT / ".runtime/expansion"
    runtime.mkdir(parents=True, exist_ok=True)
    guard = (runtime / "supervisor.lock").open("a")
    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
    budget = runtime / "budget.json"
    now = datetime.now(timezone.utc)
    if not budget.exists():
        save(budget, {"started_utc": now.isoformat(),
                      "deadline_utc": (now + timedelta(hours=args.max_hours)).isoformat()})
    deadline = datetime.fromisoformat(json.loads(budget.read_text())["deadline_utc"])
    previous = json.loads((runtime / "status.json").read_text()) if (runtime / "status.json").exists() else {}
    state = {"schema_version": 2, "host": socket.gethostname(), "pid": os.getpid(),
             "deadline_utc": deadline.isoformat(), "jobs": queue, "attempts": previous.get("attempts", [])}
    save(runtime / "lease.json", {"host": socket.gethostname(), "pid": os.getpid(), "active": True})
    def update(phase, job=None, **extra):
        state.update(phase=phase, current=job, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
        save(runtime / "status.json", state)
        with (runtime / "events.jsonl").open("a") as events:
            events.write(json.dumps({"utc": state["updated_utc"], "phase": phase, "job": job, **extra}) + "\n")
        print(json.dumps({"phase": phase, "current": job, **extra}), flush=True)
    try:
        for job in queue:
            output = ROOT / job["output"]
            if complete(output, job["method"], job["dataset"]):
                update("skip_complete", job)
                continue
            config = json.loads((ROOT / job["config"]).read_text())
            operational = dict(config)
            if args.minimum_free_gpu_mib is not None:
                if args.minimum_free_gpu_mib < 1:
                    raise ValueError("The VRAM admission threshold must be positive")
                operational["minimum_free_gpu_mib"] = args.minimum_free_gpu_mib
            while True:
                if (runtime / "pause_new_runs").exists():
                    update("paused_for_handoff", job)
                    return 0
                if datetime.now(timezone.utc) >= deadline:
                    update("budget_exhausted", job)
                    return 0
                resources = resource_state()
                if lock_available() and ready(resources, operational):
                    break
                update("waiting_resources", job, resources=resources)
                time.sleep(30)
            for smoke in ((True, False) if not job["legacy"] else (False,)):
                target = output.parent / (output.name + "-smoke") if smoke else output
                if complete(target, job["method"], job["dataset"], 2 if smoke else 100):
                    continue
                if (runtime / "pause_new_runs").exists() or datetime.now(timezone.utc) >= deadline:
                    update("paused_before_next_run", job)
                    return 0
                resume = (target / "last.pt").is_file()
                if resume:
                    expected_config = dict(config)
                    if smoke:
                        expected_config.update(smoke_check=True, epochs=2, batch_size=4, microbatch_size=2,
                                               validation_batch_size=4, warmup_epochs=0,
                                               purpose="integration_smoke_test_not_benchmark")
                    if json.loads((target / "config.json").read_text()) != expected_config:
                        raise ValueError(f"Resume config mismatch: {target}")
                elif target.exists():
                    previous = json.loads((target / "status.json").read_text()) if (target / "status.json").exists() else {}
                    if previous.get("phase") != "blocked_resources":
                        raise ValueError(f"Inspect incomplete output without checkpoint: {target}")
                    archive = ROOT / "runs/attempts" / f"{job['dataset']}-{job['method']}-{time.time_ns()}"
                    archive.parent.mkdir(parents=True, exist_ok=True)
                    target.rename(archive)
                command = [sys.executable, "scripts/train_validation.py", "--method", job["method"],
                           "--config", job["config"], "--output", str(target.relative_to(ROOT)),
                           "--trim-host-allocator"]
                if smoke:
                    command.append("--smoke-check")
                if args.minimum_free_gpu_mib is not None:
                    command.extend(["--minimum-free-gpu-mib", str(args.minimum_free_gpu_mib)])
                if resume:
                    command.append("--resume")
                references = [ROOT / f"runs/{job['dataset']}/linear/result.json",
                              ROOT / f"runs/development/{job['dataset']}/bitfit/seed42/result.json"]
                if job["dataset"] == "oxford_flowers102":
                    references.append(ROOT / "results/checkpoint1/linear/result.json")
                for reference in references:
                    if reference.exists() and reference != target / "result.json":
                        recorded = json.loads(reference.read_text())
                        if recorded.get("phase") == "complete" and recorded.get("seed") == config["seed"]:
                            command.extend(["--reference-result", str(reference)])
                            break
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                profile = ROOT / "runs/profiles" / f"{job['dataset']}-{job['method']}-{'smoke-' if smoke else ''}{stamp}"
                profile_command = [sys.executable, "scripts/profile_training.py", "--gpu", args.gpu,
                                   "--output", str(profile), "--result", str(target / "result.json"), "--", *command]
                update("running_smoke" if smoke else "running", job, profile=str(profile.relative_to(ROOT)))
                code = subprocess.call(profile_command, cwd=ROOT, stdin=subprocess.DEVNULL)
                state["attempts"].append({"job": job, "smoke": smoke, "returncode": code,
                                          "profile": str(profile.relative_to(ROOT))})
                if code:
                    status = json.loads((target / "status.json").read_text()) if (target / "status.json").exists() else {}
                    if status.get("phase") == "blocked_resources":
                        update("blocked_resources", job, reason=status.get("reason"))
                        return 3
                    update("failed", job, returncode=code)
                    return code
                if not complete(target, job["method"], job["dataset"], 2 if smoke else 100):
                    raise ValueError("Child exited without complete, reload-verified artifacts")
            update("method_complete", job)
            subprocess.run([sys.executable, "scripts/expansion_analysis.py"], cwd=ROOT, check=True)
        update("complete")
        return 0
    finally:
        save(runtime / "lease.json", {"host": socket.gethostname(), "pid": os.getpid(), "active": False})


if __name__ == "__main__":
    while True:
        code = main()
        if code != 3:
            sys.exit(code)
        time.sleep(30)
