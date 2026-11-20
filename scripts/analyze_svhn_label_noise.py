"""Compare routing controls on clean SVHN validation using saved epoch-100 logits."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoint2_analysis import load_run, write_json
from calibration_routing_pilot import ROUTING_SEEDS, _threshold_for_budget, softmax
from multi_pair_pilot import calibrate, matched_random_mask, metric_row, stable_seed
from run_svhn_label_noise import METHODS, INIT, NOISE_SEED, SPLITS, corrupt, read


ROOT = Path(__file__).resolve().parents[1]
BUDGET = 0.25
MEASURES = ("accuracy", "nll", "brier", "fallback_fraction", "forward_passes_per_image")


def load_noisy(path, checks, clean, audit):
    result, provenance = (read(path / f"{key}.json") for key in ("result", "provenance"))
    method = clean["result"]["method"]
    config = result["config"]
    manifest = checks["manifest"]
    expected_split = {**manifest["clean_split_sha256"], "train800.txt": manifest["noisy_train_sha256"]}
    if ((result["dataset"], result["method"], result["seed"], result["epochs"], result["phase"])
            != ("svhn", method, 42, 100, "complete") or not result["checkpoint_reload_exact"]
            or result["rows"]["test_used"] or provenance["split_sha256"] != expected_split
            or config["label_noise_manifest_sha256"] != checks["sha256"]
            or not result["frozen_backbone_unchanged"] or not all(result[k] == clean["result"][k] for k in INIT)):
        raise ValueError(f"Noisy run failed provenance checks: {method}")
    excluded = {"experiment", "purpose", "label_noise_manifest_sha256"}
    if {k: v for k, v in config.items() if k not in excluded} != {
            k: v for k, v in clean["result"]["config"].items() if k not in excluded}:
        raise ValueError(f"Noisy training recipe differs: {method}")
    with np.load(path / "validation_last.npz") as a, np.load(path / "validation_last_reloaded.npz") as b:
        logits = a["logits"].astype(np.float64)
        if (logits.shape != (len(clean["targets"]), audit["class_count"]) or not np.isfinite(logits).all()
                or not np.array_equal(a["targets"], clean["targets"])
                or not np.array_equal(a["image_paths"], clean["paths"])
                or set(a.files) != set(b.files) or any(not np.array_equal(a[k], b[k]) for k in a.files)):
            raise ValueError(f"Noisy logits or image order differ: {method}")
    return {"logits": logits, "result": result}


def compare(runs, targets, paths, noise_percent):
    data = {"dataset": "svhn", "subset": "validation", "runs": runs, "targets": targets, "ordered_paths": paths}
    caches, calibration, temperatures, _, _ = calibrate(data, METHODS)
    first = METHODS[0]
    logits = runs[first]["logits"]
    raw_confidence = softmax(logits).max(axis=1)
    scores, gates = [], []
    for seed, cache in caches.items():
        p = cache["probabilities"]
        fused = np.mean([p[method] for method in METHODS], axis=0)
        guess = [p[method].argmax(axis=1) for method in METHODS]
        majority = np.where(guess[1] == guess[2], guess[1], guess[0])
        base = {"training_noise_percent": noise_percent, "split_seed": seed}

        def record(strategy, probabilities, mask, forward_passes, **extra):
            scores.append(metric_row(probabilities, targets, mask, **base, strategy=strategy,
                                     forward_passes_per_image=forward_passes, **extra))

        zero = np.zeros(len(targets), dtype=bool)
        for method in METHODS:
            record(method, p[method], zero, 1)
        record("always_on_averaging", fused, ~zero, 3)
        scores.append({**base, "strategy": "hard_majority_vote", "accuracy": float((majority == targets).mean()),
                       "nll": np.nan, "brier": np.nan, "fallback_fraction": 1.0, "forward_passes_per_image": 3})

        gate, raw_gate = zero.copy(), zero.copy()
        for fold, evaluation in enumerate(cache["folds"]):
            fitting = np.concatenate([indices for i, indices in enumerate(cache["folds"]) if i != fold])
            threshold = _threshold_for_budget(
                softmax(logits[fitting], cache["temperatures"][first, fold]).max(axis=1), BUDGET)
            gate[evaluation] = p[first][evaluation].max(axis=1) < threshold
            raw_gate[evaluation] = raw_confidence[evaluation] < _threshold_for_budget(raw_confidence[fitting], BUDGET)

        first_correct = guess[0] == targets
        fused_correct = fused.argmax(axis=1) == targets
        fixable = ~first_correct & fused_correct
        breakable = first_correct & ~fused_correct
        for strategy, mask in (("selective", gate), ("raw_gate_calibrated_fusion", raw_gate)):
            record(strategy, np.where(mask[:, None], fused, p[first]), mask, 1 + 2 * mask.mean())
            gates.append({**base, "strategy": strategy, "n": len(mask), "selected": int(mask.sum()),
                          "first_errors": int((~first_correct).sum()),
                          "errors_selected": int((mask & ~first_correct).sum()),
                          "errors_missed": int((~mask & ~first_correct).sum()),
                          "errors_fixed": int((mask & fixable).sum()),
                          "correct_broken": int((mask & breakable).sum()),
                          "selection_differs_from_raw": int((mask != raw_gate).sum())})
        for random_seed in ROUTING_SEEDS:
            mask = zero.copy()
            for fold, evaluation in enumerate(cache["folds"]):
                mask[evaluation] = matched_random_mask(
                    gate[evaluation], stable_seed("svhn_label_noise", seed, fold, random_seed))
                assert mask[evaluation].sum() == gate[evaluation].sum()
            record("matched_random", np.where(mask[:, None], fused, p[first]), mask, 1 + 2 * mask.mean(),
                   random_seed=random_seed)
    for row in calibration + temperatures:
        row["training_noise_percent"] = noise_percent
    return scores, gates, calibration, temperatures


def build(root, noisy, output):
    audit = read(root / "results/dataset_audits/svhn_validation.json")
    clean = {m: load_run(root, "svhn", m, 42, audit) for m in METHODS}
    targets, paths = (clean[METHODS[0]][k] for k in ("targets", "paths"))
    scores, gates, calibration, temperatures = compare(clean, targets, paths, 0)
    present = [m for m in METHODS if (noisy / m / "result.json").is_file()]
    noise_summary = []
    manifest_path = noisy / "corruption.json"
    if present or manifest_path.is_file():
        manifest = read(manifest_path)
        training = [(row["path"], row["label"]) for row in audit["development_records"][SPLITS[0]]]
        noisy_labels, expected_changes = corrupt(training)
        noisy_split = "".join(f"{path} {label}\n" for path, label in noisy_labels).encode()
        if (manifest["seed"] != NOISE_SEED or manifest["changed_rows"] != 160
                or manifest["changes"] != expected_changes or manifest["clean_split_sha256"] != {
                    name: audit["splits"][name]["list_sha256"] for name in SPLITS}
                or manifest["noisy_train_sha256"] != hashlib.sha256(noisy_split).hexdigest()):
            raise ValueError("Corruption manifest disagrees with the audited training split and fixed seed")
        checks = {"manifest": manifest, "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    if present:
        noise_runs = {m: load_noisy(noisy / m, checks, clean[m], audit) for m in present}
        noise_summary = [{"method": m, "accuracy": noise_runs[m]["result"]["last_epoch_validation_top1_percent"] / 100}
                         for m in present]
        if len(present) == len(METHODS):
            for base, extra in zip((scores, gates, calibration, temperatures), compare(noise_runs, targets, paths, 20)):
                base.extend(extra)
        else:
            data = {"dataset": "svhn", "subset": "validation", "runs": noise_runs,
                    "targets": targets, "ordered_paths": paths}
            _, single_calibration, single_temperatures, _, _ = calibrate(data, present)
            calibration.extend({**row, "training_noise_percent": 20} for row in single_calibration)
            temperatures.extend({**row, "training_noise_percent": 20} for row in single_temperatures)
    output.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(scores)
    scores.to_csv(output / "scores_by_split_and_random_seed.csv", index=False)
    summary = scores.groupby(["training_noise_percent", "strategy", "split_seed"])[list(MEASURES)].mean()
    summary.groupby(level=[0, 1]).mean().to_csv(output / "strategy_summary.csv")
    pd.DataFrame(gates).to_csv(output / "gate_diagnostics.csv", index=False)
    pd.DataFrame(calibration).to_csv(output / "calibration.csv", index=False)
    pd.DataFrame(temperatures).to_csv(output / "temperatures.csv", index=False)
    write_json({"validation_n": len(targets), "training_seed": 42, "target_fallback_fraction": BUDGET,
                "methods": list(METHODS), "verified_noisy_runs": noise_summary,
                "noisy_ensemble_evaluated": len(present) == len(METHODS), "final_test_used": False},
               output / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy", type=Path, default=ROOT / "results/label_noise/svhn/runs")
    parser.add_argument("--output", type=Path, default=ROOT / "results/label_noise/svhn/analysis")
    args = parser.parse_args()
    build(ROOT, args.noisy, args.output)
