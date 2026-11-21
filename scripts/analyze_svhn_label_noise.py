"""Compare clean and label-noise SVHN experiments using saved epoch-100 logits."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoint2_analysis import load_run, write_json
from calibration_routing_pilot import ROUTING_SEEDS, _threshold_for_budget, softmax
from multi_pair_pilot import calibrate, evaluate_pairs, matched_random_mask, metric_row, stable_seed
from run_svhn_label_noise import METHODS, INIT, NOISE_SEED, SPLITS, corrupt, read


ROOT = Path(__file__).resolve().parents[1]
BUDGET = 0.25
PAIR = ("convpass", "ssf")
MEASURES = ("accuracy", "nll", "brier", "fallback_fraction", "forward_passes_per_image")


def corruption_checks(audit, manifest_path):
    manifest = read(manifest_path)
    training = [(row["path"], row["label"]) for row in audit["development_records"][SPLITS[0]]]
    noisy_labels, expected_changes = corrupt(training)
    noisy_split = "".join(f"{path} {label}\n" for path, label in noisy_labels).encode()
    if (manifest["seed"] != NOISE_SEED or manifest["changed_rows"] != 160
            or manifest["changes"] != expected_changes or manifest["clean_split_sha256"] != {
                name: audit["splits"][name]["list_sha256"] for name in SPLITS}
            or manifest["noisy_train_sha256"] != hashlib.sha256(noisy_split).hexdigest()):
        raise ValueError("Corruption manifest disagrees with the audited training split and fixed seed")
    return {"manifest": manifest, "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}


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
    accuracy = 100 * (logits.argmax(axis=1) == clean["targets"]).mean()
    if abs(accuracy - result["last_epoch_validation_top1_percent"]) > 1e-6:
        raise ValueError(f"Noisy accuracy disagrees with stored logits: {method}")
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
        checks = corruption_checks(audit, manifest_path)
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


def build_pairwise(root, noisy, output):
    audit = read(root / "results/dataset_audits/svhn_validation.json")
    clean = {method: load_run(root, "svhn", method, 42, audit) for method in PAIR}
    checks = corruption_checks(audit, noisy / "corruption.json")
    corrupted = {method: load_noisy(noisy / method, checks, clean[method], audit) for method in PAIR}
    first, second = PAIR
    targets, paths = (clean[first][key] for key in ("targets", "paths"))
    scores, diagnostics = [], []
    for noise, runs in ((0, clean), (20, corrupted)):
        data = {"dataset": "svhn", "subset": "validation", "runs": runs, "targets": targets,
                "ordered_paths": paths}
        caches, _, _, _, _ = calibrate(data, PAIR)
        rows, _, _ = evaluate_pairs(data, caches, [PAIR], ["averaging"], rates=(BUDGET,))
        for row in rows:
            strategy = row["strategy"]
            fallback = 0.0 if strategy in {"calibrated_first", "calibrated_second"} else row["fallback_fraction"]
            scores.append({"training_noise_percent": noise, "split_seed": row["split_seed"],
                           "strategy": strategy, "routing_seed": row["routing_seed"],
                           **{key: row[key] for key in ("accuracy", "nll", "brier")},
                           "fallback_fraction": fallback, "forward_passes_per_image": 1 + fallback})
            if strategy == "confidence_routing":
                diagnostics.append({"training_noise_percent": noise, "split_seed": row["split_seed"],
                                    **{key: row[key] for key in ("fallback_count", "errors_fixed",
                                         "correct_predictions_broken", "selected_errors_not_fixed",
                                         "first_errors_missed")}})

    output.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(scores)
    groups = ["training_noise_percent", "strategy", "split_seed"]
    by_split = scores.groupby(groups)[list(MEASURES)].mean()
    scores.to_csv(output / "scores_by_split_and_routing_seed.csv", index=False)
    by_split.groupby(level=[0, 1]).mean().to_csv(output / "strategy_summary.csv")
    pd.DataFrame(diagnostics).to_csv(output / "gate_diagnostics.csv", index=False)
    write_json({"dataset": "svhn", "validation_n": len(targets), "training_seed": 42,
                "calibration_split_seeds": sorted(scores["split_seed"].unique().tolist()),
                "corruption_seed": NOISE_SEED, "corruption_manifest_sha256": checks["sha256"],
                "first_method": first, "second_method": second, "fallback_rule": "averaging",
                "target_fallback_fraction": BUDGET, "matched_random_draws_per_split": len(ROUTING_SEEDS),
                "training_noise_percent": [0, 20], "final_test_used": False}, output / "status.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noisy", type=Path, default=ROOT / "results/label_noise/svhn/runs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pairwise", action="store_true", help="Compare ConvPass → SSF at 0%% and 20%% label noise")
    args = parser.parse_args()
    output_root = ROOT / "results/label_noise/svhn"
    if args.pairwise:
        build_pairwise(ROOT, args.noisy, args.output or output_root / "pairwise")
    else:
        build(ROOT, args.noisy, args.output or output_root / "analysis")
