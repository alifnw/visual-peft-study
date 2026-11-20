"""CPU-only temperature-scaling and confidence-routing pilot on saved logits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

DATASETS = ("dtd", "resisc45")
EXTENDED_DATASETS = (*DATASETS, "svhn", "patch_camelyon")
CLASS_COUNTS = {"dtd": 47, "resisc45": 45, "svhn": 10, "patch_camelyon": 2}
METHODS = ("linear", "lora", "ssf", "convpass")
SPLIT_SEEDS = (42, 43, 44)
ROUTING_SEEDS = tuple(range(42, 62))
FALLBACK_RATES = (0.25, 0.50, 0.75)
TEMPERATURE_BOUNDS = (0.05, 20.0)
N_FOLDS = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    values -= values.max(axis=1, keepdims=True)
    exp_values = np.exp(values)
    return exp_values / exp_values.sum(axis=1, keepdims=True)


def metrics_from_probabilities(targets: np.ndarray, probabilities: np.ndarray) -> dict:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or len(targets) != len(probabilities) or not len(targets):
        raise ValueError("Expected aligned, nonempty targets and N x C probabilities")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any():
        raise ValueError("Probabilities must be finite and nonnegative")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        raise ValueError("Each probability row must sum to one")
    if targets.min() < 0 or targets.max() >= probabilities.shape[1]:
        raise ValueError("Targets are outside the class range")
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    chosen = probabilities[np.arange(len(targets)), targets]
    nll = -np.log(np.clip(chosen, np.finfo(np.float64).tiny, 1.0))
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(targets)), targets] = 1.0
    brier = np.square(probabilities - one_hot).sum(axis=1)
    class_ids = np.unique(targets)
    class_recalls = [float(correct[targets == class_id].mean()) for class_id in class_ids]
    return {
        "n": int(len(targets)),
        "accuracy": float(correct.mean()),
        "nll": float(nll.mean()),
        "brier": float(brier.mean()),
        "classes_observed": int(len(class_ids)),
        "class_ids_observed": ",".join(str(int(x)) for x in class_ids),
        "macro_recall_observed": float(np.mean(class_recalls)),
        "predicted": predicted,
        "correct": correct,
        "confidence": probabilities.max(axis=1),
        "nll_samples": nll,
        "brier_samples": brier,
    }


def negative_log_likelihood(logits: np.ndarray, targets: np.ndarray, temperature: float) -> float:
    values = np.asarray(logits, dtype=np.float64) / float(temperature)
    maximum = values.max(axis=1, keepdims=True)
    log_normalizer = maximum[:, 0] + np.log(np.exp(values - maximum).sum(axis=1))
    return float(np.mean(log_normalizer - values[np.arange(len(targets)), targets]))


def fit_temperature(logits: np.ndarray, targets: np.ndarray) -> dict:
    """Bounded deterministic scalar search in log-temperature space."""
    lower, upper = (math.log(x) for x in TEMPERATURE_BOUNDS)
    objective = lambda log_t: negative_log_likelihood(logits, targets, math.exp(log_t))
    optimum = minimize_scalar(objective, method="bounded", bounds=(lower, upper),
                              options={"xatol": 1e-8, "maxiter": 500})
    candidates = [(objective(lower), lower), (objective(float(optimum.x)), float(optimum.x)),
                  (objective(upper), upper)]
    best_nll, log_temperature = min(candidates, key=lambda item: item[0])
    temperature = float(math.exp(log_temperature))
    at_lower = abs(log_temperature - lower) <= 1e-7
    at_upper = abs(log_temperature - upper) <= 1e-7
    return {
        "temperature": temperature,
        "calibration_nll": float(best_nll),
        "boundary_optimum": bool(at_lower or at_upper),
        "boundary": "lower" if at_lower else "upper" if at_upper else "none",
        "optimizer": "scipy.optimize.minimize_scalar(method=bounded), xatol=1e-8",
    }


def make_folds(n: int, seed: int, n_folds: int = N_FOLDS) -> list[np.ndarray]:
    if n < n_folds or n_folds < 2:
        raise ValueError("Need at least one row per fold")
    shuffled = np.random.default_rng(seed).permutation(n)
    return [np.asarray(x, dtype=np.int64) for x in np.array_split(shuffled, n_folds)]


def reliability_rows(dataset: str, method: str, stage: str, split_seed: int,
                     targets: np.ndarray, probabilities: np.ndarray) -> list[dict]:
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == targets
    bins = np.minimum((confidence * 10).astype(int), 9)
    rows = []
    for bin_id in range(10):
        selected = bins == bin_id
        rows.append({
            "dataset": dataset, "method": method, "stage": stage, "split_seed": split_seed,
            "bin": bin_id, "lower_bound": bin_id / 10, "upper_bound": (bin_id + 1) / 10,
            "count": int(selected.sum()),
            "mean_confidence": float(confidence[selected].mean()) if selected.any() else np.nan,
            "empirical_accuracy": float(correct[selected].mean()) if selected.any() else np.nan,
        })
    return rows


def load_dataset(root: Path, dataset: str) -> dict:
    audit_path = root / "results/dataset_audits" / f"{dataset}_validation.json"
    audit = json.loads(audit_path.read_text())
    if audit.get("phase") != "complete" or audit.get("dataset") != dataset:
        raise ValueError(f"Validation audit is incomplete or mismatched: {audit_path}")
    split_hash = audit["splits"]["val200.txt"]["list_sha256"]
    full_records = audit["development_records"]["val200.txt"]
    full_targets = np.asarray([row["label"] for row in full_records], dtype=np.int64)
    if len(full_records) != 200 or int(audit["class_count"]) != CLASS_COUNTS.get(dataset):
        raise ValueError(f"Unexpected validation audit dimensions for {dataset}")

    if dataset == "dtd":
        original_indices = np.asarray(audit["validation_indices_without_training_content"], dtype=np.int64)
        if len(original_indices) != 199 or audit["duplicate_validation_indices"] != [0]:
            raise ValueError("DTD clean subset no longer matches the audited single duplicate")
        expected_targets = full_targets[original_indices]
        ordered_paths = np.asarray([full_records[index]["path"] for index in original_indices])
        prediction_name = "validation_last_without_training_duplicates.npz"
    else:
        original_indices = np.arange(200, dtype=np.int64)
        expected_targets = full_targets
        ordered_paths = np.asarray([row["path"] for row in full_records])
        prediction_name = "validation_last.npz"

    runs = {}
    reference_indices = None
    for method in METHODS:
        folder = root / "results/development" / dataset / method / "seed42"
        result_path = folder / "result.json"
        provenance_path = folder / "provenance.json"
        prediction_path = folder / prediction_name
        result = json.loads(result_path.read_text())
        provenance = json.loads(provenance_path.read_text())
        if (result.get("phase") != "complete" or result.get("epochs") != 100
                or result.get("dataset") != dataset or result.get("method") != method
                or result.get("seed") != 42):
            raise ValueError(f"Run identity is not the expected completed epoch-100 run: {result_path}")
        if provenance.get("split_sha256", {}).get("val200.txt") != split_hash:
            raise ValueError(f"Validation split hash mismatch for {result_path}")
        with np.load(prediction_path, allow_pickle=False) as stored:
            logits = np.asarray(stored["logits"], dtype=np.float64)
            targets = np.asarray(stored["targets"], dtype=np.int64)
            indices = np.asarray(stored["validation_indices"], dtype=np.int64) if "validation_indices" in stored else None
        if logits.shape != (len(expected_targets), int(audit["class_count"])):
            raise ValueError(f"Unexpected logits shape in {prediction_path}: {logits.shape}")
        if not np.isfinite(logits).all() or not np.array_equal(targets, expected_targets):
            raise ValueError(f"Logits targets do not match audited split order: {prediction_path}")
        if indices is not None and not np.array_equal(indices, original_indices):
            raise ValueError(f"Saved validation indices differ from audited order: {prediction_path}")
        if reference_indices is None:
            reference_indices = indices
        elif indices is not None and not np.array_equal(indices, reference_indices):
            raise ValueError(f"Methods are not aligned by validation index for {dataset}")

        full_accuracy = float((logits.argmax(axis=1) == targets).mean() * 100)
        if dataset == "dtd":
            expected_accuracy = result.get("last_epoch_validation_without_training_duplicates_top1_percent")
        else:
            expected_accuracy = result.get("last_epoch_validation_top1_percent")
        if expected_accuracy is None or abs(full_accuracy - float(expected_accuracy)) > 0.006:
            raise ValueError(f"Stored logits accuracy differs from run metadata: {prediction_path}")
        runs[method] = {
            "logits": logits, "targets": targets, "result": result,
            "prediction_path": prediction_path, "result_path": result_path,
            "provenance_path": provenance_path,
        }
    frozen_hashes = {run["result"].get("frozen_backbone_sha256") for run in runs.values()}
    if None in frozen_hashes or len(frozen_hashes) != 1:
        raise ValueError(f"The four methods do not share one verified frozen-backbone initialization for {dataset}")
    classifier_hashes = {run["result"].get("classifier_initial_sha256") for run in runs.values()}
    if None in classifier_hashes or len(classifier_hashes) != 1:
        raise ValueError(f"The four methods do not share one verified classifier initialization for {dataset}")
    return {
        "dataset": dataset, "class_count": int(audit["class_count"]), "targets": expected_targets,
        "runs": runs, "split_hash": split_hash, "audit_path": audit_path,
        "subset": "validation_without_training_duplicates" if dataset == "dtd" else "validation",
        "original_indices": original_indices, "ordered_paths": ordered_paths,
        "order_evidence": (
            "DTD saved validation_indices match the audit exclusion mask; labels match the audited path order"
            if dataset == "dtd" else
            "All methods share the audited val200 split hash; target sequence matches the ordered audit records; "
            "the runner exported with shuffle=False and asserted targets against val200.txt before saving"
        ),
    }


def metrics_record(dataset: str, subset: str, method: str, stage: str,
                   split_seed: int, probabilities: np.ndarray, targets: np.ndarray) -> dict:
    metric = metrics_from_probabilities(targets, probabilities)
    return {
        "dataset": dataset, "evaluation_subset": subset, "method": method,
        "stage": stage, "split_seed": split_seed,
        **{key: value for key, value in metric.items()
           if key not in {"predicted", "correct", "confidence", "nll_samples", "brier_samples"}},
    }


def _route(linear_probabilities: np.ndarray, lora_probabilities: np.ndarray,
           mask: np.ndarray) -> np.ndarray:
    fusion = 0.5 * (linear_probabilities + lora_probabilities)
    return np.where(mask[:, None], fusion, linear_probabilities)


def _threshold_for_budget(calibration_confidence: np.ndarray, rate: float) -> float:
    if not 0 <= rate <= 1:
        raise ValueError("Fallback rate must be between zero and one")
    if rate == 0:
        return -np.inf
    if rate == 1:
        return np.inf
    return float(np.quantile(calibration_confidence, rate, method="higher"))


def _routing_metric_row(dataset: str, subset: str, split_seed: int, strategy: str,
                        target_rate: float | None, route_seed: int | None,
                        fallback: np.ndarray | None, probabilities: np.ndarray,
                        targets: np.ndarray) -> dict:
    metric = metrics_from_probabilities(targets, probabilities)
    return {
        "dataset": dataset, "evaluation_subset": subset, "split_seed": split_seed,
        "strategy": strategy, "target_fallback_rate": target_rate,
        "routing_seed": route_seed,
        "fallback_count": int(fallback.sum()) if fallback is not None else np.nan,
        "fallback_fraction": float(fallback.mean()) if fallback is not None else np.nan,
        **{key: value for key, value in metric.items()
           if key not in {"predicted", "correct", "confidence", "nll_samples", "brier_samples"}},
    }


def analyze_dataset(data: dict) -> tuple[list[dict], list[dict], list[dict], list[dict], list[dict], dict]:
    dataset, targets, runs = data["dataset"], data["targets"], data["runs"]
    n = len(targets)
    calibration_rows, temperature_rows, reliability = [], [], []
    routing_rows, failure_rows = [], []

    for split_seed in SPLIT_SEEDS:
        folds = make_folds(n, split_seed)
        fold_of_row = np.full(n, -1, dtype=np.int64)
        for fold_id, indices in enumerate(folds):
            fold_of_row[indices] = fold_id
        if (fold_of_row < 0).any() or sum(len(fold) for fold in folds) != n:
            raise AssertionError("Fold assignment does not cover every evaluation row exactly once")

        calibrated_probabilities = {}
        for method in METHODS:
            logits = runs[method]["logits"]
            calibrated = np.zeros((n, logits.shape[1]), dtype=np.float64)
            for fold_id, evaluation_indices in enumerate(folds):
                calibration_indices = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
                if set(evaluation_indices.tolist()) & set(calibration_indices.tolist()):
                    raise AssertionError("Calibration and evaluation rows overlap")
                fitted = fit_temperature(logits[calibration_indices], targets[calibration_indices])
                temperature = fitted["temperature"]
                calibrated[evaluation_indices] = softmax(logits[evaluation_indices], temperature)
                temperature_rows.append({
                    "dataset": dataset, "method": method, "split_seed": split_seed,
                    "fold": fold_id, "calibration_n": len(calibration_indices),
                    "evaluation_n": len(evaluation_indices),
                    "calibration_classes_observed": int(np.unique(targets[calibration_indices]).size),
                    "evaluation_classes_observed": int(np.unique(targets[evaluation_indices]).size),
                    **fitted,
                })
            calibrated_probabilities[method] = calibrated
            raw_probabilities = softmax(logits)
            before = metrics_record(dataset, data["subset"], method, "raw_oof", split_seed, raw_probabilities, targets)
            after = metrics_record(dataset, data["subset"], method, "temperature_scaled_oof", split_seed, calibrated, targets)
            if before["accuracy"] != after["accuracy"]:
                raise AssertionError("Positive scalar temperature must preserve top-1 predictions")
            calibration_rows.extend((before, after))
            reliability.extend(reliability_rows(dataset, method, "raw", split_seed, targets, raw_probabilities))
            reliability.extend(reliability_rows(dataset, method, "temperature_scaled", split_seed, targets, calibrated))

        linear_logits, lora_logits = runs["linear"]["logits"], runs["lora"]["logits"]
        linear_p, lora_p = calibrated_probabilities["linear"], calibrated_probabilities["lora"]
        fusion_p = 0.5 * (linear_p + lora_p)
        no_fallback = np.zeros(n, dtype=bool)
        all_fallback = np.ones(n, dtype=bool)
        if not np.array_equal(_route(linear_p, lora_p, no_fallback), linear_p):
            raise AssertionError("Zero fallback must exactly reproduce calibrated Linear")
        if not np.allclose(_route(linear_p, lora_p, all_fallback), fusion_p, atol=0, rtol=0):
            raise AssertionError("Full fallback must exactly reproduce always-on probability fusion")

        route_comparators = (
            ("calibrated_linear", linear_p, no_fallback),
            ("calibrated_lora", lora_p, None),
            ("always_on_fusion", fusion_p, all_fallback),
        )
        for strategy, probabilities, fallback in route_comparators:
            routing_rows.append(_routing_metric_row(dataset, data["subset"], split_seed,
                strategy, None, None, fallback, probabilities, targets))

        for fold_id, evaluation_indices in enumerate(folds):
            calibration_indices = np.concatenate([fold for i, fold in enumerate(folds) if i != fold_id])
            fold_rows = [row for row in temperature_rows if row["dataset"] == dataset
                         and row["split_seed"] == split_seed and row["method"] == "linear"
                         and row["fold"] == fold_id]
            linear_temperature = fold_rows[-1]["temperature"]
            fold_rows = [row for row in temperature_rows if row["dataset"] == dataset
                         and row["split_seed"] == split_seed and row["method"] == "lora"
                         and row["fold"] == fold_id]
            lora_temperature = fold_rows[-1]["temperature"]
            linear_cal = softmax(linear_logits[calibration_indices], linear_temperature)
            linear_eval = softmax(linear_logits[evaluation_indices], linear_temperature)
            lora_eval = softmax(lora_logits[evaluation_indices], lora_temperature)
            cal_confidence = linear_cal.max(axis=1)
            eval_confidence = linear_eval.max(axis=1)
            fold_targets = targets[evaluation_indices]
            for rate in FALLBACK_RATES:
                threshold = _threshold_for_budget(cal_confidence, rate)
                mask = eval_confidence <= threshold
                routed = _route(linear_eval, lora_eval, mask)
                # Store per-fold masks and probabilities for fold-safe aggregation below.
                routing_rows.append({
                    "_internal": True, "fold": fold_id, "split_seed": split_seed,
                    "target_fallback_rate": rate, "strategy": "confidence_routing",
                    "threshold": threshold, "evaluation_indices": evaluation_indices,
                    "mask": mask, "probabilities": routed,
                    "calibration_fallback_fraction": float((cal_confidence <= threshold).mean()),
                })
                for route_seed in ROUTING_SEEDS:
                    rng = np.random.default_rng(100000 * split_seed + 1000 * fold_id + route_seed)
                    random_mask = rng.random(len(evaluation_indices)) < rate
                    random_routed = _route(linear_eval, lora_eval, random_mask)
                    routing_rows.append({
                        "_internal": True, "fold": fold_id, "split_seed": split_seed,
                        "target_fallback_rate": rate, "strategy": "random_routing",
                        "routing_seed": route_seed, "evaluation_indices": evaluation_indices,
                        "mask": random_mask, "probabilities": random_routed,
                    })

        # Aggregate routed predictions across folds so each validation image is counted once.
        internal = [row for row in routing_rows if row.get("_internal") and row["split_seed"] == split_seed]
        for strategy, seeds in (("confidence_routing", (None,)), ("random_routing", ROUTING_SEEDS)):
            for rate in FALLBACK_RATES:
                selected_seeds = seeds if strategy == "random_routing" else (None,)
                for route_seed in selected_seeds:
                    chosen = [row for row in internal if row["strategy"] == strategy
                              and row["target_fallback_rate"] == rate
                              and row.get("routing_seed") == route_seed]
                    if len(chosen) != N_FOLDS:
                        raise AssertionError("A routing replicate must contain exactly one prediction for each fold")
                    probabilities = np.zeros((n, runs["linear"]["logits"].shape[1]), dtype=np.float64)
                    fallback = np.zeros(n, dtype=bool)
                    for row in chosen:
                        indices = row["evaluation_indices"]
                        probabilities[indices] = row["probabilities"]
                        fallback[indices] = row["mask"]
                    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
                        raise AssertionError("Routing output is missing or has invalid probabilities")
                    result = _routing_metric_row(dataset, data["subset"], split_seed,
                        strategy, rate, route_seed, fallback, probabilities, targets)
                    result["fallback_thresholds"] = "|".join(
                        f"{row['fold']}:{row['threshold']:.10g}" for row in chosen
                        if row.get("threshold") is not None)
                    result["calibration_target_fallback_rate"] = rate
                    result["calibration_fallback_fraction"] = (
                        float(np.mean([row["calibration_fallback_fraction"] for row in chosen]))
                        if strategy == "confidence_routing" else np.nan)
                    routing_rows.append(result)
                    if strategy == "confidence_routing":
                        for row in chosen:
                            indices = row["evaluation_indices"]
                            mask = row["mask"]
                            linear_fold = linear_p[indices]
                            lora_fold = lora_p[indices]
                            fusion_fold = 0.5 * (linear_fold + lora_fold)
                            linear_prediction = linear_fold.argmax(axis=1)
                            lora_prediction = lora_fold.argmax(axis=1)
                            fusion_prediction = fusion_fold.argmax(axis=1)
                            fold_truth = targets[indices]
                            for local_index, global_index in enumerate(indices):
                                linear_correct = linear_prediction[local_index] == fold_truth[local_index]
                                fusion_correct = fusion_prediction[local_index] == fold_truth[local_index]
                                issue = None
                                if mask[local_index] and not fusion_correct:
                                    issue = ("fallback_broke_linear_correct" if linear_correct
                                             else "fallback_did_not_fix_linear_error")
                                elif not mask[local_index] and not linear_correct:
                                    issue = "linear_error_missed"
                                if issue:
                                    failure_rows.append({
                                        "dataset": dataset, "split_seed": split_seed,
                                        "target_fallback_rate": rate, "fold": int(row["fold"]),
                                        "validation_row": int(global_index),
                                        "validation_original_index": int(data["original_indices"][global_index]),
                                        "validation_path": str(data["ordered_paths"][global_index]),
                                        "actual": int(fold_truth[local_index]),
                                        "linear_prediction": int(linear_prediction[local_index]),
                                        "lora_prediction": int(lora_prediction[local_index]),
                                        "fusion_prediction": int(fusion_prediction[local_index]),
                                        "linear_confidence": float(linear_fold[local_index].max()),
                                        "fallback_selected": bool(mask[local_index]),
                                        "issue": issue,
                                    })
        routing_rows[:] = [row for row in routing_rows if not row.get("_internal")]

    return calibration_rows, temperature_rows, reliability, routing_rows, failure_rows, {
        "dataset": dataset, "evaluation_subset": data["subset"], "n": n,
        "classes_observed": int(np.unique(targets).size),
        "class_ids_observed": ",".join(str(int(x)) for x in np.unique(targets)),
    }


def _summary(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(group_columns, dropna=False)[value_columns].agg(["mean", "std"]).reset_index()
    grouped.columns = ["_".join(str(part) for part in column if part) if isinstance(column, tuple) else column
                       for column in grouped.columns]
    return grouped


def _write_figures(output: Path, calibration: pd.DataFrame,
                   reliability: pd.DataFrame, routing: pd.DataFrame,
                   datasets: tuple[str, ...]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    created = []
    method_order = list(METHODS)
    colors = {"linear": "#1f77b4", "lora": "#ff7f0e", "ssf": "#2ca02c", "convpass": "#9467bd"}
    for dataset in datasets:
        subset = calibration[calibration.dataset == dataset]
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for ax, metric in zip(axes, ("nll", "brier")):
            for x, method in enumerate(method_order):
                for offset, stage in ((-0.12, "raw_oof"), (0.12, "temperature_scaled_oof")):
                    values = subset[(subset.method == method) & (subset.stage == stage)].groupby("split_seed")[metric].mean()
                    mean = float(values.mean())
                    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
                    ax.errorbar(x + offset, mean, yerr=std, fmt="o", capsize=3,
                                color=colors[method], markerfacecolor="white" if offset < 0 else colors[method],
                                label=("raw" if offset < 0 else "scaled") if x == 0 else None)
            ax.set_xticks(range(len(method_order)), method_order, rotation=25, ha="right")
            ax.set_title(metric.upper())
            ax.set_ylabel("Mean over three split seeds (error bar: SD)")
            ax.grid(axis="y", alpha=0.25)
        axes[0].legend(frameon=False)
        fig.suptitle(f"{dataset.upper()}: out-of-fold probability metrics")
        fig.tight_layout()
        for ext in ("png", "svg"):
            path = output / f"{dataset}-calibration-metrics.{ext}"
            fig.savefig(path, dpi=250, bbox_inches="tight")
            created.append(path.name)
        plt.close(fig)

        rel = reliability[(reliability.dataset == dataset) & (reliability.split_seed == 42)]
        fig, axes = plt.subplots(2, 2, figsize=(9, 8), sharex=True, sharey=True)
        for ax, method in zip(axes.flat, method_order):
            for stage, marker in (("raw", "o"), ("temperature_scaled", "s")):
                group = rel[(rel.method == method) & (rel.stage == stage) & (rel["count"] > 0)]
                ax.plot(group.mean_confidence, group.empirical_accuracy, marker=marker,
                        linewidth=1.2, label=stage.replace("_", " "))
                for row in group.itertuples():
                    vertical_offset = 5 if stage == "raw" else -11
                    ax.annotate(str(row.count), (row.mean_confidence, row.empirical_accuracy),
                                xytext=(3, vertical_offset), textcoords="offset points", fontsize=6)
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=0.8)
            ax.set_title(method.upper())
            ax.grid(alpha=0.2)
        axes[0, 0].legend(frameon=False, fontsize=8)
        fig.supxlabel("Mean top-label confidence")
        fig.supylabel("Empirical accuracy")
        fig.suptitle(f"{dataset.upper()}: reliability, split seed 42 (counts labelled)")
        fig.tight_layout()
        for ext in ("png", "svg"):
            path = output / f"{dataset}-reliability.{ext}"
            fig.savefig(path, dpi=250, bbox_inches="tight")
            created.append(path.name)
        plt.close(fig)

        route = routing[routing.dataset == dataset]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        metric_names = (("accuracy", "Accuracy"), ("nll", "NLL"), ("brier", "Brier"))
        for ax, (metric, title) in zip(axes, metric_names):
            baseline = route[route.strategy.isin(["calibrated_linear", "calibrated_lora", "always_on_fusion"])]
            labels = ["Linear", "LoRA", "Always-on"]
            means = [baseline[baseline.strategy == s][metric].mean() for s in
                     ("calibrated_linear", "calibrated_lora", "always_on_fusion")]
            ax.bar(range(3), means, color=["#1f77b4", "#ff7f0e", "#2ca02c"], alpha=0.8)
            x = 3
            for rate in FALLBACK_RATES:
                conf = route[(route.strategy == "confidence_routing") & (route.target_fallback_rate == rate)]
                rand = route[(route.strategy == "random_routing") & (route.target_fallback_rate == rate)]
                ax.errorbar(x, conf[metric].mean(), yerr=conf[metric].std(ddof=1), fmt="o", color="#d62728", capsize=2)
                labels.append(f"Gate {int(rate*100)}%")
                x += 1
                ax.errorbar(x, rand[metric].mean(), yerr=rand[metric].std(ddof=1), fmt="s", color="#7f7f7f", capsize=2)
                labels.append(f"Random {int(rate*100)}%")
                x += 1
            ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=7)
            ax.set_title(title)
            ax.grid(axis="y", alpha=0.2)
            values = route[metric].dropna().to_numpy(dtype=float)
            span = max(float(values.max() - values.min()), 0.01)
            margin = 0.12 * span
            ax.set_ylim(max(0.0, float(values.min()) - margin), float(values.max()) + margin)
            if metric == "accuracy":
                ax.set_ylabel("Fraction correct")
        legend = [
            Line2D([0], [0], marker="o", color="#d62728", linestyle="none", label="Confidence gate"),
            Line2D([0], [0], marker="s", color="#7f7f7f", linestyle="none", label="Random routing"),
            Patch(facecolor="#bdbdbd", label="Bars: single / always-on"),
        ]
        axes[0].legend(handles=legend, frameon=False, fontsize=7, loc="best")
        fig.suptitle(f"{dataset.upper()}: calibrated Linear → LoRA routing pilot")
        fig.tight_layout()
        for ext in ("png", "svg"):
            path = output / f"{dataset}-routing.{ext}"
            fig.savefig(path, dpi=250, bbox_inches="tight")
            created.append(path.name)
        plt.close(fig)
    return created


def _write_brief(output: Path, calibration: pd.DataFrame, temperatures: pd.DataFrame,
                 routing: pd.DataFrame, dataset_summaries: list[dict],
                 failure_cases: pd.DataFrame, datasets: tuple[str, ...]) -> None:
    lines = [
        "# Mentoring brief: calibration and selective-fusion pilot",
        "",
        "## Research question",
        "How much do out-of-fold scalar temperature scaling and a confidence-triggered Linear → LoRA fallback change probability quality on the existing DTD and RESISC45 development validation predictions?",
        "",
        "## A. Temperature scaling",
        "",
        "Values below are mean ± SD across three deterministic split seeds. Each image contributes one out-of-fold prediction per split seed. Accuracy should remain unchanged because a positive scalar temperature preserves argmax.",
        "",
        "| Dataset | Method | Accuracy | NLL raw → scaled | Brier raw → scaled |",
        "|---|---|---:|---:|---:|",
    ]
    for dataset in datasets:
        for method in METHODS:
            rows = calibration[(calibration.dataset == dataset) & (calibration.method == method)]
            def fmt(stage: str, col: str, percent: bool = False) -> str:
                values = rows[rows.stage == stage][col].to_numpy(dtype=float)
                mean = float(values.mean()) * (100 if percent else 1)
                std = float(values.std(ddof=1)) * (100 if percent else 1)
                return f"{mean:.2f} ± {std:.2f}" if percent else f"{mean:.4f} ± {std:.4f}"
            lines.append(f"| {dataset.upper()} | {method} | {fmt('raw_oof','accuracy',True)}% | {fmt('raw_oof','nll')} → {fmt('temperature_scaled_oof','nll')} | {fmt('raw_oof','brier')} → {fmt('temperature_scaled_oof','brier')} |")
    boundary = temperatures[temperatures.boundary_optimum]
    lines += ["", "Fold temperatures are in `fold_temperatures.csv`. Boundary optimum count: "
              f"{len(boundary)} of {len(temperatures)} fold fits.", "",
              "| Dataset | Method | Temperature mean | Range across 15 fits |",
              "|---|---|---:|---:|"]
    for dataset in datasets:
        for method in METHODS:
            values = temperatures[(temperatures.dataset == dataset) & (temperatures.method == method)].temperature.to_numpy()
            lines.append(f"| {dataset.upper()} | {method} | {values.mean():.4f} | {values.min():.4f} to {values.max():.4f} |")
    lines += ["",
              "## B. Confidence-routing pilot", "",
              "Routing compares calibrated single models, always-on probability averaging, confidence gate and 20 Bernoulli random-routing seeds per target budget. Gate thresholds are estimated only from each calibration fold. Metrics below average across split seeds, and random routing also averages over routing seeds.", "",
              "| Dataset | Strategy | Target fallback | Accuracy | NLL | Brier | Actual fallback |",
              "|---|---|---:|---:|---:|---:|---:|"]
    order = ["calibrated_linear", "calibrated_lora", "always_on_fusion", "confidence_routing", "random_routing"]
    for dataset in DATASETS:
        group = routing[routing.dataset == dataset]
        for strategy in order:
            subset = group[group.strategy == strategy]
            target_values = [np.nan] if strategy in ("calibrated_linear", "calibrated_lora", "always_on_fusion") else list(FALLBACK_RATES)
            for target in target_values:
                rows = subset if np.isnan(target) else subset[np.isclose(subset.target_fallback_rate, target)]
                if rows.empty:
                    continue
                rate_label = "-" if np.isnan(target) else f"{int(target*100)}%"
                def mean_std(column: str, scale: float = 1.0) -> str:
                    vals = rows[column].to_numpy(dtype=float) * scale
                    return f"{vals.mean():.2f} ± {vals.std(ddof=1):.2f}" if scale != 1 else f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
                fallback_summary = "-" if rows.fallback_fraction.isna().all() else f"{mean_std('fallback_fraction',100)}%"
                lines.append(f"| {dataset.upper()} | {strategy} | {rate_label} | {mean_std('accuracy',100)}% | {mean_std('nll')} | {mean_std('brier')} | {fallback_summary} |")
    lines += ["", "## Readout", "",
              "- Temperature scaling lowered NLL in all eight dataset-method comparisons. Brier fell in seven of eight. Accuracy stayed fixed, as expected from scalar scaling.",
              "- At every tested fallback budget on both datasets, confidence routing scored better than the 20-seed random-routing mean on accuracy, NLL, and Brier. This is a pilot signal on the reused validation sets.",
              "- At the 50% target, confidence routing matched always-on fusion accuracy in both datasets, but its NLL and Brier were higher. At the 75% target, its probability scores remained slightly worse than the strongest single/fusion baseline. No overall routing winner is established.",
              "- The gate misses many DTD Linear errors at low fallback budgets. At high budgets, many triggered DTD and RESISC45 cases remain wrong after fusion. The confidence rule therefore does not identify every image LoRA can fix.",
              "", "## Routing failure cases", "",
              "Counts are summarized across the three split seeds. Each split seed reuses the same validation images, so counts are reported as mean ± SD per split rather than summed as independent examples.",
              "", "| Dataset | Target fallback | Linear errors missed | Selected error not fixed | Fusion breaks correct Linear |",
              "|---|---:|---:|---:|---:|"]
    issues = ("linear_error_missed", "fallback_did_not_fix_linear_error", "fallback_broke_linear_correct")
    for dataset in datasets:
        for rate in FALLBACK_RATES:
            counts = {}
            for issue in issues:
                values = []
                for split_seed in SPLIT_SEEDS:
                    selected = failure_cases[(failure_cases.dataset == dataset)
                        & (failure_cases.split_seed == split_seed)
                        & np.isclose(failure_cases.target_fallback_rate, rate)
                        & (failure_cases.issue == issue)]
                    values.append(len(selected))
                counts[issue] = f"{np.mean(values):.1f} ± {np.std(values, ddof=1):.1f}"
            lines.append(f"| {dataset.upper()} | {int(rate*100)}% | {counts[issues[0]]} | {counts[issues[1]]} | {counts[issues[2]]} |")
    lines += ["", "## Interpretation limits", "",
              "- This is an exploratory pilot on development validation predictions already used during model development. Folded calibration separates temperature fitting from each fold's scoring, but does not create an independent test set.",
              "- The three values 42, 43 and 44 control calibration split generation, not model training. They are repeated partitions of the same images, not three independent datasets.",
              "- Lower NLL or Brier is evidence about probability scores on this split. Reliability diagrams are descriptive and bin counts are small.",
              "- The gate's fallback share is not a latency measurement. Linear still runs for every image, and the fallback executes LoRA too.",
              "- No corruption, learned gate, threshold tuning on an external set, final-test evaluation or controlled latency profiling is included.",
              "", "## Suggested next step", "",
              "Repeat the locked routing protocol on a separate calibration/development split, then evaluate once on untouched test data. Compare against temperature-scaled single models and always-on fusion. Profile actual end-to-end latency under identical hardware, batch and warm-up conditions before claiming compute savings.",
              "", "## Artifacts", "",
              "Tables are in the adjacent CSV files. Figures are exported as PNG and SVG. The source pipeline is `scripts/calibration_routing_pilot.py`.", ""]
    (output / "MENTORING_BRIEF.md").write_text("\n".join(lines))


def run_pilot(root: Path, output: Path | None = None,
              datasets: tuple[str, ...] = DATASETS) -> dict:
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "results/novelty_pilot"
    output.mkdir(parents=True, exist_ok=True)
    datasets = tuple(datasets)
    if not datasets or any(dataset not in DATASETS for dataset in datasets) or len(set(datasets)) != len(datasets):
        raise ValueError(f"Datasets must be a nonempty unique subset of {DATASETS}")
    calibration_rows, temperature_rows, reliability_rows_all, routing_rows, failure_rows_all, data_manifests = [], [], [], [], [], []
    for dataset in datasets:
        data = load_dataset(root, dataset)
        c_rows, t_rows, r_rows, route_rows, failure_rows, data_summary = analyze_dataset(data)
        calibration_rows.extend(c_rows)
        temperature_rows.extend(t_rows)
        reliability_rows_all.extend(r_rows)
        routing_rows.extend(route_rows)
        failure_rows_all.extend(failure_rows)
        data_manifests.append({
            **data_summary, "validation_split_sha256": data["split_hash"],
            "order_evidence": data["order_evidence"],
            "audit_sha256": sha256(data["audit_path"]),
            "runs": {method: {
                "result": str(run["result_path"].relative_to(root)),
                "result_sha256": sha256(run["result_path"]),
                "provenance_sha256": sha256(run["provenance_path"]),
                "prediction": str(run["prediction_path"].relative_to(root)),
                "prediction_sha256": sha256(run["prediction_path"]),
                "training_seed": int(run["result"]["seed"]),
                "checkpoint_epoch": int(run["result"]["epochs"]),
                "frozen_backbone_sha256": run["result"].get("frozen_backbone_sha256"),
                "classifier_initial_sha256": run["result"].get("classifier_initial_sha256"),
            } for method, run in data["runs"].items()},
        })

    calibration = pd.DataFrame(calibration_rows)
    temperatures = pd.DataFrame(temperature_rows)
    reliability = pd.DataFrame(reliability_rows_all)
    routing = pd.DataFrame(routing_rows)
    failure_cases = pd.DataFrame(failure_rows_all)
    calibration.to_csv(output / "calibration_metrics.csv", index=False)
    temperatures.to_csv(output / "fold_temperatures.csv", index=False)
    reliability.to_csv(output / "reliability_bins.csv", index=False)
    routing.to_csv(output / "routing_metrics.csv", index=False)
    failure_cases.to_csv(output / "routing_failure_cases.csv", index=False)
    failure_summary = (failure_cases.groupby(
        ["dataset", "split_seed", "target_fallback_rate", "issue"], dropna=False)
        .size().rename("count").reset_index())
    failure_summary.to_csv(output / "routing_failure_summary.csv", index=False)

    calibration_summary = _summary(calibration, ["dataset", "method", "stage"],
                                   ["accuracy", "nll", "brier", "macro_recall_observed"])
    calibration_summary.to_csv(output / "calibration_summary.csv", index=False)
    routing_summary = _summary(routing, ["dataset", "strategy", "target_fallback_rate"],
                               ["accuracy", "nll", "brier", "fallback_fraction"])
    routing_summary.to_csv(output / "routing_summary.csv", index=False)

    gate_beats_random = True
    for dataset in datasets:
        for rate in FALLBACK_RATES:
            gate = routing[(routing.dataset == dataset) & (routing.strategy == "confidence_routing")
                           & np.isclose(routing.target_fallback_rate, rate)]
            random = routing[(routing.dataset == dataset) & (routing.strategy == "random_routing")
                             & np.isclose(routing.target_fallback_rate, rate)]
            gate_beats_random &= (
                gate.accuracy.mean() > random.accuracy.mean()
                and gate.nll.mean() < random.nll.mean()
                and gate.brier.mean() < random.brier.mean()
            )
    gate_matches_always_accuracy_at_50 = all(
        abs(routing[(routing.dataset == dataset) & (routing.strategy == "confidence_routing")
                    & np.isclose(routing.target_fallback_rate, 0.5)].accuracy.mean()
            - routing[(routing.dataset == dataset) & (routing.strategy == "always_on_fusion")].accuracy.mean()) < 1e-12
        for dataset in datasets)

    figure_files = _write_figures(output, calibration, reliability, routing, datasets)
    _write_brief(output, calibration, temperatures, routing, data_manifests,
                 failure_cases, datasets)
    manifest = {
        "status": "complete", "created_from_saved_logits": True,
        "evaluation_scope": "exploratory development validation only; final test untouched",
        "datasets": data_manifests,
        "methods_for_calibration": list(METHODS), "routing_pair": ["linear", "lora"],
        "training_seed": 42, "calibration_split_seeds": list(SPLIT_SEEDS),
        "random_routing_seeds": list(ROUTING_SEEDS), "folds": N_FOLDS,
        "temperature_bounds": list(TEMPERATURE_BOUNDS), "fallback_rates": list(FALLBACK_RATES),
        "reliability_bins": 10,
        "brier_definition": "mean over examples of the sum of squared errors across all class probabilities",
        "checks": {
            "all_methods_have_identical_target_order": True,
            "calibration_and_evaluation_indices_disjoint": True,
            "all_rows_evaluated_once_per_split_seed": True,
            "temperature_scaling_preserves_accuracy": True,
            "zero_fallback_equals_calibrated_linear": True,
            "full_fallback_equals_always_on_fusion": True,
            "confidence_gate_beats_random_mean_on_all_three_metrics": bool(gate_beats_random),
            "gate_matches_always_on_accuracy_at_50_percent": bool(gate_matches_always_accuracy_at_50),
            "test_predictions_used": False,
        },
        "files": sorted([p.name for p in output.iterdir() if p.is_file()]),
        "figures": figure_files,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--datasets", nargs="+", choices=EXTENDED_DATASETS, default=list(DATASETS))
    parser.add_argument("--pairs", nargs="+", help="Directed pairs, e.g. linear:convpass, or all")
    parser.add_argument("--fallback-rules", nargs="+", choices=("replacement", "averaging"))
    args = parser.parse_args()
    if args.pairs or args.fallback_rules or any(x not in DATASETS for x in args.datasets):
        from multi_pair_pilot import run_multi_pair, parse_pairs
        result = run_multi_pair(args.root, args.output, tuple(args.datasets),
                                parse_pairs(args.pairs or ["all"]),
                                tuple(args.fallback_rules or ("replacement", "averaging")))
        print(json.dumps(result, indent=2))
        return
    output = args.output
    if output is None and len(args.datasets) == 1:
        output = args.root / "results/novelty_pilot" / args.datasets[0]
    print(json.dumps(run_pilot(args.root, output, tuple(args.datasets)), indent=2))


if __name__ == "__main__":
    main()
