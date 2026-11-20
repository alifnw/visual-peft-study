"""Compare calibrated routing pairs using saved development logits on CPU."""
from __future__ import annotations

import hashlib
from itertools import combinations, permutations
import json
import importlib.metadata
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from calibration_routing_pilot import (
    EXTENDED_DATASETS, METHODS, SPLIT_SEEDS, ROUTING_SEEDS, FALLBACK_RATES,
    TEMPERATURE_BOUNDS, N_FOLDS, load_dataset, fit_temperature, softmax,
    make_folds, metrics_from_probabilities, metrics_record, reliability_rows,
    _threshold_for_budget, sha256,
)

RULES = ("replacement", "averaging")
METRICS = ("accuracy", "nll", "brier", "macro_recall_observed", "fallback_fraction")


def parse_pairs(values):
    if list(values) == ["all"]:
        return tuple(permutations(METHODS, 2))
    pairs = []
    for value in values:
        parts = value.split(":")
        if len(parts) != 2 or any(x not in METHODS for x in parts) or parts[0] == parts[1]:
            raise ValueError(f"Invalid directed pair {value!r}. Use distinct methods separated by ':'")
        pair = tuple(parts)
        if pair in pairs:
            raise ValueError(f"Duplicate pair: {value}")
        pairs.append(pair)
    if not pairs:
        raise ValueError("At least one directed pair is required")
    return tuple(pairs)


def route(first, second, mask, rule):
    if rule not in RULES:
        raise ValueError(f"Unknown fallback rule: {rule}")
    if first.shape != second.shape or mask.shape != (len(first),):
        raise ValueError("Routing probabilities and mask must be aligned")
    fallback = second if rule == "replacement" else 0.5 * (first + second)
    return np.where(mask[:, None], fallback, first)


def matched_random_mask(mask, seed):
    mask = np.asarray(mask, dtype=bool)
    result = np.zeros(len(mask), dtype=bool)
    result[np.random.default_rng(seed).choice(len(mask), int(mask.sum()), replace=False)] = True
    return result


def stable_seed(*values):
    digest = hashlib.sha256(json.dumps(values).encode()).digest()
    return int.from_bytes(digest[:8], "little")


def metric_row(probabilities, targets, mask=None, **identity):
    metrics = metrics_from_probabilities(targets, probabilities)
    excluded = {"predicted", "correct", "confidence", "nll_samples", "brier_samples"}
    return {**identity, **{k: v for k, v in metrics.items() if k not in excluded},
            "fallback_count": int(mask.sum()) if mask is not None else np.nan,
            "fallback_fraction": float(mask.mean()) if mask is not None else np.nan}


def summarize(frame, groups, values):
    summary = frame.groupby(groups, dropna=False)[values].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(str(x) for x in col if x) if isinstance(col, tuple) else col
                       for col in summary.columns]
    return summary


def calibrate(data, methods=METHODS):
    targets, n = data["targets"], len(data["targets"])
    calibration, temperatures, reliability, folds_rows, caches = [], [], [], [], {}
    for seed in SPLIT_SEEDS:
        folds = make_folds(n, seed)
        cache = {"folds": folds, "probabilities": {}, "temperatures": {}}
        for fold, evaluation in enumerate(folds):
            fitting = np.concatenate([indices for i, indices in enumerate(folds) if i != fold])
            if np.intersect1d(fitting, evaluation).size:
                raise AssertionError("Calibration and evaluation overlap")
            for index in evaluation:
                folds_rows.append({"dataset": data["dataset"], "split_seed": seed,
                                   "sample_index": int(index), "fold": fold,
                                   "image_path": str(data["ordered_paths"][index]),
                                   "target": int(targets[index])})
        for method in methods:
            logits = data["runs"][method]["logits"]
            probabilities = np.zeros_like(logits)
            for fold, evaluation in enumerate(folds):
                fitting = np.concatenate([indices for i, indices in enumerate(folds) if i != fold])
                fitted = fit_temperature(logits[fitting], targets[fitting])
                cache["temperatures"][method, fold] = fitted["temperature"]
                probabilities[evaluation] = softmax(logits[evaluation], fitted["temperature"])
                temperatures.append({"dataset": data["dataset"], "method": method,
                                     "split_seed": seed, "fold": fold,
                                     "calibration_n": len(fitting), "evaluation_n": len(evaluation),
                                     "calibration_classes_observed": len(np.unique(targets[fitting])),
                                     "evaluation_classes_observed": len(np.unique(targets[evaluation])),
                                     **fitted})
            raw = softmax(logits)
            if not np.array_equal(raw.argmax(1), probabilities.argmax(1)):
                raise AssertionError("Temperature changed top-1 predictions")
            cache["probabilities"][method] = probabilities
            for stage, p in (("raw_oof", raw), ("temperature_scaled_oof", probabilities)):
                calibration.append(metrics_record(data["dataset"], data["subset"], method, stage, seed, p, targets))
                reliability.extend(reliability_rows(data["dataset"], method, stage, seed, targets, p))
        caches[seed] = cache
    return caches, calibration, temperatures, reliability, folds_rows


def evaluate_pairs(data, caches, pairs, rules, rates=FALLBACK_RATES):
    targets, n = data["targets"], len(data["targets"])
    rows, decisions, oracles = [], [], []
    for first, second in pairs:
        first_correct = data["runs"][first]["logits"].argmax(1) == targets
        second_correct = data["runs"][second]["logits"].argmax(1) == targets
        oracles.append({"dataset": data["dataset"], "first_method": first, "second_method": second,
                        "n": n, "first_accuracy": float(first_correct.mean()),
                        "second_accuracy": float(second_correct.mean()),
                        "oracle_two_single_predictions_accuracy": float((first_correct | second_correct).mean()),
                        "both_wrong": int((~first_correct & ~second_correct).sum()),
                        "second_fixes_first": int((~first_correct & second_correct).sum()),
                        "first_fixes_second": int((first_correct & ~second_correct).sum()),
                        "diagnostic_uses_labels": True})
        for seed, cache in caches.items():
            first_p, second_p = (cache["probabilities"][m] for m in (first, second))
            base_id = {"dataset": data["dataset"], "evaluation_subset": data["subset"],
                       "first_method": first, "second_method": second, "split_seed": seed}
            for rule in rules:
                identity = {**base_id, "fallback_rule": rule}
                zero, full = np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
                if not np.array_equal(route(first_p, second_p, zero, rule), first_p):
                    raise AssertionError("Zero fallback endpoint differs")
                expected = second_p if rule == "replacement" else 0.5 * (first_p + second_p)
                if not np.array_equal(route(first_p, second_p, full, rule), expected):
                    raise AssertionError("Full fallback endpoint differs")
                for strategy, p, mask in (("calibrated_first", first_p, zero),
                                          ("calibrated_second", second_p, None),
                                          ("always_on_averaging", 0.5 * (first_p + second_p), full)):
                    rows.append(metric_row(p, targets, mask, **identity, strategy=strategy,
                                           target_fallback_rate=np.nan, routing_seed=np.nan,
                                           random_protocol="none"))
                for rate in rates:
                    gate = np.zeros(n, dtype=bool)
                    thresholds, fold_ids = np.zeros(n), np.zeros(n, dtype=int)
                    random_masks = {s: np.zeros(n, dtype=bool) for s in ROUTING_SEEDS}
                    for fold, evaluation in enumerate(cache["folds"]):
                        fitting = np.concatenate([v for i, v in enumerate(cache["folds"]) if i != fold])
                        temperature = cache["temperatures"][first, fold]
                        calibration_confidence = softmax(data["runs"][first]["logits"][fitting], temperature).max(1)
                        threshold = _threshold_for_budget(calibration_confidence, rate)
                        selected = first_p[evaluation].max(1) < threshold
                        gate[evaluation], thresholds[evaluation], fold_ids[evaluation] = selected, threshold, fold
                        for route_seed in ROUTING_SEEDS:
                            # The paired rules use identical masks to isolate the fallback operation.
                            random_masks[route_seed][evaluation] = matched_random_mask(
                                selected, stable_seed(data["dataset"], first, second, seed, fold, rate, route_seed))
                    probabilities = route(first_p, second_p, gate, rule)
                    correct = probabilities.argmax(1) == targets
                    counts = {"errors_fixed": int((~first_correct & correct & gate).sum()),
                              "correct_predictions_broken": int((first_correct & ~correct & gate).sum()),
                              "selected_errors_not_fixed": int((~first_correct & ~correct & gate).sum()),
                              "first_errors_missed": int((~first_correct & ~gate).sum())}
                    rows.append(metric_row(probabilities, targets, gate, **identity,
                                           strategy="confidence_routing", target_fallback_rate=rate,
                                           routing_seed=np.nan, random_protocol="none", **counts))
                    for index in range(n):
                        decisions.append({**identity, "target_fallback_rate": rate,
                                          "sample_index": index, "image_path": str(data["ordered_paths"][index]),
                                          "target": int(targets[index]), "fold": int(fold_ids[index]),
                                          "threshold": float(thresholds[index]),
                                          "first_confidence": float(first_p[index].max()),
                                          "fallback": bool(gate[index]),
                                          "first_correct": bool(first_correct[index]),
                                          "second_correct": bool(second_correct[index]),
                                          "routed_correct": bool(correct[index])})
                    for route_seed, mask in random_masks.items():
                        for evaluation in cache["folds"]:
                            if mask[evaluation].sum() != gate[evaluation].sum():
                                raise AssertionError("Random routing fallback count differs from gate")
                        rows.append(metric_row(route(first_p, second_p, mask, rule), targets, mask,
                                               **identity, strategy="random_routing", target_fallback_rate=rate,
                                               routing_seed=route_seed, random_protocol="exact_count_per_fold"))
    return rows, decisions, oracles


def export_figures(output, calibration_summary, reliability, routing_summary, oracles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    files = []
    for dataset in calibration_summary.dataset.unique():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        subset = calibration_summary[calibration_summary.dataset == dataset]
        x = np.arange(len(METHODS))
        for axis, metric in zip(axes, ("nll", "brier")):
            for offset, stage, label in ((-0.18, "raw_oof", "Raw"), (0.18, "temperature_scaled_oof", "OOF temperature-scaled")):
                group = subset[subset.stage == stage].set_index("method").loc[list(METHODS)]
                axis.bar(x + offset, group[metric + "_mean"], width=0.36, label=label)
            axis.set(xticks=x, xticklabels=METHODS, ylabel=metric.upper(), title=dataset)
            axis.legend(fontsize=8)
        fig.tight_layout()
        files.extend(save_figure(fig, output, dataset + "-calibration"))
        fig, axes = plt.subplots(1, 4, figsize=(15, 4))
        for axis, method in zip(axes, METHODS):
            axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
            r = reliability[(reliability.dataset == dataset) & (reliability.method == method)]
            for stage in ("raw_oof", "temperature_scaled_oof"):
                # Each split reuses the same images. Pooling is descriptive, not a larger sample.
                points = []
                for _, group in r[r.stage == stage].groupby("bin"):
                    valid = group[group["count"] > 0]
                    if not valid.empty:
                        count = valid["count"].sum()
                        points.append((np.average(valid.mean_confidence, weights=valid["count"]),
                                       np.average(valid.empirical_accuracy, weights=valid["count"]),
                                       count / len(SPLIT_SEEDS)))
                for confidence, accuracy, count in points:
                    axis.scatter(confidence, accuracy, marker="o" if stage == "raw_oof" else "x",
                                 color="tab:blue" if stage == "raw_oof" else "tab:orange")
                    axis.annotate(f"{count:.0f}", (confidence, accuracy), fontsize=7)
            axis.set(title=method, xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy")
        fig.suptitle(dataset + ": raw (blue), scaled (orange). Labels: mean bin count per split")
        fig.tight_layout()
        files.extend(save_figure(fig, output, dataset + "-reliability"))
        pairs = list(dict.fromkeys(zip(routing_summary.first_method, routing_summary.second_method)))
        columns = 4
        fig, axes = plt.subplots(int(np.ceil(len(pairs) / columns)), columns, figsize=(16, 3.1 * int(np.ceil(len(pairs) / columns))), squeeze=False)
        for axis, (first, second) in zip(axes.flat, pairs):
            group = routing_summary[(routing_summary.dataset == dataset) & (routing_summary.first_method == first)
                                    & (routing_summary.second_method == second)]
            for rule in sorted(group.fallback_rule.unique()):
                color = "tab:blue" if rule == "averaging" else "tab:orange"
                g = group[(group.fallback_rule == rule) & (group.strategy == "confidence_routing")].sort_values("target_fallback_rate")
                axis.plot(g.fallback_fraction_mean, g.nll_mean, marker="o", color=color, label=rule)
                random = group[(group.fallback_rule == rule) & (group.strategy == "random_routing")].sort_values("target_fallback_rate")
                axis.plot(random.fallback_fraction_mean, random.nll_mean, linestyle=":", color=color, alpha=0.7)
            for strategy, color in (("calibrated_first", "gray"), ("calibrated_second", "green"), ("always_on_averaging", "purple")):
                g = group[group.strategy == strategy]
                if len(g): axis.axhline(g.nll_mean.iloc[0], color=color, linestyle="--", alpha=0.6)
            axis.set(title=f"{first} → {second}", xlabel="Actual fallback", ylabel="NLL")
            axis.legend(fontsize=7)
        for axis in list(axes.flat)[len(pairs):]: axis.set_visible(False)
        fig.suptitle(dataset + ": solid confidence, dotted matched random. Dashed gray:first, green:second, purple:averaging")
        fig.tight_layout()
        files.extend(save_figure(fig, output, dataset + "-routing"))
        fig, axis = plt.subplots(figsize=(5, 4))
        matrix = np.full((len(METHODS), len(METHODS)), np.nan)
        for row in oracles[oracles.dataset == dataset].itertuples():
            matrix[METHODS.index(row.first_method), METHODS.index(row.second_method)] = row.both_wrong
        image = axis.imshow(matrix, cmap="Blues")
        for i in range(len(METHODS)):
            for j in range(len(METHODS)):
                if np.isfinite(matrix[i, j]): axis.text(j, i, str(int(matrix[i, j])), ha="center", va="center")
        axis.set(xticks=range(4), yticks=range(4), xticklabels=METHODS, yticklabels=METHODS,
                 title=dataset + ": shared error counts")
        fig.colorbar(image, ax=axis)
        fig.tight_layout()
        files.extend(save_figure(fig, output, dataset + "-error-overlap"))
    return files


def save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    files = []
    for extension in ("png", "svg"):
        path = output / f"{name}.{extension}"
        fig.savefig(path, dpi=220, bbox_inches="tight")
        files.append(path.name)
    plt.close(fig)
    return files


def write_brief(output, calibration, routing):
    lines = ["# Four-dataset calibration and routing pilot", "",
             "Development validation, checkpoint epoch 100, training seed 42. Split seeds 42/43/44 reuse the same images.", "",
             "## Calibration", "", "| Dataset | Method | NLL raw | NLL scaled | Brier raw | Brier scaled |",
             "|---|---|---:|---:|---:|---:|"]
    for (dataset, method), g in calibration.groupby(["dataset", "method"]):
        g = g.set_index("stage")
        lines.append(f"| {dataset} | {method} | {g.loc['raw_oof','nll_mean']:.4f} | {g.loc['temperature_scaled_oof','nll_mean']:.4f} | {g.loc['raw_oof','brier_mean']:.4f} | {g.loc['temperature_scaled_oof','brier_mean']:.4f} |")
    lines += ["", "## Routing candidates", "",
              "The table shows the lowest development NLL among tested confidence-routing configurations per dataset. All combinations remain in CSV. Selection is exploratory.", "",
              "| Dataset | Direction | Rule | Target fallback | Accuracy | NLL | Brier | First-model NLL | Second-model NLL | Always-on NLL | Best single NLL (all four) |",
              "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for dataset, group in routing.groupby("dataset"):
        gated = group[group.strategy == "confidence_routing"].sort_values(["nll_mean", "first_method", "second_method", "fallback_rule", "target_fallback_rate"])
        best = gated.iloc[0]
        controls = group[(group.first_method == best.first_method) & (group.second_method == best.second_method)
                         & (group.fallback_rule == best.fallback_rule)].set_index("strategy")
        second = controls.loc["calibrated_second", "nll_mean"]
        first = controls.loc["calibrated_first", "nll_mean"]
        always = controls.loc["always_on_averaging", "nll_mean"]
        best_single = calibration[(calibration.dataset == dataset) & (calibration.stage == "temperature_scaled_oof")].nll_mean.min()
        lines.append(f"| {dataset} | {best.first_method} → {best.second_method} | {best.fallback_rule} | {best.target_fallback_rate:.0%} | {best.accuracy_mean:.2%} | {best.nll_mean:.4f} | {best.brier_mean:.4f} | {first:.4f} | {second:.4f} | {always:.4f} | {best_single:.4f} |")
    lines += ["", "## Interpretation", "",
              "- No independent test predictions are used. Pair, direction and budget exploration increases selection bias.",
              "- Temperature fitting and scoring are separated within each fold. Model development already used these validation sets.",
              "- Random routing matches the realized gate fallback count per fold. This is a diagnostic matched-budget comparison, not a deployable random policy with knowledge of future batch counts.",
              "- Replacement uses the second model at fallback. Averaging combines both calibrated probability vectors.",
              "- Oracle accuracy uses labels and can select only between the two single-model predictions. It bounds replacement accuracy, not all possible averaging predictions.",
              "- Fallback is an additional-model invocation count. Both models run a full ViT and latency has not been measured.",
              "- Raw NLL and Brier magnitudes across different class counts are not a cross-dataset ranking.",
              "- Report each dataset separately. Split-seed SD measures partition sensitivity, not training-seed uncertainty.", ""]
    (output / "MENTORING_BRIEF.md").write_text("\n".join(lines))


def run_multi_pair(root, output=None, datasets=EXTENDED_DATASETS, pairs=None, rules=RULES):
    root = Path(root).resolve()
    output = Path(output).resolve() if output else root / "results/novelty_pilot_four_datasets"
    if not datasets or len(set(datasets)) != len(datasets) or any(x not in EXTENDED_DATASETS for x in datasets):
        raise ValueError("Datasets must be a unique, nonempty subset of the supported datasets")
    pairs = parse_pairs([f"{a}:{b}" for a, b in pairs]) if pairs is not None else parse_pairs(["all"])
    if not rules or len(set(rules)) != len(rules) or any(x not in RULES for x in rules):
        raise ValueError("Fallback rules must be a unique, nonempty subset of replacement and averaging")
    if output == root / "results/novelty_pilot" or root / "results/development" in (output, *output.parents):
        raise ValueError("Choose a new analysis output directory, not historical pilot or input artefacts")
    # Validate every input before creating output. Missing predictions never trigger training.
    loaded = [load_dataset(root, dataset) for dataset in datasets]
    output.mkdir(parents=True, exist_ok=True)
    configuration = {"datasets": list(datasets), "pairs": [list(p) for p in pairs], "fallback_rules": list(rules)}
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if previous.get("configuration") != configuration:
            raise ValueError("Output directory belongs to a different experiment. Choose another --output")
    tables = {name: [] for name in ("calibration_metrics", "fold_temperatures", "reliability_bins", "fold_assignments", "routing_metrics", "routing_decisions", "oracle_diagnostics", "class_coverage")}
    inputs = []
    for data in loaded:
        print(f"Analyzing saved logits: {data['dataset']}", flush=True)
        caches, calibration, temperatures, reliability, folds = calibrate(data)
        routes, decisions, oracles = evaluate_pairs(data, caches, pairs, rules)
        for name, records in (("calibration_metrics", calibration), ("fold_temperatures", temperatures),
                              ("reliability_bins", reliability), ("fold_assignments", folds),
                              ("routing_metrics", routes), ("routing_decisions", decisions),
                              ("oracle_diagnostics", oracles)):
            tables[name].extend(records)
        counts = np.bincount(data["targets"], minlength=data["class_count"])
        tables["class_coverage"].extend({"dataset": data["dataset"], "class_id": i, "count": int(count)} for i, count in enumerate(counts))
        np.savez_compressed(output / f"{data['dataset']}-calibrated-oof.npz",
                            targets=data["targets"], image_paths=data["ordered_paths"],
                            **{f"{m}_split{s}": cache["probabilities"][m] for s, cache in caches.items() for m in METHODS})
        inputs.append({"dataset": data["dataset"], "n": len(data["targets"]), "class_count": data["class_count"],
                       "validation_split_sha256": data["split_hash"], "order_evidence": data["order_evidence"],
                       "audit_sha256": sha256(data["audit_path"]),
                       "runs": {m: {"prediction": str(r["prediction_path"].relative_to(root)),
                                     "prediction_sha256": sha256(r["prediction_path"]),
                                     "result_sha256": sha256(r["result_path"]),
                                     "provenance_sha256": sha256(r["provenance_path"]),
                                     "checkpoint_epoch": 100, "training_seed": 42} for m, r in data["runs"].items()}})
    frames = {name: pd.DataFrame(rows) for name, rows in tables.items()}
    calibration_summary = summarize(frames["calibration_metrics"], ["dataset", "method", "stage"], ["accuracy", "nll", "brier", "macro_recall_observed"])
    groups = ["dataset", "first_method", "second_method", "fallback_rule", "strategy", "target_fallback_rate"]
    routing_summary = summarize(frames["routing_metrics"], groups, list(METRICS))
    frames.update(calibration_summary=calibration_summary, routing_summary=routing_summary)
    for name, frame in frames.items(): frame.to_csv(output / f"{name}.csv", index=False)
    figures = export_figures(output, calibration_summary, frames["reliability_bins"], routing_summary, frames["oracle_diagnostics"])
    write_brief(output, calibration_summary, routing_summary)
    manifest = {"schema_version": 2, "status": "complete", "configuration": configuration,
                "created_from_saved_logits": True, "training_launched": False, "model_checkpoints_loaded": False,
                "test_predictions_used": False, "training_seed": 42, "calibration_split_seeds": list(SPLIT_SEEDS),
                "random_routing_seeds": list(ROUTING_SEEDS), "random_protocol": "exact_count_per_fold",
                "folds": N_FOLDS, "temperature_bounds": list(TEMPERATURE_BOUNDS), "fallback_rates": list(FALLBACK_RATES),
                "evaluation_scope": "exploratory development validation", "datasets": inputs,
                "environment": {"python": sys.version.split()[0], **{
                    name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "matplotlib")}},
                "checks": {"label_and_audit_order_match": True, "common_initialization": True,
                           "calibration_evaluation_disjoint": True, "positive_temperature_preserves_argmax": True,
                           "routing_endpoints_match": True, "random_fallback_counts_match_each_fold": True},
                "source_sha256": {p: sha256(root / p) for p in ("scripts/calibration_routing_pilot.py", "scripts/multi_pair_pilot.py")},
                "row_counts": {name: len(frame) for name, frame in frames.items()}, "figures": figures}
    manifest["files_sha256"] = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "manifest.json"}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
