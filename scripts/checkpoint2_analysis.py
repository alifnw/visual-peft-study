"""Build Checkpoint 2 tables and figures from audited development logits."""

from itertools import product
from pathlib import Path
import argparse
import hashlib
import json

import numpy as np
import pandas as pd

from expansion_analysis import prediction_metrics
from multi_pair_pilot import calibrate, evaluate_pairs


DATASETS = ("oxford_flowers102", "dtd", "resisc45", "svhn", "patch_camelyon")
METHODS = ("linear", "lora", "ssf", "convpass", "bitfit", "adaptformer", "vpt_deep", "fact_tt")
PAIRS = {"svhn": ("convpass", "ssf"), "patch_camelyon": ("convpass", "linear")}
DATASET_NAMES = dict(zip(DATASETS, ("Flowers102", "DTD", "RESISC45", "SVHN", "PatchCamelyon")))
METHOD_NAMES = dict(zip(METHODS, ("Linear", "LoRA", "SSF", "ConvPass", "BitFit", "AdaptFormer",
                                  "VPT-Deep", "FacT-TT")))
SCORES = ["accuracy", "nll", "brier", "fallback_fraction"]
STRATEGIES = {"calibrated_first": "First expert", "calibrated_second": "Second expert",
              "always_on_averaging": "Always-on", "confidence_routing": "Selective",
              "random_routing": "Matched random"}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(value, path):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def run_directory(root, dataset, method, seed):
    if dataset == "oxford_flowers102" and method in METHODS[:3] and seed == 42:
        return root / "results/checkpoint1" / method
    return root / "results/development" / dataset / method / f"seed{seed}"


def load_run(root, dataset, method, seed, audit):
    folder = run_directory(root, dataset, method, seed)
    result_path = folder / "result.json"
    provenance_path = folder / "provenance.json"
    prediction = folder / "validation_last.npz"
    reloaded = folder / "validation_last_reloaded.npz"
    result = read(result_path)
    actual = tuple(result[key] for key in ("dataset", "method", "seed", "epochs", "phase"))
    if actual != (dataset, method, seed, 100, "complete"):
        raise ValueError(f"Run identity mismatch: {result_path}")
    if result.get("smoke_check") or result.get("rows", {}).get("test_used", 0):
        raise ValueError(f"Unexpected evaluation scope: {result_path}")
    if result.get("checkpoint_reload_exact") is not True:
        raise ValueError(f"Checkpoint reload was not verified: {result_path}")

    provenance = read(provenance_path)
    for name in ("train800.txt", "val200.txt"):
        if provenance["split_sha256"][name] != audit["splits"][name]["list_sha256"]:
            raise ValueError(f"Split hash mismatch: {result_path} / {name}")
    records = audit["development_records"]["val200.txt"]
    expected_targets = np.array([row["label"] for row in records])
    expected_paths = np.array([row["path"] for row in records])
    with np.load(prediction, allow_pickle=False) as arrays:
        logits, targets = arrays["logits"], arrays["targets"]
        if logits.shape != (len(records), audit["class_count"]) or not np.isfinite(logits).all():
            raise ValueError(f"Invalid logits: {prediction}")
        if not np.array_equal(targets, expected_targets):
            raise ValueError(f"Validation targets out of order: {prediction}")
        if "validation_indices" in arrays and not np.array_equal(arrays["validation_indices"], np.arange(len(records))):
            raise ValueError(f"Validation indices out of order: {prediction}")
        if "image_paths" in arrays and not np.array_equal(arrays["image_paths"], expected_paths):
            raise ValueError(f"Validation paths out of order: {prediction}")
    if reloaded.is_file():
        with np.load(reloaded, allow_pickle=False) as arrays:
            if not np.array_equal(logits, arrays["logits"]) or not np.array_equal(targets, arrays["targets"]):
                raise ValueError(f"Reloaded logits differ: {prediction}")
    raw_accuracy = 100 * (logits.argmax(axis=1) == targets).mean()
    if abs(raw_accuracy - result["last_epoch_validation_top1_percent"]) > 1e-6:
        raise ValueError(f"Accuracy disagrees with stored logits: {prediction}")
    indices = np.array(audit["validation_indices_without_training_content"]) if dataset == "dtd" else np.arange(len(records))
    if dataset == "dtd":
        clean = folder / "validation_last_without_training_duplicates.npz"
        with np.load(clean, allow_pickle=False) as arrays:
            if not np.array_equal(logits[indices], arrays["logits"]) or not np.array_equal(targets[indices], arrays["targets"]):
                raise ValueError(f"DTD overlap mask disagrees with stored clean subset: {clean}")
        if audit["duplicate_validation_indices"] != [0] or len(indices) != 199:
            raise ValueError(f"Invalid DTD overlap audit: {prediction}")
    source_paths = [result_path, provenance_path, prediction, folder / "history.csv"]
    if reloaded.is_file():
        source_paths.append(reloaded)
    if dataset == "dtd":
        source_paths.append(clean)
    sources = {p.relative_to(root).as_posix(): digest(p) for p in source_paths}
    return {"result": result, "logits": logits[indices].astype(np.float64), "targets": targets[indices],
            "paths": expected_paths[indices], "sources": sources, "folder": folder}


def write_baselines(root, destination, audited):
    rows, recipes, history, inputs = [], [], [], {}
    for dataset, method in product(DATASETS, METHODS):
        audit = audited[dataset]
        run = load_run(root, dataset, method, 42, audit)
        r = run["result"]
        m = prediction_metrics(run["logits"], run["targets"])
        identity = {"dataset": dataset, "method": method, "training_seed": 42}
        rows.append({**identity, "validation_n": m["n"], "classes_observed": m["classes_observed"],
                     "accuracy_percent": m["accuracy_percent"], "nll": m["nll"], "brier": m["brier"],
                     "macro_recall_observed": m["macro_recall_observed"],
                     "trainable_parameters": r["trainable_parameters_including_head"],
                     "snapshot": run["folder"].relative_to(root).as_posix()})
        cfg = r["config"]
        recipes.append({**identity, "epochs": r["epochs"], "learning_rate": cfg.get("learning_rate"),
                        "weight_decay": cfg.get("weight_decay"), "effective_batch": r.get("effective_batch_size"),
                        "microbatch": r.get("microbatch_size"),
                        "method_settings": json.dumps({key: cfg[key] for key in
                            ("lora_rank", "convpass", "vpt_num", "adaptformer", "fact_tt") if key in cfg},
                            sort_keys=True)})
        curve = pd.read_csv(run["folder"] / "history.csv")
        if curve["epoch"].tolist() != list(range(1, 101)):
            raise ValueError(f"Unexpected epoch history in {run['folder']}")
        history.append(curve.assign(**identity))
        inputs.update(run["sources"])
    metrics = pd.DataFrame(rows)
    metrics.to_csv(destination / "metrics.csv", index=False)
    pd.DataFrame(recipes).to_csv(destination / "recipes.csv", index=False)
    pd.concat(history, ignore_index=True).to_csv(destination / "learning_curves.csv", index=False)
    return metrics, inputs


def write_pilot(destination):
    frames, summaries = [], []
    keys = ["dataset", "first_method", "second_method", "target_fallback_rate"]
    calibration = None
    for owner in ("gipta", "charleen"):
        source = destination / "source" / owner
        manifest = read(source / "original_manifest.json")
        if (manifest["status"] != "complete" or manifest["training_seed"] != 42
                or manifest["test_predictions_used"] or manifest["training_launched"]):
            raise ValueError(f"Incompatible source manifest: {owner}")
        for p in source.glob("*.csv"):
            if digest(p) != manifest["files_sha256"][p.name]:
                raise ValueError(f"Source checksum mismatch: {p}")
        routing = pd.read_csv(source / "routing_summary.csv").assign(contributor=owner)
        current = pd.read_csv(source / "calibration_summary.csv")
        if calibration is None:
            calibration = current
        else:
            pd.testing.assert_frame_equal(calibration, current)
        paired = routing[routing.strategy == "confidence_routing"].pivot(index=keys, columns="fallback_rule",
                                                                         values="nll_mean")
        if paired.isna().any().any() or len(paired) != 72:
            raise ValueError(f"Incomplete averaging vs replacement comparison: {owner}")
        summaries.append({"contributor": owner, "averaging_lower_nll": int((paired.averaging < paired.replacement).sum()),
                          "matched_configurations": len(paired)})
        frames.append(routing)
    routing = pd.concat(frames, ignore_index=True)
    if routing.duplicated(keys + ["strategy", "fallback_rule"]).any():
        raise ValueError("The two contributions contain duplicate routing configurations")
    routing.to_csv(destination / "routing_summary.csv", index=False)
    calibration.to_csv(destination / "calibration_summary.csv", index=False)
    best = (routing[routing.strategy == "confidence_routing"].sort_values("nll_mean")
            .groupby("dataset", sort=False).head(1).copy())
    for i, row in best.iterrows():
        match = routing[(routing.dataset == row.dataset) & (routing.first_method == row.first_method)
                        & (routing.second_method == row.second_method) & (routing.fallback_rule == row.fallback_rule)]
        best.loc[i, "always_on_nll"] = match.loc[match.strategy == "always_on_averaging", "nll_mean"].item()
        best.loc[i, "matched_random_nll"] = match.loc[(match.strategy == "random_routing")
            & (match.target_fallback_rate == row.target_fallback_rate), "nll_mean"].item()
    best.to_csv(destination / "selected_candidates.csv", index=False)
    ownership = pd.DataFrame(summaries)
    ownership.loc[len(ownership)] = ["team", ownership.averaging_lower_nll.sum(),
                                     ownership.matched_configurations.sum()]
    ownership.to_csv(destination / "averaging_comparison.csv", index=False)
    verify_cross_checks(destination)
    return routing


def verify_cross_checks(destination):
    checks = pd.read_csv(destination / "source/crosscheck_decisions.csv")
    keys = ["contributor", "reviewer", "dataset", "first_method", "second_method",
            "split_seed", "fallback_rule", "target_fallback_rate"]
    rows = []
    for values, items in checks.groupby(keys):
        identity = dict(zip(keys, values))
        owner = identity["contributor"]
        scores = pd.read_csv(destination / "source" / owner / "routing_metrics.csv")
        same = scores.strategy == "confidence_routing"
        for key, value in identity.items():
            if key not in ("contributor", "reviewer"):
                same &= scores[key] == value
        score = scores.loc[same].iloc[0]
        if len(items) != 200 or items.sample_index.nunique() != 200:
            raise ValueError("Incomplete per-image cross-check")
        if not np.allclose([items.routed_correct.mean(), items.fallback.mean()],
                           [score.accuracy, score.fallback_fraction], atol=1e-12, rtol=0):
            raise ValueError("Cross-check differs from pilot metrics")
        rows.append({**identity, "n": len(items), "accuracy": score.accuracy,
                     "fallback_fraction": score.fallback_fraction})
    if len(rows) != 2:
        raise ValueError("Both contributor cross-checks are required")
    pd.DataFrame(rows).to_csv(destination / "cross_checks.csv", index=False)


def write_confirmation(root, destination, audited, pilot):
    availability, scores, temps, masks, inputs = [], [], [], [], {}
    for dataset, methods in PAIRS.items():
        audit = audited[dataset]
        for seed in (42, 43, 44):
            ident = {"dataset": dataset, "training_seed": seed}
            runs = {}
            for method in methods:
                folder = run_directory(root, dataset, method, seed)
                if not (folder / "result.json").is_file():
                    continue
                run = load_run(root, dataset, method, seed, audit)
                baseline = run_directory(root, dataset, method, 42)
                reference = read(baseline / "result.json")["config"]
                if {**reference, "seed": seed} != run["result"]["config"]:
                    raise ValueError(f"Training recipe changed beyond seed: {folder}")
                runs[method] = run
                inputs.update(run["sources"])
            missing = [method for method in methods if method not in runs]
            availability.append({**ident, "first_method": methods[0], "second_method": methods[1],
                                 "status": "incomplete" if missing else "evaluated",
                                 "missing_epoch100_artifact": ", ".join(missing)})
            if missing:
                continue
            for key in ("classifier_initial_sha256", "frozen_backbone_sha256"):
                hashes = {run["result"].get(key) for run in runs.values()}
                if len(hashes) != 1 or None in hashes:
                    raise ValueError(f"Pair initialization mismatch: {ident} / {key}")
            first = runs[methods[0]]
            data = {"dataset": dataset, "subset": "validation", "runs": runs,
                    "targets": first["targets"], "ordered_paths": first["paths"]}
            caches, _, temperatures, _, _ = calibrate(data, methods)
            rows, decisions, _ = evaluate_pairs(data, caches, [methods], ["averaging"], rates=(0.25,))
            frame = pd.DataFrame(rows)
            frame["training_seed"] = seed
            frame.loc[frame.strategy == "calibrated_second", "fallback_fraction"] = 0
            scores.append(frame)
            temps.extend({**row, "training_seed": seed} for row in temperatures)
            masks.extend({**row, "training_seed": seed} for row in decisions)
    status = pd.DataFrame(availability)
    status.to_csv(destination / "availability.csv", index=False)
    all_scores = pd.concat(scores, ignore_index=True)
    all_scores.to_csv(destination / "metrics_by_split.csv", index=False)
    pd.DataFrame(temps).to_csv(destination / "temperatures.csv", index=False)
    decisions = pd.DataFrame(masks)
    decisions.to_csv(destination / "routing_decisions.csv", index=False)
    folds = ["dataset", "training_seed", "split_seed", "fold", "first_method", "second_method", "threshold"]
    (decisions.groupby(folds, as_index=False).agg(evaluation_n=("sample_index", "size"),
                                                  fallback_n=("fallback", "sum"))
     .to_csv(destination / "routing_thresholds.csv", index=False))
    grouping = ["dataset", "training_seed", "strategy"]
    splits = all_scores.groupby(grouping + ["split_seed"], as_index=False)[SCORES].mean()
    summary = splits.groupby(grouping)[SCORES].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join(x for x in col if x) for col in summary.columns]
    summary = summary.rename(columns={f"{x}_std": f"{x}_split_sd" for x in SCORES})
    summary.to_csv(destination / "metrics_by_training_seed.csv", index=False)
    for dataset, (first, second) in PAIRS.items():
        previous = pilot[(pilot.dataset == dataset) & (pilot.first_method == first)
                         & (pilot.second_method == second) & (pilot.fallback_rule == "averaging")]
        previous = previous[previous.target_fallback_rate.isna() | (previous.target_fallback_rate == 0.25)]
        current = summary[(summary.dataset == dataset) & (summary.training_seed == 42)]
        columns = ["accuracy_mean", "nll_mean", "brier_mean"]
        left = previous.set_index("strategy")[columns].sort_index()
        right = current.set_index("strategy")[columns].sort_index()
        pd.testing.assert_frame_equal(left, right, atol=1e-12, rtol=0)
    for dataset in PAIRS:
        for seed in (42, 43, 44):
            expected = status.loc[(status.dataset == dataset) & (status.training_seed == seed), "status"].item()
            count = len(summary[(summary.dataset == dataset) & (summary.training_seed == seed)])
            if count != (5 if expected == "evaluated" else 0):
                raise ValueError(f"Incorrect number of strategies for {dataset} seed {seed}")
    evaluated = status[status.status == "evaluated"].groupby("dataset").size()
    across = summary[summary.dataset.isin(evaluated[evaluated == 3].index)]
    across = across.groupby(["dataset", "strategy"])[[f"{x}_mean" for x in SCORES]].agg(["mean", "std"]).reset_index()
    across.columns = ["_".join(x for x in col if x).replace("_mean_std", "_training_seed_sd")
                      .replace("_mean_mean", "_mean") for col in across.columns]
    across.to_csv(destination / "summary_three_training_seeds.csv", index=False)
    return status, summary, inputs


def figures(destination, baseline, confirmation):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = baseline.pivot(index="dataset", columns="method", values="accuracy_percent").loc[list(DATASETS), list(METHODS)]
    fig, ax = plt.subplots(figsize=(9.4, 3.7), layout="constrained")
    scale = ax.imshow(matrix, cmap="YlGnBu", vmin=40, vmax=100, aspect="auto")
    ax.set_xticks(range(len(METHODS)), [METHOD_NAMES[x] for x in METHODS])
    ax.set_yticks(range(len(DATASETS)), [DATASET_NAMES[x] for x in DATASETS])
    for i, j in product(range(len(DATASETS)), range(len(METHODS))):
        v = matrix.iloc[i, j]
        ax.text(j, i, f"{v:.1f}", color="white" if v >= 79 else "black", ha="center", va="center")
    ax.set_title("Epoch 100 validation accuracy (%) · training seed 42")
    fig.colorbar(scale, ax=ax, shrink=.82)
    for ext in ("png", "svg"):
        fig.savefig(destination / f"baseline_accuracy.{ext}", dpi=160)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9.4, 3.7), sharey=True, layout="constrained")
    for (dataset, methods), ax in zip(PAIRS.items(), axs):
        frame = confirmation[confirmation.dataset == dataset].pivot(index="training_seed", columns="strategy",
                                                                    values="nll_mean")
        for strategy, marker in (("calibrated_first", "o"), ("calibrated_second", "^"),
                                 ("always_on_averaging", "s"), ("confidence_routing", "D"), ("random_routing", "x")):
            ax.plot((42, 43, 44), frame[strategy].reindex([42, 43, 44]), marker=marker,
                    label=STRATEGIES[strategy], linewidth=1)
        ax.set(title=f"{DATASET_NAMES[dataset]} · {METHOD_NAMES[methods[0]]} → {METHOD_NAMES[methods[1]]}",
               xlabel="Training seed", xticks=(42, 43, 44))
        ax.grid(axis="y", alpha=.2)
    axs[0].set_ylabel("NLL (lower is better)")
    fig.legend(*axs[0].get_legend_handles_labels(), ncol=3, loc="outside lower center", fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(destination / f"nll_by_training_seed.{ext}", dpi=160)
    plt.close(fig)


def build(root):
    dest = root / "results/checkpoint2"
    artifacts = root / "assets/checkpoint2"
    for p in (dest / "baselines", dest / "confirmation", artifacts):
        p.mkdir(parents=True, exist_ok=True)
    audits = {ds: root / "results/dataset_audits" / f"{ds}_validation.json" for ds in DATASETS}
    audited = {ds: read(path) for ds, path in audits.items()}
    for ds, audit in audited.items():
        if audit["dataset"] != ds or audit["phase"] != "complete":
            raise ValueError(f"Incomplete or mismatched audit: {ds}")
    baseline, inputs = write_baselines(root, dest / "baselines", audited)
    pilot = write_pilot(dest / "pilot")
    status, confirmation, seed_inputs = write_confirmation(root, dest / "confirmation", audited, pilot)
    figures(artifacts, baseline, confirmation)
    inputs.update(seed_inputs)
    inputs.update({p.relative_to(root).as_posix(): digest(p) for p in audits.values()})
    inputs.update({p.relative_to(root).as_posix(): digest(p) for p in (dest / "pilot/source").rglob("*") if p.is_file()})
    outputs = {p.relative_to(root).as_posix(): digest(p) for folder in (dest, artifacts)
               for p in folder.rglob("*") if p.suffix in (".csv", ".png", ".svg") and "source" not in p.parts}
    manifest = {"evaluation": "development_validation", "final_test_used": False,
                "baseline_count": len(baseline), "confirmation_pairs_evaluated": int((status.status == "evaluated").sum()),
                "confirmation_pairs_target": len(status), "input_sha256": inputs, "output_sha256": outputs}
    write_json(manifest, dest / "manifest.json")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    report = build(args.root.resolve())
    print(f"{report['baseline_count']} baselines, {report['confirmation_pairs_evaluated']}/{report['confirmation_pairs_target']} training-seed pairs")
