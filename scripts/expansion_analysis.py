"""Replay completed development predictions without importing the training stack."""

import argparse
from itertools import combinations
import json
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd

METHOD_LABELS = {"linear": "Linear", "lora": "LoRA", "ssf": "SSF", "convpass": "ConvPass",
                 "bitfit": "BitFit", "adaptformer": "AdaptFormer", "vpt_deep": "VPT-Deep", "fact_tt": "FacT-TT"}


def prediction_metrics(logits, targets):
    logits, targets = np.asarray(logits, dtype=np.float64), np.asarray(targets)
    if logits.ndim != 2 or not len(targets) or logits.shape[0] != len(targets):
        raise ValueError("Predictions must be a nonempty N x C array aligned with targets")
    if targets.dtype.kind not in "iu" or not np.isfinite(logits).all() or targets.min() < 0 or targets.max() >= logits.shape[1]:
        raise ValueError("Nonfinite logits or invalid integer labels")
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_prob = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    probabilities = np.exp(log_prob)
    nll = -log_prob[np.arange(len(targets)), targets]
    brier = (probabilities ** 2).sum(axis=1) - 2 * probabilities[np.arange(len(targets)), targets] + 1
    predicted = logits.argmax(axis=1)
    correct = predicted == targets
    classes = np.arange(logits.shape[1])
    recall = np.array([correct[targets == c].mean() if (targets == c).any() else np.nan for c in classes])
    n, p = len(targets), correct.mean()
    z = NormalDist().inv_cdf(0.975)
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"accuracy_percent": float(100*p), "nll": float(nll.mean()), "brier": float(brier.mean()),
            "macro_recall_observed": float(np.nanmean(recall)), "classes_observed": int(np.isfinite(recall).sum()),
            "accuracy_ci95_low": float(100*(center-half)), "accuracy_ci95_high": float(100*(center+half)),
            "n": n, "nll_samples": nll, "brier_samples": brier, "recall": recall,
            "confidence": probabilities.max(axis=1), "predicted": predicted, "correct": correct}


def paired_interval(first, second, seed=42, repeats=10000):
    delta = np.asarray(first) - np.asarray(second)
    if delta.ndim != 1 or not len(delta) or not np.isfinite(delta).all():
        raise ValueError("Paired losses must be finite and aligned")
    rng = np.random.default_rng(seed)
    means = np.concatenate([delta[rng.integers(0, len(delta), size=(min(500, repeats-i), len(delta)))].mean(axis=1)
                            for i in range(0, repeats, 500)])
    low, high = np.quantile(means, [0.025, 0.975])
    return {"difference": float(delta.mean()), "ci95_low": float(low), "ci95_high": float(high),
            "bootstrap_repeats": repeats, "bootstrap_seed": seed}


def collect(root):
    root = Path(root)
    records = []
    for parent in (root / "results", root / "runs"):
        if not parent.exists():
            continue
        for path in sorted(parent.rglob("result.json")):
            if any(part in ("attempts", "_attempts") for part in path.parts):
                continue
            value = json.loads(path.read_text())
            if value.get("phase") != "complete" or value.get("smoke_check") or value.get("epochs") != 100:
                continue
            if value.get("method") not in METHOD_LABELS or value.get("stage", "development") != "development":
                continue
            output = path.parent
            prediction = output / "validation_last.npz"
            if not prediction.is_file():
                continue
            with np.load(prediction, allow_pickle=False) as stored:
                logits, targets = stored["logits"].copy(), stored["targets"].copy()
                image_paths = stored["image_paths"].copy() if "image_paths" in stored else None
            metrics = prediction_metrics(logits, targets)
            if abs(metrics["accuracy_percent"] - value["last_epoch_validation_top1_percent"]) > 1e-6:
                raise ValueError(f"Recorded accuracy differs from stored predictions: {path}")
            provenance_path = output / "provenance.json"
            provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
            split_hash = provenance.get("split_sha256", {}).get("val200.txt")
            audit_file = root / "results/dataset_audits" / f"{value['dataset']}_validation.json"
            if image_paths is None and audit_file.exists():
                audit = json.loads(audit_file.read_text())
                if audit["splits"]["val200.txt"]["list_sha256"] == split_hash:
                    rows = audit.get("development_records", {}).get("val200.txt", [])
                    if rows:
                        if not np.array_equal(np.asarray([r["label"] for r in rows]), targets):
                            raise ValueError("Audit labels differ from stored targets")
                        image_paths = np.asarray([r["path"] for r in rows])
            records.append({"result": value, "folder": output, "metrics": metrics, "targets": targets,
                            "paths": image_paths, "split_hash": split_hash})
    unique = {}
    for record in records:
        value = record["result"]
        key = (value["dataset"], value["method"], value.get("seed"), record["split_hash"],
               json.dumps(value.get("config", {}), sort_keys=True))
        if key in unique:
            if not np.array_equal(unique[key]["metrics"]["nll_samples"], record["metrics"]["nll_samples"]):
                raise ValueError("Duplicate run identity has different predictions")
            continue
        unique[key] = record
    return list(unique.values())


def export(root, output=None):
    root = Path(root)
    output = Path(output) if output else root / "assets/expansion"
    output.mkdir(parents=True, exist_ok=True)
    records = collect(root)
    rows, bins, recalls, errors, pair_rows = [], [], [], [], []
    array_keys = ("nll_samples", "brier_samples", "recall", "confidence", "predicted", "correct")
    for record in records:
        result, metric = record["result"], record["metrics"]
        identity = {"dataset": result["dataset"], "method": result["method"], "seed": result.get("seed", 42),
                    "run": str(record["folder"].relative_to(root))}
        rows.append({**identity, **{k: v for k, v in metric.items() if k not in array_keys},
                     "evaluation_subset": "original_validation",
                     "parameters_excluding_head": result["trainable_parameters_excluding_head"],
                     "parameters_including_head": result["trainable_parameters_including_head"],
                     "observed_total_epoch_minutes": result["total_epoch_seconds"] / 60,
                     "observed_mean_epoch_seconds": result["total_epoch_seconds"] / result["epochs"],
                     "image_order_verified": record["paths"] is not None})
        clean_prediction = record["folder"] / "validation_last_without_training_duplicates.npz"
        if result["dataset"] == "dtd" and clean_prediction.exists():
            with np.load(clean_prediction, allow_pickle=False) as stored:
                clean_metric = prediction_metrics(stored["logits"], stored["targets"])
            rows.append({**rows[-1], **{k: v for k, v in clean_metric.items() if k not in array_keys},
                         "evaluation_subset": "without_training_duplicates"})
        for c, recall in enumerate(metric["recall"]):
            recalls.append({**identity, "class": c, "recall": recall,
                            "support": int((record["targets"] == c).sum())})
        indices = np.minimum((metric["confidence"] * 10).astype(int), 9)
        for index in range(10):
            selected = indices == index
            bins.append({**identity, "bin": index, "count": int(selected.sum()),
                         "confidence": float(metric["confidence"][selected].mean()) if selected.any() else np.nan,
                         "accuracy": float(metric["correct"][selected].mean()) if selected.any() else np.nan})
        for index in np.flatnonzero(~metric["correct"]):
            errors.append({**identity, "row": int(index), "actual": int(record["targets"][index]),
                           "predicted": int(metric["predicted"][index]),
                           "confidence": float(metric["confidence"][index]),
                           "image": str(record["paths"][index]) if record["paths"] is not None else "unverified"})
    for a, b in combinations(records, 2):
        x, y = a["result"], b["result"]
        if x["dataset"] != y["dataset"] or x.get("seed") != y.get("seed") or x["method"] == y["method"]:
            continue
        aligned = (a["split_hash"] is not None and a["split_hash"] == b["split_hash"]
                   and np.array_equal(a["targets"], b["targets"]))
        if not aligned:
            continue
        for loss in ("nll", "brier"):
            pair_rows.append({"dataset": x["dataset"], "seed": x.get("seed"), "first": x["method"],
                              "second": y["method"], "metric": loss,
                              **paired_interval(a["metrics"][loss+"_samples"], b["metrics"][loss+"_samples"]),
                              "disagreement_count": int((a["metrics"]["predicted"] != b["metrics"]["predicted"]).sum()),
                              "shared_error_count": int((~a["metrics"]["correct"] & ~b["metrics"]["correct"]).sum())})
    tables = {"metrics": rows, "reliability": bins, "class_recall": recalls, "errors": errors,
              "paired_differences": pair_rows}
    for name, values in tables.items():
        pd.DataFrame(values).to_csv(output / f"{name}.csv", index=False)
    status = {"completed_runs": len(records), "additional_seed42_target": 20,
              "additional_completed": sum(r["result"]["method"] in ("bitfit", "adaptformer", "vpt_deep", "fact_tt")
                                          and r["result"].get("seed") == 42 for r in records),
              "evaluation": "Reused development validation, no final test evaluation",
              "brier_definition": "Mean over images of the sum of squared class probability errors",
              "bootstrap": {"repeats": 10000, "seed": 42, "interval": "95% percentile, descriptive"}}
    (output / "manifest.json").write_text(json.dumps(status, indent=2) + "\n")
    if rows:
        import matplotlib.pyplot as plt
        frame = pd.DataFrame(rows)
        for dataset, group in frame.groupby("dataset", sort=True):
            group = group[group["evaluation_subset"] == "original_validation"]
            fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
            for axis, column in zip(axes, ("accuracy_percent", "nll", "brier")):
                axis.bar(group["method"], group[column])
                axis.set_title(column)
                axis.tick_params(axis="x", labelrotation=60)
            fig.suptitle(dataset + ": epoch-100 development predictions")
            fig.tight_layout()
            for extension in ("png", "svg"):
                fig.savefig(output / f"{dataset}-metrics.{extension}", dpi=250, bbox_inches="tight")
            plt.close(fig)
            selected = [r for r in records if r["result"]["dataset"] == dataset]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for record in selected:
                method = record["result"]["method"]
                history = record["folder"] / "history.csv"
                if history.exists():
                    curve = pd.read_csv(history)
                    axes[0].plot(curve["epoch"], curve["val_top1"], label=method)
                metric = record["metrics"]
                indices = np.minimum((metric["confidence"] * 10).astype(int), 9)
                conf, acc, sizes = [], [], []
                for i in range(10):
                    mask = indices == i
                    if mask.any():
                        conf.append(metric["confidence"][mask].mean())
                        acc.append(metric["correct"][mask].mean())
                        sizes.append(mask.sum())
                axes[1].scatter(conf, acc, s=np.asarray(sizes) + 10, label=method, alpha=0.7)
            axes[0].set(xlabel="Epoch", ylabel="Validation accuracy (%)")
            axes[1].plot([0, 1], [0, 1], color="gray", linestyle="--")
            axes[1].set(xlabel="Mean confidence", ylabel="Empirical accuracy", xlim=(0, 1), ylim=(0, 1))
            axes[0].legend(fontsize=8)
            axes[1].legend(fontsize=8)
            fig.suptitle(dataset + ": learning curves and reliability (bin counts in CSV)")
            fig.tight_layout()
            for extension in ("png", "svg"):
                fig.savefig(output / f"{dataset}-diagnostics.{extension}", dpi=250, bbox_inches="tight")
            plt.close(fig)
    return pd.DataFrame(rows), status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    _, status = export(args.root, args.output)
    print(json.dumps(status, indent=2))
