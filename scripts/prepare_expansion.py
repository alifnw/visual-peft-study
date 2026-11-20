"""Generate dataset-specific recipes from the pinned authors' result records."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
COMMIT = "cea7e21b2ddecef8172f5d177da5a29d531c4279"
DATASETS = {"dtd": 47, "resisc45": 45, "svhn": 10, "patch_camelyon": 2, "oxford_flowers102": 102}
ADDITIONAL_METHODS = ("bitfit", "adaptformer", "vpt_deep", "fact_tt")


def generate(root=ROOT, check=False):
    root = Path(root)
    code = root / "code"
    actual = subprocess.check_output(["git", "-C", str(code), "rev-parse", "HEAD"], text=True).strip()
    if actual != COMMIT:
        raise ValueError("The upstream checkout does not match the pinned recipe source")
    base = json.loads((root / "configs/dtd_linear.json").read_text())
    default_path = code / "experiment/config/default_vtab_processed.yml"
    records = []
    for method in ADDITIONAL_METHODS:
        for dataset, classes in DATASETS.items():
            relative = f"tune_output/default_vtab/processed_vtab/{dataset}/{method}/final_result.json"
            source = code / relative
            raw = source.read_bytes()
            published = json.loads(raw)
            chosen = published["best_tune"]
            config = {**base, "experiment": f"{dataset}-{method}-seed42-published-recipe",
                      "dataset": dataset, "expected_class_count": classes, "methods": [method],
                      "dataset_audit_manifest": f"manifests/{dataset}_validation.json",
                      "require_all_classes_in_development": dataset != "oxford_flowers102",
                      "expected_duplicate_validation_rows": 1 if dataset == "dtd" else 0,
                      "learning_rate": chosen["lr"], "weight_decay": chosen["wd"],
                      "purpose": "single_seed_development_baseline_using_published_task_recipe",
                      "stage": "development", "schema_version": 2,
                      "recipe_source": {"commit": COMMIT, "path": relative,
                                        "sha256": hashlib.sha256(raw).hexdigest()},
                      "published_adaptation_parameters": published["inserted_parameters"]}
            if method == "vpt_deep":
                config["vpt_num"] = chosen["vpt_num"]
            elif method == "adaptformer":
                config["adaptformer"] = {"bottleneck": chosen["adapter_bottleneck"],
                                          "scale": chosen["adapter_scaler"]}
            elif method == "fact_tt":
                config["fact_tt"] = {"dim": chosen["fact_dim"], "scale": chosen["fact_scaler"]}
            path = root / "configs" / f"{dataset}_{method}.json"
            text = json.dumps(config, indent=2) + "\n"
            if check:
                if path.read_text() != text:
                    raise ValueError(f"Configuration differs from the pinned recipe: {path}")
            else:
                path.write_text(text)
            records.append({"dataset": dataset, "method": method,
                            "config": path.relative_to(root).as_posix(),
                            "config_sha256": hashlib.sha256(text.encode()).hexdigest(),
                            "source": config["recipe_source"], "best_tune": chosen,
                            "published_adaptation_parameters": published["inserted_parameters"]})
    manifest = {"schema_version": 2, "upstream_commit": COMMIT,
                "default_yaml_sha256": hashlib.sha256(default_path.read_bytes()).hexdigest(),
                "selection": "Published recipes, not locally repeated hyperparameter search",
                "records": records}
    output = root / "docs/PUBLISHED_RECIPES.json"
    text = json.dumps(manifest, indent=2) + "\n"
    if check:
        if output.read_text() != text:
            raise ValueError("Published recipe manifest is stale")
    else:
        output.parent.mkdir(exist_ok=True)
        output.write_text(text)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    manifest = generate(check=args.check)
    print(json.dumps({"recipes": len(manifest["records"]), "verified": args.check}))
