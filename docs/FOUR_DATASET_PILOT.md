# Four-dataset calibration and selective-routing pilot

This analysis reads saved epoch-100 development logits for Linear, LoRA, SSF, and ConvPass. It runs on CPU and does not load model checkpoints, read image datasets, launch training, or evaluate final test data.

## Inputs and protocol

The input snapshots are under `results/development/<dataset>/<method>/seed42/`. Dataset audits are under `results/dataset_audits/`. Missing or inconsistent inputs stop analysis instead of triggering prediction generation.

DTD uses the audited 199-image subset without the known training duplicate. RESISC45, SVHN, and PatchCamelyon each use 200 validation images. Run identity, split hash, target order, initialization hashes, and recorded accuracy are checked before analysis.

Temperature scaling uses five folds and partition seeds 42, 43, and 44. Each fold fits one temperature per method on the other four folds, minimizing NLL within 0.05 to 20. Calibration and scoring are disjoint within each fold. Positive scalar temperature preserves accuracy. Partition seeds are not training seeds.

All six unordered pairs can be evaluated in both directions. Low first-model confidence triggers fallback. Thresholds are derived from calibration confidence for target rates 25%, 50%, and 75%. Replacement uses the second model's calibrated probabilities. Averaging uses equal-weight calibrated probabilities. Both rules use the same gate.

Random routing uses 20 seeds and exactly matches the gate's realized fallback count in each evaluation fold. It is a matched-budget diagnostic using evaluation batch counts. The original pilot used a different Bernoulli random comparator and remains separately archived.

## Run

From the repository root with the analysis environment:

```bash
.venv/bin/python scripts/calibration_routing_pilot.py \
  --datasets dtd resisc45 svhn patch_camelyon \
  --pairs all \
  --fallback-rules replacement averaging \
  --output results/checkpoint2/pilot/replay
```

Specify directed pairs to run a subset. For example, `--pairs lora:ssf ssf:lora`. An existing output directory is reusable only for the same dataset, pair and rule configuration. Use a new output folder for another configuration.

The original two-dataset invocation without new arguments retains the initial Linear-to-LoRA pilot behavior. Do not use its historical output folder for the expanded protocol.

## Outputs

The output contains calibration scores, fold temperatures, reliability bins with counts, fold assignments, class coverage, calibrated OOF probability arrays, routing scores, per-image gate decisions, oracle diagnostics, PNG/SVG figures, and a generated brief. The manifest records source/input hashes, package versions, protocol configuration, and checks.

`routing_summary.csv` aggregates repeated partitions and random seeds. Split SD describes partition sensitivity. Random SD also includes routing randomness. Neither represents uncertainty across independent model training seeds.

Oracle accuracy selects between the two single-model predictions using labels. It is a diagnostic upper bound for replacement on that pair. Averaging can predict another class and is not bounded by that oracle.

The pilot section of the [Checkpoint 2 notebook](../notebooks/02_selective_fusion_checkpoint2.ipynb) reads the verified joint results. The [artifact guide](CHECKPOINT2_ARTIFACTS.md) describes the contributor snapshots and the separate evaluation across training seeds.

## Interpretation

All pair, rule, direction, and budget comparisons use development validation previously seen during model development. Report the full tested scope and treat the selected configuration as a hypothesis for independent evaluation. NLL/Brier comparisons should be made within each dataset because class counts differ. Reliability plots are descriptive, with small bin counts.

Fallback is an additional-model invocation rate. The first model executes a full ViT and fallback executes another. No latency or inference-efficiency benefit is established by these CPU analyses.
