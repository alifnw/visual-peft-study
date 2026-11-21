# Checkpoint 2 artifacts

[Checkpoint 1](../notebooks/01_flowers102_checkpoint1.ipynb) preserves the original Flowers102 analysis. [Checkpoint 2](../notebooks/02_selective_fusion_checkpoint2.ipynb) combines the baseline inventory, joint pair screening, and evaluation across training seeds.

| Location | Contents |
|---|---|
| `results/checkpoint1/` | Original Flowers102 Linear, LoRA, and SSF snapshots |
| `results/development/<dataset>/<method>/seed<N>/` | Completed epoch-100 logits, histories, result metadata, and provenance |
| `results/dataset_audits/` | Split hashes, ordered validation labels and paths, and duplicate audit |
| `results/checkpoint2/baselines/` | Forty seed-42 baseline results, recipes, and per-epoch learning curves |
| `results/checkpoint2/pilot/` | Joint calibration and routing tables, contributor manifests, and two independent cross-checks |
| `results/checkpoint2/confirmation/` | Pair coverage, per-seed metrics, calibration temperatures, thresholds, and per-image decisions |
| `results/label_noise/svhn/` | Corruption mask, completed noisy run snapshots, calibration metrics, and a matched ConvPass–SSF routing comparison |
| `assets/checkpoint2/` | PNG and SVG figures with hashes in the result manifest |

## Reproduce on CPU

From the repository root with the dependencies in `requirements-analysis.txt`:

```bash
python scripts/checkpoint2_analysis.py
python scripts/analyze_svhn_label_noise.py
python scripts/analyze_svhn_label_noise.py --pairwise
```

Open the Checkpoint 2 notebook from the repository root or the notebook folder and run all cells. The notebook reads the generated tables and figures. The builder verifies epoch, training seed, split hashes, label and image order, missing runs, and metadata accuracy against the logits. Model weights are needed for fresh inference.

## Contributions and source records

Alif Negifta Wibawaputra built the shared analysis pipeline and evaluated Linear/LoRA, Linear/SSF, and LoRA/SSF. Charleen Samaeri Gulo evaluated the pairs containing ConvPass. The contributor manifests identify the inputs and outputs for each contributor's experiments. The [consolidated manifest](../results/checkpoint2/manifest.json) records hashes for the files used to build the report.

Public run provenance uses a repository-relative location for the pretrained backbone. Weight, split, source, and configuration hashes retain their original values.

[Cross-check records](../results/checkpoint2/pilot/cross_checks.csv) compare per-image decisions against saved metrics for one configuration from each contributor.

## Evaluation scope

The baseline and pair screening use development validation at training seed 42. DTD statistics exclude the audited training duplicate. The repeated-seed evaluation freezes the SVHN and PatchCamelyon pairs, averaging, and the 25% target fallback. Each training seed is evaluated on three OOF calibration partitions. The three-seed summary is emitted for a dataset only after all three pairs have complete artifacts. Fallback counts measure second-expert calls.

The [SVHN test manifest](../results/checkpoint2/holdout/manifest.json) is separate from the development manifest. The test uses a fixed ConvPass → SSF pair and per-seed temperatures and thresholds fitted on all 200 validation images. The [per-seed results](../results/checkpoint2/holdout/svhn-test-per-seed.csv) cover all 26,032 test images. The [split audit](../results/checkpoint2/holdout/test-split-integrity.jsonl) records why PatchCamelyon test was withheld.
