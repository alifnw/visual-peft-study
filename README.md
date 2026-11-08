# Visual PEFT Study

**Kelompok 1: Alif Negifta Wibawaputra, Charleen Samaeri Gulo**

Partial reproduction with exploratory analysis of [Lessons and Insights from a Unifying Study of Parameter-Efficient Fine-Tuning in Visual Recognition](https://arxiv.org/abs/2409.16434), CVPR 2025.

The [notebook](notebooks/01_flowers102_checkpoint1.ipynb) reports the research summary, related work, dataset audit, ViT and adaptation methods, training protocol, results, error analysis and proposed extensions.

Research question: under a fixed training configuration, how do Linear, LoRA and SSF differ in probability quality and prediction errors despite achieving the same validation accuracy?

## DTD baseline preparation

Four DTD recipes are available for Linear, LoRA, SSF, and ConvPass, using a new 47-class classifier. The data audit, configuration settings, ordered training commands, and output checks are documented in the [DTD protocol](docs/DTD_PROTOCOL.md). No completed DTD results have been recorded in this checkout.

## Flowers102: Checkpoint 1 progress

- Linear probing, LoRA and SSF completed 100 epochs on the same 800-image training and 200-image validation split.
- Stored logits, histories, initialization hashes and checkpoint-reload predictions were checked.
- CPU analysis includes macro recall, NLL, Brier, reliability plots, paired bootstrap and two ensemble aggregation rules.
- The report distinguishes CPU replay of stored predictions from model inference. Model checkpoint distribution and a new-image inference demo remain outside this checkpoint.
- Hyperparameter search, repeated training seeds, trainval retraining and final test evaluation remain pending.
- ConvPass stopped at epoch 92 and is excluded from the completed-method comparison.

## Configuration

The runs use supervised ImageNet-21K ViT-B/16, seed 42, AdamW, learning rate 0.001, weight decay 0.0001, effective batch 64, cosine decay, 10 warmup epochs and drop path 0.1. LoRA uses rank 8. LoRA and SSF use microbatch 8. All runs use FP32.

Learning rate, weight decay and rank are baseline choices within the official grid. They have not been tuned in this experiment. Matching settings does not establish equally well-tuned methods or an optimal parameter budget. Native LoRA updates Q, K and V through a shared low-rank projection. See [configuration notes](docs/CONFIGURATION_NOTES.md) and the [training protocol](docs/TRAINING_PROTOCOL.md).

## Validation results

All three methods correctly classify **198 of 200 images: 99% top-1 accuracy at epoch 100**.

- Linear: NLL 0.1162, multiclass Brier 0.03572, 78,438 trainable parameters.
- LoRA: NLL 0.04099, Brier 0.01373, 373,350 trainable parameters.
- SSF: NLL 0.03558, Brier 0.01345, 284,262 trainable parameters.

Parameter counts include the classifier. Brier sums over 102 classes before averaging over images. Macro recall averages only over the 90 classes observed in validation.

Uniform probability averaging and logit averaging both give 200/200 correct for Linear-LoRA and 199/200 for the other three combinations. Their NLL values differ. These are exploratory results on reused validation data.

Linear-LoRA probability averaging raises the correct count by two images but has worse NLL than single LoRA: 0.0734 versus 0.0410. Each model makes only two errors, so different error locations provide limited evidence for systematic complementarity.

LoRA and SSF have lower NLL and Brier than Linear. These scores do not isolate calibration or establish improved representations as the cause. Temperature-scaled Linear remains an untested control. SSF has the lowest single-model NLL point estimate, but the paired-bootstrap interval for SSF minus LoRA includes zero. The experiment does not establish an overall winner or an inference-efficiency benefit.

## Run the CPU notebook

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python -m ipykernel install --user --name peft-checkpoint1-cpu --display-name "PEFT Checkpoint 1 (CPU)"
```

Open the notebook, select **PEFT Checkpoint 1 (CPU)**, restart the kernel and Run All. Start from the repository root or the notebooks folder. Analysis reads local snapshots and does not start training or download data.

Metric figures and CSV tables are regenerated under assets/checkpoint1. Mermaid diagrams are loaded from validated local PNG/SVG files, with their editable .mmd sources included. The asset index records sources, producing sections, interpretation limits and hashes. Training code is separate and its GPU environment is documented in the training protocol.

CPU checks:

```bash
python -m unittest discover -s tests -p 'test_checkpoint*.py' -v
```

## Optional diagram rendering

The four diagrams use Mermaid CLI 12.0.0. Their sources and rendering configuration are under assets/checkpoint1/diagrams. Rebuilding requires Node.js 22.13 or newer and an installed Chromium browser. Normal notebook execution requires neither Node nor browser access.

```bash
PUPPETEER_SKIP_DOWNLOAD=1 npm ci --ignore-scripts
python scripts/render_diagrams.py
```

Use --chromium PATH if the browser is not on PATH. After editing a diagram, render it and rerun the notebook to refresh its outputs and asset index.

## Interpretation limits

This is a fixed-configuration, single-seed comparison on one VTAB-1K task. Twelve classes have no validation samples. Exact mirror split membership against the authors' archive is unverified.

Test metadata and sample files were checked during integrity preparation. Test data were not used for training, configuration selection or performance evaluation. Validation scores are not directly comparable with the paper's tuned and retrained test results. Runtime and memory logs come from shared-node conditions.

The CP1 research proposals include selective fusion, class-preserving weighting under controlled label noise, and resource-aware recipe selection. Their evaluation plans and literature context are retained in [research directions](docs/RESEARCH_DIRECTIONS.md). The extension direction is being reconsidered while DTD baselines are prepared. None of these interventions has been evaluated yet.

## Sources

- [Paper](https://arxiv.org/abs/2409.16434) and [official code](https://github.com/OSU-MLB/ViT_PEFT_Vision), commit cea7e21b2ddecef8172f5d177da5a29d531c4279.
- [Oxford Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and [VTAB-1K mirror](https://huggingface.co/datasets/XiN0919/VTAB-1k), revision a1c257fbbeebd3260c0a201fe6a6c9916b56eee7.
- [Image attribution](assets/checkpoint1/ATTRIBUTION.md), [reference metadata](docs/references.json) and [verification record](docs/CHECKPOINT1_VERIFICATION.json).
