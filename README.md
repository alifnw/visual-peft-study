# Visual PEFT Study

Kelompok 1: Alif Negifta Wibawaputra and Charleen Samaeri Gulo

Partial reproduction of [Mai et al., CVPR 2025](https://arxiv.org/abs/2409.16434), with experiments on probability calibration and confidence-based selective fusion for visual recognition.

The [Checkpoint 2 report in FILKOM format](reports/checkpoint2-filkom-report.pdf) summarizes the study, evaluation, and limitations.

- [Checkpoint 1: Flowers102](notebooks/01_flowers102_checkpoint1.ipynb) compares Linear, LoRA, and SSF under a fixed recipe.
- [Checkpoint 2: Calibration and Selective Fusion](notebooks/02_selective_fusion_checkpoint2.ipynb) covers 40 epoch-100 baseline runs across five datasets, pair screening on four datasets, and evaluation across training seeds 42, 43, and 44.

## Results

On Flowers102, all three Checkpoint 1 models classified 198 of 200 validation images correctly. LoRA and SSF achieved lower negative log-likelihood (NLL) and Brier scores than Linear despite identical accuracy. Twelve of the 102 classes had no validation samples.

Checkpoint 2 evaluates ConvPass followed by SSF on SVHN and ConvPass followed by Linear on PatchCamelyon. At a 25% target fallback rate, selective averaging matched always-on averaging in validation accuracy in all six seed-by-dataset comparisons and had lower NLL in five. For PatchCamelyon seed 43, ConvPass alone achieved 91.0% accuracy, compared with 89.5% for selective averaging.

With 20% corrupted training labels on SVHN, ConvPass accuracy fell from 90.0% to 83.0%, and SSF fell from 91.0% to 77.5%, on the same clean validation images. At a 25% target fallback rate, ConvPass followed by SSF reached 91.5% accuracy with clean training labels and 82.83% with noisy training labels. Under noise, it had slightly lower Brier and slightly higher NLL than ConvPass alone. The three-expert noisy ensemble remains unevaluated because the LoRA run did not complete. The paired comparison covers one training seed and one corruption mask.

On the frozen SVHN test of 26,032 images, selective fusion and always-on averaging both round to 91.68% mean accuracy across three training seeds. Selective makes 1.256 calls per image compared with 2.000 for always-on, with NLL of 0.2885 versus 0.2869. Matched random reaches 90.77% at the same call budget as selective. The PatchCamelyon mirror test was withheld after the split audit found an exact training overlap. The label-noise results and pair screening use development validation. Calls per image measure the number of forward passes.

[Baseline metrics](results/checkpoint2/baselines/metrics.csv) · [Results by training seed](results/checkpoint2/confirmation/metrics_by_training_seed.csv) · [SVHN test](results/checkpoint2/holdout/svhn-test-per-seed.csv) · [Label-noise routing results](results/label_noise/svhn/pairwise/strategy_summary.csv)

## Reproduce the analysis

Use Python 3.10 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/checkpoint2_analysis.py
python scripts/analyze_svhn_label_noise.py
python scripts/analyze_svhn_label_noise.py --pairwise
```

For interactive review, register the environment as a notebook kernel:

```bash
python -m ipykernel install --user --name visual-peft-cpu --display-name "Visual PEFT (CPU)"
```

Open either notebook in a Jupyter-compatible editor, select **Visual PEFT (CPU)**, and run all cells from the repository root or the `notebooks/` directory. Both notebooks use the saved predictions and artifacts in this repository. They require no GPU, dataset download, or model weights.

## Experimental protocols

- [Artifact guide](docs/CHECKPOINT2_ARTIFACTS.md): data, tables, figures, and provenance.
- [Training protocol](docs/TRAINING_PROTOCOL.md) and [multi-dataset recipes](docs/EXPANSION_PROTOCOL.md): backbone, data preparation, and epoch-100 runs.
- [Routing protocol](docs/FOUR_DATASET_PILOT.md): out-of-fold temperature fitting, fallback thresholds, and the matched-budget random control.

The experiments use the [authors' upstream code](https://github.com/OSU-MLB/ViT_PEFT_Vision/tree/cea7e21b2ddecef8172f5d177da5a29d531c4279) and the [VTAB-1K mirror](https://huggingface.co/datasets/XiN0919/VTAB-1k) at revision `a1c257fbbeebd3260c0a201fe6a6c9916b56eee7`. See the [reference metadata](docs/references.json) and [Flowers102 image attribution](assets/checkpoint1/ATTRIBUTION.md).
