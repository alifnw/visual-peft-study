# Multi-dataset development baselines

This is a partial reproduction of Mai et al., CVPR 2025, using published recipes. It does not repeat the authors' hyperparameter search. The research question is how adaptation mechanisms differ in accuracy, probability quality, and errors across visual domains with limited training data.

## Experimental design

Datasets: Oxford Flowers102, DTD, RESISC45, SVHN, and PatchCamelyon. Additional methods: BitFit, AdaptFormer, VPT-Deep, and FacT-TT. Existing Linear, LoRA, SSF, and ConvPass results retain their original configurations and provenance. Linear is the classifier-only baseline, giving seven PEFT methods and eight approaches in total.

All additional development runs use 800 training images, 200 validation images, seed 42, 100 epochs, supervised ImageNet-21K ViT-B/16, AdamW, cosine decay, 10 warmup epochs, drop path 0.1, FP32, effective batch 64, microbatch 8, and zero loader workers. Resize is bicubic 224 x 224 followed by ToTensor. Normalization and augmentation are disabled. With upstream drop_last=True, each shuffled epoch uses 768 images and 12 optimizer updates. Stochastic gradient accumulation is not asserted to reproduce a direct-batch trajectory exactly.

Learning rate, weight decay, and method size/scale come from each dataset-method best_tune record at upstream commit cea7e21b2ddecef8172f5d177da5a29d531c4279. The static source, hashes, and settings are in [PUBLISHED_RECIPES.json](PUBLISHED_RECIPES.json). The pinned upstream defaults specify remaining architecture and initialization settings. We preserve constructor behavior rather than replacing upstream initialization routines. Local performance is not assumed to match the published result.

BitFit tunes pretrained biases and the classifier. AdaptFormer uses the upstream parallel FFN adapter with lora_kaiming initialization. VPT-Deep replaces layer-specific prompt tokens at each block. FacT-TT reuses the upstream shared factor module. Unique trainable tensors are counted and checkpointed once. Parameter counts always distinguish the adaptation parameters from the classifier.

## Data integrity

The pinned mirror archive is checksum verified before extracting missing tasks. Audits decode all development images, check label bounds and split hashes, compare trainval content with the train/validation union, and check exact development duplicates. Test split metadata is read for path overlap checks. Test images are not decoded and no test predictions are generated.

DTD has one known training-validation duplicate. Preserve the original 200-row validation and also report the 199-row subset without that training image. Unexpected duplicates or conflicting labels stop preparation. Flowers102 validation class coverage is reported rather than assumed complete. PatchCamelyon patient/slide separation remains unverified unless supporting grouping metadata is found and checked. Mirror split identity with the authors' seed remains unverified.

```bash
.venv/bin/python scripts/prepare_expansion.py --check
.venv/bin/python scripts/audit_expansion.py --prepare-missing
.venv/bin/python -m unittest discover -s tests -p 'test_expansion*.py' -v
```

## Training

```bash
python scripts/train_validation.py --config configs/svhn_bitfit.json \
  --method bitfit --output runs/development/svhn/bitfit/seed42
```

Use the matching file in `configs/` for each dataset and method. Use a new output directory for a fresh run. Resume requires the original configuration, split hashes, initialization, and runner version. Integration smoke checks use two epochs and four images and do not contribute benchmark scores.

## Evaluation

Epoch-100 validation accuracy is primary. Best-validation checkpoints are supplementary. Reports include stable log-softmax NLL, multiclass Brier summed across classes then averaged across images, recall per class, macro recall over observed classes, confidence bins with counts, learning curves, errors, and prediction disagreement. Missing-class recall is unavailable.

Accuracy uses a Wilson 95% interval. Paired NLL/Brier differences use 10,000 image resamples, seed 42, and percentile 95% intervals. Paired analysis requires matching validation split hashes and label order. These are descriptive intervals on reused development data, without correction for the multiple exploratory comparisons. Lower NLL/Brier alone does not establish calibration or the mechanism causing a change. Raw scores are compared within a dataset. Any cross-dataset aggregate gives equal weight to each dataset.

Resource profiles distinguish process RAM/VRAM from whole-device utilization and separate queue waiting from child runtime. Shared-node observations do not establish intrinsic efficiency rankings. Original recipe budgets differ, so these runs do not isolate parameter capacity or a specific architecture component.

```bash
python scripts/expansion_analysis.py
```

The baseline section of the [Checkpoint 2 notebook](../notebooks/02_selective_fusion_checkpoint2.ipynb) reads stored artifacts. The consolidated export uses [baseline CSVs](../results/checkpoint2/baselines/) and [figures](../assets/checkpoint2/). See the [artifact guide](CHECKPOINT2_ARTIFACTS.md) for CPU reproduction.
