# DTD development baselines

These are new DTD runs. The completed Flowers102 runs remain the CP1 record.

Use the existing pinned training environment and upstream commit described in [the training setup](TRAINING_PROTOCOL.md). DTD uses 800 training images, 200 validation images, and a new 47-class classifier. Initialize every method from the same supervised ImageNet-21K ViT-B/16 checkpoint.

All runs use seed 42, 100 epochs, AdamW, cosine decay, 10 warmup epochs, drop path 0.1, FP32, effective batch 64, microbatch 8, validation batch 8, and zero loader workers. Preprocessing remains the upstream resize to 224 x 224, ToTensor, no normalization, and no augmentation. Each epoch uses 768 images because the upstream loader drops the incomplete effective batch.

## Task-specific recipes

The learning rate, weight decay, LoRA rank, and ConvPass settings come from the [published DTD configurations](https://github.com/OSU-MLB/ViT_PEFT_Vision/tree/cea7e21b2ddecef8172f5d177da5a29d531c4279/tune_output/default_vtab/processed_vtab/dtd). These initial runs do not repeat the authors' hyperparameter search or trainval retraining.

- Linear: learning rate 0.001, weight decay 0.0001.
- LoRA: learning rate 0.001, weight decay 0.0001, rank 1.
- SSF: learning rate 0.01, weight decay 0.001.
- ConvPass: learning rate 0.01, weight decay 0.001, bottleneck 8, scale 1, Xavier initialization disabled.

Adapter counts exclude the common 36,143-parameter classifier: Linear 0, LoRA 36,864, SSF 205,824, and ConvPass 327,552. The recipes use different adaptation budgets.

## Audit existing data

Reuse the prepared VTAB directory and pretrained checkpoint on the training node. The audit checks split metadata, decodes every training and validation image, and rejects identical training-validation image content. It reads test split metadata for overlap checks and does not open test images or generate test predictions.

```bash
.venv/bin/python scripts/audit_dtd.py
```

The audit writes manifests/dtd_validation.json. The runner checks the audit, current split hashes, and the 47-class classifier before training. If VTAB has not been prepared, follow the download procedure in the training setup first. The community mirror's exact split identity with the authors' archive remains unverified.

## DTD smoke tests

These checks use the DTD configs and a 47-class classifier. They run two epochs on four training and four validation images, with effective batch 4 and microbatch 2, and exercise checkpoint export and reload. Their quality_score and primary_metric are null.

```bash
.venv/bin/python scripts/train_validation.py --method lora --config configs/dtd_lora.json --output runs/dtd/smoke-lora --smoke-check --trim-host-allocator
.venv/bin/python scripts/train_validation.py --method ssf --config configs/dtd_ssf.json --output runs/dtd/smoke-ssf --smoke-check --trim-host-allocator
```

## Full runs

Run one training process at a time in this order. Linear establishes the reference initialization for the remaining methods. Use the same checkout and dataset directory so the training lock covers all four runs.

```bash
.venv/bin/python scripts/train_validation.py --method linear --config configs/dtd_linear.json --output runs/dtd/linear --trim-host-allocator
.venv/bin/python scripts/train_validation.py --method lora --config configs/dtd_lora.json --output runs/dtd/lora --reference-result runs/dtd/linear/result.json --trim-host-allocator
.venv/bin/python scripts/train_validation.py --method ssf --config configs/dtd_ssf.json --output runs/dtd/ssf --reference-result runs/dtd/linear/result.json --trim-host-allocator
.venv/bin/python scripts/train_validation.py --method convpass --config configs/dtd_convpass.json --output runs/dtd/convpass --reference-result runs/dtd/linear/result.json --trim-host-allocator
```

The runner saves status.json, history.csv, last.pt, best.pt, provenance.json, and validation logits. A completed run has phase complete, 100 epochs, zero test images used, and validation_last.npz with logits shaped 200 x 47. Epoch 100 is the primary checkpoint. Best-validation checkpoints remain diagnostic.

Append --resume to the same command after an interrupted run with last.pt. Preserve the original config. Startup and runtime resource guards remain enabled.

## Evaluation outputs

Record accuracy, NLL, multiclass Brier, learning curves, and error analysis for each completed DTD run. Keep the outputs separate from results/checkpoint1 and retain the model checkpoints on the training node.

Final test evaluation, repeated seeds, and hyperparameter search are outside this baseline stage. The DTD training status must be recorded from live run outputs.
