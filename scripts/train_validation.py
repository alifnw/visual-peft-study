"""Train native upstream PEFT models with a fixed train-validation-only protocol."""

import argparse
import csv
import ctypes
import fcntl
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import time
import traceback

from smoke_test import hash_frozen, resource_state, save
from gradient_accumulation import accumulated_step
from resource_guard import snapshot as runtime_snapshot, stop_reason
from peft_methods import METHODS, method_settings, is_method_parameter, expected_method_parameters
from initialization_reference import load_reference_result, validate_initialization_reference
from audit_dtd import validate_audit_manifest


ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def memory_guard(initial_oom):
    state = runtime_snapshot()
    reason = stop_reason(state, initial_oom)
    if reason:
        raise RuntimeError(reason + ". Stopping this run.")


def run(config, method, output, initial, resume, resource_policy, reference_result=None, dataset_audit=None):
    import numpy as np
    import torch
    import timm
    import yaml
    from dotwiz import DotWiz

    sys.path.insert(0, str(ROOT / "code"))
    import model.vision_transformer
    from data.dataset.vtab_processed import get_processed_VTAB
    from engine.trainer import Trainer
    from timm.utils import accuracy

    def trim_host_memory(stage):
        if resource_policy.get("trim_host_allocator"):
            gc.collect()
            libc = ctypes.CDLL("libc.so.6")
            libc.malloc_trim.argtypes = [ctypes.c_size_t]
            libc.malloc_trim.restype = ctypes.c_int
            libc.malloc_trim(0)
        print(json.dumps({"loading_stage": stage,
                          "rss_peak_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}), flush=True)

    trim_host_memory("imports_complete")

    seed = config["seed"]
    torch.set_num_threads(1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.reset_peak_memory_stats()

    params = DotWiz(yaml.safe_load((ROOT / "code/experiment/config/default_vtab_processed.yml").read_text()))
    params.update({
        "data": "processed_vtab-" + config["dataset"],
        "data_path": str(ROOT / "data/vtab_processed"),
        "final_run": False,
        "device": 0,
        "patch_size": 16,
        "epoch": config["epochs"],
        "warmup_epoch": config["warmup_epochs"],
        "warmup_lr_init": config["warmup_learning_rate"],
        "lr_min": config["minimum_learning_rate"],
        "lr": config["learning_rate"],
        "wd": config["weight_decay"],
        "batch_size": config["batch_size"],
        "test_batch_size": config["validation_batch_size"],
        "drop_path_rate": config["drop_path_rate"],
        "normalized": config["normalized"],
        "random_seed": seed,
        "output_dir": str(output),
        "debug": False,
    })
    params.update(method_settings(config, method))
    train_data = get_processed_VTAB(params, "train")
    val_data = get_processed_VTAB(params, "val")
    assert len(train_data) == 800 and len(val_data) == 200
    if config.get("expected_class_count") is not None and params.class_num != config["expected_class_count"]:
        raise ValueError("Upstream classifier class count differs from the dataset configuration")
    assert not config["test_set_access"]
    smoke_check = config.get("smoke_check", False)
    if smoke_check:
        assert config["epochs"] == 2 and config["batch_size"] == 4
        train_data = torch.utils.data.Subset(train_data, range(4))
        val_data = torch.utils.data.Subset(val_data, range(4))
    sampler_generator = torch.Generator()
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=config["batch_size"],
        shuffle=True, drop_last=True, num_workers=0, pin_memory=False, generator=sampler_generator)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=config["validation_batch_size"],
        shuffle=False, num_workers=0, pin_memory=False)

    print(f"Building {method} using the upstream PETL model", flush=True)
    model = timm.create_model("vit_base_patch16_224_in21k_petl", pretrained=False,
        drop_path_rate=params.drop_path_rate, params=params)
    trim_host_memory("model_constructed")
    model.load_pretrained(str(ROOT / "code/pretrained_weights/ViT-B_16_in21k.npz"))
    trim_host_memory("pretrained_weights_loaded")
    torch.manual_seed(seed)
    model.reset_classifier(params.class_num)
    classifier_initial_hash = hashlib.sha256(b"".join(
        value.detach().cpu().numpy().tobytes() for value in model.head.parameters())).hexdigest()
    for name, value in model.named_parameters():
        value.requires_grad_(name.startswith("head.") or is_method_parameter(name, method))
    frozen_initial = hash_frozen(model)
    reference_matched = validate_initialization_reference(reference_result, classifier_initial_hash, frozen_initial)
    model = model.cuda()
    trim_host_memory("model_on_gpu")
    trainable = [p for p in model.parameters() if p.requires_grad]
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in trainable)
    trainable_names = [n for n, p in model.named_parameters() if p.requires_grad]
    expected_adapters = expected_method_parameters(config, method)
    assert sum(p.numel() for n, p in model.named_parameters()
               if p.requires_grad and is_method_parameter(n, method)) == expected_adapters
    assert trainable_parameters == expected_adapters + sum(p.numel() for p in model.head.parameters())
    initial_tuned = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}

    class GuardedTrainer(Trainer):
        def validate_gradients(self):
            memory_guard(initial["oom_kill_count"])
            for name, value in self.model.named_parameters():
                if value.requires_grad:
                    if value.grad is None or not torch.isfinite(value.grad).all():
                        raise RuntimeError(f"Missing or nonfinite gradient: {name}")
                elif value.grad is not None:
                    raise RuntimeError(f"Frozen backbone received a gradient: {name}")

        def forward_one_batch(self, samples, targets, is_train):
            memory_guard(initial["oom_kill_count"])
            if is_train and config.get("microbatch_size"):
                loss, outputs = accumulated_step(
                    self.model, self.cls_criterion, self.optimizer, samples, targets,
                    self.device, config["microbatch_size"],
                    before_microbatch=lambda: memory_guard(initial["oom_kill_count"]),
                    before_optimizer_step=self.validate_gradients)
                result = loss, outputs, accuracy(outputs, targets.cpu(), topk=(1, 5))
            else:
                result = super().forward_one_batch(samples, targets, is_train)
            loss = result[0]
            if isinstance(loss, int) or not torch.isfinite(loss):
                raise RuntimeError("The upstream trainer encountered a nonfinite batch loss")
            if is_train:
                self.validate_gradients()
            return result

    trainer = GuardedTrainer(model, trainable, params)

    def tuned_state():
        return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}

    def restore_tuned(state):
        assert set(state) == set(trainable_names)
        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(state[n].to(p.device))

    def save_checkpoint(path, epoch, best, history):
        payload = {"epoch": epoch, "trainable_state": tuned_state(),
            "optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.scheduler.state_dict(),
            "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(), "python_rng": random.getstate(),
            "best_validation": best, "history": history, "config": config, "method": method,
            "classifier_initial_sha256": classifier_initial_hash,
            "frozen_backbone_sha256": frozen_initial}
        temporary = path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        temporary.replace(path)

    start_epoch = 0
    history = []
    best = {"top1": -1, "epoch": None}
    if resume:
        checkpoint = torch.load(output / "last.pt", map_location="cpu")
        assert checkpoint["config"] == config and checkpoint["method"] == method
        assert checkpoint["classifier_initial_sha256"] == classifier_initial_hash
        assert checkpoint["frozen_backbone_sha256"] == frozen_initial
        restore_tuned(checkpoint["trainable_state"])
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        trainer.scheduler.load_state_dict(checkpoint["scheduler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        start_epoch = checkpoint["epoch"]
        history, best = checkpoint["history"], checkpoint["best_validation"]

    versions = {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "timm", "numpy", "Pillow")}
    provenance = {
        "dataset": config["dataset"], "class_count": params.class_num,
        "dataset_audit": dataset_audit,
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT / "code"), "rev-parse", "HEAD"], text=True).strip(),
        "tracked_source_diff": subprocess.check_output(["git", "-C", str(ROOT / "code"), "diff"], text=True),
        "python": sys.version,
        "versions": versions,
        "script_sha256": digest(Path(__file__)),
        "shared_helpers_sha256": digest(ROOT / "scripts/smoke_test.py"),
        "gradient_accumulation_source_sha256": digest(ROOT / "scripts/gradient_accumulation.py"),
        "resource_guard_source_sha256": digest(ROOT / "scripts/resource_guard.py"),
        "method_helpers_sha256": digest(ROOT / "scripts/peft_methods.py"),
        "method_settings": method_settings(config, method),
        "trainable_parameter_names": trainable_names,
        "smoke_check": smoke_check,
        "batching": {
            "effective_batch_size": config["batch_size"],
            "microbatch_size": config.get("microbatch_size", config["batch_size"]),
            "optimizer_updates_per_epoch": len(train_loader),
            "validation_batch_size": config["validation_batch_size"],
        },
        "native_source_sha256": {str(p.relative_to(ROOT)): digest(p) for p in [
            ROOT / "code/engine/trainer.py", ROOT / "code/engine/optimizer.py",
            ROOT / "code/model/vision_transformer.py", ROOT / "code/model/lora.py",
            ROOT / "code/model/ssf.py", ROOT / "code/model/convpass.py",
            ROOT / "code/model/block.py", ROOT / "code/model/attention.py",
            ROOT / "code/model/mlp.py", ROOT / "code/model/patch_embed.py",
            ROOT / "code/data/dataset/vtab_processed.py"]},
        "split_sha256": {name: digest(ROOT / "data/vtab_processed" / config["dataset"] / name)
                          for name in ("train800.txt", "val200.txt")},
        "weights_manifest": json.loads((ROOT / "manifests/pretrained_weights.json").read_text()),
        "frozen_backbone_sha256": frozen_initial,
        "classifier_initial_sha256": classifier_initial_hash,
        "initialization_reference": ({**reference_result, "matched": reference_matched}
                                     if reference_result is not None else None),
        "initial_resources": initial,
        "resource_policy": resource_policy,
        "protocol_differences": [
            "Fixed single configuration, no hyperparameter search and no final train+validation fit",
            "Python 3.10 instead of the setup script's Python 3.7",
            "Zero data-loader workers and no pinned CPU memory",
            f"Validation batch size {config['validation_batch_size']} instead of 512",
            f"Validation recorded after every epoch, fixed {config['epochs']}-epoch budget without early stopping",
            "Classifier reset with its own seed and independently seeded per-epoch sampler",
            "Native train_one_epoch, optimizer and cosine scheduler are reused",
            ("Training uses sample-weighted gradient accumulation, one optimizer update per effective batch. "
             "Dropout draws and numerical summation differ from direct full batches."
             if config.get("microbatch_size") else "Native forward_one_batch is reused"),
            "Best validation is supplementary, primary comparison uses the last epoch",
        ],
    }
    if resume:
        archived = output / f"provenance-before-resume-epoch{start_epoch}.json"
        if not archived.exists():
            shutil.copyfile(output / "provenance.json", archived)
        shutil.copyfile(__file__, output / f"train_validation.resume_epoch{start_epoch}.source.py")
    save(output / "provenance.json", provenance)
    shutil.copyfile(ROOT / "scripts/resource_guard.py", output / f"resource_guard.epoch{start_epoch}.source.py")
    if not resume:
        shutil.copyfile(__file__, output / "train_validation.source.py")
        shutil.copyfile(ROOT / "scripts/smoke_test.py", output / "smoke_test_helpers.source.py")
        shutil.copyfile(ROOT / "scripts/gradient_accumulation.py", output / "gradient_accumulation.source.py")
        shutil.copyfile(ROOT / "scripts/peft_methods.py", output / "peft_methods.source.py")
    started = time.monotonic()
    for epoch in range(start_epoch, config["epochs"]):
        epoch_started = time.monotonic()
        sampler_generator.manual_seed(seed + epoch)
        lr_used = trainer.optimizer.param_groups[0]["lr"]
        train_metrics = trainer.train_one_epoch(epoch, train_loader)
        val_metrics = trainer.eval_classifier(val_loader, "val")
        row = {"epoch": epoch + 1, "learning_rate": lr_used,
            **{"train_" + k: v for k, v in train_metrics.items()},
            **{"val_" + k: v for k, v in val_metrics.items()},
            "epoch_seconds": time.monotonic() - epoch_started}
        assert all(math.isfinite(float(v)) for v in row.values())
        history.append(row)
        is_best = val_metrics["top1"] > best["top1"]
        if is_best:
            best = {**val_metrics, "epoch": epoch + 1}
        trainer.scheduler.step(epoch)
        save_checkpoint(output / "last.pt", epoch + 1, best, history)
        if is_best:
            save_checkpoint(output / "best.pt", epoch + 1, best, history)
        with (output / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        status = {"phase": "training", "method": method, "epoch": epoch + 1,
            "total_epochs": config["epochs"], "last": row, "best_validation": best,
            "elapsed_this_process_seconds": time.monotonic() - started,
            "initial_resources": initial, "resource_policy": resource_policy}
        save(output / "status.json", status)
        print(json.dumps({"method": method, **row}), flush=True)

    assert hash_frozen(model) == frozen_initial
    changed_parameters = [n for n, p in model.named_parameters() if p.requires_grad
                          and not torch.equal(initial_tuned[n], p.detach().cpu())]
    assert any(n.startswith("head.") for n in changed_parameters), "Classifier did not update"
    assert method == "linear" or any(is_method_parameter(n, method) for n in changed_parameters), "PEFT weights did not update"
    del initial_tuned
    trainer.optimizer.zero_grad(set_to_none=True)

    def export_predictions(label):
        model.eval()
        predictions, target_values = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                memory_guard(initial["oom_kill_count"])
                predictions.append(model(images.cuda()).cpu().numpy())
                target_values.append(targets.numpy())
        logits = np.concatenate(predictions)
        targets = np.concatenate(target_values)
        assert logits.shape == (len(val_data), params.class_num) and np.isfinite(logits).all()
        np.savez_compressed(output / f"validation_{label}.npz", logits=logits, targets=targets)
        return float((logits.argmax(axis=1) == targets).mean() * 100)

    last_accuracy = export_predictions("last")
    assert abs(last_accuracy - history[-1]["val_top1"]) < 0.006
    last_state = torch.load(output / "last.pt", map_location="cpu")
    restore_tuned(last_state["trainable_state"])
    reloaded_last_accuracy = export_predictions("last_reloaded")
    before = np.load(output / "validation_last.npz")["logits"]
    after = np.load(output / "validation_last_reloaded.npz")["logits"]
    assert np.array_equal(before, after)
    assert reloaded_last_accuracy == last_accuracy
    best_state = torch.load(output / "best.pt", map_location="cpu")
    restore_tuned(best_state["trainable_state"])
    best_accuracy = export_predictions("best")
    assert abs(best_accuracy - best["top1"]) < 0.006
    final_resources = resource_state()
    result = {
        "phase": "complete", "method": method, "dataset": config["dataset"],
        "class_count": params.class_num,
        "purpose": config["purpose"], "seed": seed, "epochs": len(history),
        "smoke_check": smoke_check,
        **({"quality_score": None} if smoke_check else {}),
        "rows": {"train": len(train_data), "validation": len(val_data), "test_used": 0},
        "training_images_used_per_epoch": len(train_loader) * config["batch_size"],
        "drop_last": True, "total_parameters": total_parameters,
        "effective_batch_size": config["batch_size"],
        "microbatch_size": config.get("microbatch_size", config["batch_size"]),
        "optimizer_updates_per_epoch": len(train_loader),
        "trainable_parameters_including_head": trainable_parameters,
        "trainable_parameters_excluding_head": expected_adapters,
        "trainable_percent": trainable_parameters / total_parameters * 100,
        "primary_metric": None if smoke_check else "last_epoch_validation_top1_percent",
        "last_epoch_validation_top1_percent": last_accuracy,
        "best_validation_top1_percent": best_accuracy,
        "best_validation_epoch": best["epoch"],
        "final_train_metrics": {k: history[-1]["train_" + k] for k in ("loss", "top1", "top5")},
        "final_validation_metrics": {k: history[-1]["val_" + k] for k in ("loss", "top1", "top5")},
        "total_epoch_seconds": sum(row["epoch_seconds"] for row in history),
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "frozen_backbone_unchanged": True, "checkpoint_reload_exact": True,
        "classifier_parameters_updated": True,
        "method_parameters_updated": method != "linear",
        "classifier_initial_sha256": classifier_initial_hash,
        "frozen_backbone_sha256": frozen_initial,
        "new_oom_kills": final_resources["oom_kill_count"] - initial["oom_kill_count"],
        "final_resources": final_resources,
        "config": config,
        "resource_policy": resource_policy,
    }
    assert result["new_oom_kills"] == 0
    save(output / "result.json", result)
    save(output / "status.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/flowers102_stage3.json")
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-check", action="store_true",
                        help="Run two integration epochs on four development images without reporting a benchmark score.")
    parser.add_argument("--reference-result", type=Path,
                        help="Optional result JSON used to compare initial classifier and backbone hashes.")
    parser.add_argument("--skip-startup-ram-check", action="store_true",
                        help="Explicit override of the initial RAM gate only. Runtime guards remain enabled.")
    parser.add_argument("--trim-host-allocator", action="store_true",
                        help="Return freed host allocator pages at model-loading stage boundaries.")
    args = parser.parse_args()
    config_path = ROOT / args.config
    config = json.loads(config_path.read_text())
    if args.smoke_check:
        config.update(smoke_check=True, epochs=2, batch_size=4, microbatch_size=2,
                      validation_batch_size=4, warmup_epochs=0,
                      purpose="integration_smoke_test_not_benchmark")
    method_settings(config, args.method)
    dataset_audit = validate_audit_manifest(config, ROOT)
    reference_path = args.reference_result
    if reference_path is not None and not reference_path.is_absolute():
        reference_path = ROOT / reference_path
    reference_result = load_reference_result(reference_path)
    (ROOT / "runs").mkdir(exist_ok=True)
    lock = (ROOT / "runs/.training.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output = ROOT / args.output
    output = output.resolve()
    if not output.is_relative_to(ROOT / "runs"):
        raise ValueError("Output must be inside the project runs directory")
    output.mkdir(parents=True, exist_ok=args.resume)
    if args.resume:
        if (output / "result.json").exists():
            raise ValueError("Completed runs cannot be resumed")
        previous_config = json.loads((output / "config.json").read_text())
        if previous_config != config:
            raise ValueError("Resume configuration differs from saved configuration")
    else:
        save(output / "config.json", config)
    initial = resource_state()
    resource_policy = {
        "startup_ram_check_skipped": args.skip_startup_ram_check,
        "configured_startup_ram_headroom_gib": config["minimum_conservative_ram_headroom_gib"],
        "runtime_minimum_headroom_gib": 0.35,
        "runtime_oom_guard_enabled": True,
        "server_memory_limit_unchanged": True,
        "trim_host_allocator": args.trim_host_allocator,
        "runtime_cache_credit_fraction": 0.25,
        "runtime_cache_credit_cap_mib": 512,
        "runtime_resident_headroom_minimum_gib": 0.75,
        "runtime_full_pressure_maximum_avg10_percent": 10.0,
    }
    status = {"phase": "preflight", "method": args.method, "initial_resources": initial,
              "resource_policy": resource_policy}
    save(output / "status.json", status)
    if not args.skip_startup_ram_check and initial["conservative_headroom_gib"] < config["minimum_conservative_ram_headroom_gib"]:
        status.update(phase="blocked_resources", reason="Insufficient cgroup RAM headroom")
        save(output / "status.json", status)
        print(json.dumps(status, indent=2), flush=True)
        return 3
    if initial["gpu_total_mib"] - initial["gpu_used_mib"] < config["minimum_free_gpu_mib"] or initial["gpu_utilization_percent"] > 20:
        status.update(phase="blocked_resources", reason="GPU busy or insufficient free VRAM")
        save(output / "status.json", status)
        print(json.dumps(status, indent=2), flush=True)
        return 3
    try:
        memory_guard(initial["oom_kill_count"])
        run(config, args.method, output, initial, args.resume, resource_policy, reference_result, dataset_audit)
    except Exception as exc:
        status.update(phase="failed", error=repr(exc), traceback=traceback.format_exc())
        save(output / "status.json", status)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
