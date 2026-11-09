"""Bounded native-model integration check, deliberately not a benchmark experiment."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

from resource_guard import snapshot as memory_snapshot


ROOT = Path(__file__).resolve().parents[1]
GIB = 1024 ** 3


def save(path, content):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(content, indent=2) + "\n")
    temporary.replace(path)


def resource_state():
    memory = memory_snapshot()
    gpu_selector = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if not gpu_selector:
        raise RuntimeError("CUDA_VISIBLE_DEVICES does not expose a GPU")
    gpu = subprocess.check_output([
        "nvidia-smi", "--id=" + gpu_selector,
        "--query-gpu=memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"
    ], text=True).strip()
    gpu_used, gpu_total, gpu_utilization = [int(x.strip()) for x in gpu.splitlines()[0].split(",")]
    return {
        "cgroup_limit_bytes": memory["limit_bytes"],
        "cgroup_used_bytes": memory["used_bytes"],
        "inactive_file_bytes": memory["inactive_file_bytes"],
        "conservative_headroom_gib": memory["headroom_gib"],
        "oom_kill_count": memory["oom_kill_count"],
        "memory_scope": memory["scope"],
        "memory_limits_checked": memory["limits_checked"],
        "gpu_selector": gpu_selector,
        "gpu_used_mib": gpu_used,
        "gpu_total_mib": gpu_total,
        "gpu_utilization_percent": gpu_utilization,
    }


def hash_frozen(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if not value.requires_grad:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def run_check(args, output, initial):
    import numpy as np
    import torch
    import timm
    import yaml
    from dotwiz import DotWiz

    sys.path.insert(0, str(ROOT / "code"))
    import model.vision_transformer  # Registers the paper's native PETL backbone.
    from data.dataset.vtab_processed import get_processed_VTAB
    from engine.optimizer import make_optimizer

    torch.set_num_threads(1)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = False
    assert torch.cuda.is_available(), "CUDA is unavailable"
    torch.cuda.reset_peak_memory_stats()

    params = DotWiz(yaml.safe_load((ROOT / "code/experiment/config/default_vtab_processed.yml").read_text()))
    params.update({
        "data": "processed_vtab-oxford_flowers102",
        "data_path": str(ROOT / "data/vtab_processed"),
        "final_run": False,
        "patch_size": 16,
        "lr": 0.001,
        "wd": 0.0001,
        "device": 0,
        "lora_bottleneck": 8 if args.method == "lora" else 0,
        "batch_size": 2,
        "test_batch_size": 2,
    })
    train = get_processed_VTAB(params, "train")
    validation = get_processed_VTAB(params, "val")
    assert len(train) == 800 and len(validation) == 200
    train_subset = torch.utils.data.Subset(train, range(4))
    val_subset = torch.utils.data.Subset(validation, range(4))
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=2, shuffle=False, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=2, shuffle=False, num_workers=0)

    print(f"Building {args.method} with native ViT-B/16 and official ImageNet-21K weights", flush=True)
    model = timm.create_model("vit_base_patch16_224_in21k_petl", pretrained=False,
                              drop_path_rate=params.drop_path_rate, params=params)
    model.load_pretrained(str(ROOT / "code/pretrained_weights/ViT-B_16_in21k.npz"))
    model.reset_classifier(params.class_num)
    trainable_names = []
    for name, value in model.named_parameters():
        value.requires_grad_("head" in name or (args.method == "lora" and "lora" in name))
        if value.requires_grad:
            trainable_names.append(name)
    assert trainable_names and all("head" in n or "lora" in n for n in trainable_names)
    if args.method == "linear":
        assert all(n.startswith("head.") for n in trainable_names)
    else:
        assert len([n for n in trainable_names if "lora" in n]) == 24
    frozen_before = hash_frozen(model)
    model = model.cuda()
    trainable = [p for p in model.parameters() if p.requires_grad]
    first_trainable_before = {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}
    optimizer = make_optimizer(trainable, params)
    criterion = torch.nn.CrossEntropyLoss()
    losses = []
    gradient_counts = []
    lora_gradient_verified = False
    started = time.monotonic()
    model.train()
    for index, (images, labels) in enumerate(train_loader):
        images, labels = images.cuda(), labels.cuda()
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        assert tuple(logits.shape) == (2, 102)
        assert torch.isfinite(logits).all()
        loss = criterion(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        for name, value in model.named_parameters():
            if value.requires_grad:
                assert value.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(value.grad).all(), f"Nonfinite gradient for {name}"
                if "lora" in name and value.grad.abs().sum().item() > 0:
                    lora_gradient_verified = True
            else:
                assert value.grad is None, f"Frozen parameter has gradient: {name}"
        gradient_counts.append(sum(p.grad is not None for p in trainable))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        print(f"Step {index + 1}/2 loss={losses[-1]:.6f}", flush=True)
    assert args.method != "lora" or lora_gradient_verified
    changed = [name for name, p in model.named_parameters() if p.requires_grad and
               not torch.equal(first_trainable_before[name], p.detach().cpu())]
    assert any(n.startswith("head.") for n in changed)
    assert args.method != "lora" or any("lora" in n for n in changed)
    frozen_after = hash_frozen(model)
    assert frozen_before == frozen_after, "Frozen backbone changed"

    model.eval()
    all_outputs = []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.cuda())
            assert torch.isfinite(logits).all()
            all_outputs.append(logits.cpu())
    torch.save({name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}, output / "trainable_state.pt")
    restored = torch.load(output / "trainable_state.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        for name, value in model.named_parameters():
            if value.requires_grad:
                value.zero_()
                value.copy_(restored[name].to(value.device))
        images, _ = next(iter(val_loader))
        reload_logits = model(images.cuda()).cpu()
    assert torch.equal(all_outputs[0], reload_logits), "Reload changed predictions"
    torch.cuda.synchronize()
    final = resource_state()
    assert final["oom_kill_count"] == initial["oom_kill_count"], "OOM event occurred during the check"
    total_parameters = sum(p.numel() for p in model.parameters())
    tuned_parameters = sum(p.numel() for p in trainable)
    return {
        "phase": "passed",
        "purpose": "integration_smoke_test_not_benchmark",
        "quality_score": None,
        "dataset": params.data,
        "dataset_rows": {"train": len(train), "validation": len(validation)},
        "used_rows": {"train": 4, "validation": 4, "test": 0},
        "method": args.method,
        "steps": 2,
        "batch_size": 2,
        "seed": 42,
        "normalized": params.normalized,
        "drop_path_rate": params.drop_path_rate,
        "learning_rate": params.lr,
        "weight_decay": params.wd,
        "lora_rank": params.lora_bottleneck,
        "total_parameters": total_parameters,
        "trainable_parameters_including_head": tuned_parameters,
        "trainable_parameters_excluding_head": sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "head" not in n),
        "trainable_percent": tuned_parameters / total_parameters * 100,
        "trainable_names": trainable_names,
        "training_losses": losses,
        "finite_gradient_tensor_counts": gradient_counts,
        "head_updated": True,
        "lora_updated": any("lora" in n for n in changed),
        "frozen_backbone_unchanged": True,
        "frozen_backbone_sha256": frozen_after,
        "checkpoint_reload_exact": True,
        "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "peak_gpu_reserved_mib": torch.cuda.max_memory_reserved() / 1024 ** 2,
        "peak_process_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "step_and_verification_seconds": time.monotonic() - started,
        "gpu": torch.cuda.get_device_name(0),
        "versions": {name: importlib.metadata.version(name) for name in ("torch", "torchvision", "timm", "numpy", "Pillow")},
        "source_commit": subprocess.check_output(["git", "-C", str(ROOT / "code"), "rev-parse", "HEAD"], text=True).strip(),
        "source_dirty": subprocess.check_output(["git", "-C", str(ROOT / "code"), "status", "--short"], text=True).strip(),
        "weights_manifest": json.loads((ROOT / "manifests/pretrained_weights.json").read_text()),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "initial_resources": initial,
        "final_resources": final,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["linear", "lora"], required=True)
    args = parser.parse_args()
    output = ROOT / "runs" / (time.strftime("%Y%m%dT%H%M%S") + "-smoke-" + args.method)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "result.json"
    initial = resource_state()
    status = {"phase": "preflight", "method": args.method, "quality_score": None,
              "purpose": "integration_smoke_test_not_benchmark", "initial_resources": initial}
    save(path, status)
    print(f"Output: {output}", flush=True)
    headroom = initial["conservative_headroom_gib"]
    if headroom is not None and headroom < 2.5:
        status.update(phase="blocked_resources", reason="Need 2.5 GiB conservative cgroup RAM headroom before importing the training stack")
        save(path, status)
        print(json.dumps(status, indent=2), flush=True)
        return 3
    if initial["gpu_total_mib"] - initial["gpu_used_mib"] < 6000 or initial["gpu_utilization_percent"] > 20:
        status.update(phase="blocked_resources", reason="GPU is busy or has less than 6 GiB free")
        save(path, status)
        print(json.dumps(status, indent=2), flush=True)
        return 3
    try:
        status = run_check(args, output, initial)
        save(path, status)
        print(json.dumps(status, indent=2), flush=True)
    except Exception as exc:
        status.update(phase="failed", error=repr(exc), traceback=traceback.format_exc())
        save(path, status)
        raise
    return 0


if __name__ == "__main__":
    sys.exit(main())
