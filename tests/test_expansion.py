import gc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from peft_methods import (method_settings, is_method_parameter, expected_method_parameters,
                          hash_pretrained_backbone, is_pretrained_backbone_parameter)
from prepare_expansion import DATASETS, ADDITIONAL_METHODS, generate
from initialization_reference import validate_initialization_reference
from run_expansion import ready, jobs, complete


class RecipeAndQueueTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("numpy"), "NumPy required for artifact validation")
    def test_completion_requires_matching_predictions_and_full_history(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            result = {"phase": "complete", "method": "bitfit", "dataset": "patch_camelyon",
                      "epochs": 100, "checkpoint_reload_exact": True,
                      "last_epoch_validation_top1_percent": 100}
            (root / "result.json").write_text(json.dumps(result))
            (root / "last.pt").write_bytes(b"synthetic-test-placeholder")
            (root / "history.csv").write_text("epoch\n" + "\n".join(map(str, range(1, 101))) + "\n")
            logits = np.tile([2., 0.], (200, 1))
            targets = np.zeros(200, dtype=np.int64)
            np.savez(root / "validation_last.npz", logits=logits, targets=targets)
            self.assertFalse(complete(root, "bitfit", "patch_camelyon"))
            np.savez(root / "validation_last_reloaded.npz", logits=logits, targets=targets)
            self.assertIs(complete(root, "bitfit", "patch_camelyon"), True)
            json.dumps({"complete": complete(root, "bitfit", "patch_camelyon")})
            np.savez(root / "validation_last_reloaded.npz", logits=logits+1, targets=targets)
            self.assertFalse(complete(root, "bitfit", "patch_camelyon"))

    def test_all_twenty_recipes_and_counts(self):
        manifest = json.loads((ROOT / "docs/PUBLISHED_RECIPES.json").read_text())
        self.assertEqual(len(manifest["records"]), 20)
        self.assertEqual({(r["dataset"], r["method"]) for r in manifest["records"]},
                         {(d, m) for d in DATASETS for m in ADDITIONAL_METHODS})
        for item in manifest["records"]:
            config = json.loads((ROOT / item["config"]).read_text())
            self.assertEqual(expected_method_parameters(config, item["method"]),
                             item["published_adaptation_parameters"])
            self.assertEqual(config["epochs"], 100)
            self.assertEqual(config["batch_size"], 64)
            self.assertFalse(config["test_set_access"])
        if (ROOT / "code/.git").exists():
            generate(check=True)

    def test_parameter_masks_separate_pretrained_biases_and_adapters(self):
        self.assertTrue(is_method_parameter("blocks.0.attn.qkv.bias", "bitfit"))
        self.assertFalse(is_method_parameter("head.bias", "bitfit"))
        self.assertFalse(is_method_parameter("blocks.0.norm1.weight", "bitfit"))
        self.assertTrue(is_pretrained_backbone_parameter("blocks.0.norm1.bias"))
        for name in ("blocks.0.attn.fact.FacTu.weight", "vpt.prompt_embeddings",
                     "blocks.0.ft_mlp_module.up_proj.weight"):
            self.assertFalse(is_pretrained_backbone_parameter(name))

    def test_common_initialization_does_not_require_identical_frozen_masks(self):
        ref = {"classifier_initial_sha256": "a" * 64, "frozen_backbone_sha256": "b" * 64,
               "pretrained_backbone_initial_sha256": "c" * 64}
        self.assertTrue(validate_initialization_reference(ref, "a" * 64, "d" * 64, "c" * 64))
        with self.assertRaises(ValueError):
            validate_initialization_reference(ref, "a" * 64, "d" * 64, "e" * 64)

    def test_queue_order_and_resource_gate(self):
        new = [j for j in jobs() if not j["legacy"]]
        self.assertEqual([(j["method"], j["dataset"]) for j in new],
                         [(m, d) for m in ADDITIONAL_METHODS for d in DATASETS])
        config = {"minimum_conservative_ram_headroom_gib": 3.25, "minimum_free_gpu_mib": 12000}
        state = {"conservative_headroom_gib": 4, "gpu_total_mib": 81920,
                 "gpu_used_mib": 30000, "gpu_utilization_percent": 100}
        self.assertFalse(ready(state, config))
        state["gpu_utilization_percent"] = 0
        self.assertTrue(ready(state, config))
        self.assertFalse(complete(ROOT / "runs/nonexistent", "bitfit", "dtd"))


@unittest.skipUnless(importlib.util.find_spec("torch") and (ROOT / "code/model").exists(),
                     "Requires the pinned training environment and upstream checkout")
class NativeIntegrationTests(unittest.TestCase):
    def test_native_counts_for_all_dataset_recipes(self):
        import torch
        import timm
        import yaml
        from dotwiz import DotWiz
        sys.path.insert(0, str(ROOT / "code"))
        import model.vision_transformer
        torch.set_num_threads(1)
        for method in ADDITIONAL_METHODS:
            for dataset, classes in DATASETS.items():
                with self.subTest(method=method, dataset=dataset):
                    config = json.loads((ROOT / f"configs/{dataset}_{method}.json").read_text())
                    params = DotWiz(yaml.safe_load((ROOT / "code/experiment/config/default_vtab_processed.yml").read_text()))
                    params.update(method_settings(config, method))
                    net = timm.create_model("vit_base_patch16_224_in21k_petl", pretrained=False, params=params)
                    net.reset_classifier(classes)
                    count = sum(p.numel() for n, p in net.named_parameters() if is_method_parameter(n, method))
                    self.assertEqual(count, config["published_adaptation_parameters"])
                    self.assertEqual(sum(p.numel() for p in net.head.parameters()), 769 * classes)
                    del net
                    gc.collect()

    def test_counts_gradient_prompt_length_shared_factors_and_reload(self):
        import torch
        import timm
        import yaml
        from dotwiz import DotWiz
        sys.path.insert(0, str(ROOT / "code"))
        import model.vision_transformer
        from smoke_test import hash_frozen
        torch.set_num_threads(1)
        pretrained_hashes = []
        for method in ADDITIONAL_METHODS:
            config = json.loads((ROOT / f"configs/patch_camelyon_{method}.json").read_text())
            params = DotWiz(yaml.safe_load((ROOT / "code/experiment/config/default_vtab_processed.yml").read_text()))
            params.update(method_settings(config, method))
            torch.manual_seed(42)
            net = timm.create_model("vit_base_patch16_224_in21k_petl", pretrained=False, params=params)
            weight_path = ROOT / "code/pretrained_weights/ViT-B_16_in21k.npz"
            if weight_path.is_file():
                net.load_pretrained(str(weight_path))
                pretrained_hashes.append(hash_pretrained_backbone(net))
            torch.manual_seed(42)
            net.reset_classifier(2)
            for name, value in net.named_parameters():
                value.requires_grad_(name.startswith("head.") or is_method_parameter(name, method))
            actual = sum(p.numel() for n, p in net.named_parameters() if is_method_parameter(n, method))
            self.assertEqual(actual, expected_method_parameters(config, method))
            frozen = hash_frozen(net)
            lengths, hooks = [], []
            if method == "vpt_deep":
                for block in net.blocks:
                    hooks.append(block.register_forward_pre_hook(lambda _, inputs: lengths.append(inputs[0].shape[1])))
            if method == "fact_tt":
                self.assertEqual(len({id(block.fact) for block in net.blocks}), 1)
                params_with_aliases = list(net.state_dict(keep_vars=True).items())
                shared_ids = [id(p) for n, p in params_with_aliases if is_method_parameter(n, method)]
                self.assertGreater(len(shared_ids), len(set(shared_ids)))
            net.train()
            image = torch.randn(1, 3, 224, 224)
            target = torch.tensor([1])
            loss = torch.nn.functional.cross_entropy(net(image), target)
            loss.backward()
            adaptation_grad = []
            for name, value in net.named_parameters():
                if value.requires_grad:
                    self.assertIsNotNone(value.grad, name)
                    self.assertTrue(torch.isfinite(value.grad).all(), name)
                    if is_method_parameter(name, method):
                        adaptation_grad.append(value.grad.abs().sum().item())
                else:
                    self.assertIsNone(value.grad, name)
            self.assertGreater(sum(adaptation_grad), 0)
            optimizer = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], lr=0.001)
            optimizer.step()
            self.assertEqual(hash_frozen(net), frozen)
            if method == "vpt_deep":
                self.assertEqual(lengths, [197 + config["vpt_num"]] * 12)
            for hook in hooks:
                hook.remove()
            net.eval()
            with torch.no_grad():
                before = net(image).clone()
                saved = {n: p.clone() for n, p in net.named_parameters() if p.requires_grad}
                for name, value in net.named_parameters():
                    if value.requires_grad:
                        value.zero_()
                        value.copy_(saved[name])
                after = net(image)
            self.assertTrue(torch.equal(before, after))
            del net, optimizer, saved, before, after, loss
            gc.collect()
        if pretrained_hashes:
            self.assertEqual(len(set(pretrained_hashes)), 1)


if __name__ == "__main__":
    unittest.main()
