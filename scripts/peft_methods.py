"""Explicit method settings and parameter boundaries for the native paper models."""

METHODS = ("linear", "lora", "ssf", "convpass", "vpt_deep", "adaptformer", "bitfit", "fact_tt")


def method_settings(config, method):
    if method not in METHODS or method not in config["methods"]:
        raise ValueError("Method must be supported and listed in the configuration")
    settings = {"ssf": False, "lora_bottleneck": 0,
                "ft_attn_module": None, "ft_mlp_module": None,
                "vpt_mode": None, "fact_type": None, "bitfit": False,
                "ln": False, "difffit": False, "vqt_num": 0, "full": False}
    if method == "lora":
        rank = config["lora_rank"]
        if not isinstance(rank, int) or rank < 1:
            raise ValueError("LoRA rank must be a positive integer")
        settings["lora_bottleneck"] = rank
    elif method == "ssf":
        settings["ssf"] = True
    elif method == "vpt_deep":
        count = config["vpt_num"]
        if type(count) is not int or count not in (5, 10, 50, 100):
            raise ValueError("Use a VPT-Deep prompt count from the paper grid")
        settings.update(vpt_mode="deep", vpt_num=count, vpt_layer=None, vpt_dropout=0.1)
    elif method == "adaptformer":
        chosen = config["adaptformer"]
        if chosen["bottleneck"] not in (4, 16, 32) or chosen["scale"] not in (0.05, 0.1, 0.2):
            raise ValueError("Use AdaptFormer settings from the paper grid")
        settings.update(ft_mlp_module="adapter", ft_mlp_mode="parallel", ft_mlp_ln="before",
                        adapter_bottleneck=chosen["bottleneck"], adapter_scaler=chosen["scale"],
                        adapter_init="lora_kaiming")
    elif method == "bitfit":
        settings["bitfit"] = True
    elif method == "fact_tt":
        chosen = config["fact_tt"]
        if chosen["dim"] not in (8, 16, 32) or chosen["scale"] not in (0.01, 0.1, 1, 10, 100):
            raise ValueError("Use FacT-TT settings from the paper grid")
        settings.update(fact_type="tt", fact_dim=chosen["dim"], fact_scaler=chosen["scale"])
    elif method == "convpass":
        chosen = config["convpass"]
        if chosen["bottleneck"] not in (8, 16):
            raise ValueError("Use a ConvPass bottleneck from the paper's VTAB grid")
        if chosen["scale"] not in (0.01, 0.1, 1, 10, 100):
            raise ValueError("Use a ConvPass scale from the paper's VTAB grid")
        if not isinstance(chosen["xavier_init"], bool):
            raise ValueError("ConvPass initialization flag must be a boolean")
        settings.update(ft_attn_module="convpass", ft_attn_mode="parallel", ft_attn_ln="after",
                        ft_mlp_module="convpass", ft_mlp_mode="parallel", ft_mlp_ln="after",
                        convpass_bottleneck=chosen["bottleneck"],
                        convpass_scaler=chosen["scale"],
                        convpass_xavier_init=chosen["xavier_init"],
                        convpass_init="lora_xavier")
    return settings


def is_method_parameter(name, method):
    if method not in METHODS:
        raise ValueError("Unsupported method")
    parts = name.split(".")
    if method == "lora":
        return "lora" in parts
    if method == "ssf":
        return parts[-1].startswith(("ssf_scale", "ssf_shift"))
    if method == "convpass":
        return any(part in ("ft_attn_module", "ft_mlp_module") for part in parts)
    if method == "adaptformer":
        return "ft_mlp_module" in parts
    if method == "vpt_deep":
        return parts[0] == "vpt" and parts[-1] == "prompt_embeddings"
    if method == "fact_tt":
        return "fact" in parts
    if method == "bitfit":
        return parts[0] != "head" and parts[-1] == "bias"
    return False


def expected_method_parameters(config, method, width=768, depth=12):
    method_settings(config, method)
    if method == "lora":
        return depth * 4 * width * config["lora_rank"]
    if method == "ssf":
        return depth * 22 * width + 4 * width
    if method == "bitfit":
        return (11 * depth + 2) * width
    if method == "vpt_deep":
        return depth * config["vpt_num"] * width
    if method == "adaptformer":
        rank = config["adaptformer"]["bottleneck"]
        return depth * (2 * width * rank + rank + width)
    if method == "fact_tt":
        rank = config["fact_tt"]["dim"]
        return 2 * width * rank + depth * 12 * rank ** 2
    if method == "convpass":
        rank = config["convpass"]["bottleneck"]
        per_module = 2 * width * rank + 9 * rank ** 2 + 2 * rank + width
        return 2 * depth * per_module
    return 0


def is_pretrained_backbone_parameter(name):
    """Identify common pretrained tensors independently of the trainability mask."""
    if name.startswith("head."):
        return False
    return not any(is_method_parameter(name, method) for method in METHODS if method != "bitfit")


def hash_pretrained_backbone(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        if is_pretrained_backbone_parameter(name):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
