"""Optional cross-run initialization checks, independent of the training environment."""

import hashlib
import json
from pathlib import Path


HASH_FIELDS = ("classifier_initial_sha256", "frozen_backbone_sha256")


def load_reference_result(path):
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Initialization reference not found: {path}")
    raw = path.read_bytes()
    reference = json.loads(raw)
    if not isinstance(reference, dict):
        raise ValueError("Initialization reference must be a result object")
    for field in HASH_FIELDS:
        value = reference.get(field)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError(f"Reference requires a valid {field}")
    extra = {}
    if "pretrained_backbone_initial_sha256" in reference:
        value = reference["pretrained_backbone_initial_sha256"]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("Reference has an invalid pretrained backbone hash")
        extra["pretrained_backbone_initial_sha256"] = value
    return {**{field: reference[field] for field in HASH_FIELDS}, **extra,
            "source_file": path.name, "source_sha256": hashlib.sha256(raw).hexdigest()}


def validate_initialization_reference(reference, classifier_hash, backbone_hash, pretrained_hash=None):
    if reference is None:
        return False
    if reference[HASH_FIELDS[0]] != classifier_hash:
        raise ValueError(f"Initialization differs from reference: {HASH_FIELDS[0]}")
    expected = reference.get("pretrained_backbone_initial_sha256", reference[HASH_FIELDS[1]])
    actual = pretrained_hash if pretrained_hash is not None else backbone_hash
    if expected != actual:
        raise ValueError("Initialization differs from reference: frozen_backbone_sha256 / pretrained backbone")
    return True
