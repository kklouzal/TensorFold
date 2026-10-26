"""Pinned target semantic completion shared by container and text preparation.

This layer reads no files and constructs no tokenizer. An exact ordered source
inventory binds missing fields; family names alone never supply generation EOS
or a normalization program.
"""

import hashlib
import json


_INVENTORY = "9ffc48f421092228fa8431da41bd0760384094c39cbf5a1be27160470d4b10ac"
_SOURCE = {"repository": "Qwen/Qwen3.8-Flash-Next",
           "revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
           "file": "tokenizer.json",
           "sha256": "0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3"}


def semantic_source(metadata):
    """Return the validated supplementary source, or leave an unknown inventory unresolved."""
    if metadata.get("tokenizer.ggml.model") != "gpt2" or metadata.get("tokenizer.ggml.pre") != "qwen35":
        return None
    inventory = [metadata.get("tokenizer.ggml." + key) for key in ("tokens", "token_type", "merges")]
    identity = hashlib.sha256(json.dumps(inventory, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return dict(_SOURCE) if identity == _INVENTORY else None


def generation_defaults(metadata):
    """Complete only the inspected target's independent generation-stop role.

    Geometry-only imports need no tokenizer construction. Unknown inventories
    acquire no invented EOS defaults from a family name, PLE or padding role.
    """
    if semantic_source(metadata) is None:
        return {}
    expected = {"eos": 248046, "bos": 248044, "padding": 248044}
    if any(metadata.get(f"tokenizer.ggml.{role}_token_id") != value for role, value in expected.items()):
        raise ValueError("bound target tokenizer has conflicting BOS/EOS/padding roles")
    return {"eos_token_id": [248046, 248044],
            "tensorfold_source": {"repository": _SOURCE["repository"], "revision": _SOURCE["revision"],
                                  "file": "generation_config.json",
                                  "sha256": "e70c136c1b78ddc1fb0905bac8e733a4dc448d4f852a5dd75143fffc70be550e"}}
