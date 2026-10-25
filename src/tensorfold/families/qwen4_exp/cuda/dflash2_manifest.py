"""DFlash2 pre-load pairing authority in the existing execution manifest.

This consumes ``draft_pairing`` plus the existing exact ``artifacts`` closure;
it is not another manifest system or an execution-eligibility certificate.
Content authority is issued by model_identity and remains reusable after load.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, read_metadata_json
from tensorfold.engine.model_identity import ModelIdentity, canonical_runtime_json, capture_model_identity

from .dflash2_config import DFlash2Config
from .execution_identity import SCHEMA, VERSION

PAIRING_KEYS = frozenset({"method", "target_artifacts", "metadata_artifacts", "draft_config", "draft_weights",
                          "target_config_file", "tokenizer_file", "tokenizer", "features", "runtime", "selector", "sharing", "target_numerics"})


def digest(value):
    return hashlib.sha256(canonical_runtime_json(value, max_bytes=16 << 20, max_nodes=2000000).encode()).hexdigest()


def select_method(raw_config, requested="auto"):
    """Classify DFlash without ever falling back to an MTP loader."""
    if not isinstance(raw_config, dict):
        raise ValueError("draft model configuration must be an object")
    architectures = raw_config.get("architectures")
    architectures = [] if architectures is None else architectures
    if not isinstance(architectures, list) or any(not isinstance(name, str) for name in architectures):
        raise ValueError("draft architectures must be a list of names")
    if architectures == ["DFlash2DraftModel"]:
        if requested not in ("auto", "dflash", "dflash2"):
            raise ValueError("explicit speculative method differs from the DFlash2 checkpoint")
        return "dflash2"
    if requested in ("dflash", "dflash2") or any(
            "dflash" in name.lower() for name in (architectures or ()) if isinstance(name, str)):
        raise ValueError("this DFlash architecture needs a separately implemented pairing profile")
    return None  # Caller retains its existing explicit non-DFlash classifier.


def tokenizer_contract(tokenizer, *, vocab_size, mask_id):
    vocabulary = tokenizer.get_vocab(with_added_tokens=True)
    if (not isinstance(vocabulary, dict) or not vocabulary or len(vocabulary) > vocab_size
            or any(not isinstance(token, str) or type(index) is not int or not 0 <= index < vocab_size
                   for token, index in vocabulary.items())):
        raise ValueError("DFlash2 actual tokenizer IDs do not fit its declared target vocabulary")
    piece = tokenizer.id_to_token(mask_id)
    if not isinstance(piece, str) or vocabulary.get(piece) != mask_id:
        raise ValueError("DFlash2 mask ID is not the declared target tokenizer token")
    # Sparse mapping retains explicit gaps/aliases without allocating one entry
    # per potentially padded model-vocabulary row.
    mapping = [[token, index] for token, index in sorted(vocabulary.items())]
    return {"vocab_size": vocab_size, "mask_token_id": mask_id, "mask_token": piece,
            "token_ids_sha256": digest({"model_vocab_size": vocab_size, "token_to_id": mapping})}


def pairing_contract(config, *, target_config, tokenizer, context, kv_identity, target_numerics, tp=8):
    """The exact implemented semantic contract; no publisher-name dispatch."""
    if tp != 8:
        raise ValueError("the initial DFlash2 pairing requires fixed TP8")
    if not isinstance(kv_identity, str) or not kv_identity:
        raise ValueError("DFlash2 pairing needs the selected target codec/schema identity")
    if (not isinstance(target_numerics, dict) or target_numerics.get("execution_dtype") != "float16"
            or target_numerics.get("recurrent_state_dtype") != "float32"
            or not isinstance(target_numerics.get("rope"), dict)
            or not isinstance(target_numerics["rope"].get("rope_theta"), (int, float))
            or isinstance(target_numerics["rope"].get("rope_theta"), bool)):
        raise ValueError("the initial DFlash2 pairing requires explicit FP16 target execution")
    if type(context) is not int or not 1 <= context <= config.context:
        raise ValueError("DFlash2 pairing context exceeds supported draft positions")
    text = target_config.get("text_config", target_config)
    config.check_target(SimpleNamespace(hidden=text.get("hidden_size"), vocab=text.get("vocab_size"),
        layers=text.get("num_hidden_layers"), streams=text.get("hc_count", 4)))
    from ..rope import RopeParameters
    target_rope = RopeParameters.from_config(target_config)
    if target_numerics["rope"]["rope_theta"] != target_rope.theta:
        raise ValueError("DFlash2 draft RoPE must not replace the target's positional theta")
    canonical_runtime_json(target_numerics)
    return {"method": "dflash2", "tokenizer": tokenizer,
            "features": {"completed_layers": list(config.taps), "hc_streams": config.hc_streams,
                         "reduction": "mean", "order": "declared-tap-order", "width": len(config.taps) * config.hidden,
                         "boundary": "completed-block-before-final-hc-output-mixer"},
            "runtime": {"tp": tp, "draft_weight_dtype": "float16", "draft_kv_dtype": "float16",
                        "block_rows": 4, "context_tokens": context, "physical_window_rows": min(context, config.window),
                        "window_policy": "accepted-prefix-fp16-ring", "draft_rope_theta": config.theta,
                        "target_kv_identity": kv_identity, "k_norm": "per-head-rms", "v_scale": None},
            "selector": {"rank": config.selector_rank, "top_k": config.top_k,
                         "codebook_token_ids": "target-vocabulary", "proposal_probabilities": "actual-conditional-selector"},
            "sharing": {"embedding": "actual-target-storage", "head": "actual-target-storage"},
            "target_numerics": target_numerics}


@dataclass(frozen=True)
class PairingBinding:
    identity: str
    authority: ModelIdentity
    contract_json: str
    source_sha256: str
    draft_weights_name: str
    plan: object

    @property
    def qualification(self):
        return "unqualified-candidate"

    @property
    def contract(self):
        return json.loads(self.contract_json)  # callers cannot mutate issued contract storage

    def verify_unchanged(self):
        self.authority.verify_unchanged()

    def receipt(self):
        return {"identity": self.identity, "content_identity": self.authority.model_prefix,
                "qualification": self.qualification, "draft_sha256": self.source_sha256,
                "eligible_execution": "requires-independent-live-correctness-resources-performance-evidence"}


def preflight_pairing(manifest, *, files, target_config, draft_plan, context, kv_identity, target_numerics,
                      authorize, max_file_bytes, max_total_bytes, reuse=None):
    """Verify exact files and pairing before device allocation.

    ``files`` is the actual loader-derived target/draft/prepared-metadata closure,
    with the existing logical names. Paths are authorized by the owning engine.
    The caller preserves immutability through load and subsequent reuse.
    """
    if manifest.get("schema") != SCHEMA or type(manifest.get("version")) is not int or manifest["version"] != VERSION:
        raise ValueError("DFlash2 pairing extends the existing execution-profile manifest version1")
    section, expected = manifest.get("draft_pairing"), manifest.get("artifacts")
    if not isinstance(section, dict) or set(section) != PAIRING_KEYS:
        raise ValueError("DFlash2 requires an explicit complete draft_pairing contract")
    if not isinstance(files, dict) or not files or not isinstance(expected, dict) or set(expected) != set(files):
        raise ValueError("DFlash2 manifest artifacts must equal the actual complete loader input closure")
    if any(not isinstance(name, str) or not isinstance(path, Path) for name, path in files.items()):
        raise ValueError("DFlash2 actual inputs need logical names and owned Path entries")
    for key in ("draft_config", "draft_weights", "target_config_file", "tokenizer_file"):
        if not isinstance(section[key], str) or not section[key]:
            raise ValueError("DFlash2 named artifact roles must be nonempty logical strings")
    target_names = sorted(name for name in files if name.startswith("target/"))
    metadata_names = sorted(name for name in files if name.startswith("metadata/"))
    draft_names = {name for name in files if name.startswith("draft/")}
    declared_draft = {section["draft_config"], section["draft_weights"]}
    if (not target_names or not metadata_names or section["target_artifacts"] != target_names
            or section["metadata_artifacts"] != metadata_names or draft_names != declared_draft
            or len(target_names) + len(metadata_names) + len(draft_names) != len(files)):
        raise ValueError("DFlash2 named target/draft/tokenizer closure differs from the selected loader")
    tokenizer_name = section["tokenizer_file"]
    if (tokenizer_name not in metadata_names or section["target_config_file"] not in metadata_names
            or Path(files[section["draft_weights"]]).resolve() != draft_plan.path.resolve()):
        raise ValueError("DFlash2 actual tokenizer or draft file is outside the declared input role")
    for name, item in expected.items():
        if (not isinstance(item, dict) or set(item) != {"size", "sha256"} or type(item["size"]) is not int
                or item["size"] < 0 or not isinstance(item["sha256"], str) or len(item["sha256"]) != 64
                or any(char not in "0123456789abcdef" for char in item["sha256"])):
            raise ValueError("DFlash2 artifact size/SHA256 is invalid: " + name)
    if expected[tokenizer_name]["size"] > MAX_HEADER_BYTES:
        raise ValueError("DFlash2 tokenizer metadata exceeds the existing checkpoint metadata bound")
    if expected[section["draft_config"]]["sha256"] != draft_plan.config_sha256:
        raise ValueError("DFlash2 parsed draft configuration differs from the pinned content")
    source_hash = expected[section["draft_weights"]]["sha256"]
    if draft_plan.source_sha256 is not None and draft_plan.source_sha256 != source_hash:
        raise ValueError("DFlash2 source digest conflicts with the selected artifact plan")
    role_keys = ("target_artifacts", "metadata_artifacts", "draft_config", "draft_weights", "target_config_file", "tokenizer_file")
    declared = {key: value for key, value in section.items() if key not in role_keys}
    # Reject cheap semantic mismatches before streaming large model payloads.
    preliminary = pairing_contract(draft_plan.config, target_config=target_config, tokenizer=section["tokenizer"],
        context=context, kv_identity=kv_identity, target_numerics=target_numerics, tp=draft_plan.tp[1])
    if canonical_runtime_json(declared, max_bytes=16 << 20) != canonical_runtime_json(preliminary, max_bytes=16 << 20):
        raise ValueError("DFlash2 declared feature/token-ID/numerical contract differs from the selected profile")
    authority = capture_model_identity(files, runtime_identity={"schema": SCHEMA, "version": VERSION},
        authorize=authorize, max_file_bytes=max_file_bytes, max_total_bytes=max_total_bytes,
        reuse=reuse, reuse_is_immutable=reuse is not None)
    observed = {item.logical_name: {"size": item.size, "sha256": item.sha256} for item in authority.files}
    if observed != expected:
        raise ValueError("DFlash2 actual content differs from the named pinned closure")
    actual_config = DFlash2Config.from_dict(read_metadata_json(files[section["draft_config"]]), block=draft_plan.config.block)
    if actual_config != draft_plan.config:
        raise ValueError("DFlash2 parsed draft contract changed before content binding")
    source_record = next(item for item in authority.files if item.logical_name == section["draft_weights"])
    if draft_plan.source_stat is not None and source_record.stat_identity != draft_plan.source_stat:
        raise ValueError("DFlash2 source header lifetime differs from the captured content")
    actual_target = read_metadata_json(files[section["target_config_file"]])
    if canonical_runtime_json({"text": actual_target.get("text_config", actual_target)}, max_bytes=16 << 20) != \
            canonical_runtime_json({"text": target_config.get("text_config", target_config)}, max_bytes=16 << 20):
        raise ValueError("DFlash2 target configuration differs from actual prepared loader metadata")
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(files[tokenizer_name]))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    token_contract = tokenizer_contract(tokenizer, vocab_size=draft_plan.config.vocab, mask_id=draft_plan.config.mask)
    contract = pairing_contract(draft_plan.config, target_config=target_config, tokenizer=token_contract,
                               context=context, kv_identity=kv_identity, target_numerics=target_numerics, tp=draft_plan.tp[1])
    encoded = canonical_runtime_json(contract, max_bytes=16 << 20)
    if canonical_runtime_json(declared, max_bytes=16 << 20) != encoded:
        raise ValueError("DFlash2 declared feature/token-ID/numerical contract differs from actual selected inputs")
    authority.verify_unchanged()
    identity = digest({"contents": observed, "pairing": contract})
    bound_plan = replace(draft_plan, source_sha256=source_hash)
    return PairingBinding(identity, authority, encoded, source_hash, section["draft_weights"], bound_plan)
