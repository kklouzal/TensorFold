"""Pairing/content boundaries in the existing startup-manifest schema."""
import copy
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import hashlib
import json
import threading
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer, models

from tensorfold.engine.model_identity import ModelIdentity
from tensorfold.families.qwen4_exp.cuda.dflash2_manifest import (
    SCHEMA, pairing_contract, preflight_pairing, select_method, tokenizer_contract,
)
from tensorfold.families.qwen4_exp.cuda.dflash2_weights import DFlash2Plan, load_shard
from tensorfold.families.qwen4_exp.cuda.dflash2_runtime import DFlash2Runtime
from tensorfold.families.qwen4_exp.rope import RopeParameters

from test_flashnext_dflash2 import Head, make_weights, target


@pytest.fixture
def packet(tmp_path):
    draft, metadata = tmp_path / "draft", tmp_path / "metadata"
    draft.mkdir()
    metadata.mkdir()
    c, _ = make_weights(draft)
    plan = DFlash2Plan.read(draft, tp=(0, 8))
    target = {"hidden_size": c.hidden, "vocab_size": c.vocab, "num_hidden_layers": c.target_layers,
              "hc_count": c.hc_streams, "num_attention_heads": 8, "head_dim": 4,
              "max_position_embeddings": 4096,
              "rope_parameters": {"rope_theta": 10000000, "partial_rotary_factor": 1.0, "mrope_section": [0, 0, 2]}}
    (metadata / "config.json").write_text(json.dumps(target))
    (metadata / "tokenizer_config.json").write_text(json.dumps({"chat_template": "{{messages}}"}))
    tokenizer = Tokenizer(models.WordLevel({"token" + str(i): i for i in range(c.vocab)}, unk_token="token0"))
    tokenizer.save(str(metadata / "tokenizer.json"))
    target_file = tmp_path / "tiny.gguf"
    target_file.write_bytes(b"fixture content only; not a model or a quality result")
    files = {"target/tiny.gguf": target_file,
             "draft/config.json": draft / "config.json", "draft/model.safetensors": draft / "model.safetensors",
             **{"metadata/" + path.name: path for path in metadata.iterdir()}}
    artifacts = {name: {"size": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for name, path in files.items()}
    numerics = {"execution_dtype": "float16", "recurrent_state_dtype": "float32",
                "rope": RopeParameters.from_config(target).metadata()}
    contract = pairing_contract(c, target_config=target,
        tokenizer=tokenizer_contract(tokenizer, vocab_size=c.vocab, mask_id=c.mask),
        context=64, kv_identity="actual-int8-h32-schema-v1", target_numerics=numerics)
    section = {**contract, "target_artifacts": sorted(name for name in files if name.startswith("target/")),
               "metadata_artifacts": sorted(name for name in files if name.startswith("metadata/")),
               "draft_config": "draft/config.json", "draft_weights": "draft/model.safetensors",
               "target_config_file": "metadata/config.json", "tokenizer_file": "metadata/tokenizer.json"}
    manifest = {"schema": SCHEMA, "version": 1, "artifacts": artifacts, "draft_pairing": section}
    params = {"files": files, "target_config": target, "draft_plan": plan, "context": 64,
              "kv_identity": "actual-int8-h32-schema-v1", "target_numerics": numerics,
              "authorize": lambda _name, path: path.resolve(strict=True),
              "max_file_bytes": max(record["size"] for record in artifacts.values()),
              "max_total_bytes": sum(record["size"] for record in artifacts.values())}
    return manifest, params


def test_preload_closure_contract_and_issued_authority_reuse_never_grant_eligibility(packet):
    manifest, params = packet
    binding = preflight_pairing(manifest, **params)
    assert type(binding.authority) is ModelIdentity
    assert binding.plan.source_sha256 == manifest["artifacts"]["draft/model.safetensors"]["sha256"]
    assert binding.qualification == "unqualified-candidate"
    assert "requires-independent-live" in binding.receipt()["eligible_execution"]
    binding.verify_unchanged()
    reused = preflight_pairing(manifest, reuse=binding.authority, **params)
    assert reused.authority.bytes_read == 0 and reused.authority.hashes_reused == len(params["files"])
    assert reused.identity == binding.identity
    weights = load_shard(binding.plan, device="cpu", source_authority=binding.authority, chunk_bytes=4096)
    assert weights.nbytes == binding.plan.resident_bytes
    view = binding.contract
    view["features"]["completed_layers"] = [0]
    assert binding.contract["features"]["completed_layers"] != [0]


@pytest.mark.parametrize("bad_rank", [None, 3])
def test_cpu_pairing_phase_refusal_cannot_strand_other_ranks_in_content_exchange(packet, monkeypatch, bad_rank):
    from tensorfold.families.qwen4_exp.cuda import execution_identity as identity

    manifest, params = packet
    values, condition = {}, threading.Condition()

    class Store:
        def set(self, key, value):
            with condition:
                values[key] = value.encode() if isinstance(value, str) else value
                condition.notify_all()

        def wait(self, keys, timeout):
            with condition:
                if not condition.wait_for(lambda: all(key in values for key in keys), timeout=5):
                    raise RuntimeError("test phase wait did not retire")

        def get(self, key):
            self.wait([key], None)
            with condition:
                return values[key]

        def delete_key(self, key):
            with condition:
                return values.pop(key, None) is not None

    store = Store()
    monkeypatch.setattr(identity, "artifact_files", lambda engine: (params["files"], None))
    engines = [SimpleNamespace(rank=rank, tp=8, multi=None, _startup_store=store,
                              kv_pair=SimpleNamespace(identity=params["kv_identity"]),
                              rope=SimpleNamespace(metadata=lambda: params["target_numerics"]["rope"]))
               for rank in range(8)]

    def prepare(engine):
        return identity.prepare_dflash_pairing(engine, manifest, target_config=params["target_config"],
            plan=replace(params["draft_plan"], tp=(engine.rank, 8)),
            context=params["context"] + int(engine.rank == bad_rank))

    with ThreadPoolExecutor(max_workers=8) as workers:
        tasks = [workers.submit(prepare, engine) for engine in engines]
        if bad_rank is not None:
            for task in tasks:
                with pytest.raises(ValueError, match="preflight refused"):
                    task.result(timeout=10)
            assert all(not hasattr(engine, "_dflash_pairing") for engine in engines)
        else:
            plans = [task.result(timeout=10) for task in tasks]
            assert len({engine._dflash_pairing.identity for engine in engines}) == 1
            assert all(plan.source_sha256 == manifest["artifacts"]["draft/model.safetensors"]["sha256"] for plan in plans)
            assert sum(engine._dflash_pairing.authority.bytes_read for engine in engines) == params["max_total_bytes"]
            assert all(engine._dflash_pairing.authority.bytes_read == 0 for engine in engines[1:])
            repeated = [workers.submit(prepare, engine) for engine in engines]
            assert all(task.result(timeout=10).source_sha256 == plans[0].source_sha256 for task in repeated)
            assert all(engine._dflash_pairing.authority.bytes_read == 0 for engine in engines)
    assert not values  # Every outcome and retirement acknowledgement was consumed.


@pytest.mark.parametrize("mutation", ["omitted-input", "bad-digest", "mask-piece", "mapping", "prelayer-features",
                                     "unbounded-kv", "nominal-q", "qualified-boolean", "target-theta"])
def test_forged_or_incomplete_pairings_refuse_before_any_device_work(packet, mutation):
    manifest, params = packet
    body = copy.deepcopy(manifest)
    section = body["draft_pairing"]
    if mutation == "omitted-input":
        del body["artifacts"]["metadata/tokenizer_config.json"]
    elif mutation == "bad-digest":
        body["artifacts"]["draft/model.safetensors"]["sha256"] = "0" * 64
    elif mutation == "mask-piece":
        section["tokenizer"]["mask_token"] = "another-token"
    elif mutation == "mapping":
        section["tokenizer"]["token_ids_sha256"] = "0" * 64
    elif mutation == "prelayer-features":
        section["features"]["boundary"] = "pre-layer-input"
    elif mutation == "unbounded-kv":
        section["runtime"]["window_policy"] = "full-prefix-allocation-with-mask"
    elif mutation == "nominal-q":
        section["selector"]["proposal_probabilities"] = "pre-selector-logits"
    elif mutation == "qualified-boolean":
        section["qualified"] = True
    else:
        params = {**params, "target_numerics": copy.deepcopy(params["target_numerics"])}
        params["target_numerics"]["rope"]["rope_theta"] = 10000
        section["target_numerics"]["rope"]["rope_theta"] = 10000
    with pytest.raises(ValueError):
        preflight_pairing(body, **params)


def test_content_authority_detects_changed_inputs_and_rejects_disk_json_hash_authority(packet):
    manifest, params = packet
    binding = preflight_pairing(manifest, **params)
    path = params["files"]["metadata/tokenizer_config.json"]
    path.write_text(json.dumps({"chat_template": "changed"}))
    with pytest.raises(ValueError, match="changed"):
        binding.verify_unchanged()
    with pytest.raises(ValueError, match="process"):
        load_shard(binding.plan, device="cpu", source_authority={"passed": True}, chunk_bytes=4096)


def test_method_selection_never_routes_dflash_into_stock_mtp():
    assert select_method({"architectures": ["DFlash2DraftModel"]}) == "dflash2"
    assert select_method({"architectures": ["DSparkDraftModel"]}) is None
    for raw, choice in (({"architectures": ["DFlash2DraftModel"]}, "mtp"),
                        ({"architectures": ["DFlashDraftModel"]}, "auto"),
                        ({"architectures": ["StockMTP"]}, "dflash")):
        with pytest.raises(ValueError):
            select_method(raw, choice)


def test_unpinned_source_fails_before_cuda_allocation(packet):
    _manifest, params = packet
    with pytest.raises(ValueError, match="pinned before"):
        load_shard(params["draft_plan"], device="cuda", chunk_bytes=4096)


def test_runtime_consumes_pairing_plan_and_preserves_runtime_local_snapshot_scope(packet):
    manifest, params = packet
    binding = preflight_pairing(manifest, **params)
    w = target(binding.plan.config)
    w.comm = SimpleNamespace(world=8, rank=0)
    w.head = Head(w.head.weight[:binding.plan.config.vocab // 8])
    runtime = DFlash2Runtime(w, binding.plan, context=64, pairing=binding)
    state = SimpleNamespace(pos=0, slot_generation=0, dflash2=None)
    runtime.init_state(state)
    assert runtime.nbytes == binding.plan.resident_bytes
    assert runtime.record(state)["snapshot_scope"] == "current-runtime-numerical-profile-rank"
    runtime.close_state(state)
    runtime.close()
