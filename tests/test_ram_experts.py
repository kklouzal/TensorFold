"""Public offload boundaries and independent header-only CPU/GPU memory oracles."""

import json
import math
import struct
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli, serve_options
from tensorfold.cuda import capacity
from tensorfold.cuda.geometry import indexed_weights
from tensorfold.families import qwen4_exp
from tensorfold.families.qwen4_exp import ram_experts


def checkpoint(path, *, mtp=True, missing=None, bits=4, group=32):
    text = {"hidden_size": 256, "moe_intermediate_size": 64, "shared_expert_intermediate_size": 64,
            "num_experts": 3, "num_experts_per_tok": 2, "num_hidden_layers": 2,
            "max_position_embeddings": 128, "quantization": {"bits": bits, "group_size": group}}
    (path / "config.json").write_text(json.dumps(text))
    entries, offset = {}, 0
    def add(name, dtype, shape):
        nonlocal offset
        if name == missing:
            return
        size = math.prod(shape) * (4 if dtype == "U32" else 2)
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    add("model.embed_tokens.weight", "U32", [32, 32])
    bases = [f"model.layers.{i}.mlp" for i in range(2)] + (["mtp.layers.0.mlp"] if mtp else [])
    for base in bases:
        for kind, prefix in (("switch_mlp", [3]), ("shared_expert", [])):
            for part, rows, cols in (("gate_proj", 64, 256), ("up_proj", 64, 256), ("down_proj", 256, 64)):
                for field, dtype, n in (("weight", "U32", cols // 8), ("scales", "BF16", cols // 32),
                                        ("biases", "BF16", cols // 32)):
                    add(f"{base}.{kind}.{part}.{field}", dtype, prefix + [rows, n])
        add(base + ".gate.weight", "BF16", [3, 256])
        for field, dtype, n in (("weight", "U32", 32), ("scales", "BF16", 8), ("biases", "BF16", 8)):
            add(base + ".shared_expert_gate." + field, dtype, [1, n])
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return text, entries


@pytest.mark.parametrize("mtp,count", [(False, 8), (True, 12)])
def test_pool_host_and_staging_bytes_match_quantized_parameter_oracle(tmp_path, mtp, count):
    _, entries = checkpoint(tmp_path)
    # 3 projections * 256*64 entries, each 4-bit + two BF16 affine values /32.
    entry = 3 * 256 * 64 * (4 * 32 + 2 * 16) // (32 * 8)
    plan = ram_experts.layout(tmp_path, (2 * entry + 17) / 2**30, mtp=mtp)
    assert plan.entry_bytes == entry
    assert plan.slots == 2 and plan.gpu_bytes == 2 * entry
    assert plan.host_bytes == count * entry and plan.staging_bytes == 2 * entry
    native = indexed_weights(1, mtp, mapped_tables=False)
    expected = sum(native(name, info)[0] for name, info in entries.items()
                   if not ram_experts.expert_tensor(name)) + 2 * entry
    sized = capacity.estimate_weights(tmp_path, plan.transform(native))
    assert sized.resident + plan.gpu_bytes == expected
    assert sized.mapped == 0
    assert plan.transform(native)("model.layers.0.mlp.gate.weight", entries["model.layers.0.mlp.gate.weight"]) == (1536, 0)
    assert plan.transform(native)("model.layers.0.mlp.shared_expert_gate.scales",
                                 entries["model.layers.0.mlp.shared_expert_gate.scales"])[0] > 0


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "1"])
def test_invalid_cache_budget_refused_before_checkpoint_read(value):
    with pytest.raises(ValueError, match="finite positive"):
        ram_experts.check("/does/not/exist", value)


def test_subexpert_pool_missing_projection_and_wrong_formats_are_refused(tmp_path):
    checkpoint(tmp_path)
    with pytest.raises(ValueError, match="at least one"):
        ram_experts.layout(tmp_path, 1 / 2**30)
    checkpoint(tmp_path, missing="model.layers.1.mlp.switch_mlp.down_proj.biases")
    with pytest.raises(ValueError, match="invalid affine4 projection"):
        ram_experts.layout(tmp_path, 1)
    checkpoint(tmp_path, group=64)
    with pytest.raises(ValueError, match="groups of 32"):
        ram_experts.check(tmp_path, 1)
    with pytest.raises(ValueError, match="one CUDA GPU"):
        ram_experts.check(tmp_path, 1, tp=2)


@pytest.mark.parametrize("field,value", [("num_experts", 0), ("num_hidden_layers", -1),
    ("hidden_size", 256.0), ("num_experts_per_tok", 4), ("moe_intermediate_size", True)])
def test_invalid_configuration_geometry_precedes_payload_allocation(tmp_path, field, value):
    text, _ = checkpoint(tmp_path)
    text[field] = value
    (tmp_path / "config.json").write_text(json.dumps(text))
    with pytest.raises(ValueError, match="positive integer|must not exceed"):
        ram_experts.layout(tmp_path, 1)


def test_expert_count_limit_matches_native_plan_contract(tmp_path):
    import re
    from pathlib import Path
    native_source = Path(capacity.__file__).with_name("experts.cu").read_text()
    assert int(re.search(r"EMAX\s*=\s*(\d+)", native_source).group(1)) == ram_experts.MAX_ROUTED_EXPERTS + 1
    text, _ = checkpoint(tmp_path)
    text["num_experts"] = ram_experts.MAX_ROUTED_EXPERTS + 1
    (tmp_path / "config.json").write_text(json.dumps(text))
    with pytest.raises(ValueError, match="at most"):
        ram_experts.layout(tmp_path, 1)


def test_cli_rejects_unsupported_backends_and_silent_cuda_ssd_flag(tmp_path):
    checkpoint(tmp_path)
    family = NS(title=qwen4_exp.TITLE, model_type="qwen4_exp", package=qwen4_exp)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--ram-experts", "0.01"])
    assert serve_options.check(args, family, "cuda", tmp_path) is None
    with pytest.raises(ValueError, match="CUDA only"):
        serve_options.check(args, family, "mlx", tmp_path)
    with pytest.raises(ValueError, match="CUDA only"):
        serve_options.check(args, NS(package=NS()), "cuda", tmp_path)
    args.ram_experts, args.ssd_experts = None, 1
    with pytest.raises(ValueError, match="MLX only"):
        serve_options.check(args, family, "cuda", tmp_path)


def test_ram_option_reaches_cuda_engine(tmp_path, monkeypatch):
    from tensorfold.cuda import server
    received = []
    family = NS(title="test", model_type="test", package=NS(cuda_engine=lambda *a, **k:
        received.append(k) or NS(max_len=128)))
    monkeypatch.setattr(server, "App", lambda *a, **k: NS(effective_context_window=128))
    monkeypatch.setattr(server, "serve", lambda *a: None)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--no-drafts", "--ram-experts", "0.25"])
    assert cli._serve_cuda(args, family, tmp_path, 128) == 0
    assert received[0]["ram_experts"] == 0.25


@pytest.mark.parametrize("unified", [False, True])
def test_admission_accounts_host_authority_on_correct_memory_pool(tmp_path, monkeypatch, unified):
    checkpoint(tmp_path)
    monkeypatch.setattr("tensorfold.cuda.build.refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "unified", lambda *a: unified)
    monkeypatch.setattr(capacity, "available_bytes", lambda *a: 2**30)
    monkeypatch.setattr(capacity, "page_room", lambda *a: None)
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: 2**30)
    geometry = capacity.Geometry(lambda _: 0, 0)
    receipt = capacity.admit(tmp_path, 128, True, object(), geometry, lambda *a: (0, 0),
                             host_resident=64 * 2**20, host_extra_staging=2**20)
    assert receipt["weight_bytes_estimate"] == (64 * 2**20 if unified else 0)
    assert receipt["loading_bytes_estimate"] == (2**20 if unified else 0)
    if not unified:
        monkeypatch.setattr(capacity, "host_stream_bytes", lambda: 32 * 2**20)
        with pytest.raises(ValueError, match="host weights/staging"):
            capacity.admit(tmp_path, 128, True, object(), geometry, lambda *a: (0, 0), host_resident=64 * 2**20)


def test_host_cgroup_refusal_precedes_any_cuda_allocation(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    monkeypatch.setattr("tensorfold.cuda.build.refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "unified", lambda *a: False)
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda: 1024)
    with pytest.raises(ValueError, match="container/cgroup"):
        capacity.admit(tmp_path, 128, True, object(), capacity.Geometry(lambda _: 0, 0),
                       lambda *a: (0, 0), host_resident=2048)


def test_fixed_pool_does_not_manufacture_weight_packing_staging(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    monkeypatch.setattr("tensorfold.cuda.build.refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "unified", lambda *a: False)
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: 2**30)
    monkeypatch.setattr(capacity, "available_bytes", lambda *a: 2**30)
    monkeypatch.setattr(capacity, "page_room", lambda *a: None)
    plan = capacity.admit(tmp_path, 128, True, object(), capacity.Geometry(lambda _: 0, 0),
                          lambda *a: (0, 0), resident_extra=64 * 2**20)
    assert plan["weight_bytes_estimate"] == 64 * 2**20
    assert plan["loading_bytes_estimate"] == 0


def test_engine_shutdown_joins_worker_before_closing_cache():
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    calls = []
    engine = FlashNextEngine.__new__(FlashNextEngine)
    engine.scheduler = NS(close=lambda: calls.append("joined"))
    engine.w = NS(meta={"expert_cache": NS(close=lambda: calls.append("cache"))})
    engine.close()
    assert calls == ["joined", "cache"]
    assert engine.scheduler is None


def test_failed_worker_join_keeps_expert_cache_alive():
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    def fail():
        raise RuntimeError("worker still active")
    engine = FlashNextEngine.__new__(FlashNextEngine)
    engine.scheduler = NS(close=fail)
    closed = []
    engine.w = NS(meta={"expert_cache": NS(close=lambda: closed.append(True))})
    with pytest.raises(RuntimeError, match="worker still active"):
        engine.close()
    assert not closed and engine.scheduler is not None


def cgroup_files(monkeypatch, membership, entries):
    """Isolated kernel-file snapshot; never inspect or mutate the test host's quota."""

    from pathlib import Path

    files = {"/proc/self/cgroup": membership, **entries}
    def read(path, *args, **kwargs):
        value = files.get(str(path), FileNotFoundError(str(path)))
        if isinstance(value, BaseException):
            raise value
        return value
    monkeypatch.setattr(Path, "read_text", read)


@pytest.mark.parametrize("membership,entries,expected", [
    ("0::/\n", {"/sys/fs/cgroup/memory.max": "2048", "/sys/fs/cgroup/memory.current": "512"}, 1536),
    ("0::/team/job\n", {"/sys/fs/cgroup/team/job/memory.max": "2048",
        "/sys/fs/cgroup/team/job/memory.current": "512", "/sys/fs/cgroup/team/memory.max": "4096",
        "/sys/fs/cgroup/team/memory.current": "3072"}, 1024),
    ("0::/team/job\n", {"/sys/fs/cgroup/team/job/memory.max": "max",
        "/sys/fs/cgroup/team/memory.max": "2048", "/sys/fs/cgroup/team/memory.current": "2304"}, 0),
    ("0::/team/job\n", {"/sys/fs/cgroup/team/job/memory.max": "max"}, None),
    ("2:memory:/legacy\n", {"/sys/fs/cgroup/memory.max": "1024"}, None),
])
def test_host_cgroup_uses_process_membership_and_limiting_ancestor(monkeypatch, membership, entries, expected):
    cgroup_files(monkeypatch, membership, entries)
    assert capacity.host_cgroup_bytes() == expected


@pytest.mark.parametrize("used", [FileNotFoundError("gone"), PermissionError("denied"), "invalid"])
def test_exposed_cgroup_limit_cannot_silently_lose_its_usage(monkeypatch, used):
    cgroup_files(monkeypatch, "0::/job\n", {"/sys/fs/cgroup/job/memory.max": "2048",
                                          "/sys/fs/cgroup/job/memory.current": used})
    with pytest.raises(ValueError, match="cgroup-v2 memory.*at /sys/fs/cgroup/job"):
        capacity.host_cgroup_bytes()


@pytest.mark.parametrize("membership", ["0::relative\n", "0::/../job\n", "0::/\n0::/job\n"])
def test_invalid_cgroup_membership_cannot_escape_controller_root(monkeypatch, membership):
    cgroup_files(monkeypatch, membership, {})
    with pytest.raises(ValueError, match="process membership"):
        capacity.host_cgroup_bytes()


@pytest.fixture
def ram_loader(monkeypatch):
    """CPU-only loader fault injection, including real success-path final cleanup."""

    torch = pytest.importorskip("torch")
    from tensorfold.cuda import expert_cache
    from tensorfold.families.qwen4_exp.cuda import exl3, weights

    calls = []
    reader = NS(has=lambda name: False, layer_names=lambda *a: [],
                get=lambda name: torch.ones((2, 8)), close=lambda: calls.append("reader close"),
                release=lambda: calls.append("reader release"))
    cache = NS(close=lambda: calls.append("cache close"))
    def create_cache(*args):
        calls.append("cache acquire")
        return cache
    rope = NS(inverse_frequencies=lambda framework: torch.zeros(1), metadata=lambda: {})
    cfg = NS(rope=rope, layers=0, quant="mlx", vocab=2)
    monkeypatch.setattr(ram_experts, "check", lambda *a, **k: None)
    monkeypatch.setattr(ram_experts, "layout", lambda *a, **k: NS(gpu_bytes=2048, entry_bytes=1024))
    monkeypatch.setattr(exl3, "is_exl3", lambda *a: False)
    monkeypatch.setattr(weights.Config, "read", lambda *a, **k: cfg)
    monkeypatch.setattr(weights, "_Reader", lambda *a: reader)
    monkeypatch.setattr(weights, "norms_around_one", lambda *a: True)
    monkeypatch.setattr(weights, "make_q4", lambda *a: NS())
    monkeypatch.setattr(weights, "stack_q4", lambda *a: NS())
    monkeypatch.setattr(expert_cache, "HostExpertCache", create_cache)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return NS(weights=weights, reader=reader, cache=cache, calls=calls)


def test_reader_initialization_failure_precedes_expert_cache_acquisition(tmp_path, monkeypatch, ram_loader):
    def fail(*args):
        raise OSError("checkpoint index unavailable")
    monkeypatch.setattr(ram_loader.weights, "_Reader", fail)
    with pytest.raises(OSError, match="checkpoint index unavailable"):
        ram_loader.weights.load(tmp_path, "cpu", ram_experts=1, mtp=False, draft_vocab=None)
    assert ram_loader.calls == []


def test_norm_probe_failure_closes_reader_before_expert_cache_acquisition(tmp_path, monkeypatch, ram_loader):
    def fail(*args):
        raise ValueError("invalid norm encoding")
    monkeypatch.setattr(ram_loader.weights, "norms_around_one", fail)
    with pytest.raises(ValueError, match="invalid norm encoding"):
        ram_loader.weights.load(tmp_path, "cpu", ram_experts=1, mtp=False, draft_vocab=None)
    assert ram_loader.calls == ["reader close"]


def test_final_reader_close_failure_also_closes_expert_cache(tmp_path, ram_loader):
    def fail_first():
        ram_loader.calls.append("reader close")
        if ram_loader.calls.count("reader close") == 1:
            raise RuntimeError("reader completion failed")
    ram_loader.reader.close = fail_first
    with pytest.raises(RuntimeError, match="reader completion failed"):
        ram_loader.weights.load(tmp_path, "cpu", ram_experts=1, mtp=False, draft_vocab=None)
    assert ram_loader.calls == ["cache acquire", "reader close", "reader close", "cache close"]


def test_payload_failure_preserves_primary_when_both_cleanups_fail(tmp_path, ram_loader):
    primary = OSError("payload read failed")
    def payload(name):
        raise primary
    def reader_close():
        ram_loader.calls.append("reader close")
        raise RuntimeError("reader cleanup failed")
    def cache_close():
        ram_loader.calls.append("cache close")
        raise RuntimeError("cache cleanup failed")
    ram_loader.reader.get, ram_loader.reader.close, ram_loader.cache.close = payload, reader_close, cache_close
    with pytest.raises(OSError, match="payload read failed") as caught:
        ram_loader.weights.load(tmp_path, "cpu", ram_experts=1, mtp=False, draft_vocab=None)
    assert caught.value is primary
    assert ram_loader.calls == ["cache acquire", "reader close", "cache close"]
    assert any("reader cleanup failed" in note for note in primary.__notes__)
    assert any("cache cleanup failed" in note for note in primary.__notes__)
