"""Sparkulator owned data/metadata/commit boundaries; no model payload or GPU."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.gemma4.sparkulator_metadata import SparkulatorConfig  # noqa:E402
from tensorfold.families.gemma4.cuda.sparkulator import decompress_int4_group128  # noqa:E402
from tensorfold.families.gemma4.cuda.sparkulator_bridge import SparkulatorState, BridgeSnapshot  # noqa:E402

FIXTURE = Path(__file__).parent / "fixtures/gemma4_sparkulator_26b_config.json"


def test_actual_native_unmasked_profile_and88_tensor_contract():
    cfg = SparkulatorConfig.from_dict(json.loads(FIXTURE.read_text()))
    shapes = cfg.shapes()
    assert len(shapes) == 88 and len(cfg.matrix_shapes()) == 20
    assert sum(dtype == "I32" for dtype, _ in shapes.values()) == 20
    assert sum(dtype == "I64" for dtype, _ in shapes.values()) == 20
    assert sum(dtype == "BF16" for dtype, _ in shapes.values()) == 48
    assert shapes["model.layers.3.self_attn.q_proj.weight_packed"] == ("I32", [8192, 128])
    assert shapes["model.layers.3.self_attn.o_proj.weight_scale"] == ("BF16", [1024, 64])
    assert not any("centroid" in name or "ordering" in name for name in shapes)
    resident, peak = cfg.memory_bytes()
    assert peak >= resident >= cfg.parameter_bytes
    assert cfg.snapshot_bytes(65536) == 5632
    assert cfg.state_bytes(65536) >= 256 * (1 << 20)
    # Global MATH: repeated BF16 K/V + FP32 K/V + FP32 scaled key;
    # retained local gathers and auxiliary owners are separate named charges.
    assert cfg.workspace_bytes(65536) == (8 << 30) + (8 << 20) + (16 << 20)
    assert cfg.workspace_bytes(131072) == (16 << 30) + (8 << 20) + (16 << 20)
    for length in (1, 1023, 1024, 1025):
        local = min(length, cfg.window)
        assert cfg.workspace_bytes(length) == max(length * 16 * 512 * 16, local * 16 * 256 * 16) + local * 8192 + (16 << 20)
    assert cfg.workspace_bytes(65536, global_gqa=True) == (88 << 20) + 65552
    assert cfg.workspace_bytes(131072, global_gqa=True) == (88 << 20) + 65552
    with pytest.raises(ValueError, match="context"):
        cfg.workspace_bytes(131073, global_gqa=True)
    with pytest.raises(ValueError, match="boolean"):
        cfg.workspace_bytes(65536, global_gqa=1)
    with pytest.raises(ValueError):
        cfg.memory_bytes(8)


@pytest.mark.parametrize(
    "patch",
    [
        {"use_ordered_embeddings": True},
        {"backbone_hidden_size": 2560},
        {"text_config": {"hidden_size": 256}},
        {"quantization_config": {"format": "float-quantized"}},
    ],
)
def test_other_head_formats_are_not_silently_broadened(patch):
    raw = json.loads(FIXTURE.read_text())
    for name, value in patch.items():
        if isinstance(value, dict):
            raw[name].update(value)
        else:
            raw[name] = value
    with pytest.raises(ValueError):
        SparkulatorConfig.from_dict(raw)


def test_ct_signed_nibbles_are_offset_biased_low_first_not_twos_complement():
    # Independently hand-encoded CT words: FEDCBA98 -> signed0..7;
    # 76543210 -> signed-8..-1. Negative int32 word preserves high-bit cases.
    words = torch.tensor([0xFEDCBA98 - (1 << 32), 0x76543210] * 8, dtype=torch.int32)[None].repeat(2, 1)
    scales = torch.tensor([[0.5], [2.0]], dtype=torch.bfloat16)
    decoded = decompress_int4_group128(words, scales, torch.tensor([2, 128]), (2, 128))
    integers = torch.tensor([*range(8), *range(-8, 0)] * 8, dtype=torch.bfloat16)
    expected = integers[None] * scales
    assert torch.equal(decoded.view(torch.int16), expected.view(torch.int16))


def test_group_boundaries_row64_tail_and_signed_zero_match_native_bf16_multiply():
    words = torch.zeros((65, 32), dtype=torch.int32)  # nibble0 means signed-8, not0
    scales = torch.tensor([0.25, 0.0], dtype=torch.bfloat16)[None].repeat(65, 1)
    decoded = decompress_int4_group128(words, scales, torch.tensor([65, 256]), (65, 256))
    expected = (torch.full((65, 2, 128), -8, dtype=torch.bfloat16) * scales[:, :, None]).reshape(65, 256)
    assert torch.equal(decoded.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("change", ["shape", "negative", "nan", "overflow"])
def test_decoder_rejects_invalid_shape_or_nonfinite_weight(change):
    words = torch.zeros((1, 16), dtype=torch.int32)
    shape = torch.tensor([1, 128], dtype=torch.int64)
    scale = torch.ones((1, 1), dtype=torch.bfloat16)
    if change == "shape":
        shape[1] = 64
    elif change == "negative":
        scale[0, 0] = -1
    elif change == "nan":
        scale[0, 0] = float("nan")
    else:
        scale[0, 0] = torch.finfo(torch.bfloat16).max
    with pytest.raises(ValueError):
        decompress_int4_group128(words, scale, shape, (1, 128))


def bare_bridge():
    state = object.__new__(SparkulatorState)
    state.owner = SimpleNamespace(device=torch.device("cpu"), context_limit=100)
    state.device = torch.device("cpu")
    state.context_limit = 100
    state.window = 1024
    state._snapshot_token = object()
    state.target_state = SimpleNamespace(position=2, poisoned=False)
    state.position = 0
    state.valid_full = 0
    state.last_hidden = None
    state.poisoned = False
    return state


def test_committed_hidden_is_last_normalized_row_not_emitted_or_graph_alias():
    state = bare_bridge()
    hidden = torch.arange(2 * 2816, dtype=torch.float32).reshape(2, 2816).to(torch.bfloat16)
    state.append(hidden)
    expected = hidden[-1:].clone()[None]
    hidden.zero_()  # later graph replay must not change committed authority
    assert state.position == 2 and torch.equal(state.last_hidden, expected)
    snap = state.snapshot()
    state.last_hidden.zero_()
    state.valid_full = 80
    state.restore(snap)
    assert state.position == 2 and state.valid_full == 0 and torch.equal(state.last_hidden, expected)


def test_stale_main_or_uncommitted_hidden_cannot_publish():
    state = bare_bridge()
    with pytest.raises(ValueError):
        state.append(torch.zeros(1, 2816, dtype=torch.bfloat16))
    assert state.position == 0 and state.last_hidden is None
    with pytest.raises(RuntimeError):
        state.view()
    with pytest.raises(ValueError):
        state.restore(BridgeSnapshot(3, None))


def test_original_finite_validation_is_bounded_cpu_chunks(monkeypatch):
    from tensorfold.families.gemma4.cuda.sparkulator import finite_cpu

    real = torch.isfinite
    sizes = []

    def observed(value):
        sizes.append(value.numel())
        return real(value)

    monkeypatch.setattr(torch, "isfinite", observed)
    value = torch.ones(13, dtype=torch.bfloat16)
    finite_cpu(value, chunk_elements=4)
    assert sizes == [4, 4, 4, 1]
    value[-1] = float("nan")
    with pytest.raises(ValueError, match="nonfinite"):
        finite_cpu(value, chunk_elements=4)


@pytest.mark.parametrize("transport_fails", [False, True])
def test_unpublished_upload_survives_publication_and_retirement_failure(monkeypatch, tmp_path, transport_fails):
    import weakref
    from tensorfold.families.gemma4.cuda import sparkulator as module

    raw = json.loads(FIXTURE.read_text())
    status = module.LoadStatus()
    primary = RuntimeError("parameter publication failed")
    native = KeyError("original native cause")
    primary.__cause__ = native
    retirement = OSError("CUDA retirement failed")
    sync = []
    held = []

    class Source:
        consumed = {}

        def __init__(self, *args, **kwargs):
            pass

        def tensor(self, *args, **kwargs):
            return torch.ones((1, 1), dtype=torch.bfloat16)

        def close(self):
            raise AssertionError("reader must not close after failed drain")

    monkeypatch.setattr(module, "read_json", lambda *args, **kwargs: raw)
    monkeypatch.setattr(module, "_VerifiedTensors", Source)

    def publish(parent, leaf, value, actual_status):
        assert actual_status is status and status.pending[0] is value
        held.append(weakref.ref(value))
        raise primary

    def fail_sync(*args):
        sync.append("one attempt")
        raise retirement

    monkeypatch.setattr(module, "_publish_parameter", publish)
    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    if transport_fails:

        def fail_transport(*args):
            raise MemoryError("rollback entry allocation")

        monkeypatch.setattr(module, "rollback", fail_transport)
    cfg = SimpleNamespace(shapes=lambda: {"pre_projection.weight": ("BF16", [1, 1])})
    caught = None
    try:
        module._load(tmp_path, cfg, {}, torch.device("cpu"), status)
    except BaseException as error:
        caught = error
        assert error is primary and error.__traceback__ is not None
    assert caught is primary
    assert not status.safe_to_release and status.model is not None and status.source is not None
    assert held[0]() is status.pending[0] and status.pending[1] is not None
    assert sync == ([] if transport_fails else ["one attempt"])
    assert primary.__cause__ is not None


def test_same_position_restore_clone_failure_preserves_tuple_but_poisons_combined_state(monkeypatch):
    state = bare_bridge()
    state.position = 2
    state.last_hidden = torch.ones((1, 1, 2816), dtype=torch.bfloat16)
    state.valid_full = 2
    original = state.last_hidden
    snapshot = BridgeSnapshot(2, torch.zeros_like(original), state._snapshot_token)
    failure = MemoryError("snapshot clone")

    def fail_clone(*args, **kwargs):
        raise failure

    original_clone = torch.Tensor.clone
    # The main snapshot has already changed branch at the SAME position2.
    state.target_state.branch = "restored-different-prefix"
    monkeypatch.setattr(torch.Tensor, "clone", fail_clone)
    caught = None
    try:
        state.restore(snapshot)
    except BaseException as error:
        caught = error
        assert error is failure
    assert caught is failure
    assert state.last_hidden is original and state.position == 2 and state.valid_full == 2
    assert state.poisoned
    with pytest.raises(RuntimeError, match="healthy committed"):
        state.view()
    with pytest.raises(RuntimeError, match="matching committed"):
        state.snapshot()
    monkeypatch.setattr(torch.Tensor, "clone", original_clone)
    state.restore(snapshot)
    assert not state.poisoned and state.position == 2 and state.valid_full == 0
    assert torch.equal(state.last_hidden, snapshot.hidden)
    state.poisoned = True
    state.target_state.position = 0
    state.reset()
    assert not state.poisoned and state.position == 0 and state.last_hidden is None
    assert state.snapshot().position == 0


def test_finite_rejection_drops_owned_cpu_views_with_primary_retained():
    import gc
    import weakref
    from tensorfold.families.gemma4.cuda.sparkulator import finite_cpu

    value = torch.tensor([float("nan")], dtype=torch.bfloat16)
    ref = weakref.ref(value)
    caught = None
    try:
        finite_cpu(value, chunk_elements=1)
    except BaseException as error:
        caught = error
        assert isinstance(error, ValueError) and error.__traceback__ is not None
    value = None
    gc.collect()
    assert caught is not None and ref() is None


@pytest.mark.parametrize("target_k,proposal_k", [(0, 256), (1, 1), (64, 64), (256, 256), (512, 256)])
def test_sampled_q_dispatch_preserves_target_policy_and_bounded_support(monkeypatch, target_k, proposal_k):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.gemma4.cuda import sparkulator, assistant_sampled

    head = object.__new__(sparkulator.Sparkulator)
    head._closed = False
    head._bound = object()
    head.block = 16
    head.context_limit = 131072
    head.cfg = SimpleNamespace(vocab=262144)
    head.sampled_q = True
    head.model = object()
    def callback(ids):
        return ids
    head.target = SimpleNamespace(embed_tokens=callback)
    held = {"last_hidden_state": torch.ones((1, 1, 2816), dtype=torch.bfloat16),
            "position": 65536, "shared_kv_states": object(), "attention_mask": None}
    state = SimpleNamespace(owner=head, position=65536, view=lambda: held, poisoned=False)
    history = object()
    policy = Sampling(seed=371, temperature=1, top_k=target_k, top_p=.95)
    calls = []
    packet = object()
    def chain(*args, **kwargs):
        calls.append((args, kwargs))
        return packet
    monkeypatch.setattr(assistant_sampled, "sampled_assistant_chain", chain)
    assert head.propose(state, 258884, 15, history, sampled=True, sampling=policy) is packet
    args, options = calls[0]
    assert args[1] is callback and args[2] is held["last_hidden_state"]
    assert args[6:9] == (15, 65536, policy) and args[8] is policy and args[9] is history
    assert options == {"enabled": True, "temperature_factor": 1.0, "proposal_top_k": proposal_k, "max_count": 15}
    assert policy.top_k == target_k and policy.temperature == 1 and policy.top_p == .95
    assert not state.poisoned
