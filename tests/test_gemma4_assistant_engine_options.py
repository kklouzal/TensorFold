"""Private official-head options and retained full-support q memory admission."""
from itertools import product
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.gemma4.cuda import e2b_engine  # noqa: E402 - optional runtime admitted above
from tensorfold.engine.rejection import ChainPacket, ProbabilityRow  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.assistant_sampled import RecordedSampledChain  # noqa: E402 - optional runtime admitted above
from test_gemma4_e2b_metadata import e2b_config  # noqa: E402 - optional runtime admitted above
from test_gemma4_e4b import e4b_config  # noqa: E402 - optional runtime admitted above

FLAGS = ("assistant_attention_gqa", "assistant_device_chain", "assistant_sampled_q")


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("value", [1, None, "true"])
def test_assistant_options_are_strict_boolean_before_any_gpu(monkeypatch, tmp_path, flag, value):
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("invalid switch touched GPU"))
    with pytest.raises(ValueError, match="boolean"):
        e2b_engine.E2BEngine(tmp_path, **{flag: value})


@pytest.mark.parametrize("flag", FLAGS)
@pytest.mark.parametrize("mode", ["no-head", "disabled", "dflash2", "eagle3"])
def test_assistant_options_refuse_absent_disabled_and_other_heads_before_gpu(monkeypatch, tmp_path, flag, mode):
    raw = e2b_config() if mode == "dflash2" else e4b_config()
    (tmp_path/"config.json").write_text(json.dumps(raw))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("wrong head reached GPU"))
    if mode == "eagle3":
        cfg = SimpleNamespace(block=4, memory_bytes=lambda _: (1, 2))
        monkeypatch.setattr(e2b_engine, "companion_metadata", lambda *_: ("eagle3", cfg))
    options = {flag: True, "drafter": "" if mode == "no-head" else "named-companion",
               "no_drafts": mode == "disabled"}
    with pytest.raises(ValueError, match="official"):
        e2b_engine.E2BEngine(tmp_path, **options)


@pytest.mark.parametrize("values,changed_policy",[(values,changed) for values in product((False,True),repeat=3)
                                                for changed in (False,True) if not changed or values[2]])
def test_official_options_forward_exactly_and_admission_only_adds_sampled_q(monkeypatch, tmp_path, values,changed_policy):
    import tensorfold.cuda.chat_template as template
    import tensorfold.token_metadata as tokens
    from tensorfold.families.gemma4.cuda import assistant
    flags = dict(zip(FLAGS, values))
    extra = {"assistant_q_temperature_factor":.75,"assistant_q_top_k":16} if changed_policy else {}
    (tmp_path/"config.json").write_text(json.dumps(e4b_config()))
    cfg = SimpleNamespace(block=8, memory_bytes=lambda _: (1, 2))
    monkeypatch.setattr(e2b_engine, "companion_metadata", lambda *_: ("assistant", cfg))
    calls, admitted, closed = [], [], []
    stop = RuntimeError("late stream failure after constructor contracts")

    class Target:
        device = torch.device("cpu")

        def __init__(self, path, **options):
            pass

        def make_state(self, context, *, assistant=False):
            assert assistant and context == 64
            return SimpleNamespace(owner=self)

        def close(self):
            closed.append(True)

    class Head:
        def __init__(self, directory, target, context, *, device, **options):
            calls.append(options)
            self.target = target

        def bind_state(self, state):
            assert state.owner is self.target
            self.bound = state

        def make_state(self):
            assert self.bound.owner is self.target
            return object()

    def admission(*args, **kwargs):
        admitted.append(kwargs)
        return {"context_window": 64}

    def fail_stream(**_):
        raise stop

    monkeypatch.setattr(e2b_engine, "E2BTarget", Target)
    monkeypatch.setattr(assistant, "Gemma4AssistantHead", Head)
    monkeypatch.setattr(e2b_engine, "admit", admission)
    monkeypatch.setattr(template, "prepare_chat_template", lambda *_, **__: object())
    monkeypatch.setattr(tokens, "load_tokenizer", lambda _: SimpleNamespace(token_to_id=lambda _: 2))
    monkeypatch.setattr(tokens, "resolve_eos", lambda *_, **__: [1])
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    monkeypatch.setattr(torch.cuda, "Stream", fail_stream)
    with pytest.raises(RuntimeError) as caught:
        e2b_engine.E2BEngine(tmp_path, drafter="native-assistant", **flags,**extra)
    assert caught.value is stop and closed == [True]
    assert calls == [{**flags,**extra}]
    expected={"assistant_sampled_q":True} if flags["assistant_sampled_q"] else {}
    if changed_policy:
        expected["assistant_q_policy"]=True
    assert admitted == [expected]


@pytest.mark.parametrize("count,width", [(1, 1), (6, 257), (7, 1024)])
@pytest.mark.parametrize("proposal_policy",[False,True])
def test_q_budget_covers_actual_bytes_backed_arrays_packet_and_record_owners(count, width,proposal_policy):
    from tensorfold.engine.exact_sampling import Sampling
    rows = tuple(ProbabilityRow(np.arange(width), np.ones(width)) for _ in range(count))
    packet = ChainPacket(tuple(int(str(i % width)) for i in range(count)), rows, 17)
    original=Sampling(seed=1)
    derived=Sampling(seed=1,temperature=.75,top_k=16) if proposal_policy else original
    record = RecordedSampledChain(packet, original, None, proposal_sampling=derived)
    actual = (sys.getsizeof(record) + sys.getsizeof(record.__dict__) + sys.getsizeof(packet)
              + sys.getsizeof(packet.first_position) + sys.getsizeof(packet.tokens) + sys.getsizeof(packet.q_rows)
              + sum(sys.getsizeof(token) for token in packet.tokens))
    if proposal_policy:
        actual+=sys.getsizeof(derived)+sys.getsizeof(derived.__dict__)+sys.getsizeof(derived.temperature)
    for row in rows:
        actual += sys.getsizeof(row)
        actual += sum(sys.getsizeof(array) + sys.getsizeof(array.base) for array in (row.ids, row.probs))
        assert row.ids.dtype == np.int64 and row.probs.dtype == np.float64
        assert isinstance(row.ids.base, bytes) and isinstance(row.probs.base, bytes)
    assert e2b_engine.assistant_q_host_bytes(count, width,proposal_policy=proposal_policy) >= actual


def test_full_vocabulary_q_payload_not_old_256_candidate_estimate():
    count, width = 6, 262144
    charge = e2b_engine.assistant_q_host_bytes(count, width)
    assert charge >= count*width*(np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize)
    assert charge >= 24*1024*1024
    with pytest.raises(ValueError):
        e2b_engine.assistant_q_host_bytes(0, width)


def test_admission_retains_full_q_beside_p_but_reuses_sequential_device_preparation(monkeypatch, tmp_path):
    import tensorfold.cuda.capacity as capacity
    from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig
    cfg = E2BConfig.from_dict(e4b_config())
    draft = SimpleNamespace(hidden=256, backbone_hidden=2560, vocab=262144, block=8, parameter_bytes=1,
                            memory_bytes=lambda _: (1, 2), state_bytes=lambda _: 0)
    monkeypatch.setattr(e2b_engine, "companion_metadata", lambda *_: ("assistant", draft))
    monkeypatch.setattr(capacity, "headers", lambda _: {"fake": {"shape": [1], "dtype": "BF16"}})
    calls = []

    def admitted(*args, **kwargs):
        calls.append(kwargs)
        return {"context_window": 4096}

    monkeypatch.setattr(capacity, "admit", admitted)
    e2b_engine.admit(tmp_path, cfg, 4096, True, torch, tmp_path, 7, False)
    e2b_engine.admit(tmp_path, cfg, 4096, True, torch, tmp_path, 7, False, assistant_sampled_q=True)
    baseline, sampled = calls
    assert baseline["device_workspace_bytes"] == sampled["device_workspace_bytes"]
    assert sampled["host_workspace_bytes"]-baseline["host_workspace_bytes"] >= 6*262144*16


@pytest.mark.parametrize("options",[{"assistant_q_temperature_factor":True},{"assistant_q_temperature_factor":0},
                                    {"assistant_q_temperature_factor":float("inf")},{"assistant_q_top_k":True},
                                    {"assistant_q_top_k":-1}])
def test_invalid_q_parameters_reject_before_any_gpu(monkeypatch,tmp_path,options):
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:pytest.fail("invalid q parameters reached GPU"))
    with pytest.raises(ValueError):
        e2b_engine.E2BEngine(tmp_path,**options)


@pytest.mark.parametrize("options",[{"assistant_q_temperature_factor":.75},{"assistant_q_top_k":16}])
def test_nondefault_q_parameters_require_enabled_sampled_official_head_before_gpu(monkeypatch,tmp_path,options):
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:pytest.fail("unused q parameters reached GPU"))
    with pytest.raises(ValueError,match="sampled_q"):
        e2b_engine.E2BEngine(tmp_path,**options)


def test_q_vocab_override_bound_validated_before_loading_head_or_gpu(monkeypatch,tmp_path):
    (tmp_path/"config.json").write_text(json.dumps(e4b_config()))
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:pytest.fail("bad q vocabulary reached GPU"))
    monkeypatch.setattr(e2b_engine,"companion_metadata",lambda *_:pytest.fail("bad q vocabulary read checkpoint headers"))
    with pytest.raises(ValueError,match="vocabulary"):
        e2b_engine.E2BEngine(tmp_path,drafter="official",assistant_sampled_q=True,assistant_q_top_k=262145)


def test_q_temperature_overflow_rejects_before_target_prefill_and_reference_bypasses_candidate(monkeypatch):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
    engine=object.__new__(e2b_engine.E2BEngine)
    engine._assistant_sampled_q=True
    engine._assistant_q_parameters=(2.0,None)
    engine.head=SimpleNamespace(_reference_depth=0)
    engine.target=SimpleNamespace(config=SimpleNamespace(vocab=262144))
    calls=[]
    monkeypatch.setattr(Gemma4Engine,"generate",lambda *args,**kwargs:calls.append(True) or "native")
    sampling=Sampling(seed=1,temperature=1e308)
    with pytest.raises(ValueError,match="product"):
        engine.generate([1],2,sampling,lambda _:False)
    assert not calls
    engine.head._reference_depth=1
    assert engine.generate([1],2,sampling,lambda _:False)=="native"
    assert calls==[True]


def test_closed_engine_without_head_preserves_base_closed_error(monkeypatch):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
    engine=object.__new__(e2b_engine.E2BEngine)
    engine._assistant_sampled_q=True
    engine.head=None

    def closed(*args,**kwargs):
        raise RuntimeError("engine is closed")

    monkeypatch.setattr(Gemma4Engine,"generate",closed)
    with pytest.raises(RuntimeError,match="closed"):
        engine.generate([1],2,Sampling(seed=1),lambda _:False)
