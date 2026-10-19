"""DFlash2 proposal-head selection, ownership and admission on CPU provider controls; native math is separate."""

import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from safetensors.torch import save_file

from tensorfold.cuda.requant import Head, resident_bytes
from tensorfold.families import qwen3_5
from tensorfold.families.qwen3_5.cuda import b16, dflash2, draft_tree
from tensorfold.families.qwen3_5.cuda.weights import Plain

H, V, RANK = 64, 512, 8
SPANS = ((0, 256), (500, 512))


@pytest.fixture
def setup(tmp_path, monkeypatch):
    weights = {"fc.weight": torch.zeros(H, 5 * H, dtype=torch.bfloat16),
               "hidden_norm.weight": torch.ones(H, dtype=torch.bfloat16),
               "norm.weight": torch.ones(H, dtype=torch.bfloat16),
               "candidate_selector.hidden_projection.weight": torch.zeros(RANK, H, dtype=torch.bfloat16),
               "candidate_selector.predecessor_codebook": torch.zeros(V, RANK, dtype=torch.bfloat16),
               "candidate_selector.successor_codebook": torch.zeros(V, RANK, dtype=torch.bfloat16)}
    save_file(weights, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "hidden_size": H, "head_dim": 64, "num_attention_heads": 1, "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000}, "num_hidden_layers": 0,
        "sliding_window": 64, "dflash_config": {"mask_token_id": V - 1, "conv_group_size": 16}}))
    head = torch.zeros(V, H, dtype=torch.bfloat16)
    head[:16, 0] = torch.arange(1, 17).bfloat16() / 16
    head[300:316, 0] = torch.arange(16, 32).bfloat16()
    target = NS(config=NS(vocab=V), head=Plain(head), embed=object(), norm=torch.ones(1), quant="nvfp4")
    monkeypatch.setattr(draft_tree, "HOT_SPANS", SPANS)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    return tmp_path, target


@pytest.mark.parametrize("vocab", ["truncated", "full"])
@pytest.mark.parametrize("bits", [16, 8])
def test_selection_borrows_the_original_or_owns_only_the_requested_fp8_ranges(setup, vocab, bits):
    path, target = setup
    original, saved = target.head, target.head.weight.clone()
    draft = dflash2.DFlash2(path, target, bits=16, fast=False, draft_vocab=vocab, head_bits=bits)
    assert target.head is original and torch.equal(target.head.weight, saved)
    assert draft.in_vocab(7) and draft.in_vocab(505)
    assert draft.in_vocab(300) is (vocab == "full") and not draft.in_vocab(V)
    if bits == 16:
        if vocab == "full":
            assert draft.full_head is original and draft.head_ids is None and draft.sub_parts is None
        else:
            for part, lo, hi in draft.sub_parts:
                assert isinstance(part, Plain)
                assert part.weight.untyped_storage().data_ptr() == original.weight.untyped_storage().data_ptr()
                assert lo == 0 and hi == part.n
    else:
        ranges = ((0, V),) if vocab == "full" else SPANS
        copies = [draft.full_head] if vocab == "full" else [part for part, _, _ in draft.sub_parts]
        assert all(isinstance(part, Head) for part in copies)
        assert sum(part.nbytes() for part in copies) == sum(resident_bytes("fp8", b - a, H) for a, b in ranges)
        for part, (a, b) in zip(copies, ranges):
            top = saved[a:b].float().abs().amax(1)
            scale = torch.where(top > 0, top / 448, torch.ones_like(top))
            assert torch.equal(part.linear.cols[:b - a], scale)  # the selected rows, without an offset/remap error
            assert part.linear.w8.data_ptr() != original.weight.data_ptr()


def test_original_and_full_policies_map_native_head_columns_to_target_ids(setup, monkeypatch):
    path, target = setup
    monkeypatch.setattr(b16, "matmul", lambda x, weight: torch.nn.functional.linear(x, weight))
    h = torch.zeros(3, H, dtype=torch.bfloat16)
    h[:, 0] = 1
    for vocab, expected in (("truncated", set(range(16))), ("full", set(range(300, 316)))):
        draft = dflash2.DFlash2(path, target, bits=16, fast=False, draft_vocab=vocab)
        values, ids = draft._candidates(h)
        assert all(set(row.tolist()) == expected for row in ids)
        assert values.dtype == torch.float32
        for i in range(h.shape[0]):
            assert torch.equal(values[i], target.head(h[i:i + 1]).float()[0, ids[i]])


@pytest.mark.parametrize("vocab", [500, 600])
def test_new_profiles_refuse_disagreement_between_stored_head_and_declared_vocabulary(setup, vocab):
    path, target = setup
    target.config.vocab = vocab
    for bits in (16, 8):
        with pytest.raises(ValueError, match="rows must match"):
            dflash2.DFlash2(path / "absent", target, draft_vocab="full", head_bits=bits)


@pytest.mark.parametrize("options", [{"head_bits": True}, {"head_bits": 4}, {"draft_vocab": "adaptive"},
                                     {"world": 2, "draft_vocab": "full"}])
def test_invalid_or_unsupported_choices_refuse_before_opening_draft_files(setup, options):
    path, target = setup
    with pytest.raises(ValueError):
        dflash2.DFlash2(path / "absent", target, **options)


def test_private_copy_refuses_a_stored_quantized_head_before_reading_drafter(setup):
    path, target = setup
    target.head = object()
    with pytest.raises(ValueError, match="floating target head"):
        dflash2.DFlash2(path / "absent", target, head_bits=8)


@pytest.mark.parametrize("book", ["predecessor", "successor"])
@pytest.mark.parametrize("missing", [False, True])
def test_requested_vocabulary_must_be_covered_by_the_candidate_codebooks(setup, monkeypatch, book, missing):
    from safetensors.torch import load_file

    path, target = setup
    weights = load_file(path / "model.safetensors")
    name = f"candidate_selector.{book}_codebook"
    if missing:
        del weights[name]
    else:
        weights[name] = weights[name][:-1]
    save_file(weights, str(path / "model.safetensors"))
    monkeypatch.setattr("tensorfold.cuda.requant.Head", lambda *a, **kw: pytest.fail("copy built before coverage refusal"))
    with pytest.raises(ValueError, match="does not cover"):
        dflash2.DFlash2(path, target, draft_vocab="full", head_bits=8)


def test_family_passes_static_head_options_to_the_actual_cuda_constructor(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(qwen3_5, "gb10", lambda: False)
    monkeypatch.setattr("tensorfold.families.qwen3_5.cuda.exl3_load.quant_config", lambda path: None)
    monkeypatch.setattr("tensorfold.families.qwen3_5.cuda.engine.Qwen27Engine", lambda *a, **kw: seen.append(kw))
    qwen3_5.cuda_engine(tmp_path, drafter="secondary", draft_vocab="full", draft_head_bits=8)
    assert seen[0]["draft_vocab"] == "full" and seen[0]["draft_head_bits"] == 8
