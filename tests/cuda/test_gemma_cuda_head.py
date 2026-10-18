"""Actual Gemma vocabulary boundary for explicitly prepared GB10 head candidates."""

from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 1):
    pytest.skip("Gemma head candidate is scoped to GB10SM12.1", allow_module_level=True)

from tensorfold.families.gemma4.cuda.forward import Gemma4Target
from tensorfold.families.gemma4.cuda.head import prepare_head
from tensorfold.families.qwen3_5.cuda.weights import Plain


@pytest.fixture(scope="module")
def actual_head():
    generator = torch.Generator(device="cuda").manual_seed(181026)
    weight = torch.empty((262144, 2816), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.02, 0.02, generator=generator)
    hidden = torch.empty((17, 2816), dtype=torch.bfloat16, device="cuda").uniform_(
        -0.5, 0.5, generator=generator)
    incumbent = Plain(weight)
    singleton = torch.cat([incumbent(row[None]) for row in hidden])
    return incumbent, hidden, singleton


@pytest.mark.parametrize("reuse_rows", [2, 4, 8, 16])
def test_actual_head_keeps_singleton_bits_for_partitions_and_permutations(actual_head, reuse_rows, monkeypatch):
    incumbent, hidden, singleton = actual_head
    prepared = prepare_head(incumbent, reuse_rows=reuse_rows)
    assert prepared.incumbent is incumbent and prepared.weight is incumbent.weight
    assert prepared.nbytes() == incumbent.nbytes()
    # Preparation owns capability selection; hot calls never rediscover it.
    with monkeypatch.context() as scope:
        scope.setattr(torch.cuda, "get_device_capability", lambda *_: pytest.fail("hot capability discovery"))
        for rows in (1, 2, 3, 4, 7, 8, 15, 16, 17):
            assert torch.equal(prepared(hidden[:rows]).view(torch.int16), singleton[:rows].view(torch.int16)), rows
        partition = torch.cat([prepared(hidden[:1]), prepared(hidden[1:4]),
                               prepared(hidden[4:9]), prepared(hidden[9:])])
        assert torch.equal(partition.view(torch.int16), singleton.view(torch.int16))
        order = torch.tensor([16, 0, 15, 1, 14, 2, 13, 3, 12, 4, 11, 5, 10, 6, 9, 7, 8], device="cuda")
        assert torch.equal(prepared(hidden[order]).view(torch.int16), singleton[order].view(torch.int16))


def test_target_head_softcap_remains_exactly_once(actual_head):
    incumbent, hidden, singleton = actual_head
    prepared = prepare_head(incumbent, reuse_rows=8)
    target = object.__new__(Gemma4Target)
    target.config = SimpleNamespace(softcap=30.0)
    target.weights = SimpleNamespace(embed=prepared)
    actual = target.project_logits(hidden[:15])
    expected = torch.tanh(singleton[:15] / 30.0) * 30.0
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


@pytest.mark.parametrize("reuse_rows", [True, 0, 1, 3, 32])
def test_explicit_invalid_reuse_configuration_is_refused(reuse_rows):
    with pytest.raises(ValueError, match="reuse_rows"):
        prepare_head(None, reuse_rows=reuse_rows)


def test_explicit_unrelated_head_is_refused():
    unrelated = Plain(torch.zeros((128, 64), dtype=torch.bfloat16, device="cuda"))
    with pytest.raises(ValueError, match="262144,2816"):
        prepare_head(unrelated, reuse_rows=4)


def test_selected_ct_head_keeps_original_authority_and_raw_bits(actual_head, monkeypatch):
    from tensorfold.families.gemma4.cuda.head import select_head

    incumbent, hidden, singleton = actual_head
    selected = select_head(incumbent)
    assert selected.incumbent is incumbent and selected.weight is incumbent.weight
    assert selected.reuse_rows == 16 and selected.nbytes() == incumbent.nbytes()
    assert select_head(selected) is selected
    with monkeypatch.context() as scope:
        scope.setattr(torch.cuda, "get_device_capability", lambda *_: pytest.fail("hot capability discovery"))
        for rows in (1, 5, 16):
            assert torch.equal(selected(hidden[:rows]).view(torch.int16), singleton[:rows].view(torch.int16))


@pytest.mark.parametrize("capability", [(12, 0), (12, 2), (9, 0)])
def test_unqualified_device_selection_retains_original_head(actual_head, monkeypatch, capability):
    from tensorfold.families.gemma4.cuda.head import select_head

    incumbent, _, _ = actual_head
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: capability)
    assert select_head(incumbent) is incumbent


def test_unqualified_shape_selection_keeps_original_without_capability_probe(monkeypatch):
    from tensorfold.families.gemma4.cuda.head import select_head

    incumbent = Plain(torch.zeros((128, 64), dtype=torch.bfloat16, device="cuda"))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: pytest.fail("unnecessary capability probe"))
    assert select_head(incumbent) is incumbent
