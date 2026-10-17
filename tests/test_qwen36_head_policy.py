"""Preparation and row dispatch of the actual head object without a GPU."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.qwen3_5.cuda import b16
from tensorfold.families.qwen3_5.cuda.weights import Plain
from tensorfold.families.qwen3_5_moe.cuda import head as policy


def weight(*, shape=(248320, 2048), dtype=torch.bfloat16, device="cuda", contiguous=True):
    return NS(shape=shape, dtype=dtype, device=NS(type=device), is_contiguous=lambda: contiguous)


def test_selected_head_preserves_weight_ownership_and_binds_device_once(monkeypatch):
    devices, prepared = [], []
    native = lambda *args: prepared.append(args) or "selected"
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: devices.append(dev) or (12, 0))
    monkeypatch.setattr(b16, "_ext", lambda: NS(b16_linear_rows=native))
    source = weight()
    target = NS(head=Plain(source))
    target.head = policy.prepare_head(target.head)
    drafter_head = target.head
    assert isinstance(target.head, Plain)
    assert target.head.weight is source
    assert drafter_head is target.head
    assert devices == [source.device]
    for rows, reuse in ((2, 2), (3, 8), (4, 8), (7, 8), (8, 8), (12, 16), (15, 16), (16, 16)):
        x = torch.zeros(rows, 2048, dtype=torch.float32)
        assert target.head(x) == "selected"
        assert prepared[-1][0].dtype == torch.bfloat16
        assert prepared[-1][1] is source
        assert prepared[-1][2] == reuse
    assert devices == [source.device]
    assert policy.prepare_head(target.head) is target.head


@pytest.mark.parametrize("options", [{"device": "cpu"}, {"dtype": torch.float16}, {"shape": (512, 2048)},
                                   {"shape": (248320, 1024)}, {"contiguous": False}])
def test_other_storage_regions_do_not_query_device_or_compile(monkeypatch, options):
    def unexpected(*args):
        pytest.fail("unselected head queried a device or compiled an extension")

    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected)
    monkeypatch.setattr(b16, "_ext", unexpected)
    original = Plain(weight(**options))
    assert policy.prepare_head(original) is original


@pytest.mark.parametrize("capability", [(8, 9), (9, 0), (10, 0), (12, 1)])
def test_unmeasured_device_keeps_original_object(monkeypatch, capability):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: capability)
    monkeypatch.setattr(b16, "_ext", lambda: pytest.fail("unmeasured device compiled a specialization"))
    original = Plain(weight())
    assert policy.prepare_head(original) is original


def test_singleton_and_unmeasured_rows_keep_original_dispatch(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: (12, 0))
    monkeypatch.setattr(b16, "_ext", lambda: NS(b16_linear_rows=lambda *args: pytest.fail("unmeasured row dispatch")))
    monkeypatch.setattr(b16, "matmul", lambda *args: calls.append(args) or "incumbent")
    prepared = policy.prepare_head(Plain(weight()))
    for rows in (1, 5, 6, 9, 10, 11, 13, 14, 17, 32, 64, 128):
        x = torch.zeros(rows, 2048, dtype=torch.bfloat16)
        assert prepared(x) == "incumbent"
        assert calls[-1] == (x, prepared.weight)


def test_selected_prompt_policy_is_inherited_unchanged(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: (12, 0))
    monkeypatch.setattr(b16, "_ext", lambda: NS(b16_linear_rows=None))
    monkeypatch.setattr(b16, "prompt", lambda *args: args)
    prepared = policy.prepare_head(Plain(weight()))
    x = torch.zeros(4, 2048, dtype=torch.bfloat16)
    assert prepared.prefill(x) == (x, prepared.weight)


def test_preparation_failure_keeps_the_original_head_owned(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda dev: (12, 0))
    monkeypatch.setattr(b16, "_ext", lambda: (_ for _ in ()).throw(RuntimeError("compile")))
    target = NS(head=Plain(weight()))
    original = target.head
    with pytest.raises(RuntimeError, match="compile"):
        target.head = policy.prepare_head(target.head)
    assert target.head is original
