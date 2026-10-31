"""SM70 attention carries independent immutable Q4 format fields to native code."""

from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.sm70 import attention_native as native


def operands(kbits, vbits, kcodec=0, vcodec=0):
    workspace = NS(capacity=8192, scores=object(), probabilities=object(), partials=object())
    scratch = NS(native_workspace=workspace, sparse=object(), nk=object(), ids=object(), qsa=True)
    q = NS(shape=(3, 3, 256), device="cuda")
    key = NS(shape=(8192, 1, 128 if kcodec else 256))
    value = NS(shape=(8192, 1, 128 if vcodec else 256))
    return [q, key, value, object(), scratch, 3, 0.0625, object(), 8192,
            object() if kbits else None, object() if vbits else None, kbits, vbits]


@pytest.mark.parametrize("kb,vb", [(0, 0), (0, 8), (8, 0), (8, 8)])
def test_raw_and_int8_keep_the_existing_native_entry_and_argument_order(monkeypatch, kb, vb):
    calls = []
    monkeypatch.setattr(native, "extension", lambda: NS(run=lambda *args: calls.append(args)))
    args = operands(kb, vb)
    assert native.attention(*args) is args[7]
    packet = calls[0]
    assert len(packet) == 19
    assert packet[:4] == tuple(args[:4])
    assert packet[11:17] == (args[9], args[10], 3, 8192, kb, vb)
    assert packet[17:] == (True, 0.0625)


@pytest.mark.parametrize("kb,kc,vb,vc", [(4, 4, 4, 4), (4, 4, 0, 0), (4, 4, 8, 0),
                                       (0, 0, 4, 4), (8, 0, 4, 4), (4, 5, 4, 5),
                                       (4, 5, 4, 4), (4, 4, 4, 5), (4, 5, 0, 0),
                                       (0, 0, 4, 5), (4, 5, 8, 0), (8, 0, 4, 5)])
def test_packed_sides_keep_independent_codecs_and_scales(monkeypatch, kb, kc, vb, vc):
    calls = []
    monkeypatch.setattr(native, "extension", lambda: NS(run_rotor=lambda *args: calls.append(args)))
    args = operands(kb, vb, kc, vc)
    assert native.attention(*args, kcodec=kc, vcodec=vc) is args[7]
    packet = calls[0]
    assert len(packet) == 21 and packet[-2:] == (kc, vc)
    assert packet[11:17] == (args[9], args[10], 3, 8192, kb, vb)
    assert args[4].native_workspace.capacity == 8192


@pytest.mark.parametrize("kb,kc,vb,vc", [(4, 0, 0, 0), (4, 3, 4, 4), (6, 4, 4, 4),
                                       (8, 4, 0, 0), (4, True, 0, 0)])
def test_unqualified_codec_cannot_allocate_or_reach_native(monkeypatch, kb, kc, vb, vc):
    monkeypatch.setattr(native, "extension", lambda: pytest.fail("invalid format reached native"))
    args = operands(kb, vb, kc, vc)
    del args[4].native_workspace
    with pytest.raises(ValueError, match="SM70 KV"):
        native.attention(*args, kcodec=kc, vcodec=vc)
    assert not hasattr(args[4], "native_workspace")


@pytest.mark.parametrize("side", [1, 2])
def test_physical_width_must_match_each_side_before_workspace_allocation(monkeypatch, side):
    monkeypatch.setattr(native, "extension", lambda: pytest.fail("invalid shape reached native"))
    args = operands(4, 4, 4, 4)
    args[side].shape = (8192, 1, 256)
    del args[4].native_workspace
    with pytest.raises(ValueError, match="qualified"):
        native.attention(*args, kcodec=4, vcodec=4)
    assert not hasattr(args[4], "native_workspace")
