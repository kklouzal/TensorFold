"""A decode window's prefill-form expert items by chip (host): 16 pairs only where measured (dev-01's sm_120 with 34
SMs); every other chip, the GB10 among them, keeps the 64-pair items, and no CUDA device means no kernel to pick for."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import experts  # noqa: E402


def test_only_the_measured_chip_takes_items_of_16(monkeypatch):
    chips = {(12, 0, 34): experts.TILE, (12, 1, 48): experts.PREFILL_TILE, (12, 0, 188): experts.PREFILL_TILE,
             (12, 0, 36): experts.PREFILL_TILE, (9, 0, 132): experts.PREFILL_TILE, (8, 9, 128): experts.PREFILL_TILE}
    for (major, minor, sms), want in chips.items():
        props = SimpleNamespace(major=major, minor=minor, multi_processor_count=sms)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device, p=props: p)
        assert experts.window_tile("cuda:0") == want, (major, minor, sms)
    assert experts.window_tile("cpu") == experts.PREFILL_TILE
