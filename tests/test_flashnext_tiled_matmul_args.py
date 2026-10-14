"""Flash Next's tiled matmul refuses launch overrides its kernel cannot take (a K split dropping or overrunning
groups, a program tile not inside one stored tile, an unknown load-ahead depth) before any allocation or launch."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402

N, K = 64, 256                       # 8 groups of 32 inputs


@pytest.fixture
def launches(monkeypatch):
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: calls.append((grid, kwargs))

    def group_sums(x):
        calls.append("group_sums")
        return torch.zeros((x.shape[0], x.shape[1] // qmm.GS))

    monkeypatch.setattr(qmm, "_qmm", Kernel())
    monkeypatch.setattr(qmm, "group_sums", group_sums)
    return calls


def _tiled():
    scales = torch.ones((N, K // qmm.GS), dtype=torch.bfloat16)
    return qmm.make_q4(torch.zeros((N, K // 8), dtype=torch.int32), scales, scales, "tiled")


@pytest.mark.parametrize("override", [dict(pd=3), dict(pd=-1), dict(block_n=8), dict(block_n=48),
                                      dict(block_n=128), dict(sk=3), dict(sk=16), dict(sk=-8)], ids=str)
def test_bad_overrides_are_refused_before_any_launch(launches, override):
    with pytest.raises(ValueError, match="^matmul: "):
        qmm.matmul(torch.zeros((5, K), dtype=torch.bfloat16), _tiled(), reduce=False, **override)
    assert launches == []


@pytest.mark.parametrize("override", [dict(), dict(pd=0), dict(pd=2), dict(block_n=16), dict(block_n=32),
                                      dict(sk=1), dict(sk=8)], ids=str)
def test_valid_overrides_reach_the_kernel(launches, override):
    qmm.matmul(torch.zeros((5, K), dtype=torch.bfloat16), _tiled(), reduce=False, **override)
    (grid, kwargs), = [call for call in launches if call != "group_sums"]
    assert grid[2] == kwargs["SK"]
    for name, constexpr in (("pd", "PD"), ("block_n", "BLOCK_N"), ("sk", "SK")):
        if name in override:
            assert kwargs[constexpr] == override[name]
