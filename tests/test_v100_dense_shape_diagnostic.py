"""Bounded case/control contracts; actual CUDA arithmetic is Root-only pending."""
import json
import pytest
from tools import v100_dense_shape_diagnostic as d


def test_source_geometry_outputs_and_small_m_shapes_are_explicit():
    assert len(d.GEOMETRIES) == 7
    by_role = {g.role: g for g in d.GEOMETRIES}
    assert (by_role["hc_down"].n, by_role["hc_down"].k, by_role["hc_down"].output) == (320, 10240, "float32")
    assert by_role["hc_inject"].n == 4 and by_role["vocab_head_rank"].n == 31040
    assert by_role["mtp_fc_h_physical"].rows == (4, 8, 12)
    assert d.shapes(1) == (1, 4, 8, 16, 32)
    assert d.shapes(3) == (3, 4, 8, 16, 32)
    assert d.shapes(12) == (12, 16, 32)
    assert all(d.payload_bound(g) < d.GPU_LIMIT for g in d.GEOMETRIES)
    assert d.SAMPLES == 3 and d.UNROLL == 4 and d.REPLAYS == 8


@pytest.mark.parametrize("rows", [0, 33, True, 1.5])
def test_invalid_geometry_refuses_without_gpu_or_side_effect(rows):
    with pytest.raises(ValueError):
        d.shapes(rows)


def test_actual_metric_keeps_bits_errors_and_nonfinite_guard():
    torch = pytest.importorskip("torch")
    a = torch.tensor([[1., 2.]], dtype=torch.float16)
    b = torch.tensor([[1., 2.]], dtype=torch.float16)
    record = d.metrics(torch, a, b)
    assert record["bits_exact"] and record["max_abs"] == 0
    b[0, 1] = 3
    record = d.metrics(torch, a, b)
    assert not record["bits_exact"] and record["different_elements"] == 1 and record["max_abs"] == 1
    with pytest.raises(ValueError, match="nonfinite"):
        d.metrics(torch, a, torch.full_like(a, float("nan")))


def test_report_bound_and_units_are_not_unbounded(tmp_path, monkeypatch):
    path = tmp_path / "bounded.json"
    d.write(path, {"ns": 123, "scope": "microdiagnostic only"})
    assert json.loads(path.read_text())["ns"] == 123
    monkeypatch.setattr(d, "REPORT_LIMIT", 4)
    with pytest.raises(ValueError, match="envelope"):
        d.write(path, {"too_large": True})
