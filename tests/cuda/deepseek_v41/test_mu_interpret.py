"""Multi-request (MU) Triton kernels on the CPU under the Triton interpreter (ported from the friend overlay's
tests/test_ds41mu_interpret.py): each check script runs in a subprocess with TRITON_INTERPRET=1 and compares
the batched kernels with the vendored upstream single-sequence kernels (bit-identical where the overlay
requires it), plus cross-request isolation and validation bits. GPU form of the same scripts:
DS41MU_DEVICE=cuda python tests/cuda/deepseek_v41/<script> (see each script's docstring)."""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))

try:
    import triton
except ImportError:  # pragma: no cover
    triton = None


def _run(script, *sections, timeout=3600, **extra):
    env = dict(os.environ, TRITON_INTERPRET="1", CUDA_VISIBLE_DEVICES="", DS41MU_DEVICE="cpu", **extra)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, script), *sections],
        check=False, env=env, capture_output=True, text=True, timeout=timeout,
    )
    print(proc.stdout)
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    assert "ALL OK" in proc.stdout
    return proc.stdout


@pytest.mark.skipif(triton is None, reason="needs triton (interpreter mode)")
def test_mu_state_and_paged_kernels_under_triton_interpreter():
    out = _run("mu_interpret_check.py")
    for name in ("token_req", "engram", "c2", "paged", "hardening", "prefix", "spec"):
        assert f"{name} OK" in out


@pytest.mark.skipif(triton is None, reason="needs triton (interpreter mode)")
def test_mu_packed_kernels_under_triton_interpreter():
    out = _run("mu_packed_check.py")
    for name in ("lossless", "insert", "rows", "prefill", "gather", "ring", "index", "select"):
        assert f"{name} OK" in out


@pytest.mark.skipif(triton is None, reason="needs triton (interpreter mode)")
def test_mu_reindex_and_topk_under_triton_interpreter():
    out = _run("mu_reindex_check.py")
    for name in ("topkseg", "scores", "blocks", "prefill", "select"):
        assert f"{name} OK" in out


@pytest.mark.skipif(triton is None, reason="needs triton (interpreter mode)")
def test_mu_reindex_tp8_index_heads_under_triton_interpreter():
    """TensorFold TP8: 4 indexer heads per rank (scores incl. FP32 out, and the select glue)."""
    out = _run("mu_reindex_check.py", "scores", "select", DS41MU_INDEX_HEADS="4")
    for name in ("scores", "select"):
        assert f"{name} OK" in out
