"""Cold boundary checks; no TileLang import, compiler, CUDA or invented error bound."""
import ast
import hashlib
import json
from pathlib import Path

import pytest

from tensorfold.cuda.sm70 import gdn_chunk64 as g3


@pytest.mark.parametrize("rows",[1,63,64,65,4096,8192])
def test_layout_exact_finite_payload(rows):
    plan=g3.layout(rows)
    assert plan.chunks==(rows+63)//64
    assert plan.workspace_bytes==rows*792+21


@pytest.mark.parametrize("rows",[0,-1,True,1.0,2**31])
def test_invalid_geometry_refused_before_dependencies(rows):
    with pytest.raises(ValueError):
        g3.prepare(rows,"cuda:0",workspace_budget_bytes=0,compiler=None)


def test_credit_and_interior_cut_refused_before_compiler_or_allocation():
    with pytest.raises(ValueError,match="credit"):
        g3.prepare(65,"cuda:0",workspace_budget_bytes=g3.layout(65).workspace_bytes-1,compiler=None)
    with pytest.raises(ValueError,match="cuts"):
        g3.prepare(65,"cuda:0",workspace_budget_bytes=10**9,compiler=None,cuts=(32,))


def test_generators_preserve_recorded_minimal_api_adaptation():
    root=Path(g3.__file__).parent
    source=(root/"gdn_chunk64_kernels.py").read_text()
    funcs={f.name:f for f in ast.parse(source).body if isinstance(f,ast.FunctionDef)}
    provenance=json.loads((root/"gdn_chunk64_provenance.json").read_text())
    for item in provenance["extractions"]:
        f=funcs[item["name"]]
        start=min([f.lineno]+[d.lineno for d in f.decorator_list])
        text="\n".join(source.splitlines()[start-1:f.end_lineno])
        assert hashlib.sha256(text.encode()).hexdigest()==item["adapted_sha256"]
    # Extraction drops original import-time CUDA query and serving dependencies.
    assert "T.gemm_v1(" not in source
    assert "flash_qla" not in source
    assert "get_device_properties" not in source


def test_independent_serial_reference_known_one_row_and_decay_order():
    torch=pytest.importorskip("torch")
    from tools.v100_gdn_chunk64_trial import reference
    q=torch.zeros(1,1,2,128,dtype=torch.float16)
    k=torch.zeros_like(q)
    v=torch.zeros(1,1,6,128,dtype=torch.float16)
    q[...,0]=1;k[...,0]=1;v[...,0]=1
    g=torch.zeros(1,1,6,dtype=torch.float32)
    beta=torch.ones_like(g)
    initial=torch.zeros(1,6,128,128,dtype=torch.float32)
    out,final=reference(q,k,v,g,beta,initial)
    assert torch.equal(initial,torch.zeros_like(initial))
    assert torch.equal(final[...,0,0],torch.ones(1,6,dtype=torch.float64))
    assert torch.equal(out[...,0],torch.full((1,1,6),128**-0.5,dtype=torch.float64))
    # Existing state decays before prediction. beta=1 then replaces its key
    # direction with v=1, irrespective of that decayed prediction.
    initial[...,0,0]=4
    g.fill_(-1)
    out2,final2=reference(q,k,v,g,beta,initial)
    assert torch.equal(out2,out)
    assert torch.equal(final2,final)


@pytest.mark.parametrize("failure",[None,"cumsum","kkt","core"])
def test_bound_nondefault_ffi_scope_restores_after_success_or_launch_failure(monkeypatch,failure):
    """Check our binding to the supported FFI scope, without pretending GPU proof."""
    import sys
    import threading
    from contextlib import contextmanager
    from types import SimpleNamespace

    current={"ffi":41}
    calls=[]

    @contextmanager
    def stream_context(device,stream):
        assert device=="ffi-cuda:0"
        previous=current["ffi"]
        current["ffi"]=stream
        try:
            yield
        finally:
            current["ffi"]=previous

    def kernel(name):
        def launch(*args):
            assert current["ffi"]==17  # an admitted nondefault raw stream
            calls.append(name)
            if failure==name:
                raise RuntimeError("injected launch failure")
        return launch

    fake=SimpleNamespace(
        cuda=SimpleNamespace(current_device=lambda:0,is_current_stream_capturing=lambda:False,
                             current_stream=lambda device:SimpleNamespace(cuda_stream=17)),
        is_inference_mode_enabled=lambda:True,
        autograd=SimpleNamespace(graph=SimpleNamespace(increment_version=lambda values:calls.append("versions"))))
    monkeypatch.setitem(sys.modules,"torch",fake)
    prepared=SimpleNamespace(device=SimpleNamespace(index=0),owner=threading.current_thread(),stream=17,
                             stream_context=stream_context,ffi_device="ffi-cuda:0",
                             metadata=("cu","offsets","dummy","has_initial"),layout=SimpleNamespace(chunks=2),
                             gsum="gsum",a="A",h="h",cumsum=kernel("cumsum"),kkt=kernel("kkt"),core=kernel("core"))
    bound=g3.Bound(prepared,tuple(range(8)))
    if failure is None:
        assert bound.run()==(6,7)
        assert calls==["cumsum","kkt","core","versions"]
    else:
        with pytest.raises(RuntimeError,match="injected launch failure"):
            bound.run()
        assert "versions" not in calls
    assert current["ffi"]==41
