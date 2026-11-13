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


def test_native_initial_preservation_rejects_signed_zero_bit_changes():
    import torch

    from tools import v100_gdn_chunk64_trial as trial

    initial=torch.tensor([0.0,-0.0],dtype=torch.float32)
    changed=torch.tensor([-0.0,0.0],dtype=torch.float32)
    assert torch.equal(initial,changed)  # Value equality cannot prove read-only storage.
    assert not trial.unchanged_initial(initial,changed)
    assert trial.unchanged_initial(initial,initial.clone())


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


@pytest.mark.parametrize("mutate_and_restore",[False,True])
def test_trial_keeps_full128_and129_cases_and_non_aligned_split_extents(monkeypatch,mutate_and_restore):
    """CPU protocol control; synthetic calls do not qualify the GPU algorithm."""
    from types import SimpleNamespace

    import torch

    from tools import v100_gdn_chunk64_trial as trial

    original_reference=trial.reference
    latest_before=[None]
    restored=[]

    def observe_reference(q,k,v,gate,beta,initial):
        # The serial oracle must consume the saved input, even if the native
        # procedure illegally changed its own initial bank.
        assert trial.unchanged_initial(initial,latest_before[0])
        return original_reference(q,k,v,gate,beta,initial)

    class Compiler:
        def __init__(self, *args):
            pass

        def inspect(self):
            return {"scope": "CPU protocol fixture only"}

    class Prepared:
        def __init__(self, rows):
            self.rows, self.layout = rows, g3.layout(rows)

        def bind(self, q, k, v, gate, beta, initial, output, final):
            assert all(tensor.shape[1] == self.rows for tensor in (q, k, v, gate, beta, output))
            if final is initial:
                if mutate_and_restore:
                    initial.copy_(self.original)
                    restored.append(True)
                raise ValueError("fixture alias refusal")
            self.original=initial.clone()

            def run():
                latest_before[0]=self.original
                values, state = original_reference(q, k, v, gate, beta, initial)
                output.copy_(values)
                final.copy_(state)
                if mutate_and_restore:
                    initial.add_(1)

            return SimpleNamespace(run=run)

    monkeypatch.setattr(g3, "Compiler", Compiler)
    monkeypatch.setattr(trial,"reference",observe_reference)
    monkeypatch.setattr(g3, "prepare", lambda rows, *args, **kwargs: Prepared(rows))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *args: "CPU fixture")
    monkeypatch.setattr(trial, "nondefault_stream_trial",
                        lambda *args: {"scope": "unrun in CPU protocol fixture"})
    args = SimpleNamespace(owned_tilelang_attestation=None, cuda_home="unused", nvcc_sha256="unused",
                           device="cpu")
    result = trial.run(args, {"results": []})
    records = {(row["rows"], row["case"]): row for row in result["results"]}
    assert set(records) == {(rows, case) for rows in (1, 63, 64, 65, 128, 129)
                            for case in ("zero", "nonzero_initial", "raw_underflow")}
    assert all(row["initial_unchanged"] is (not mutate_and_restore) for row in records.values())
    assert len(restored)==(18 if mutate_and_restore else 0)
    for rows, expected in ((128, [65, 63]), (129, [65, 64])):
        record = records[rows, "nonzero_initial"]
        assert record["continuation"]["part_rows"] == expected
        assert record["continuation"]["initial_unchanged"] is (not mutate_and_restore)
        assert record["continuation"]["output_vs_serial"]["finite"]
        assert record["continuation"]["final_vs_serial"]["finite"]


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


@pytest.mark.parametrize("ffi",["0.1.10","0.1.15","0.1.16","0.1.17","0.2.0"])
def test_compiler_refuses_unknown_ffi_or_unattested_tilelang_before_stream_api_use(tmp_path,monkeypatch,ffi):
    """Dependency-only control; mock toolkit metadata, never execute a compiler."""
    from types import SimpleNamespace

    class DependencyInspectionReached(Exception):
        pass

    nvcc=tmp_path/"bin/nvcc"
    nvcc.parent.mkdir()
    nvcc.write_bytes(b"mock-nvcc-identity")
    monkeypatch.setenv("CUDA_HOME",str(tmp_path))
    monkeypatch.setattr(g3.shutil,"which",lambda name:str(nvcc))
    monkeypatch.setattr(g3.subprocess,"run",lambda *args,**kwargs:SimpleNamespace(stdout="release 12.9, mock"))
    packages={"tilelang":"0.1.10","apache-tvm-ffi":ffi,"torch":"2.16.dev-test","triton":"3.2.0"}
    monkeypatch.setattr(g3.importlib.metadata,"version",packages.__getitem__)

    def requires(name):
        raise DependencyInspectionReached

    monkeypatch.setattr(g3.importlib.metadata,"requires",requires)
    compiler=g3.Compiler(str(tmp_path),hashlib.sha256(nvcc.read_bytes()).hexdigest())
    if ffi=="0.1.16":
        with pytest.raises(RuntimeError,match="explicit owned artifact"):
            compiler.inspect()
    else:
        with pytest.raises(RuntimeError,match="apache-tvm-ffi==0.1.16"):
            compiler.inspect()
