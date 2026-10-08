"""Finite original FP16 scale boundaries shared by resident/compact loaders."""
import contextlib
import struct
from types import SimpleNamespace

import pytest

torch=pytest.importorskip("torch")
from flashnext_exl3_fixture import write_checkpoint  # noqa: E402
from tensorfold.cuda.exl3 import experts as native,host_experts as compact  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import exl3  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack  # noqa: E402
from test_exl3_host_experts import Checkpoint,PREFIX,SHARED  # noqa: E402


@pytest.mark.parametrize("value",[float("nan"),float("inf"),-float("inf")])
def test_source_scale_validator_rejects_nonfinite_with_exact_part_context(value):
    source=torch.tensor([1.,value,-0.],dtype=torch.float16)
    with pytest.raises(ValueError,match="layer.17.shared_expert.down_proj.svh:.*nonfinite"):
        native.validate_scale_payload(source,"layer.17.shared_expert.down_proj.svh")
    assert not torch.cuda.is_initialized()


def test_source_validator_preserves_every_finite_half_bit_pattern():
    # All finite half patterns, including negative zero/subnormals/maxima.
    bits=torch.arange(65536,dtype=torch.int32).short()
    values=bits.view(torch.float16)
    finite=values[torch.isfinite(values)].contiguous()
    original=finite.view(torch.int16).clone()
    assert native.validate_scale_payload(finite,"finite.bits") is finite
    assert torch.equal(finite.view(torch.int16),original)


@pytest.mark.parametrize("source",[torch.ones(16,dtype=torch.float32),torch.ones((2,8),dtype=torch.float16),
                                    torch.ones(16,dtype=torch.float16)[::2],torch.empty(8,device="meta",dtype=torch.float16),
                                    torch.empty(0,dtype=torch.float16)])
def test_source_validator_rejects_unproved_representation_without_coercion(source):
    with pytest.raises(ValueError,match="contiguous CPU FP16 vector"):
        native.validate_scale_payload(source,"bad.representation")


@pytest.fixture(scope="module")
def complete_checkpoint(tmp_path_factory):
    path=tmp_path_factory.mktemp("exl3-finite-source")
    write_checkpoint(path)
    return path


@contextlib.contextmanager
def corrupt_half(path,name,value):
    pack=Pack(path)
    file,begin,end,dtype,shape=pack.entry(name)
    assert dtype=="F16" and end-begin==shape[0]*2
    target=path/file
    with target.open("r+b") as stream:
        stream.seek(begin)
        original=stream.read(2)
        stream.seek(begin)
        stream.write(struct.pack("<e",value))
    try:
        yield
    finally:
        with target.open("r+b") as stream:
            stream.seek(begin)
            stream.write(original)


@pytest.mark.parametrize("base",["model.language_model.layers.0.mlp","model.language_model.layers.1.mlp","mtp.layers.0.mlp"])
@pytest.mark.parametrize("expert",["experts.0","shared_expert"])
@pytest.mark.parametrize("projection",["gate_proj","up_proj","down_proj"])
@pytest.mark.parametrize("part",["suh","svh"])
@pytest.mark.parametrize("value",[float("nan"),float("inf"),-float("inf")])
def test_both_file_loaders_reject_early_late_main_mtp_routed_shared_source_scales(
        complete_checkpoint,base,expert,projection,part,value,monkeypatch):
    name=f"{base}.{expert}.{projection}.{part}"
    def forbidden_prepare(*args,**kwargs):
        raise AssertionError("native GPU metadata construction began before source rejection")
    monkeypatch.setattr(native,"prepare",forbidden_prepare)
    with corrupt_half(complete_checkpoint,name,value):
        with pytest.raises(ValueError,match="nonfinite") as resident:
            exl3.expert_table(Pack(complete_checkpoint),base+".experts",3,base+".shared_expert","cpu")
        assert name in str(resident.value)
        with pytest.raises(ValueError,match="nonfinite") as offload:
            compact.load_compact(Pack(complete_checkpoint),base+".experts",3,base+".shared_expert",device="cpu")
        assert name in str(offload.value)
    assert not torch.cuda.is_initialized()


def test_packed_sign_scale_expansion_is_finite_and_bit_equal_for_both_loaders(tmp_path,monkeypatch):
    # A minimal real indexed safetensors file made from the existing independent
    # packed-sign expert fixture; native prepare is observed at its CPU boundary.
    import json
    pk=Checkpoint(signs=True)
    header,chunks,offset={},[],0
    for name,value in pk.values.items():
        raw=value.reshape(-1).view(torch.uint8).numpy().tobytes()
        dtype="I32" if value.dtype==torch.int32 else "I16"
        header[name]={"dtype":dtype,"shape":list(value.shape),"data_offsets":[offset,offset+len(raw)]}
        offset+=len(raw)
        chunks.append(raw)
    encoded=json.dumps(header).encode()
    (tmp_path/"model.safetensors").write_bytes(struct.pack("<Q",len(encoded))+encoded+b"".join(chunks))
    (tmp_path/"model.safetensors.index.json").write_text(json.dumps({"weight_map":{key:"model.safetensors" for key in header}}))
    captured=[]
    def capture_prepare(gate,up,down,*args,**kwargs):
        captured.extend((gate,up,down))
        return SimpleNamespace(keep=[])
    monkeypatch.setattr(native,"prepare",capture_prepare)
    exl3.expert_table(Pack(tmp_path),PREFIX,pk.count,SHARED,"cpu")
    _,tables=compact.load_compact(Pack(tmp_path),PREFIX,pk.count,SHARED,device="cpu")
    targets=((tables.suh_g,tables.svh_g),(tables.suh_u,tables.svh_u),(tables.suh_d,tables.svh_d))
    for projection,(inputs,outputs) in zip(captured,targets):
        for expert,(_,source_in,source_out) in enumerate(projection):
            assert set(source_in.tolist())<= {-1.,1.} and set(source_out.tolist())<= {-1.,1.}
            assert torch.equal(source_in.view(torch.int16),inputs[expert].view(torch.int16))
            assert torch.equal(source_out.view(torch.int16),outputs[expert].view(torch.int16))
    words=pk.values[PREFIX+".0.gate_proj.su"].tolist()
    expected=[-1. if (int(word)&65535)&(1<<bit) else 1. for word in words for bit in range(16)]
    assert torch.equal(tables.suh_g[0],torch.tensor(expected,dtype=torch.float16))
