"""Real safetensors/headers/readback verification of the offline EXL3 fixture."""
import json
import math

import pytest

torch = pytest.importorskip("torch")
from flashnext_exl3_fixture import EXPERT_WIDTHS, PROJECTIONS, write_checkpoint  # noqa: E402
from tensorfold.cuda.exl3 import format as fmt  # noqa: E402
from tensorfold.families.qwen4_exp import ram_experts  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weight_types import Config  # noqa: E402


@pytest.mark.parametrize("mtp", [False,True])
def test_serialized_complete_fixture_is_deterministic_readable_and_admitted(tmp_path,mtp):
    a,b = tmp_path / "a",tmp_path / "b"
    receipt = write_checkpoint(a,mtp=mtp)
    assert receipt == write_checkpoint(b,mtp=mtp)
    pack = Pack(a)
    config = Config.read(a)
    assert config.layers == 2 and config.head_dim == 256 and config.ple_layers == []
    assert config.nv == 24 and config.nk == 8 and config.conv_dim == 5120
    assert pack.has("mtp.fc_embedding.trellis") is mtp
    headers = fmt.read_header(a / "model.safetensors")
    scanned = fmt.scan(a)
    assert len(headers) == receipt["tensor_count"]
    assert len(scanned.groups) >= 3 * 4 * (2 + int(mtp))
    largest = 3 * 256 * 128 * 4 // 8
    plan = ram_experts.layout(a,(3*largest+13)/2**30,mtp=mtp)
    assert plan.entry_bytes == largest and plan.slots == 3 and plan.gpu_bytes == 3*largest
    expected = (2+int(mtp))*sum(sum(widths)*256*128//16 for widths in EXPERT_WIDTHS)
    assert plan.host_bytes == expected and plan.host_bytes < (2+int(mtp))*4*largest
    assert plan.metadata_host_bytes == 28*4*(2+int(mtp))
    assert plan.control_device_bytes == 96 and plan.control_host_bytes == 192
    base = "model.language_model.layers.0.mlp.experts.0"
    for part,k2 in zip(PROJECTIONS,EXPERT_WIDTHS[0]):
        name = base + "." + part
        trellis = pack.get(name + ".trellis")
        assert fmt.bits_of(trellis.shape) == k2 / 2
        assert pack.get(name + ".mul1").item() == fmt.MARKERS["mul1"] - 2**32
        assert pack.get(name + ".suh").dtype == torch.float16
        file,begin,end,dtype,shape = pack.entry(name + ".trellis")
        assert end-begin == math.prod(shape)*2 and dtype == "I16"
        assert file == "model.safetensors"
    index = json.loads((a / "model.safetensors.index.json").read_text())
    assert set(index["weight_map"]) == set(headers)
    assert not torch.cuda.is_initialized()
