"""Small source-workflow fixture, never representative component timing."""
from types import SimpleNamespace

import pytest
import torch

from tools import gemma4_cuda_moe_bench as bench

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip('component workflow requires one native SM12 device', allow_module_level=True)

from nvfp4_a4_inputs import PREFIX, Context
from test_gemma_cuda_nvfp4_experts import _prepared

from tensorfold.cuda.nvfp4.grouped_a4 import _ext
from tensorfold.families.gemma4.cuda.nvfp4_experts import _combine, load_experts


@pytest.fixture(scope='module')
def prepared_owners():
    owners = []
    for bank_index in range(2):
        source = Context(count=128, width=64, dims=128, uniform=True)
        if bank_index:
            for name, value in source.payload.items():
                if name.endswith(('gate_proj.input_global_scale', 'up_proj.input_global_scale')):
                    value.mul_(2)  # Distinct fixture calibrations, never altered real-bank data.
        wrapped = load_experts(source.read, PREFIX + '.experts', count=128, width=64, dims=128,
                               device='cuda:0', calibrated=_prepared(source))
        reader = SimpleNamespace(get=lambda name, device='cpu', values=source.payload: values[name].to(device))
        group = [bench.prepare_owner(torch, bank_index, wrapped.bank, 1, route_index, 'scan', 1137)
                 for route_index in (0, 1)]
        bench._reference(torch, reader, PREFIX + '.experts', group)
        owners.extend(group)
    assert len({owner.buf.pick.data_ptr() for owner in owners}) == 4
    assert len({owner.up_base[9].data_ptr() for owner in owners}) == 4
    yield owners
    torch.cuda.synchronize()


@pytest.fixture
def graph():
    value = torch.cuda.CUDAGraph(keep_graph=True, num_rng_states=0)
    yield value
    # Pytest owns aggregation of test and fixture-teardown failures. A failed
    # sync stops dependent reset; its traceback retains the unretired graph.
    torch.cuda.synchronize()
    value.reset()


@pytest.mark.parametrize('arm', (bench.PUBLIC, *bench.ARMS))
def test_prebound_real_dependency_chain_and_graph_replay(prepared_owners, graph, arm):
    ext = _ext()
    operations = bench.bindings(ext, prepared_owners)[arm]
    down, combine = ext.down_plan, _combine[(1, 1)]
    bench.poison_owners(prepared_owners)
    bench._macro(operations, down, combine, 128)
    torch.cuda.synchronize()
    bench.validate_owners(torch, prepared_owners)
    with bench._OwnedContext(torch.cuda.graph(graph)):
        bench._macro(operations, down, combine, 128)
    graph.instantiate()
    metadata = graph.get_graph_data()
    assert len(metadata['nodes']) >= 3 * len(prepared_owners)
    bench.poison_owners(prepared_owners)
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    bench.validate_owners(torch, prepared_owners)


def test_full_byte_validation_detects_stale_output(prepared_owners):
    ext = _ext()
    bench._macro(bench.bindings(ext, prepared_owners)[bench.PUBLIC], ext.down_plan, _combine[(1, 1)], 128)
    torch.cuda.synchronize()
    bench.validate_owners(torch, prepared_owners)
    prepared_owners[0].up_base[9].view(-1)[0].bitwise_xor_(1)
    with pytest.raises(bench.OutputMismatchError) as caught:
        bench.validate_owners(torch, prepared_owners)
    assert caught.value.owner == 0 and caught.value.field == 'codes'
