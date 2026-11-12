"""Optional explicit-artifact FA26.10 component gates, not model/PPL/P3 claims.

Set TENSORFOLD_FA_V100_ARTIFACT to a controlled source-built native ELF. The
normal checkout does not install or select this optional external provider.
"""
import importlib.util
import json
import os
from pathlib import Path
import threading

import pytest
import torch

from tensorfold.cuda.sm70 import flash_v100_forward as fa

pytestmark = pytest.mark.skipif(not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (7, 0), reason='requires actual SM70')


@pytest.fixture(scope='module')
def native():
    value = os.environ.get('TENSORFOLD_FA_V100_ARTIFACT')
    if value is None:
        pytest.skip('unselected FA26.10 component requires an explicitly verified source-built ELF')
    path = Path(value).resolve(strict=True)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tensors(batch=1, rows=1, heads=3, kvheads=1, keys=1):
    q = torch.zeros(batch, rows, heads, 256, device='cuda', dtype=torch.float16)
    k = torch.zeros(batch, keys, kvheads, 256, device='cuda', dtype=torch.float16)
    v = torch.ones_like(k)
    return q, k, v, torch.full_like(q, -711.), torch.full((batch, heads, rows), -713., device='cuda')


@pytest.mark.parametrize('batch,rows,heads,kvheads', [(1, 1, 1, 1), (1, 4, 12, 1), (2, 3, 12, 2), (1, 33, 3, 1)])
def test_single_key_gqa_batch_and_query_tile_mapping_exact_known_answer(native, batch, rows, heads, kvheads):
    q, k, v, out, lse = tensors(batch, rows, heads, kvheads)
    for index in range(kvheads):
        v[:, :, index].fill_(index + 1)
    expected = v[:, :1].repeat_interleave(heads // kvheads, dim=2).expand_as(out)
    with fa.prepare(native, q, k, v, out, lse) as owner:
        owner.run(q, k, v, out, lse)
        assert torch.equal(out, expected)
        assert torch.equal(lse, torch.zeros_like(lse))
        info = owner.settings()
        assert info['shared_bytes'] == 90368 and info['threads'] == 512
        assert info['source_commit'] == fa.SOURCE_COMMIT and info['binary_version'] == 70


def reference(q, k, v):
    # Independent CPU Float64 SDPA; no optional CUDA attention backend/oracle.
    q, k, v = (value.cpu().double() for value in (q, k, v))
    copies = q.shape[2] // k.shape[2]
    k, v = (value.repeat_interleave(copies, dim=2) for value in (k, v))
    scores = q.permute(0, 2, 1, 3) @ k.permute(0, 2, 3, 1) * 0.0625
    return ((scores.softmax(-1) @ v.permute(0, 2, 1, 3)).permute(0, 2, 1, 3), scores.logsumexp(-1))


@pytest.mark.parametrize('keys,mode', [(17, 'normal'), (64, 'normal'), (65, 'normal'), (257, 'normal'), (17, 'tiny'), (65, 'large')])
def test_finite_tile_boundaries_and_adverse_values_report_unfitted_error(native, keys, mode, record_property):
    q, k, v, out, lse = tensors(rows=2, heads=12, keys=keys)
    generator = torch.Generator(device='cuda').manual_seed(26010 + keys)
    if mode == 'large':
        q.fill_(4096.)
        k.fill_(4096.)
    else:
        factor = 1e-6 if mode == 'tiny' else .1
        q.copy_(torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator) * factor)
        k.copy_(torch.randn(k.shape, dtype=k.dtype, device=k.device, generator=generator) * factor)
    v.copy_(torch.randn(v.shape, dtype=v.dtype, device=v.device, generator=generator) * .1)
    gold, gold_lse = reference(q, k, v)
    with fa.prepare(native, q, k, v, out, lse) as owner:
        owner.run(q, k, v, out, lse)
        actual, actual_lse = out.cpu().double(), lse.cpu().double()
        assert torch.isfinite(actual).all() and torch.isfinite(actual_lse).all()
        record_property('fa26_forward_diagnostic', json.dumps({'keys': keys, 'mode': mode,
            'out_max_abs_vs_CPU_float64': float((actual - gold).abs().max().item()),
            'lse_max_abs_vs_CPU_float64': float((actual_lse - gold_lse).abs().max().item()),
            'settings': owner.settings(), 'quality_accepted': False, 'PPL_qualified': False, 'P3_qualified': False}, sort_keys=True))


def test_query_batch_regroup_exact_bits_for_serial_verify_law(native, record_property):
    generator = torch.Generator(device='cuda').manual_seed(2710)
    single = list(tensors(heads=12, keys=65))
    for value in single[:3]:
        value.copy_(torch.randn(value.shape, device=value.device, dtype=value.dtype, generator=generator) * .1)
    with fa.prepare(native, *single) as one:
        one.run(*single)
        q, k, v, out, lse = tensors(batch=4, heads=12, keys=65)
        for target, source in zip((q, k, v), single[:3]):
            target.copy_(source.expand_as(target))
        with fa.prepare(native, q, k, v, out, lse) as many:
            many.run(q, k, v, out, lse)
            mismatch_out = int((out[2].view(torch.int16) != single[3][0].view(torch.int16)).sum().item())
            mismatch_lse = int((lse[2].view(torch.int32) != single[4][0].view(torch.int32)).sum().item())
            record_property('fa26_row_regroup', json.dumps({'output_bit_mismatches': mismatch_out,
                'LSE_bit_mismatches': mismatch_lse, 'MTP_model_qualified': False}))
            assert mismatch_out == mismatch_lse == 0


def test_shared_kv_heads_as_query_rows_layout_diagnostic(native, record_property):
    args = tensors(heads=12, keys=65)
    generator = torch.Generator(device='cuda').manual_seed(712610)
    for value in args[:3]:
        value.copy_(torch.randn(value.shape, dtype=value.dtype, device=value.device, generator=generator) * .1)
    with fa.prepare(native, *args) as ordinary:
        ordinary.run(*args)
        q = args[0].view(1, 12, 1, 256)
        out = torch.empty_like(q)
        lse = torch.empty(1, 1, 12, device='cuda')
        with fa.prepare(native, q, args[1], args[2], out, lse) as folded:
            folded.run(q, args[1], args[2], out, lse)
            assert torch.isfinite(out).all() and torch.isfinite(lse).all()
            record_property('fa26_head_fold_layout', json.dumps({
                'output_bit_mismatches': int((out.flatten().view(torch.int16) != args[3].flatten().view(torch.int16)).sum().item()),
                'LSE_bit_mismatches': int((lse.flatten().view(torch.int32) != args[4].flatten().view(torch.int32)).sum().item()),
                'quality_accepted': False, 'performance_qualified': False,
                'semantic_precondition': 'same selected KV for all heads; causal/window/alibi/softcap disabled'}))


def test_other_stream_capture_and_changed_value_replay(native):
    args = tensors(keys=17)
    with fa.prepare(native, *args) as owner:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            owner.run(*args)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            owner.run(*args)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(args[3], torch.ones_like(args[3]))
        args[2].fill_(2.)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(args[3], torch.full_like(args[3], 2.))
        graph.reset()


def test_temporary_preparation_stream_then_scheduler_handoff_and_control_close(native):
    args = tensors()
    temporary = torch.cuda.Stream()
    temporary.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(temporary):
        owner = fa.prepare(native, *args)
    del temporary
    errors = []
    def worker():
        try:
            owner.handoff_to_current_thread()
            owner.run(*args)
            torch.cuda.synchronize()
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    try:
        assert errors == [] and torch.equal(args[3], torch.ones_like(args[3]))
    finally:
        owner.close()


@pytest.mark.parametrize('index', [0, 3])
def test_full_contiguous_but_eight_not_sixteen_byte_alignment_refused_without_writes(native, index):
    args = list(tensors())
    base = torch.zeros(args[index].numel() + 4, dtype=torch.float16, device='cuda')
    args[index] = base[4:].view_as(args[index])
    assert args[index].is_contiguous() and args[index].data_ptr() % 16 == 8
    before_out, before_lse = args[3].clone(), args[4].clone()
    with pytest.raises(RuntimeError, match='16-byte'):
        fa.prepare(native, *args)
    assert torch.equal(args[3], before_out) and torch.equal(args[4], before_lse)


def test_empty_shards_are_explicitly_outside_positive_component_domain(native):
    args = tensors(keys=0)
    before_out, before_lse = args[3].clone(), args[4].clone()
    with pytest.raises(RuntimeError, match='empty shards handled outside'):
        fa.prepare(native, *args)
    assert torch.equal(args[3], before_out) and torch.equal(args[4], before_lse)


def test_changed_geometry_rejected_before_output_mutation(native):
    native.validate_geometry(1, 1, 32768, 256, 32768)
    with pytest.raises(RuntimeError, match='KV tile offset overflow'):
        native.validate_geometry(1, 1, 32768, 257, 32768)
    args = tensors(rows=2)
    with fa.prepare(native, *args) as owner:
        changed = tensors(rows=1)
        before = changed[3].clone()
        with pytest.raises(RuntimeError, match='prepared geometry'):
            owner.run(*changed)
        assert torch.equal(changed[3], before)
