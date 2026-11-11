"""Synthetic SC117-local HC graph component; no model/TP transport/P3 claim.

The caller grant is 2GiB of additional graph residency, with actual fixture
Tensor residency paid separately and a 512MiB free-device floor. It is NOT a
production grant for all24 States or all8 ranks. All97 real HC/Matrix keys use
unchanged original ATen arithmetic; comparisons include mutation publication.
"""
from contextlib import contextmanager
import json
import math
import time
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda.forward import _readout_plain
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat
from tensorfold.families.qwen4_exp.cuda.hc_plans import readout_graphs
from tensorfold.families.qwen4_exp.cuda.hc_readout_graphs import HCReadoutGraphs
from tensorfold.families.qwen4_exp.cuda.weight_types import HC

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='actual SM70 graph component required')

GRAPH_GRANT = 2 << 30
FREE_FLOOR = 512 << 20
ROWS = 24
SHAPES = {'h': (ROWS, 10240), 'streams': (ROWS, 10240), 'pss': (ROWS, 10, 4),
          'normed': (ROWS, 10240), 'xs_normed': (ROWS, 320), 'dn': (ROWS, 324),
          'dn_mix': (ROWS, 320), 'act': (ROWS, 320), 'xs_act': (ROWS, 10),
          'inj_a': (ROWS, 4), 'inj_m': (ROWS, 4), 'up': (ROWS, 10240),
          'mixed': (ROWS, 2560), 'xs_mixed': (ROWS, 80)}
F32 = {'pss', 'xs_normed', 'xs_act', 'xs_mixed'}


def buffer_payload():
    return sum(math.prod(shape) * (4 if name in F32 else 2)
               for name, shape in SHAPES.items())


def buffers():
    return NS(rows=ROWS, prefill=False, hc_plans=None, padded_linears=None,
              **{name: torch.zeros(shape, dtype=torch.float32 if name in F32 else torch.float16,
                                   device='cuda') for name, shape in SHAPES.items()})


def catalogue(generator):
    def matrix(n, k):
        weight = torch.empty((n, k), dtype=torch.float16, device='cuda')
        weight.normal_(mean=0, std=.003, generator=generator)
        return GGUFDense(weight)
    def hc(inject):
        # Exactly the actual GGUF loader's split320+4 down projection. The
        # aligned concatenation owns fused storage but implementation=reference
        # executes the original separate M32 part GEMMs, not its fused selector.
        down = GGUFConcat([matrix(320, 10240), matrix(4, 10240)]) if inject else matrix(320, 10240)
        up = matrix(10240, 320)
        scale = torch.empty(10240, dtype=torch.float32, device='cuda')
        scale.normal_(mean=1, std=.01, generator=generator)
        return HC(down, up, scale, inject, down, up)
    return NS(cfg=NS(hidden=2560, streams=4, low=320, eps=1e-6), mtp=None, x3=None,
              comm=NS(world=8, rank=0), meta={'execution_provider': 'sm70'},
              layers=[NS(attn_hc=hc(True), mlp_hc=hc(True)) for _ in range(48)], mixer=hc(False))


@contextmanager
def prepared(record_property):
    assert torch.cuda.get_device_capability() == (7, 0), 'actual SM70 component region required'
    assert family_ops._NATIVE_FAMILY is False, 'this component retains the incumbent Torch HC family'
    errors, owner, primary = [None], None, None
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    free_before = torch.cuda.mem_get_info()[0]
    # Tensor payload plus the largest single Concat construction overlap.
    # This is an allocation-plan preflight, not a worst-case allocator bound.
    weight_payload = 96*(10240*324*2 + 10240*320*2 + 10240*4)
    weight_payload += 10240*320*2 + 10240*320*2 + 10240*4
    planned = weight_payload + 2*buffer_payload() + 10240*324*2
    if free_before < planned + GRAPH_GRANT + FREE_FLOOR:
        raise MemoryError('fixture tensors plus explicit graph grant and free floor do not fit')
    try:
        generator = torch.Generator(device='cuda').manual_seed(9381)
        w, b, gold = catalogue(generator), buffers(), buffers()
        torch.cuda.synchronize()
        physical_free = torch.cuda.mem_get_info()[0]
        if physical_free < GRAPH_GRANT + FREE_FLOOR:
            raise MemoryError('actual settled fixture tensors leave insufficient graph grant/floor')
        owner = readout_graphs(w, b, budget_bytes=GRAPH_GRANT)
        record = owner.prepare(quiescent=lambda: True)
        assert record['captured_keys'] == record['covered_keys'] == 97 and record['observed_peak_delta_bytes'] > 0
        assert record['captured_templates'] == record['template_count'] == 3
        assert record['captured_r1_graphs'] == record['r1_graph_count'] == 97
        assert record['captured_graphs'] == 100 and record['bank_group_count'] == 2
        assert record['bank_tensor_bytes'] > 0
        assert 0 < record['settled_charge_bytes'] <= GRAPH_GRANT
        assert torch.cuda.mem_get_info()[0] >= FREE_FLOOR
        record_property('HC_graph_component_admission', json.dumps({
            'planned_fixture_bytes': planned, 'physical_free_before_fixture': free_before,
            'physical_free_after_fixture': physical_free, 'graph_grant_bytes': GRAPH_GRANT,
            'free_floor_bytes': FREE_FLOOR, 'record': record}, sort_keys=True))
        b.hc_plans = owner  # The actual handled-bool hook, after preparation/admission.
        yield owner, w, b, gold, generator
    except BaseException as error:
        primary = error
    finally:
        if owner is not None:
            try:
                owner.close()
            except BaseException as error:
                errors[0] = error
        if errors[0] is not None:
            raise_failures(primary, errors)
        if primary is not None:
            raise_failures(primary, ())


def exact(a, b):
    word = torch.int16 if a.dtype == torch.float16 else torch.int32
    assert torch.equal(a.view(word), b.view(word))


def inputs(b, gold, generator, iteration):
    b.h.normal_(mean=0, std=.25, generator=generator)
    b.streams.normal_(mean=0, std=.25, generator=generator)
    if iteration == 1:
        b.h[:, ::2].neg_()  # Changed live inputs, cancellation and signed values.
    elif iteration == 3:
        values = torch.tensor([0., -0., 2**-24, -2**-24, 2**-14, -2**-14,
                               65504., -65504., 32752., -32752., 1., -1., .25, -.25, 16., -16.],
                              device='cuda', dtype=torch.float16)
        b.h.copy_(values.repeat(b.h.numel() // len(values)).reshape_as(b.h))
        b.streams.copy_(b.h)
    elif iteration == 4:
        b.h[0, 0] = float('nan')
        b.h[1, 1] = float('inf')
        b.streams[0, 0] = -float('inf')
    elif iteration == 2:
        b.h.zero_()
        b.h[0, :4] = torch.tensor([0., -0., 2**-24, -2**-24], device='cuda', dtype=torch.float16)
    h = b.h.float().view(ROWS, 4, 10, 256)
    b.pss.copy_((h*h).sum(-1).transpose(1, 2))
    b.inj_a.fill_(-7)
    b.inj_m.fill_(3)
    for name in SHAPES:
        getattr(gold, name).copy_(getattr(b, name))


def keys(w, b):
    for layer in w.layers:
        yield layer.attn_hc, b.h, b.inj_a[:1]
        yield layer.mlp_hc, b.h, b.inj_m[:1]
    yield w.mixer, b.streams, None



def measure_copy_and_full97(owner, b, gold, generator, record_property):
    """Component-only R1 costs; not a model/P3 benchmark or a projected gain."""
    inputs(b, gold, generator, 0)
    entries = list(owner._entries)
    payload = sum(value.numel()*value.element_size() for entry in entries for value in entry.sources)
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    started = time.perf_counter()
    begin.record()
    for entry in entries:
        owner._stage_key(entry.template, entry.sources)
    end.record()
    end.synchronize()
    copy_wall, copy_ms = time.perf_counter()-started, begin.elapsed_time(end)
    before = owner.record()['replays']
    before_stages = owner.record()['bank_stages']
    started = time.perf_counter()
    begin.record()
    for entry in entries:
        inject = entry.inject[:1] if entry.inject is not None else None
        assert owner(entry.hc, entry.h, 1, owner.eps, 4, 320, inject)
    end.record()
    end.synchronize()
    full_wall, full_ms = time.perf_counter()-started, begin.elapsed_time(end)
    assert owner.record()['replays'] == before+97
    assert owner.record()['bank_stages'] == before_stages
    assert torch.isfinite(b.mixed[:1]).all()
    record_property('HC_template_R1_copy_full97_diagnostic', json.dumps({
        'key_count': 97, 'weight_copy_bytes_per_rank': payload,
        'copy_only_wall_seconds': copy_wall, 'copy_only_inclusive_stream_ms': copy_ms,
        'full97_wall_seconds': full_wall, 'full97_inclusive_stream_ms': full_ms,
        'r1_bank_stages': owner.record()['bank_stages']-before_stages,
        'scope': 'synthetic component copy-only/M32staging diagnostic plus full97directR1(no bankcopies); no TP/model/P3 claim'}))

def test_all97_keys_dynamic_inputs_raw_outputs_versions_and_close(record_property):
    with prepared(record_property) as (owner, w, b, gold, generator):
        pointers = {name: getattr(b, name).data_ptr() for name in SHAPES}
        for iteration in range(3):
            for index, (hc, h, inject) in enumerate(keys(w, b)):
                inputs(b, gold, generator, iteration)
                gh = gold.streams if index == 96 else gold.h
                gi = None if inject is None else (gold.inj_a if index % 2 == 0 else gold.inj_m)[:1]
                before = {name: getattr(b, name)._version for name in SHAPES}
                expected_before = {name: getattr(gold, name)._version for name in SHAPES}
                _readout_plain(hc, gold, gh, 1, w.cfg.eps, 4, 320, gi)
                before_replays = owner.record()['replays']
                _readout_plain(hc, b, h, 1, w.cfg.eps, 4, 320, inject)
                assert owner.record()['replays'] == before_replays+1
                for name in SHAPES:
                    assert getattr(b, name)._version-before[name] == getattr(gold, name)._version-expected_before[name]
                    # Entire buffers include untouched unsupported second rows,
                    # other injection role and input/status-like PSS payloads.
                    exact(getattr(b, name), getattr(gold, name))
                assert {name: getattr(b, name).data_ptr() for name in SHAPES} == pointers
        assert owner.record()['replays'] == 3*97
        measure_copy_and_full97(owner, b, gold, generator, record_property)
    assert owner.closed and not owner._owners and not owner._leases



@pytest.mark.parametrize('inference', [False, True])
def test_live_rows1_through24_and_poisoned_suffix_match_original_callback(record_property, inference):
    with torch.inference_mode(inference):
        with prepared(record_property) as (owner, w, b, gold, generator):
            catalogue = [(getattr(layer, role), index*2+offset) for index, layer in enumerate(w.layers)
                         for offset, role in enumerate(('attn_hc', 'mlp_hc'))] + [(w.mixer, 96)]
            order = [value for pair in zip(range(24, 12, -1), range(1, 13)) for value in pair]
            schedule = [(hc, key, 24, 0) for hc, key in catalogue]
            for key in (0, 1, 96):
                schedule.extend((catalogue[key][0], key, rows, pattern) for rows in order for pattern in (1, 3, 4))
            schedule.extend((catalogue[key][0], key, rows, 3) for key, rows in
                            ((0, 1), (2, 24), (0, 1), (2, 3), (0, 24)))
            for entry in owner._entries:
                assert all(target.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
                           for target, source in zip(entry.template.banks, entry.sources, strict=True))
            source_versions = {id(source): source._version for entry in owner._entries
                               for source in entry.sources if not source.is_inference()}
            before_count = owner.record()['replays']
            scalar_version, queued_rows, fills = owner._live_rows._version, owner._queued_rows, 0
            for hc, key, rows, pattern in schedule:
                inputs(b, gold, generator, pattern)
                # Every inactive word is initialized and must remain unchanged;
                # bitwise checks catch half/float signaling-NaN quieting in masks.
                for name in SHAPES:
                    value = getattr(b, name)[rows:]
                    word = value.view(torch.int16 if value.dtype == torch.float16 else torch.int32)
                    word.fill_(0x7d35 if value.dtype == torch.float16 else 0x7fa12345)
                    getattr(gold, name)[rows:].copy_(value)
                versions = {name: None if getattr(b, name).is_inference() else getattr(b, name)._version
                            for name in SHAPES}
                expected_versions = {name: None if getattr(gold, name).is_inference() else getattr(gold, name)._version
                                     for name in SHAPES}
                hname = 'streams' if key == 96 else 'h'
                iname = 'inj_a' if key % 2 == 0 else 'inj_m'
                inject = getattr(b, iname)[:rows] if hc.inject else None
                expected_inject = getattr(gold, iname)[:rows] if hc.inject else None
                _readout_plain(hc, gold, getattr(gold, hname), rows, w.cfg.eps, 4, 320, expected_inject)
                count = owner.record()['replays']
                before_stages = owner.record()['bank_stages']
                if rows != 1 and queued_rows != rows:
                    fills += 1
                    queued_rows = rows
                _readout_plain(hc, b, getattr(b, hname), rows, w.cfg.eps, 4, 320, inject)
                assert owner.record()['replays'] == count+1
                assert owner.record()['bank_stages'] == before_stages+(rows != 1)
                for name in SHAPES:
                    actual, expected = getattr(b, name), getattr(gold, name)
                    exact(actual, expected)
                    if versions[name] is not None:
                        assert actual._version-versions[name] == expected._version-expected_versions[name]
                # The shared physical input frame has defined zero padding even
                # when R decreases or a different HC key/injection role follows.
                if rows != 1:
                    physical_h = getattr(owner._scratch, hname)
                    assert torch.count_nonzero(physical_h[rows:].view(torch.int16)) == 0
                    assert torch.count_nonzero(owner._scratch.pss[rows:].view(torch.int32)) == 0
            assert all(source._version == source_versions[id(source)] for entry in owner._entries
                       for source in entry.sources if not source.is_inference())
            assert owner.record()['template_count'] == owner.record()['captured_templates'] == 3
            assert owner._live_rows._version == scalar_version+fills
            assert owner._live_rows.item() == queued_rows
            assert owner.record()['replays'] == before_count+len(schedule)
            assert len(owner._entries) == owner.record()['captured_keys'] == 97
            assert owner.record()['rows'] == list(range(1, 25)) and owner.record()['physical_rows'] == 32

def test_misses_alias_pointer_rebind_and_explicit_budget_refusal(record_property):
    with prepared(record_property) as (owner, w, b, gold, generator):
        hc, h, inject = next(keys(w, b))
        count = owner.record()['replays']
        assert not owner(hc, h, 25, w.cfg.eps, 4, 320, inject)
        assert not owner(hc, h, 1, w.cfg.eps, 4, 320, inject, normed=True)
        assert not owner(hc, h.clone(), 1, w.cfg.eps, 4, 320, inject)
        assert not owner(hc, h, 1, w.cfg.eps, 4, 320, inject.clone())
        assert owner.record()['replays'] == count
        inputs(b, gold, generator, 0)
        for rows, normed in ((24, True), (1, True)):
            _readout_plain(hc, gold, gold.h, rows, w.cfg.eps, 4, 320, gold.inj_a[:rows], normed=normed)
            _readout_plain(hc, b, b.h, rows, w.cfg.eps, 4, 320, b.inj_a[:rows], normed=normed)
            assert owner.record()['replays'] == count
            for name in SHAPES:
                exact(getattr(b, name), getattr(gold, name))
        owner.close()
        b.hc_plans = None  # Positive retirement precedes fixture re-enrollment.
        tiny = HCReadoutGraphs(w, b, _readout_plain, budget_bytes=1)
        primary = None
        try:
            with pytest.raises(MemoryError, match='admitted'):
                tiny.prepare(quiescent=lambda: True)
            assert tiny.closed and tiny.record()['observed_peak_delta_bytes'] > 1
            record_property('HC_graph_one_byte_refusal', json.dumps(tiny.record(), sort_keys=True))
        except BaseException as error:
            primary = error
        finally:
            try:
                tiny.close()
            except BaseException as cleanup:
                raise_failures(primary, (cleanup,))
            if primary is not None:
                raise_failures(primary, ())
        failure = OSError('controlled CPU readout failure during legal CUDA capture')
        calls = 0
        def fail_after_original(*args, **kwargs):
            nonlocal calls
            calls += 1
            _readout_plain(*args, **kwargs)
            if calls == 2:  # First key warmup, then its capture body.
                raise failure
        broken = HCReadoutGraphs(w, b, fail_after_original, budget_bytes=GRAPH_GRANT)
        primary = None
        try:
            with pytest.raises(OSError) as raised:
                broken.prepare(quiescent=lambda: True)
            assert raised.value is failure and broken.closed and not broken._owners
        except BaseException as error:
            primary = error
        finally:
            try:
                broken.close()
            except BaseException as cleanup:
                raise_failures(primary, (cleanup,))
            if primary is not None:
                raise_failures(primary, ())
