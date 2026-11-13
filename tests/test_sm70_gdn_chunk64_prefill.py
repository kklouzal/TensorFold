"""CPU span/State protocol controls; synthetic launches are not native evidence."""
import threading
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import gdn_chunk64 as core
from tensorfold.cuda.sm70 import gdn_chunk64_half as half
from tensorfold.cuda.sm70.family_ops import _reference
from tensorfold.cuda.sm70.owned_family import PreparedFamily
from tensorfold.families.qwen4_exp.cuda import gdn_chunk64_prefill as adapter
from tensorfold.families.qwen4_exp.cuda.forward import Cut
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_sm70_gdn_chunk64_half import operands
from tools.v100_gdn_chunk64_trial import half_epilogue_reference, half_frontend_reference, reference


class Event:
    def __init__(self):
        self.recorded = self.waited = 0
        self.failure = None
    def record(self, stream):
        self.recorded += 1
    def synchronize(self):
        self.waited += 1
        if self.failure is not None:
            raise self.failure


def cpu_state(bank=0):
    state = object.__new__(State)
    state.rec = torch.zeros(2,1,6,128,128)
    state.rec[:,0,:,0,0] = 0.25
    state.conv = torch.zeros(1,3,1280,dtype=torch.float16)
    state.conv[0,:,0] = torch.tensor([0.5,1.,2.])
    state.cur = [bank]
    state.execution_dtype = torch.float16
    state.slot_generation = 0
    state._history_error = state._kv_error = state._resize_error = None
    state.kv_status = state.compact_indexer = state.indexer_history_owner = None
    state._kv_pending = False
    state.history_native_readers = weakref.WeakSet()
    state.scratch = []
    state.capacity = state.logical_context = 256
    state.pos = state.mtp_len = state.mtp_drafted = 0
    state.pos_dev = torch.tensor(0)
    state.mtp_pos = torch.tensor(0)
    state.rope_delta_dev = torch.tensor(0)
    state.ple_tail = torch.zeros(1)
    state.ple_history = None
    state.kc = []
    state.kv_identity = ('CPU-protocol-control',)
    return state


def cpu_prepared(monkeypatch, rows, capacity=None):
    """Real tensor/State metadata plus independent CPU numerical surrogates.

    Only CUDA context/event and launch interfaces are substituted. This cannot
    establish native math, device ordering, launch validity, or performance.
    """
    capacity = rows if capacity is None else capacity
    workspace,tensors = operands(rows,capacity)
    p,_,weight,_parameter,_bias,_norm,_out,_sums = tensors
    p[:,0] = 2
    p[:,256] = 2
    p[:,512:1280] = 0.5
    weight[:,2:] = 0.5
    calls = []
    stream = SimpleNamespace(cuda_stream=17,synchronize=lambda: calls.append('stream-wait'))
    monkeypatch.setattr(half,'_context',lambda device: None)
    monkeypatch.setattr(torch.cuda,'current_device',lambda: None)
    monkeypatch.setattr(torch.cuda,'current_stream',lambda device: stream)
    monkeypatch.setattr(torch.cuda,'is_current_stream_capturing',lambda: False)
    def front(p,history,w,a,b,r,q,k,v,g,beta,negative):
        calls.append(('front',r))
        expected = half_frontend_reference(p,history,w,a,b,negative_decay=negative)
        for destination,source in zip((q,k,v,g,beta),expected):
            destination[:r].copy_(source[0])
    def back(p,norm,eps,r,raw,out,sums):
        calls.append(('back',r))
        y,groups = half_epilogue_reference(raw[:r],p,norm,eps)
        out.copy_(y);sums.copy_(groups)
    frontend = half.Prepared(SimpleNamespace(chunk64_front=front,chunk64_back=back),torch.device('cpu'))
    prepared = core.Prepared()
    prepared.device = torch.device('cpu')
    prepared.owner = threading.current_thread()
    prepared.stream = 17
    prepared.error = None
    prepared.layout = core.layout(capacity)
    prepared.a = torch.full((1,capacity,6,64),9,dtype=torch.float16)
    prepared.gsum = torch.full((1,capacity,6),9.)
    prepared.h = torch.empty(1,0,6,128,128,dtype=torch.float16)
    prepared.metadata = (torch.tensor([0,capacity],dtype=torch.int32),
                         torch.tensor([0,(capacity+63)//64],dtype=torch.int32),
                         torch.zeros(1,dtype=torch.int32),torch.ones(1,dtype=torch.bool))
    prepared.ffi_device = 'CPU-control'
    prepared.stream_context = lambda device,stream: nullcontext()
    def cumsum(g,gsum,chunks):
        calls.append(('cumsum',g.shape[1],chunks))
        prepared.raw = g.clone()
        for start in range(0,g.shape[1],64):
            gsum[:,start:start+64].copy_(g[:,start:start+64].cumsum(1))
    def kkt(k,b,a,chunks):
        calls.append(('kkt',k.shape[1],chunks))
        a.zero_()
    def run(q,k,v,a,g,b,initial,cu,offsets,dummy,rawcu,indices,has_initial,output,h,final):
        calls.append(('core',q.shape[1]))
        y,state = reference(q,k,v,prepared.raw,b,initial)
        output.copy_(y.half());final.copy_(state.float())
    prepared.cumsum,prepared.kkt,prepared.core = cumsum,kkt,run
    family = PreparedFamily(SimpleNamespace(shift=_reference['shift_windows']),torch.device('cpu'))
    event = Event()
    return adapter.Prepared(frontend,prepared,workspace,family,event,stream),tensors,calls


def bind(prepared,state,tensors,cuts=(),credit=None):
    p,_,weight,parameter,bias,norm,out,sums = tensors
    return prepared.bind(state,0,p,weight,bias,norm,1e-6,out,sums,a_log=parameter,cuts=cuts,
                         cut_budget_bytes=adapter.cut_bytes(1,len(cuts)) if credit is None else credit)


@pytest.mark.parametrize('rows,points,bank',[(1,(),0),(63,(1,31),0),(64,(3,63),1),(65,(1,63,64),0)])
def test_full_prompt_frontend_segmented_banks_exact_cut_tails_and_publication(monkeypatch,rows,points,bank):
    prepared,tensors,calls = cpu_prepared(monkeypatch,rows,rows+2)
    state = cpu_state(bank)
    original = state.conv.clone()
    initial = state.rec[bank,0].unsqueeze(0).clone()
    cuts = tuple(Cut(row,rec=torch.empty_like(state.rec[0]),conv=torch.empty_like(state.conv)) for row in points)
    metadata = tuple(t.clone() for t in prepared.core.metadata)
    with torch.inference_mode():
        frame = bind(prepared,state,tensors,cuts)
        assert prepared.event.recorded == prepared.event.waited == 0
        frame.run()
    assert prepared.event.recorded == 1 and prepared.event.waited == 0
    assert state.cur == [(bank+len(points)+1)%2]
    assert calls.count(('front',rows)) == calls.count(('back',rows)) == 1
    assert all(torch.equal(a,b) for a,b in zip(metadata,prepared.core.metadata))
    assert torch.equal(prepared.core.a[:,rows:],torch.full_like(prepared.core.a[:,rows:],9))
    assert torch.equal(prepared.core.gsum[:,rows:],torch.full_like(prepared.core.gsum[:,rows:],9))
    p,_,weight,parameter,bias,norm,out,sums = tensors
    values = half_frontend_reference(p,original[0],weight,parameter,bias,negative_decay=False)
    q,k,v,g,beta = values
    expected = initial
    raw = torch.empty_like(v)
    start = 0
    for index,end in enumerate((*points,rows)):
        y,expected = reference(q[:,start:end],k[:,start:end],v[:,start:end],g[:,start:end],beta[:,start:end],expected)
        expected = expected.float()
        raw[:,start:end].copy_(y.half())
        if index < len(cuts):
            assert torch.equal(cuts[index].rec[0],expected[0])
            tail = torch.cat((original,p[None,:,:1280]),dim=1)[:,end:end+3]
            assert torch.equal(cuts[index].conv,tail)
        start = end
    assert torch.equal(state.rec[state.cur[0],0],expected[0])
    assert torch.equal(state.conv,torch.cat((original,p[None,:,:1280]),dim=1)[:,rows:rows+3])
    wanted,groups = half_epilogue_reference(raw,p,norm,1e-6)
    assert torch.equal(out,wanted) and torch.equal(sums,groups)
    with torch.inference_mode(),pytest.raises(RuntimeError,match='retirement'):
        bind(prepared,state,tensors)
    frame.wait_native_reuse()
    assert prepared.active is None and not state.history_native_readers


@pytest.mark.parametrize('fault',['zero','last','descending','missing','wrong-at','undercredit','alias'])
def test_bad_cut_policy_refuses_before_any_frontend_or_state_mutation(monkeypatch,fault):
    prepared,tensors,calls = cpu_prepared(monkeypatch,5)
    state = cpu_state()
    cut = Cut(2,rec=torch.empty_like(state.rec[0]),conv=torch.empty_like(state.conv))
    cuts = (cut,)
    credit = None
    if fault == 'zero':cut.row = 0
    elif fault == 'last':cut.row = 5
    elif fault == 'descending':cuts = (cut,Cut(1,rec=torch.empty_like(cut.rec),conv=torch.empty_like(cut.conv)))
    elif fault == 'missing':cut.rec = None
    elif fault == 'wrong-at':cut.at = 1
    elif fault == 'undercredit':credit = adapter.cut_bytes(1,1)-1
    else:cut.conv = state.conv
    before = state.rec.clone(),state.conv.clone()
    with torch.inference_mode(),pytest.raises((ValueError,MemoryError)):
        bind(prepared,state,tensors,cuts,credit)
    assert calls == [] and prepared.active is None
    assert torch.equal(state.rec,before[0]) and torch.equal(state.conv,before[1])


@pytest.mark.parametrize('fault',['cut-row','cut-owner','state-owner','bank','generation','span-view'])
def test_changed_bound_state_cut_or_span_refused_before_writes(monkeypatch,fault):
    prepared,tensors,calls = cpu_prepared(monkeypatch,5)
    state = cpu_state()
    cut = Cut(2,rec=torch.empty_like(state.rec[0]),conv=torch.empty_like(state.conv))
    with torch.inference_mode():frame = bind(prepared,state,tensors,(cut,))
    if fault == 'cut-row':cut.row = 3
    elif fault == 'cut-owner':cut.rec = cut.rec.clone()
    elif fault == 'state-owner':state.rec = state.rec.clone()
    elif fault == 'bank':state.cur[0] = 1
    elif fault == 'generation':state.slot_generation += 1
    else:frame.spans[0].operands[0].resize_(frame.spans[0].operands[0].numel())
    with torch.inference_mode(),pytest.raises(RuntimeError,match='changed'):
        frame.run()
    assert calls == [] and not frame.submitted
    frame.wait_native_reuse()  # Never submitted: no GPU work/event to wait for.


@pytest.mark.parametrize('cleanup_fails',[False,True])
def test_launch_and_retirement_failure_keep_primary_poison_and_borrowers(monkeypatch,cleanup_fails):
    prepared,tensors,calls = cpu_prepared(monkeypatch,5)
    state = cpu_state()
    cut = Cut(2,rec=torch.empty_like(state.rec[0]),conv=torch.empty_like(state.conv))
    primary = RuntimeError('authentic second-span failure')
    native = prepared.core.core
    def fail_second(*args):
        if ('core',3) in calls:raise primary
        native(*args)
        if args[0].shape[1] == 2:calls.append(('core',3))
    prepared.core.core = fail_second
    original = state.conv.clone()
    with torch.inference_mode():
        frame = bind(prepared,state,tensors,(cut,))
        with pytest.raises(RuntimeError) as caught:frame.run()
    assert caught.value is primary and state._history_error is state._kv_error is primary
    assert state.cur == [0] and torch.equal(state.conv,original)
    assert prepared.active is frame and not frame.recorded
    cleanup = OSError('authentic stream-retirement failure')
    if cleanup_fails:
        def fail_wait():
            calls.append('stream-wait')
            raise cleanup
        prepared.stream.synchronize = fail_wait
    with pytest.raises(RuntimeError) as caught:frame.wait_native_reuse()
    assert caught.value is primary and 'stream-wait' in calls and prepared.active is frame
    if cleanup_fails:
        assert caught.value.__cause__ is cleanup


def test_async_retirement_error_poison_and_actual_state_clone_restore_continuation(monkeypatch):
    prepared,tensors,_ = cpu_prepared(monkeypatch,3)
    state = cpu_state()
    with torch.inference_mode():
        frame = bind(prepared,state,tensors)
        frame.run()
    frame.wait_native_reuse()
    snapshot = state.snapshot()
    clone = state.clone()
    assert clone.rec.data_ptr() != state.rec.data_ptr() and torch.equal(clone.conv,state.conv)
    state.restore(snapshot)
    assert state.cur == [0] and torch.equal(state.rec[0],snapshot['rec'])
    with torch.inference_mode():
        second = bind(prepared,state,tensors)
        second.run()
    failure = RuntimeError('authentic completion failure')
    prepared.event.failure = failure
    with pytest.raises(RuntimeError) as caught:second.wait_native_reuse()
    assert caught.value is failure and state._history_error is failure and prepared.active is second
    assert torch.equal(clone.conv,snapshot['conv'])


def test_complete_workspace_credit_is_checked_before_any_provider_or_allocation():
    for credit in (0,True,4936*65):
        with pytest.raises(MemoryError):
            adapter.prepare(65,'cuda:0',workspace_budget_bytes=credit,compiler=None,family=None)


@pytest.mark.parametrize('fault',['core-metadata','core-h','workspace','family','event','stream'])
def test_prepared_owner_replacement_is_refused_before_dispatch(monkeypatch,fault):
    prepared,tensors,calls = cpu_prepared(monkeypatch,3)
    state = cpu_state()
    with torch.inference_mode():frame = bind(prepared,state,tensors)
    if fault == 'core-metadata':prepared.core.metadata = tuple(t.clone() for t in prepared.core.metadata)
    elif fault == 'core-h':prepared.core.h = torch.empty_like(prepared.core.h)
    elif fault == 'workspace':prepared.workspace = operands(3)[0]
    elif fault == 'family':prepared.family = PreparedFamily(prepared.family.module,torch.device('cpu'))
    elif fault == 'event':prepared.event = Event()
    else:prepared.stream = SimpleNamespace(cuda_stream=17)
    with torch.inference_mode(),pytest.raises(RuntimeError,match='changed'):
        frame.run()
    assert calls == [] and not frame.submitted


@pytest.mark.parametrize('fault',['zero','over-capacity','misaligned','metadata-replaced','h-replaced','view-resized'])
def test_core_maximum_span_and_retained_metadata_contract(monkeypatch,fault):
    prepared,_tensors,calls = cpu_prepared(monkeypatch,3,5)
    state = cpu_state()
    q,k,v,g,b,raw = (tensor[:,:3] for tensor in (prepared.workspace.q,prepared.workspace.k,prepared.workspace.v,
                     prepared.workspace.raw_gate,prepared.workspace.beta,prepared.workspace.raw_output))
    if fault == 'zero':q = q[:,:0]
    elif fault == 'over-capacity':q = torch.empty(1,6,2,128,dtype=torch.float16)
    elif fault == 'misaligned':q = torch.empty(3*256+1,dtype=torch.float16)[1:].view(1,3,2,128)
    with torch.inference_mode():
        if fault in ('zero','over-capacity','misaligned'):
            with pytest.raises(ValueError):
                prepared.core.bind(q,k,v,g,b,state.rec[0,0][None],raw,state.rec[1,0][None])
        else:
            span = prepared.core.bind(q,k,v,g,b,state.rec[0,0][None],raw,state.rec[1,0][None])
            if fault == 'metadata-replaced':prepared.core.metadata = tuple(t.clone() for t in prepared.core.metadata)
            elif fault == 'h-replaced':prepared.core.h = torch.empty_like(prepared.core.h)
            else:span.scratch[0].resize_(span.scratch[0].numel())
            with pytest.raises(RuntimeError,match='changed'):span.run()
    assert calls == []
