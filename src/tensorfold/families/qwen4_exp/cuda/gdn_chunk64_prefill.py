"""Explicit unqualified G3 prefill adapter; no engine/serving selection.

One actor owns one accepted prompt piece and its already funded State/cut
storage. Frontend generation reads the original conv history once. Contiguous
core spans alternate the two existing FP32 banks; prefix cuts copy the exact
bank and conv tail before the final conv/bank publication. Partial submission
is not transactional: errors poison the State and retain all borrowers.

No graph, MTP/decode or concurrent reuse is admitted. Binding creates host
views only, before any writes; kernels and maximum-row workspaces are prepared
once. A caller explicitly retires the completion event before reusing this
adapter. Submission itself never waits for GPU completion. This is still an
unqualified numerical component; native, model, quality, fit and P3 gates bind.
"""

from dataclasses import dataclass

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sm70 import gdn_chunk64, gdn_chunk64_half


def cut_bytes(layers, cuts):
    if type(layers) is not int or layers <= 0 or type(cuts) is not int or cuts < 0:
        raise ValueError("G3 cut credit requires actual positive layer/count geometry")
    return layers*cuts*(6*128*128*4+3*1280*2)


def _plain(tensor, shape, dtype, device):
    import torch

    return (type(tensor) is torch.Tensor and tuple(tensor.shape) == shape and tensor.dtype == dtype
            and tensor.device == device and tensor.is_contiguous() and not tensor.requires_grad
            and not tensor.is_conj() and not tensor.is_neg())


def _disjoint(owners, reads):
    for index,tensor in enumerate(owners):
        lo,hi = gdn_chunk64._span(tensor)
        if any(lo < gdn_chunk64._span(other)[1] and gdn_chunk64._span(other)[0] < hi
               for other in (*reads,*owners[:index]) if other.numel()):
            raise ValueError("G3 State/cuts/workspaces/output storage must be disjoint")


@dataclass
class Prepared:
    frontend: gdn_chunk64_half.Prepared
    core: gdn_chunk64.Prepared
    workspace: gdn_chunk64_half.Workspace
    family: object
    event: object
    stream: object
    active: object = None
    error: BaseException | None = None

    def bind(self, state, layer, projection, weight, bias, norm, eps, output, sums, *,
             a_log=None, a_neg=None, cuts=(), cut_budget_bytes=0, at=0):
        """Bind one actual State/layer/prompt and prefunded cut destinations.

        All cuts are relative to this complete prompt, strictly interior and
        ascending. Cut.rec/conv are full-layer snapshot owners populated layer
        by layer; this method never lazily allocates them or treats other
        unfilled layers as initialized snapshot payload.
        """
        import torch

        from tensorfold.cuda.sm70.owned_family import PreparedFamily

        from .forward import Cut
        from .state import State

        if self.error is not None:
            raise RuntimeError("G3 prefill adapter is poisoned; exclusive containment required") from self.error
        if self.active is not None:
            raise RuntimeError("G3 prefill borrowers require explicit positive retirement before reuse")
        if type(state) is not State or type(layer) is not int:
            raise TypeError("G3 prefill requires its actual State and layer owner")
        state.kv_check()
        if type(at) is not int or at < 0:
            raise ValueError("G3 prompt buffer origin must be a nonnegative row")
        if type(cuts) is not tuple:
            raise ValueError("G3 prefill cut policy must be an owned fixed tuple")
        device = self.frontend.device
        if type(self.family) is not PreparedFamily or self.family.device != device:
            raise TypeError("G3 prefill requires the already prepared same-device native shift owner")
        rec,conv = state.rec,state.conv
        if (type(rec) is not torch.Tensor or rec.dim() != 5 or rec.shape[0] != 2
                or not 0 <= layer < rec.shape[1]):
            raise ValueError("G3 requires two real State banks and an existing linear layer")
        layers = rec.shape[1]
        if (not _plain(rec,(2,layers,6,128,128),torch.float32,device)
                or not _plain(conv,(layers,3,1280),torch.float16,device)
                or state.execution_dtype != torch.float16 or type(state.cur) is not list
                or len(state.cur) != layers or any(type(index) is not int or index not in (0,1) for index in state.cur)):
            raise ValueError("G3 prefill State differs from TP8 Half conv/FP32 VLK banks")
        half = self.frontend.bind(self.workspace,projection,conv[layer],weight,bias,norm,eps,output,sums,
                                  a_log=a_log,a_neg=a_neg)
        rows = half.rows
        if type(cut_budget_bytes) is not int or cut_budget_bytes < cut_bytes(layers,len(cuts)):
            raise MemoryError("G3 prefix cuts lack explicit full-layer snapshot credit")
        last = 0
        for cut in cuts:
            if (type(cut) is not Cut or type(cut.row) is not int or not last < cut.row < rows or type(cut.at) is not int or cut.at != at
                    or not _plain(cut.rec,(layers,6,128,128),torch.float32,device)
                    or not _plain(cut.conv,(layers,3,1280),torch.float16,device)):
                raise ValueError("G3 cuts require ascending interior rows and prefunded exact snapshot destinations")
            last = cut.row
        parameter = a_log if a_neg is None else a_neg
        exclusive = (rec,conv,*(tensor for cut in cuts for tensor in (cut.rec,cut.conv)),
                     self.workspace.q,self.workspace.k,self.workspace.v,self.workspace.raw_gate,
                     self.workspace.beta,self.workspace.raw_output,self.core.a,self.core.gsum,self.core.h,
                     *self.core.metadata,output,sums)
        _disjoint(exclusive,(projection,weight,parameter,bias,norm))
        cur = state.cur[layer]
        banks = (rec[0,layer].unsqueeze(0),rec[1,layer].unsqueeze(0))
        spans,cut_views = [],[]
        start = 0
        for cut in (*cuts,None):
            end = rows if cut is None else cut.row
            q,k,v,gate,beta,raw = (tensor[:,start:end] for tensor in half.core_operands)
            spans.append(self.core.bind(q,k,v,gate,beta,banks[cur],raw,banks[1-cur]))
            cur = 1-cur
            if cut is not None:
                cut_views.append((cut.rec[layer],cut.conv[layer],projection[None,:end],banks[cur][0],cut.conv[layer:layer+1]))
            start = end
        # This immutable history view is not advanced until all prefix tails
        # have been copied and the full raw core output has been consumed.
        history = conv[layer]
        final_tail = conv[layer:layer+1]
        full_projection = projection[None]
        retained = (*exclusive,projection,weight,parameter,bias,norm,history,final_tail,full_projection,
                    *banks,*(tensor for cut in cut_views for tensor in cut),
                    *(tensor for span in spans for tensor in span.owners))
        bound = Bound(self,state,state.cur,layer,state.cur[layer],cur,state.slot_generation,half,tuple(spans),
                      tuple(cut_views),tuple((cut,cut.row,cut.at,cut.rec,cut.conv) for cut in cuts),
                      history,final_tail,full_projection,
                      (self.frontend,self.core,self.workspace,self.family,self.event,self.stream),retained,
                      tuple(map(gdn_chunk64._signature,retained)))
        state.history_native_readers.add(bound)
        self.active = bound
        return bound


@dataclass(eq=False)
class Bound:
    prepared: Prepared
    state: object
    cur_owner: list
    layer: int
    original_bank: int
    final_bank: int
    generation: int
    half: gdn_chunk64_half.Bound
    spans: tuple
    cuts: tuple
    cut_records: tuple
    history: object
    final_tail: object
    projection: object
    prepared_owners: tuple
    retained: tuple
    signatures: tuple
    submitted: bool = False
    recorded: bool = False
    retired: bool = False
    error: BaseException | None = None

    def _poison(self, error):
        self.error = self.prepared.error = error
        self.state._history_error = self.state._kv_error = error

    def run(self):
        """Enqueue once; no wait, device allocation, compilation or rollback."""
        if self.submitted or self.retired:
            raise RuntimeError("G3 prefill frame is single-use")
        self.state.kv_check()
        self.half._ready()
        current = (self.prepared.frontend,self.prepared.core,self.prepared.workspace,
                   self.prepared.family,self.prepared.event,self.prepared.stream)
        core = self.spans[0].prepared
        core_owners = (core.a,core.gsum,core.h,*core.metadata)
        if (any(now is not original for now,original in zip(current,self.prepared_owners))
                or len(core_owners) != len(self.spans[0].prepared_owners)
                or any(now is not original for now,original in zip(core_owners,self.spans[0].prepared_owners))
                or self.prepared.active is not self or self.state.cur is not self.cur_owner
                or self.state.rec is not self.retained[0] or self.state.conv is not self.retained[1]
                or type(self.state.cur[self.layer]) is not int or self.state.cur[self.layer] != self.original_bank
                or type(self.state.slot_generation) is not int or self.state.slot_generation != self.generation
                or any(type(cut.row) is not int or type(cut.at) is not int or cut.row != row or cut.at != at
                       or cut.rec is not rec or cut.conv is not conv
                       for cut,row,at,rec,conv in self.cut_records)
                or tuple(map(gdn_chunk64._signature,self.retained)) != self.signatures):
            raise RuntimeError("G3 prefill State/cut/span binding changed before submission")
        self.submitted = True
        try:
            self.half.front()
            for index,span in enumerate(self.spans):
                span.run()
                if index < len(self.cuts):
                    rec,conv,prefix,source,tail = self.cuts[index]
                    rec.copy_(source)
                    conv.copy_(self.history)
                    self.prepared.family.shift_windows(tail,prefix,prefix.shape[1],1280)
            self.half.back()
            self.prepared.family.shift_windows(self.final_tail,self.projection,self.half.rows,1280)
            self.prepared_owners[4].record(self.prepared_owners[5])
            self.recorded = True
            self.state.cur[self.layer] = self.final_bank
        except BaseException as error:
            self._poison(error)
            raise
        return self.half.output_pair

    def wait_native_reuse(self):
        """Explicit retirement, also used by the existing State reader journal.

        Failure retains the active journal and every borrower; no reset/retry
        becomes legal. A recorded event proves the final consumer, while a
        failed record requires positive owning-stream retirement instead.
        """
        if self.retired:
            return
        if self.prepared.active is not self:
            raise RuntimeError("G3 active borrower journal changed before retirement")
        primary = self.error
        try:
            if self.recorded:
                self.prepared_owners[4].synchronize()
            elif self.submitted:
                self.prepared_owners[5].synchronize()
        except BaseException as cleanup:
            self._poison(primary if primary is not None else cleanup)
            if primary is None:
                raise
            raise_failures(primary,[cleanup])
        if primary is not None:
            raise primary
        self.retired = True
        self.state.history_native_readers.discard(self)
        self.prepared.active = None


def prepare(capacity, device, *, workspace_budget_bytes, compiler, family):
    """Cold explicit component opt-in; compiler/runtime retention is separate."""
    total = gdn_chunk64_half.layout(capacity)["device_bytes"]+gdn_chunk64.layout(capacity).workspace_bytes
    if type(workspace_budget_bytes) is not int or workspace_budget_bytes < total:
        raise MemoryError("G3 segmented prefill lacks complete frontend/core/raw-output credit")
    import torch

    from tensorfold.cuda.sm70.owned_family import PreparedFamily

    if type(family) is not PreparedFamily or family.device != torch.device(device):
        raise TypeError("G3 prefill requires an already prepared explicit native-family shift owner")
    frontend = gdn_chunk64_half.prepare(device)
    workspace = gdn_chunk64_half.allocate_workspace(capacity,frontend.device,
        device_budget_bytes=gdn_chunk64_half.layout(capacity)["device_bytes"])
    core = gdn_chunk64.prepare(capacity,frontend.device,compiler=compiler,
                              workspace_budget_bytes=gdn_chunk64.layout(capacity).workspace_bytes)
    stream = torch.cuda.current_stream(frontend.device)
    event = torch.cuda.Event()
    event.record(stream)  # Resolve the finite event resource during cold preparation.
    event.synchronize()
    return Prepared(frontend,core,workspace,family,event,stream)
