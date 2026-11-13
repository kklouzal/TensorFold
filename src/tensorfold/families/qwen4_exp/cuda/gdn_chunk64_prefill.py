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

from tensorfold.cleanup import finish, raise_failures, rollback
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
    group_layers: int = 0
    group_segments: int = 0
    group_journal: object = None
    group_host_budget: int | None = None

    def configure_groups(self, layers, segments, *, host_budget_bytes=None):
        if (self.active is not None or self.group_journal is not None or type(layers) is not int or layers <= 0
                or type(segments) is not int or segments <= 0):
            raise ValueError("G3 accepted-frame journal must be bounded once during cold preparation")
        if host_budget_bytes is not None and (type(host_budget_bytes) is not int or host_budget_bytes <= 0):
            raise ValueError("G3 journal host credit must be explicit positive bytes")
        self.group_host_budget = host_budget_bytes
        self.group_layers,self.group_segments = layers,segments
        self.group_journal = [None]*(layers*segments)

    def begin_group(self, segs, cuts, *, cut_budget_bytes):
        return Group.begin(self,segs,cuts,cut_budget_bytes=cut_budget_bytes)

    def bind(self, state, layer, projection, weight, bias, norm, eps, output, sums, *,
             a_log=None, a_neg=None, cuts=(), cut_budget_bytes=0, at=0, _group=None):
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
        if self.active is not None and (type(_group) is not Group or self.active is not _group):
            raise RuntimeError("G3 prefill borrowers require explicit positive retirement before reuse")
        if _group is not None:
            _group.authorize(state,layer,at,projection.shape[0],cuts)
        if type(state) is not State or type(layer) is not int:
            raise TypeError("G3 prefill requires its actual State and layer owner")
        if _group is None:
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
                      tuple(map(gdn_chunk64._signature,retained)),group=_group)
        if _group is None:
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
    group: object = None

    def _poison(self, error):
        self.error = self.prepared.error = error
        self.state._history_error = self.state._kv_error = error
        if self.group is not None:
            self.group.fail(error)

    def run(self):
        """Enqueue once; no wait, device allocation, compilation or rollback."""
        if self.submitted or self.retired:
            raise RuntimeError("G3 prefill frame is single-use")
        if self.group is None:
            self.state.kv_check()
        else:
            self.group.authorize(self.state,self.layer,self.group.origins[self.state],self.half.rows,
                                 tuple(record[0] for record in self.cut_records))
        self.half._ready()
        current = (self.prepared.frontend,self.prepared.core,self.prepared.workspace,
                   self.prepared.family,self.prepared.event,self.prepared.stream)
        core = self.spans[0].prepared
        core_owners = (core.a,core.gsum,core.h,*core.metadata)
        if (any(now is not original for now,original in zip(current,self.prepared_owners))
                or len(core_owners) != len(self.spans[0].prepared_owners)
                or any(now is not original for now,original in zip(core_owners,self.spans[0].prepared_owners))
                or self.prepared.active is not (self if self.group is None else self.group)
                or self.state.cur is not self.cur_owner
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
            if self.group is None:
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


def _python_metadata_bytes(value):
    """Bound caller-owned Python headers; Torch/FFI native metadata is separate.

    Shared cold preparers/States/storage are not recursively recharged here.
    Per-command repeated headers may be conservatively counted again; this is
    an admission charge, not a claim of exact RSS or C++ allocator accounting.
    """
    import sys

    import torch

    from tensorfold.cuda.sm70.gdn_chunk64_half import Bound as HalfBound
    seen = set()
    def size(item):
        if id(item) in seen:
            return 0
        seen.add(id(item))
        if type(item) in (Bound,HalfBound,gdn_chunk64.Bound):
            return sys.getsizeof(item)+sys.getsizeof(vars(item))+sum(
                size(v) for k,v in vars(item).items() if k not in ('prepared','group','state','workspace','thread'))
        if type(item) in (tuple,list):
            return sys.getsizeof(item)+sum(map(size,item))
        if type(item) is dict:
            return sys.getsizeof(item)+sum(size(k)+size(v) for k,v in item.items())
        if type(item) is torch.Tensor or type(item) in (int,float,str,bool,type(None)):
            return sys.getsizeof(item)
        return 0
    return size(value)


@dataclass(eq=False)
class Group:
    """One bounded accepted-target frame; GPU scratch reuse is stream ordered.

    State health is checked before the target frame marks codec writes pending.
    Layer submissions never perform kv_check/readback or event synchronization.
    All view/kernel/storage borrowers remain journaled until one final event.
    Cuts leave this owner only by ordinary snapshot/retention ownership transfer.
    """
    prepared: Prepared
    segments: tuple
    cuts: dict
    origins: dict
    indices: dict
    generations: tuple
    cut_budget_bytes: int
    prepared_owners: tuple
    core_owners: tuple
    cut_records: tuple
    count: int = 0
    python_metadata_bytes: int = 0
    validated: bool = False
    sealed: bool = False
    retired: bool = False
    submitted: bool = False
    error: BaseException | None = None

    @classmethod
    def begin(cls, prepared, segs, cuts, *, cut_budget_bytes):
        import threading

        import torch

        from .forward import Cut
        from .state import State

        if prepared.error is not None:
            raise RuntimeError("G3 preparation is poisoned") from prepared.error
        if (threading.current_thread() is not prepared.core.owner
                or torch.cuda.current_device() != prepared.frontend.device.index
                or torch.cuda.is_current_stream_capturing()
                or torch.cuda.current_stream(prepared.frontend.device).cuda_stream != prepared.core.stream
                or prepared.stream.cuda_stream != prepared.core.stream):
            raise RuntimeError("G3 target frame requires its admitted actor thread/device/stream")
        if prepared.active is not None or prepared.group_journal is None:
            raise RuntimeError("G3 target frame requires its cold journal and exclusive actor boundary")
        if not 0 < len(segs) <= prepared.group_segments or len({id(s) for s,_,_ in segs}) != len(segs):
            raise ValueError("G3 target frame exceeds its unique State segment bound")
        last = 0
        for state,start,end in segs:
            if (type(state) is not State or type(start) is not int or type(end) is not int
                    or start != last or not start < end <= prepared.workspace.capacity
                    or len(state.cur) != prepared.group_layers):
                raise ValueError("G3 target frame differs from its actual packed pbuf/State geometry")
            state.kv_check()
            layers = prepared.group_layers
            device = prepared.frontend.device
            if (not _plain(state.rec,(2,layers,6,128,128),torch.float32,device)
                    or not _plain(state.conv,(layers,3,1280),torch.float16,device)
                    or state.execution_dtype != torch.float16 or type(state.cur) is not list
                    or any(type(index) is not int or index not in (0,1) for index in state.cur)):
                raise ValueError("G3 accepted-target State geometry differs before submission")
            last = end
        if type(cut_budget_bytes) is not int or cut_budget_bytes < cut_bytes(prepared.group_layers,len(cuts)):
            raise MemoryError("G3 target cuts lack their existing retained-version payload credit")
        by_origin = {start:[] for _,start,_ in segs}
        for cut in cuts:
            if type(cut) is not Cut or cut.at not in by_origin:
                raise ValueError("G3 target cut belongs to another packed State")
            by_origin[cut.at].append(cut)
        for state,start,end in segs:
            points = by_origin[start]
            if len(points) > 3 or any(type(c.row) is not int or not 0 < c.row < end-start for c in points):
                raise ValueError("G3 target cut count/rows exceed the accepted prompt policy")
            if [c.row for c in points] != sorted({c.row for c in points}):
                raise ValueError("G3 target cuts must be strictly ascending")
            for cut in points:
                if (cut.rec is None) != (cut.conv is None):
                    raise ValueError("G3 cut requires a complete snapshot pair")
                if cut.rec is None:
                    # Existing snapshot category pays these owners. They escape
                    # through Cut into retention; event retirement never pools/reuses them.
                    cut.rec,cut.conv = torch.empty_like(state.rec[0]),torch.empty_like(state.conv)
                if (not _plain(cut.rec,(prepared.group_layers,6,128,128),torch.float32,prepared.frontend.device)
                        or not _plain(cut.conv,(prepared.group_layers,3,1280),torch.float16,prepared.frontend.device)):
                    raise ValueError("G3 target cut destinations differ before any layer submission")
        owned = tuple(tensor for state,_,_ in segs for tensor in (state.rec,state.conv))
        owned += tuple(tensor for cut in cuts for tensor in (cut.rec,cut.conv))
        _disjoint(owned,(prepared.workspace.q,prepared.workspace.k,prepared.workspace.v,prepared.workspace.raw_gate,
                         prepared.workspace.beta,prepared.workspace.raw_output,prepared.core.a,prepared.core.gsum,
                         prepared.core.h,*prepared.core.metadata))
        group = cls(prepared,tuple(segs),{k:tuple(v) for k,v in by_origin.items()},
                    {state:start for state,start,_ in segs},{state:index for index,(state,_,_) in enumerate(segs)},
                    tuple((state,state.slot_generation,state.rec,state.conv,state.cur) for state,_,_ in segs),cut_budget_bytes,
                    (prepared.frontend,prepared.core,prepared.workspace,prepared.family,prepared.event,prepared.stream,
                     prepared.group_journal,prepared.group_layers,prepared.group_segments,prepared.group_host_budget),
                    (prepared.core.a,prepared.core.gsum,prepared.core.h,*prepared.core.metadata),
                    tuple((cut,cut.row,cut.at,cut.rec,cut.conv,
                           gdn_chunk64._signature(cut.rec),gdn_chunk64._signature(cut.conv)) for cut in cuts))
        group.python_metadata_bytes = _python_metadata_bytes((group.segments,group.cuts,group.origins,group.indices,
            group.generations,group.cut_records,prepared.group_journal))
        if prepared.group_host_budget is not None and group.python_metadata_bytes > prepared.group_host_budget:
            raise MemoryError("G3 accepted frame exceeds its explicit Python journal reserve")
        readers = tuple(state.history_native_readers for state,_,_ in segs)
        try:
            for reader in readers:
                reader.add(group)
        except BaseException as error:  # noqa: BLE001 - rollback preserves and re-raises this primary after reader cleanup.
            rollback(group,error,lambda: finish(lambda reader=reader: reader.discard(group) for reader in readers))
        prepared.active = group
        return group

    def authorize(self, state, layer, at, rows, cuts):
        if (self.retired or self.sealed or self.error is not None or self.prepared.active is not self
                or state not in self.indices or type(layer) is not int or not 0 <= layer < self.prepared.group_layers):
            raise RuntimeError("G3 accepted-target frame lease is not live")
        current = (self.prepared.frontend,self.prepared.core,self.prepared.workspace,self.prepared.family,
                   self.prepared.event,self.prepared.stream,self.prepared.group_journal,
                   self.prepared.group_layers,self.prepared.group_segments,self.prepared.group_host_budget)
        core = self.prepared.core
        if (any(a is not b for a,b in zip(current[:7],self.prepared_owners[:7]))
                or current[7:] != self.prepared_owners[7:]
                or len(core.metadata)+3 != len(self.core_owners)
                or any(a is not b for a,b in zip((core.a,core.gsum,core.h,*core.metadata),self.core_owners))
                or any(type(c.row) is not int or type(c.at) is not int or c.row != row or c.at != origin
                       or c.rec is not rec or c.conv is not conv or gdn_chunk64._signature(rec) != rs
                       or gdn_chunk64._signature(conv) != cs
                       for c,row,origin,rec,conv,rs,cs in self.cut_records)):
            raise RuntimeError("G3 accepted-target prepared/cut ownership changed")
        expected = self.segments[self.indices[state]]
        if at != expected[1] or rows != expected[2]-expected[1] or cuts != self.cuts[at]:
            raise ValueError("G3 submission differs from its complete packed target frame")
        for owned,generation,rec,conv,cur in self.generations:
            primary = getattr(owned,'_history_error',None)
            if primary is None:
                primary = getattr(owned,'_kv_error',None)
            if primary is not None:
                self.fail(primary)
                raise primary
            if (owned.slot_generation != generation or owned.rec is not rec or owned.conv is not conv
                    or owned.cur is not cur):
                raise RuntimeError("G3 accepted-target State ownership was invalidated")

    def submit(self, state, layer, projection, weight, bias, norm, eps, output, sums, *,
               a_log=None, a_neg=None, at=0):
        cuts = self.cuts.get(at,())
        self.authorize(state,layer,at,projection.shape[0],cuts)
        slot = layer*self.prepared.group_segments+self.indices[state]
        if self.prepared.group_journal[slot] is not None:
            raise RuntimeError("G3 layer/State submitted twice in one accepted frame")
        command = self.prepared.bind(state,layer,projection,weight,bias,norm,eps,output,sums,
            a_log=a_log,a_neg=a_neg,cuts=cuts,cut_budget_bytes=self.cut_budget_bytes,at=at,_group=self)
        charge = _python_metadata_bytes(command)
        if (self.prepared.group_host_budget is not None
                and self.python_metadata_bytes+charge > self.prepared.group_host_budget):
            raise MemoryError("G3 command would exceed its prepaid Python journal reserve")
        self.python_metadata_bytes += charge
        self.prepared.group_journal[slot] = command
        self.submitted = True
        try:
            result = command.run()
        except BaseException as error:
            self.fail(error)
            raise
        self.count += 1
        return result

    def fail(self, error):
        self.error = self.prepared.error = error
        for state,_,_ in self.segments:
            state._history_error = state._kv_error = error

    def target_started(self):
        if self.sealed or self.retired or self.error is not None:
            raise RuntimeError("G3 accepted-target frame is not issuable")
        self.submitted = True

    def mark_validated(self):
        if (not self.sealed or self.error is not None
                or any(getattr(state,'_kv_pending',False) or getattr(state,'_history_error',None) is not None
                       or getattr(state,'_kv_error',None) is not None for state,_,_ in self.segments)):
            raise RuntimeError("G3 target frame requires the original authoritative whole-frame codec check")
        self.validated = True

    def seal(self):
        if self.sealed or self.retired or self.error is not None:
            raise RuntimeError("G3 accepted-target frame cannot be sealed")
        if self.count != self.prepared.group_layers*len(self.segments):
            raise RuntimeError("G3 target frame did not submit every linear layer/State")
        state,start,end = self.segments[0]
        self.authorize(state,0,start,end-start,self.cuts[start])
        try:
            self.prepared_owners[4].record(self.prepared_owners[5])
            self.sealed = True
        except BaseException as error:
            self.fail(error)
            raise

    def wait_native_reuse(self):
        if self.retired:
            return
        if self.prepared.active is not self:
            raise RuntimeError("G3 accepted-target journal owner changed")
        if not self.sealed and self.error is None and self.submitted:
            raise RuntimeError("G3 State cannot be reused before target submission is sealed")
        if self.error is None and self.submitted and not self.validated:
            raise RuntimeError("G3 target State reuse requires authoritative whole-frame validation")
        primary = self.error
        try:
            if self.sealed:
                self.prepared_owners[4].synchronize()
            elif self.submitted:
                self.prepared_owners[5].synchronize()
        except BaseException as cleanup:
            self.fail(primary if primary is not None else cleanup)
            if primary is None:
                raise
            raise_failures(primary,[cleanup])
        if primary is not None:
            raise primary
        for state,_,_ in self.segments:
            state.history_native_readers.discard(self)
        for index in range(len(self.prepared.group_journal)):
            self.prepared.group_journal[index] = None
        self.retired = True
        self.prepared.active = None


def prepare(capacity, device, *, workspace_budget_bytes, compiler, family):
    """Cold explicit component opt-in; compiler/runtime retention is separate."""
    total = gdn_chunk64_half.layout(capacity)["device_bytes"]+gdn_chunk64.layout(capacity).workspace_bytes
    if type(workspace_budget_bytes) is not int or workspace_budget_bytes < total:
        raise MemoryError("G3 segmented prefill lacks complete frontend/core/raw-output credit")
    import torch

    from tensorfold.cuda.sm70.owned_family import PreparedFamily

    if family is not None and (type(family) is not PreparedFamily or family.device != torch.device(device)):
        raise TypeError("G3 prefill requires an already prepared explicit native-family shift owner")
    # Protected compiler/artifact admission precedes frontend imports/allocations.
    # One returned core is shared across all accepted-frame layer/State commands.
    core = gdn_chunk64.prepare(capacity,device,compiler=compiler,
                              workspace_budget_bytes=gdn_chunk64.layout(capacity).workspace_bytes)
    if family is None:
        from tensorfold.cuda.sm70.family_native import extension
        family = PreparedFamily(extension(),core.device)
    frontend = gdn_chunk64_half.prepare(core.device)
    workspace = gdn_chunk64_half.allocate_workspace(capacity,frontend.device,
        device_budget_bytes=gdn_chunk64_half.layout(capacity)["device_bytes"])
    stream = torch.cuda.current_stream(frontend.device)
    event = torch.cuda.Event()
    event.record(stream)  # Resolve the finite event resource during cold preparation.
    event.synchronize()
    return Prepared(frontend,core,workspace,family,event,stream)
