"""Closed default-off G3 target-prefill policy, preparation and adoption.

This is an unqualified numerical mode, not an attention/transport selector.
Compiler/library private retention is observed in the existing preparation
commitment and remains subject to actual model/resource qualification.
"""
import struct
import sys
import threading
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

POLICY = 'g3-target-prefill-half-raw-v1'


def _sha(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class Request:
    policy: str
    tensor_budget_bytes: int
    journal_host_budget_bytes: int
    cold_host_budget_bytes: int
    warmup_device_budget_bytes: int
    cuda_home: str
    nvcc_sha256: str
    artifact: dict

    @classmethod
    def parse(cls, value):
        from tensorfold.cuda.sm70.gdn_chunk64_artifact import MAX_METADATA_BYTES, OwnedArtifact

        fields = set(cls.__dataclass_fields__)
        if type(value) is not dict or set(value) != fields:
            raise ValueError('G3 target-prefill requires its complete explicit private artifact/resource contract')
        result = cls(**value)
        if (result.policy != POLICY or not _sha(result.nvcc_sha256) or type(result.cuda_home) is not str
                or not Path(result.cuda_home).is_absolute() or '\x00' in result.cuda_home
                or any(type(getattr(result,key)) is not int or not 0 < getattr(result,key) < 2**63 for key in
                       ('tensor_budget_bytes','journal_host_budget_bytes','cold_host_budget_bytes','warmup_device_budget_bytes'))):
            raise ValueError('G3 target-prefill needs positive bounded credits and the protected CUDA identity')
        required = set(OwnedArtifact.__dataclass_fields__)-{'native_base_wheel'}
        artifact = result.artifact
        if type(artifact) is not dict or not required <= artifact.keys() or set(artifact)-required-{'native_base_wheel'}:
            raise ValueError('G3 target-prefill needs the existing closed OwnedArtifact contract')
        if (type(artifact['attestation_bytes']) is not int or not 0 < artifact['attestation_bytes'] <= MAX_METADATA_BYTES
                or type(artifact['hash_byte_credit']) is not int or not 0 < artifact['hash_byte_credit'] < 2**63
                or not _sha(artifact['attestation_sha256'])
                or any(type(artifact.get(key)) is not str or not Path(artifact[key]).is_absolute()
                       or '\x00' in artifact[key] for key in ('attestation','wheel'))
                or artifact.get('native_base_wheel') is not None and
                   (type(artifact['native_base_wheel']) is not str or not Path(artifact['native_base_wheel']).is_absolute())):
            raise ValueError('G3 target-prefill artifact identities/credit are invalid')
        return result

    def as_dict(self):
        return asdict(self)


def forecast(text, rows, segments, request):
    from .gdn_chunk64_prefill import cut_bytes

    kinds = text.get('layer_types',())
    layers = sum(kind == 'linear_attention' for kind in kinds)
    if (type(rows) is not int or not 1 <= rows <= 65535 or type(segments) is not int or not 1 <= segments <= rows
            or len(kinds) != 48 or layers != 36
            or tuple(text.get(key) for key in ('linear_num_key_heads','linear_num_value_heads','linear_key_head_dim',
                                               'linear_value_head_dim','linear_conv_kernel_dim')) != (16,48,128,128,4)):
        raise ValueError('G3 target-prefill requires actual SC117 TP8 geometry and bounded pbuf/slots')
    payload = 4936*rows+21
    warmup = 5752*rows+804656
    commands = layers*segments
    # This exact root list is charged separately from the explicit reserve for
    # transient view/native-wrapper metadata; it is not a total C++ heap claim.
    journal_root = sys.getsizeof([])+commands*struct.calcsize('P')
    if (payload > request.tensor_budget_bytes or warmup > request.warmup_device_budget_bytes
            or journal_root > request.journal_host_budget_bytes):
        raise MemoryError('G3 target-prefill explicit workspace/warmup/journal credit is insufficient')
    return {'rows':rows,'segments':segments,'layers':layers,'tensor_payload_bytes':payload,
            'legacy_replay_bytes_omitted':2608*rows,'warmup_temporary_bytes':warmup,
            'journal_commands':commands,'maximum_spans':commands*4,'journal_root_bytes':journal_root,
            'journal_host_budget_bytes':request.journal_host_budget_bytes,
            'cold_host_budget_bytes':request.cold_host_budget_bytes,
            'cut_snapshot_payload_credit':0,
            'maximum_cut_snapshot_payload':cut_bytes(layers,segments*3),
            'cut_accounting':'existing retained-recovery/POINTS category; ownership escapes through Cut',
            'runtime_private_metadata':'bounded frame count; measured model/host gates required'}


class Owner:
    def __init__(self, request, facts):
        self.request,self.facts = request,facts
        self.prepared = self.weights = self.buffers = None
        self.closed = False
        self.started = False
        self.host_admission = None

    def admit_host(self):
        from tensorfold.cuda.capacity import admit_host_commitment

        if self.host_admission is not None or self.started or self.closed:
            raise RuntimeError('G3 host admission must precede its one-time cold preparation')
        self.host_admission = admit_host_commitment(8*self.request.cold_host_budget_bytes)
        return deepcopy(self.host_admission)

    def prepare(self, device, family=None):
        import torch

        from tensorfold.cuda.sm70.gdn_chunk64 import Compiler
        from tensorfold.cuda.sm70.gdn_chunk64_artifact import OwnedArtifact

        from .gdn_chunk64_prefill import prepare

        if self.started or self.prepared is not None or self.closed:
            raise RuntimeError('G3 serving preparation is one-time')
        if self.host_admission is None:
            raise RuntimeError('G3 cold preparation requires its fresh cohort host commitment admission')
        self.started = True
        compiler = Compiler(self.request.cuda_home,self.request.nvcc_sha256,OwnedArtifact(**self.request.artifact))
        # inspect refuses absent/unattested packages BEFORE importing TileLang.
        # No installation or library/algorithm fallback belongs to this mode.
        with torch.inference_mode():
            from importlib.metadata import PackageNotFoundError
            try:
                self.prepared = prepare(self.facts['rows'],device,
                    workspace_budget_bytes=self.request.tensor_budget_bytes,compiler=compiler,family=family)
            except PackageNotFoundError as error:
                raise RuntimeError('requested private G3 prefill requires the protected owned GDN image') from error
            self.prepared.configure_groups(self.facts['layers'],self.facts['segments'],
                                           host_budget_bytes=self.request.journal_host_budget_bytes)
            self._warm()

    def _warm(self):
        import torch

        p = self.prepared
        rows,device = self.facts['rows'],p.frontend.device
        projection = torch.zeros(rows,2060,dtype=torch.float16,device=device)
        history = torch.zeros(3,1280,dtype=torch.float16,device=device)
        weight = torch.zeros(1280,4,dtype=torch.float16,device=device)
        parameter = torch.full((6,),-0.001,dtype=torch.float32,device=device)
        bias = torch.zeros(6,device=device)
        norm = torch.ones(128,dtype=torch.float16,device=device)
        out = torch.empty(rows,768,dtype=torch.float16,device=device)
        sums = torch.empty(rows,24,device=device)
        initial = torch.zeros(1,6,128,128,device=device)
        final = torch.empty_like(initial)
        half = p.frontend.bind(p.workspace,projection,history,weight,bias,norm,1e-6,out,sums,a_neg=parameter)
        q,k,v,g,beta,raw = half.core_operands
        bound = p.core.bind(q,k,v,g,beta,initial,raw,final)
        half.front();bound.run();half.back()
        p.family.shift_windows(history[None],projection[None,:1],1,1280)
        p.event.record(p.stream)
        p.event.synchronize()  # Cold residency, never a serving layer wait.

    def bind_weights(self, weights):
        if self.prepared is None or self.weights is not None:
            raise RuntimeError('G3 target weights require one completed cold preparation')
        linear = tuple(layer for layer in weights.layers if layer.linear)
        if (len(linear) != self.facts['layers'] or weights.comm is None or weights.comm.world != 8
                or any(layer.gdn is None for layer in linear)):
            raise ValueError('G3 target layer/rank ownership differs from prepared SC117')
        import torch

        device = self.prepared.frontend.device
        from .gdn_chunk64_prefill import _plain
        for layer in linear:
            g = layer.gdn
            if (g.a_log is None) == (g.a_neg is None):
                raise ValueError('G3 target requires exactly one typed decay source')
            parameter = g.a_log if g.a_neg is None else g.a_neg
            if any(not _plain(t,shape,dtype,device) for t,shape,dtype in
                   ((g.conv,(1280,4),torch.float16),(parameter,(6,),torch.float32),
                    (g.dt_bias,(6,),torch.float32),(g.norm,(128,),torch.float16))):
                raise ValueError('G3 target weights differ from protected Half per-rank geometry')
        self.weights = weights
        self.linear = linear
        weights._private_g3_prefill = self

    def adopt(self, weights, decoder):
        from .gdn_chunk64_prefill import cut_bytes

        buffers = decoder.pbuf
        if self.weights is not weights or buffers.rows != self.facts['rows'] or not buffers.prefill:
            raise ValueError('G3 serving pbuf differs from its actual paid target row capacity')
        self.buffers = buffers
        self.states = frozenset(decoder.states)
        self.facts['adopted_segments'] = min(len(self.states),buffers.rows)
        self.facts['cut_snapshot_payload_credit'] = cut_bytes(self.facts['layers'],self.facts['adopted_segments']*3)
        buffers.g3_prefill_owner = self

    def begin(self, weights, buffers, segs, cuts):
        if self.closed or self.weights is not weights or self.buffers is not buffers or buffers.split is not None:
            raise RuntimeError('G3 accepted-target frame differs from its adopted eager pbuf')
        if len(segs) > self.facts['adopted_segments'] or any(state not in self.states for state,_,_ in segs):
            raise ValueError('G3 target frame must borrow the actually admitted resident State slots')
        return self.prepared.begin_group(segs,cuts,cut_budget_bytes=self.facts['cut_snapshot_payload_credit'])

    def submit(self, frame, gdn, state, layer, buffers, start, end, cfg):
        if (self.buffers is not buffers or frame is not self.prepared.active
                or not 0 <= layer < len(self.linear) or self.linear[layer].gdn is not gdn):
            raise RuntimeError('G3 target submission lost its actual layer/weight/frame owner')
        import torch

        # The existing model warmup uses no_grad, whereas the G3 primitive's
        # explicit API requires inference_mode. Scope it to this no-grad native
        # component; ordinary model buffer/output owners stay unchanged.
        with torch.inference_mode():
            return frame.submit(state,layer,buffers.proj[0,start:end],gdn.conv,gdn.dt_bias,gdn.norm,cfg.eps,
                buffers.gout[start:end],buffers.gxs[start:end],a_log=gdn.a_log,a_neg=gdn.a_neg,at=start)

    def handoff_to_current_thread(self):
        import torch

        p = self.prepared
        if (p.active is not None or p.error is not None
                or torch.cuda.current_device() != p.frontend.device.index
                or torch.cuda.is_current_stream_capturing()):
            raise RuntimeError('G3 worker handoff requires the current uncaptured device and no live borrower')
        p.event.synchronize()
        p.stream = torch.cuda.current_stream(p.frontend.device)
        p.core.stream,p.core.owner = p.stream.cuda_stream,threading.current_thread()

    def record(self):
        p = self.prepared
        return {'policy':POLICY,'scope':'accepted-target-prefill-only;single-actor-stream-group',
                'numerical_mode':'g3-half-unscaled-qk-raw-gate-rounded-beta-v1',
                'forecast':dict(self.facts),'artifact':dict(self.request.artifact),
                'cold_host_admission':deepcopy(self.host_admission),
                'compiler':None if p is None else deepcopy(p.core.compiler_report),
                'compiled_kernel_sets':int(p is not None),'workspace_slabs':int(p is not None),
                'qualification':'private-unqualified; native group/model/quality/full-window/P3 gates pending'}

    def close(self):
        if self.closed:
            return
        if self.prepared is not None:
            if self.prepared.active is not None:
                self.prepared.active.wait_native_reuse()
            self.prepared.event.synchronize()
        self.closed = True
