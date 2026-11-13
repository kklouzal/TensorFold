#!/usr/bin/env python3
"""Explicit native G3 numerical trial. No model/serving hook or speed claim.

Run only under a coordinator-granted GPU/resource lease. Error measurements do
not constitute numerical acceptance: this new mode has no authorized tolerance.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def reference(q, k, v, raw_log_gate, beta, initial):
    """Independent CPU float64 serial delta rule, on the actual component inputs."""
    import torch
    q,k,v,g,b,s = [x.detach().cpu().to(torch.float64) for x in (q,k,v,raw_log_gate,beta,initial)]
    s = s.clone()
    outputs = []
    for row in range(q.shape[1]):
        key = k[0,row].repeat_interleave(3,dim=0)
        query = q[0,row].repeat_interleave(3,dim=0) * (128**-0.5)
        s = s * torch.exp(g[0,row])[None,:,None,None]
        prediction = torch.einsum("bhvk,hk->bhv",s,key)
        delta = (v[:,row] - prediction) * b[:,row,:,None]
        s = s + delta[:,:,:,None] * key[None,:,None,:]
        outputs.append(torch.einsum("bhvk,hk->bhv",s,query))
    return torch.stack(outputs,dim=1),s


def half_frontend_reference(projection, history, weight, parameter, bias, *, negative_decay):
    """Independent CPU boundary oracle; no native equivalence/tolerance claim.

    Ordered FP32 convolution and explicit Half cast points model the producer.
    Torch transcendental/reduction implementations are diagnostic references,
    not assumed bit-identical to CUDA intrinsics and warp reduction trees.
    """
    import torch

    p,h,w,a,dt = [value.detach().cpu().float() for value in (projection,history,weight,parameter,bias)]
    rows = p.shape[0]
    source = torch.cat((h,p[:,:1280]),dim=0)
    conv = torch.zeros((rows,1280),dtype=torch.float32)
    for tap in range(4):
        conv = conv + source[tap:tap+rows]*w[:,tap]
    activated = (conv/(1+torch.exp(-conv))).half().float()
    q = activated[:,:256].reshape(1,rows,2,128)
    k = activated[:,256:512].reshape(1,rows,2,128)
    q = (q/torch.sqrt(q.square().sum(-1,keepdim=True)+1e-6)).half()
    k = (k/torch.sqrt(k.square().sum(-1,keepdim=True)+1e-6)).half()
    v = activated[:,512:].reshape(1,rows,6,128).half()
    shifted = p[:,2054:2060]+dt
    softplus = torch.where(shifted>20,shifted,torch.log1p(torch.exp(shifted)))
    rate = a if negative_decay else -torch.exp(a)
    raw = (rate*softplus)[None]
    beta = (1/(1+torch.exp(-p[:,2048:2054]))).half().float()[None]
    return q,k,v,raw,beta


def half_epilogue_reference(raw, projection, norm, eps):
    """Independent CPU Half cast/gated-RMSNorm/group32 boundary diagnostic."""
    import torch

    y,p,n = [value.detach().cpu().float() for value in (raw,projection,norm)]
    rows = p.shape[0]
    y = y.reshape(rows,6,128)
    inv = 1/torch.sqrt(y.square().mean(-1,keepdim=True)+eps)
    normalized = ((y*inv).half().float()*n).half().float()
    z = p[:,1280:2048].reshape(rows,6,128)
    output = (normalized/(1+torch.exp(-z))).half().reshape(rows,768)
    sums = output.float().reshape(rows,24,32).sum(-1)
    return output,sums


def unchanged_initial(initial, original):
    """Compare storage bits of the admitted contiguous FP32 state banks."""
    import torch
    return bool(torch.equal(initial.view(torch.int32),original.view(torch.int32)))


def errors(actual, expected):
    import torch
    a=actual.detach().cpu().to(torch.float64)
    e=expected.detach().cpu().to(torch.float64)
    d=(a-e).abs()
    finite=torch.isfinite(a) & torch.isfinite(e)
    selected=d[finite]
    return {"finite": bool(torch.isfinite(a).all()), "nonfinite_elements": int((~torch.isfinite(a)).sum()),
            "error_elements": int(finite.sum()),
            "max_abs": float(selected.max()) if selected.numel() else None,
            "rms": float((selected.square().mean()).sqrt()) if selected.numel() else None,
            "reference_max_abs": float(e.abs().max())}



def cuda_entry(source):
    """Require one unique entry; generated declarations may repeat its name."""
    import re

    pattern=r'(?:extern\s+"C"\s+)?__global__\s+void\s+(?:__launch_bounds__\([^\)]*\)\s+)?(\w+)'
    names=set(re.findall(pattern,source))
    if len(names)!=1:
        raise RuntimeError("cannot bind exactly one generated CUDA entry per G3 stage")
    return names.pop()


def nondefault_stream_trial(args, compiler, progress):
    """Actual producer/core/consumer ordering plus recorded CUDA stream evidence.

    This diagnostic refuses to substitute profiler headers or CPU annotations
    for actual native GPU records. It does not qualify physical clock accuracy.
    """
    import torch

    from tensorfold.cuda.sm70.gdn_chunk64 import layout, prepare

    stream=torch.cuda.Stream(device=args.device)
    default=torch.cuda.default_stream(args.device)
    if stream.cuda_stream==default.cuda_stream:
        raise RuntimeError("nondefault stream control did not create a distinct stream")
    with torch.inference_mode(),torch.cuda.stream(stream):
        rows=65
        q=torch.full((1,rows,2,128),128**-0.5,dtype=torch.float16,device=args.device)
        k=q.clone()
        v=torch.ones((1,rows,6,128),dtype=torch.float16,device=args.device)
        g=torch.zeros((1,rows,6),dtype=torch.float32,device=args.device)
        beta=torch.ones_like(g)
        initial=torch.zeros((1,6,128,128),dtype=torch.float32,device=args.device)
        output=torch.full_like(v,float("nan"));final=torch.full_like(initial,float("nan"))
        prepared=prepare(rows,args.device,workspace_budget_bytes=layout(rows).workspace_bytes,compiler=compiler)
        bound=prepared.bind(q,k,v,g,beta,initial,output,final)
        symbols=[]
        for kernel in (prepared.cumsum,prepared.kkt,prepared.core):
            symbols.append(cuda_entry(kernel.get_kernel_source()))
        torch.cuda.synchronize(args.device)
        progress["current"]={"stage":"nondefault-stream-profile","rows":rows}
        # CUDA must be requested by the process's FIRST Kineto initialization.
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            with torch.profiler.record_function("g3.stream.producer"):
                v.zero_()
            with torch.profiler.record_function("g3.stream.component"):
                bound.run()
            with torch.profiler.record_function("g3.stream.consumer"):
                checksum=output.float().square().sum()
                done=torch.cuda.Event()
                done.record(stream)
            done.synchronize()
        profile.export_chrome_trace(str(args.stream_trace_output))
        ordering_ok=bool(checksum.item()==0 and (final==0).all() and (initial==0).all())
    raw=args.stream_trace_output.read_bytes()
    events=json.loads(raw)["traceEvents"]
    kernels=[event for event in events if event.get("cat")=="kernel"]
    native=[event for event in kernels if any(symbol in event.get("name","") for symbol in symbols)]
    streams={event.get("args",{}).get("stream") for event in kernels}
    devices={event.get("args",{}).get("device") for event in kernels}
    # Producer and consumer force additional Torch GPU work; a single recorded
    # stream for ALL kernel records ties the G3 launches to those Torch controls.
    proof=(len(native)==3 and len(kernels)>3 and len(streams)==1 and None not in streams
           and len(devices)==1 and None not in devices)
    result={"torch_raw_stream":stream.cuda_stream,"torch_default_raw_stream":default.cuda_stream,
            "generated_entry_names":symbols,"actual_kernel_count":len(kernels),
            "actual_g3_kernel_count":len(native),"recorded_stream_ids":sorted(streams,key=str),
            "recorded_device_ids":sorted(devices,key=str),"producer_core_consumer_exact_zero":ordering_ok,
            "actual_single_nondefault_stream_proof":proof,
            "trace_path":str(args.stream_trace_output),"trace_sha256":hashlib.sha256(raw).hexdigest(),
            "scope":"Recorded native activity and exact ordering control only; no general no-loss/physical-clock claim"}
    progress["nondefault_stream"]=result
    return result


SPAN_STATE_CAPACITY = 131
SPAN_STATE_LAYERS = 2
SPAN_STATE_CASES = ((63,31),(64,3),(65,1),(129,63),(129,64),(129,65),(131,65))
SPAN_STATE_SCHEMA = "tensorfold.gdn-chunk64-spans-state-native-diagnostics-v1"


class ObservedCUDAEvent:
    """Diagnostic API counter forwarding every call to a real CUDA event.

    Counts describe this adapter event's API only, never all runtime/FFI waits.
    The optional host exception follows positive genuine CUDA completion; it
    does not simulate a CUDA fault or replace native numerical execution.
    """
    def __init__(self, event):
        self.event = event
        self.record_calls = self.synchronize_calls = self.completed_synchronizations = 0
        self.host_failure = None

    def record(self, stream):
        self.record_calls += 1
        return self.event.record(stream)

    def synchronize(self):
        self.synchronize_calls += 1
        self.event.synchronize()
        self.completed_synchronizations += 1
        if self.host_failure is not None:
            raise self.host_failure


def retire_controlled_host_failure(frame, primary):
    """Consume only the declared identity; unexpected native errors propagate."""
    try:
        frame.wait_native_reuse()
    except RuntimeError as observed:
        if observed is not primary:
            raise
        return True
    return False


def diagnostic_state(device):
    """Use the real State allocator for two unweighted GDN-only layers.

    This finite local geometry models a TP8 rank's GDN slice. There are no
    attention/MTP caches, loaded model weights or TP/model qualification claims.
    """
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda.state import State

    cfg = SimpleNamespace(hidden=2560,streams=4,nk=2,nv=6,dk=128,dv=128,conv_kernel=4,
        conv_dim=1280,index_ratio=4,index_dim=128,kv_heads=2,head_dim=256,
        ple_kernel=1,ngram_size=1,ple_layers=[])
    weights = SimpleNamespace(cfg=cfg,device=device,mtp=None,comm=None,
        layers=[SimpleNamespace(index=index,linear=True) for index in range(SPAN_STATE_LAYERS)],
        meta={"execution_dtype":"float16","execution_provider":"sm70"})
    return State(weights,SPAN_STATE_CAPACITY,SPAN_STATE_CAPACITY,kv_dtype="fp16")


def same_storage_bits(left, right):
    import torch

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    kind = {torch.float16:torch.int16,torch.float32:torch.int32}[left.dtype]
    return bool(torch.equal(left.detach().cpu().contiguous().view(kind),
                            right.detach().cpu().contiguous().view(kind)))


def boundary_errors(actual, expected):
    """Reuse the serial diagnostics while retaining JSON-safe nonfinite facts."""
    import math

    import torch

    result = errors(actual,expected)
    e = expected.detach().cpu().double()
    result["reference_nonfinite_elements"] = int((~torch.isfinite(e)).sum())
    if not math.isfinite(result["reference_max_abs"]):
        result["reference_max_abs"] = None
    return result


def spans_state_trial(args, compiler, progress):
    """Real CUDA two-layer submissions, exact copies and measured errors only."""
    import torch

    from tensorfold.cuda.sm70 import gdn_chunk64
    from tensorfold.cuda.sm70.owned_family import prepare as prepare_family
    from tensorfold.families.qwen4_exp.cuda import gdn_chunk64_prefill as adapter
    from tensorfold.families.qwen4_exp.cuda.forward import Cut

    stream = torch.cuda.Stream(device=args.device)
    default = torch.cuda.default_stream(args.device)
    if stream.cuda_stream == default.cuda_stream:
        raise RuntimeError("span/State trial requires an actual distinct nondefault stream")
    records = progress["results"]
    with torch.inference_mode(),torch.cuda.stream(stream):
        family = prepare_family(args.device)
        credit = SPAN_STATE_CAPACITY*4936+21
        owners = [adapter.prepare(SPAN_STATE_CAPACITY,args.device,workspace_budget_bytes=credit,
                                  compiler=compiler,family=family) for _ in range(SPAN_STATE_LAYERS)]
        generated = [{name:{"entry":cuda_entry(kernel.get_kernel_source()),
                              "source_sha256":hashlib.sha256(kernel.get_kernel_source().encode()).hexdigest()}
                      for name,kernel in zip(("cumsum","kkt","core"),
                                            (owner.core.cumsum,owner.core.kkt,owner.core.core))}
                     for owner in owners]
        # Instrument only this owned event API, before any binding seals it.
        for owner in owners:
            owner.event = ObservedCUDAEvent(owner.event)
        for case_index,(rows,cut_row) in enumerate(SPAN_STATE_CASES):
            progress["current"] = {"stage":"span-State-case","rows":rows,"cut":cut_row}
            state = diagnostic_state(torch.device(args.device))
            state.rec.normal_(std=0.125)
            state.rec[...,0,0] = -0.0
            state.conv.normal_(std=0.05)
            state.conv[...,0] = -0.0
            original_bank = case_index%2
            state.cur = [original_bank]*SPAN_STATE_LAYERS
            original_all_banks = state.rec.clone()
            original_rec = state.rec[original_bank].clone()
            original_conv = state.conv.clone()
            cut = Cut(cut_row,rec=torch.empty_like(state.rec[0]),conv=torch.empty_like(state.conv))
            frames,inputs,outputs,immutable = [],[],[],[]
            alias_refused = bad_cut_refused = preflight_unchanged = False
            for layer,owner in enumerate(owners):
                p = torch.randn(rows,2060,dtype=torch.float16,device=args.device)*0.125
                w = torch.randn(1280,4,dtype=torch.float16,device=args.device)*0.125
                parameter = torch.full((6,),-0.5 if layer == 0 else 0.,dtype=torch.float32,device=args.device)
                decay_source = {"a_neg" if layer == 0 else "a_log":parameter}
                bias = torch.linspace(-1,1,6,device=args.device)
                norm = torch.ones(128,dtype=torch.float16,device=args.device)
                out = torch.empty(rows,768,dtype=torch.float16,device=args.device)
                sums = torch.empty(rows,24,dtype=torch.float32,device=args.device)
                inputs.append((p,w,parameter,bias,norm))
                immutable.append(tuple(tensor.clone() for tensor in inputs[-1]))
                outputs.append((out,sums))
                for tensor in (owner.workspace.q,owner.workspace.k,owner.workspace.v,owner.workspace.raw_output):
                    tensor.fill_(7)
                for tensor in (owner.workspace.raw_gate,owner.workspace.beta):
                    tensor.fill_(13)
                owner.core.a.fill_(9);owner.core.gsum.fill_(9)
                if layer == 0:
                    try:
                        owner.bind(state,layer,p,w,bias,norm,1e-6,p.view(-1)[:rows*768].view(rows,768),sums,
                                   **decay_source)
                    except ValueError:
                        alias_refused = True
                    bad = Cut(cut_row,rec=torch.empty_like(state.rec[0]),conv=state.conv)
                    try:
                        owner.bind(state,layer,p,w,bias,norm,1e-6,out,sums,**decay_source,
                                   cuts=(bad,),cut_budget_bytes=adapter.cut_bytes(SPAN_STATE_LAYERS,1))
                    except ValueError:
                        bad_cut_refused = True
                    preflight_unchanged = (same_storage_bits(state.rec,original_all_banks)
                                           and same_storage_bits(state.conv,original_conv))
                frame = owner.bind(state,layer,p,w,bias,norm,1e-6,out,sums,**decay_source,
                    cuts=(cut,),cut_budget_bytes=adapter.cut_bytes(SPAN_STATE_LAYERS,1))
                frames.append(frame)
            waits_before = tuple(owner.event.synchronize_calls for owner in owners)
            for frame in frames:
                frame.run()  # Both real layers submit before any explicit retirement.
            no_adapter_event_wait = tuple(owner.event.synchronize_calls for owner in owners) == waits_before
            both_active = all(owner.active is frame and frame.recorded and not frame.retired
                              for owner,frame in zip(owners,frames))
            try:
                owner = owners[0]
                p,w,parameter,bias,norm = inputs[0]
                owner.bind(state,0,p,w,bias,norm,1e-6,*outputs[0],a_neg=parameter)
            except RuntimeError as refusal:
                reuse_refused = "retirement" in str(refusal)
            else:
                reuse_refused = False
            for frame in frames:
                frame.wait_native_reuse()
            layers = []
            for layer,(owner,source,before,pair) in enumerate(zip(owners,inputs,immutable,outputs)):
                p,w,parameter,bias,norm = source
                out,sums = pair
                q,k,v,g,beta,raw = (tensor[:,:rows] for tensor in
                    (owner.workspace.q,owner.workspace.k,owner.workspace.v,owner.workspace.raw_gate,
                     owner.workspace.beta,owner.workspace.raw_output))
                # The one cut bank is not overwritten by the final span. This
                # compares the actual copied native result, including -0 bits.
                actual_cut_bank = state.rec[1-original_bank,layer]
                cut_copy = same_storage_bits(cut.rec[layer],actual_cut_bank)
                prefix_tail = torch.cat((original_conv[layer:layer+1],p[None,:,:1280]),dim=1)[:,cut_row:cut_row+3]
                final_tail = torch.cat((original_conv[layer:layer+1],p[None,:,:1280]),dim=1)[:,rows:rows+3]
                expected,expected_state = reference(q,k,v,g,beta,original_rec[layer:layer+1])
                _,expected_cut = reference(q[:,:cut_row],k[:,:cut_row],v[:,:cut_row],g[:,:cut_row],
                                           beta[:,:cut_row],original_rec[layer:layer+1])
                front = half_frontend_reference(p,original_conv[layer],w,parameter,bias,negative_decay=layer == 0)
                expected_out,expected_sums = half_epilogue_reference(expected.half(),p,norm,1e-6)
                actual_raw_out,actual_raw_sums = half_epilogue_reference(raw,p,norm,1e-6)
                max_span = max(cut_row,rows-cut_row)
                inactive = [*[(tensor[:,rows:],7) for tensor in (owner.workspace.q,owner.workspace.k,
                             owner.workspace.v,owner.workspace.raw_output)],
                            (owner.workspace.raw_gate[:,rows:],13),(owner.workspace.beta[:,rows:],13),
                            (owner.core.a[:,max_span:],9),(owner.core.gsum[:,max_span:],9)]
                layer_record = {"layer":layer,"decay_representation":"GGUF-A_neg" if layer == 0 else "HF-A_log",
                    "actual_cut_bank_copy_bits":cut_copy,
                    "last_span_initial_bank_unchanged_vs_cut_copy_bits":cut_copy,
                    "cut_conv_tail_bits":same_storage_bits(cut.conv[layer:layer+1],prefix_tail),
                    "final_conv_tail_bits":same_storage_bits(state.conv[layer:layer+1],final_tail),
                    "immutable_projection_weights_parameters_bits":all(same_storage_bits(a,b) for a,b in zip(source,before)),
                    "inactive_capacity_bits":all(same_storage_bits(t,torch.full_like(t,value)) for t,value in inactive),
                    "frontend":{name:boundary_errors(actual,ref) for name,actual,ref in
                                zip(("q","k","v","raw_gate","beta"),(q,k,v,g,beta),front)},
                    "output_vs_serial":boundary_errors(out,expected_out),
                    "sums_vs_serial":boundary_errors(sums,expected_sums),
                    "raw_vs_serial":boundary_errors(raw,expected),
                    "final_state_vs_serial":boundary_errors(state.rec[state.cur[layer],layer:layer+1],expected_state),
                    "cut_state_vs_serial":boundary_errors(cut.rec[layer:layer+1],expected_cut),
                    "epilogue_actual_raw_output":boundary_errors(out,actual_raw_out),
                    "epilogue_actual_raw_sums":boundary_errors(sums,actual_raw_sums)}
                layers.append(layer_record)
            metadata_unchanged = all(tuple(map(gdn_chunk64._signature,frame.retained)) == frame.signatures
                                     for frame in frames)
            record = {"rows":rows,"cut":cut_row,"capacity":SPAN_STATE_CAPACITY,"layers":layers,
                "storage_metadata_unchanged":metadata_unchanged,
                "bank_parity":state.cur == [original_bank]*SPAN_STATE_LAYERS,
                "both_layers_submitted_before_explicit_retirement":both_active,
                "adapter_event_synchronize_calls_unchanged_before_retirement":no_adapter_event_wait,
                "active_reuse_refused":reuse_refused,"alias_refused":alias_refused,
                "bad_cut_refused":bad_cut_refused,"preflight_state_bits_unchanged":preflight_unchanged,
                "positive_retirement":all(frame.retired and owner.active is None for owner,frame in zip(owners,frames)),
                "state_reader_journal_empty":not state.history_native_readers}
            records.append(record)
        # A declared host observer failure AFTER genuine event completion tests
        # containment/retention. It is never reported as a CUDA failure/control.
        progress["current"] = {"stage":"controlled-host-retirement-failure"}
        owner = owners[0]
        p,w,parameter,bias,norm = inputs[0]
        control_bank = state.cur[0]
        state.rec[control_bank,0,...,0,0] = -0.0
        control_initial = state.rec[control_bank,0].clone()
        frame = owner.bind(state,0,p,w,bias,norm,1e-6,*outputs[0],a_neg=parameter)
        frame.run()  # One span: the original bank must remain readonly in this control.
        primary = RuntimeError("declared host observer failure after actual CUDA completion")
        owner.event.host_failure = primary
        completions_before = owner.event.completed_synchronizations
        primary_preserved = retire_controlled_host_failure(frame,primary)
        host_failure = {"scope":"Host observer exception after genuine CUDA event completion; no CUDA fault injected",
            "genuine_completion_before_host_exception":owner.event.completed_synchronizations == completions_before+1,
            "single_span_bank_publication":state.cur[0] == 1-control_bank,
            "single_span_initial_bank_readonly_bits":same_storage_bits(state.rec[control_bank,0],control_initial),
            "primary_preserved":primary_preserved,"state_poisoned":state._history_error is primary and state._kv_error is primary,
            "owners_retained":owner.active is frame and bool(frame.retained) and not frame.retired}
        progress["host_failure_control"] = host_failure
        return {"schema":SPAN_STATE_SCHEMA,"phase":"spans-state","main_numerical_cases":"unrun",
            "quality_acceptance":False,"performance_qualified":False,"case_count":len(records),
            "capacity":SPAN_STATE_CAPACITY,"layers":SPAN_STATE_LAYERS,"seed":20261113,
            "stream_trace_status":"unrun; this phase requests no profiler",
            "stream":{"actual_raw":stream.cuda_stream,"default_raw":default.cuda_stream,
                      "scope":"Same serialized actor stream; only owned adapter event API waits are counted"},
            "workspace_payload_bytes_per_owner":credit,"cut_payload_bytes":adapter.cut_bytes(SPAN_STATE_LAYERS,1),
            "actual_state_recurrent_payload_bytes":sum(t.numel()*t.element_size() for t in (state.rec,state.conv)),
            "actual_state_replay_scratch_bytes":sum(t.numel()*t.element_size() for scratch in state.scratch
                                                    for t in vars(scratch).values()),
            "family":family.record(),"frontend_native_module":owners[0].frontend.module.__name__,
            "compiler":compiler.inspect(),"generated_core":generated,
            "prepared_compiler_reports":[owner.core.compiler_report for owner in owners],
            "torch":torch.__version__,"device":torch.cuda.get_device_name(args.device),
            "source_inputs":progress["source_inputs"],"results":records,"host_failure_control":host_failure,
            "reason":"Native diagnostics only; numerical errors have no invented tolerance or model/quality/P3 acceptance"}


def validate_spans_state_result(result):
    if (result.get("schema") != SPAN_STATE_SCHEMA or result.get("phase") != "spans-state"
            or result.get("main_numerical_cases") != "unrun" or result.get("quality_acceptance") is not False
            or result.get("performance_qualified") is not False or result.get("case_count") != len(SPAN_STATE_CASES)
            or result.get("capacity") != SPAN_STATE_CAPACITY or result.get("layers") != SPAN_STATE_LAYERS
            or [(record["rows"],record["cut"]) for record in result["results"]] != list(SPAN_STATE_CASES)):
        raise SystemExit("G3 span diagnostics require their exact finite census; legacy18 cases remain unrun")
    flags = ("bank_parity","both_layers_submitted_before_explicit_retirement",
             "adapter_event_synchronize_calls_unchanged_before_retirement","active_reuse_refused",
             "alias_refused","bad_cut_refused","preflight_state_bits_unchanged","positive_retirement",
             "storage_metadata_unchanged","state_reader_journal_empty")
    bits = ("actual_cut_bank_copy_bits","last_span_initial_bank_unchanged_vs_cut_copy_bits",
            "cut_conv_tail_bits","final_conv_tail_bits","immutable_projection_weights_parameters_bits","inactive_capacity_bits")
    metrics = ("output_vs_serial","sums_vs_serial","raw_vs_serial","final_state_vs_serial","cut_state_vs_serial",
               "epilogue_actual_raw_output","epilogue_actual_raw_sums")
    for record in result["results"]:
        if (any(record[key] is not True for key in flags) or record["capacity"] != SPAN_STATE_CAPACITY
                or [layer["layer"] for layer in record["layers"]] != list(range(SPAN_STATE_LAYERS))):
            raise SystemExit("G3 native span/State ownership invariant failed; preserve report")
        for layer in record["layers"]:
            if (layer["decay_representation"] != ("GGUF-A_neg" if layer["layer"] == 0 else "HF-A_log")
                    or set(layer["frontend"]) != {"q","k","v","raw_gate","beta"}
                    or any(layer[key] is not True for key in bits)
                    or any(not layer[key]["finite"] or layer[key]["reference_nonfinite_elements"] for key in metrics)
                    or any(not item["finite"] or item["reference_nonfinite_elements"] for item in layer["frontend"].values())):
                raise SystemExit("G3 native copy/storage/finite invariant failed; numerical error magnitude remains diagnostic")
    if any(result["host_failure_control"][key] is not True for key in ("genuine_completion_before_host_exception","single_span_bank_publication",
                     "single_span_initial_bank_readonly_bits","primary_preserved","state_poisoned","owners_retained")):
        raise SystemExit("G3 native host-failure containment invariant failed; preserve report")


def run(args, progress):
    progress["phase"]=args.phase
    progress["main_numerical_cases"]="unrun" if args.phase in ("nondefault-stream","spans-state") else "incomplete"
    from tensorfold.cuda.sm70.gdn_chunk64 import Compiler, layout, prepare
    from tensorfold.cuda.sm70.gdn_chunk64_artifact import OwnedArtifact
    owned=(OwnedArtifact(str(args.owned_tilelang_attestation),args.owned_attestation_bytes,
                         args.owned_attestation_sha256,str(args.owned_tilelang_wheel),args.owned_hash_byte_credit,
                         str(args.owned_native_base_wheel) if args.owned_native_base_wheel is not None else None)
           if args.owned_tilelang_attestation is not None else None)
    compiler=Compiler(args.cuda_home,args.nvcc_sha256,owned)
    import tensorfold.cuda.sm70.gdn_chunk64 as component
    root=Path(component.__file__).parent
    progress["source_inputs"]={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                              for name in ("gdn_chunk64.py","gdn_chunk64_artifact.py",
                                           "gdn_chunk64_kernels.py","gdn_chunk64_provenance.json")}
    progress["compiler_admission"]=compiler.inspect()
    import torch
    torch.cuda.set_device(args.device)
    torch.manual_seed(20261113)
    if args.phase == "spans-state":
        from tensorfold.cuda.sm70 import gdn_chunk64_half
        from tensorfold.families.qwen4_exp.cuda import gdn_chunk64_prefill as adapter
        progress["source_inputs"].update({str(path.relative_to(root.parent.parent.parent)):
            hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (Path(adapter.__file__),Path(gdn_chunk64_half.__file__),root/"gdn_value8.cpp",root/"gdn_value8.cu")})
        return spans_state_trial(args,compiler,progress)
    results=progress["results"]
    for rows in ((1,63,64,65,128,129) if args.phase=="full" else ()):
        for case in ("zero", "nonzero_initial", "raw_underflow"):
            progress["current"]={"rows":rows,"case":case,"stage":"operands"}
            with torch.inference_mode():
                q=torch.nn.functional.normalize(torch.randn(1,rows,2,128,device=args.device),dim=-1).half()
                k=torch.nn.functional.normalize(torch.randn_like(q,dtype=torch.float32),dim=-1).half()
                v=torch.randn(1,rows,6,128,device=args.device,dtype=torch.float16)*0.125
                g=-torch.nn.functional.softplus(torch.randn(1,rows,6,device=args.device))
                b=torch.sigmoid(torch.randn_like(g))
                initial=torch.randn(1,6,128,128,device=args.device,dtype=torch.float32)*0.125
                initial[...,0,0]=-0.0
                if case == "zero":
                    v.zero_();initial.zero_()
                elif case == "raw_underflow":
                    g.fill_(-1000.0)
                original=initial.clone()
                output=torch.full_like(v,float("nan"))
                final=torch.full_like(initial,float("nan"))
                before=time.monotonic()
                progress["current"]["stage"]="prepare"
                p=prepare(rows,args.device,workspace_budget_bytes=layout(rows).workspace_bytes,compiler=compiler)
                progress["current"]["stage"]="bind"
                bound=p.bind(q,k,v,g,b,initial,output,final)
                progress["current"]["stage"]="launch-and-synchronize"
                bound.run()
                torch.cuda.synchronize()
                elapsed=time.monotonic()-before
                progress["current"]["stage"]="reference"
                expected,expected_state=reference(q,k,v,g,b,original)
                record={"rows":rows,"case":case,"cold_prepare_and_run_seconds":elapsed,
                        "workspace_bytes":p.layout.workspace_bytes,
                        "initial_unchanged":unchanged_initial(initial,original),
                        "output":errors(output,expected),"final":errors(final,expected_state)}
                if case == "zero":
                    record["exact_zero_output_and_final"] = bool((output==0).all() and (final==0).all())
                progress["current_record"]=record
                if rows in (128,129) and case == "nonzero_initial":
                    # Separate-final continuation through a non-chunk-aligned
                    # boundary. Record numerical differences, never relabel
                    # tolerance-free differences as equivalent.
                    progress["current"]["stage"]="continuation"
                    middle=torch.empty_like(initial)
                    second=torch.empty_like(initial)
                    parts=[]
                    for start,end,state_in,state_out in ((0,65,initial,middle),(65,rows,middle,second)):
                        count=end-start
                        fragment=torch.empty((1,count,6,128),device=args.device,dtype=torch.float16)
                        part=prepare(count,args.device,workspace_budget_bytes=layout(count).workspace_bytes,compiler=compiler)
                        part_bound=part.bind(q[:,start:end].contiguous(),k[:,start:end].contiguous(),
                                  v[:,start:end].contiguous(),g[:,start:end].contiguous(),
                                  b[:,start:end].contiguous(),state_in,fragment,state_out)
                        part_bound.run()
                        # Keep launch/workspace/operand owners live until GPU
                        # completion, then transfer only output/state forward.
                        torch.cuda.synchronize()
                        parts.append(fragment)
                    split_output=torch.cat(parts,dim=1)
                    record["continuation"]={"boundary":65,"part_rows":[65,rows-65],
                        "initial_unchanged":unchanged_initial(initial,original),
                        "output_vs_serial":errors(split_output,expected),
                        "final_vs_serial":errors(second,expected_state),
                        "output_vs_unsplit":errors(split_output,output),
                        "final_vs_unsplit":errors(second,final)}
                results.append(record)
                # Alias refusal precedes dispatch; capture is unsupported.
                progress["current"]["stage"]="alias-negative-control"
                try:
                    p.bind(q,k,v,g,b,initial,output,initial)
                except ValueError:
                    record["initial_alias_refused"] = True
                else:
                    record["initial_alias_refused"] = False
                record["initial_unchanged"]=record["initial_unchanged"] and unchanged_initial(initial,original)
                progress.pop("current_record",None)
                progress["current"]["stage"]="complete"
    if args.phase=="full":
        progress["main_numerical_cases"]="completed"
    stream_result=nondefault_stream_trial(args,compiler,progress)
    return {"schema":"tensorfold.gdn-chunk64-native-numerical-trial-v1",
            "phase":args.phase,"main_numerical_cases":progress["main_numerical_cases"],
            "quality_acceptance":False,"performance_qualified":False,
            "reason":"Errors reported without invented tolerance; frontend/model/state-cut integration unqualified",
            "compiler":compiler.inspect(),"device":torch.cuda.get_device_name(args.device),
            "nondefault_stream":stream_result,"torch":torch.__version__,"seed":20261113,"source_inputs":progress["source_inputs"],"results":results}


def normalize_attestation(argv):
    """One shared structural schema, fed only actual verified producer DATA."""
    from tensorfold.cuda.sm70.gdn_chunk64_artifact import publish, read_attestation
    parser=argparse.ArgumentParser(description="Normalize owned artifact DATA; no artifact or GPU admission")
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--input-bytes",type=int,required=True)
    parser.add_argument("--input-sha256",required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(argv)
    value=read_attestation(args.input,args.input_bytes,args.input_sha256)
    if value["evidence"]["producer"]["sha256"]!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError("G3 attestation must name this exact maintained normalization producer")
    result=publish(args.output,value)
    print(json.dumps(result,sort_keys=True))
    return result


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=="attest-owned":
        return normalize_attestation(argv[1:])
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda-home",required=True)
    p.add_argument("--nvcc-sha256",required=True)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--phase",choices=("full","nondefault-stream","spans-state"),default="full",
                   help="full runs all18 numerical cases and both splits; nondefault-stream runs only the stream control; spans-state runs the separate finite State diagnostics; legacy main cases remain unrun in both")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--stream-trace-output",type=Path,required=True)
    p.add_argument("--owned-tilelang-attestation",type=Path,
                   help="Explicit immutable postbuild attestation; unattested TileLang is refused before import")
    p.add_argument("--owned-attestation-bytes",type=int,help="Exact attestation metadata byte credit (at most16MiB)")
    p.add_argument("--owned-attestation-sha256",help="Externally selected actual attestation SHA256")
    p.add_argument("--owned-tilelang-wheel",type=Path,help="Actual retained owned wheel whose bytes must match")
    p.add_argument("--owned-hash-byte-credit",type=int,
                   help="SHA256 byte credit per inspection: wheels, installed members and v2 archive member payloads")
    p.add_argument("--owned-native-base-wheel",type=Path,
                   help="Retained native base wheel, required only by closed Python-derivation attestation v2")
    args=p.parse_args(argv)
    selections=(args.owned_tilelang_attestation,args.owned_attestation_bytes,args.owned_attestation_sha256,
                args.owned_tilelang_wheel,args.owned_hash_byte_credit)
    if any(value is not None for value in selections) and any(value is None for value in selections):
        p.error("owned TileLang admission requires all five explicit artifact/credit fields")
    if args.owned_native_base_wheel is not None and args.owned_tilelang_attestation is None:
        p.error("a native base wheel requires the complete explicit owned artifact selection")
    if args.stream_trace_output.resolve()==args.output.resolve() or args.stream_trace_output.exists():
        p.error("stream trace requires its own unused output path")
    if args.output.exists():
        p.error("output already exists; preserve historical trials")
    progress={"results":[],"current":{"stage":"cold-import-and-admission"}}
    try:
        result=run(args, progress)
    except Exception as exc:
        result={"schema":SPAN_STATE_SCHEMA if args.phase=="spans-state" else "tensorfold.gdn-chunk64-native-numerical-trial-v1",
                "phase":args.phase,"main_numerical_cases":progress.get("main_numerical_cases","unrun"),
                "quality_acceptance":False,"performance_qualified":False,
                "failure":{"type":type(exc).__name__,"message":str(exc)},
                "progress":progress}
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        raise
    result["tool_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    if args.phase == "spans-state":
        validate_spans_state_result(result)
        return
    if not result["nondefault_stream"]["actual_single_nondefault_stream_proof"] or not result["nondefault_stream"]["producer_core_consumer_exact_zero"]:
        raise SystemExit("G3 nondefault stream proof failed; preserve report, do not adopt")
    if args.phase=="nondefault-stream":
        if result["results"] or result["main_numerical_cases"]!="unrun":
            raise SystemExit("G3 stream-only report must not claim main numerical cases")
        return
    expected={(rows,case) for rows in (1,63,64,65,128,129)
              for case in ("zero","nonzero_initial","raw_underflow")}
    if (len(result["results"])!=len(expected)
            or {(r["rows"],r["case"]) for r in result["results"]}!=expected
            or result["main_numerical_cases"]!="completed"):
        raise SystemExit("G3 full trial requires all18 main numerical cases")
    if any(not r["initial_unchanged"] or not r["initial_alias_refused"]
           or not r["output"]["finite"] or not r["final"]["finite"]
           or not r.get("exact_zero_output_and_final",True)
           or ("continuation" in r and (not r["continuation"]["initial_unchanged"]
               or not r["continuation"]["output_vs_serial"]["finite"]
               or not r["continuation"]["final_vs_serial"]["finite"]))
           for r in result["results"]):
        raise SystemExit("G3 numerical/state trial has failures; preserve report, do not adopt")


if __name__ == "__main__":
    main()
