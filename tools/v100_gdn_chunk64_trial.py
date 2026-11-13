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


def run(args, progress):
    progress["phase"]=args.phase
    progress["main_numerical_cases"]="unrun" if args.phase=="nondefault-stream" else "incomplete"
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
    p.add_argument("--phase",choices=("full","nondefault-stream"),default="full",
                   help="full runs all18 numerical cases and both splits; nondefault-stream runs only the stream control, with main cases explicitly unrun")
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
        result={"schema":"tensorfold.gdn-chunk64-native-numerical-trial-v1",
                "phase":args.phase,"main_numerical_cases":progress.get("main_numerical_cases","unrun"),
                "quality_acceptance":False,"performance_qualified":False,
                "failure":{"type":type(exc).__name__,"message":str(exc)},
                "progress":progress}
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        raise
    result["tool_sha256"]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
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
