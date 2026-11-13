"""Explicit G3 numerical-trial component, with no serving selection or frontend.

Inputs are normalized, unscaled FP16 q/k, FP16 v, FP32 raw log gates
and beta. The caller owns the frontend's numerical contract. State is FP32
VLK [1,6,128,128]; initial and final must be distinct. Only one sequence,
Hq2/Hv6/D128, full-prefix retention and 128-thread SM70 kernels are admitted.
This component neither updates conv history nor publishes a State bank.

Cold preparation validates and allocates once, compiles before binding, and
retains all operands. Bound invocation is serial on the preparation CUDA stream;
rebind after any tensor replacement. No concurrent reuse, resize or storage
mutation is permitted until the stream has completed. CUDA failure poisons the
trial's outputs; the caller must contain the run without publishing its State.
"""

import importlib.metadata
import shlex
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

from .gdn_chunk64_artifact import OwnedArtifact


@dataclass(frozen=True)
class Layout:
    rows: int
    chunks: int
    workspace_bytes: int


def layout(rows, *, cuts=()):
    """Exact component-owned tensor payload, excluding inputs/output/state."""
    if type(rows) is not int or not 1 <= rows <= 2**31 - 1:
        raise ValueError("G3 rows must be a positive int32")
    if type(cuts) is not tuple or cuts:
        raise ValueError("G3 does not yet support retained interior cuts")
    # A Half[1,R,6,64], cumulative gates Float[1,R,6], int32 cu/offsets,
    # int32 single-sequence dummy, bool has-initial; h has zero elements.
    return Layout(rows, (rows + 63) // 64, rows * (6 * 64 * 2 + 6 * 4) + 21)


@dataclass(frozen=True)
class Compiler:
    """A cold-process compiler identity; never rewrites TileLang callbacks/env.

    Use a dedicated component process with CUDA_HOME and PATH already selecting
    this SDK before importing TileLang. Supplying an identity is not admission:
    inspect the dependency report and actual generated SM70 code first.
    """
    cuda_home: str
    nvcc_sha256: str
    owned_artifact: OwnedArtifact | None = None

    def _nvcc(self):
        import hashlib
        import os
        home = Path(self.cuda_home).resolve(strict=True)
        nvcc = home / "bin/nvcc"
        if Path(os.environ.get("CUDA_HOME", "")).resolve() != home:
            raise RuntimeError("G3 CUDA_HOME must select the protected SDK before import")
        if shutil.which("nvcc") is None or Path(shutil.which("nvcc")).resolve() != nvcc.resolve():
            raise RuntimeError("G3 PATH must select the same protected nvcc")
        if hashlib.sha256(nvcc.read_bytes()).hexdigest() != self.nvcc_sha256:
            raise RuntimeError("G3 protected nvcc identity mismatch")
        version = subprocess.run([str(nvcc), "--version"], check=True, capture_output=True,
                                 text=True, timeout=15).stdout
        if "release 12.9," not in version:
            raise RuntimeError("G3 initial trial requires protected CUDA12.9")
        return {"cuda_home": str(home), "nvcc_sha256": self.nvcc_sha256, "nvcc_version": version}

    @staticmethod
    def _headers():
        import os

        from .foundation_headers import selected_headers
        selected = selected_headers()
        if selected is None:
            raise RuntimeError("G3 requires the protected selected-header manifest")
        headers = selected.record()
        roots = [Path(root) for root in headers["include_roots"]]
        cutlass = [root for root in roots if (root / "cutlass/cutlass.h").is_file()]
        cccl = [root for root in roots if (root / "cuda/std/type_traits").is_file()]
        if len(cutlass) != 1 or len(cccl) != 1:
            raise RuntimeError("G3 requires unambiguous selected CUTLASS/CCCL include roots")
        configured = os.environ.get("TL_CUTLASS_PATH")
        if not configured or Path(configured).resolve() != cutlass[0].resolve():
            raise RuntimeError("G3 TL_CUTLASS_PATH must select protected CUTLASS before TileLang import")
        return headers, str(cutlass[0]), str(cccl[0])

    def environment(self):
        """Metadata-only compiler/header descriptor for the attestation producer."""
        nvcc = self._nvcc()
        headers, _, _ = self._headers()
        return {key: nvcc[key] for key in ("cuda_home", "nvcc_sha256")} | {"headers": headers}

    def inspect(self):
        nvcc = self._nvcc()
        packages = {name: importlib.metadata.version(name)
                    for name in ("tilelang", "apache-tvm-ffi", "torch", "triton")}
        if packages["triton"] != "3.2.0":
            raise RuntimeError("G3 requires retained Triton3.2.0")
        if packages["apache-tvm-ffi"] != "0.1.16":
            raise RuntimeError("G3 scoped stream API requires verified apache-tvm-ffi==0.1.16")
        # The unpatched public0.1.10/FFI0.1.16 combination actually aborted at
        # import. No version string, including that historical default, proves
        # patched ownership. Exact externally selected artifact evidence must
        # precede every TileLang import; unknown/unattested builds stay closed.
        if self.owned_artifact is None:
            raise RuntimeError("G3 TileLang requires an explicit owned artifact attestation; "
                               "unpatched TileLang0.1.10/FFI0.1.16 is not admitted")
        if type(self.owned_artifact) is not OwnedArtifact:
            raise TypeError("G3 requires its closed owned artifact identity")
        headers, cutlass, cccl = self._headers()
        environment = {key: nvcc[key] for key in ("cuda_home", "nvcc_sha256")} | {"headers": headers}
        owned = self.owned_artifact.inspect(environment, packages)
        from packaging.requirements import Requirement
        requirements = []
        for text in importlib.metadata.requires("tilelang") or ():
            requirement = Requirement(text)
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            installed = importlib.metadata.version(requirement.name)
            if not requirement.specifier.contains(installed):
                raise RuntimeError(f"G3 TileLang base requirement is unsatisfied: {text}; installed {installed}")
            requirements.append({"requirement": text, "installed": installed})
        import sys
        if any(name == "flash_qla" or name.startswith("flash_qla.") for name in sys.modules):
            raise RuntimeError("G3 dedicated process must not import the upstream compiler override")
        return {"tilelang_base_requirements": requirements, "packages": packages,
                **nvcc, "headers": headers, "owned_artifact": owned,
                "cutlass_root": cutlass, "cccl_root": cccl,
                "target": {"kind": "cuda", "arch": "sm_70"}, "execution_backend": "tvm_ffi"}


class Prepared:
    """Cold allocated/compiled single-stream component. Construct with prepare()."""

    def bind(self, q, k, v, raw_log_gate, beta, initial, output, final):
        import torch
        if torch.cuda.current_device() != self.device.index:
            raise RuntimeError("G3 requires its preparation CUDA device to be current")
        if threading.current_thread() is not self.owner:
            raise RuntimeError("G3 requires its original CPU owner thread")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("G3 binding must precede capture")
        if torch.cuda.current_stream(self.device).cuda_stream != self.stream:
            raise RuntimeError("G3 bind must use its preparation stream")
        r = self.layout.rows
        specs = [(q, (1,r,2,128), torch.float16), (k, (1,r,2,128), torch.float16),
                 (v, (1,r,6,128), torch.float16), (raw_log_gate, (1,r,6), torch.float32),
                 (beta, (1,r,6), torch.float32), (initial, (1,6,128,128), torch.float32),
                 (output, (1,r,6,128), torch.float16), (final, (1,6,128,128), torch.float32)]
        for tensor, shape, dtype in specs:
            if (type(tensor) is not torch.Tensor or tensor.shape != shape or tensor.dtype != dtype
                    or tensor.device != self.device or not tensor.is_contiguous()
                    or tensor.requires_grad or tensor.is_neg() or tensor.is_conj()):
                raise ValueError("G3 requires contiguous, untransformed, no-grad operands of exact shape/device/dtype")
        # All writes are disjoint from every other operand. Read-read aliasing is
        # harmless, but outputs/state/scratch can never overwrite an input.
        reads = [q,k,v,raw_log_gate,beta,initial]
        writes = [output,final,self.a,self.gsum,*self.metadata]
        for i, tensor in enumerate(writes):
            lo, hi = _span(tensor)
            if any(lo < _span(other)[1] and _span(other)[0] < hi
                   for other in reads + writes[:i] if other.numel()):
                raise ValueError("G3 writes must not alias inputs, another write or initial state")
        return Bound(self, (q,k,v,raw_log_gate,beta,initial,output,final))


def _span(tensor):
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


@dataclass(frozen=True)
class Bound:
    prepared: Prepared
    operands: tuple

    def run(self):
        """Enqueue on the original stream; caller waits before host use/retirement.

        Tensor shapes/storage/ownership must remain as cold-bound. Input values
        may change after preceding work on this stream. raw_log_gate is the raw
        finite nonpositive producer value, never log(saved_exp_gate); beta is
        the producer's FP32 representation. No checks synchronize CUDA here.
        """
        import torch
        p = self.prepared
        if torch.cuda.current_device() != p.device.index:
            raise RuntimeError("G3 requires its preparation CUDA device to be current")
        if threading.current_thread() is not p.owner:
            raise RuntimeError("G3 requires its original CPU owner thread")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("G3 initial numerical trial does not admit capture")
        if not torch.is_inference_mode_enabled():
            raise RuntimeError("G3 is an inference-only numerical trial")
        if torch.cuda.current_stream(p.device).cuda_stream != p.stream:
            raise RuntimeError("G3 cannot execute on a different stream")
        q,k,v,g,b,initial,output,final = self.operands
        cu,offsets,dummy,has_initial = p.metadata
        # TVM-FFI launches consume TVMFFIEnvGetStream. Bind that supported
        # thread-local context explicitly; the pinned TileLang adapter's prose
        # about a Torch stream thunk does not implement such a thunk. The FFI
        # context restores the caller's prior stream even if a launch raises.
        with p.stream_context(p.ffi_device, p.stream):
            p.cumsum(g, p.gsum, p.layout.chunks)
            p.kkt(k, b, p.a, p.layout.chunks)
            p.core(q,k,v,p.a,p.gsum,b,initial,cu,offsets,dummy,cu,dummy,has_initial,output,p.h,final)
        torch.autograd.graph.increment_version((output, final, p.gsum, p.a))
        return output, final


def prepare(rows, device, *, workspace_budget_bytes, compiler, cuts=()):
    """Cold only: admit SDK, allocate finite scratch and compile three kernels.

    No hidden global cache or fallback. Caller supplies a budget in addition to
    ordinary model admission. Runtime allocator/compiler retention is separate
    from tensor payload and must be measured/contained in the trial process.
    No kernel runs here. Failure releases local tensors through normal Python
    ownership; successful return retains all scratch and compiled launchers.
    """
    plan = layout(rows, cuts=cuts)
    if type(workspace_budget_bytes) is not int or workspace_budget_bytes < plan.workspace_bytes:
        raise ValueError("G3 workspace lacks explicit tensor-payload credit")
    if type(compiler) is not Compiler:
        raise TypeError("G3 requires a protected compiler identity")
    report = compiler.inspect()
    import torch
    device = torch.device(device)
    if device.type != "cuda" or torch.cuda.is_current_stream_capturing():
        raise RuntimeError("G3 prepare requires an uncaptured CUDA device")
    device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
    if torch.cuda.get_device_capability(device) != (7,0):
        raise RuntimeError("G3 initial candidate is SM70 only")
    with torch.cuda.device(device):
        # This module has no import-time CUDA query or global compiler override.
        import tilelang
        import tvm_ffi
        from tilelang.env import CUDA_HOME, CUTLASS_INCLUDE_DIR
        from tilelang.env import env as tilelang_environment
        if (Path(CUTLASS_INCLUDE_DIR).resolve() != Path(report["cutlass_root"]).resolve()
                or Path(CUDA_HOME).resolve() != Path(report["cuda_home"]).resolve()
                or Path(tilelang_environment.TILELANG_CACHE_DIR) != Path(report["owned_artifact"]["cache_directory"])
                or Path(tilelang_environment.TILELANG_TMP_DIR) != Path(report["owned_artifact"]["temporary_directory"])):
            raise RuntimeError("G3 TileLang imported a different compiler/header binding")
        from .gdn_chunk64_artifact import loaded_compiler
        report["owned_artifact"]["loaded_compiler"] = loaded_compiler(report["owned_artifact"])
        from . import gdn_chunk64_kernels as kernels

        def compile_kernel(generator, *args, **kwargs):
            # Public 0.1.10 API, explicit target/backend; no JITImpl mutation or
            # callback replacement. TVM-FFI's default NVCC callback places this
            # selected CCCL include ahead of implicit CUDA toolkit includes.
            return tilelang.compile(generator.get_tir(*args, **kwargs),
                                    target=report["target"], execution_backend="tvm_ffi",
                                    pass_configs=generator.pass_configs,
                                    compile_flags=["-I" + shlex.quote(report["cccl_root"])])

        p = Prepared()
        p.layout, p.device, p.compiler_report = plan, device, report
        p.stream = torch.cuda.current_stream(device).cuda_stream
        p.owner = threading.current_thread()
        p.ffi_device = tvm_ffi.device(str(device))
        p.stream_context = tvm_ffi.use_raw_stream
        p.a = torch.empty((1,rows,6,64), dtype=torch.float16, device=device)
        p.gsum = torch.empty((1,rows,6), dtype=torch.float32, device=device)
        p.h = torch.empty((1,0,6,128,128), dtype=torch.float16, device=device)
        p.metadata = (torch.tensor([0,rows],dtype=torch.int32,device=device),
                      torch.tensor([0,plan.chunks],dtype=torch.int32,device=device),
                      torch.zeros(1,dtype=torch.int32,device=device),
                      torch.ones(1,dtype=torch.bool,device=device))
        p.cumsum = compile_kernel(kernels.tilelang_chunk_local_cumsum,6,64,"float32","float32","int32",False,False)
        p.kkt = compile_kernel(kernels.tilelang_kkt_solve_sm70,6,2,128,64,"float32","float16","float32","int32",False)
        p.core = compile_kernel(kernels.tilelang_fused_chunk_gdr_fwd_sm70,
            H=6,Hg=2,DK=128,DV=128,chunk_size=64,scale=128**-0.5,
            accum_dtype="float32",qkva_dtype="float16",g_dtype="float32",b_dtype="float32",
            h0_dtype="float32",ht_dtype="float32",h_dtype="float16",o_dtype="float16",
            seqlen_dtype="int32",use_initial_state=True,store_final_state=True,
            store_h=False,store_o=True,is_varlen=False,is_cp=False,state_layout_vlk=True,
            indexed_state=False,block_DV=32,threads=128)
        return p
