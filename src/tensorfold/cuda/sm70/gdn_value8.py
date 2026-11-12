"""Unselected TF-exact H2/HV6 GDN component with eight value rows per CTA.

Only an explicit ``prepare`` selects this component. Incumbent dispatch is
untouched. Caller owns every tensor and the funded Q workspace exclusively on
the current stream through completion and capture/replay retirement. There is
no state cache, hidden allocation during execution, or cross-stream handoff.
"""
from dataclasses import dataclass
import hashlib
from pathlib import Path

MAX_ROWS = 65535  # The preprocessing/output kernels use CUDA grid dimension y.


def layout(rows):
    if type(rows) is not int or not 0 <= rows <= MAX_ROWS:
        raise ValueError("value8 component requires a bounded R0..65535 frame")
    return {"rows": rows, "key_heads": 2, "value_heads": 6, "head_dim": 128,
            "projection_width": 2060, "q_shape": (rows, 2, 128),
            "additional_device_bytes": rows * 2 * 128 * 4,
            "core_threads": 64, "value_rows_per_cta": 8,
            "core_ctas_per_sequence": 96,
            "preload_rows": next((n for n in (1, 2, 4, 5, 8) if 0 < rows <= n), 0),
            "raw_output": "existing caller Half output reused before gated-norm epilogue",
            "qualification": "private-unqualified"}


@dataclass(frozen=True)
class Workspace:
    q: object
    capacity: int


def allocate_workspace(rows, device, *, device_budget_bytes):
    facts = layout(rows)
    if type(device_budget_bytes) is not int or device_budget_bytes < facts["additional_device_bytes"]:
        raise MemoryError("explicit device credit cannot contain the value8 Q workspace")
    import torch

    return Workspace(torch.empty(facts["q_shape"], dtype=torch.float32, device=device), rows)


@dataclass(frozen=True)
class Prepared:
    module: object
    device: object

    def chain(self, projection, conv, weight, initial, parameter, bias, norm, eps, rows,
              scratch, final, output, sums, workspace, *, negative_decay=False):
        layout(rows)
        if type(negative_decay) is not bool or type(workspace) is not Workspace or workspace.capacity < rows:
            raise ValueError("value8 requires its declared decay mode and sufficient caller workspace")
        import torch

        keep = getattr(scratch, "keep_rows", None)
        tensors = (projection, conv, weight, initial, parameter, bias, norm, final, output, sums,
                   scratch.k, scratch.v, scratch.g, scratch.b, workspace.q)
        if keep is not None:
            tensors += (keep,)
        if any(type(t) is not torch.Tensor or t.device != self.device or t.requires_grad
               or t.is_neg() or t.is_conj() for t in tensors):
            raise TypeError("value8 requires plain same-device forward-only operands")
        writes = (final, output, sums, scratch.k, scratch.v, scratch.g, scratch.b, workspace.q)
        if not torch.is_inference_mode_enabled() and any(t.is_inference() for t in writes):
            raise RuntimeError("inference output mutation outside inference mode")
        self.module.chain(projection, conv, weight, initial, parameter, bias, norm, eps, rows,
                          output, sums, final, scratch.k, scratch.v, scratch.g, scratch.b,
                          workspace.q, keep, negative_decay)
        torch.autograd.graph.increment_version(writes if rows else (final,))
        return output

    def replay(self, initial, scratch, rows, final):
        layout(rows)
        import torch

        if any(type(t) is not torch.Tensor or t.device != self.device or t.requires_grad
               or t.is_neg() or t.is_conj() for t in (initial, scratch.k, scratch.v, scratch.g, scratch.b, final)):
            raise TypeError("value8 replay requires plain same-device forward-only operands")
        if final.is_inference() and not torch.is_inference_mode_enabled():
            raise RuntimeError("inference output mutation outside inference mode")
        self.module.replay(initial, scratch.k, scratch.v, scratch.g, scratch.b, rows, final)
        torch.autograd.graph.increment_version(final)
        return final


def prepare(device):
    """Resolve this private native component before caller admission/capture."""
    import torch
    from tensorfold.cuda import build
    from .foundation_headers import selected_headers
    from .source_cache import stage_sources

    device = torch.device(device)
    with torch.cuda.device(device):
        device = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.is_current_stream_capturing() or torch.cuda.get_device_capability(device) != (7, 0):
            raise ValueError("value8 preparation requires uncaptured actual SM70")
        selected_headers()
        here = Path(__file__).parent
        sources = [here / "gdn_value8.cpp", here / "gdn_value8.cu"]
        headers = [here.parent / "kernels/tensor_contracts.cuh"]
        digest = hashlib.sha256()
        for path in [*sources, *headers]:
            digest.update(path.name.encode() + b"\0" + path.read_bytes())
        staged = stage_sources(sources, headers=headers, source_root=here.parent)
        module = build.load("tf_sm70_gdn_value8_" + digest.hexdigest()[:20], staged,
                            need=(7, 0), extra_cflags=["-O3"], extra_cuda_cflags=["-O3", "--fmad=false"])
        module.prepare()
    return Prepared(module, device)
