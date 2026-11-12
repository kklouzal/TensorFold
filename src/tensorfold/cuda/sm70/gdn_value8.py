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
MAX_BATCH_STATES, MAX_BATCH_WINDOW = 24, 5


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


def batch_layout(rows, participants, width):
    """Stable launch bounds for every ragged segmentation of this exact key."""
    if (any(type(value) is not int for value in (rows, participants, width))
            or not 1 <= participants <= MAX_BATCH_STATES or not 1 <= width <= MAX_BATCH_WINDOW
            or not participants <= rows <= participants*width):
        raise ValueError('value8 batch requires nonempty bounded 24-State/five-row windows')
    bound = min(width, rows-participants+1)
    return {'rows': rows, 'participants': participants, 'max_window_rows': bound,
            'preload_rows': next(value for value in (1, 2, 4, 5) if bound <= value),
            'minimum_active_q_bytes': rows*1024, 'q_bytes_per_capacity_row': 1024,
            'front_grid': [2, bound, participants], 'core_grid': [16, 6, participants],
            'back_grid': [6, bound, participants]}


def _span(tensor):
    if not tensor.numel():
        return None
    offsets = [(size-1)*stride for size, stride in zip(tensor.shape, tensor.stride())]
    return (tensor.data_ptr()+sum(min(0, offset) for offset in offsets)*tensor.element_size(),
            tensor.data_ptr()+(1+sum(max(0, offset) for offset in offsets))*tensor.element_size())


@dataclass(frozen=True)
class BatchPrepared:
    """One exact row/count key, borrowing the original registered table owner.

    No cold Frame/State subset or parity is retained. The actor must validate a
    freshly published NativeTables frame before each submission/replay; side
    capture uses these fixed GPU views without an actor-stream check. All
    descriptor/pointee/Q/output storage outlives positive consumer retirement.
    Tensor mutation counters follow the incumbent raw-pointer batch ABI; this
    forward-only component does not promise per-replay Python version updates.
    """
    prepared: object
    static: object
    buffers: object
    layer: int
    projection: object
    output: object
    sums: object
    workspace: Workspace
    pointers: object
    starts: object
    rows: int
    participants: int
    table_width: int
    max_window_rows: int
    inference_storage: bool
    bound_addresses: tuple

    def validate_frame(self, frame, *, segs=None):
        """Actor-side metadata only: no upload, allocation or device readback."""
        from tensorfold.families.qwen4_exp.cuda.gdn_multi import NativeTables

        if (type(frame) is not NativeTables or frame.static is not self.static
                or frame.buffers is not self.buffers or frame.rows != self.rows
                or len(frame.segs) != self.participants or frame.static.width != self.table_width):
            raise ValueError('value8 batch replay differs from its registered exact geometry')
        frame.validate_current(frame.segs if segs is None else segs, self.buffers)
        b = self.buffers
        if (frame.pointers.stride(0) != self.pointers.stride(0)
                or frame.pointers.data_ptr()+self.layer*frame.pointers.stride(1)*frame.pointers.element_size() != self.bound_addresses[4]
                or frame.starts.data_ptr() != self.bound_addresses[5]
                or b.proj.data_ptr()+self.layer*b.proj.stride(0)*b.proj.element_size() != self.bound_addresses[0]
                or b.gout.data_ptr() != self.bound_addresses[1] or b.gxs.data_ptr() != self.bound_addresses[2]
                or self.workspace.q.data_ptr() != self.bound_addresses[3]):
            raise ValueError('value8 batch fixed descriptor/output storage changed')

    def chain_prepared(self, weight, parameter, bias, norm, eps, *, negative_decay=False):
        """Launch unchanged arithmetic on current stream, including capture.

        No device allocation, host readback or descriptor publication occurs.
        Unlike validate_frame, this submission may run on the capture stream.
        """
        import torch

        if type(negative_decay) is not bool:
            raise ValueError('value8 batch decay mode must be explicit boolean')
        if any(type(t) is not torch.Tensor or t.device != self.prepared.device or t.requires_grad
               or t.is_neg() or t.is_conj() for t in (weight, parameter, bias, norm)):
            raise TypeError('value8 batch weights require plain same-device forward-only operands')
        if self.inference_storage and not torch.is_inference_mode_enabled():
            raise RuntimeError('inference output mutation outside inference mode')
        self.prepared.module.batch_chain(self.projection, weight, parameter, bias, norm, eps,
            self.pointers, self.starts, self.max_window_rows, self.output, self.sums, self.workspace.q, negative_decay)
        return self.output


@dataclass(frozen=True)
class Prepared:
    module: object
    device: object
    batch_ready: bool = False

    def batch_prepare(self, frame, layer, projection, output, sums, workspace):
        """Cold-bind existing NativeTables views after paid workspace admission.

        prepare(..., batch=True) resolves batch kernel residency before State
        capacity admission. This method only validates/retains existing owners;
        it neither publishes pointers nor allocates Q or other device storage.
        """
        import torch
        from tensorfold.families.qwen4_exp.cuda.gdn_multi import NativeTables

        if not self.batch_ready:
            raise RuntimeError('value8 batch kernels must be resident before admission')
        if type(frame) is not NativeTables or type(layer) is not int:
            raise TypeError('value8 batch preparation requires its current registered NativeTables frame')
        frame.validate_current(frame.segs, frame.buffers)
        facts = batch_layout(frame.rows, len(frame.segs), frame.static.width)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('value8 batch binding belongs outside capture')
        if (type(workspace) is not Workspace or type(workspace.capacity) is not int
                or not frame.rows <= workspace.capacity <= MAX_ROWS):
            raise ValueError('value8 batch requires its funded Q workspace capacity')
        tensors = (projection, output, sums, workspace.q)
        if any(type(t) is not torch.Tensor or t.device != self.device or not t.is_contiguous() or t.requires_grad
               or t.is_neg() or t.is_conj() for t in tensors):
            raise TypeError('value8 batch requires plain same-device forward-only operands')
        if (projection.shape != (frame.rows, 2060) or projection.dtype != torch.float16
                or output.shape != (frame.rows, 768) or output.dtype != torch.float16
                or sums.shape != (frame.rows, 24) or sums.dtype != torch.float32
                or workspace.q.shape != (workspace.capacity, 2, 128) or workspace.q.dtype != torch.float32):
            raise ValueError('value8 batch differs from H2/HV6 output or Q workspace geometry')
        pointers, starts = frame.operands(layer, projection, output, sums)
        # Q is an extra exclusive allocation. Protect every registered State's
        # replay history, all layers' pointer words and the decoder's existing
        # direct tensor owners, not just the initially selected State subset.
        owners = [frame.static.i64, frame.static.i32]
        owners.extend(t for t in vars(frame.buffers).values() if type(t) is torch.Tensor and t is not workspace.q)
        writes = [output, sums, workspace.q]
        for record in frame.static.native_states:
            owners.extend((record[1], record[2]))
            writes.append(record[2])
            for saved in record[4]:
                owners.extend(t for t in saved[1:] if t is not None)
                writes.extend(saved[1:5])
        qspan = _span(workspace.q)
        for tensor in owners:
            span = _span(tensor) if tensor.device == self.device else None
            if span is not None and qspan[0] < span[1] and span[0] < qspan[1]:
                raise ValueError('value8 funded Q workspace overlaps registered inference storage')
        return BatchPrepared(self, frame.static, frame.buffers, layer, projection, output, sums, workspace,
                             pointers, starts, frame.rows, len(frame.segs), frame.static.width, facts['max_window_rows'],
                             any(t.is_inference() for t in writes),
                             (projection.data_ptr(), output.data_ptr(), sums.data_ptr(), workspace.q.data_ptr(),
                              pointers.data_ptr(), starts.data_ptr()))

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


def prepare(device, *, batch=False):
    """Resolve this private native component before caller admission/capture."""
    import torch
    from tensorfold.cuda import build
    from .foundation_headers import selected_headers
    from .source_cache import stage_sources

    if type(batch) is not bool:
        raise ValueError('value8 batch preparation mode must be explicit boolean')
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
        if batch:
            module.prepare_batch()
    return Prepared(module, device, batch)
