"""Private target-only numerical owner; all authority follows the TP cohort veto.

Two serial roles share retained weight bytes but never scratch. A status word
is never cleared in a frame. Containment replaces exceptional floats only in a
poisoned frame, which can never commit or expose output. It is not recovery.
"""
from contextlib import contextmanager
import sys

import torch

from tensorfold.cuda.sm70.dp4a_experts import CONTRACT, eligible, extension, _signature

from .prefix_policy import DP4A_PROFILE as PROFILE
POLICY = CONTRACT + ":target-only:selected-blocks2w-shared-v1:finite-containment-terminal-fatal-v1:cohort2words-v1"
ROLE_BYTES = 412672
# Explicit additional construction/retention operating commitment: fixed48-bank
# catalogue, two192-view tables, retained native/Tensor metadata and diagnostics.
# It is charged before floor admission; observed cold RSS/HWM must fit it.
HOST_BYTES = 3 << 20
MAX_TENSOR_HANDLES = 558  #12 buffers+384 scratchviews+66 sharedpickviews+96 outputviews.
NATIVE_HANDLE_BYTES = 2048
METADATA_TEMP_BYTES = 1 << 20
MAX_METADATA_OBJECTS = 8192


def funding(slots=24, world=8, *, feedback_bytes=0):
    if type(slots) is not int or slots != 24 or world != 8 or type(feedback_bytes) is not int or feedback_bytes < 0:
        raise ValueError("DP4A target owner requires admitted TP8/native24 geometry")
    def granule(n):
        return (n + 511) // 512 * 512
    packet = granule(8 * slots + 8) - granule(8 * slots)
    packet += granule(8 * slots * world + feedback_bytes + 8 * world) - granule(8 * slots * world + feedback_bytes)
    return {"roles_device_bytes": 2 * ROLE_BYTES, "status_device_delta_bytes": packet,
            "host_commitment_bytes": HOST_BYTES + 8 * world,
            "raw_status_host_delta_bytes": 8 * world, "policy": POLICY}


def _layout(bank):
    return (bank.count, bank.width, bank.dims, bank.implementation, bank.limit, bank.swiglu,
            *((p.ggml_type, p.n, p.k, p.column_offset) for p in (bank.gate, bank.up, bank.down)),
            tuple((p.n, p.k, p.implementation) for p in bank.shared))


class Role:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        device = owner.weights.device
        self.qx = torch.empty((32, 80, 36), dtype=torch.uint8, device=device)
        self.qh = torch.empty((32 * 11, 3, 36), dtype=torch.uint8, device=device)
        self.gate = torch.empty((32, 11, 80), dtype=torch.float32, device=device)
        self.up = torch.empty_like(self.gate)
        self.hidden = torch.empty((32, 11, 80), dtype=torch.float16, device=device)
        self.status = torch.zeros(1, dtype=torch.int32, device=device)
        # No shape cache or new internal tensor view is needed in a hot call.
        self.views = tuple((self.qx[:r], self.qh[:r * 11], self.gate[:r], self.up[:r],
                            self.hidden[:r], self.hidden[:r].view(-1, 80)) for r in range(1, 33))
        self.shared_picks = self.gate.view(torch.int32).view(-1)[:32].view(32, 1)
        self.shared_pick_views = tuple(self.shared_picks[:r] for r in range(1, 33))
        self.outputs = {}  # At most canonical and bound target output owners; cold only.
        self.original_views = self.views
        self.buffers = (self.qx, self.qh, self.gate, self.up, self.hidden, self.status)
        self.signatures = tuple(_signature(value) for value in self.buffers)
        self.calls = self.exact_calls = 0
        self.counter_overflowed = False

    def bind_output(self, output):
        self.owner.check_alive()
        if (torch.cuda.is_current_stream_capturing() or output.ndim != 3 or output.shape[1:] != (11, 2560)
                or output.dtype not in (torch.float16, torch.float32) or output.requires_grad):
            raise ValueError("DP4A output binding must be cold canonical11-slot storage")
        key = (output.untyped_storage().data_ptr(), output.data_ptr())
        if key not in self.outputs:
            if len(self.outputs) == (1 if self.name == "main" else 2) or not 1 <= output.shape[0] <= 32 or not output.is_contiguous():
                raise ValueError("DP4A output binding exceeds its two fixed buffer families")
            try:
                self.outputs[key] = (output, _signature(output), tuple(
                    output.view(-1, 2560)[:r].view(r, 1, 2560) for r in range(1, output.shape[0] + 1)))
                self.owner.account_metadata()
            except BaseException as error:
                self.owner.failure = error
                raise

    def _shared_only(self, bank, x, picks, out, module):
        from tensorfold.cuda.sm70 import gguf_ops
        key = (out.untyped_storage().data_ptr(), out.data_ptr())
        prepared = self.outputs.get(key)
        if prepared is None or out.shape[0] > len(prepared[2]):
            raise ValueError("DP4A output lacks its original cold binding")
        shared_picks = self.shared_pick_views[x.shape[0] - 1]
        # Gate scratch is idle until the following DP4A gate projection. No
        # permanent GPU word, weight copy or scratch allocation is introduced.
        shared_picks.fill_(512)
        gguf_ops.experts(x, shared_picks, bank.gate, bank.up, bank.down, bank.shared,
                        prepared[2][x.shape[0] - 1], blockwise=True, projection_warps=2)
        module.place_shared(out, picks)

    def _count(self, name):
        count = getattr(self, name)
        if count == (1 << 63) - 1:
            self.counter_overflowed = True
        else:
            setattr(self, name, count + 1)

    def check(self):
        owner = self.owner
        owner.check_alive()
        stream = torch.cuda.current_stream(owner.weights.device)
        if self.name == "main":
            if stream != owner.stream or torch.cuda.is_current_stream_capturing():
                raise RuntimeError("DP4A main role cannot be borrowed by capture/foreign stream")
        elif owner.capture_parent is None or stream != owner.capture_parent.side or not owner.borrowed:
            raise RuntimeError("DP4A capture role requires its explicitly active original graph borrower")
        if torch.cuda.is_current_stream_capturing() and not owner.capturing:
            raise RuntimeError("DP4A role refuses a capture not begun by its active borrower")
        owner.pending = True  # Before any submitted writer, including cold capture bodies.

    def input(self, value):
        self.check()
        try:
            extension().contain(value, self.status, 1)
        except BaseException as error:
            self.owner.failure = error
            raise

    def terminal(self, value):
        self.check()
        try:
            extension().contain(value, self.status, 32)
        except BaseException as error:
            self.owner.failure = error
            raise

    def run(self, bank, x, picks, out):
        self.check()
        try:
            return self._run(bank, x, picks, out)
        except BaseException as error:
            self.owner.failure = error
            raise

    def _run(self, bank, x, picks, out):
        owner = self.owner
        if owner.bank_ids.get(id(bank)) is not bank:
            raise ValueError("DP4A role lost original target bank")
        module = extension()
        if eligible(bank, "target"):
            self._shared_only(bank, x, picks, out, module)
            qx, qh, gate, up, hidden, flat_hidden = self.views[x.shape[0] - 1]
            module.quantize(x, qx, 0, self.status)
            module.project(qx, picks, bank.gate.data, bank.gate.ggml_type, False, gate, self.status)
            module.project(qx, picks, bank.up.data, bank.up.ggml_type, False, up, self.status)
            module.activate(gate, up, hidden)
            module.quantize(flat_hidden, qh, bank.down.column_offset, self.status)
            module.project_routed(qh, picks, bank.down.data, 20, True, out, self.status)
            self._count("calls")
        else:
            bank.forward(x, picks, out=out)
            self._count("exact_calls")
        # Preserve every finite result bit. Exceptional output is contained but
        # the monotonic fatal word survives and vetoes ALL ranks at the cohort.
        module.contain(out, self.status, 16)
        return out


class Owner:
    def __init__(self, weights):
        from .capture_lifetime import host_memory
        from tensorfold.cuda.capacity import admit_host_commitment

        self.weights = weights
        self.failure = None
        self.ready = False
        self.retiring = self.closed = self.pending = self.borrowed = self.capturing = False
        self.capture_parent = self.cohort = None
        self.main = self.capture = None
        self.comm = weights.comm
        if (weights.meta.get("sm70_profile") != PROFILE or self.comm is None or self.comm.world != 8
                or weights.cfg.execution_dtype != "float16" or len(weights.layers) != 48
                or any(type(getattr(weights.cfg, key, None)) is not int or getattr(weights.cfg, key) != value
                       for key, value in (("hidden", 2560), ("moe_width", 80), ("experts", 512), ("top_k", 10)))):
            raise ValueError("DP4A target owner requires its explicit TP8 FP16 numerical profile")
        self.stream = torch.cuda.current_stream(weights.device)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DP4A owner construction refuses foreign capture")
        self.host_preflight = admit_host_commitment(8 * funding()["host_commitment_bytes"])
        before = host_memory()
        self.banks = tuple(layer.moe.experts for layer in weights.layers)
        self.eligible_mask = tuple(eligible(bank, "target") for bank in self.banks)
        if not any(self.eligible_mask):
            raise ValueError("DP4A target profile has no eligible retained hot-format bank")
        self.graph_submitted_hot_layer_calls = 0
        self.graph_counter_overflowed = False
        self.bank_ids = {id(bank): bank for bank in self.banks}
        self.tensors = tuple((t, t.untyped_storage(), _signature(t)) for bank in self.banks for t in
                             (bank.gate.data, bank.up.data, bank.down.data, *(p.weight for p in bank.shared)))
        self.parts = tuple((bank, (bank.gate, bank.up, bank.down, *bank.shared),
                            (bank.gate.data, bank.up.data, bank.down.data, *(part.weight for part in bank.shared)))
                           for bank in self.banks)
        if any(bank.count != 513 or bank.width != 80 or bank.dims != 2560 or len(bank.shared) != 3 or bank.implementation != "packed_blocks2w" for bank in self.banks):
            raise ValueError("DP4A target requires all48 original513/80/2560 banks and exact shared owners")
        if any(t.requires_grad for t, _storage, _span in self.tensors):
            raise ValueError("DP4A target bank owners must be inference-only")
        self.layouts = tuple(_layout(bank) for bank in self.banks)
        self.math = (torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                     torch.backends.cuda.matmul.allow_fp16_accumulation)
        if self.math != (True, False):
            raise ValueError("DP4A shared exact path requires selected True/False math policy")
        self.main = object.__new__(Role)
        self.main.__init__(self, "main")
        self.capture = object.__new__(Role)
        self.capture.__init__(self, "capture")
        self.status_words = (self.main.status, self.capture.status)
        after = host_memory()
        self.host_observed_bytes = max(0, after['VmRSS'] - before['VmRSS'], after['VmHWM'] - before['VmHWM'])
        # Process RSS/HWM also includes the independently funded concurrent
        # whole-file hash workers; it is observational, not attribution to this
        # owner. The controlled bound below is the actual per-owner gate.
        self.account_metadata()
        self.ready = True

    def account_metadata(self):
        """Cold bounded object accounting, never a hot frame walk.

        Borrowed model/weight/output payloads are excluded. Runtime-owned native
        Tensor/view headers get2048B each (actual selected SDK sizeof is checked),
        plus1MiB bounded construction/accounting temporary allowance. All of
        this fits the fixed3MiB commitment or startup refuses before publication.
        """
        if extension().metadata_header_bytes() > NATIVE_HANDLE_BYTES:
            raise MemoryError("DP4A selected native header exceeds its explicit allowance")
        external = {id(self.weights), id(self.comm), id(self), id(self.capture_parent), id(self.cohort)}
        for bank, parts, tensors in self.parts:
            external.add(id(bank))
            external.update(id(value) for value in (*parts, *tensors))
        for role in (self.main, self.capture):
            external.update(id(row[0]) for row in role.outputs.values())
        seen, tensor_count = set(), 0

        def visit(value):
            nonlocal tensor_count
            key = id(value)
            if key in external or key in seen or value is None or isinstance(value, (bool, str, torch.UntypedStorage)):
                return 0
            if len(seen) == MAX_METADATA_OBJECTS:
                raise MemoryError("DP4A metadata object count exceeds its finite commitment")
            seen.add(key)
            size = sys.getsizeof(value)
            if isinstance(value, torch.Tensor):
                tensor_count += 1
            elif isinstance(value, dict):
                size += sum(visit(k) + visit(v) for k, v in value.items())
            elif isinstance(value, (tuple, list, set, frozenset)):
                size += sum(visit(v) for v in value)
            return size

        python_bytes = visit(vars(self)) + visit(vars(self.main)) + visit(vars(self.capture))
        bound = python_bytes + tensor_count * NATIVE_HANDLE_BYTES + METADATA_TEMP_BYTES
        if tensor_count > MAX_TENSOR_HANDLES or bound > HOST_BYTES:
            raise MemoryError("DP4A controlled metadata exceeds its exact host commitment")
        self.metadata_receipt = {"python_owned_bytes": python_bytes, "native_tensor_handles": tensor_count,
                                 "native_header_allowance": NATIVE_HANDLE_BYTES,
                                 "construction_temporary_bytes": METADATA_TEMP_BYTES,
                                 "controlled_bound_bytes": bound, "committed_bytes": HOST_BYTES}

    def check_alive(self):
        if not self.ready or self.closed or self.retiring or self.failure is not None:
            raise RuntimeError("DP4A target numerical owner requires retirement") from self.failure
        if (type(self.status_words) is not tuple or len(self.status_words) != 2
                or self.status_words[0] is not self.main.status or self.status_words[1] is not self.capture.status):
            raise RuntimeError("DP4A original role-status bindings changed")

    def _check_math(self):
        if (torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                torch.backends.cuda.matmul.allow_fp16_accumulation) != self.math:
            raise RuntimeError("DP4A exact shared math policy changed")

    def validate_frame(self):
        self.check_alive()
        self._check_math()
        if self.weights.meta.get("sm70_profile") != PROFILE or self.weights.meta.get("sm70_dp4a_target_trial", POLICY) != POLICY:
            raise RuntimeError("DP4A original numerical profile changed")
        for role in (self.main, self.capture):
            for output, signature, _views in role.outputs.values():
                if _signature(output) != signature:
                    raise RuntimeError("DP4A original output buffer storage changed")
            current = (role.qx, role.qh, role.gate, role.up, role.hidden, role.status)
            if (role.views is not role.original_views or any(a is not b for a, b in zip(current, role.buffers))
                    or any(_signature(value) != signature for value, signature in zip(current, role.signatures))):
                raise RuntimeError("DP4A original scratch storage generation changed")
        if tuple(layer.moe.experts for layer in self.weights.layers) != self.banks:
            raise RuntimeError("DP4A original layer bank owners changed")
        for bank, owners, tensors in self.parts:
            current = (bank.gate, bank.up, bank.down, *bank.shared)
            data = (bank.gate.data, bank.up.data, bank.down.data, *(part.weight for part in bank.shared))
            if len(current) != len(owners) or any(a is not b for a, b in zip(current, owners)) or any(a is not b for a, b in zip(data, tensors)):
                raise RuntimeError("DP4A original packed/shared owners changed")
        if tuple(_layout(bank) for bank in self.banks) != self.layouts:
            raise RuntimeError("DP4A original bank layout changed")
        for tensor, _storage, signature in self.tensors:
            if _signature(tensor) != signature:
                raise RuntimeError("DP4A original tensor storage generation changed")

    def count_graph(self):
        count = sum(self.eligible_mask)
        if self.graph_submitted_hot_layer_calls > (1 << 63) - 1 - count:
            self.graph_counter_overflowed = True
        else:
            self.graph_submitted_hot_layer_calls += count

    def assert_exposure(self):
        self.check_alive()
        if self.pending:
            raise RuntimeError("DP4A frame must complete its all-rank numeric veto before authority")

    def check_received(self, peers):
        self.check_alive()
        if len(peers) != 8 or any(len(row) != 2 for row in peers):
            raise ValueError("DP4A completed status packet has wrong rank/role layout")
        failed = next(((rank, role, bits) for rank, words in enumerate(peers)
                       for role, bits in enumerate(words) if bits), None)
        if failed is not None:
            rank, role, bits = failed
            self.failure = RuntimeError(f"DP4A numerical frame failed: rank={rank}, role={role}, bits={bits}; decoder retirement required")
            raise self.failure
        self.pending = False

    def bind_capture(self, parent):
        self.check_alive()
        if self.capture_parent is not None and self.capture_parent is not parent:
            raise RuntimeError("DP4A capture borrower cannot change")
        if parent.dec.w is not self.weights:
            raise ValueError("DP4A graph borrower must own this exact model")
        self.capture_parent = parent

    @contextmanager
    def borrow(self, parent):
        self.check_alive()
        if self.capture_parent is not parent or self.borrowed or torch.cuda.is_current_stream_capturing():
            raise RuntimeError("DP4A capture requires its original nonnested graph borrower")
        self.borrowed = True
        try:
            yield
        except BaseException as error:
            self.failure = error
            raise
        finally:
            self.borrowed = self.capturing = False

    def close(self):
        if self.closed:
            return
        parent = getattr(self, "capture_parent", None)
        if parent is not None and (getattr(parent, "pending", ()) or getattr(parent, "verify", {}) or getattr(parent, "mtp", {})
                or getattr(parent, "bound", None) is not None and not parent.bound.closed
                or getattr(parent, "moe_regions", None) is not None and not parent.moe_regions.closed):
            raise RuntimeError("DP4A graph borrowers must positively retire before owner close")
        self.retiring = True
        try:
            torch.cuda.synchronize(self.weights.device)
        except BaseException as error:
            self.failure = error
            raise
        # Engine calls only after all graph borrowers have positively retired.
        self.main = self.capture = self.status_words = self.tensors = self.layouts = self.banks = self.bank_ids = None
        self.parts = self.capture_parent = self.cohort = self.weights = None
        self.closed = True
