"""CUDA graphs for Flash Next's concurrent decode rounds: a pure decode round's verify forward and each MTP step.

A forward is captured at startup in pieces, one key per (streams, rows) and, for the verify, the DeltaNet scratch
parity: everything that changes within a key reaches the kernels through the decoder's static tables
(``gdn_multi.RoundTables``, ``attn_multi.StepTables``) and its static buffers, written before each replay. What a
step's own host data shapes runs eagerly between the pieces, on the current step's data: each attention layer's block
selection (its grid follows the longest context), each stream's n-gram conv tail (a State pointer) and the late
n-gram reads (``forward.stage_late``). Replays run exactly the kernels and launch parameters of the static-table eager
path for the same key, so their bits are its bits.

Pieces of every key share one private memory pool (``torch.cuda.graph_pool_handle``): a round replays one key's
pieces in order and no piece's pool memory outlives it as an output (outputs are the static buffers). Keys are
captured in order of expected use (fewer streams first) while their measured memory stays within the budget
(``TENSORFOLD_MULTI_GRAPH_MIB`` or the remaining automatic service allowance, recorded in startup admission);
the rest run eagerly with a recorded reason. A key's memory is the
drop of free memory across its capture alone (its pool blocks and driver graph memory), measured after its eager
warm-up's own allocator blocks are in place; the stream caches' room loses the larger of the kept keys' sum and the
whole capture's settled drop. Rounds with a prompt pass, image rows or the lookup decoder, and steps whose key was
not captured, run the same eager path.
"""

from __future__ import annotations

import gc
import time
from functools import partial

from .profile_construction import replay as profile_replay

import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda import pdl
# re-exported for the decoder (multi.py): the captured rounds' budget, geometry's variable and parser (one source)
from tensorfold.cuda.geometry import MULTI_GRAPH_ENV as ENV, multi_graph_request as budget_bytes  # noqa: F401

from . import attn_multi, gdn_multi, history_runtime
from .forward import _kv_begin, _kv_check, compute, ple_conv_streams, stage, stage_late
from .mtp import mtp_compute, mtp_stage
from .service_memory import failure_frame_values, private_pool_spare

MIB = 1 << 20


def free_bytes() -> int:
    """The memory graphs take from: a unified GPU's MemAvailable (the host's), else the card's free memory. Uncapped
    by TENSORFOLD_CUDA_MEMORY_LIMIT_GB, so a key's driver graph memory shows beside its private pool's blocks."""

    from tensorfold.cuda.capacity import page_room

    room = page_room(torch)
    return int(room) if room is not None else int(torch.cuda.mem_get_info()[0])


def prepare_capture() -> None:
    """Return ordinary warmup cache before choosing physical room; private graph cache is never credited."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()


def live_bytes() -> int:
    from tensorfold.cuda.capacity import available_bytes, cuda_limit_bytes

    room = int(available_bytes(torch))
    limit = cuda_limit_bytes()
    return max(0, min(room, limit - int(torch.cuda.memory_allocated()))) if limit is not None else room


def service_live_bytes(pool):
    if pool.private_spare_bytes is None:
        # Backends without the native segment proof retain physical-only
        # admission. Never infer zero private storage from an empty snapshot.
        return live_bytes
    from tensorfold.cuda.capacity import available_bytes
    from tensorfold.cuda.memory_gate import torch_live

    return torch_live(torch, available_bytes, unavailable_spare_bytes=pool.private_spare_bytes, owner=pool)


def capturable(w) -> str | None:
    """Why this model's rounds cannot be captured (None: they can): experts that read their plan on the host."""

    if w.meta.get("history_layout") == "unique":
        return "unique history owner execution has no qualified padded capture contract"
    if w.comm is not None and w.meta.get("execution_provider") != "sm70":
        return "tensor-parallel ranks"
    if w.meta.get("execution_provider") == "sm70" and not w.meta.get("capture_ready", False):
        return w.meta.get("capture_reason", "the SM70 family provider has not prepared capture-safe state")
    if any(getattr(layer.moe.experts, "capturable", True) is False for layer in w.layers):
        return "this MoE reads its plan's item list on the host, which a capture rejects"
    if w.mtp is not None and getattr(w.mtp.layer.moe.experts, "capturable", True) is False:
        return "the MTP head's MoE reads its plan's item list on the host"
    if w.meta.get("execution_provider") == "sm70" and sum(layer.ple is not None for layer in w.layers) > 1:
        return "bound SM70 captures require at most one PLE table"
    return None


class Pieces:
    """One key's captured pieces, the inserts between them (``(kind, layer)``) and its output view."""

    __slots__ = ("graphs", "inserts", "out")

    def __init__(self) -> None:
        self.graphs: list[torch.cuda.CUDAGraph] = []
        self.inserts: list[tuple] = []
        self.out: torch.Tensor | None = None


class RoundGraphs:
    """A ``MultiDecoder``'s captured verify forwards and MTP steps (``capture`` at its warm-up, before any request)."""

    def __init__(self, dec, budget: int, *, spare=None) -> None:
        self.dec, self.budget = dec, int(budget)
        self.spare = spare                              # live bytes after required future service, preparation only
        self.allocator_backend = torch.cuda.memory.get_allocator_backend()
        self.pool = torch.cuda.graph_pool_handle()
        self.side = torch.cuda.Stream()
        self.verify: dict[tuple[bool, int, int, int], Pieces] = {}    # (PDL, streams, rows, DeltaNet parity)
        self.mtp: dict[tuple[bool, int, int], Pieces] = {}             # (PDL, streams, rows)
        self.bytes = 0                                   # the kept keys' measured memory (each >= 0): within budget
        self.settled_bytes = 0                           # the whole capture's drop of free memory, settled (raw)
        self.room_bytes = 0                              # what the stream caches' room loses: the larger of the two
        self.capture_s = 0.0
        self.measured: list[dict] = []                   # per key: raw bytes (< 0: outside noise), pieces, seconds
        self.stopped: str | None = None                  # why capture stopped before the last key (the budget)
        self.rejected: dict | None = None                # the attempted key, including its raw memory observation
        self.private_spare_bytes: int | None = None
        self.allocator_record: dict = {}
        self.admitted = False
        self.failure = self.failure_traceback = self.failure_cause = self.failure_context = None
        self.failure_frames = None
        self.pending = []
        self.bound = None
        if dec.w.meta.get("execution_provider") == "sm70":
            from .bound_graphs import BoundStateGraphs

            self.bound = BoundStateGraphs(self)

    def retain_failure(self, primary) -> None:
        """Own the original tensor-bearing frames independently of later exception traceback clearing."""
        if self.failure is None:
            self.failure_traceback = BaseException.__traceback__.__get__(primary)
            self.failure_cause = BaseException.__cause__.__get__(primary)
            self.failure_context = BaseException.__context__.__get__(primary)
            self.failure = primary

    # -- replay ------------------------------------------------------------------------------------------------
    def verify_forward(self, segs) -> torch.Tensor | None:
        """The round's verify logits from its captured key (its tables written), or None: this round runs eagerly."""

        if getattr(self, "bound", None) is not None:
            return self.bound.forward("verify", segs)
        b = self.dec.buf
        if b.rotor_lookup or b.attn_step.vision:
            return None
        pieces = self.verify.get((pdl.ON, len(segs), segs[-1][2], self.dec.gdn.parity))
        if pieces is None:
            return None
        self._replay(pieces, segs, b)
        return pieces.out

    def mtp_step(self, segs, *, mtp_positions=None, device_tokens=None) -> torch.Tensor | None:
        if getattr(self, "bound", None) is not None:
            return self.bound.forward("mtp", segs, mtp_positions=mtp_positions, device_tokens=device_tokens)
        if mtp_positions is not None or device_tokens is not None:
            raise ValueError("provisional MTP positions and device tokens require bound SM70 graphs")
        b = self.dec.mbuf
        if b.rotor_lookup or b.attn_step.vision:
            return None
        pieces = self.mtp.get((pdl.ON, len(segs), segs[-1][2]))
        if pieces is None:
            return None
        self._replay(pieces, segs, b, mtp=True)
        return pieces.out

    def _replay(self, pieces: Pieces, segs, b, *, mtp=False) -> None:
        w = self.dec.w
        archive = getattr(w, "meta", {}).get("indexer_history") == "ram"
        if archive:
            history_runtime.begin(segs, prefill=b.prefill, mtp=mtp)
        _kv_begin(segs)                                  # the frame's codec status: checked by the caller
        for graph, insert in zip(pieces.graphs, pieces.inserts + [None]):
            profile_replay(graph, self.dec.w.comm)
            if insert is None:
                continue
            kind, layer = insert
            if kind == "select":
                attn_multi.select(layer, w, b, b.attn_step)
            elif kind == "ple_conv":
                ple_conv_streams(layer, w, segs, b)
            elif b.ple_late is not None:                 # "ple_stage": the late n-gram reads (stage(late=True))
                stage_late(w, b)
        if archive:
            history_runtime.submitted(segs, prefill=b.prefill, mtp=mtp)

    # -- capture -----------------------------------------------------------------------------------------------
    def _pieces(self, run, b, *, split=True) -> Pieces:
        """``run()`` captured in pieces, one ending at each ``b.split`` insert. Only a begun capture is ended, once;
        a failure inside ``run`` stays the raised error, with a failed end of its open capture attached."""

        pieces = Pieces()
        pending = getattr(self, "pending", None)
        if pending is None:
            pending = self.pending = []
        pending.append(pieces)                          # retain native owners before any fallible capture call
        open_graph: list[torch.cuda.CUDAGraph] = []      # the piece being captured (begun, not yet ended)

        def begin() -> None:
            graph = torch.cuda.CUDAGraph()
            open_graph.append(graph)
            pieces.graphs.append(graph)
            graph.capture_begin(pool=self.pool, capture_error_mode="thread_local")

        def end() -> None:
            open_graph.pop().capture_end()               # popped first: a failed end is never retried

        def insert_piece(insert) -> None:
            end()
            pieces.inserts.append(insert)
            begin()

        b.split = insert_piece if split else None
        try:
            with torch.cuda.stream(self.side):
                begin()
                try:
                    pieces.out = run()
                except BaseException as primary:
                    errors = []
                    if open_graph:
                        try:
                            end()
                        except BaseException as error:
                            errors.append(error)
                    raise_failures(primary, errors)
                end()
        finally:
            b.split = None
        pending.remove(pieces)
        return pieces

    def capture(self, *, prepared: bool = False) -> None:
        """Capture every key in order (streams 1, 2, ...: verify at both parities, then the MTP step) on the free
        slots while the kept keys' measured memory stays within the budget; a key past it is dropped and capture
        stops. A key is measured across its capture alone (``free_bytes`` after its eager warm-up's allocator blocks,
        then after its capture): its private pool blocks and driver graph memory, plus any outside change in the
        same interval, recorded raw (a negative figure is outside noise and counts as 0 against the budget). The
        slots are reset afterwards; the settled drop of the whole capture (after ``empty_cache``) is recorded, and
        the stream caches' room loses the larger of it and the keys' sum (``room_bytes``).

        Keys are captured with the decoder's PDL policy (``dec.pdl``, which its rounds launch with). A failure stays
        the raised error; a failed restore of the slots afterwards is attached to it."""

        dec = self.dec
        states = list(dec.free)
        cleanup_errors = [None]
        parity, launch = dec.gdn.parity, pdl.ON
        started = time.perf_counter()
        if not prepared:
            prepare_capture()
        before = free_bytes()
        enabled = gc.isenabled()
        gc.disable()                                     # collecting a graph's exec invalidates an active capture
        primary = None
        try:
            self._keys(states)
        except BaseException as error:
            primary = error
            self.retain_failure(primary)
        finally:
            if enabled:
                gc.enable()
            pdl.ON = launch
            dec.gdn.parity = parity
            dec.buf.gdn_tables = dec.buf.attn_step = None
            if dec.mbuf is not None:
                dec.mbuf.attn_step = None
        try:
            torch.cuda.synchronize()
            for st in states:                            # the warm-up forwards wrote their caches and states
                st.reset(dec.w)
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()                     # a dropped key's exec and the warm-ups' blocks back
            self.settled_bytes = before - free_bytes()
            self.room_bytes = max(self.bytes, self.settled_bytes)
            self.capture_s = time.perf_counter() - started
            self._account_pool()
            if self.room_bytes > self.budget:
                raise MemoryError(f"captured rounds retain {self.room_bytes} bytes beyond their {self.budget}-byte "
                                  "admitted budget after settlement")
            if self.spare is not None and self.spare() < 0:
                raise MemoryError("captured rounds cannot retain the required native window and service workspace; "
                                  "live memory changed during capture")
            self.admitted = True
        except BaseException as error:
            if primary is None:
                raise
            cleanup_errors[0] = error
            raise_failures(primary, cleanup_errors)
        if primary is not None:
            raise_failures(primary, [])

    def _account_pool(self) -> None:
        if self.failure is not None and self.failure_frames is None:
            self.failure_frames = failure_frame_values(self.failure, self.failure_traceback,
                                                        self.failure_cause, self.failure_context)
        if self.allocator_backend == "native":
            try:
                segments = torch.cuda.memory_snapshot(mempool_id=self.pool, include_traces=False)
                self.private_spare_bytes = private_pool_spare(segments, self.pool, self.dec.w.device.index or 0)
            except (AttributeError, TypeError, ValueError) as error:
                raise MemoryError("captured-round private allocator storage cannot be accounted for") from error
        self.allocator_record = {"backend": self.allocator_backend,
                                 "accounting": "native-pool-inactive" if self.private_spare_bytes is not None else "physical-only",
                                 "physical_free_bytes": free_bytes(), "available_after_floor_bytes": live_bytes(),
                                 "reserved_bytes": int(torch.cuda.memory_reserved()),
                                 "allocated_bytes": int(torch.cuda.memory_allocated()),
                                 "private_pool_spare_bytes": self.private_spare_bytes}

    def _has_room(self) -> bool:
        if self.spare is not None and self.spare() <= 0:
            self.stopped = "live memory has no room after the required native window and service workspace"
            return False
        return True

    def _keys(self, states) -> None:
        if getattr(self, "bound", None) is not None:
            self.bound.warm(states)
            return
        dec = self.dec
        slots, width, launch = len(states), dec.depth + 1, dec.pdl
        pdl.ON = launch
        for n in range(1, slots + 1):
            for rows in range(n, n * width + 1):
                keys = [("verify", rows, p) for p in (0, 1)] + ([("mtp", rows, 0)] if dec.mbuf is not None else [])
                for kind, R, p in keys:
                    if not self._has_room():
                        return
                    t0 = time.perf_counter()
                    run, b = self._warm(kind, states[:n], R, p)
                    if not self._has_room():
                        return
                    # the warm-up's allocator blocks and module loads are taken as it is enqueued: no wait
                    free = free_bytes()
                    pieces = self._pieces(run, b)
                    torch.cuda.synchronize()
                    took = free - free_bytes()
                    if pieces.out is None:
                        raise RuntimeError(f"a captured {kind} step returned no logits")
                    remaining = self.spare() if self.spare is not None else None
                    if self.bytes + max(0, took) > self.budget or (remaining is not None and remaining < 0):
                        del pieces                       # this key's graphs go; the earlier keys stay
                        self.rejected = {"key": [kind, launch, n, R, p], "bytes": took,
                                         "live_spare_bytes": remaining}
                        self.stopped = (f"{(self.bytes + took) / MIB:.0f} MiB with the next key exceeds the "
                                        f"{self.budget // MIB} MiB budget" if remaining is None or remaining >= 0 else
                                        "the next key would consume required native-window or service memory")
                        return
                    if kind == "verify":
                        self.verify[(launch, n, R, p)] = pieces
                    else:
                        self.mtp[(launch, n, R)] = pieces
                    self.measured.append({"key": [kind, launch, n, R, p], "pieces": len(pieces.graphs),
                                          "bytes": took, "s": round(time.perf_counter() - t0, 4)})
                    self.bytes += max(0, took)

    def _warm(self, kind: str, states, R: int, parity: int):
        """One key's eager warm-up forward (compiles and configures its kernels) on its tables; returns the forward
        to capture and its buffers."""

        dec = self.dec
        w, n = dec.w, len(states)
        counts = [R // n + (s < R % n) for s in range(n)]
        if kind == "verify":
            b = dec.buf
            segs = stage(w, b, [(st, [0] * c) for st, c in zip(states, counts)])
            dec.gdn.parity = parity
            b.gdn_tables = gdn_multi.Tables(dec.gdn, segs, [[] for _ in states], static=dec.round_tables)
            b.attn_step = attn_multi.Step(w, segs, mtp=False, tables=dec.step_tables)
            run = partial(compute, w, segs, b, check=False)
        else:
            b = dec.mbuf
            at, windows, zeros = 0, [], torch.zeros_like(dec.buf.streams[:R])      # defined input streams
            for st, c in zip(states, counts):
                windows.append((st, [0] * c, zeros[at:at + c]))
                at += c
            segs = mtp_stage(b, windows)
            b.attn_step = attn_multi.Step(w, segs, mtp=True, tables=dec.mtp_tables)
            run = partial(mtp_compute, w, segs, b, check=False)
        b.rotor_lookup = False
        run()
        _kv_check(segs)
        return run, b

    def record(self) -> dict:
        """The startup record (capacity plan): captured keys, measured bytes, capture time, per-key measures."""

        return {"verify_keys": len(self.verify), "mtp_keys": len(self.mtp), "bytes": self.bytes,
                "settled_bytes": self.settled_bytes, "room_bytes": self.room_bytes,
                "budget_bytes": self.budget, "capture_s": round(self.capture_s, 3), "stopped": self.stopped,
                "rejected": self.rejected,
                "allocator": self.allocator_record,
                "pieces": sum(len(p.graphs) for p in (*self.verify.values(), *self.mtp.values())),
                "keys": self.measured}

    def invalidate_state(self, state):
        if getattr(self, "bound", None) is not None:
            self.bound.invalidate(state)

    def close(self):
        """After replay admission closes, fence and retire graphs before borrowed native owners."""
        torch.cuda.synchronize(self.dec.w.device)
        if getattr(self.dec.w, "dspark", None) is not None:
            # A failed capture can retain graphs outside the bound catalogue.
            # Retire those borrowers before bound.close releases tap anchors.
            for pieces in tuple(getattr(self, "pending", ())):
                for graph in pieces.graphs:
                    graph.reset()
                self.pending.remove(pieces)
        if getattr(self, "bound", None) is not None:
            self.bound.close()
        else:
            for mapping in (self.verify, self.mtp):
                for key, pieces in tuple(mapping.items()):
                    for graph in pieces.graphs:
                        graph.reset()
                    del mapping[key]
        for pieces in tuple(getattr(self, "pending", ())):
            for graph in pieces.graphs:
                graph.reset()
            self.pending.remove(pieces)
