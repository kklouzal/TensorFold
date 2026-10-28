"""Exact target-only greedy verification candidate; MTP confidence is unchanged."""

import numpy as np
import torch
import weakref

from tensorfold.cuda.sampling import validate_policy, validate_rows

POLICY = "source-target-greedy-int-key-chain-v1"


def workspace_bytes(vocabulary):
    """EOS payload upper bound; all packet/feedback data reuse admitted slabs.

    Config resolves unique generation EOS IDs within the full vocabulary.
    Source04cf metadata: EOS GPU/temporary CPU tensor headers2*368,
    three retained views3*248, owner128, two weakrefs160, old/new counter72, <=8 transient
    view headers8*248. Allocator/driver RSS remains separately unqualified.
    """
    if type(vocabulary) is not int or not 0 < vocabulary <= np.iinfo(np.int32).max:
        raise ValueError("target greedy vocabulary requires a positive int32 domain")
    return 4 * vocabulary, 4 * vocabulary + 2 * 368 + 3 * 248 + 128 + 2 * 80 + 72 + 8 * 248


class TargetGreedy:
    """One serialized decoder owns borrowed packet storage and immutable EOS."""
    __slots__ = ("_weights", "_buffers", "plan", "native", "eos", "error", "completed", "local", "frames",
                 "feedback", "retired", "offset")

    def __init__(self, decoder):
        from tensorfold.cuda.sm70.greedy_native import extension

        self._weights, self._buffers = weakref.ref(decoder.w), weakref.ref(decoder.buf)
        self.plan = decoder._sample_plan
        self.error, self.completed = None, 0
        self.native = self.eos = self.local = self.frames = self.feedback = None
        self.retired = False
        if (self.plan is None or self.plan.mapping is not None or not self.plan.graph_ids_safe
                or self.w.comm is None or self.w.comm.world != 8
                or self.w.meta.get("sm70_target_greedy_trial", {}).get("policy") != POLICY):
            raise ValueError("target greedy owner requires its explicit prepared source-stack trial")
        vocabulary = self.w.cfg.vocab
        workspace_bytes(vocabulary)
        if self.plan.offset != sum(self.plan.widths[:self.w.comm.rank]) or self.plan.total_width != vocabulary:
            raise ValueError("target greedy requires complete disjoint contiguous original vocabulary shards")
        self.offset = self.plan.offset
        eos = self.w.cfg.eos
        if (type(eos) is not tuple or len(eos) > vocabulary
                or any(type(value) is not int or not 0 <= value < vocabulary for value in eos)):
            raise ValueError("target greedy owner requires resolved generation EOS within its prepaid domain")
        self.native = extension()
        self.eos = torch.tensor(eos, dtype=torch.int32, device=self.w.device)
        rows, world = self.buf.rows, self.w.comm.world
        self.local = self.buf.cand.view(-1)[:4 * rows].view(torch.int64).view(rows, 2)
        self.frames = self.buf.cand_all.view(-1)[:4 * world * rows].view(torch.int64)
        # Feedback reuses local-packet bytes only after their NCCL gather.
        # Every submission stays on the actor's same device stream, so that
        # reader has retired before the accepting kernel overwrites the slab.
        self.feedback = self.buf.cand.view(torch.int32).view(-1)[:rows + 2]

    @property
    def w(self):
        value = self._weights()
        if value is None:
            raise RuntimeError("target greedy weight owner has retired")
        return value

    @property
    def buf(self):
        value = self._buffers()
        if value is None:
            raise RuntimeError("target greedy buffer owner has retired")
        return value

    def eligible(self, decoder, live, *, pieces, grammars, dflash):
        if self.retired:
            raise RuntimeError("target greedy owner has retired")
        if self.error is not None:
            raise RuntimeError("failed target greedy owner requires model retirement") from self.error
        if (len(live) != 1 or pieces or grammars or dflash is not None
                or getattr(self.w, "dspark", None) is not None):
            return False
        if decoder.graphs is not None:
            from .prefix_policy import OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, OWNER_TARGET_DENSE_PROFILE, OWNER_BATCH_PRELUDE_PROFILE, ROUTER_BLAS_PROFILE, SHARED_FUSION_PROFILE, HC_SMALLROW_VENDOR_PROFILE

            bound = getattr(decoder.graphs, "bound", None)
            if (self.w.meta.get("sm70_profile") not in (OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, OWNER_TARGET_DENSE_PROFILE, OWNER_BATCH_PRELUDE_PROFILE, ROUTER_BLAS_PROFILE, SHARED_FUSION_PROFILE, HC_SMALLROW_VENDOR_PROFILE)
                    or bound is None or not bound.unique or not bound.native_owner):
                return False
        stream = live[0]
        if (stream.owed or stream.constraint is not None or stream.probabilities is not None
                or stream.token_limit is not None or getattr(stream, "_tp_token_limit", False)
                or getattr(stream, "copies", None) is not None or stream.vision is not None
                or getattr(stream.st, "media_profile_identity", None) is not None):
            return False
        sampling = stream.sampling
        try:
            validate_policy([stream.st.pos + 1 + row for row in range(1 + len(stream.drafts))], sampling)
            return sampling is None or bool(sampling.temperature <= 0)
        except (AttributeError, TypeError, ValueError, OverflowError):
            # Preserve the ordinary producer/codec/policy failure precedence.
            return False

    def gather(self, logits, rows):
        try:
            if (self.buf.cand.data_ptr() != self.local.data_ptr()
                    or self.buf.cand_all.data_ptr() != self.frames.data_ptr() or not 0 < rows <= self.buf.rows):
                raise RuntimeError("target greedy borrowed arena changed after preparation")
            packet = self.local[:rows]
            self.native.target_local(logits[:rows], packet, self.offset, self.w.cfg.vocab)
            self.w.comm.all_gather(packet, self.frames[:self.w.comm.world * rows * 2])
        except BaseException as error:
            self.error = error  # Unknown submitted writers cannot become an optional fallback.
            raise

    def finish(self, decoder, segs, stream, logits):
        rows = segs[-1][2]
        output = self.feedback[:rows + 2]
        try:
            self.native.target_accept(self.frames[:self.w.comm.world * rows * 2].view(self.w.comm.world, rows, 2),
                self.buf.ids[:rows], self.eos, output, stream.count - len(stream.out), self.w.cfg.vocab, stream.stop_eos)
        except BaseException as error:
            self.error = error
            raise
        # The existing owner latches every rank's codec status before any
        # sampler policy/score/feedback interpretation. No pinned view escapes.
        feedback = decoder.kv_cohort.read_feedback([stream.st], output, comm=self.w.comm)
        try:
            positions = [stream.st.pos + 1 + row for row in range(rows)]
            validate_policy(positions, stream.sampling)
            validate_rows(logits, positions, stream.sampling)
            self.plan.check_model(self.w, logits)
            if self.plan.offset != self.offset:
                raise ValueError("target greedy vocabulary shard changed after preparation")
            if feedback.dtype != np.int32 or feedback.shape != (rows + 2,):
                raise ValueError("target greedy feedback differs from its bounded int32 geometry")
            keep, flags = int(feedback[0]), int(feedback[1])
            if flags & ~31:
                raise ValueError("target greedy feedback has unknown score-domain flags")
            if flags & 7:
                raise ValueError("sampling needs finite scores or -inf masks, with one finite score per complete row")
            if flags & 8:
                raise ValueError("target greedy maximum is outside the original global vocabulary")
            if flags & 16:
                raise ValueError("gathered global token IDs must be distinct")
            if not 1 <= keep <= min(rows, stream.count - len(stream.out)):
                raise ValueError("target greedy accepted prefix exceeds its producing rows/output room")
            accepted = feedback[2:2 + keep].tolist()
            if any(not 0 <= token < self.w.cfg.vocab for token in accepted):
                raise ValueError("target greedy committed IDs exceed the original vocabulary")
            self.completed += 1
            return (list(range(keep)), accepted[-1]), accepted
        finally:
            feedback = None  # No target escape copy survives into proposal readback.

    def retire_after_fence(self, *, fenced):
        if fenced is not True:
            raise ValueError("target greedy retirement requires its actual model device fence")
        self.retired = True
        self.eos = self.local = self.frames = self.feedback = self.native = self.plan = None

    def record(self):
        return {"policy": POLICY, "completed": self.completed, "retired": self.retired,
                "qualification": "unqualified-candidate", "target_only": True,
                "mtp_confidence": "original-fp32-lse-numpy-fp64"}
