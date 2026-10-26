"""Model-owned cuBLAS arithmetic for the initial SM70 media profile.

These Torch flags are process global. The engine owns the sole active model in
this process and drains its consumers before an in-process replacement. Other
hardware/profile paths do not create this scope. This owner allocates no device
storage and grants no software, numerical-quality or performance qualification.
"""
from __future__ import annotations

import os

import torch

from tensorfold.cleanup import finish, rollback

from .mmproj_vision import vision_arithmetic


def _flags():
    matmul = torch.backends.cuda.matmul
    values = (matmul.allow_fp16_accumulation, matmul.allow_fp16_reduced_precision_reduction,
              matmul.allow_fp16_reduced_precision_reduction_split_k)
    if any(type(value) is not bool for value in values):
        raise ValueError("media arithmetic requires the pinned Torch boolean cuBLAS flag contract")
    return values


class MediaNumericScope:
    @classmethod
    def create(cls, *, register):
        """Publish the actual partial owner before any global flag change."""
        if not callable(register):
            raise ValueError("media arithmetic requires the model's owner journal")
        owner = cls.__new__(cls)
        owner.pid = os.getpid()
        owner.original = None
        owner.active = owner.closed = owner.retired = False
        owner.accumulation_attempted = owner.reduction_attempted = False
        try:
            register(owner)
            owner.original = _flags()
            owner.active = True
            owner.accumulation_attempted = True
            torch.backends.cuda.matmul.allow_fp16_accumulation = False
            owner.reduction_attempted = True
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, owner.original[2])
            owner.verify()
        except BaseException as primary:
            # No model math has started, so only this constructor's verified
            # partial flag changes need retirement. Preserve the primary and
            # retain the journaled owner if restoration cannot finish.
            rollback(owner, primary, owner._rollback_creation)
        return owner

    def _check_process(self):
        if os.getpid() != self.pid:
            raise ValueError("media arithmetic scope belongs to another process")

    def verify(self):
        """Check actual current arithmetic; split-K remains independent."""
        self._check_process()
        if not self.active or self.closed or self.retired:
            raise ValueError("media arithmetic scope is not active")
        arithmetic = vision_arithmetic()
        actual = (arithmetic["linear_fp16_accumulation"], arithmetic["linear_fp16_reduced_precision"],
                  arithmetic["linear_fp16_split_k"])
        if actual != (False, False, self.original[2]):
            raise ValueError("media arithmetic changed after model adoption")
        return arithmetic

    def record(self):
        """Observation for full profile identity, never release authority."""
        return {"schema": "tensorfold.sm70-media-numerics", "version": 1,
                "torch_version": str(torch.__version__), "policy": "cublas-fp16-f32acc-reduction-v1",
                "scope": "one-model-process-global-drained-lifecycle", "arithmetic": self.verify()}

    def _restore_reduction(self):
        if not self.reduction_attempted:
            return
        current = _flags()[1:]
        original = self.original[1:]
        if current not in (original, (False, original[1])):
            raise ValueError("media reduction flags changed outside the owned partial transition")
        if current != original:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = original
        if _flags()[1:] != original:
            raise ValueError("media reduction flag restoration did not complete")

    def _restore_accumulation(self):
        if not self.accumulation_attempted:
            return
        current = _flags()[0]
        if current not in (self.original[0], False):
            raise ValueError("media accumulation flag changed outside the owned partial transition")
        if current != self.original[0]:
            torch.backends.cuda.matmul.allow_fp16_accumulation = self.original[0]
        if _flags()[0] != self.original[0]:
            raise ValueError("media accumulation flag restoration did not complete")

    def _restore(self):
        if self.original is not None:
            finish((self._restore_reduction, self._restore_accumulation))
            if _flags() != self.original:
                raise ValueError("media arithmetic changed outside the owned transition")
        self.active = False
        self.closed = True

    def _rollback_creation(self):
        self._check_process()
        self.retired = True  # Construction precedes all model math.
        self._restore()

    def close(self, *, retire):
        """Fence/drain all owning-engine consumers before exact restoration.

        A failed retirement leaves the active policy and owner intact. A
        restoration failure retains the already-retired owner for retry and
        propagates; a successful fence is never repeated by that retry.
        """
        self._check_process()
        if self.closed:
            return
        if not callable(retire):
            raise ValueError("media arithmetic retirement requires the owning engine's fence/drain")
        if not self.retired:
            retire()
            self.retired = True
        self._restore()
