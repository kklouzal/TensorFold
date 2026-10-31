"""Explicit process ownership of the pinned Blackwell BF16 BLAS policy.

PyTorch's preferences are process-global. One E2B engine owns this lease from
construction until all work drains and target closes; overlapping leases are
rejected. Other independent Torch CUDA work in the same process is outside this
engine contract. Separate service processes retain their own preferences.
"""
import threading

from tensorfold.cleanup import raise_failures, rollback


_POLICY_GUARD = threading.Lock()


class E2BBlasPolicy:
    def __init__(self, torch):
        self.backend = torch.backends.cuda
        self._before = None
        self._held = False
        if not _POLICY_GUARD.acquire(blocking=False):
            raise RuntimeError("Gemma E2B BLAS policy already belongs to an active engine in this process")
        self._held = True
        try:
            self._before = (self.backend.preferred_blas_library(),
                            self.backend.matmul.allow_bf16_reduced_precision_reduction,
                            self.backend.matmul.allow_bf16_reduced_precision_reduction_split_k)
            self.backend.preferred_blas_library("cublaslt")
            self.backend.matmul.allow_bf16_reduced_precision_reduction = (False, False)
        except BaseException as primary:
            rollback(self, primary, self.close)

    def close(self):
        if not self._held:
            return
        errors = []
        if self._before is not None:
            library, reduced, split_k = self._before
            try:
                self.backend.matmul.allow_bf16_reduced_precision_reduction = (reduced, split_k)
            except BaseException as error:
                errors.append(error)
            try:
                self.backend.preferred_blas_library(library)
            except BaseException as error:
                errors.append(error)
        # A failed restore retains its process guard and owner for explicit
        # retry/terminal teardown, preventing an ambiguous replacement lease.
        raise_failures(None, errors)
        self._held = False
        _POLICY_GUARD.release()
