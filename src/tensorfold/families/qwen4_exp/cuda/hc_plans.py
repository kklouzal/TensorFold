"""Startup-owned exact F16 HC launches over stable model/Buffers allocations.

Matrices, tensor metadata and compilation settings stay fixed for the buffer
lifetime. State KV relocation does not move these scratch tensors. Dynamic
CompiledKernel launch hooks are retained; JIT pre-run hooks select the original
readout. Unsupported optional enrollment regions retain the original path.
Explicit close follows accepted-work drain and fences before releasing views.
"""
from dataclasses import dataclass


ARGUMENTS = ('X', 'W', 'OUT', 'M', 'N', 'x_stride', 'o_stride', 'K', 'KS', 'BM', 'BN', 'BK', 'F32')


@dataclass(frozen=True)
class _F16Plan:
    kernel: object
    runner: object
    arguments: tuple
    owners: tuple
    grid: tuple

    @classmethod
    def prepare(cls, matrix, x, out, *, f32=False):
        import torch
        import triton
        from tensorfold.families.qwen4_exp.cuda.exl3_mm import _f16_mm

        if not triton.__version__.startswith('3.9.0') or tuple(_f16_mm.arg_names) != ARGUMENTS:
            raise ValueError('exact reviewed Triton3.9 F16 argument ABI required')
        if _f16_mm.pre_run_hooks:
            raise ValueError('JIT pre-run hooks require original JIT path')
        m, n, k, sk = x.shape[0], matrix.n, matrix.k, matrix.sk
        if not 1 <= m <= 128 or n <= 0 or k <= 0 or sk <= 0 or k % (sk * 64) or not f32 and sk != 1:
            raise ValueError('bounded valid original F16 matrix geometry required')
        if x.device.type != 'cuda' or x.device != out.device or x.device != matrix.w.device:
            raise ValueError('one fixed CUDA device required')
        if x.ndim != 2 or x.shape[1] != k or x.stride(1) != 1:
            raise ValueError('fixed input must satisfy original F16 layout')
        if matrix.w.dtype != torch.float16 or tuple(matrix.w.shape) != (n, k) or not matrix.w.is_contiguous():
            raise ValueError('original contiguous FP16 F16 weight required')
        expected = (sk, m, n) if f32 else (m, n)
        if tuple(out.shape) != expected or out.stride(-1) != 1 or f32 and not out.is_contiguous():
            raise ValueError('fixed output must satisfy original F16 slice layout')
        if f32 and out.dtype != torch.float32 or x.requires_grad or out.requires_grad or matrix.w.requires_grad:
            raise ValueError('non-gradient original output representation required')
        grid = ((m + 15) // 16, (n + 63) // 64, sk)
        args = (x, matrix.w, out, m, n, x.stride(0), n if f32 else out.stride(0), k, k // sk, 16, 64, 64, f32)
        kernel = _f16_mm[grid](*args[:7], K=k, KS=k // sk, BM=16, BN=64, BK=64, F32=f32,
                               num_warps=4, num_stages=3)
        if kernel.src.fn is not _f16_mm or kernel.metadata.num_warps != 4 or kernel.metadata.num_stages != 3:
            raise ValueError('original compiled specialization/provider differs')
        runner = kernel[grid]  # Same binary, launcher and packed metadata; no JIT cache/binder on replay.
        return cls(kernel, runner, args, (matrix, x, out, matrix.w), grid)

    def __call__(self):
        fn = self.kernel.src.fn
        if fn.pre_run_hooks:
            # A launch hook may register a JIT pre-run hook between HC's down
            # and up calls. Preserve the original positional/constexpr packing.
            x, weight, out, m, n, xs, os, k, ks, bm, bn, bk, f32 = self.arguments
            fn[self.grid](x, weight, out, m, n, xs, os, K=k, KS=ks, BM=bm, BN=bn, BK=bk, F32=f32,
                          num_warps=4, num_stages=3)
        else:
            self.runner(*self.arguments)


class _HCPlans:
    def __init__(self, w, b, hcs):
        import torch
        from . import glue
        from .exl3_mm import _f16_mm

        self._glue, self._jit, self._device = glue, _f16_mm, w.device
        self._entries, self._owners = {}, tuple(hcs)
        self._closed = False
        with torch.no_grad():
            b.normed.zero_()
            b.act.zero_()
            try:
                for hc in hcs:
                    rows = [None]
                    for n in range(1, b.rows + 1):
                        part = w.x3.part[:hc.down.sk * n * hc.down.n].view(hc.down.sk, n, hc.down.n)
                        down = _F16Plan.prepare(hc.down, b.normed[:n], part, f32=True)
                        up = _F16Plan.prepare(hc.up, b.act[:n], b.up[:n])
                        rows.append((down, up, part, b.act[:n], b.xs_act[:n], b.up[:n], b.normed[:n],
                                     b.mixed[:n], b.xs_mixed[:n], b.pss[:n], b.xs_normed[:n]))
                    self._entries[id(hc)] = tuple(rows)
                torch.cuda.synchronize(self._device)
            except BaseException as primary:
                # Preserve tensor owners until every startup launch is fenced.
                try:
                    torch.cuda.synchronize(self._device)
                except BaseException as cleanup:
                    raise primary from cleanup
                raise

    @property
    def enabled(self):
        return not self._closed and not self._jit.pre_run_hooks

    def __call__(self, hc, h, R, eps, streams, low, inject, normed=False):
        down, up, part, act, xs_act, out, normalized, mixed, xs_mixed, pss, xs_normed = self._entries[id(hc)][R]
        if not normed:
            self._glue.hc_normed(h[:R], pss, hc.scale, normalized, xs_normed, streams, eps)
        down()
        self._glue.hc_reduce_act(part, act, xs_act, inject, streams, low)
        up()
        self._glue.hc_mix(out, normalized, mixed, xs_mixed, streams)

    def close(self):
        if self._closed:
            return
        import torch
        torch.cuda.synchronize(self._device)
        self._entries.clear()
        self._owners = ()
        self._closed = True


def enroll(w, b, *, mtp=False):
    """Select once before admission/warm; optional limits never reject a model."""
    if b is None or b.prefill or w.x3 is None or w.comm is not None:
        return
    import triton
    from .exl3_mm import F16, _f16_mm

    hcs = ([getattr(layer, field) for layer in w.layers for field in ('attn_hc', 'mlp_hc')] + [w.mixer]
           if not mtp else [w.mtp.layer.attn_hc, w.mtp.layer.mlp_hc, w.mtp.mixer])
    # Outside the qualified region the buffer keeps the original readout:
    if not triton.__version__.startswith('3.9.0') or tuple(_f16_mm.arg_names) != ARGUMENTS:
        return                      # an unqualified optional compiled-launch ABI
    if _f16_mm.pre_run_hooks:
        return                      # JIT pre-run hooks require the original readout
    if not 1 <= b.rows <= 128 or 2 * b.rows * len(hcs) > 8192:
        return                      # past the optional startup plan count/row budget
    if any(not isinstance(hc.down, F16) or not isinstance(hc.up, F16) or hc.up.sk != 1
           or hc.down.sc is not w.x3 or hc.up.sc is not w.x3
           or hc.down.k % (hc.down.sk * 64) or hc.up.k % 64 for hc in hcs):
        return                      # HC geometry outside the qualified fixed-view F16 region
    b.hc_plans = _HCPlans(w, b, hcs)
