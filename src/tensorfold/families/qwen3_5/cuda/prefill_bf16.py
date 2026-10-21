"""BF16 prompt inputs; projections own their required quantization and need no affine lane group sums."""

from . import glue


def add_rmsnorm(x, residual, weight, eps):
    saved, normalized, _ = glue.add_rmsnorm(x, residual, weight, eps, group_sums=False)
    return saved, normalized


def swiglu(gate, up):
    return glue.swiglu(gate, up, group_sums=False)[0]


def gated_norm(y, z, weight, eps):
    return glue.gated_norm(y, z, weight, eps, group_sums=False)[0]


def gate_mul(o, qg, *, heads, head_dim):
    return glue.gate_mul(o, qg, heads=heads, head_dim=head_dim, group_sums=False)[0]
