"""Bounded batch-axis plans for the original GGUF reference-prefill arithmetic.

This provider is deliberately unselected until its full-model qualification.
The private policy is read once by the engine before admission and loading;
it is not a request-time switch. No quantized coefficients or per-dot matrix
dimensions change when more independent expert slots share one BMM launch.
"""

from dataclasses import asdict, dataclass


# Qualification harnesses may select a budget before constructing an engine.
# Ordinary callers retain the existing provider until that qualification passes.
WORKSPACE_BUDGET_BYTES = None
DECODE_BYTES = 32 << 20
BLAS_SCRATCH_BYTES = 32 << 20
POLICY = "gguf-reference-prefill-live-slots-v3"


def _positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"GGUF prefill {name} must be a positive integer")
    return value


def original_pairs(width, dims):
    """The original bounded provider's batch count, retained as its fallback."""
    _positive(width, "width")
    _positive(dims, "hidden width")
    capacity = max(1, DECODE_BYTES // (6 * width * dims))
    return min(16, 1 << (capacity.bit_length() - 1))


def _components(width, dims, rows, slots, pairs):
    decoded = 8 * pairs * width * dims
    tile = 28 * pairs * dims + 40 * pairs * width + 64 * pairs + 8 * (width + dims)
    shared = 32 * (width + dims) * 16 + rows * (width + dims) * 32 + rows * slots * dims * 4
    return decoded, tile, shared, BLAS_SCRATCH_BYTES


@dataclass(frozen=True)
class PrefillPlan:
    width: int
    dims: int
    rows: int
    slots: int
    pairs: int
    budget_bytes: int

    def __post_init__(self):
        for name in ("width", "dims", "rows", "slots", "pairs", "budget_bytes"):
            _positive(getattr(self, name), name)
        allowed = (original_pairs(self.width, self.dims), 32, 64, 128)
        if self.pairs not in allowed:
            raise ValueError("GGUF prefill batch count must be original, 32, 64 or 128")
        if self.workspace_bytes > self.budget_bytes:
            raise ValueError("GGUF prefill workspace exceeds its admitted budget")

    @property
    def decoded_replacement_bytes(self):
        # Python loop replacement can retain old gate/up/down and one newly
        # decoded FP16 bank simultaneously: four banks, not three.
        return _components(self.width, self.dims, self.rows, self.slots, self.pairs)[0]

    @property
    def tile_bytes(self):
        # Index vectors/expanded bank rows, selected input/result rows, two
        # FP32 projections, SwiGLU intermediates and selected FP16 activation.
        return _components(self.width, self.dims, self.rows, self.slots, self.pairs)[1]

    @property
    def shared_bytes(self):
        # The last routed tile remains live while the original shared expert
        # runs fixed-M32 GEMMs. Include full-row intermediates and the FP32
        # torch.where temporary. Caller-owned MoEBuffers.y is charged elsewhere.
        return _components(self.width, self.dims, self.rows, self.slots, self.pairs)[2]

    @property
    def workspace_bytes(self):
        return sum(_components(self.width, self.dims, self.rows, self.slots, self.pairs))

    def validate(self, *, width, dims, rows=None, slots=None):
        if (width, dims) != (self.width, self.dims):
            raise ValueError("GGUF prefill plan does not match the loaded expert geometry")
        if rows is not None and (type(rows) is not int or not 0 <= rows <= self.rows):
            raise ValueError("GGUF prefill rows exceed the admitted workspace plan")
        if slots is not None and (type(slots) is not int or not 0 < slots <= self.slots):
            raise ValueError("GGUF prefill slots exceed the admitted workspace plan")

    def receipt(self):
        return {**asdict(self), "policy": POLICY,
                "original_pairs": original_pairs(self.width, self.dims),
                "nominal_three_decoded_banks_bytes": 6 * self.pairs * self.width * self.dims,
                "decoded_replacement_bytes": self.decoded_replacement_bytes,
                "tile_bytes": self.tile_bytes, "shared_bytes": self.shared_bytes,
                "blas_scratch_bytes": BLAS_SCRATCH_BYTES, "workspace_bytes": self.workspace_bytes,
                "caller_output_included": False, "host_bytes": 0, "resident_bytes": 0}


def choose(*, width, dims, rows, slots, budget_bytes):
    """Prefer the largest proven batch count that fits an explicit scratch cap."""
    for name, value in (("width", width), ("hidden width", dims), ("rows", rows),
                        ("slots", slots), ("budget", budget_bytes)):
        _positive(value, name)
    for pairs in (128, 64, 32, original_pairs(width, dims)):
        if sum(_components(width, dims, rows, slots, pairs)) <= budget_bytes:
            return PrefillPlan(width, dims, rows, slots, pairs, budget_bytes)
    raise ValueError("GGUF prefill budget cannot fit the original reference provider")


def plan(text, *, rows, world):
    """Engine hook: create one immutable plan before admission, or retain default."""
    if WORKSPACE_BUDGET_BYTES is None:
        return None
    if type(world) is not int or world not in (1, 2, 4, 8):
        raise ValueError("GGUF prefill tensor-parallel world must be 1, 2, 4 or 8")
    width = _positive(text["moe_intermediate_size"], "expert width")
    shared = _positive(text["shared_expert_intermediate_size"], "shared expert width")
    if width % world or shared != width:
        raise ValueError("GGUF prefill requires equal routed/shared widths divisible by tensor parallelism")
    return choose(width=width // world, dims=text["hidden_size"], rows=rows,
                  slots=_positive(text["num_experts_per_tok"], "routed slots") + 1,
                  budget_bytes=WORKSPACE_BUDGET_BYTES)


def call_pairs(plan, rows, slots, *, width=None, dims=None):
    """Spend the admitted maximum only when a prompt has enough live slots."""
    plan.validate(width=plan.width if width is None else width,
                  dims=plan.dims if dims is None else dims, rows=rows, slots=slots)
    rounded = 1 << (max(1, rows * slots) - 1).bit_length()
    return min(plan.pairs, max(original_pairs(plan.width, plan.dims), rounded))
