"""Private DSpark greedy proposal captures in their target RoundGraphs owner's pool.

Only proposal arithmetic is bucketed. Target verification, its position-keyed
sampling, accepted-prefix publication and all fallback policies remain external.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import dspark_reference as ref
from .dspark_cuda import _borrow, _record

_ENABLED = False


def graph_host_bytes(slots, block, world, capacity):
    if (any(type(value) is not int or value < 1 for value in (slots, block, world, capacity))
            or slots > 64 or block > 32 or world > 8):
        raise ValueError("DSpark graph host forecast requires admitted slots/block/world/capacity")
    buckets = max(1, (capacity - 1).bit_length() - 7)
    return slots * buckets * (block * world * 5 + block * 2 + 1) * 8


def capture_fits(owner, took):
    """Driver/private-pool growth must leave the common future service floor."""
    return (owner.room_bytes + took <= owner.budget
            and (owner.spare is None or owner.spare() >= 0))


def context_bucket(length, capacity):
    if (type(length) is not int or type(capacity) is not int
            or not 0 <= length < capacity):
        raise ValueError("DSpark proposal bucket requires an admitted committed cursor")
    return min(capacity, max(256, 1 << max(0, length - 1).bit_length()))


def policy_eligible(length, count, sampling, confidence):
    from tensorfold.cuda.sampling import validate_policy

    if type(count) is not int or count < 1 or type(length) is not int or length < 0:
        return False
    try:
        validate_policy([length + 1 + j for j in range(count)], sampling)
    except (TypeError, ValueError, OverflowError):
        # Retain the ordinary provider's full validation/error ordering.
        return False
    return confidence == 0 and (sampling is None or sampling.temperature <= 0)


def masked_gqa(q, prefix_k, prefix_v, block_k, block_v, valid):
    """FP32 bidirectional block+prefix attention, ignoring unused cache bytes.

    This deliberately has a fixed bucket-shaped FP32 reduction. It can differ
    from the eager draft's exact-length reduction; it never changes target math.
    """
    rows, heads, dim = q.shape
    kv = prefix_k.shape[1]
    prefix_k = torch.where(valid[:, None, None], prefix_k, 0)
    prefix_v = torch.where(valid[:, None, None], prefix_v, 0)
    keys, values = torch.cat((prefix_k, block_k)), torch.cat((prefix_v, block_v))
    mask = torch.cat((valid, torch.ones(rows, dtype=torch.bool, device=q.device)))
    grouped = q.float().reshape(rows, kv, heads // kv, dim)
    scores = torch.einsum("rhgd,chd->hgrc", grouped, keys.float()) * (dim ** -0.5)
    scores.masked_fill_(~mask, -torch.inf)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hgrc,chd->rhgd", probabilities, values.float()).reshape(rows, heads * dim).half()


def bucketed_hidden(provider, ws, embeds):
    c, t = provider.config, provider.weights.tensors
    positions = ws.length + ws.block_offsets
    valid = ws.prefix_offsets < ws.length
    hidden, residual = embeds, None
    for layer in range(c.layers):
        base = f"layers.{layer}."
        x, residual = ref.add_norm(hidden, residual, t[base + "input_layernorm.weight"], c.eps)
        ab = base + "self_attn."
        q = provider._linear(x, ab + "q_proj.weight").view(-1, provider.local["heads"], c.head_dim)
        k = provider._linear(x, ab + "k_proj.weight").view(-1, provider.local["kv_heads"], c.head_dim)
        v = provider._linear(x, ab + "v_proj.weight").view(-1, provider.local["kv_heads"], c.head_dim)
        q = ref.rope(ref.rms(q, t[ab + "q_norm.weight"], c.eps), positions, c.theta)
        k = ref.rope(ref.rms(k, t[ab + "k_norm.weight"], c.eps), positions, c.theta)
        attended = masked_gqa(q, ws.keys[layer, :ws.bucket], ws.values[layer, :ws.bucket], k, v, valid)
        hidden = provider._linear(attended, ab + "o_proj.weight", reduce=True)
        x, residual = ref.add_norm(hidden, residual, t[base + "post_attention_layernorm.weight"], c.eps)
        gate = provider._linear(x, base + "mlp.gate_proj.weight")
        up = provider._linear(x, base + "mlp.up_proj.weight")
        hidden = provider._linear((F.silu(gate.float()) * up.float()).half(), base + "mlp.down_proj.weight", reduce=True)
    return ref.add_norm(hidden, residual, t["norm.weight"], c.eps)[0]


def latch(ws, *values):
    for value in values:
        bad = (~torch.isfinite(value)).any().to(torch.int32).reshape(1)
        ws.status[:1].bitwise_or_(bad)


def proposal_body(provider, ws):
    """GPU-only operations, fixed safe collective order, no target publication."""
    ws.status.zero_()
    ws.status[1].fill_(-2)
    embeds = provider.embed(ws.inputs)
    latch(ws, embeds)
    hidden = bucketed_hidden(provider, ws, embeds)
    latch(ws, hidden)
    base = provider.head(hidden, provider.shard).half()
    latch(ws, base)
    ws.previous.copy_(ws.inputs[:1])
    for step in range(ws.block):
        # Every previous token comes from a validated host bonus, or the native
        # global winner's bounded safe-input word, including invalid score paths.
        prev = provider.weights.tensors["markov_head.markov_w1.weight"].index_select(0, ws.previous)
        bias = provider._linear(prev, "markov_head.markov_w2.weight")
        torch.add(base[step:step + 1].float(), bias.float(), out=ws.logits)
        ws.corrected[step:step + 1].copy_(ws.logits)
        ws.logits.copy_(ws.corrected[step:step + 1])  # existing FP16 corrected-logit storage boundary
        latch(ws, ws.logits)
        ws.native.local(ws.logits, ws.ids, ws.norm, ws.status, ws.send)
        frame = ws.frames[step]
        if provider.world == 1:
            frame[0].copy_(ws.send)
        else:
            provider.comm.all_gather(ws.send, frame.view(-1))
        ws.native.global_winner(frame, ws.tokens[step:step + 1], ws.safe_input, provider.config.vocab)
        ws.previous.copy_(ws.safe_input)  # conversion of an already bounded device ID
    return ws.tokens


class ProposalWorkspace:
    """Stable input/packet owners; graph-pool intermediates never escape a replay."""
    def __init__(self, provider, state, block, bucket):
        self.state, self.keys, self.values = state, state.keys, state.values
        self.block, self.bucket = block, bucket
        self.binding = self.key(provider, state, block, bucket)
        self.owned = {}
        self.native = None
        self.closed = self.busy = False
        self.readbacks = self.steps = self.calls = 0
        self._device, self._provider = provider.device, provider
        self.split = None
        self.pieces = None

    @staticmethod
    def key(provider, state, block, bucket):
        return (id(state.target), state.target.version, provider.identity,
                state.keys.data_ptr(), state.values.data_ptr(), state.capacity, block, bucket)

    @staticmethod
    def allocation_sizes(width, block, bucket, world):
        # Immutable full-global ID map; static inputs/offsets; corrected Half
        # oracle/output rows; FP32 one-row local selection; native frame words.
        return (width * 8, block * 8, 8, block * 8, bucket * 8, 8, 4,
                block * width * 2, width * 4, 4, 8, 5 * 8, (block * world * 5 + block) * 8)

    @staticmethod
    def device_bytes(width, block, bucket, world):
        return sum(ProposalWorkspace.allocation_sizes(width, block, bucket, world))

    def allocate(self):
        p, device = self._provider, self._device
        b, width = self.block, p.shard.width
        from tensorfold.cuda.sm70.greedy_native import extension
        self.native = extension()

        def put(name, value):
            self.owned[name] = value
            setattr(self, name, value)
            return value

        put("ids", torch.arange(p.shard.start, p.shard.stop, dtype=torch.int64, device=device))
        put("inputs", torch.full((b,), p.config.mask, dtype=torch.int64, device=device))
        put("length", torch.zeros(1, dtype=torch.int64, device=device))
        put("block_offsets", torch.arange(b, dtype=torch.int64, device=device))
        put("prefix_offsets", torch.arange(self.bucket, dtype=torch.int64, device=device))
        put("previous", torch.zeros(1, dtype=torch.int64, device=device))
        put("safe_input", torch.zeros(1, dtype=torch.int32, device=device))
        put("corrected", torch.empty((b, width), dtype=torch.float16, device=device))
        put("logits", torch.empty((1, width), dtype=torch.float32, device=device))
        put("norm", torch.zeros(1, dtype=torch.float32, device=device))
        put("status", torch.zeros(2, dtype=torch.int32, device=device))
        put("send", torch.empty(5, dtype=torch.int64, device=device))
        put("storage", torch.empty(b * p.world * 5 + b, dtype=torch.int64, device=device))
        self.frames = self.storage[:b * p.world * 5].view(b, p.world, 5)
        self.tokens = self.storage[-b:]
        self.host = torch.empty_like(self.storage, device="cpu", pin_memory=True)
        self.host_inputs = torch.empty(b, dtype=torch.int64, pin_memory=True)
        self.host_length = torch.empty(1, dtype=torch.int64, pin_memory=True)

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.owned.values())

    @property
    def host_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.host, self.host_inputs, self.host_length))

    def prepare(self, bonus):
        p, state = self._provider, self.state
        p._check(state)
        if (self.closed or self.busy or state.pending is not None
                or self.binding != self.key(p, state, self.block, self.bucket)
                or state.length > self.bucket or state.length + self.block + 1 > state.capacity
                or type(bonus) is not int or not 0 <= bonus < p.config.vocab):
            raise ValueError("DSpark proposal capture lost its admitted committed slot or bonus")
        self.host_inputs.fill_(p.config.mask)
        self.host_inputs[0] = bonus
        self.host_length[0] = state.length
        self.inputs.copy_(self.host_inputs, non_blocking=True)
        self.length.copy_(self.host_length, non_blocking=True)

    def finish(self):
        from .device_greedy import recorded_token
        self.host.copy_(self.storage, non_blocking=True)
        torch.cuda.current_stream(self._device).synchronize()
        self.readbacks += 1
        frame = self.host[:self.block * self._provider.world * 5].view(self.block, self._provider.world, 5).numpy()
        tokens = self.host[-self.block:].tolist()
        for rows, token in zip(frame, tokens, strict=True):
            if rows[:, 3].any():
                raise ValueError("DSpark nonfinite captured proposal; no draft IDs published")
            if not 0 <= token < self._provider.config.vocab:
                raise ValueError("DSpark captured proposal global ID is outside the full target vocabulary")
            recorded_token(rows, token)
        self.steps += self.block
        self.calls += 1
        return tokens

    def close(self):
        if self.busy:
            raise ValueError("DSpark proposal graph is still replaying")
        if self.pieces is not None:
            for graph in self.pieces.graphs:
                graph.reset()
        self.pieces = self.keys = self.values = self.state = None
        self.owned.clear()
        for name in ("ids", "inputs", "length", "block_offsets", "prefix_offsets", "previous", "safe_input",
                     "corrected", "logits", "norm", "status", "send", "storage", "frames", "tokens",
                     "host", "host_inputs", "host_length"):
            setattr(self, name, None)
        self.closed = True


class DSparkProposalGraphs:
    """Optional per-slot proposal keys, charged to the existing bound graph owner."""
    def __init__(self, runtime, bound):
        self.runtime, self.bound, self.owner = runtime, bound, bound.owner
        self.provider = runtime.provider
        self.block = min(bound.dec.depth, runtime.config.block)
        self.slots = {}
        self.declined = set()
        from .lane_admission import prepare
        self.allocator = (getattr(bound, "lane_allocator", None) or prepare(torch)
                          if self.provider.device.type == "cuda" else None)
        self.closed = False
        self.hits = self.misses = self.fallbacks = 0

    def _slot(self, state, *, capture):
        bucket = context_bucket(state.length, state.capacity)
        key = ProposalWorkspace.key(self.provider, state, self.block, bucket)
        result = self.slots.get(key)
        if result is not None or not capture or key in self.declined:
            return result
        root, bound = self.owner, self.bound
        required = ProposalWorkspace.device_bytes(self.provider.shard.width, self.block, bucket, self.provider.world)
        sizes = ProposalWorkspace.allocation_sizes(self.provider.shard.width, self.block, bucket, self.provider.world)
        bounds = self.allocator.bounds(sizes) if self.allocator is not None else None
        remaining = None if root.spare is None else root.spare()
        allowed = (bounds is not None and self.allocator.unchanged(torch)
                   and root.room_bytes + bounds[0] <= root.budget and root._has_room()
                   and (remaining is None or remaining >= max(bounds)))
        if not bound._agree(allowed):
            self.declined.add(key)
            return None
        result = self.slots[key] = ProposalWorkspace(self.provider, state, self.block, bucket)
        before = torch.cuda.memory_allocated(self.provider.device)
        result.allocate()
        bound._resident(max(0, torch.cuda.memory_allocated(self.provider.device) - before))
        assert result.nbytes == required
        result.prepare(0)
        def run():
            return proposal_body(self.provider, result)
        run()  # load all exact kernels/collective geometry before capture
        torch.cuda.synchronize(self.provider.device)
        from .multi_graphs import free_bytes
        before = free_bytes()
        pieces = root._pieces(run, result, split=False)
        result.pieces = pieces  # retain graph owners before any later failure
        torch.cuda.synchronize(self.provider.device)
        took = max(0, before - free_bytes())
        if not bound._agree(capture_fits(root, took)):
            raise MemoryError("DSpark proposal capture exceeded the common graph budget or future service memory")
        root.bytes += took
        root.room_bytes += took
        root.measured.append({"key": ["dspark", repr(key)], "pieces": len(pieces.graphs), "bytes": took})
        self.misses += 1
        return result

    def warm(self, states):
        if self.closed or self.block < 1:
            return
        for target in states:
            self._slot(self.runtime._state(target), capture=True)

    def draft(self, target, bonus, count, sampling, confidence):
        if self.closed or count != self.block or not policy_eligible(target.pos, count, sampling, confidence):
            self.fallbacks += 1
            return None
        state = self.runtime._state(target)
        if state.length + self.block + 1 > state.capacity:
            self.fallbacks += 1
            return None
        previous = self.owner.room_bytes
        try:
            ws = self._slot(state, capture=True)
        except BaseException as error:
            self.owner.retain_failure(error)
            self.bound.dec._failed_graphs, self.bound.dec._memory_error = self.owner, error
            raise
        if self.owner.room_bytes != previous:
            self.bound._account(previous)
        if ws is None:
            self.fallbacks += 1
            return None
        if ws.pieces is None:
            raise RuntimeError("DSpark proposal slot has an incomplete captured owner")
        ws.prepare(bonus)
        ws.busy = True
        try:
            _borrow((ws.keys, ws.values))
            for graph in ws.pieces.graphs:
                graph.replay()
            result = ws.finish()
            self.hits += 1
            return result
        finally:
            _record(state.ready, self.provider.device)
            ws.busy = False

    def invalidate(self, target):
        torch.cuda.synchronize(self.provider.device)
        for key, ws in tuple(self.slots.items()):
            if ws.state.target is target:
                ws.close()
                del self.slots[key]
        self.declined = {key for key in self.declined if key[0] != id(target)}

    def close(self):
        if self.closed:
            return
        torch.cuda.synchronize(self.provider.device)
        for key, ws in tuple(self.slots.items()):
            ws.close()
            del self.slots[key]
        self.closed = True
        self.declined.clear()

    def record(self):
        return {"keys": len(self.slots), "hits": self.hits, "misses": self.misses, "fallbacks": self.fallbacks,
                "declined_keys": len(self.declined),
                "device_bytes": sum(ws.nbytes for ws in self.slots.values()),
                "host_bytes": sum(ws.host_bytes for ws in self.slots.values()),
                "calls": sum(ws.calls for ws in self.slots.values()),
                "steps": sum(ws.steps for ws in self.slots.values()),
                "readbacks": sum(ws.readbacks for ws in self.slots.values())}
