"""Unique Flash Next history ownership independent of TP8 weight placement.

All indices here are logical ranks. Physical rank ordering is supplied by the
launcher after topology qualification. The thirteenth (MTP) cache is striped
independently; it is outside the target-layer quartet balance proof.
"""

from __future__ import annotations

from dataclasses import dataclass

MIGRATION_OVERLAP_BYTES = 64 << 20  # initial specified candidate, not a measured optimum


def _int(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def compact_cache_bytes(dim: int, *, proposal_rows: int = 0, mtp: bool = False) -> int:
    """Fixed raw ring/protected carry and the selected method's bounded span."""
    _int(dim, "indexer width", 1)
    _int(proposal_rows, "proposal span rows")
    row = dim * 2 + 3 * 8
    return 8 * row + int(mtp) * (4 + proposal_rows) * row


def compact_stage_bytes(rows: int, caches: int, *, mtp: bool = False, mtp_only: bool = False) -> int:
    _int(rows, "execution rows", 1)
    _int(caches, "indexed caches", 1)
    channels = 0 if mtp_only else (caches + 7) // 8
    return rows * (channels * 128 * 2 + int(not mtp_only) * 3 * 8 + int(mtp) * (128 * 2 + 3 * 8))


@dataclass(frozen=True)
class HistoryPlacement:
    world: int
    global_kv_heads: int
    target_layers: int
    degree: int = 1
    page_tokens: int = 128

    def __post_init__(self) -> None:
        if (self.world, self.global_kv_heads) != (8, 2):
            raise ValueError("unique SM70 history currently requires TP8 and two global KV heads")
        _int(self.target_layers, "target attention layers", 1)
        if self.degree not in (1, 2, 4) or type(self.degree) is not int:
            raise ValueError("attention degree must be 1, 2, or 4")
        if type(self.page_tokens) is not int or self.page_tokens <= 0 or self.page_tokens % 4:
            raise ValueError("history pages must contain a positive multiple of four tokens")

    @property
    def group_size(self) -> int:
        return self.world // self.global_kv_heads

    def _indices(self, slot: int, layer: int, head: int = 0, page: int = 0) -> None:
        _int(slot, "session slot")
        _int(page, "logical page")
        if type(layer) is not int or not 0 <= layer <= self.target_layers:
            raise ValueError("attention layer is outside the target/MTP cache inventory")
        if type(head) is not int or not 0 <= head < self.global_kv_heads:
            raise ValueError("global KV head is outside the model inventory")

    def kv_owner(self, slot: int, layer: int, head: int, page: int = 0) -> int:
        self._indices(slot, layer, head, page)
        degree = self.group_size if layer == self.target_layers else self.degree
        base = (slot + layer) % self.group_size
        return head * self.group_size + (base ^ (page % degree))

    def indexer_owner(self, slot: int, layer: int, *, include_mtp: bool = True) -> int:
        self._indices(slot, layer)
        layers = self.target_layers + int(include_mtp)
        if layer >= layers:
            raise ValueError("MTP indexer requested for a target-only placement")
        return (slot * layers + layer) % self.world

    def page_rows(self, rows: int, page: int) -> int:
        """Logical valid extent; a partly valid page still owns a full class."""
        _int(rows, "logical rows")
        _int(page, "logical page")
        return min(self.page_tokens, max(0, rows - page * self.page_tokens))

    def physical_page_rows(self, rows: int, page: int) -> int:
        return self.page_tokens if self.page_rows(rows, page) else 0

    def local_rows(self, slot: int, layer: int, rows: int, rank: int) -> int:
        self._indices(slot, layer)
        _int(rows, "logical rows")
        if type(rank) is not int or not 0 <= rank < self.world:
            raise ValueError("owner rank is outside TP8")
        degree = self.group_size if layer == self.target_layers else self.degree
        residue = ((slot + layer) % self.group_size) ^ (rank % self.group_size)
        pages = (rows + self.page_tokens - 1) // self.page_tokens
        if residue >= degree or pages <= residue:
            return 0
        count = (pages - 1 - residue) // degree + 1
        return count * self.page_tokens

    def layer_rank_bytes(self, slot: int, layer: int, rows: int, rank: int, kv_row_bytes: int,
                         kv_dummy_bytes: int = 0, *, index_dim: int = 128,
                         ratio: int = 4, include_mtp: bool = True, proposal_rows: int = 0) -> int:
        """Exact tensor bytes, including full tail pages, dummies and partial pools.

        ``kv_row_bytes`` describes one global KV head's combined K/V row,
        including codec metadata. Per-row allocator padding must already be
        included by its codec authority. Physical tail pages retain their full
        fixed class; logical validity never changes their reserved footprint.
        """
        _int(kv_row_bytes, "KV bytes per head row", 1)
        _int(kv_dummy_bytes, "KV dummy metadata bytes")
        _int(index_dim, "indexer width", 1)
        _int(ratio, "indexer pooling ratio", 1)
        if self.page_tokens % ratio:
            raise ValueError("page boundaries must align with indexer pooling groups")
        total = self.local_rows(slot, layer, rows, rank) * kv_row_bytes + kv_dummy_bytes
        total += ((rows + self.page_tokens - 1) // self.page_tokens) * 8  # stable int64 page offsets
        if rank == self.indexer_owner(slot, layer, include_mtp=include_mtp):
            total += (rows + ratio - 1) // ratio * index_dim * 2
            total += compact_cache_bytes(index_dim, mtp=layer == self.target_layers,
                                         proposal_rows=proposal_rows if layer == self.target_layers else 0)
        return total

    def rank_bytes(self, slot: int, rows: int, rank: int, kv_row_bytes: int,
                   kv_dummy_bytes: int = 0, *, index_dim: int = 128,
                   ratio: int = 4, include_mtp: bool = True, proposal_rows: int = 0) -> int:
        return sum(self.layer_rank_bytes(slot, layer, rows, rank, kv_row_bytes, kv_dummy_bytes,
                                        index_dim=index_dim, ratio=ratio, include_mtp=include_mtp,
                                        proposal_rows=proposal_rows)
                   for layer in range(self.target_layers + int(include_mtp)))


def for_weights(weights) -> HistoryPlacement | None:
    """Explicit startup policy; other hardware retains its existing allocator."""
    mode = weights.meta.get("history_layout", "replicated")
    if mode == "replicated":
        return None
    if mode != "unique":
        raise ValueError("history layout must be replicated or unique")
    full = weights.meta.get("full")
    if (weights.meta.get("execution_provider") != "sm70" or full is None
            or weights.meta.get("world") != 8 or full.kv_heads != 2
            or weights.cfg.kv_heads != 1 or weights.meta.get("indexer_history", "gpu") != "gpu"):
        raise ValueError("unique history requires the SM70 TP8/two-head GPU-history execution plan")
    return HistoryPlacement(8, 2, sum(not layer.linear for layer in weights.layers),
                            degree=weights.meta.get("attention_degree", 1))


def native_owner_workspace_bytes(rows: int) -> dict[str, int]:
    """Exact added native tables; communication and output slabs are reused."""
    _int(rows, "native execution rows", 1)
    from tensorfold.cuda.sm70.paged_attention import DESCRIPTOR_BYTES, LAYER_BYTES, workspace_bytes

    chunk = min(rows, 256)
    descriptors, operator = chunk * DESCRIPTOR_BYTES, workspace_bytes(chunk)
    return {"descriptor_bytes": descriptors, "global_statistics_bytes": operator - descriptors,
            "lane_control_bytes": chunk * 4, "layer_control_bytes": LAYER_BYTES,
            "device_bytes": operator + chunk * 4 + LAYER_BYTES, "host_staging_bytes": descriptors}


def owner_workspace_bytes(text, rows: int, world: int = 8, *, native: bool = False) -> dict[str, int]:
    """Construction-derived owner slabs and a conservative ATen work envelope.

    ATen/native library-private allocations require the runtime's independent
    workspace reserve. This describes project-controlled tensors only. The
    temporary bound sums potentially live intermediates rather than assuming
    allocator reuse; it covers full indexer sorting, bounded KV decode tiles,
    packed preparation frames, and prefix-copy staging.
    """
    _int(rows, "execution rows", 1)
    if world != 8:
        raise ValueError("owner workspace describes the supported TP8 plan")
    def field(name, default=None):
        if isinstance(text, dict):
            value = text.get(name, default)
        else:
            value = getattr(text, name, default)
        return _int(value, name, 1)
    heads = field("heads")
    dims = field("head_dim")
    index_heads, index_dim = field("index_heads"), field("index_dim")
    budget, ratio = field("index_budget"), field("index_ratio")
    context = field("context", getattr(text, "max_position_embeddings", None) if not isinstance(text, dict)
                    else text.get("max_position_embeddings"))
    chunk, idw = min(rows, 256), budget + ratio
    persistent = (world * chunk * heads * dims * 2
                  + (world + 1) * chunk * (idw + 2) * 4
                  + (world + 1) * chunk * 4 * heads * max(dims, 2) * 4 + 4)
    native_bytes = native_owner_workspace_bytes(rows) if native else {"device_bytes": 0, "host_staging_bytes": 0}
    if native and (heads != 3 or dims != 256):
        raise ValueError("native owner workspace requires H3/D256")
    persistent += native_bytes["device_bytes"]
    prepare = rows * (2 * heads * dims + 8 * dims + 2 * (index_heads + 1) * index_dim) * 32
    pooled = (context + ratio - 1) // ratio
    # Progressive pooling gathers only groups completed by the append plus
    # their <=ratio-1 carry rows. The small CUDA normalization pad preserves
    # the old logical-output reduction mode without a full raw-history mirror.
    pool_groups = min(pooled, (rows + ratio - 1) // ratio)
    pool_work = pool_groups * ratio * index_dim * 16 + max(pool_groups, min(pooled, 16)) * index_dim * 64
    indexer = pool_work + chunk * pooled * 32
    attention = 128 * dims * 64 + 4 * heads * (128 + dims) * 32
    prefix = 128 * (2 * dims * 2 + 2 * index_dim)
    return {"persistent_bytes": persistent, "host_staging_bytes": native_bytes["host_staging_bytes"],
            "native_device_bytes": native_bytes["device_bytes"],
            "temporary_bytes": prepare + indexer + attention + prefix}
