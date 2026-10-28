"""Independent descriptor ABI oracle and minimal real-State metadata fixture."""
from types import SimpleNamespace as NS
import torch


def weights(device="cpu", *, mtp=True):
    cfg = NS(hidden=512, streams=4, conv_kernel=4, conv_dim=1024, nk=2, nv=4, dk=128, dv=128,
             kv_heads=1, head_dim=256, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3,
             ple_layers=[], eos=(0,), heads=3, index_heads=4, index_budget=2048, eps=1e-6,
             mrope_section=(1, 1, 1), rope_attention_factor=1.0, execution_dtype="float16")
    return NS(cfg=cfg, device=torch.device(device), layers=[NS(index=i, linear=False) for i in range(12)],
              mtp=NS(layer=NS(index=12, linear=False)) if mtp else None, positions=None,
              meta={"world": 8, "rank": 0, "full": NS(kv_heads=2), "execution_provider": "sm70",
                    "execution_dtype": "float16", "history_layout": "unique", "attention_degree": 1},
              comm=NS(world=8, rank=0), inv_freq=torch.tensor([1.0], device=device))


def original_words(owner, segments, layer, base, count, bucket, *, mtp=False):
    # Independent field-by-field ABI oracle: does not call either optimized
    # builder or the extracted production word helper.
    table = torch.zeros((bucket, 16), dtype=torch.int64)
    lanes = torch.zeros(bucket, dtype=torch.int32)
    lane = 0
    for state, first, last in segments:
        begin, end = max(base, first), min(base + count, last)
        if begin >= end:
            continue
        ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
        cache = state.mtp_kc if mtp else state.kc[ai]
        lanes[lane] = last - first
        for row in range(begin, end):
            fields = {
                0: cache.k.data_ptr(), 1: cache.v.data_ptr(), 2: cache.ks.data_ptr(), 3: cache.vs.data_ptr(),
                4: cache.page_offsets.data_ptr(), 5: state.history_generations.data_ptr(),
                6: state.kv_status.data_ptr(), 7: (state.mtp_pos if mtp else state.pos_dev).data_ptr(),
                8: owner.lane_rows.data_ptr() + 4 * lane, 9: state.slot_generation,
                10: state.history_generation, 11: state.capacity, 12: cache.capacity,
                13: cache.page_offsets.numel(), 14: row - first,
                15: state.history_placement.indexer_owner(state.history_slot, ai, include_mtp=hasattr(state, "mtp_kc"))}
            for field, value in fields.items():
                table[row - base, field] = value
        lane += 1
    return table, lanes

