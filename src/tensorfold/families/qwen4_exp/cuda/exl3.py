"""Flash Next from an EXL3 pack into the MLX path's ``Weights``: trellis matrices, fp16 tensors, routed experts at their own widths."""

from __future__ import annotations

from contextlib import nullcontext

import time
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.exl3 import experts as x3experts
from tensorfold.cuda.exl3 import format as fmt

from .exl3_mm import Scratch, f16, stack, x3
from .exl3_pack import _DT, NgramTable, Pack, is_exl3
from ..rope import RopeParameters

PREFILL_ROWS = 2048       # the prompt buffers' rows (``decode.PREFILL_ROWS``): the n-gram staging holds as many

__all__ = ["is_exl3", "load"]


def expert_table(pk: Pack, prefix: str, count: int, shared: str, device) -> x3experts.Exl3RoutedExperts:
    """Experts ``prefix.{0..count-1}`` and the shared expert (as expert ``count``), trellises read in large runs into one buffer."""

    names = [f"{prefix}.{e}" for e in range(count)] + [shared]
    projs = ("gate_proj", "up_proj", "down_proj")
    codebooks = {pk.codebook(f"{nm}.{p}") for nm in names for p in projs}
    if len(codebooks) > 1:
        raise ValueError(f"{prefix}: the experts mix EXL3 codebooks ({', '.join(sorted(codebooks))}); the grouped "
                         "expert kernel takes one codebook a layer")
    parts = {nm + "." + p: (("suh" if pk.has(f"{nm}.{p}.suh") else "su"), ("svh" if pk.has(f"{nm}.{p}.svh") else "sv"))
             for nm in names for p in projs}
    entries = {f"{m}.{part}": pk.entry(f"{m}.{part}") for m, (i, o) in parts.items() for part in ("trellis", i, o)}
    place, total = {}, 0
    for k in entries:
        if k.endswith(".trellis"):
            place[k] = total
            total += -(-(entries[k][2] - entries[k][1]) // 256) * 256
    big = torch.empty((total,), dtype=torch.uint8, device=device)
    small: dict[str, torch.Tensor] = {}
    by_file: dict[str, list[str]] = {}
    for key, (file, *_rest) in entries.items():
        by_file.setdefault(file, []).append(key)
    for file, keys in by_file.items():
        keys.sort(key=lambda k: entries[k][1])
        run: list[str] = []

        def flush(run: list[str]) -> None:
            if run:
                b0, b1 = entries[run[0]][1], max(entries[k][2] for k in run)
                host = pk.read(file, b0, b1)
                # Validate/expand scales before these bytes reach the device.
                # Every logical scale table is transferred only after all runs
                # have validated; no corrupt payload becomes native metadata.
                for k in run:
                    if not k.endswith(".trellis"):
                        _, b, e, dtype, shape = entries[k]
                        value = host[b - b0:e - b0].clone().view(_DT[dtype]).reshape(shape)
                        if k.endswith((".su", ".sv")):
                            value = torch.from_numpy(fmt.unpack_signs(value.numpy()))
                        small[k] = x3experts.validate_scale_payload(value, k)
                dev = host.to(device)
                for k in run:
                    _, b, e, dtype, shape = entries[k]
                    if k.endswith(".trellis"):
                        big[place[k]:place[k] + (e - b)].copy_(dev[b - b0:e - b0])
                del dev, host

        for k in keys:                     # a run spans at most ~2 GB and skips at most 16 MB of other tensors
            if run and (entries[k][2] - entries[run[0]][1] > (2 << 30)
                        or entries[k][1] - max(entries[j][2] for j in run[-4:]) > (16 << 20)):
                flush(run)
                run = []
            run.append(k)
        flush(run)

    def trellis(k: str) -> torch.Tensor:
        _, b, e, dtype, shape = entries[k]
        if dtype != "I16":
            raise ValueError(f"{k}: trellis dtype {dtype}")
        return big[place[k]:place[k] + (e - b)].view(torch.int16).view(shape)

    def scales(m: str, part: str) -> torch.Tensor:
        return small[f"{m}.{part}"].to(device)

    lists = {p: [(trellis(f"{nm}.{p}.trellis"), scales(f"{nm}.{p}", parts[f"{nm}.{p}"][0]),
                  scales(f"{nm}.{p}", parts[f"{nm}.{p}"][1])) for nm in names] for p in projs}
    ex = x3experts.prepare(lists["gate_proj"], lists["up_proj"], lists["down_proj"], codebooks.pop(), device=device)
    ex.keep.append(big)
    return ex


def centred_offset(pk: Pack, names: list[str]) -> float:
    """1.0 when the pack stores the centred norms as gamma - 1 (every EXL3 pack seen), 0.0 when as gamma."""

    means = np.array([float(pk.get(n).float().mean()) for n in names if pk.has(n)])
    if not len(means):
        return 1.0
    around_zero = (means > 0.5).mean() <= 0.1 and -0.5 <= float(np.median(means)) <= 0.25
    around_one = (means > 0.5).mean() >= 0.9 and 0.75 <= float(np.median(means)) <= 1.5
    if around_zero == around_one:
        raise ValueError(f"cannot tell how the pack stores its norm weights (median mean {np.median(means):.3f})")
    return 1.0 if around_zero else 0.0


def requant_rows(head, ids: torch.Tensor, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The head's rows ``ids`` decoded through its own EXL3 linear, as MLX 4-bit groups of 32: the draft head (drafts only)."""

    from .exl3_mm import ROWS

    k = head.k
    rows = torch.empty((len(ids), k), dtype=torch.float32, device=device)
    eye = torch.eye(ROWS, dtype=torch.bfloat16, device=device)
    out = torch.empty((ROWS, head.n), dtype=torch.float32, device=device)
    for k0 in range(0, k, ROWS):
        x = torch.zeros((ROWS, k), dtype=torch.bfloat16, device=device)
        x[:, k0:k0 + ROWS] = eye
        head(x, out)
        rows[:, k0:k0 + ROWS] = out[:, ids].t()
    g = rows.view(len(ids), k // 32, 32)
    lo, hi = g.amin(dim=-1), g.amax(dim=-1)
    scale = ((hi - lo) / 15).clamp_min(1e-8).to(torch.bfloat16)
    bias = lo.to(torch.bfloat16)
    q = torch.round((g - bias.float()[..., None]) / scale.float()[..., None]).clamp(0, 15).to(torch.int64)
    words = (q.view(len(ids), k // 8, 8) << (torch.arange(8, device=device, dtype=torch.int64) * 4)).sum(dim=-1)
    words = words & 0xFFFFFFFF
    words = torch.where(words >= 2 ** 31, words - 2 ** 32, words).to(torch.int32)
    return words.contiguous(), scale.contiguous(), bias.contiguous()


def load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True, tp: tuple[int, int] | None = None,
         draft_vocab: int | str | None = None, table_reads: list | None = None,
         rope: RopeParameters | None = None, vram_experts: float | str | None = None, _ram_layout=None):
    """Load native EXL3 weights, optionally retaining original expert streams on CPU."""

    from .weight_types import Config, draft_token_ids

    cfg = Config.read(model_dir, rope=rope)
    ids = draft_token_ids(draft_vocab, cfg.vocab)
    cache = None
    if vram_experts is not None:
        from ..ram_experts import check, layout
        from tensorfold.cuda.exl3.host_experts import Exl3HostExpertCache

        check(model_dir, vram_experts, tp=tp[1] if tp is not None else 1)
        plan = layout(model_dir, vram_experts, mtp=mtp) if _ram_layout is None else _ram_layout
        # Keep only one bootstrap cell while the immutable host authority and
        # fixed device weights load. The final arena replaces it once, before
        # any lease, so numeric budgets do not require two full pools at once.
        cache = Exl3HostExpertCache(plan.entry_bytes, plan.entry_bytes, device)
    try:
        weights = _load(model_dir, device, mtp=mtp, tp=tp, _config=cfg, _draft_ids=ids,
                        table_reads=table_reads, rope=rope, expert_cache=cache)
        if cache is not None and not plan.automatic:
            # _load has sealed every registration, drained its read owner and
            # released Pack/load temporaries. Auto defers this same transition
            # until all decoder buffers and future service bounds are known.
            cache.configure_before_use(plan.gpu_bytes)
        return weights
    except BaseException as primary:
        if cache is not None:
            try:
                cache.close()
            except BaseException as cleanup:
                BaseException.add_note(primary, "EXL3 expert cache cleanup also failed; its owner remains retained")
                raise primary from cleanup
        raise


def _load(model_dir: str | Path, device: str = "cuda", *, mtp: bool = True, tp: tuple[int, int] | None = None,
          _config: object, _draft_ids: object, table_reads: list | None = None,
          rope: RopeParameters | None = None, expert_cache=None):
    from .qmm import make_q4
    from tensorfold.cuda.direct_read import in_background

    from .weights import GDNW, HC, AttnW, LayerW, MoEW, MTPW, PLEW, Weights

    if tp is not None and tp[1] > 1:
        raise ValueError("EXL3 packs of Flash Next run on one GPU; two ranks read the MLX checkpoint")
    model_dir = Path(model_dir)
    cfg, ids = _config, _draft_ids
    inv = cfg.rope.inverse_frequencies(torch)
    pk = Pack(model_dir)
    sc = Scratch(cfg.top_k + 1)
    T = "model.language_model."
    t0 = time.time()
    offset = centred_offset(pk, [f"{T}layers.{i}.attn_hyper_connection.hc_norm.weight" for i in range(cfg.layers)])
    next_expert_layer = 0

    def plain(name: str) -> torch.Tensor:
        return pk.get(name).to(device)

    def centred(name: str) -> torch.Tensor:
        return (pk.get(name).float() + offset).to(device).contiguous()

    def hc(name: str, inject: bool) -> HC:
        rows = [pk.get(name + ".input_mix_weight_down.weight")]
        if inject:
            rows.append(pk.get(name + ".block_inject_weight.weight"))
        down, up = f16(sc, rows, device), f16(sc, [pk.get(name + ".input_mix_weight_up.weight")], device)
        return HC(down, up, centred(name + ".hc_norm.weight"), inject, down, up)

    def moe(name: str) -> MoEW:
        nonlocal next_expert_layer
        router = torch.cat([pk.get(name + ".gate.weight").to(torch.bfloat16),
                            pk.get(name + ".shared_expert_gate.weight").to(torch.bfloat16)]).to(device).contiguous()
        if expert_cache is not None:
            from tensorfold.cuda.exl3.host_experts import load_cached

            experts = load_cached(pk, name + ".experts", cfg.experts, name + ".shared_expert",
                                  expert_cache, next_expert_layer, device, reads=read_session)
            next_expert_layer += 1
            return MoEW(router, experts)
        return MoEW(router, expert_table(pk, name + ".experts", cfg.experts, name + ".shared_expert", device))

    def attention(name: str) -> AttnW:
        proj = stack(sc, [x3(sc, pk, name + p, device) for p in (".q_proj", ".k_proj", ".v_proj",
                                                                  ".indexer.index_qk_proj")])
        return AttnW(proj, centred(name + ".q_norm.weight"), centred(name + ".k_norm.weight"),
                     centred(name + ".indexer.q_layernorm.weight"), centred(name + ".indexer.k_layernorm.weight"),
                     x3(sc, pk, name + ".o_proj", device))

    def gdn(name: str) -> GDNW:
        proj = stack(sc, [x3(sc, pk, name + ".in_proj_qkv", device), x3(sc, pk, name + ".in_proj_z", device),
                          f16(sc, [pk.get(name + ".in_proj_b.weight"), pk.get(name + ".in_proj_a.weight")], device)])
        conv = plain(name + ".conv1d.weight").reshape(cfg.conv_dim, cfg.conv_kernel).to(torch.bfloat16).contiguous()
        return GDNW(proj, conv, plain(name + ".A_log").float().contiguous(),
                    plain(name + ".dt_bias").float().contiguous(),
                    plain(name + ".norm.weight").to(torch.bfloat16).contiguous(),
                    x3(sc, pk, name + ".out_proj", device))

    def ple_layer(name: str, ple_index: int) -> PLEW:
        ngram = cfg.ngram(ple_index)
        table = NgramTable(pk, name + ".ple_embedding.ngram_embedding.", cfg.ngram_shards, device)
        ngram.check(table.multipliers, table.head_offsets, table.head_sizes)
        if table.rows != ngram.rows or table.dh != ngram.dims:
            raise ValueError(f"n-gram tables of {table.rows} rows of {table.dh} values; the config gives "
                             f"{ngram.rows} of {ngram.dims}")
        if table_reads is not None:                   # its pages come in while the weights load
            in_background(table.prefetch, table_reads)    # the caller waits for it (``wait_all``)
        conv = plain(name + ".conv1d.weight").reshape(cfg.streams * cfg.hidden, cfg.ple_kernel).to(torch.bfloat16)
        return PLEW(table, f16(sc, [pk.get(name + ".key_proj.weight")], device),
                    f16(sc, [pk.get(name + ".value_proj.weight")], device), centred(name + ".norm_key.weight"),
                    centred(name + ".norm_query.weight"), centred(name + ".norm_conv.weight"), conv.contiguous(), ngram)

    def layer(i: int, base: str, kind: str, with_ple: bool) -> LayerW:
        linear = kind == "linear"
        entry = LayerW(i, linear, hc(base + ".attn_hyper_connection", True), hc(base + ".mlp_hyper_connection", True),
                       gdn(base + ".linear_attn") if linear else None,
                       None if linear else attention(base + ".self_attn"), moe(base + ".mlp"))
        if with_ple and i in cfg.ple_layers:
            entry.ple = ple_layer(base + ".ple", cfg.ple_layers.index(i))
        return entry

    if expert_cache is not None:
        from tensorfold.cuda.exl3.host_experts import CompactReadSession

        owner = CompactReadSession(pk)
    else:
        owner = nullcontext(None)
    with owner as read_session:
        embed = plain(T + "embed_tokens.weight")
        if embed.dtype not in (torch.bfloat16, torch.float16):
            embed = embed.to(torch.bfloat16)
        loaded = []
        for i in range(cfg.layers):
            loaded.append(layer(i, f"{T}layers.{i}", cfg.layer_types[i], True))
            pk.release()
            if i % 8 == 7:                            # each release waits for the device; a layer leaves few temporaries
                torch.cuda.empty_cache()
        mixer = hc(T + "hyper_connection_mixer", False)
        head = x3(sc, pk, "lm_head", device, head=True)
        w = Weights(cfg, (embed.contiguous(),), loaded, mixer, head, inv.to(device), around_one=True)
        w.meta.update(rank=0, world=1, vocab_offset=0, full=cfg, centred_offset=offset)
        w.meta["rope"] = cfg.rope.metadata()
        if mtp and pk.has("mtp.fc_embedding.trellis"):
            w.mtp = MTPW(centred("mtp.pre_fc_norm_embedding.weight"), centred("mtp.pre_fc_norm_hidden.weight"),
                         x3(sc, pk, "mtp.fc_embedding", device), x3(sc, pk, "mtp.fc_hidden", device),
                         layer(-1, "mtp.layers.0", "attention", False), hc("mtp.hyper_connection_mixer", False))
        if expert_cache is not None:
            expert_cache.finish_loading()
            w.meta["expert_cache"] = expert_cache
        ple = next((lay.ple for lay in loaded if lay.ple is not None), None)
        sc.allocate(device, experts=loaded[0].moe.experts, rows=PREFILL_ROWS,
                    ple_words=ple.table.words_per_row if ple else 0, ple_heads=ple.ngram.heads if ple else 0,
                    ple_dim=cfg.ple_dim)
        if expert_cache is not None:
            from tensorfold.cuda.exl3.host_experts import WaveScratch

            sc.moe.host_waves = WaveScratch(sc.moe.rows, sc.moe.slots, device)
        w.x3 = sc
        if ids is not None and w.mtp is not None:
            w.draft_ids = torch.from_numpy(ids).to(device)
            w.draft_head = make_q4(*requant_rows(head, w.draft_ids, device))
        pk.release()
        torch.cuda.empty_cache()
    w.meta["load_seconds"] = time.time() - t0
    return w
