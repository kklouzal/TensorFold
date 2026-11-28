"""DeepSeek-V4.1-Flash on eight V100s with TP8 attention/dense and selectable expert tensor parallelism.

Rank 0 drives; ranks 1-7 run ``follow``. Every model operation is announced to the followers through the
same-host control pipes as one small frame before rank 0 issues its own identical kernels and collectives
(the Gemma 4 SM70 provider's protocol), so all ranks submit the same NCCL sequence. Only rank 0 samples.

Per-stream state lives in slots (patch 85's SlotTable semantics): KV pages, the window rings, the Engram n-gram
history and the ratio-2 compressor partials. A slot is reset when a stream takes it; the last slot is scratch.

Serving (``serve=True``, the family's ``cuda_engine``): rank 0 owns the shared CUDA Scheduler and this provider's
``Decoder`` (the Gemma 4 SM70 Decoder's round structure): one batched decode forward over every decoding stream a
round, and between rounds the next prompt chunk of a filling stream; while streams decode, a chunk runs in groups
of units (each layer's attention, then its experts) cut every round from measured costs so a decoding stream waits
about ``DECODE_GAP_MS`` (175 ms), not a chunk.
"""

from __future__ import annotations

import array
import os
import struct
import time
from pathlib import Path
from typing import Any, Callable

from .prefix import OPS as PREFIX_OPS, PrefixOwner, chunk_end
from .schedule import RoundCosts, UnitCosts

# The Gemma 4 SM70 operation codes (the prefix owner announces them; prefix.SlotState runs the prefix ones); OP_STEP
# (this provider's batched forward) takes Gemma's OP_FORWARD code.
from tensorfold.families.gemma4.sm70.ops import OP_ATTACH, OP_DETACH, OP_FORWARD as OP_STEP, OP_GROW, OP_LAYERS, OP_STOP
_HEAD = struct.Struct("<qqqqq")
PIPE_TIMEOUT = 600.0
# An idle follower with outstanding RotorQuant work must surface a device trap
# even while the leader is blocked on a cross-rank GPU operation.
ROTOR_ERROR_POLL_S = 1.0
F_SCORE, F_LAST = 1 << 20, 1 << 21  # OP_STEP d flags (low bits: the stop layer): score every row; and the last row
# graph decode flags: return the compact greedy packet (``round_sampling``); and also gather the full rows
F_COMPACT, F_FULL = 1 << 22, 1 << 23
F_IMAGE = 1 << 24          # a prompt chunk with image rows: their features are every rank's staged tower rows
F_AHEAD = 1 << 25          # run-ahead round: token ids from the previous replay's device greedy tokens (graphs.py)
EXPERT_SAMPLE_SHIFT = 26   # OP_STEP: 0 off, 1 prefill, 2 decode, visible only to the rank-zero observer
F_W576_FP32 = 1 << 28      # experimental continuation math, independent of output packet format
F_DRAFT_TAPS = 1 << 29     # OP_STEP appends the first local row to tap in an ordinary prompt forward
F_SAMPLE = 1 << 30         # OP_STEP appends every row's device_sampling words (a compact decode graph round)
# The TP8 vision tower's operations (``OP_TOWER``, kind in a; every rank runs its shard, ``DS41Engine._tower``)
OP_TOWER = 32
OP_EXPERT_PLACEMENT = 33
OP_DRAFT_VERIFY_BATCH, OP_DRAFT_COMMIT_BATCH = 35, 36
_EXPERT_EPOCH = struct.Struct("<q")
T_BEGIN, T_RUN, T_SPAN, T_STAGE, T_FREE = range(5)
SCORE_TOP = 20
MAX_TOKENS = 8192          # tokens per forward (the friend's --max-num-batched-tokens); also the prompt chunk
MAX_STREAMS = 32
RESERVE = 3 << 30          # bytes left free per rank for prompt transients, after the KV pool
H64_TEXT_RESERVE = 5 << 29
GRAPH_EACH_BYTES = 40 << 20 # decode graph memory per captured bucket (sizing bound; measured ~30 MiB)
GRAPH_BYTES = 128 << 20     # + the graphs' shared torch pool (measured ~40 MiB)
DRAFT_BUCKETS = (1, 2, 3)
# Experimental compact ETP4 allowance for these exact, ordered 128K H64 keys.
# Keep the catalogue independent of shape_keys so edits use the generic allowance
# until requalified. See docs/recipes/deepseek-v41-v100.md.
H64_128K_GRAPH_KEYS = (
    *((bucket, 131_072) for bucket in range(1, 4)),
    *((bucket, context) for bucket in (*range(4, 17), 24, 32) for context in (8_192, 32_768, 131_072)),
)
H64_128K_GRAPH_BYTES = 1536 << 20
TRACE = bool(os.environ.get("TENSORFOLD_DS41_SCHED_TRACE"))   # diagnostic: the Decoder's group plans and costs
TRACE_DRAFT = os.environ.get("TENSORFOLD_DS41_DRAFT_TRACE") == "1"


def draft_blockers(s) -> tuple[str, ...]:
    """Request-local reasons D3 cannot seed a draft ring from this prompt."""
    from .dspark_head import WINDOW
    from .dspark_serving import sampled_draftable

    return tuple(reason for blocked, reason in (
        (not s.draft, "draft_disabled"),
        (s.vision is not None, "image_input"),
        (s.background, "background"),
        (bool(s.owed), "replay"),
        (s.token_limit is not None, "token_limit"),
        (s.forced is not None, "forced_tokens"),
        (s.constraint is not None and not s.stop_eos, "grammar_without_eos"),
        (s.constraint is not None and not callable(getattr(s.constraint, "window", None)), "grammar_without_window"),
        (s.history is not None, "penalties"),
        (s.probabilities is not None, "probabilities"),
        (s.sampling is not None and not sampled_draftable(s.sampling), "sampling"),
        (s.head is not None, "exact_cache_hit"),
        (s.head is None and s.pos > max(0, len(s.prompt) - WINDOW), "short_uncached_tail"),
    ) if blocked)


class DS41Engine:
    tp = 8
    concurrent = True                # serving: the shared Scheduler runs concurrent streams
    supports_canonical_prefix_archive = False
    supports_teacher_forcing = True      # /v1/tensorfold/score (tools/quality_eval: the lane's perplexity gate)
    penalties_supported = True

    def __init__(self, model_dir: Path, *, rank: int, tp: int, master: str = "", master_port: int = 29551,
                 context: int | None = None, slots: int = 4, max_tokens: int = MAX_TOKENS, parallel: int | None = None,
                 serve: bool = False, queue_timeout_s: float | None = None, graphs: bool = True,
                 keep_slots: int | None = None, prefix_host_bytes: int | None = None, vision: bool = False,
                 vision_urls: bool = False, vision_offload: bool = False, yarn_factor: float | None = None,
                 kv_dtype: str = "bf16", expert_tensor_parallel_size: int | None = None,
                 expert_placement: Path | None = None, expert_balance_state: Path | None = None,
                 expert_balance_sample_every: int = 64, expert_balance_half_life: int = 262_144,
                 no_expert_balance: bool = False,
                 kv_full_windows: str | None = None,
                 draft_root: Path | None = None, draft_depth: int | None = None,
                 **_options) -> None:
        from .. import ROTOR_KV_DTYPE

        expert_balance_live = serve and not no_expert_balance
        if kv_full_windows is not None and (kv_full_windows != "auto" or not serve):
            raise ValueError("DeepSeek-V4.1 supports only --kv-full-windows auto in CUDA serving")
        full_windows_auto = kv_full_windows == "auto"
        if ((draft_root is None) != (draft_depth is None)
                or draft_depth is not None and (type(draft_depth) is not int or draft_depth != 3 or not full_windows_auto)):
            raise ValueError("DeepSeek DSpark requires an FT bundle and depth 3 with --kv-full-windows auto")
        # The container may also serve the native codec. Its execution must
        # skip the experiments scoped to the compact RotorQuant profile.
        rotorquant = kv_dtype == ROTOR_KV_DTYPE
        w576_fp32 = rotorquant and os.environ.get("TENSORFOLD_DS41_EXPERIMENT_W576_FP32") == "1"
        sp_fp32 = rotorquant and os.environ.get("TENSORFOLD_DS41_EXPERIMENT_SP_FP32") == "1"
        if no_expert_balance and (expert_placement is not None or expert_balance_state is not None):
            raise ValueError("disabled expert balancing cannot accept a placement or history file")
        if not expert_balance_live and expert_balance_state is not None:
            raise ValueError("expert balance history requires live serving")
        if expert_balance_live and expert_balance_state is None:
            from .weights import checked_etp

            degree = checked_etp(expert_tensor_parallel_size)
            model_path = Path(model_dir).resolve(strict=True)
            name = f"expert-balance-etp{degree}.json"
            expert_balance_state = model_path / name if model_path.is_dir() else model_path.with_name(f"{model_path.name}.{name}")
        options = dict(rank=rank, tp=tp, master=master, master_port=master_port, context=context, slots=slots,
                       max_tokens=max_tokens, parallel=parallel, serve=serve, queue_timeout_s=queue_timeout_s,
                       graphs=graphs, keep_slots=keep_slots, prefix_host_bytes=prefix_host_bytes, vision=vision,
                       vision_urls=vision_urls, vision_offload=vision_offload, yarn_factor=yarn_factor,
                       kv_dtype=kv_dtype, expert_tensor_parallel_size=expert_tensor_parallel_size,
                       expert_placement=expert_placement, expert_balance_state=expert_balance_state,
                       expert_balance_sample_every=expert_balance_sample_every,
                       expert_balance_half_life=expert_balance_half_life, expert_balance_live=expert_balance_live,
                       w576_fp32=w576_fp32, sp_fp32=sp_fp32, full_windows_auto=full_windows_auto,
                       draft_root=draft_root, draft_depth=draft_depth)
        if expert_placement is None and expert_balance_state is None and not (w576_fp32 or sp_fp32 or full_windows_auto):
            self._initialize(model_dir, startup=None, **options)
            return

        from tensorfold.cleanup import rollback
        from .startup import ExpertStartup

        startup = ExpertStartup()
        try:
            self._initialize(model_dir, startup=startup, **options)
        except BaseException as primary:
            rollback(startup, primary, lambda: startup.abort(self, primary))

    def _initialize(self, model_dir, *, rank, tp, master, master_port, context, slots, max_tokens, parallel,
                    serve, queue_timeout_s, graphs, keep_slots, prefix_host_bytes, vision, vision_urls,
                    vision_offload, yarn_factor, kv_dtype, expert_tensor_parallel_size, expert_placement,
                    expert_balance_state, expert_balance_sample_every, expert_balance_half_life, expert_balance_live,
                    w576_fp32, sp_fp32, full_windows_auto, draft_root, draft_depth, startup):
        import torch
        from tensorfold.checkpoint_metadata import read_config
        from tensorfold.cuda.comm import open_comm
        from tensorfold.cuda.control_pipes import ControlPipes
        from .. import CUDA_KV_DTYPES, NATIVE_KV_DTYPE, ROTOR_KV_DTYPE
        from . import weights as W
        from .engram import HostEngram
        from .collectives import Collectives
        from .target import Target
        from .startup import Timeline

        if tp != W.WORLD or not 0 <= rank < tp:
            raise ValueError("DeepSeek-V4.1 SM70 serving requires --tp 8")
        etp = W.checked_etp(expert_tensor_parallel_size)
        if draft_root is not None and etp != W.ETP:
            raise ValueError("DeepSeek DSpark requires ETP4 × EP2")
        self._experimental_w576_fp32 = w576_fp32
        experiment = w576_fp32 or sp_fp32
        if expert_balance_state is not None:
            if (not serve or type(expert_balance_sample_every) is not int or not 1 <= expert_balance_sample_every <= 1 << 20
                    or type(expert_balance_half_life) is not int or not 1 <= expert_balance_half_life <= 1 << 52):
                raise ValueError("DeepSeek expert balancing requires serving and positive bounded sampling settings")
        self.expert_tensor_parallel_size = etp
        model_dir = Path(model_dir)
        if parallel is not None:
            if type(parallel) is not int or not 1 <= parallel <= MAX_STREAMS:
                raise ValueError(f"DeepSeek-V4.1 SM70 serves --parallel 1 to {MAX_STREAMS}")
            slots = parallel
        elif full_windows_auto:
            slots = MAX_STREAMS
        self.streams = slots
        raw = read_config(model_dir)
        self.geometry = g = W.Geometry.from_config(raw)
        if kv_dtype not in CUDA_KV_DTYPES:
            raise ValueError(f"DeepSeek-V4.1 does not support --kv-dtype {kv_dtype!r}")
        rotorquant = kv_dtype == ROTOR_KV_DTYPE
        self.model_kv_dtype = ROTOR_KV_DTYPE if rotorquant else NATIVE_KV_DTYPE
        rope = g.rope_parameters(yarn_factor)
        self.yarn_factor = rope.factor
        self.context = self.context_window = int(context or rope.context_limit)
        if not 1 <= self.context <= rope.context_limit:
            raise ValueError(f"DeepSeek-V4.1 context must be within the {rope.context_limit}-token YaRN window")
        if full_windows_auto and (not rotorquant or vision_offload or not graphs or yarn_factor is not None
                                  or self.context != rope.context_limit or max_tokens != MAX_TOKENS or experiment):
            raise ValueError("DeepSeek-V4.1 --kv-full-windows auto requires RotorQuant Q4 at native context "
                             "with decode graphs, the ordinary prompt chunk, native YaRN and no vision offload")
        self.rank, self.world = rank, tp
        self.vocab = g.vocab
        self.max_tokens = max_tokens
        if rank == 0:              # metadata errors fail before the weights load
            from tensorfold.token_metadata import resolve_eos
            self.eos = resolve_eos(model_dir, g.eos, vocab_size=self.vocab)
        torch.cuda.set_device(0)
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        device = torch.device("cuda", 0)
        started = time.perf_counter()
        self.timeline = clock = Timeline(rank)
        clock.add("init", started, 0.0)
        # The Flash Next / Gemma 4 NVLink small all-gather mailbox (collectives.py: small gathers and the LL row
        # all-reduce); every rank sets the same value before the communicator opens (its open checks they agree).
        from tensorfold.cuda import small_gather
        os.environ.setdefault(small_gather.ENV, str(small_gather.MAX_CAPACITY))
        from .startup import Prebuild
        prebuild = Prebuild(clock)         # every native extension, built/loaded in parallel beside the startup
        prebuild.wait(small_gather.__name__)
        with clock.phase("comm"):
            self.comm = open_comm(rank, tp, master or "127.0.0.1", master_port)
            mode = (int(expert_balance_live) | (int(w576_fp32) << 1) | (int(sp_fp32) << 2)
                    | (int(rotorquant) << 3) | (int(full_windows_auto) << 4) | (int(draft_root is not None) << 5)
                    | (int(vision) << 6))
            settings = [etp, int(expert_placement is not None), mode]
            local = torch.tensor(settings, dtype=torch.int64, device=device)
            agreed = torch.empty(tp * len(settings), dtype=torch.int64, device=device)
            self.comm.all_gather(local, agreed)
            if agreed.tolist() != settings * tp:
                raise ValueError("DeepSeek-V4.1 ranks must agree on KV storage, full-window, draft and vision modes, "
                                 "ETP degree, expert placement and experimental math")
            del local, agreed
        self.coll = Collectives(self.comm)
        # Prompt-prefix reuse (prefix.py): ``keep_slots`` more slots keep idle conversations' chains.
        from .prefix import HostSlab, default_keep_slots
        requested_keep_slots = keep_slots
        if keep_slots is None:
            keep_slots = default_keep_slots(slots)
        if type(keep_slots) is not int or keep_slots < 0:
            raise ValueError("keep_slots must be a nonnegative integer")
        if full_windows_auto and keep_slots >= 1 << 63:
            raise ValueError("DeepSeek-V4.1 keep_slots cannot fit the rank capacity plan")
        self.keep_slots = keep_slots
        # Development smoke runs load only the first N decoder layers.
        smoke = int(os.environ.get("TENSORFOLD_DS41_SMOKE_LAYERS") or 0)
        if smoke > 0 and expert_balance_state is not None:
            raise ValueError("DeepSeek expert balancing requires all model layers")
        capture = graphs and not os.environ.get("TENSORFOLD_DS41_ENGRAM_TRACE")
        if full_windows_auto and (not capture or smoke > 0):
            raise ValueError("DeepSeek-V4.1 --kv-full-windows auto requires all decoder layers and CUDA graph capture")
        h64_text = rotorquant and not vision and capture and self.context <= 131_072 and max_tokens <= MAX_TOKENS
        from .graphs import shape_keys
        graph_keys = shape_keys(slots, self.context) if capture else ()
        compact_profile = (etp == W.ETP and h64_text and tuple(graph_keys) == H64_128K_GRAPH_KEYS
                           and self.context == 131_072 and slots == MAX_STREAMS
                           and keep_slots == default_keep_slots(slots) and max_tokens == MAX_TOKENS
                           and yarn_factor is None and self.yarn_factor == 16.0 and smoke <= 0
                           and prefix_host_bytes == 16 << 30)
        # After rank mode agreement, reject unsupported experiments before opening the checkpoint.
        if w576_fp32 and not (serve and compact_profile):
            raise ValueError("experimental W576 FP32 requires the native compact 128K H64 text serving profile")
        if sp_fp32 and not compact_profile:
            raise ValueError("experimental SP FP32 requires the native compact 128K H64 text profile")
        if experiment and vision_offload:
            raise ValueError("DeepSeek-V4.1 SM70 keeps its vision tower in VRAM: --vision-offload is unsupported")
        entry = raw.get("gguf_file") or raw.get("gguf_entrypoint")
        engram_file = raw.get("engram_file")
        if not entry or not engram_file:
            raise ValueError("the DeepSeek-V4.1 model directory's config.json must name gguf_file (the backbone's "
                             "first shard) and engram_file (the llama.cpp Engram GGUF)")
        resolve = lambda p: Path(p) if Path(p).is_absolute() else model_dir / p
        bind_files = startup is not None
        shards = W.Shards(resolve(entry), resolve(engram_file), bind_files=True) if bind_files else W.Shards(
            resolve(entry), resolve(engram_file))
        placement_options = {}
        if bind_files:
            startup.shards = shards
            from .expert_placement import agree_placement, placement_identity

            checkpoint = placement_identity(shards.placement_files(), raw)
            identity = list(bytes.fromhex(checkpoint))
            local = torch.tensor(identity, dtype=torch.int32, device=device)
            agreed = torch.empty(tp * len(identity), dtype=torch.int32, device=device)
            self.comm.all_gather(local, agreed)
            if agreed.tolist() != identity * tp:
                raise ValueError("DeepSeek-V4.1 ranks loaded different expert-balance checkpoint inputs")
            del local, agreed
        if expert_placement is not None:
            placement_options["expert_placement"] = agree_placement(
                Path(expert_placement), store=self.comm.store, rank=rank, checkpoint=checkpoint,
                layers=g.layers, experts=g.experts, etp=etp,
            )
        self.expert_observer = None
        self.expert_live = None
        self._draft_verify = None
        if expert_balance_state is not None:
            key = "deepseek-v41/expert-history-startup-v1"
            if rank == 0:
                from .expert_balance_state import ExpertBalanceState
                from .expert_observer import ExpertObserver

                try:
                    history = ExpertBalanceState(Path(expert_balance_state), checkpoint=checkpoint, etp=etp,
                                                 layers=g.layers, experts=g.experts, half_life_routes=expert_balance_half_life,
                                                 current=placement_options.get("expert_placement"))
                    startup.history = history
                    self.expert_observer = ExpertObserver(device, history, sample_every=expert_balance_sample_every,
                                                          minimum_routes=expert_balance_half_life)
                    startup.observer = self.expert_observer
                except (OSError, ValueError, RuntimeError) as exc:
                    from tensorfold.cleanup import rollback

                    error = str(exc).encode(errors="replace")[:1024]
                    rollback(self, exc, lambda: self.comm.store.set(key, b"!" + error))
                from .expert_placement import encode_placement

                active_epoch, active_placement = history.placement_snapshot()
                status = b"+" + _EXPERT_EPOCH.pack(active_epoch) + encode_placement(
                    active_placement, checkpoint=checkpoint, etp=etp)
                self.comm.store.set(key, status)
            else:
                status = self.comm.store.get(key)
                if status.startswith(b"!"):
                    error = status[1:].decode(errors="replace")
                    raise RuntimeError(f"DeepSeek expert history initialization failed on rank zero: {error}")
                from .expert_placement import decode_placement

                if not status.startswith(b"+") or len(status) <= 1 + _EXPERT_EPOCH.size:
                    raise RuntimeError("DeepSeek ranks received an incomplete live expert history")
                active_epoch, = _EXPERT_EPOCH.unpack_from(status, 1)
                if active_epoch < 0:
                    raise RuntimeError("DeepSeek expert history contains an invalid active epoch")
                active_placement = decode_placement(status[1 + _EXPERT_EPOCH.size:], checkpoint=checkpoint,
                                                    layers=g.layers, experts=g.experts, etp=etp,
                                                    source="DeepSeek live expert history on rank 0")
            identity = tuple(range(g.experts))
            if all(row == identity for row in active_placement):
                placement_options.pop("expert_placement", None)
            else:
                placement_options["expert_placement"] = active_placement
        layers = range(min(smoke, g.layers)) if smoke > 0 else None
        self.engram = None
        mapping = None
        if layers is None or len(layers) > min(g.engram_layers):
            # The Engram mapping (page tables, ordered registrations: host and driver work, no device memory) runs on
            # its own thread while this one loads the weights; the forward first reads it in _warm, after the join.
            # The page-table populate (and, from a cold page cache, the file reads) overlap the load; the driver
            # serializes the registration itself with the load's CUDA calls (start/diag loadprof: W.load alone 8-10 s,
            # 20-28 s beside a 7-16 s registration).
            from concurrent.futures import ThreadPoolExecutor
            rows = W.engram_split(shards, g)
            engram_path, engram_offset = shards.engram.path, shards.engram.offset

            def map_engram(open_fd=None):
                with clock.phase("engram") as note:
                    options = {} if open_fd is None else {"open_fd": open_fd}
                    e = HostEngram(engram_path, engram_offset, rows, rank, self.comm.store, **options)
                    if startup is not None:
                        startup.engram = e
                    note[0] = f"fill {e.fill_seconds:.1f} wait {e.wait_seconds:.1f} register {e.register_seconds:.1f}"
                return e
            if startup is None:
                mapping = ThreadPoolExecutor(1, thread_name_prefix="ds41-engram")
                engram = mapping.submit(map_engram)
                mapping.shutdown(wait=False)
            else:
                engram = startup.map_engram(shards.duplicate_file(engram_path), map_engram)
        if layers is not None:
            print(f"[tensorfold] rank {rank}: DEVELOPMENT SMOKE: {len(layers)} of {g.layers} layers loaded", flush=True)
        prebuild.wait()
        with clock.phase("weights"):
            self.w = W.load(shards, g, rank, device, layers=layers, expert_tensor_parallel_size=etp,
                            compact_expert_down=h64_text and layers is None, **placement_options)
            torch.cuda.synchronize()
        if mapping is not None or (startup is not None and startup.mapper is not None):
            self.engram = engram.result()
            if startup is not None:
                startup.retire_mapper()
            print(f"[tensorfold] rank {rank}: Engram table mapped and registered in {self.engram.seconds:.0f} s", flush=True)
        if bind_files:
            shards.verify_placement_files()
        shards.close()
        if startup is not None:
            startup.shards = None
        if experiment and not self.w.compact_expert_down:
            raise ValueError("experimental FP32 requires actual compact expert banks")
        # Image input (--vision): the vision router bias and a TP8 part of the mmproj tower on every rank (each rank
        # reads only its part). Every rank encodes in step (``_tower``), so each ends with the same span and a prompt
        # chunk's image rows need no broadcast. Loaded before the KV pool is sized, so the pool pays for it; the pool
        # is sized by the smallest rank, which with the tower split is every rank's 1/8.
        self.tower = self.vision = None
        self.bias_vl = None
        self.tower_states, self.tower_spans = {}, {}     # handle -> paused encode / finished span (every rank)
        if vision_offload:
            raise ValueError("DeepSeek-V4.1 SM70 keeps its vision tower in VRAM: --vision-offload is unsupported")
        if vision:
            from . import vision as V
            V.check_config(raw)
            mmproj, bias_file = raw.get("mmproj_file"), raw.get("vision_bias_file")
            if not mmproj or not bias_file:
                raise ValueError("--vision needs config.json mmproj_file (the deepseek41v mmproj GGUF) and "
                                 "vision_bias_file (exp_probs_b_vl.f32, 40 x 384 F32)")
            table, digest = V.read_bias_vl(resolve(bias_file))
            if table.shape[0] != g.layers:
                raise ValueError(f"the vision router bias has {table.shape[0]} layers; the model has {g.layers}")
            self.bias_vl = torch.from_numpy(table).to(device)
            mmproj = resolve(mmproj).resolve(strict=True)

            def stamp():
                st = mmproj.stat()
                return struct.pack("<QQQqq", st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)

            # TP shards must use the same stable local checkpoint incarnation. The deployment keeps its files
            # immutable while loading. Check for ordinary path/file changes across the inventory and data reads.
            before = stamp()
            tensors = V.read_mmproj(mmproj, rank, tp)
            if stamp() != before:
                raise ValueError("the DeepSeek-V4.1 mmproj changed during startup")
            fingerprint = torch.tensor(list(before + bytes.fromhex(digest)), dtype=torch.uint8, device=device)
            fingerprints = torch.empty((tp, fingerprint.numel()), dtype=torch.uint8, device=device)
            self.comm.all_gather(fingerprint, fingerprints)
            if not bool((fingerprints == fingerprint).all()):
                raise ValueError("the DeepSeek-V4.1 ranks loaded different vision checkpoints")
            self.tower = V.VisionTower(tensors, device, rank=rank, world=tp, comm=self.coll)
            del tensors
            if rank == 0:
                self.vision = V.Frontend(model_dir, allow_urls=vision_urls)
                print(f"[tensorfold] DeepSeek-V4.1 SM70: vision tower TP{tp}, {self.tower.nbytes() / 2**30:.3f} GiB "
                      f"per rank (FP16 GEMMs, FP32 residual, feature gathers); vision router bias "
                      f"sha256 {'matches' if digest == V.BIAS_VL_SHA256 else 'DIFFERS from'} upstream's", flush=True)
        with clock.phase("barrier"):
            self._barrier()
        # The host tier's pinned slab (Gemma 4 SM70 sizing, on by default) pins on its own thread from here (every
        # rank's Engram registration is done: pinning while the Engram table registers slowed m-pf1's registration
        # 4x) until before the graph capture.
        slab_started = time.perf_counter()
        if experiment:
            slab = HostSlab(self.comm, rank, tp, prefix_host_bytes, required_per_rank=2 << 30, startup=startup)
        elif full_windows_auto:
            slab = HostSlab(self.comm, rank, tp, prefix_host_bytes, startup=startup)
        else:
            slab = HostSlab(self.comm, rank, tp, prefix_host_bytes)
        # The lattice prefill experts' persistent canonical-bank workspace exists before the KV pool is
        # sized, so it does not come out of RESERVE.
        from tensorfold.cuda.sm70 import ds41_lattice_moe
        with clock.phase("lattice-ws"):
            if etp == W.ETP:
                lattice = ds41_lattice_moe.Workspace(device)
            else:
                lattice = ds41_lattice_moe.Workspace(device, tile=len(self.w.local_experts),
                                                      intermediate=g.moe_inter // etp)
            torch.cuda.synchronize()
        draft_weights = None
        if draft_root is not None:
            from .dspark_weights import load_ft

            with clock.phase("dspark-weights"):
                draft_weights = load_ft(draft_root, g, rank=rank, world=tp, vocab_first=self.w.embed_first,
                                        device=device)
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        # Sample after both optional model companions are resident. Tower warm-up uses the prompt reserve.
        free = torch.cuda.mem_get_info()[0]
        sizing_inputs = []
        if full_windows_auto:
            sizing_inputs = [self.context, slots, -1 if requested_keep_slots is None else requested_keep_slots]
        mine = torch.tensor([free, *sizing_inputs], dtype=torch.int64, device=device)
        every = torch.empty(tp * mine.numel(), dtype=torch.int64, device=device)
        self.comm.all_gather(mine, every)
        from .target import BLOCK, window_ring_bytes
        from .kernels.mu_packed import PackedLayout
        layout = PackedLayout({l: g.compress_ratios[l] for l in g.kv_sources}, BLOCK, rotorquant=rotorquant)
        page_bytes = layout.page_bytes
        ring_bytes_per_slot = g.layers * window_ring_bytes(rotorquant)
        # The decode graphs are captured after the KV pool exists, so their memory comes out of the free memory too:
        # ~30 MiB per captured graph outside the torch pool (graph/NCCL resources) + ~40 MiB pool (ds41-graphs g2:
        # 606 MiB for 18 buckets on one capture stream).
        compact_128k = compact_profile and self.w.compact_expert_down and slab.per_rank == 2 << 30
        graph_bytes = GRAPH_BYTES + GRAPH_EACH_BYTES * len(graph_keys) if capture else 0
        prompt_reserve = RESERVE
        funding = None
        draft_budget = {}
        c2_sources = sum(g.compress_ratios[layer] == 2 for layer in g.kv_sources)
        if draft_weights is not None:
            from .dspark_weights import RING, STAGES
            from .kernels.kv_pack import ROTOR_ROW
            from .speculation import Speculation

            def draft_state(physical_slots):
                ring = physical_slots * STAGES * RING * g.head_dim * 2
                return ring + Speculation.state_bytes(physical_slots - 1, g.layers, c2_sources, ROTOR_ROW)

            draft_budget = dict(draft_graph_bytes=len(DRAFT_BUCKETS) * (GRAPH_BYTES + GRAPH_EACH_BYTES),
                                draft_state_bytes=draft_state, min_lanes=DRAFT_BUCKETS[-1])
        if full_windows_auto:
            from .prefix import full_window_plan

            records = every.view(tp, mine.numel()).tolist()
            if any(row[1:] != sizing_inputs for row in records):
                raise ValueError("DeepSeek-V4.1 ranks must agree on full-window sizing inputs")
            funding = full_window_plan(min(row[0] for row in records), self.context, slots, requested_keep_slots,
                                       layout, ring_bytes_per_slot, c2_sources,
                                       prompt_reserve=prompt_reserve, graph_fixed=GRAPH_BYTES, graph_each=GRAPH_EACH_BYTES,
                                       **draft_budget)
            slots, keep_slots = funding["funded_lanes"], funding["keep_slots"]
            self.streams, self.keep_slots = slots, keep_slots
            graph_keys, graph_bytes, pages = funding["graph_keys"], funding["graph_allowance_bytes"], funding["pool_pages"]
        else:
            if etp == W.ETP and h64_text:
                prompt_reserve = H64_TEXT_RESERVE
                if compact_128k:
                    graph_bytes = H64_128K_GRAPH_BYTES
            ring_bytes = (slots + keep_slots + 1) * ring_bytes_per_slot
            pages = max(1, (int(every.min()) - prompt_reserve - graph_bytes - ring_bytes) // page_bytes)
        self.slots = slots + keep_slots + 1          # streams, kept chains, the scratch slot
        target_started = time.perf_counter()
        self.target = Target(self.w, self.coll, slots=self.slots, pages=pages, max_tokens=max_tokens,
                             context=self.context, engram=self.engram, device=device, lattice=lattice,
                             bias_vl=self.bias_vl, rope=rope, layout=layout, fp32_sp=sp_fp32)
        if self.expert_observer is not None:
            self.target.expert_usage = self.expert_observer.usage
        clock.add("target", target_started, time.perf_counter() - target_started)
        self.image_features = None          # every rank: the next F_IMAGE step's image rows [n, 5120] FP16
        self.layer_count = len(self.w.layers)
        self.units = self.target.units      # pause points of a prompt forward (attention, experts per layer)
        from .prefix import SlotState
        self.prefix = SlotState(self.target)
        if funding is not None:
            pool = self.target.pool
            if (pool.cache.nbytes != pages * page_bytes
                    or sum(ring.nbytes for ring in pool.rings) != funding["ring_bytes"]
                    or self.prefix.pages != funding["snapshot_pages_each"]
                    or self.prefix.raw != funding["snapshot_raw_bytes"]
                    or sum(part.nbytes for part in self.prefix._parts(0)) != self.prefix.raw):
                raise RuntimeError("DeepSeek-V4.1 actual KV rings or snapshot layout differs from the full-window plan")
        self.layout = self.target.pool.layout
        self.token_bytes = self.layout.bytes_per_token
        self.snap_bytes = self.prefix.snap_bytes
        self.budget = pages * page_bytes            # the prefix owner's ledger: slot pages and snapshot pages
        # Prompt chunks split at most prompt_rows apart (no point kept there) and where prefix points are kept
        # (prefix.chunk_splits: the prompt's first and last message boundary), independent of load.
        self.prompt_rows = max_tokens
        self.part = None                    # a paused prompt forward (state, logits rows, slots), same on all ranks
        self.layer_marks = None             # rank 0: per-unit CUDA events of the prompt forward being timed
        from .device_sampling import DeviceSampler
        from .graphs import DecodeGraphs, SlotPages
        self.sampler = DeviceSampler(self.comm, rank=rank, world=tp, shard=self.vocab // tp, device=device)
        self.slot_pages = SlotPages(self.slots, -(-self.target.hash_state.max_context // BLOCK), device)
        self.graphs = None
        self._warm()
        with clock.phase("slab-join"):
            self.host_per_rank, self.host_slab = slab.finish()
            if experiment or full_windows_auto:
                startup.retire_slab()
        clock.add("slab", slab_started, time.perf_counter() - slab_started)
        self.prefix.slab = self.host_slab
        if rank == 0:
            print(f"[tensorfold] DeepSeek-V4.1 SM70: host prefix tier {self.host_per_rank >> 20} MiB/rank pinned "
                  f"(beside the warm-up: {time.perf_counter() - slab_started:.1f} s)", flush=True)
        # Decode forwards replay startup CUDA graphs (every rank, same order, no control frames; never lazily).
        # The Engram lookup trace (rank 0) reads ids on the host every forward, so tracing keeps the eager decode;
        # every rank decides from the same environment (all ranks capture, or none).
        if capture:
            with clock.phase("graphs") as note:
                self.graphs = DecodeGraphs(self, slots)
                note[0] = f"{len(self.graphs.keys)}, warm-up runs {self.graphs.warm_seconds:.1f}"
            spare = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
            if self.graphs.bytes > graph_bytes or spare < prompt_reserve:
                raise RuntimeError(f"DeepSeek-V4.1 SM70 rank {rank}: decode graphs took {self.graphs.bytes >> 20} MiB "
                                   f"(sized {graph_bytes >> 20} MiB); {spare >> 20} MiB left for prompt transients, "
                                   f"{prompt_reserve >> 20} MiB required")
            if funding is not None:
                if self.graphs.keys != graph_keys:
                    raise RuntimeError("DeepSeek-V4.1 actual graph catalogue differs from the full-window plan")
                self.capacity_plan = funding
        elif rank == 0:
            print("[tensorfold] DeepSeek-V4.1 SM70: eager decode (" +
                  ("graphs disabled)" if not graphs else "TENSORFOLD_DS41_ENGRAM_TRACE set)"), flush=True)
        if draft_weights is not None:
            from .dspark_serving import prepare_verifier

            self._draft_graph_budget = funding["draft_graph_allowance_bytes"]
            with clock.phase("dspark-capture"):
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                before = torch.cuda.mem_get_info()[0]
                self._draft_verify = prepare_verifier(self, draft_weights, DRAFT_BUCKETS, draft_depth)
                spec = self._draft_verify.shape.spec
                state = self._draft_verify.head.ring.nbytes + spec.nbytes
                if state != funding["draft_state_bytes"]:
                    raise RuntimeError("DeepSeek DSpark state differs from the full-window plan")
                self._capture_draft_graphs(prompt_reserve, before=before, state=state)
            self.depth = draft_depth
        if expert_balance_live:
            from .expert_live import LiveExpertPlacement

            self.expert_live = LiveExpertPlacement(self, epoch=active_epoch, placement=active_placement,
                                                   operation=OP_EXPERT_PLACEMENT)
            self._expert_graph_keys = tuple(self.graphs.keys) if self.graphs is not None else None
            self._expert_graph_bytes = graph_bytes
            self._expert_prompt_reserve = prompt_reserve
        with clock.phase("pipes"):
            self.pipes = ControlPipes(self.comm.store, rank, tp, timeout_s=PIPE_TIMEOUT)
        if rank == 0 and sorted(self.pipes.piped()) != list(range(1, tp)):
            raise RuntimeError("DeepSeek-V4.1 SM70 requires all eight ranks on one host (control pipes)")
        self.load_seconds = time.perf_counter() - started
        print(clock.line(), flush=True)
        self.pages = pages
        self.page_bytes = page_bytes
        self.decoder = self.scheduler = None
        self._closed = False
        if rank == 0:
            print(f"[tensorfold] DeepSeek-V4.1 SM70: routed experts ETP{etp} x EP{tp // etp}", flush=True)
            if self.graphs is not None:
                print(f"[tensorfold] DeepSeek-V4.1 SM70: decode graphs for batches {list(self.graphs.sizes)} "
                      f"(index classes {self.graphs.classes[self.graphs.sizes[-1]]}; {len(self.graphs.keys)} graphs) "
                      f"captured in {self.graphs.seconds:.1f} s, {self.graphs.bytes / 2**20:.0f} MiB", flush=True)
            engram = ("not mapped (development smoke)" if self.engram is None else
                      f"{self.engram.size / 2**30:.1f} GiB in {self.engram.seconds:.0f} s, host RAM only: VRAM residency "
                      f"{self.engram.device_bytes} B")
            print(f"[tensorfold] DeepSeek-V4.1 SM70: loaded in {self.load_seconds:.0f} s (Engram table {engram} + per-step working buffer "
                  f"{self.target.engram_rows.numel() * 2 >> 20} MiB); KV pool {pages} pages of "
                  f"{page_bytes} B ({pages * 64} tokens), {slots} streams", flush=True)
        if rank == 0 and serve:
            from tensorfold.cuda.scheduler import Scheduler
            # The V4.1 encoder's message boundaries of a prompt's ids (kept-point and chunk-split candidates).
            from tensorfold.token_metadata import load_tokenizer
            from ..prompts_v41 import message_points
            self.points = message_points(load_tokenizer(model_dir).token_to_id)
            self.decoder = FixedDraftDecoder(self) if self._draft_verify is not None else Decoder(self)
            # Requests beyond --parallel wait in the scheduler's queue; --queue-timeout-s bounds that wait (503 after).
            options = {}
            if self.expert_live is not None:
                from threading import Lock

                self._expert_proposal_lock = Lock()
                self._expert_proposal = None
                self._expert_refused = None
                options["worker_maintenance"] = self._expert_maintenance
                self.expert_observer.publish_live = self._publish_expert_proposal
            self.scheduler = Scheduler(self.decoder, max_streams=slots, queue_timeout_s=queue_timeout_s, **options)
            try:
                self.scheduler.start()
            except BaseException as primary:
                if startup is None:
                    try:
                        self.scheduler.close()
                    except BaseException as cleanup:
                        raise primary from cleanup
                raise
            if self.expert_observer is not None:
                self.expert_observer.active = True

    def _warm(self):
        """Startup, every rank, no control frames: build the native extensions and compile the forward's Triton
        kernels on slot 0 (a prompt chunk past the grouped-experts and prefill-indexer thresholds, then one decode
        row), so the first request builds nothing. Slot 0 is reset by its next attach.

        The Decoder's first per-unit cost sample is timed on a second identical chunk: the first one's units carry
        the process's first-use host stalls (kernel loads and compiles, library handles, allocator growth; seconds
        on layer 0's units), and that sample stays the small cost anchor, so a chunk planned from it ran unit 0 and
        unit 1 alone and then, its measured progress scaled by those inflated predictions, every other unit in one
        group (``Decoder._stop``): a whole 4096-row chunk between two decode rounds."""
        import torch
        from tensorfold.cuda.sm70 import ds41_lattice_moe, ds41_sparse_attention, onecat_ds41
        from .target import DECODE_MAX
        clock = self.timeline
        with clock.phase("warm-ext"):
            ds41_lattice_moe.prepare()
            ds41_sparse_attention.prepare()
            onecat_ds41.prepare()
        warm_started = time.perf_counter()
        rows = max(ds41_lattice_moe.MIN_TOKENS, DECODE_MAX + 1) * 2
        with torch.inference_mode():
            self._attach(0, rows + 1)
            self._step([(0, 0, rows)], [self.geometry.bos] * rows, [rows - 1])
            self._attach(0, rows + 1)
            self.layer_marks = [] if self.rank == 0 else None
            self._step([(0, 0, rows)], [self.geometry.bos] * rows, [rows - 1])
            if self.rank == 0:          # the Decoder's first per-unit prompt cost sample (ms at ``rows`` rows)
                torch.cuda.synchronize()
                (_, events), = self.layer_marks
                self.warm_unit_ms = (rows, [events[j - 1].elapsed_time(events[j]) for j in range(1, len(events))])
            self.layer_marks = None
            self._step([(0, rows, 1)], [self.geometry.bos], [0])
            self._detach(0)
            if self.tower is not None:
                self._warm_tower()
        torch.cuda.synchronize()
        clock.add("warm-forward", warm_started, time.perf_counter() - warm_started)

    def _warm_tower(self):
        """Every rank warms the tower at the supported patch-count bound through the full serving path.

        The Decoder scales rank 0's largest observed block cost by patch count for smaller frames.
        """
        import numpy as np
        import torch
        from . import vision as V
        height, width = V.MAX_PATCH_GRID
        patches = height * width
        frame = V.ImageFrame(np.zeros((patches, 3, V.PATCH, V.PATCH), np.uint8), height, width, "warm")
        free = torch.cuda.mem_get_info()[0]
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(V.LAYERS + 1)]
        self._tower(T_BEGIN, -1, height, width, (), V.pinned_patches(frame) if self.rank == 0 else None)
        events[0].record()
        for b in range(V.LAYERS):
            self._tower(T_RUN, -1, b + 1)
            events[b + 1].record()
        self._tower(T_SPAN, -1)
        self._tower(T_STAGE, 0, 0, 0, [-1, 0, frame.tokens, 1])
        span, self.image_features = self.image_features, None
        torch.cuda.synchronize()
        if span.shape != (frame.tokens, V.OUT_DIM) or not bool(torch.isfinite(span).all()) or self.tower_spans:
            raise RuntimeError("DeepSeek-V4.1 vision tower warm-up produced an invalid span")
        self.tower_patches = patches
        ms = [events[b].elapsed_time(events[b + 1]) for b in range(V.LAYERS)]
        self.tower_block_ms = max(ms)
        self.tower_peak = torch.cuda.max_memory_allocated() - before
        del span
        print(f"[tensorfold] DeepSeek-V4.1 SM70: vision tower warm ({frame.tokens}-token image): rank {self.rank}, "
              f"blocks {sum(ms):.0f} ms ({self.tower_block_ms:.1f} ms max), "
              f"transient peak {self.tower_peak >> 20} MiB of {free >> 20} MiB free", flush=True)
        if self.tower_peak > RESERVE:
            raise RuntimeError("the DeepSeek-V4.1 vision tower's transients exceed the prompt reserve")

    def _tower(self, kind, handle, x=0, y=0, ids=(), pixels=None):
        """One vision tower operation on this rank's TP part (every rank, same order: the collectives pair up).
        T_BEGIN: encode ``handle`` of an x * y patch grid; rank 0's uint8 ``pixels`` (pinned CPU) are broadcast
        from GPU 0 to every rank. T_RUN: its blocks up to x. T_SPAN: finish it into its span. T_STAGE: ``ids`` =
        (handle, first, end, release) quadruples: the next F_IMAGE step's rows are those span rows in order;
        released spans are dropped. T_FREE: drop the encodes and spans of handles ``ids``."""
        import torch
        from .vision import PATCH
        if kind == T_BEGIN:
            if handle in self.tower_states or handle in self.tower_spans:
                raise RuntimeError(f"vision encode {handle} already exists")
            device = self.target.device
            if self.rank == 0:
                patches = pixels.to(device, non_blocking=True)
            else:
                patches = torch.empty((x * y, 3, PATCH, PATCH), dtype=torch.uint8, device=device)
            self.comm.broadcast(patches, 0)
            self.tower_states[handle] = self.tower.begin(patches, x, y)
        elif kind == T_RUN:
            self.tower.run(self.tower_states[handle], x)
        elif kind == T_SPAN:
            self.tower_spans[handle] = self.tower.span(self.tower_states.pop(handle))
        elif kind == T_STAGE:
            rows = []
            for i in range(0, len(ids), 4):
                h, first, end, release = ids[i:i + 4]
                rows.append(self.tower_spans[h][first:end])
                if release:
                    del self.tower_spans[h]
            self.image_features = torch.cat(rows).contiguous()
        elif kind == T_FREE:
            for h in ids:
                self.tower_states.pop(h, None)
                self.tower_spans.pop(h, None)
        else:
            raise RuntimeError(f"unknown DeepSeek-V4.1 vision tower operation {kind}")

    def _barrier(self):
        import torch
        x = torch.zeros(1, dtype=torch.int64, device="cuda")
        y = torch.empty(self.world, dtype=torch.int64, device="cuda")
        self.comm.all_gather(x, y)
        torch.cuda.synchronize()

    # ---------------------------------------------------------------------------------------- operations
    def _attach(self, slot, tokens):
        self.target.pool.detach(slot)
        self.slot_pages.set(slot, self.target.pool.attach(slot, tokens))
        self.target.reset_slot(slot)
        if self._draft_verify is not None:
            self._draft_verify.head.reset_slot(slot)

    def _detach(self, slot):
        self.target.pool.detach(slot)
        self.slot_pages.set(slot, ())

    def slot_bytes(self, capacity: int) -> int:
        """Ledger bytes of a slot holding ``capacity`` positions: its pages."""
        return -(-capacity // 64) * self.page_bytes

    def _step(self, entries, ids, logits_rows, stop=0, tap_from=None, sample=None):
        flags, stop = stop & ~0xFFFF, stop & 0xFFFF
        """entries: [(slot, start, count)] in row order; ids: the tokens; logits_rows: token rows to score.
        Returns rank 0's full-vocabulary FP32 logits [len(logits_rows), vocab] (None elsewhere). With
        0 < ``stop`` < units the forward pauses after unit ``stop`` (``Target.layers``: attention and experts of
        each layer; ``_layers`` continues it) and returns None; the pause changes no kernel or row. One row per
        entry, every row scored and ``stop`` 0 (a decode batch, or a one-token prompt chunk) replays the batch's
        startup graph; ``stop`` = units runs the same forward eagerly."""
        import torch
        from .target import Step, DECODE_MAX
        if tap_from is not None:
            from .dspark_head import WINDOW

            if (self._draft_verify is None or len(entries) != 1
                    or entries[0][2] != len(ids) or not 0 <= tap_from < len(ids)
                    or len(ids) - tap_from > WINDOW or flags & (F_IMAGE | F_SCORE | F_COMPACT | F_FULL | F_AHEAD)):
                raise ValueError("DeepSeek DSpark taps require an ordinary text prompt tail")
        device = self.target.device
        pool = self.target.pool
        positions, token_req, qsl = [], [], [0]
        for slot, start, count in entries:
            if start + count > len(pool.owned.get(slot, ())) * 64:
                raise RuntimeError(f"slot {slot}: positions up to {start + count} exceed its KV pages")
        observer = self.expert_observer
        expert_sample = (flags >> EXPERT_SAMPLE_SHIFT) & 3 if observer is not None else 0
        retain_routed = bool(flags & F_W576_FP32)
        if retain_routed and (not self._experimental_w576_fp32 or flags & (F_SCORE | F_LAST | F_IMAGE)):
            raise ValueError("experimental W576 FP32 is only for continuation frames")
        if (tap_from is None and stop == 0 and not flags & F_IMAGE and self.graphs is not None
                and len(entries) <= self.graphs.sizes[-1]
                and all(count == 1 for _, _, count in entries) and list(logits_rows) == list(range(len(entries)))):
            compact = bool(flags & F_COMPACT)
            return self.graphs.run(entries, ids, compact, full=not compact or bool(flags & F_FULL),
                                   ahead=bool(flags & F_AHEAD), expert_sample=expert_sample, retain_routed=retain_routed,
                                   sample=sample if compact else None)
        # The experiment applies only to graph continuations. Eager and prompt forwards keep stock arithmetic.
        if retain_routed:
            raise RuntimeError("experimental W576 FP32 continuation must replay a startup graph")
        if flags & F_AHEAD:
            raise RuntimeError("a run-ahead decode round must replay a startup graph")
        for r, (slot, start, count) in enumerate(entries):
            positions.extend(range(start, start + count))
            token_req.extend([r] * count)
            qsl.append(qsl[-1] + count)
        width = max(len(pool.owned[slot]) for slot, _, _ in entries)
        table = [pool.owned[slot] + [0] * (width - len(pool.owned[slot])) for slot, _, _ in entries]
        rows = len(entries) + 8
        req_slot = [slot for slot, _, _ in entries] + [self.slots - 1] * 8
        t = lambda v, dt: torch.tensor(v, dtype=dt).to(device, non_blocking=True)
        prefill = [(r, qsl[r], qsl[r + 1], entries[r][1] + entries[r][2] - 1) for r in range(len(entries))
                   if qsl[r + 1] - qsl[r] > DECODE_MAX]
        decode_index = None
        if prefill:
            inside = set()
            for _, s0, s1, _ in prefill:
                inside.update(range(s0, s1))
            rest = [i for i in range(qsl[-1]) if i not in inside]
            decode_index = t(rest, torch.long) if rest else torch.empty(0, dtype=torch.long, device=device)
        image = image_rows = features = None
        if flags & F_IMAGE:          # image rows carry prompt-cache ids >= vocab (``vision.prefix_ids``)
            from .vision import IMAGE_TOKEN_ID, OUT_DIM
            rows_at = [i for i, token in enumerate(ids) if token >= self.vocab]
            if not rows_at:
                raise RuntimeError("an image step without image rows")
            ids = [IMAGE_TOKEN_ID if token >= self.vocab else token for token in ids]
            features, self.image_features = self.image_features, None       # staged by T_STAGE on every rank
            if features is None or features.shape != (len(rows_at), OUT_DIM) or features.dtype != torch.float16:
                raise RuntimeError("the image step's features do not match its image rows")
            image_rows = t(rows_at, torch.long)
            image = torch.zeros(len(ids), dtype=torch.bool, device=device)
            image[image_rows] = True
        step = Step(ids=t(ids, torch.long), positions=t(positions, torch.long),
                    valid=torch.ones(len(ids), dtype=torch.bool, device=device), token_req=t(token_req, torch.int32),
                    qsl=t(qsl, torch.int32), qsl_cpu=qsl, num_reqs=len(entries), req_slot=t(req_slot, torch.int32),
                    block_table=t(table, torch.int32), prefill=prefill, decode_index=decode_index,
                    image=image, image_rows=image_rows, features=features)
        if expert_sample:
            step.expert_sample = observer.eager_flags[expert_sample]
            step.expert_prefill_phase = expert_sample - 1
        del rows
        state = self.target.begin(step)
        if tap_from is not None:
            from .target import DRAFT_TAP_LAYERS

            indices = torch.arange(tap_from, len(ids), dtype=torch.long, device=device)
            taps = torch.empty((len(indices), len(DRAFT_TAP_LAYERS) * self.geometry.hidden), dtype=torch.float16,
                               device=device)
            state["draft_context"] = (entries[0][0], indices, taps)
        if flags & F_SCORE:                # logits_rows carries each row's next token: score all rows
            rows = (logits_rows, bool(flags & F_LAST))
        else:
            rows = t(logits_rows, torch.long) if logits_rows else None
        marks = self.layer_marks if qsl[-1] > len(entries) else None     # prompt chunks only, never decode rows
        if 0 < stop < self.units:
            if self.part is not None:
                raise RuntimeError("a DeepSeek-V4.1 prompt forward began while another one is paused")
            self.part = (state, rows, tuple(slot for slot, _, _ in entries) if self._draft_verify is not None else ())
            self._advance_layers(state, stop, marks)
            return None
        self._advance_layers(state, self.units, marks)
        return self._end(state, rows)

    def _advance_layers(self, state, stop, marks):
        context = state.get("draft_context") if self._draft_verify is not None else None
        if context is None:
            self.target.layers(state, stop, marks)
            return
        import torch

        slot, indices, taps = context
        self.target.layers_with_draft_taps(state, stop, indices, taps, marks)
        if stop == self.units:
            positions = state["step"].positions.index_select(0, indices)
            self._draft_verify.head.insert_context(torch.full_like(positions, slot), taps, positions)

    def _layers(self, stop):
        """Advance the paused prompt forward to unit ``stop`` (< 0: drop it); at the last unit finish it. Decode
        forwards may run between its units (other slots; ``Target.begin``)."""
        if stop < 0:
            self.part = None
            return None
        state, rows, _ = self.part
        self._advance_layers(state, stop, self.layer_marks)
        if stop < self.units:
            return None
        self.part = None
        return self._end(state, rows)

    def _end(self, state, rows):
        import torch
        if isinstance(rows, tuple):                 # F_SCORE: (targets, last)
            targets, last = rows
            local = self.target.end(state, torch.arange(len(targets), device=self.target.device))
            scores = self._score_rows(local, targets)
            if not last:
                return scores
            # the last row's logits as an unscored chunk computes them (a one-row head GEMM), so a scored request
            # and a prefix-cached repeat of an unscored one start their replies from the same bits
            last_row = torch.tensor([len(targets) - 1], dtype=torch.long, device=self.target.device)
            full = self.coll.gather(self.target.end(state, last_row))
            return (scores, full) if self.rank == 0 else None
        local = self.target.end(state, rows)
        if local is None:
            return None
        full = self.coll.gather(local)
        return full if self.rank == 0 else None

    def _score_rows(self, local, targets):
        """Per row: (target log probability, top log probabilities, top ids) merged across the vocabulary shards
        (each rank sends its log-sum-exp, target logit and top entries; the Gemma 4 SM70 provider's packet)."""
        import torch
        rows, shard = local.shape
        base = self.rank * shard
        lse = torch.logsumexp(local, -1)
        t = torch.tensor(targets, dtype=torch.int64, device=local.device) - base
        inside = (t >= 0) & (t < shard)
        tl = torch.where(inside, local.gather(1, t.clamp(0, shard - 1)[:, None])[:, 0],
                         torch.full_like(lse, float("-inf")))
        tv, ti = torch.topk(local, SCORE_TOP, -1)
        packet = torch.cat([lse[:, None], tl[:, None], tv, (ti + base).float()], 1).contiguous()
        recv = torch.empty((self.world,) + tuple(packet.shape), dtype=torch.float32, device=local.device)
        self.comm.all_gather(packet, recv)
        glse = torch.logsumexp(recv[:, :, 0], 0)
        target = recv[:, :, 1].max(0).values - glse
        values = recv[:, :, 2:2 + SCORE_TOP].permute(1, 0, 2).reshape(rows, -1)
        index = recv[:, :, 2 + SCORE_TOP:].permute(1, 0, 2).reshape(rows, -1)
        top, order = torch.topk(values, SCORE_TOP, -1)
        return target.tolist(), (top - glse[:, None]).tolist(), index.gather(1, order).long().tolist()

    def _run(self, op, a, b, c, d, ids):
        verifier = getattr(self, "_draft_verify", None)
        if verifier is not None and verifier.pending is not None and op != OP_DRAFT_COMMIT_BATCH:
            raise RuntimeError("DeepSeek DSpark must settle its pending vector before another control operation")
        if op == OP_ATTACH:
            return self._attach(a, b)
        if op == OP_DETACH:
            return self._detach(a)
        if op == OP_STEP:
            n = 3 * a
            entries = [tuple(ids[i:i + 3]) for i in range(0, n, 3)]
            tokens = ids[n:n + b]
            rows = ids[n + b:n + b + c]
            if d & F_DRAFT_TAPS:
                if len(ids) != n + b + c + 1 or d & F_SAMPLE:
                    raise ValueError("invalid DeepSeek DSpark tap frame")
                return self._step(entries, tokens, rows, d & ~F_DRAFT_TAPS, ids[-1])
            if d & F_SAMPLE:
                from .device_sampling import WORDS

                if len(ids) != n + b + c + WORDS * a:
                    raise ValueError("invalid DeepSeek sampled decode frame")
                return self._step(entries, tokens, rows, d & ~F_SAMPLE, sample=list(ids[n + b + c:]))
            return self._step(entries, tokens, rows, d)
        if op == OP_LAYERS:
            return self._layers(a)
        if op == OP_TOWER:
            return self._tower(a, b, c, d, ids)
        if op == OP_EXPERT_PLACEMENT and self.expert_live is not None:
            receipt = self.expert_live.handle(a, b, c, d, ids)
            if self.rank != 0:
                self.pipes.acknowledge(receipt)
            return receipt
        if op in (OP_DRAFT_VERIFY_BATCH, OP_DRAFT_COMMIT_BATCH):
            return self._run_draft_vector(op, a, b, c, d, ids)
        if op in PREFIX_OPS:
            out = self.prefix.run(op, a, b, c, d, ids)
            if op == OP_GROW:                   # the slot gained pages: the graphs' device page table follows
                self.slot_pages.set(a, self.target.pool.owned[a])
            return out
        raise RuntimeError(f"unknown DeepSeek-V4.1 control operation {op}")

    def _check_draft_vector(self, op, sequence, count, c, d, ids):
        """c: lanes whose full rows rank 0 gathers (grammar); d: lanes drawing keyed tokens on the device, whose
        ``device_sampling`` words follow the entries in lane order (verification only; disjoint from c)."""
        from .device_sampling import WORDS

        verifier = getattr(self, "_draft_verify", None)
        width = 3 if op == OP_DRAFT_VERIFY_BATCH else 2
        verify = op == OP_DRAFT_VERIFY_BATCH
        if (verifier is None or not 1 <= count <= verifier.bucket or sequence < 0
                or (not (0 <= c < (1 << count) and 0 <= d < (1 << count) and not c & d) if verify
                    else c != 0 or d != 0)
                or len(ids) != count * width + (WORDS * d.bit_count() if verify else 0)):
            raise ValueError("invalid DeepSeek DSpark control frame")
        if op == OP_DRAFT_VERIFY_BATCH and self.part is not None:
            paused_slots = self.part[2]
            # A paused forward owns its activations and the KV/history
            # of these slots. The other slots' verify and commit may run
            # between groups on the same stream, as ordinary decode does.
            if any(ids[i] in paused_slots for i in range(0, len(ids), width)):
                raise RuntimeError("DeepSeek DSpark verification overlaps a paused prompt slot")
        return verifier, width

    def _run_draft_vector(self, op, sequence, count, c, d, ids):
        from .dspark_graph import VerifyEntry

        verifier, width = self._check_draft_vector(op, sequence, count, c, d, ids)
        rows = tuple(tuple(ids[i:i + width]) for i in range(0, count * width, width))
        if op == OP_DRAFT_VERIFY_BATCH:
            from .device_sampling import WORDS, lane_noise

            words = ids[count * width:]
            lanes = [i for i in range(count) if d >> i & 1]
            noise = {i: lane_noise(words[k * WORDS:(k + 1) * WORDS]) for k, i in enumerate(lanes)}
            output = verifier.verify_batch(sequence, tuple(VerifyEntry(*row, noise=noise.get(i))
                                                           for i, row in enumerate(rows)))
            if d:
                output = self._sample_draft_rows(output, rows, d, ids[count * width:], verifier.depth + 1)
            if c:
                width = verifier.depth + 1
                if c == (1 << count) - 1:
                    local = output.local_logits[:count * width]
                else:
                    import torch

                    local = torch.cat([output.local_logits[i * width:(i + 1) * width]
                                       for i in range(count) if c & (1 << i)], dim=0)
                full = self.coll.gather(local)
                if self.rank == 0:
                    output = output._replace(full_logits=full)
            return output
        verifier.commit_batch(sequence, rows)
        # commit_batch fences the local device and checks sticky errors first.
        if self.rank != 0:
            self.pipes.acknowledge(_HEAD.pack(op, sequence, count, c, d) + array.array("i", ids).tobytes())
        return None

    def _sample_draft_rows(self, output, entries, lanes, words, width):
        """Every rank: the keyed draw of each row of the ``lanes`` bitmask's lanes (row j of a lane at position
        p samples p + 1 + j, the serial decoder's key), from the vector's local logits and gathered packet."""
        from .device_sampling import WORDS

        chosen = [i for i in range(len(entries)) if lanes >> i & 1]
        index = self.sampler.upload([i * width + j for i in chosen for j in range(width)])
        positions = self.sampler.upload([entries[i][1] + 1 + j for i in chosen for j in range(width)])
        per_row = [w for k in range(len(chosen)) for _ in range(width) for w in words[k * WORDS:(k + 1) * WORDS]]
        tokens = self.sampler.sample(output.local_logits.index_select(0, index).contiguous(),
                                     output.compact[:, index, 0].max(0).values, positions,
                                     self.sampler.words(per_row, len(index)))
        return output._replace(sampled=tokens) if self.rank == 0 else output

    def announce(self, op, a=0, b=0, c=0, d=0, ids=()):
        verifier = getattr(self, "_draft_verify", None)
        if verifier is not None and verifier.pending is not None and op != OP_DRAFT_COMMIT_BATCH:
            raise RuntimeError("DeepSeek DSpark must settle its pending vector before another control operation")
        if op in (OP_DRAFT_VERIFY_BATCH, OP_DRAFT_COMMIT_BATCH):
            payload = array.array("i", ids)
            self._check_draft_vector(op, a, b, c, d, payload)
            frame = _HEAD.pack(op, a, b, c, d) + payload.tobytes()
            try:
                self.pipes.send(frame)
                result = self._run(op, a, b, c, d, payload)
                if op == OP_DRAFT_COMMIT_BATCH:
                    # Publication is permitted only after all seven followers
                    # echo the exact sequence and ordered slot/count payload.
                    if self.pipes.acknowledgments() != [frame] * (self.world - 1):
                        raise RuntimeError("invalid DeepSeek DSpark commit acknowledgment")
                return result
            except BaseException as primary:
                from tensorfold.cuda.cohort import notify_failure
                from tensorfold.cleanup import raise_failures

                self.pipes.broken = True
                try:
                    notify_failure()
                except BaseException as cleanup:
                    raise_failures(primary, [cleanup])
                raise
        self.pipes.send(_HEAD.pack(op, a, b, c, d) + array.array("i", ids).tobytes())
        return self._run(op, a, b, c, d, list(ids))

    def follow(self) -> None:
        import torch

        watch = self.target.rotor_cache is not None
        pending = False
        try:
            with torch.inference_mode():
                while True:
                    try:
                        frame = self.pipes.receive(ROTOR_ERROR_POLL_S if pending else PIPE_TIMEOUT)
                    except TimeoutError:
                        if pending:
                            pending = not torch.cuda.current_stream().query()
                        continue
                    if frame is None:
                        verifier = getattr(self, "_draft_verify", None)
                        if verifier is not None and verifier.pending is not None:
                            raise RuntimeError("DeepSeek DSpark control peer departed with an unsettled vector")
                        return
                    op, a, b, c, d = _HEAD.unpack_from(frame)
                    if op == OP_STOP:
                        verifier = getattr(self, "_draft_verify", None)
                        if verifier is not None and verifier.pending is not None:
                            raise RuntimeError("DeepSeek DSpark stopped with an unsettled verification vector")
                        return
                    payload = array.array("i")
                    payload.frombytes(frame[_HEAD.size:])
                    self._run(op, a, b, c, d, list(payload))
                    verifier = getattr(self, "_draft_verify", None)
                    pending = watch or (verifier is not None and verifier.pending is not None)
        except BaseException as primary:
            if watch or getattr(self, "expert_live", None) is not None or getattr(self, "_draft_verify", None) is not None:
                from tensorfold.cuda.cohort import notify_failure
                from tensorfold.cleanup import raise_failures

                try:
                    notify_failure()
                except BaseException as cleanup:
                    raise_failures(primary, [cleanup])
            raise

    # ---------------------------------------------------------------------------------------- leader
    def _publish_expert_proposal(self, proposal):
        def publish():
            with self._expert_proposal_lock:
                epoch, plan = proposal
                if self._expert_refused != (epoch, plan.proposed):
                    self._expert_proposal = proposal

        if not self._closed:
            try:
                self.scheduler.request_runtime(publish)
            except RuntimeError:
                if not self._closed:
                    raise

    def _expert_maintenance(self, closing, occupied):
        from .expert_rebalance_wire import PeerRefusal

        with self._expert_proposal_lock:
            if closing:
                self._expert_proposal = None
                return False
            if self._expert_proposal is None:
                return False
            if occupied:
                return True     # Existing requests can progress; new admissions wait at the worker boundary.
            proposal, self._expert_proposal = self._expert_proposal, None
        epoch, plan = proposal
        if self.expert_live.epoch != epoch or self.expert_live.placement != plan.current:
            return False        # A previous successful application superseded this queued proposal.

        try:
            self.decoder.drain_placement_work()
            self.expert_observer.pause_for_placement()
            try:
                applied = self.expert_live.apply(plan, expected_epoch=epoch, history=self.expert_observer.history)
            except PeerRefusal as refusal:
                with self._expert_proposal_lock:
                    self._expert_refused = (epoch, plan.proposed)
                print(f"[tensorfold] {refusal}", flush=True)
            else:
                if applied:
                    print(f"[tensorfold] DeepSeek live expert placement epoch {self.expert_live.epoch} committed", flush=True)
            self.expert_observer.resume_after_placement()
        except BaseException as error:
            self.decoder.broken = error
            raise
        return False

    def recapture_expert_graphs(self):
        """Rebuild the same optional decode catalogue while scheduler admission is held."""
        if self.graphs is not None:
            raise RuntimeError("expert placement still owns the previous graph catalogue")
        if self._expert_graph_keys is None:
            return
        import torch
        from .graphs import DecodeGraphs

        self.graphs = DecodeGraphs(self, self.streams)
        spare = torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()
        if (tuple(self.graphs.keys) != self._expert_graph_keys or self.graphs.bytes > self._expert_graph_bytes
                or spare < self._expert_prompt_reserve):
            raise RuntimeError("rebuilt expert graphs do not meet the original catalogue and memory budget")
        if self._draft_verify is not None:
            self._capture_draft_graphs(self._expert_prompt_reserve)

    def _capture_draft_graphs(self, prompt_reserve, *, before=None, state=0):
        import torch

        if before is None:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            before = torch.cuda.mem_get_info()[0]
        self._draft_verify.capture()
        torch.cuda.empty_cache()
        spare = torch.cuda.mem_get_info()[0]
        used = before - spare - state
        if used > self._draft_graph_budget or spare < prompt_reserve:
            raise RuntimeError(f"DeepSeek DSpark rank {self.rank}: B1/B2/B3 captures took {used >> 20} MiB "
                               f"(sized {self._draft_graph_budget >> 20} MiB); {spare >> 20} MiB left, "
                               f"{prompt_reserve >> 20} MiB required for prompt transients")

    def step(self, entries, ids, logits_rows, stop=0, *, tap_from=None, sample=None):
        payload = [v for e in entries for v in e] + list(ids) + list(logits_rows)
        if tap_from is not None:
            stop |= F_DRAFT_TAPS
            payload.append(tap_from)
        if sample is not None:
            if tap_from is not None:
                raise ValueError("a sampled decode round has no draft taps")
            stop |= F_SAMPLE
            payload.extend(sample)
        observer = self.expert_observer
        if observer is not None:
            # Polling history can fail. Prepare the sample before notifying peers to enter the TP8 forward.
            phase = int(bool(stop & F_COMPACT))
            stop |= observer.sample(phase, may_copy=self.part is None) << EXPERT_SAMPLE_SHIFT
        return self.announce(OP_STEP, len(entries), len(ids), len(logits_rows), stop, payload)

    # the vision tower (every rank runs its TP part; ``_tower``); handles are rank 0's
    def tower_begin(self, frame) -> int:
        from .vision import pinned_patches
        # The same handle travels in T_STAGE/T_FREE's signed int32 payload. The leader can reuse a retired
        # handle because every rank applies tower operations in control-pipe order on its serving stream.
        live = len(self.tower_states) + len(self.tower_spans)
        handle = next(i for i in range(live + 1) if i not in self.tower_states and i not in self.tower_spans)
        if handle >= (1 << 31):
            raise OverflowError("DeepSeek-V4.1 vision exhausted control-frame tower handles")
        pixels = pinned_patches(frame)
        self.pipes.send(_HEAD.pack(OP_TOWER, T_BEGIN, handle, frame.n_vit_h, frame.n_vit_w))
        self._tower(T_BEGIN, handle, frame.n_vit_h, frame.n_vit_w, (), pixels)
        return handle

    def tower_run(self, handle: int, stop: int) -> None:
        self.announce(OP_TOWER, T_RUN, handle, stop)

    def tower_span(self, handle: int) -> None:
        self.announce(OP_TOWER, T_SPAN, handle)

    def tower_stage(self, pieces) -> None:
        """pieces: [(handle, first, end, release)] -> every rank's ``image_features`` for the next F_IMAGE step."""
        self.announce(OP_TOWER, T_STAGE, ids=[v for piece in pieces for v in piece])

    def tower_free(self, handles) -> None:
        if handles:
            self.announce(OP_TOWER, T_FREE, ids=list(handles))

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], Any],
                 draft: bool = True, stop_eos: bool = True, *, vision=None, constraint=None,
                 background: bool = False, progress=None, penalties=None, cache_prompt: bool = True,
                 probabilities=None, **_ignored) -> dict[str, Any]:
        if self.rank != 0:
            raise RuntimeError("only rank 0 generates")
        if vision is not None:
            from .vision import PreparedVision
            if self.tower is None:
                raise ValueError("image input requires the server started with --vision")
            if type(vision) is not PreparedVision or tuple(prompt) != vision.token_ids:
                raise ValueError("DeepSeek-V4.1 image prompts must be the vision frontend's prepared prompt")
        if not isinstance(prompt, (list, tuple)) or not prompt or any(
                type(t) is not int or not 0 <= t < self.vocab for t in prompt):
            raise ValueError("DeepSeek-V4.1 needs a nonempty list of in-vocabulary prompt token IDs")
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        if len(prompt) + max_tokens > self.context:
            raise ValueError(f"prompt and reply exceed the {self.context}-token admitted context")
        if self._closed or self.scheduler is None:
            raise RuntimeError("DeepSeek-V4.1 SM70 engine is not serving")
        options = {"background": True} if background else {}
        if penalties is not None and penalties.active:
            options["penalties"] = penalties
        # This flag carries both draft and prompt reuse eligibility in the CUDA scheduler.
        return self.scheduler.submit(list(prompt), max_tokens, sampling, bool(draft and cache_prompt), on_tokens,
                                     stop_eos=stop_eos, vision=vision,
                                     constraint=constraint, probabilities=probabilities, progress=progress, **options)

    def drain(self) -> None:
        """Requests retire inside the scheduler; nothing to fence for a caller."""

    def greedy(self, prompt: list[int], max_tokens: int, slot: int = 0, on_token: Callable | None = None,
               stop_eos: bool = True, top_logprobs: int = 0, eager: bool = False) -> dict[str, Any]:
        """Single-stream greedy generation (the M1 correctness path): prompt chunks, then one token a step.
        With ``top_logprobs`` > 0 the result also carries each step's chosen-token logprob and its top entries.
        ``eager``: decode steps run the eager forward instead of the startup graphs (A/B checks)."""
        import torch
        if self.rank != 0:
            raise RuntimeError("only rank 0 generates")
        with torch.inference_mode():
            self.announce(OP_ATTACH, slot, len(prompt) + max_tokens)
            t0 = time.perf_counter()
            at = 0
            logits = None
            while at < len(prompt):
                n = min(self.max_tokens, len(prompt) - at)
                last = at + n == len(prompt)
                logits = self.step([(slot, at, n)], prompt[at:at + n], [n - 1] if last else [])
                at += n
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            out, logprobs = [], []

            def choose(row):
                if top_logprobs:
                    lp = torch.log_softmax(row.float(), -1)
                    values, ids = lp.topk(top_logprobs)
                    chosen = int(ids[0])
                    logprobs.append({"logprob": float(values[0]),
                                     "top": [[int(i), float(v)] for i, v in zip(ids.tolist(), values.tolist())]})
                    return chosen
                return int(row.argmax())

            token = choose(logits[-1])
            while True:
                out.append(token)
                if on_token is not None:
                    on_token(token)
                if len(out) >= max_tokens or (stop_eos and token in self.eos):
                    break
                logits = self.step([(slot, len(prompt) + len(out) - 1, 1)], [token], [0],
                                   self.units if eager else 0)
                token = choose(logits[-1])
            t2 = time.perf_counter()
            self.announce(OP_DETACH, slot)
        return {"tokens": out, "logprobs": logprobs, "prefill_s": t1 - t0, "decode_s": t2 - t1,
                "decode_tok_s": (len(out) - 1) / (t2 - t1) if len(out) > 1 else 0.0}

    def score(self, prompt: list[int], continuation: list[int], slot: int = 0) -> list[float]:
        """Teacher-forced log probabilities of ``continuation`` after ``prompt`` (rank 0; prompt chunks as greedy)."""
        import torch
        if self.rank != 0:
            raise RuntimeError("only rank 0 scores")
        tokens = list(prompt) + list(continuation)
        first = len(prompt) - 1                      # row whose logits predict continuation[0]
        out = []
        with torch.inference_mode():
            self.announce(OP_ATTACH, slot, len(tokens))
            at = 0
            while at < len(tokens) - 1:
                n = min(self.max_tokens, len(tokens) - 1 - at)
                rows = [r for r in range(n) if at + r >= first]
                logits = self.step([(slot, at, n)], tokens[at:at + n], rows)
                if rows:
                    targets = torch.tensor([tokens[at + r + 1] for r in rows], device=logits.device)
                    lp = torch.log_softmax(logits.float(), -1).gather(1, targets[:, None])[:, 0]
                    out += lp.tolist()
                at += n
            self.announce(OP_DETACH, slot)
        return out

    def close(self) -> None:
        import torch
        from tensorfold.cleanup import finish

        if self._closed:
            return
        self._closed = True

        def retire_device():
            # A run-ahead graph can still be queued after the scheduler stops. Fence readers before releasing
            # graphs, pinned history snapshots, or the Engram map. A failed fence retains their owners.
            torch.cuda.synchronize()
            self.graphs = None
            self._draft_verify = None
            operations = []
            if self.expert_observer is not None:
                operations.append(self.expert_observer.finish)
            if self.engram is not None:
                operations.append(self.engram.close)
            finish(operations)

        if self.rank != 0:
            retire_device()
            return

        scheduler = self.scheduler
        joined = scheduler is None
        stopped = False

        def join():
            nonlocal joined
            scheduler.close()
            joined = True

        def stop():
            nonlocal stopped
            verifier = getattr(self, "_draft_verify", None)
            if verifier is not None and verifier.pending is not None:
                raise RuntimeError("cannot stop DeepSeek DSpark with an unsettled verification")
            self.pipes.send(_HEAD.pack(OP_STOP, 0, 0, 0, 0))
            stopped = True

        def retire_if_quiescent():
            # Scheduler.close can report an earlier request failure after verified worker retirement.
            if not joined and self.expert_observer is not None:
                if scheduler._worker_done.is_set() and scheduler._cleanup_failure is None:
                    with scheduler._worker_lease:
                        pass
                else:
                    return
            elif not joined:
                return
            if stopped:
                self.scheduler = None
                retire_device()

        finish((lambda: finish((join, stop) if scheduler is not None else (stop,)), retire_if_quiescent))


class Decoder(PrefixOwner):
    """The shared Scheduler's decoder on rank 0: slots, prompt chunks and batched decode rounds.

    A round decides the batch in flight, launches the next batch over every decoding stream (one row each), then
    delivers the decided tokens and runs the next prompt chunk of the next filling request (round-robin; chunks
    split at most ``max_tokens`` apart and at the prompt's kept message boundaries: a prompt's arithmetic never
    depends on load). While streams decode, the chunk runs in groups of units (one per round; a unit is a layer's
    attention or its experts, half a layer, so the groups fit the gap where whole layers of an 8192-row chunk do
    not), each group cut when it starts from the measured costs (``schedule``), so a decoding stream waits one group
    plus its round (about ``GAP_MS``), not the whole chunk (the Gemma 4 SM70 Decoder's schedule).

    Slots and their pages come from the prompt-prefix owner (``prefix.PrefixOwner``, the Gemma 4 SM70 / Flash Next
    policy): admission places a request in a kept chain (resume in place, fork, host tier) or cold; prompt chunks
    keep points at their boundaries and the exact prompt end; a finished stream's slot stays idle while it keeps
    points."""

    def __init__(self, engine: DS41Engine):
        import torch
        from tensorfold.cuda.geometry import DECODE_GAP_MS
        self.e = engine
        self.decode_math = F_W576_FP32 if getattr(engine, "_experimental_w576_fp32", False) else 0
        self.GAP_MS = DECODE_GAP_MS
        self.streams: dict = {}
        self.filling: list = []
        self.next_id = 0
        self._broken = None
        self.eos = tuple(engine.eos)
        self.inflight = None                         # the launched decode batch (``_launch``), oldest first
        self.ahead = None                            # a run-ahead batch launched behind ``inflight`` (``_launch_ahead``)
        self.split = None                            # (stream, chunk, next unit) of a paused chunk
        self.finishing = None                        # (stream, chunk, result) of a chunk whose last unit was launched
                                                     # (None, None, None) keeps only that round's launch ordering
        self.costs = UnitCosts(*engine.warm_unit_ms)  # per-unit prompt cost model (CUDA events of past chunks)
        self.rounds = RoundCosts()                   # GPU time of a decode round per batch bucket (CUDA events)
        self.timing = None                           # (rows, per-group marks) of the chunk in flight
        self.timed = []                              # [(rows, marks)] of finished chunks, oldest first, until read
        # round timing events, alternating (two decode batches in flight under run-ahead)
        self.events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(2)]
        self.flip = 0
        self._init_prefix(engine, engine.points)

    @property
    def broken(self):
        return self._broken

    @broken.setter
    def broken(self, error):
        """Fail closed: after a failed operation the ranks may disagree; the TP supervisor retires the cohort."""
        if error is None or self._broken is not None:
            return
        self._broken = error
        print(f"[tensorfold] DeepSeek-V4.1 SM70 cohort failed: {type(error).__name__}: {error}; retiring every rank",
              flush=True)
        from tensorfold.cuda.cohort import notify_failure
        try:
            notify_failure()
        except Exception as failure:          # noqa: BLE001 - the original failure stays the served error
            print(f"[tensorfold] TP supervisor notification failed: {failure}", flush=True)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def drain_placement_work(self) -> None:
        """Fence the last DeepSeek graph and retire its metadata at a maintenance boundary."""
        import torch

        self._check()
        engine = self.e
        if (self.live() or self.ahead is not None or self.split is not None or engine.part is not None
                or self.timing is not None or engine.layer_marks is not None
                or engine.image_features is not None or engine.tower_states or engine.tower_spans
                or self.finishing not in (None, (None, None, None))):
            raise RuntimeError("expert placement requires drained prompt, decode, and image work")
        # An EOS/cancellation can retire the last logical stream after its successor was already submitted.
        # The scheduler cannot rely on another decoding round to observe that successor's completion.
        if self.inflight is not None and self._collect():
            raise RuntimeError("expert placement found a live decision at the drained decode boundary")
        torch.cuda.current_stream(engine.target.device).synchronize()
        self.finishing = None
        self._absorb()
        if self.timed:
            raise RuntimeError("expert placement found unread prompt events after the device fence")

    def usage(self) -> dict:
        """/stats' KV-memory and prefix-cache readings."""
        budget = self.e.budget
        active = sum(self._bytes(s.st) for s in (*self.streams.values(), *self.filling) if s.st is not None)
        per_token = self.e.token_bytes
        return {"kv": {"budget_bytes": budget, "held_bytes": self.held, "active_bytes": active,
                       "usage_ratio": round(active / budget, 6) if budget else None, "bytes_per_token": per_token,
                       "free_tokens_estimate": max(0, budget - active) // per_token},
                "prefix_cache": self.prefix_usage()}

    def _check(self):
        if self._broken is not None:
            raise RuntimeError("DeepSeek-V4.1 SM70 ranks may be out of step after an error; restart the server") \
                from self._broken

    def _ends(self, s):
        return self.eos if s.stop_eos else ()

    # ------------------------------------------------------------------------------------------ admission
    def admit(self, s) -> None:
        from tensorfold.cuda.memory_gate import NoRoom
        from tensorfold.engine.penalties import History
        self._check()
        e = self.e
        if s.vision is not None:
            # Image spans carry content-derived ids >= vocab in the prompt the prefix cache keys on and the forwards
            # carry (``vision.prefix_ids``), so a kept point covers its images' pixels, not just their token count.
            from .vision import prefix_ids
            if e.tower is None:
                raise ValueError("image input requires the server started with --vision")
            if hasattr(s.probabilities, "add_prompt"):
                raise ValueError("DeepSeek-V4.1 SM70 scores text prompts only")
            s.prompt = prefix_ids(s.vision, e.vocab)
            s.encoded = {}                 # frame index -> its span, while a later chunk still needs rows of it
            s.encoding = None              # (frame index, the tower's paused encode)
        if len(s.prompt) + s.count > e.context:
            raise ValueError(f"prompt and reply exceed the {e.context}-token admitted context")
        capacity = self._cap(len(s.prompt) + s.count)
        if e.slot_bytes(capacity) > e.budget:
            raise ValueError(f"a {len(s.prompt)}-token prompt with {s.count} reply tokens needs "
                             f"{-(-capacity // 64)} KV pages per GPU; {e.pages} exist")
        # Prompt reuse: eligible requests without media; a teacher-forcing sink only when it asks to borrow (it
        # scores the rows it computes). Sampling does not change prompt state.
        reuse = bool(s.draft)
        if hasattr(s.probabilities, "add_prompt") and not getattr(s.probabilities, "reuse", False):
            reuse = False
        try:
            self.place(s, capacity, reuse)
        except NoRoom:
            raise
        except BaseException as error:
            self.broken = error
            raise
        s.sid, self.next_id = self.next_id, self.next_id + 1
        s.history = (History(s.penalties, list(s.prompt if s.vision is None else s.vision.token_ids))
                     if s.penalties is not None and s.penalties.active else None)
        s.forced = getattr(s.probabilities, "forced", None)
        self._admitted(s)
        self.filling.append(s)

    def _admitted(self, s):
        if TRACE_DRAFT:
            s.dspark_admission = draft_blockers(s)
            s.dspark_shadow = True

    def cancel(self, gone) -> list:
        """Filling requests whose clients left end now (their slots retire in ``finish``)."""
        out = []
        for s in gone:
            if self.split is not None and self.split[0] is s:
                self.split = self.timing = self.e.layer_marks = None
                self.e.announce(OP_LAYERS, -1)
            if self.finishing is not None and self.finishing[0] is s:
                # A one-row prompt may have replayed a graph after the decode batch in flight. Keep the launch
                # barrier for that batch, but release the cancelled prompt's result and skip its completion.
                self.finishing = (None, None, None) if self.inflight is not None else None
            if s in self.filling:
                self._release_tower_handles(s)
                out.append(s)
        # Fallible cohort operations precede retirement. On failure drop() still finds every cancelled request.
        for s in out:
            self.filling.remove(s)
            s.done, s.finished = True, time.perf_counter()
        return out

    def finish(self, done) -> None:
        for s in done:
            self.streams.pop(getattr(s, "sid", None), None)
            if s in self.filling:
                if self._broken is None:
                    try:
                        self._release_tower_handles(s)
                    except BaseException as error:
                        self.broken = error
                        raise
                self.filling.remove(s)
            if getattr(s, "st", None) is None:
                continue
            s.slot = None
            if self._broken is None:
                try:
                    self.retire_slot(s)
                except BaseException as error:
                    self.broken = error
                    raise

    def drop(self, error=None) -> list:
        """After an error in a round: forget every live stream; the ranks can no longer be trusted to agree."""
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.streams, self.filling, self.inflight, self.ahead, self.split = {}, [], None, None, None
        self.finishing = self.timing = self.e.layer_marks = None
        self.timed = []
        if self._broken is None:
            self.broken = error or RuntimeError("a DeepSeek-V4.1 SM70 round failed")
        return live

    # ------------------------------------------------------------------------------------------ rounds
    def round(self) -> list:
        """Wait for the batch in flight and decide its tokens, launch the next batch at once, then (while it runs)
        deliver the decided tokens and run the next prompt chunk or layer group. Run-ahead: when the batch in flight
        can only continue unchanged (``_can_run_ahead``), its successor is launched before the wait, its token ids
        taken on the device from the batch in flight (graphs ``ahead``), so the GPU never idles on this host work.

        A prompt chunk whose last unit the previous round launched finishes here (``finishing``), after this round's
        decided tokens are delivered: reading its result waits for that unit, which runs behind the batch just
        decided, so a decoding stream's token never waits for the chunk's last group as well (two groups and a
        round). The next batch launches after it, with the stream that just filled."""
        import torch
        self._check()
        done = []
        with torch.inference_mode():
            try:
                finishing, self.finishing = self.finishing, None
                if self.inflight is not None and finishing is None and self._can_run_ahead():
                    self._launch_ahead()
                decided = self._collect() if self.inflight is not None else []
                self.inflight, self.ahead = self.ahead, None
                ending = {id(s) for s, chosen in decided if chosen is None or self._ends_after(s, chosen)}
                live = [s for s in self.streams.values() if not s.done and id(s) not in ending]
                if live and self.inflight is None and finishing is None:
                    self._launch(live, {id(s): chosen for s, chosen in decided if chosen is not None})
                for s, chosen in decided:
                    if chosen is not None:
                        s.take(chosen, self._ends(s))
                    if s.done:
                        done.append(s)
                if finishing is not None:
                    if finishing[0] is not None:
                        done += self._filled(*finishing)
                    live = [s for s in self.streams.values() if not s.done and id(s) not in ending]
                    if live and self.inflight is None:
                        self._launch(live, {})          # every decided token is delivered: none pending
                if self.filling or self.split is not None:
                    done += self._fill()
            except BaseException as error:
                self.broken = error
                raise
        return done

    def _ends_after(self, s, chosen):
        return len(s.out) + len(chosen) >= s.count or (s.stop_eos and chosen[-1] in self.eos)

    def _fill(self) -> list:
        e = self.e
        if self.split is not None:                  # the paused chunk continues first (one paused forward at most)
            s, chunk, at = self.split
            stop = self._stop(chunk[1] - chunk[0], at)
            t0 = time.perf_counter()
            out = e.announce(OP_LAYERS, stop)
            s.prefill_s += time.perf_counter() - t0
            if stop < e.units:
                self.split = (s, chunk, stop)
                return []
            self.split = None
            self._timed()
            self.finishing = (s, chunk, out)        # read next round, after its decided tokens (``round``)
            return []
        foreground = [x for x in self.filling if not x.background]
        s = (foreground or self.filling)[0]
        n, start = len(s.prompt), s.pos
        if s.head is not None:                     # an exact repeat: the kept prompt end's own last-row logits
            logits, s.head = s.head, None
            return self._filled(s, (start, n, True, None, False), logits)
        stop = chunk_end(s.bounds, start, n)       # the splits placement chose (prefix.chunk_splits)
        last = stop == n
        image = 0
        if s.vision is not None:
            t0 = time.perf_counter()
            ready = self._encode(s, start, stop)
            s.prefill_s += time.perf_counter() - t0
            if not ready:                          # an encode group ran this round; the chunk waits for the rest
                return []
            image = F_IMAGE if any(a < stop and b > start for a, b in s.vision.blocks) else 0
        self.keep_before(s, start)
        self._absorb()
        self.timing = (stop - start, [])
        e.layer_marks = self.timing[1]
        unit = self._stop(stop - start, 0)
        sink = s.probabilities if hasattr(s.probabilities, "add_prompt") else None
        layer = unit if unit < e.units else 0
        t0 = time.perf_counter()
        if sink is not None:
            # every row scored for its next prompt token (a last chunk's final row against the first forced token,
            # unused) in the serving chunk's own forward; a last chunk also returns its last row
            after = s.prompt[stop] if not last else (s.forced[0] if s.forced is not None else 0)
            out = e.step([(s.slot, start, stop - start)], s.prompt[start:stop], s.prompt[start + 1:stop] + [after],
                         layer | F_SCORE | (F_LAST if last else 0))
        else:
            out = self._text_chunk(s, start, stop, last, layer | image)
        s.prefill_s += time.perf_counter() - t0
        if unit < e.units:
            self.split = (s, (start, stop, last, sink, True), unit)
            return []
        self._timed()
        self.finishing = (s, (start, stop, last, sink, True), out)
        return []

    def _text_chunk(self, s, start, stop, last, flags, **options):
        return self.e.step([(s.slot, start, stop - start)], s.prompt[start:stop],
                           [stop - start - 1] if last else [], flags, **options)

    def _filled(self, s, chunk, out) -> list:
        """Prompt chunk ``chunk`` (start, stop, last, scoring sink, computed: False for an exact repeat's kept
        logits) of ``s`` is done (``out``: its result)."""
        start, stop, last, sink, computed = chunk
        n = len(s.prompt)
        logits = out
        if sink is not None:                   # scored chunk: scores, or (scores, last row) for a last chunk
            (tl, tv, ti), logits = out if last else (out, None)
            for r in range(stop - start - (1 if last else 0)):
                pos = start + r + 1
                sink.add_prompt([pos], [s.prompt[pos]], [tl[r]], [ti[r][:sink.top]], [tv[r][:sink.top]])
        if last and computed and s.keep_end:
            self.keep_end(s, logits)
        s.pos = stop
        self.filling.remove(s)
        if s.on_fill is not None:
            s.on_fill(stop)
        if not last:
            self.filling.append(s)                     # round-robin: back of the line
            return []
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        self.streams[s.sid] = s
        try:
            first = self._choose(s, logits[-1:], n)
        except Exception as error:          # noqa: BLE001  (this request fails alone)
            s.error, s.done, s.finished = error, True, time.perf_counter()
            return [s]
        s.take([first], self._ends(s))
        return [s] if s.done else []

    def _encode(self, s, start: int, stop: int) -> bool:
        """The image rows of prompt chunk [start, stop) of ``s`` into every rank's ``e.image_features`` (None: no
        image row in it). Frames are encoded (every rank its TP part of the tower) in tower blocks; while streams
        decode, one group of blocks a round, each group's predicted cost at most the decode gap (as prompt units
        are cut), so an image never stalls decoding streams for its whole encode. Returns False while an encode is
        unfinished. Only spans a later chunk still reads stay (``s.encoded``: frame index -> span handle)."""
        from tensorfold.cuda.geometry import group_target
        from .vision import LAYERS
        e = self.e
        need = [i for i, (a, b) in enumerate(s.vision.blocks) if a < stop and b > start]
        for i in need:
            if i in s.encoded:
                continue
            frame = s.vision.frames[i]
            if s.encoding is None or s.encoding[0] != i:
                if s.encoding is not None:
                    e.tower_free([s.encoding[1]])
                s.encoding = (i, e.tower_begin(frame), 0)
            _, handle, done = s.encoding
            end = LAYERS
            if self.streams:
                block = e.tower_block_ms * (frame.n_vit_h * frame.n_vit_w) / e.tower_patches
                target = group_target(self.GAP_MS, self.rounds.estimate(self._bucket(len(self.streams))))
                end = min(LAYERS, done + max(1, int(target // max(block, 1e-3))))
            e.tower_run(handle, end)
            if end < LAYERS:
                s.encoding = (i, handle, end)
                return False
            e.tower_span(handle)
            s.encoded[i], s.encoding = handle, None
            if self.streams:                       # the aligner closed this round's group
                return False
        pieces = []
        for i in need:
            a, b = s.vision.blocks[i]
            pieces.append((s.encoded[i], max(a, start) - a, min(b, stop) - a, int(b <= stop)))
            if b <= stop:
                del s.encoded[i]
        if pieces:
            e.tower_stage(pieces)
        return True

    def _release_tower_handles(self, s) -> None:
        """Drop the tower encodes and spans an unfinished image request still holds (every rank)."""
        if getattr(s, "vision", None) is None:
            return
        handles = list(s.encoded.values()) + ([s.encoding[1]] if s.encoding is not None else [])
        self.e.tower_free(handles)
        s.encoded, s.encoding = {}, None

    def _stop(self, rows, at) -> int:
        """The unit after which the prompt chunk of ``rows`` rows, now at unit ``at``, pauses next: every remaining
        unit with no stream decoding; otherwise the first of equal-cost groups of the rest, each at most a gap
        less the next decode round (``layer_groups``), planned again every round from the measured costs, this
        chunk's measured progress and the batch now decoding. The pause changes no kernel, so it may follow load."""
        from tensorfold.cuda.geometry import group_target, layer_groups
        e = self.e
        if not self.streams:
            return e.units
        cost = self.costs.predict(rows)
        done = [(first, events) for first, events in self.timing[1] if events[-1].query()]
        if done:                                    # this chunk so far: measured / predicted (context, load)
            spent = sum(events[0].elapsed_time(events[-1]) for _, events in done)
            planned = sum(sum(cost[first:first + len(events) - 1]) for first, events in done)
            if planned > 0:
                cost = [c * spent / planned for c in cost]
        round_ms = self._prefill_round_ms()
        stop = at + layer_groups(cost[at:], group_target(self.GAP_MS, round_ms))[0]
        if TRACE:
            print(f"[ds41 sched] rows {rows} units {at}-{stop} predicted {sum(cost[at:stop]):.1f} ms round "
                  f"{round_ms:.1f} ms streams {len(self.streams)}", flush=True)
        return stop

    def _prefill_round_ms(self):
        return self.rounds.estimate(self._bucket(len(self.streams)))

    def _timed(self) -> None:
        """The chunk in flight was launched to its end: its marks wait for ``_absorb`` (never waits on the GPU)
        behind any earlier chunk's still unread ones. The queue holds at most the chunks launched since the host
        last waited on the GPU: a round's decode wait or a last chunk's result read completes every earlier one, so
        only a prompt's own chunks filling with no stream decoding (at most context / MAX_TOKENS) stay unread."""
        self.timed.append(self.timing)
        self.timing = self.e.layer_marks = None

    def _absorb(self) -> None:
        """Per-unit costs of the finished chunks whose events completed, oldest first (never waits; the rest stay
        queued). No measurement is dropped: one that had not completed when the next chunk began would otherwise
        be lost, and under load every chunk run whole is still running then."""
        while self.timed and all(events[-1].query() for _, events in self.timed[0][1]):
            self._sample(*self.timed.pop(0))

    def _sample(self, rows, marks) -> None:
        ms = [None] * self.e.units
        for first, events in marks:
            for j in range(1, len(events)):
                ms[first + j - 1] = events[j - 1].elapsed_time(events[j])
        if None not in ms:
            self.costs.add(rows, ms)
            if TRACE:
                print(f"[ds41 sched] chunk rows {rows} groups {len(marks)} {sum(ms):.1f} ms "
                      f"(attention {sum(ms[0::2]):.1f}, experts {sum(ms[1::2]):.1f})", flush=True)

    def _bucket(self, count) -> int:
        """The decode batch a round over ``count`` streams runs as (its graph bucket; eager: ``count``)."""
        graphs = self.e.graphs
        if graphs is None or count > graphs.sizes[-1]:
            return count
        return next(b for b in graphs.sizes if b >= count)

    def _launch(self, live, pending):
        """Launch one decode forward over ``live`` (row order = stream order); ``pending``: decided, undelivered
        tokens. Plain greedy streams decide from the graph's compact packet; the full rows are gathered only for the
        others, whose sampling work is queued behind the forward (``round_sampling``): the round's one host wait
        (``_collect``) covers every stream."""
        from .round_sampling import RowSampling, needs_row
        e = self.e
        entries, ids = [], []
        for s in live:
            extra = pending.get(id(s), ())
            token = extra[-1] if extra else s.out[-1]
            entries.append((s.slot, len(s.prompt) + len(s.out) + len(extra) - 1, 1))
            ids.append(token)
        graph = e.graphs is not None and len(entries) <= e.graphs.sizes[-1]
        need = [g for g, s in enumerate(live) if needs_row(s, graph)]
        started, done = self._events()
        started.record()
        t0 = time.perf_counter()
        out = e.step(entries, ids, list(range(len(entries))), F_COMPACT | (F_FULL if need else 0) | self.decode_math,
                     sample=self._sample_words(live) if graph else None)
        compact, rows, tokens = out if isinstance(out, tuple) else (None, out, None)   # eager: full rows only
        if compact is None:
            need = list(range(len(live)))
        positions = [entry[1] + 1 for entry in entries]
        sampled = RowSampling(live, positions, rows, need) if need else None
        done.record()
        self.inflight = (live, positions, compact, rows, sampled, t0, (started, done, self._bucket(len(entries))),
                         entries, tokens)

    @staticmethod
    def _sample_words(live):
        """Every row's device_sampling words when a row draws on the device, else None (no sampler work)."""
        from .device_sampling import row_words
        from .round_sampling import device_sampled

        chosen = [s.sampling if device_sampled(s) else None for s in live]
        if all(c is None for c in chosen):
            return None
        return [w for c in chosen for w in row_words(c)]

    def _events(self):
        pair = self.events[self.flip]
        self.flip ^= 1
        return pair

    def _can_run_ahead(self) -> bool:
        """The batch in flight continues unchanged one more round whatever tokens it decides: every stream plain
        greedy (its token is the packet's argmax: ``round_sampling.plain``), still decoding, with at least two reply
        tokens left (no count-based end after this batch), the batch a graph replay and no stream added since its
        launch. A stream that ends on an end token in the batch in flight wastes its run-ahead row: that row writes the
        stream's state one position past its reply, inside its admitted pages, as a reply token would (the slot is
        retired behind it in stream order; prefix points never lie past a prompt end)."""
        from .round_sampling import device_decided
        live, compact, entries = self.inflight[0], self.inflight[2], self.inflight[7]
        if compact is None or self.e.graphs is None:
            return False
        current = [s for s in self.streams.values() if not s.done]
        if len(current) != len(live) or any(a is not b for a, b in zip(current, live)):
            return False
        return all(device_decided(s) and len(s.out) + 2 <= s.count for s in live)

    def _launch_ahead(self) -> None:
        """Launch the run-ahead successor of the batch in flight: same streams and rows, one position on, token ids
        on the device (F_AHEAD)."""
        e = self.e
        live, positions, old_entries = self.inflight[0], self.inflight[1], self.inflight[7]
        entries = [(slot, position, 1) for (slot, _, _), position in zip(old_entries, positions)]
        started, done = self._events()
        started.record()
        t0 = time.perf_counter()
        compact, rows, tokens = e.step(entries, [0] * len(entries), list(range(len(entries))),
                                       F_COMPACT | F_AHEAD | self.decode_math, sample=self._sample_words(live))
        done.record()
        self.ahead = (live, [entry[1] + 1 for entry in entries], compact, rows, None, t0,
                      (started, done, self._bucket(len(entries))), entries, tokens)

    def _collect(self):
        """Wait for the batch in flight; decide each stream's token: [(stream, [token] | None)]."""
        from .round_sampling import device_sampled, greedy_tokens
        live, positions, compact, rows, sampled, t0, (started, done, bucket), _, tokens = self.inflight
        self.inflight = None
        done.synchronize()
        self.rounds.add(bucket, started.elapsed_time(done))
        greedy = greedy_tokens(compact, len(live)) if compact is not None else None
        chosen_rows = sampled.choose() if sampled is not None else {}
        decided = []
        for g, s in enumerate(live):
            if s.done or self.streams.get(s.sid) is not s:
                continue                                    # ended/cancelled/yielded after this launch
            try:
                if s.forced is not None:
                    token = s.forced[len(s.out)]
                elif g in chosen_rows:
                    token = chosen_rows[g]
                else:
                    token, error = greedy[g]            # the packet's NaN/+inf/finite checks cover every row
                    if error is not None:
                        raise error
                    if device_sampled(s):
                        token = int(tokens[g])
                if isinstance(token, Exception):
                    raise token
                if s.probabilities is not None:
                    self._score(s, sampled.scored[g], positions[g] - 1, token)
                if s.constraint is not None:
                    s.constraint.advance([token])
                if s.history is not None:
                    s.history.extend([token])
                s.counted(1)
            except Exception as error:          # noqa: BLE001  (this request fails alone)
                s.error, s.done, s.finished = error, True, time.perf_counter()
                decided.append((s, None))
                continue
            decided.append((s, [token]))
        return decided

    def _choose(self, s, row, position, emitted=0) -> int:
        """Sample (or force) the token at ``position`` from one full-vocabulary row; advance grammar/penalties."""
        from tensorfold.cuda.sampling import sample_rows
        if s.forced is not None:
            token = s.forced[emitted]
        else:
            if s.constraint is not None:
                row = row.clone()
                s.constraint.mask(row)
            if s.history is not None:
                row = s.history.apply(row, copy=s.constraint is None)
            token = sample_rows(row, [position], s.sampling)[0]
        if s.probabilities is not None:
            self._score(s, row, position - 1, token)
        if s.constraint is not None:
            s.constraint.advance([token])
        if s.history is not None:
            s.history.extend([token])
        return token

    @staticmethod
    def _score(s, logits, position, token):
        """Probabilities sink: log probability of ``token`` after ``position`` (+ top)."""
        import torch
        sink = s.probabilities
        lp = torch.log_softmax(logits[-1:].float(), -1)[0]
        if sink.top:
            values, ids = torch.topk(lp, sink.top)
            tops, topv = ids.tolist(), values.tolist()
        else:
            tops, topv = [], []
        nxt = position + 1
        if nxt < sink.start:
            return
        sink.add([nxt], [token], [float(lp[token])], [tops], [topv])
        if hasattr(sink, "origins"):
            sink.origins([nxt], ["prompt-end" if nxt == len(s.prompt) else (0, 1)])


class FixedDraftDecoder(Decoder):
    """D3 serving with one complete verify and commit per scheduler turn."""

    def __init__(self, engine):
        super().__init__(engine)
        self.verify_rounds_by_active = [0] * (engine._draft_verify.bucket + 1)
        self._draft_round_ms = None

    def _admitted(self, s):
        # A kept prefix has target state only. Every row of the last draft
        # window must be recomputed for this slot before the first proposal.
        blockers = draft_blockers(s)
        s.draft_seed = not blockers
        if TRACE_DRAFT:
            s.dspark_admission = blockers

    def _text_chunk(self, s, start, stop, last, flags, **options):
        if s.draft_seed:
            from .dspark_head import WINDOW

            first = max(start, len(s.prompt) - WINDOW)
            if first < stop:
                options["tap_from"] = first - start
        return super()._text_chunk(s, start, stop, last, flags, **options)

    def _draft_entry(self, s):
        from .dspark_weights import BLOCK as DRAFT_BLOCK
        from .target import BLOCK

        e = self.e
        at = len(s.prompt) + len(s.out) - 1
        if (at + DRAFT_BLOCK > e.context
                or at + e._draft_verify.depth + 1 > len(e.target.pool.owned[s.slot]) * BLOCK):
            if TRACE_DRAFT:
                s.dspark_fallback = "insufficient_draft_room"
            return None
        return s.slot, at, s.out[-1]

    def _decode_groups(self):
        ready, ordinary, cancelled = [], [], []
        for s in self.streams.values():
            if s.done:
                continue
            if s.cancel is not None and s.cancel.is_set():
                s.done, s.finished = True, time.perf_counter()
                cancelled.append(s)
                continue
            entry = self._draft_entry(s) if s.draft_seed and s.out else None
            if entry is None:
                ordinary.append(s)
            else:
                ready.append((s, entry))
        if len(ready) + len(ordinary) > self.e._draft_verify.bucket:
            # More decoding streams than one verify vector holds: a vector would idle the waiting lanes or add a
            # second forward per turn, both slower than one batched ordinary round over every stream. Ordinary
            # decode retires their draft rings (``_launch``).
            ordinary.extend(s for s, _ in ready)
            ready = []
        return ready, ordinary, cancelled

    def _launch(self, live, pending):
        for s in live:
            # Ordinary continuation does not populate the draft ring.
            if TRACE_DRAFT and s.draft_seed and getattr(s, "dspark_fallback", None) is None:
                s.dspark_fallback = "ordinary_decode"
            s.draft_seed = False
        super()._launch(live, pending)

    def round(self):
        self._check()
        finishing = self.finishing
        joining = finishing[0] if finishing is not None else None
        if not ((joining is not None and joining.draft_seed)
                or any(s.draft_seed and not s.done for s in self.streams.values())):
            # Preserve the target-only run-ahead path when no draft lane can
            # join this turn.
            return super().round()
        import torch

        try:
            with torch.inference_mode():
                done = []
                if self.inflight is not None:
                    for s, chosen in self._collect():
                        if chosen is not None:
                            s.take(chosen, self._ends(s))
                        if s.done:
                            done.append(s)
                if finishing is not None:
                    self.finishing = None
                    if joining is not None:
                        done += self._filled(*finishing)
                ready, ordinary, cancelled = self._decode_groups()
                done.extend(cancelled)
                filling = bool(self.filling) or self.split is not None
                if ready:
                    live, entries = zip(*ready, strict=True)
                    start = time.perf_counter() if filling else None
                    done += self._fixed_round(live, entries)
                    if start is not None:
                        self._draft_round_ms = (time.perf_counter() - start) * 1000
                        if ordinary:
                            self._draft_round_ms += self.rounds.estimate(self._bucket(len(ordinary)))
                if ordinary:
                    self._launch(ordinary, {})
                if filling:
                    done += self._fill()
                return done
        except BaseException as error:
            self.broken = error
            raise
        finally:
            self._draft_round_ms = None

    def _prefill_round_ms(self):
        if self._draft_round_ms is not None:
            return self._draft_round_ms
        return super()._prefill_round_ms()

    def _fixed_round(self, live, entries):
        from .device_sampling import eligible, row_words
        from .dspark_serving import decide, decide_constrained, decide_sampled
        from .round_sampling import greedy_tokens

        e = self.e
        owner = e._draft_verify
        sequence, depth = owner.next_sequence, owner.depth
        count, width = len(entries), depth + 1
        # Sampled lanes draw keyed tokens on the device (``device_sampling``); grammar lanes and other policies
        # read their full rows (gathered in vector order); greedy lanes the compact packet.
        device = [s.constraint is None and eligible(s.sampling) for s in live]
        full = sum(1 << i for i, s in enumerate(live)
                   if not device[i] and (s.constraint is not None or s.sampling is not None))
        sampled = sum(1 << i for i in range(count) if device[i])
        words = [w for s, on in zip(live, device) if on for w in row_words(s.sampling)]
        output = e.announce(OP_DRAFT_VERIFY_BATCH, sequence, count, full, sampled,
                            ids=[*(value for entry in entries for value in entry), *words])
        choices = greedy_tokens(output.compact[:, :count * width].cpu(), count * width)
        drawn = iter(output.sampled.tolist()) if sampled else None
        proposals = output.drafts[:count].tolist()
        resolved = []
        full_at = 0
        for i, (s, (_, at, anchor), drafts) in enumerate(zip(live, entries, proposals, strict=True)):
            chosen = error = None
            rows = choices[i * width:(i + 1) * width]
            if device[i]:                       # the packet's NaN/+inf/finite errors stay with each row
                rows = [(next(drawn), error) for _, error in rows]
            logits = None
            if full >> i & 1:
                logits = output.full_logits[full_at:full_at + width]
                full_at += width
            if s.cancel is None or not s.cancel.is_set():
                try:
                    room, eos = s.count - len(s.out), self._ends(s)
                    if logits is None:
                        chosen = decide(rows, anchor, drafts, room, eos)
                    elif s.sampling is not None:
                        chosen = decide_sampled(s.sampling, logits, anchor, drafts, at, room, eos, s.constraint)
                        if s.constraint is not None and (s.cancel is None or not s.cancel.is_set()):
                            s.constraint.advance(chosen)
                    else:
                        accepted = decide_constrained(s.constraint, logits, rows, anchor, drafts, room, eos)
                        if s.cancel is None or not s.cancel.is_set():
                            s.constraint.advance(accepted)
                            chosen = accepted
                except Exception as failed:  # Roll back this slot and settle the rest of the vector.
                    error = failed
            resolved.append((chosen, error))
        for i, s in enumerate(live):
            if s.cancel is not None and s.cancel.is_set():
                resolved[i] = (None, None)
        e.announce(OP_DRAFT_COMMIT_BATCH, sequence, count,
                   ids=(value for (slot, _, _), (chosen, _) in zip(entries, resolved, strict=True)
                        for value in (slot, len(chosen) if chosen is not None else 0)))
        self.verify_rounds_by_active[count] += 1

        finished = []
        for s, (chosen, error), drafts in zip(live, resolved, proposals, strict=True):
            if chosen is None:
                s.error, s.done, s.finished = error, True, time.perf_counter()
            else:
                if len(chosen) <= depth and chosen[-1] == drafts[len(chosen) - 1]:
                    s.accepted += 1
                s.counted(width)
                s.take(chosen, self._ends(s))
            if s.done:
                finished.append(s)
        return finished
