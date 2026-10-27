"""The Flash Next CUDA engine: shared model execution with lockstep tensor-parallel peers."""

from __future__ import annotations

import json
import threading
import time
from copy import deepcopy
from datetime import timedelta
from pathlib import Path
from typing import Any, Callable

from tensorfold.cuda import prompt_precision
from tensorfold.cleanup import finish, raise_failures, rollback
from . import CONFIDENCE, DEPTH
from .retention_policy import RetentionPolicy

MAX_DEPTH = 15           # a verify window of at most 16 rows
KEEP_SERIAL = 4          # prompt states the serial engine keeps (they share its attention rows)
KEEP = 8                 # prompt states (one token before each end) a concurrent decoder keeps to resume from
# Reserve bounded tower workspace separately from its weights; override for measured deployments.
VISION_WORKSPACE = 4 * 2**30


def vision_workspace() -> int:
    import os

    value = os.environ.get("TENSORFOLD_VISION_WORKSPACE_MIB")
    if value is None or value == "":
        return VISION_WORKSPACE
    if not value.isdecimal() or int(value) > 16384:
        raise ValueError(f"TENSORFOLD_VISION_WORKSPACE_MIB: 0 to 16,384 MiB, not {value!r}")
    return int(value) * 2**20


def _sm70_prefill_gdn_bytes(text: dict, world: int, rows: int) -> int:
    """One prompt buffer's FP32 keys/gates and FP16 values, shared across GDN layers."""

    nk, nv = int(text["linear_num_key_heads"]) // world, int(text["linear_num_value_heads"]) // world
    dk, dv = int(text["linear_key_head_dim"]), int(text["linear_value_head_dim"])
    return rows * (nk * dk * 4 + nv * dv * 2 + nv * 8)


def _expert_authority(cache) -> tuple[str, dict | None]:
    """Publish the completed loader's registration result, including auto refusals."""
    policy = cache.pin_policy
    if policy is None:
        return "pageable host authority", None
    receipt = policy.receipt()
    pinned = receipt["pinned_bytes"]
    if pinned == cache.host_bytes:
        label = "pinned host authority"
    elif not pinned:
        label = "pageable host authority"
    else:
        label = f"host authority ({pinned / 2**30:.2f} GiB pinned, {(cache.host_bytes - pinned) / 2**30:.2f} GiB pageable)"
    return label, receipt


class FlashNextEngine:
    """``eos``, ``generate`` (rank 0 or one GPU) and ``follow`` (peer ranks), as the CUDA server expects."""

    def __init__(self, model_dir: Path, *, depth: int=DEPTH, confidence: float=CONFIDENCE, draft_vocab: str | int | None='default', max_len: int | None=None, context_explicit: bool | None=None, tp: int=1, rank: int=0, master: str='', port: int=29551, prefetch: bool=True, graphs: bool=True, streams: int=1, ple_on_ssd: bool=False, kv_dtype: str='bf16', share: float=0.0, vision: bool=False, vision_urls: bool=False, yarn_factor: float | None=None, vram_experts: float | str | None=None, kv_key_dtype: str | None=None, kv_value_dtype: str | None=None, expert_session=None, kv_full_windows: int | None=None, vram_experts_pin: str="off", eos_token_ids=None, chat_template_file=None, chat_template_name=None, mtp_model_dir: Path | None=None, dspark_model_dir: Path | None=None, dflash2_model_dir: Path | None=None, indexer_history: str="gpu", adaptive_sessions: bool=False, session_ceiling: int | None=None, execution_rows: int | None=None, execution_profile_manifest: Path | None=None, position_profile_manifest: Path | None=None, media_profile_manifest: Path | None=None, family_manifest: Path | None=None, _prepared_family_plan=None, serving_profile_root: Path | None=None, max_waiting_input_bytes: int | None=None, max_waiting_input_bytes_per_principal: int | None=None, output_queue_tokens: int | None=None, queue_timeout_s: float | None=None, retention_policy: RetentionPolicy | None=None) -> None:
        if serving_profile_root is not None:
            from .serving_profiles import profile_root

            serving_profile_root = profile_root(serving_profile_root)
            if not adaptive_sessions or indexer_history != "gpu" or expert_session is not None or vram_experts is not None:
                raise ValueError("serving profiles require resident adaptive Flash Next GGUF GPU histories")
        profile_options = dict(locals())
        profile_options.pop("self")
        profile_options.pop("model_dir")
        profile_options.pop("serving_profile_root")
        profile_options.pop("_prepared_family_plan")  # private startup cache is never an administrator option
        self._profile_build = None
        self.serving_profiles = None
        self._lifecycle = threading.Condition()
        self._closing = self._closed = self._close_running = False
        self._calls: dict[object, int] = {}
        self._abort_comm = False
        self._receiving = False
        self._shutdown_sent = False
        self.comm = self.vision = self.w = self.scheduler = self.multi = self.e = self.serial = None
        self.retention_catalog = None
        self._published_retention_catalog = None
        self._catalog_callback_leases = {}
        self._startup_store = None
        self._startup_store_origin = None
        self._store_use_libuv = None  # Prepared once for the lifetime of an adaptive unique-profile transport.
        self._dflash_pairing = None
        self.expert_session = expert_session
        self._indexer_history_session = None
        self._owner_preparation = None
        self.position_profile = self.position_method_binding = self._position_prepared = None
        self.coordinate_profile_binding = None
        self._media_inputs_enabled = False  # Only the admitted media-profile parser may enable this.
        self.media_profile = self._media_authority = self._media_software_binding = self._media_numeric_scope = None
        self._startup_reads: list = []
        self._prepared_family_plan = _prepared_family_plan
        try:
            self._initialize(model_dir, depth=depth, confidence=confidence, draft_vocab=draft_vocab, max_len=max_len, context_explicit=context_explicit, tp=tp, rank=rank, master=master, port=port, prefetch=prefetch, graphs=graphs, streams=streams, ple_on_ssd=ple_on_ssd, kv_dtype=kv_dtype, share=share, vision=vision, vision_urls=vision_urls, yarn_factor=yarn_factor, vram_experts=vram_experts, kv_key_dtype=kv_key_dtype, kv_value_dtype=kv_value_dtype, expert_session=expert_session, kv_full_windows=kv_full_windows, vram_experts_pin=vram_experts_pin, eos_token_ids=eos_token_ids, chat_template_file=chat_template_file, chat_template_name=chat_template_name, mtp_model_dir=mtp_model_dir, dspark_model_dir=dspark_model_dir, dflash2_model_dir=dflash2_model_dir, indexer_history=indexer_history, adaptive_sessions=adaptive_sessions, session_ceiling=session_ceiling, execution_rows=execution_rows, execution_profile_manifest=execution_profile_manifest, family_manifest=family_manifest, position_profile_manifest=position_profile_manifest, media_profile_manifest=media_profile_manifest, max_waiting_input_bytes=max_waiting_input_bytes, max_waiting_input_bytes_per_principal=max_waiting_input_bytes_per_principal, output_queue_tokens=output_queue_tokens, queue_timeout_s=queue_timeout_s, retention_policy=retention_policy)
            if serving_profile_root is not None:
                from .serving_runtime import ServingProfiles

                self.serving_profiles = ServingProfiles(self, serving_profile_root, profile_options)
        except BaseException as primary:
            self._abort_comm = True
            rollback(self, primary, self._rollback_startup)

    def _prepare_store_backend(self, implemented_admission: bool, *, rank: int, world: int) -> None:
        """Bind the live transport policy before model/device acquisition."""
        selected = False if implemented_admission else None
        if self._store_use_libuv is not None and self._store_use_libuv is not selected:
            raise ValueError("the prepared TCPStore backend is process-fixed across serving profiles")
        self._store_use_libuv = selected
        if self._profile_build is not None and selected is False:
            from tensorfold.cuda.comm import validate_store_origin

            validate_store_origin(self.comm.store_origin, self.comm.store, rank, world, verified=True)

    def _initialize(self, model_dir: Path, *, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 draft_vocab: str | int | None = "default", max_len: int | None = None,
                 context_explicit: bool | None = None, tp: int = 1, rank: int = 0, master: str = "", port: int = 29551,
                 prefetch: bool = True, graphs: bool = True, streams: int = 1, ple_on_ssd: bool = False,
                 kv_dtype: str = "bf16", share: float = 0.0, vision: bool = False, vision_urls: bool = False,
                 yarn_factor: float | None = None, vram_experts: float | str | None = None,
                 kv_key_dtype: str | None = None, kv_value_dtype: str | None = None,
                 kv_full_windows: int | None = None, expert_session=None, vram_experts_pin: str="off",
                 eos_token_ids=None, chat_template_file=None, chat_template_name=None,
                 mtp_model_dir: Path | None = None, dspark_model_dir: Path | None = None,
                 dflash2_model_dir: Path | None = None,
                 indexer_history: str = "gpu", adaptive_sessions: bool = False,
                 session_ceiling: int | None = None, execution_rows: int | None = None,
                 execution_profile_manifest: Path | None = None, family_manifest: Path | None = None, position_profile_manifest: Path | None = None, media_profile_manifest: Path | None = None,
                 max_waiting_input_bytes: int | None = None,
                 max_waiting_input_bytes_per_principal: int | None = None,
                 output_queue_tokens: int | None = None, queue_timeout_s: float | None = None,
                 retention_policy: RetentionPolicy | None = None) -> None:
        requested_graphs = graphs
        from tensorfold.cuda.chat_template import prepare_chat_template

        from ..ram_experts import check_kv_full_windows
        from tensorfold.server.request_limits import optional_limit

        optional_limit(max_waiting_input_bytes, "max_waiting_input_bytes")
        optional_limit(max_waiting_input_bytes_per_principal, "max_waiting_input_bytes_per_principal")
        optional_limit(output_queue_tokens, "output_queue_tokens")
        if queue_timeout_s is not None and (type(queue_timeout_s) not in (int, float)
                or not 0 < queue_timeout_s <= threading.TIMEOUT_MAX):
            raise ValueError("queue_timeout_s must be positive, finite and within the runtime wait range")
        if retention_policy is not None:
            if type(retention_policy) is not RetentionPolicy:
                raise ValueError("retention_policy must be an explicitly parsed RetentionPolicy")
            retention_policy.validate_world(tp)
        self.retention_policy = retention_policy

        self.position_profile = self.position_method_binding = self._position_prepared = None
        self.coordinate_profile_binding = None
        self._media_inputs_enabled = False  # Only the admitted media-profile parser may enable this.
        if indexer_history not in ("gpu", "ram"):
            raise ValueError("indexer history must be gpu or ram")
        self.indexer_history = indexer_history
        # Reject malformed counts without reading checkpoint metadata.
        if kv_full_windows is not None and (type(kv_full_windows) is not int or kv_full_windows < 1):
            check_kv_full_windows(kv_full_windows, streams=streams, vram_experts=vram_experts, tp=tp)
        self.kv_full_windows = kv_full_windows
        from .. import rope_parameters
        from ..kv_formats import get_pair
        self.kv_pair = get_pair(kv_dtype, kv_key_dtype, kv_value_dtype)
        from ..gguf import resolve_gguf
        source_is_gguf = resolve_gguf(model_dir) is not None
        self.model_dir = Path(model_dir)
        media_requested = media_profile_manifest is not None
        if media_requested and (not vision or graphs or not source_is_gguf or tp != 8
                or not adaptive_sessions or indexer_history != "gpu" or dspark_model_dir is not None
                or vram_experts is not None or expert_session is not None):
            raise ValueError("media profile requires --vision, --no-graphs, resident TP8 adaptive GGUF GPU histories and no DSpark")
        nvfp4_source = family_manifest is not None
        self.family_manifest = Path(family_manifest) if nvfp4_source else None
        self._sm70_import_plan = self._sm70_weight_plan = None
        self._sm70_weight_manifest_identity = None
        self._sm70_weight_workspace = None
        self._sm70_weight_preparation = None
        sm70 = source_is_gguf or nvfp4_source
        if source_is_gguf and nvfp4_source:
            raise ValueError("family_manifest requires its declared NVFP4 source, independently of GGUF")
        if nvfp4_source and (vision or execution_profile_manifest is not None):
            raise ValueError("this explicit SM70 NVFP4 recipe supports text reference execution; vision/measured kernels need separate qualification")
        if execution_profile_manifest is not None and not source_is_gguf:
            raise ValueError("execution profile manifests require the resident GGUF SM70 provider")
        if nvfp4_source and indexer_history != "gpu":
            raise ValueError("the explicit SM70 NVFP4 profile requires the shared compact GPU indexer policy; full raw RAM archives are unsupported")
        if nvfp4_source:
            from tensorfold.token_metadata import tokenizer_override

            if chat_template_file is not None or chat_template_name is not None or tokenizer_override() is not None:
                raise ValueError("SM70 NVFP4 numerical manifest binds the source's default tokenizer/template")
        if nvfp4_source and int(depth) > 0 and mtp_model_dir is None:
            raise ValueError("SM70 NVFP4 requires --no-drafts or its validated external stock MTP companion")
        if nvfp4_source:
            import os
            from tensorfold.cuda import precision
            from tensorfold.cuda.sm70.execution_policy import PROFILE_ENV

            if precision.mode() != precision.FULL:
                raise ValueError("SM70 NVFP4 requires explicit --precision full and its numerical manifest")
            if os.environ.get(PROFILE_ENV, "reference") != "reference":
                raise ValueError("SM70 NVFP4 has no qualified optimized execution profile; select reference")
            graphs = False  # Whole-forward graph qualification is separate for this numerical profile.
        if position_profile_manifest is not None and not sm70:
            raise ValueError("position profile manifests require an explicit Flash Next SM70 provider")
        compact_kv = any(side.compact is not None for side in (self.kv_pair.key, self.kv_pair.value))
        if compact_kv and not sm70:
            raise ValueError("compact Planar4/Iso4 KV requires the GGUF SM70 execution provider")
        if type(adaptive_sessions) is not bool or adaptive_sessions and not sm70:
            raise ValueError("automatic resident sessions require the GGUF SM70 provider")
        if session_ceiling is not None and (type(session_ceiling) is not int or session_ceiling < 1):
            raise ValueError("session ceiling must be a positive integer or automatic")
        if execution_rows is not None and (type(execution_rows) is not int or execution_rows < 1):
            raise ValueError("execution rows must be a positive integer")
        if adaptive_sessions and indexer_history != "gpu":
            raise ValueError("automatic resident sessions currently require GPU indexer history")
        concurrent = streams > 1 or adaptive_sessions
        self.adaptive_sessions = adaptive_sessions
        if indexer_history == "ram" and not sm70:
            raise ValueError("RAM indexer history requires the GGUF SM70 execution provider")
        if not adaptive_sessions:
            check_kv_full_windows(kv_full_windows, streams=streams, vram_experts=vram_experts, tp=tp,
                                  resident_gguf=sm70)
        elif kv_full_windows is not None:
            session_ceiling = min(session_ceiling, kv_full_windows) if session_ceiling is not None else kv_full_windows
        resident_windows = kv_full_windows if sm70 and vram_experts is None else None
        if mtp_model_dir is not None and not sm70:
            raise ValueError("an external Flash Next MTP checkpoint is supported by the GGUF SM70 loader only")
        if dspark_model_dir is not None and (mtp_model_dir is not None or not source_is_gguf):
            raise ValueError("DSpark requires the GGUF SM70 target and a separate DSpark drafter selection")
        self.mtp_model_dir = Path(mtp_model_dir) if mtp_model_dir is not None else None
        self.dspark_model_dir = Path(dspark_model_dir) if dspark_model_dir is not None else None
        self._dspark_plan = None
        self.dflash2_model_dir = Path(dflash2_model_dir) if dflash2_model_dir is not None else None
        self._dflash2_plan = None
        if dflash2_model_dir is not None and (mtp_model_dir is not None or dspark_model_dir is not None
                or not source_is_gguf or tp != 8 or not adaptive_sessions or indexer_history != "gpu"
                or vision and not media_requested or execution_profile_manifest is None or int(depth) > 3):
            raise ValueError("DFlash2 requires one exact pairing manifest, TP8 text-only adaptive GGUF GPU "
                             "histories, no other draft owner, and at most three proposals")
        if dflash2_model_dir is not None and draft_vocab not in (None, "default"):
            raise ValueError("DFlash2 uses the manifest-bound full target vocabulary; MTP draft_vocab does not apply")
        supported_tp = (1, 2, 4, 8) if sm70 else (1, 2)
        if type(tp) is not int or tp not in supported_tp or type(rank) is not int or rank not in range(tp):
            raise ValueError(f"rank {rank} of {tp}: Flash Next supports tensor-parallel sizes {supported_tp} "
                             f"for {'GGUF SM70' if sm70 else 'this checkpoint'}")
        if sm70:
            from .sm70_kv import check_pair

            self._sm70_kv_policy = check_pair(self.kv_pair)
        if not sm70 and any(side.name == "fp16" for side in (self.kv_pair.key, self.kv_pair.value)):
            raise ValueError("fp16 KV caches require the GGUF SM70 execution provider")
        if sm70 and (vram_experts is not None or expert_session is not None):
            raise ValueError("GGUF SM70 currently retains packed experts on each tensor-parallel device; "
                             "--vram-experts and separate expert ranks are not supported by this loader")
        if concurrent and tp > 1 and not sm70:
            raise ValueError("--parallel decodes several Flash Next requests together on one GPU; with --tp 2 it "
                             "serves one request at a time for now, so drop --parallel")
        if not 0 <= int(depth) <= MAX_DEPTH:
            raise ValueError(f"MTP drafts a round: 0 to {MAX_DEPTH}, not {depth}")
        if not 0.0 <= float(confidence) <= 1.0:
            raise ValueError(f"MTP draft confidence: a probability from 0 to 1, not {confidence}")
        if vision and not media_requested and (streams < 2 or tp != 1):
            raise ValueError("image input on Flash Next runs on one GPU with --parallel 2 or more")
        if vram_experts_pin not in ("off", "auto", "on") or vram_experts_pin != "off" and vram_experts is None:
            raise ValueError("expert pinning requires a RAM expert budget and off, auto or on")
        if tp > 1 and not master:
            raise ValueError("tensor-parallel ranks need rank 0's address (master)")
        if vram_experts is not None:
            from ..ram_experts import check

            check(model_dir, vram_experts, tp=tp)
        self.chat_template = prepare_chat_template(model_dir, template_file=chat_template_file,
                                                   template_name=chat_template_name)
        import torch
        from .exl3_pack import admission, extra_files, is_exl3
        from tensorfold.families import quant_method, read_config
        exl3 = is_exl3(model_dir)
        modelopt = not (exl3 or nvfp4_source) and quant_method(read_config(model_dir)) == "modelopt"
        if (exl3 or modelopt) and tp != 1:
            raise ValueError(f"{'EXL3 packs' if exl3 else 'NVFP4 checkpoints'} of Flash Next run on one GPU: drop --tp "
                             "2, or serve the MLX checkpoint (TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP) on two")
        if exl3 and ple_on_ssd:
            raise ValueError("--ple-on-ssd reads the MLX checkpoint's n-gram tables; an EXL3 pack maps its own table "
                             "from its file, so drop --ple-on-ssd")
        self.rope = rope_parameters(model_dir, yarn_factor)
        from .decode import Engine
        from .prompt_plan import POINTS, choose as prompt_plan
        from .weights import draft_token_ids, load
        from tensorfold.cuda.capacity import admit, config, gather_ints
        from tensorfold.cuda.geometry import (PREFILL_ROWS, gdn_geometry, indexed_prefill_rows,
                                              indexed_stream_geometry, indexed_weights, multi_graph_request)
        chunk = None if is_exl3(model_dir) else indexed_prefill_rows()
        text = config(model_dir)
        from .history_placement import HistoryPlacement

        layout_kv_heads = int(text["num_key_value_heads"])
        if source_is_gguf:
            from ..gguf import metadata_config

            actual_kv_heads = metadata_config(model_dir)["text_config"]["num_key_value_heads"]
            if layout_kv_heads != actual_kv_heads:
                raise ValueError("declared KV geometry differs from authoritative GGUF layout")
            layout_kv_heads = actual_kv_heads
        unique_positions = (sm70 and adaptive_sessions and indexer_history == "gpu"
                            and HistoryPlacement.supports(tp, layout_kv_heads))
        position_method = ("dflash2" if self.dflash2_model_dir is not None else
                           "dspark" if self.dspark_model_dir is not None and int(depth) > 0 else
                           "mtp" if int(depth) > 0 else "none")
        if position_profile_manifest is not None and (not unique_positions or position_method == "dspark"):
            raise ValueError("position profile manifests require supported unique TP8 GPU histories and an implemented method role")
        if unique_positions and position_method != "dspark":
            from ..position_profile import prepare_position_vectors, resolve_position_profile

            companions = {}
            if position_method == "mtp" and self.mtp_model_dir is not None:
                companions["mtp"] = self.mtp_model_dir
            if self.dflash2_model_dir is not None:
                companions["dflash2"] = self.dflash2_model_dir
            self.position_profile = resolve_position_profile(model_dir,
                requested_context=max_len if max_len not in (None, 0) else None, yarn_factor=yarn_factor,
                manifest_path=position_profile_manifest, companion_configs=companions)
            self.rope = self.position_profile.target.rope
            if position_profile_manifest is not None:
                # A selected manifest commits its declared rows; default/0 uses
                # those rows, never a fitted or silently shortened allowance.
                max_len, context_explicit = self.position_profile.target.max_rows, True
            lookahead = 3 if position_method == "dflash2" else int(depth)
            self.position_method_binding = self.position_profile.validate_method(
                position_method, self.kv_pair.identity, lookahead)
            self._position_prepared = prepare_position_vectors(self.position_profile, mtp_active=position_method == "mtp")
        media_resources = None
        if media_requested:
            from .media_profile import read_media_profile
            from .media_runtime import media_resources as profile_resources

            if not unique_positions or self.position_profile is None:
                raise ValueError("media profile requires the actual unique-history multimodal position provider")
            self.media_profile = read_media_profile(media_profile_manifest, model_dir=Path(resolve_gguf(model_dir)).parent,
                target_hidden=int(text["hidden_size"]), world=tp,
                context_rows=self.position_profile.target.max_rows if max_len in (None, 0) else max_len)
            media_resources = profile_resources(self.media_profile, hidden=int(text["hidden_size"]), world=tp)
        if nvfp4_source:
            from dataclasses import replace
            from .sm70_nvfp4_weights import Planner, model_plan
            from .weight_types import Config

            prepared = self.__dict__.pop("_prepared_family_plan", None)
            if prepared is None:
                prepared = model_plan(model_dir, self.family_manifest, (rank, tp),
                                      rope=self.rope, eos_token_ids=eos_token_ids)
            else:
                if (not isinstance(prepared, tuple) or len(prepared) != 2
                        or not isinstance(prepared[1], Planner)
                        or prepared[1].directory != Path(model_dir).resolve()
                        or prepared[0].meta.get("rank") != rank or prepared[0].meta.get("world") != tp):
                    raise ValueError("cached SM70 family plan belongs to another source/rank geometry")
                prepared[1].manifest.check_cached(self.family_manifest, model_dir)
                expected = replace(Config.read(model_dir, rope=self.rope, eos_token_ids=eos_token_ids),
                                   quant="sm70_nvfp4", execution_dtype="float16")
                if prepared[0].meta["full"] != expected:
                    raise ValueError("cached SM70 family plan differs from selected RoPE/EOS/configuration")
            if self.position_profile is not None:
                prepared[0].meta["position_profile"] = self.position_profile
            self._sm70_import_plan = prepared
            self._sm70_weight_manifest_identity = prepared[1].manifest.identity
            if int(depth) > 0:
                from .sm70_nvfp4_weights import check_mtp_positions

                check_mtp_positions(prepared[0], self.mtp_model_dir,
                    context=max_len if type(max_len) is int and max_len > 0 else self.rope.context_limit)
        history_placement = None
        if adaptive_sessions and HistoryPlacement.supports(tp, layout_kv_heads):
            from tensorfold.cuda.geometry import layer_counts
            from .history_placement import HistoryPlacement

            history_placement = HistoryPlacement(8, 2, layer_counts(text)[1])
            if graphs:
                print("[tensorfold] CUDA graphs off: unique history owner execution requires live row bounds", flush=True)
                graphs = False
        self._history_placement = history_placement
        implemented_admission = history_placement is not None and self.position_profile is not None
        if not implemented_admission:
            if (max_waiting_input_bytes is not None or max_waiting_input_bytes_per_principal is not None
                    or output_queue_tokens is not None or queue_timeout_s is not None or retention_policy is not None):
                raise ValueError("FIFO input/deadline and retention options require actual adaptive unique TP8 GPU histories")
        elif max_waiting_input_bytes is None:
            raise ValueError("adaptive unique-history FIFO admission requires explicit max_waiting_input_bytes")
        self._prepare_store_backend(implemented_admission, rank=rank, world=tp)
        self._admission_options = {"admission_policy": "fifo" if implemented_admission else "legacy",
                                   "max_waiting_input_bytes": max_waiting_input_bytes,
                                   "max_waiting_input_bytes_per_principal": max_waiting_input_bytes_per_principal,
                                   "output_queue_tokens": output_queue_tokens,
                                   "queue_timeout_s": queue_timeout_s}
        retention_header = None
        self._retention_native_runtime = None
        self._native_protocol_resources = None
        if implemented_admission:
            from .retention_resources import retention_header_resources

            retention_header = retention_header_resources(retention_policy, tp)
            if tp > 1:
                from .retention_resources import require_native_payload_runtime

                self._retention_native_runtime = require_native_payload_runtime()
        self._retention_header_resources = retention_header
        if sm70:
            for field in ("num_attention_heads", "linear_num_key_heads", "linear_num_value_heads",
                          "moe_intermediate_size", "shared_expert_intermediate_size"):
                if int(text[field]) % tp:
                    raise ValueError(f"GGUF {field} must divide evenly across {tp} tensor-parallel ranks")
            kv_heads = int(text["num_key_value_heads"])
            if kv_heads <= 0 or (kv_heads % tp if kv_heads >= tp else tp % kv_heads):
                raise ValueError("GGUF KV heads must shard evenly or replicate in complete rank groups")
            # Codec geometry is a pure format boundary, before any CUDA setup.
            check_pair(self.kv_pair, head_dim=int(text.get("head_dim") or
                       int(text["hidden_size"]) // int(text["num_attention_heads"])))
        dspark_plan = None
        if self.dspark_model_dir is not None and int(depth) > 0:
            from .dspark_admission import inspect as inspect_dspark

            if vision or expert_session is not None or vram_experts is not None:
                raise ValueError("DSpark with GGUF requires the ordinary text target and resident expert shards")
            if not (draft_vocab is None or type(draft_vocab) is str and draft_vocab in ("default", "0")
                    or type(draft_vocab) is int and draft_vocab == 0):
                raise ValueError("DSpark shares the complete target vocabulary; a pruned MTP vocabulary does not apply")
            dspark_plan = inspect_dspark(self.dspark_model_dir, text, tp=(rank, tp))
            if int(depth) > dspark_plan.config.block:
                raise ValueError(f"DSpark drafts at most {dspark_plan.config.block} tokens per block")
            self._dspark_plan = dspark_plan
            from .dspark_runtime import DSparkRuntime
            if not concurrent or not getattr(DSparkRuntime, "target_capturable", False):
                if graphs:
                    print("[tensorfold] CUDA graphs off: DSpark target feature capture runs eagerly", flush=True)
                graphs = False
        dflash_plan = None
        if self.dflash2_model_dir is not None:
            from .dflash2_weights import DFlash2Plan

            dflash_plan = DFlash2Plan.read(self.dflash2_model_dir, tp=(rank, tp), block=4)
            if graphs:
                print("[tensorfold] CUDA graphs off: DFlash2 target feature capture runs eagerly", flush=True)
            graphs = False
        stock_mtp = int(depth) > 0 and dspark_plan is None and dflash_plan is None
        if nvfp4_source and not (draft_vocab is None or type(draft_vocab) is int and draft_vocab == 0
                                 or type(draft_vocab) is str and draft_vocab == "default"):
            raise ValueError("SM70 NVFP4 companion MTP uses its declared shared full-vocabulary head")
        ids = draft_token_ids(draft_vocab, int(text["vocab_size"])) if stock_mtp and not nvfp4_source else None
        if ids is not None and ids.size < tp and not (
                source_is_gguf and self.mtp_model_dir is not None and resolve_gguf(self.mtp_model_dir) is not None):
            raise ValueError("the draft vocabulary must contain at least one ID per tensor-parallel rank")
        if sm70 and ids is not None:
            # Share this exact selection across admission, the settings digest
            # and loading; a file changed after startup cannot change the head.
            ids.flags.writeable = False
        from .speculation import MethodGeometry

        self.method_geometry = MethodGeometry("dflash2" if dflash_plan is not None else
                                              "dspark" if dspark_plan is not None else "mtp" if stock_mtp else "none",
                                              int(depth))
        each, mtp = self.method_geometry.target_tail_rows, stock_mtp
        if self.position_profile is not None and self.position_method_binding["method"] != self.method_geometry.method:
            raise RuntimeError("prepared positional role and actual speculative method disagree")
        owner_specification = None
        owner_preparation = None
        nvfp4_preallocated = 0
        preparation_manifest_sha = None
        if execution_profile_manifest is not None:
            from .execution_identity import begin_execution_manifest
            from .owner_preparation import parse_preparation

            self.model_dir = Path(model_dir)
            manifest = begin_execution_manifest(self, execution_profile_manifest)
            owner_specification = parse_preparation(manifest.get("preparation"))
            preparation_manifest_sha = self._execution_preload["manifest"]
            if owner_specification is not None:
                from tensorfold.cuda.sm70.paged_attention import _format

                dim = text.get("head_dim") or int(text["hidden_size"]) // int(text["num_attention_heads"])
                if (history_placement is None or tp != 8 or indexer_history != "gpu" or vision and not media_requested
                        or int(text["num_attention_heads"]) // tp != 3 or dim != 256 or not self.kv_pair.symmetric):
                    raise ValueError("native owner preparation requires the homogeneous TP8 H3/D256 GPU-history profile")
                _format(self.kv_pair.key.bits, self.kv_pair.key.codec)
        if dflash_plan is not None or media_requested:
            from torch.distributed import TCPStore
            from tensorfold.cuda.comm import create_owned_tcp_store, validate_store_origin
            from .execution_identity import prepare_dflash_pairing

            # CPU metadata/content authority must precede native preparation,
            # target loading and every CUDA allocation. NCCL borrows this same
            # listener only after the matched content phases have retired.
            self.tp, self.rank = tp, rank
            self._startup_store = self._startup_store_origin = None
            if self._profile_build is not None:
                self._startup_store = self.comm.store  # managed replacement borrows the current cohort
                if self._store_use_libuv is False:
                    self._startup_store_origin = self.comm.store_origin
            if self._startup_store is None:
                if self._store_use_libuv is False:
                    self._startup_store_origin = create_owned_tcp_store(rank, tp, master, port, timeout_s=3600)
                    self._startup_store = self._startup_store_origin.store
                else:
                    self._startup_store = TCPStore(master, port, tp, rank == 0, timeout=timedelta(seconds=3600))
            elif self._store_use_libuv is False:
                validate_store_origin(self._startup_store_origin, self._startup_store, rank, tp, verified=True)
            selected_context = self.rope.context_limit if max_len in (None, 0) else max_len
            if dflash_plan is not None:
                dflash_plan = prepare_dflash_pairing(self, manifest, target_config=read_config(model_dir),
                                                    plan=dflash_plan, context=selected_context)
                self._dflash2_plan = dflash_plan
            if media_requested:
                from .media_runtime import prepare_media_source

                self._media_authority, self._media_software_binding = prepare_media_source(self, self.media_profile)
            if self._profile_build is not None:
                self._startup_store = self._startup_store_origin = None  # Existing comm retains the managed transport.
        torch.cuda.set_device(0)
        if sm70 and adaptive_sessions:
            from tensorfold.cuda.capacity import cuda_reserve_bytes

            if cuda_reserve_bytes(int(torch.cuda.mem_get_info()[1])) < 4 * 2**30:
                raise ValueError("SM70 adaptive serving requires at least 4 GiB device headroom; a lower reserve override is unsupported")
        if media_requested:
            from .media_numerics import MediaNumericScope

            if torch.cuda.get_device_capability() != (7, 0):
                raise ValueError("media profile requires an actual SM70 compute-capability 7.0 device")
            self._media_numeric_scope = MediaNumericScope.create(
                register=lambda owner: setattr(self, "_media_numeric_scope", owner))
        if position_profile_manifest is not None and torch.cuda.get_device_capability() != (7, 0):
            raise ValueError("position profile manifests require an actual SM70 compute-capability 7.0 device")
        if (execution_profile_manifest is not None or nvfp4_source) and torch.cuda.get_device_capability() != (7, 0):
            raise ValueError("execution profile manifests require the actual SM70 V100 provider")
        if compact_kv and torch.cuda.get_device_capability() != (7, 0):
            raise ValueError("compact Planar4/Iso4 KV requires an actual SM70 compute-capability 7.0 device")
        if indexer_history == "ram" and torch.cuda.get_device_capability() != (7, 0):
            raise ValueError("RAM indexer history is supported by the GGUF SM70 CUDA provider only")
        self.tp, self.rank, self.depth, self.confidence = tp, rank, int(depth), float(confidence)
        self.execution_provider = "sm70" if sm70 else "cuda"
        unique_graph_contract = None
        if sm70:
            import os
            from tensorfold.cuda.sm70.execution_policy import (
                PROFILE_ENV, configure, unique_graph_request, verify_graph_inputs,
            )

            verify_graph_inputs(getattr(self, "_nccl_graph_inputs", None))
            requested_profile = os.environ.get(PROFILE_ENV, "reference")
            unique_graph_contract = unique_graph_request(requested_profile,
                method=self.method_geometry.method, depth=self.depth, kv_pair=self.kv_pair,
                unique_history=history_placement is not None, indexer_history=indexer_history,
                graphs=requested_graphs, concurrent=concurrent, vision=vision)
            self.sm70_profile = configure(text, tp=tp, torch=torch,
                record_graph_inputs=lambda record: setattr(self, "_nccl_graph_inputs", record))
            if self.sm70_profile != requested_profile:
                raise ValueError("SM70 profile changed during its method preparation")
            if unique_graph_contract is not None:
                graphs = requested_graphs
                print("[tensorfold] CUDA graphs on: validated exact N1 unique-history profile", flush=True)
            from tensorfold.cuda.sm70.attention_qualification import runtime_identity

            self._sm70_attention_identity = runtime_identity(torch.device("cuda", torch.cuda.current_device()))
        self.streams, self.graphs_enabled = int(streams), bool(graphs)
        managed = self._profile_build is not None
        if not managed:
            self.comm = None
        self.vision = None                   # the image tower (``QwenCudaVision``) with --vision
        if tp > 1 and not managed:
            from tensorfold.cuda.comm import open_comm

            self.comm = open_comm(rank, tp, master, port,
                **({"store": self._startup_store} if self._startup_store is not None else {}),
                **({} if self._store_use_libuv is None else {"use_libuv": self._store_use_libuv}),
                **({"store_origin": self._startup_store_origin} if self._startup_store_origin is not None else {}),
                **({"register": lambda owner: setattr(self, "comm", owner)} if self._store_use_libuv is False else {}))
            self._startup_store = None  # native communicator now owns its store reference
            self._startup_store_origin = None
            self.comm.barrier()
        if self._retention_native_runtime is not None:
            self.comm.bind_native_nofile(self._retention_native_runtime["hard_nofile"])
        native_host_bytes = (len(self.comm.store.host.encode("utf-8"))
                             if self._retention_native_runtime is not None else 0)
        gather = ((lambda values: gather_ints(torch, self.comm.all_gather, values)) if tp == 2 else
                  (lambda values: gather_ints(torch, self.comm.all_gather, values, world=tp))) if tp > 1 else None
        if sm70 and gather is not None:
            # RAM placement adds collectives before final capacity agreement.
            # All GGUF ranks must agree on that schedule before taking a branch.
            modes = gather([int(indexer_history == "ram"), int(adaptive_sessions),
                            -1 if session_ceiling is None else session_ceiling,
                            -1 if execution_rows is None else execution_rows])
            if any(row != modes[0] for row in modes[1:]):
                raise RuntimeError("tensor-parallel ranks selected different history or session execution policies")
        if execution_rows is not None and (type(execution_rows) is not int or execution_rows < each):
            raise ValueError("execution rows must fit one complete speculative verification window")

        def row_envelope(count):
            return min(count * each, execution_rows if execution_rows is not None else 32) if adaptive_sessions else (
                count * each if execution_rows is None else min(count * each, execution_rows))
        head_dim = int(text.get("head_dim") or int(text["hidden_size"]) // int(text["num_attention_heads"]))
        self.kv_pair.row_bytes(max(1, int(text["num_key_value_heads"]) // tp) if sm70
                               else int(text["num_key_value_heads"]) // tp, head_dim)
        kv_geometry = tuple((side.bits, side.group, side.scale_bytes)
                            for side in (self.kv_pair.key, self.kv_pair.value))
        kv_status = (not self.kv_pair.symmetric or bool(self.kv_pair.key.codec or self.kv_pair.value.codec)
                     or sm70 and self.kv_pair.key.quantized or indexer_history == "ram")
        rows0 = chunk or PREFILL_ROWS
        history_plan = None
        history_maximum_owners = 0
        history_geometry = {}
        if indexer_history == "ram":
            from tensorfold.cuda.geometry import layer_counts
            from .indexer_history import HistoryPlan

            history_plan = HistoryPlan(layer_counts(text)[1] + int(mtp),
                int(text.get("indexer_head_dim", 128)), int(text.get("indexer_compress_ratio", 4)),
                rows0, each if concurrent else max(8, each))
            # Concurrent retained prefixes alias their N allocated slots.
            # A serial engine can additionally create one independent twin.
            history_maximum_owners = streams if concurrent else 2
            history_geometry["indexer_raw_device_bytes"] = history_plan.device_bytes_per_layer
        self._indexer_history_plan = history_plan
        if owner_specification is not None:
            from .owner_preparation import OwnerPreparation
            from .state import ATT_ROWS

            bound = execution_rows if execution_rows is not None else 32
            if session_ceiling is not None:
                bound = min(bound, session_ceiling * each)
            roles = {"decode": min(bound, ATT_ROWS),
                     **({"mtp": min(bound, ATT_ROWS)} if mtp else {}), "prefill": min(rows0, ATT_ROWS)}
            owner_preparation = OwnerPreparation(owner_specification, role_rows=roles,
                                                  manifest_sha256=preparation_manifest_sha)
            self._owner_preparation = owner_preparation
            owner_preparation.prepare(self, text, self.kv_pair, page_tokens=history_placement.page_tokens,
                                      device=torch.device("cuda", torch.cuda.current_device()))
        if nvfp4_source:
            from . import sm70_nvfp4

            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            free_before = int(torch.cuda.mem_get_info()[0])
            allocated_before = int(torch.cuda.memory_allocated())
            self._sm70_weight_device = torch.device("cuda", torch.cuda.current_device())
            # Publish the exact owner before any fallible GPU allocation.
            prepared_workspace = object.__new__(sm70_nvfp4.Workspace)
            self._sm70_weight_workspace = prepared_workspace
            local_cfg = self._sm70_import_plan[0].cfg
            linear_kernel_family = self._sm70_import_plan[1].manifest.document["numerical"]["linear_kernel_family"]
            expert_kernel_family = self._sm70_import_plan[1].manifest.document["numerical"]["expert_kernel_family"]
            source_execution_rows = (execution_rows if execution_rows is not None else 32 if adaptive_sessions
                                     else streams * each if concurrent else max(8, each))
            selection_rows = max(rows0, source_execution_rows)
            self._sm70_selection_rows = selection_rows
            prepared_workspace.__init__(self._sm70_weight_device,
                width=max(local_cfg.moe_width, local_cfg.shared_width), dims=local_cfg.hidden,
                selection_rows=selection_rows, slots=local_cfg.top_k + 1, experts=local_cfg.experts + 1,
                kernel_family=linear_kernel_family, expert_family=expert_kernel_family)
            sm70_nvfp4.warm(prepared_workspace)
            torch.cuda.synchronize(self._sm70_weight_device)
            torch.cuda.empty_cache()
            free_after = int(torch.cuda.mem_get_info()[0])
            allocated_delta = int(torch.cuda.memory_allocated()) - allocated_before
            nvfp4_preallocated = max(0, allocated_delta, free_before - free_after)
            self._sm70_weight_preparation = {
                "scope": "startup/current provider residency; no model performance qualification",
                "logical_workspace_bytes": sm70_nvfp4.workspace_bytes(
                    width=max(local_cfg.moe_width, local_cfg.shared_width), dims=local_cfg.hidden,
                    kernel_family=linear_kernel_family, expert_family=expert_kernel_family,
                    selection_rows=selection_rows, slots=local_cfg.top_k + 1, experts=local_cfg.experts + 1),
                "host_workspace_bytes": sm70_nvfp4.host_workspace_bytes(
                    selection_rows, local_cfg.top_k + 1, local_cfg.experts + 1, expert_family=expert_kernel_family),
                "torch_allocated_delta": allocated_delta,
                "physical_free_delta": free_before - free_after,
                "preallocated_device_bytes": nvfp4_preallocated,
                "native_module": sm70_nvfp4.nvfp4_ops.extension().__name__,
                "cub_version": sm70_nvfp4.nvfp4_ops.extension().cub_version,
                "profile": sm70_nvfp4.NUMERICAL_PROFILE, "qualification": "unqualified"}
            self._sm70_weight_preparation["linear_kernel_family"] = linear_kernel_family
            self._sm70_weight_preparation["expert_kernel_family"] = expert_kernel_family
        if history_placement is not None:
            from tensorfold.cuda.sm70.compact_indexer import prepare as prepare_compact_indexer

            prepare_compact_indexer(torch.device("cuda", torch.cuda.current_device()))
        native_a4, checkpoint_math, a4_banks, a4_workspace = False, False, (), 0
        a4_scalar_host = 0
        if modelopt:
            from tensorfold.cuda import precision
            from tensorfold.cuda.nvfp4.modelopt_layout import (
                activation_banks, loading_device_bytes, resident_loading_host_bytes, runtime_workspace_bytes)

            checkpoint_math = precision.mode() == precision.CHECKPOINT
            native_a4 = checkpoint_math and precision.own_math(torch.cuda.get_device_capability())["nvfp4"]
            if expert_session is not None:
                native_a4 = expert_session.native_a4
            from tensorfold.cuda.capacity import headers

            infos = headers(model_dir)
            a4_banks = activation_banks(infos, text, native=native_a4, mtp=mtp, checkpoint=checkpoint_math)
            del infos
            a4_workspace = (runtime_workspace_bytes(text, rows=streams * each if concurrent else max(8, each),
                            prompt_rows=rows0 + (streams * each if concurrent else 0),
                            mtp_a4=any(bank.startswith("mtp.") for bank in a4_banks), cached=vram_experts is not None)
                            if a4_banks and expert_session is None else 0)
            if a4_banks and expert_session is None:
                from inspect import signature

                from tensorfold.cuda.direct_read import ALIGN, ReadAhead
                from tensorfold.cuda.nvfp4.modelopt_layout import scalar_prefetch_host_bytes

                # _Reader constructs ReadAhead with its actual default. Resolve
                # that same worker limit here without duplicating a literal.
                a4_scalar_host = scalar_prefetch_host_bytes(text["num_experts"],
                    workers=signature(ReadAhead).parameters["threads"].default, alignment=ALIGN)
        # captured rounds: bounded and planned; experts read through a host cache (--vram-experts) decline capture
        # (multi_graphs.capturable), so they plan none (the decoder's budget is checked against this after loading)
        graph_request = multi_graph_request() if concurrent and graphs else 0
        self._graph_request = graph_request
        graph_bytes = (graph_request or 0) if vram_experts is None else 0
        def target_geometry(text, count=streams):
            if concurrent:
                return indexed_stream_geometry(text, count, each, KEEP, mtp=mtp, points=POINTS,
                    kv_pair=kv_geometry, graph_bytes=graph_bytes, prefill_rows=rows0,
                    kv_status=kv_status, world=tp,
                    full_windows=count if adaptive_sessions else resident_windows,
                    workspace_bytes=2 * 2**30 if adaptive_sessions or resident_windows is not None else 0,
                    **({"execution_rows": row_envelope(count)} if adaptive_sessions or execution_rows is not None else {}),
                    device_metadata_only=adaptive_sessions,
                    **({"history_placement": history_placement, "rank": rank} if history_placement is not None else {}),
                    **history_geometry)
            return gdn_geometry(text, tp, each, indexed=True, mtp=mtp, kv_pair=kv_geometry,
                                kept=KEEP_SERIAL + 1, prefill_rows=rows0, kv_status=kv_status, **history_geometry)

        if media_resources is not None:
            original_target_geometry = target_geometry

            def target_geometry(text, count=streams):
                from tensorfold.cuda.capacity import Geometry

                base = original_target_geometry(text, count)
                static = media_resources["static_with_vision_bytes_by_rank"][rank]
                transient = media_resources["transient_with_vision_bytes_by_rank"][rank]

                def breakdown(capacity):
                    values = dict(base.breakdown_at(capacity))
                    values.update(media_static=static, media_transient=transient)
                    return values
                return Geometry(lambda capacity: base.bytes_at(capacity) + static + transient,
                    base.reserve, base.minimum_slots, breakdown if base.breakdown_at is not None else None)

        geometry = target_geometry
        if sm70:
            def sm70_geometry(text, count=streams):
                # Two global KV heads remain one local head per TP8 rank,
                # replicated across groups of four. The original metadata is
                # unchanged; this copy is solely the per-rank allocation plan.
                local = dict(text)
                local["num_key_value_heads"] = max(tp, int(text["num_key_value_heads"]))
                base = target_geometry(local, count)
                from tensorfold.cuda.sm70 import family_ops
                from . import padded_linears

                from tensorfold.cuda.capacity import Geometry
                from tensorfold.cuda.sm70.attention_native import workspace_bytes

                main_rows = row_envelope(count) if concurrent else max(8, each)
                padded_bytes = (1 + int(mtp)) * padded_linears.workspace_bytes_from_text(text, main_rows, tp)
                native_attention = (family_ops._NATIVE_FAMILY and family_ops._NATIVE_ATTENTION
                                    and history_placement is None)
                if not native_attention and not padded_bytes and history_placement is None:
                    return base
                scratch_rows = (1 + int(mtp)) * main_rows + min(rows0, 256)
                heads = int(text["num_attention_heads"]) // tp
                def extra(capacity):
                    total = padded_bytes + (workspace_bytes(scratch_rows, heads, head_dim, capacity) if native_attention else 0)
                    if history_placement is not None:
                        from .history_placement import owner_workspace_bytes

                        shape = {"heads": heads, "head_dim": head_dim,
                                 "index_heads": int(text.get("indexer_n_heads", 4)),
                                 "index_dim": int(text.get("indexer_head_dim", 128)),
                                 "index_budget": int(text.get("indexer_budget", 2048)),
                                 "index_ratio": int(text.get("indexer_compress_ratio", 4)),
                                 "context": capacity}
                        arenas = [owner_workspace_bytes(shape, r) for r in
                                  [main_rows] * (1 + int(mtp)) + [rows0]]
                        # Coordinator serializes owner computation across these
                        # arenas; slabs persist together, temporary work does not.
                        # Adopted owners are already excluded from physical
                        # free. Only future temporary work remains unallocated.
                        total += (sum(a["persistent_bytes"] for a in arenas) if owner_preparation is None else 0)
                        total += max(a["temporary_bytes"] for a in arenas)
                    return total

                def breakdown(capacity):
                    values = dict(base.breakdown_at(capacity))
                    values["sm70_native_workspace"] = extra(capacity)
                    return values
                return Geometry(lambda capacity: base.bytes_at(capacity) + extra(capacity),
                    base.reserve, base.minimum_slots, breakdown if base.breakdown_at is not None else None)
            geometry = sm70_geometry
        if a4_workspace:
            from tensorfold.cuda.capacity import Geometry

            original_geometry = geometry
            def geometry(text):
                base = original_geometry(text)
                return Geometry(lambda slots: base.bytes_at(slots) + a4_workspace, base.reserve, base.minimum_slots)
        if exl3:
            geometry = admission(geometry)
        from tensorfold.vision.qwen_cuda import capacity_geometry, weight_transform as vision_weights
        workspace = (media_resources["transient_with_vision_bytes_by_rank"][rank] if media_resources is not None
                     else vision_workspace() if vision else 0)
        rope_capacity = ({"context_limit": self.rope.context_limit, "original_context": self.rope.native_context}
                         if self.rope.rope_type != "default" else {})
        weight_rule = vision_weights(indexed_weights(tp, mtp, mapped_tables=not ple_on_ssd), vision and not media_requested, rank)
        host_capacity = ({"host_extra_staging": resident_loading_host_bytes(text)}
                         if a4_banks and vram_experts is None else {})
        automatic_experts = vram_experts == "auto"
        if automatic_experts:
            from tensorfold.cuda.capacity import unified

            if unified(torch):
                raise ValueError("--vram-experts auto requires a discrete CUDA GPU; on unified memory use a numeric "
                                 "expert budget or load the experts resident")
        automatic_capacity = {}
        numeric_capacity = {}
        if a4_banks:
            from tensorfold.cuda.capacity import Weights, estimate_weights

            def a4_loading_weights(directory):
                original = estimate_weights(directory, weight_rule, rank=rank)
                return Weights(original.resident, max(original.staging, loading_device_bytes(text)), original.mapped), original.staging

            numeric_capacity["weight_estimator"] = a4_loading_weights
        ram_layout = None
        if vram_experts is not None:
            from tensorfold.cuda.capacity import Geometry, cuda_reserve_bytes

            from ..ram_experts import AUTO_HEADROOM_BYTES, layout, plan_scratch
            from .prompt_plan import IDLE_ROWS

            ram_layout = expert_session.layout if expert_session is not None else layout(model_dir, vram_experts, mtp=mtp, native_a4=native_a4,
                                                                                           checkpoint=checkpoint_math)
            if automatic_experts:
                self.expert_headroom_bytes = cuda_reserve_bytes(int(torch.cuda.mem_get_info()[1]),
                                                               default=AUTO_HEADROOM_BYTES)
                automatic_capacity = {"memory_reserve": self.expert_headroom_bytes}
            weight_rule = ram_layout.transform(weight_rule)
            loading_device = getattr(ram_layout, "loading_device_bytes", 0)
            if loading_device:
                from tensorfold.cuda.capacity import Weights, estimate_weights
                def loading_weights(directory):
                    original = estimate_weights(directory, weight_rule, rank=rank)
                    return Weights(original.resident, max(original.staging, loading_device), original.mapped), original.staging
                numeric_capacity["weight_estimator"] = loading_weights
            if ram_layout.format == "exl3" and not automatic_experts:
                from tensorfold.cuda.capacity import Weights, estimate_weights

                def numeric_weights(directory):
                    original = estimate_weights(directory, weight_rule, rank=rank)
                    # Final replacement follows loader/read/Pack drain. Only
                    # one bootstrap payload/control cell overlaps that arena;
                    # its device allowance does not increase CPU read staging.
                    overlap = ram_layout.entry_bytes + 32
                    return Weights(original.resident, max(original.staging, overlap), original.mapped), original.staging

                numeric_capacity["weight_estimator"] = numeric_weights
            if ram_layout.format == "exl3":
                from .exl3_pack import MOE_WINDOW

                pairs = MOE_WINDOW * (text["num_experts_per_tok"] + 1)
                scratch_device, scratch_host = pairs * 4, pairs * 8
            else:
                scratch_device, scratch_host = plan_scratch(text,
                    streams * each if concurrent else max(8, each), chunk or IDLE_ROWS, mtp)
            if expert_session is not None:
                scratch_device += expert_session.workspace_bytes
            base_geometry = geometry
            def geometry(text):
                base = base_geometry(text)
                return Geometry(lambda slots: base.bytes_at(slots) + scratch_device,
                                base.reserve, base.minimum_slots)
            host_capacity = {"host_resident": ram_layout.host_bytes + ram_layout.metadata_host_bytes
                             + ram_layout.staging_bytes + ram_layout.control_host_bytes + scratch_host,
                             "host_extra_staging": ram_layout.loading_bytes,
                             "resident_extra": ram_layout.gpu_bytes + ram_layout.metadata_device_bytes
                             + ram_layout.control_device_bytes}
            if expert_session is None and vram_experts_pin != "off":
                from ..expert_authority import allocations
                from tensorfold.cuda.expert_pin import backing_bytes
                geometry_bytes = allocations(text, ram_layout)
                host_capacity["host_resident"] += max(0, sum(map(backing_bytes, geometry_bytes)) - ram_layout.host_bytes)
            if expert_session is not None:
                from tensorfold.cuda.read_budget import READER_HOST_RING_BYTES

                host_capacity["host_resident"] = expert_session.group_host_bytes + scratch_host
                # The group already owns each Session's expert read/pack
                # staging. weights.load has a separate backbone Reader whose
                # physical ring remains live beside those Session readers.
                host_capacity["host_extra_staging"] = READER_HOST_RING_BYTES
        if a4_scalar_host:
            host_capacity["host_extra_staging"] = host_capacity.get("host_extra_staging", 0) + a4_scalar_host
        self._pin_future_bytes = 0
        if ram_layout is not None and vram_experts_pin != "off":
            from tensorfold.cuda.capacity import estimate_weights
            original_estimator = numeric_capacity.get("weight_estimator")
            def pin_forecast(directory):
                if original_estimator is None:
                    weights = estimate_weights(directory, weight_rule, rank=rank)
                    host_staging = weights.staging
                else:
                    weights, host_staging = original_estimator(directory)
                self._pin_future_bytes = host_staging + a4_scalar_host
                return weights, host_staging
            numeric_capacity["weight_estimator"] = pin_forecast
        prefill_plan = None
        if sm70:
            if nvfp4_source:
                from .sm70_nvfp4_weights import capacity_options

                options, self._sm70_weight_plan = capacity_options(*self._sm70_import_plan,
                    mtp=mtp, mtp_model_dir=self.mtp_model_dir, prefill_rows=rows0,
                    execution_rows=self._sm70_selection_rows, workspace_preallocated=True)
                options["execution_capability"] = (7, 0)
                self._gguf_prefill_receipt = None
            else:
                from .gguf_admission import capacity_options
                from .gguf_prefill import plan as prefill_expert_plan

                prefill_plan = prefill_expert_plan(text, rows=rows0, world=tp)
                self._gguf_prefill_receipt = prefill_plan.receipt() if prefill_plan is not None else None
                options = capacity_options(model_dir, rank=rank, world=tp, mtp=mtp,
                                           **({"position_profile": self.position_profile}
                                              if self.position_profile is not None else {}),
                                           **({"prefill_plan": prefill_plan} if prefill_plan is not None else {}),
                                           **({"draft_ids": ids} if ids is not None else {}),
                                           **({"mtp_model_dir": self.mtp_model_dir} if self.mtp_model_dir is not None else {}))
            # The half prompt chain owns full-piece replay inputs, independent
            # of each sequence's small decode scratch. They remain live beside
            # the serving cache and the loader's shared GGUF workspaces.
            sm70_prefill_bytes = _sm70_prefill_gdn_bytes(text, tp, rows0)
            options["device_workspace_bytes"] = options.get("device_workspace_bytes", 0) + sm70_prefill_bytes
            status_device = status_host = 0
            if tp > 1 and concurrent and not adaptive_sessions:
                from .kv_status import workspace_bytes

                status_device, status_host = workspace_bytes(streams, tp)
                options["device_workspace_bytes"] += status_device
                options["host_workspace_bytes"] = options.get("host_workspace_bytes", 0) + status_host
            for key in ("host_resident", "host_extra_staging", "host_workspace_bytes"):
                if key in options:
                    host_capacity[key] = options.pop(key)
            numeric_capacity.update(options)
        position_vector_ledger = None
        if self.position_profile is not None:
            from ..position_profile import position_vector_bytes

            position_vector_ledger = position_vector_bytes(self.position_profile, mtp_active=stock_mtp)
            numeric_capacity["device_workspace_bytes"] = numeric_capacity.get("device_workspace_bytes", 0) + (
                position_vector_ledger["mtp_extra_vector_bytes"])
        if dspark_plan is not None:
            from tensorfold.cuda.capacity import Geometry

            main_rows = streams * each if concurrent else max(8, each)
            dspark_cache_copies = (min(streams, resident_windows) if resident_windows is not None else 1) if concurrent else 2
            if concurrent:
                from .multi import FIRST as dspark_first
            else:
                dspark_first = 0
            def dspark_slots(total):
                return total * dspark_cache_copies + max(0, streams - dspark_cache_copies) * min(dspark_first, total)
            # Match indexed target geometry: selected full windows plus other
            # slots' initial floors, or two full caches for a serial twin.
            dspark_context_limit = dspark_plan.config.context - each
            if max_len is None:
                max_len = min(int(text.get("max_position_embeddings", dspark_context_limit)), dspark_context_limit)
            elif max_len > dspark_context_limit:
                raise ValueError(f"DSpark prompt/reply context must not exceed {dspark_context_limit} tokens")
            from .dspark_proposal_graphs import _ENABLED as proposal_graphs_enabled, graph_host_bytes

            dspark_graph_host = 0
            if graphs and concurrent and proposal_graphs_enabled:
                # Bound per-slot/bucket pinned packets before ownership. Device
                # buffers and graph pools still use the common measured budget.
                dspark_graph_host = graph_host_bytes(streams, min(int(depth), dspark_plan.config.block),
                                                    tp, int(max_len))
                host_capacity["host_workspace_bytes"] = host_capacity.get("host_workspace_bytes", 0) + dspark_graph_host
            dspark_geometry = geometry
            def geometry(text):
                base = dspark_geometry(text)
                return Geometry(lambda total: base.bytes_at(total) + dspark_plan.capacity_bytes(
                    dspark_slots(total), main_rows=main_rows, prompt_rows=rows0, slots=streams),
                    base.reserve, base.minimum_slots)
            if resident_windows is not None:
                dspark_full_geometry = geometry
                # Original target geometry charges one target-layer replacement.
                # Auxiliary DSpark resize replaces all its layers together.
                from tensorfold.cuda.geometry import _kv_pair_bytes, layer_counts
                target_layers = layer_counts(text)[1] + int(mtp)
                pair_row, pair_dummies = _kv_pair_bytes(head_dim, 16, 32, 2, kv_geometry)
                local_kv = max(1, int(text["num_key_value_heads"]) // tp)
                index_dim = int(text.get("indexer_head_dim", 128))
                ratio = int(text.get("indexer_compress_ratio", 4))
                def geometry(text):
                    base = dspark_full_geometry(text)
                    def bytes_at(total):
                        raw_rows = total if history_plan is None else 0
                        target_copy = (total * local_kv * pair_row + pair_dummies +
                            (raw_rows + -(-total // ratio)) * index_dim * 2) if target_layers else 0
                        return base.bytes_at(total) + max(0, total * dspark_plan.cache_row_bytes - target_copy)
                    return Geometry(bytes_at, base.reserve, base.minimum_slots)
            numeric_capacity["resident_extra"] = numeric_capacity.get("resident_extra", 0) + dspark_plan.resident_bytes
            host_capacity["host_extra_staging"] = host_capacity.get("host_extra_staging", 0) + dspark_plan.loading_host_bytes
        if dflash_plan is not None:
            numeric_capacity["resident_extra"] = numeric_capacity.get("resident_extra", 0) + dflash_plan.resident_bytes
            # Target and draft load sequentially; this host conversion charge
            # belongs to the same process loading envelope, once per rank.
            host_capacity["host_extra_staging"] = host_capacity.get("host_extra_staging", 0) + tp * dflash_plan.resources(
                slots=1, context=selected_context, prefill_rows=rows0, execution_rows=row_envelope(1),
                keep=KEEP, points=POINTS)["loading_host_staging_bytes"]

        def dflash_resources(count, total):
            return dflash_plan.serving_resources(slots=count, context=min(total, selected_context),
                # The initial TP8 path cannot converge expert launches across
                # decode/prefill; its actual prompt buffer has exactly rows0.
                prefill_rows=rows0, execution_rows=row_envelope(count),
                keep=KEEP, points=POINTS, target_vocab=int(text["vocab_size"]))
        history_host_limit = history_host_per_rank = 0
        if history_plan is not None:
            # Every CPU archive owns its complete declared window up front.
            # Charge all ranks together: host/cgroup availability is shared,
            # while GPU device admission remains independently per rank.
            history_host_limit = int(max_len if max_len is not None else self.rope.context_limit) + each
            history_host_per_rank = history_maximum_owners * history_plan.host_bytes(history_host_limit)
            host_capacity["host_workspace_bytes"] = host_capacity.get("host_workspace_bytes", 0) + tp * history_host_per_rank
        adaptive_capacity = {}
        if adaptive_sessions:
            from tensorfold.cuda.capacity import Geometry
            from .kv_status import workspace_bytes as status_workspace

            def session_geometry(text, count):
                base = sm70_geometry(text, count)
                media_extra = 0
                if media_requested and mtp and self.position_profile is not None:
                    from .media_mtp import boundary_resources

                    media_extra = boundary_resources(hidden_width=int(text.get("hc_count", 4)) * int(text["hidden_size"]),
                        slots=count, keep=KEEP, points=POINTS)["additional_device_bytes"]
                status_bytes = status_workspace(count, tp)[0] if tp > 1 else 0

                def additions(total):
                    costs = {"kv_cohort_workspace": status_bytes}
                    if retention_header is not None and retention_policy is not None:
                        from .retention_resources import retained_recovery_extra

                        costs["retained_idle_recovery"] = retained_recovery_extra(
                            base.breakdown_at(total)["retained_recovery"], slots=count, keep=KEEP, points=POINTS)
                        costs["retained_restore_overlap"] = retention_header.per_rank_gpu_overlap_bytes
                    if media_extra:
                        costs["media_mtp_boundary"] = media_extra
                    if history_placement is not None:
                        from .history_placement import MIGRATION_OVERLAP_BYTES

                        costs["migration_overlap_limit"] = MIGRATION_OVERLAP_BYTES
                    if dspark_plan is not None:
                        costs["draft_history_execution"] = dspark_plan.capacity_bytes(
                            count * total, main_rows=row_envelope(count), prompt_rows=rows0, slots=count)
                        target_copy = base.breakdown_at(total)["replacement_overlap"]
                        costs["draft_replacement_overlap"] = max(0, total * dspark_plan.cache_row_bytes - target_copy)
                    if dflash_plan is not None:
                        resource = dflash_resources(count, total)
                        costs.update(dflash_window_state=resource["state_bytes"],
                                     dflash_recovery_snapshots=resource["snapshot_bytes"],
                                     dflash_execution_workspace=resource["execution_workspace_bytes"],
                                     dflash_target_probability=resource["target_probability_device_bytes"])
                        if retention_header is not None and retention_policy is not None:
                            costs["retained_idle_dflash_recovery"] = retained_recovery_extra(
                                resource["snapshot_bytes"], slots=count, keep=KEEP, points=POINTS)
                    if retention_header is not None:
                        from .retention_resources import retention_device_resources

                        device = retention_device_resources({**base.breakdown_at(total), **costs},
                            policy=retention_policy, slots=count, keep=KEEP, points=POINTS, mtp=mtp,
                            hidden_width=int(text.get("hc_count", 4)) * int(text["hidden_size"]),
                            media_inputs_enabled=media_requested)
                        costs["retained_device_dependencies"] = device["header_extra_device_bytes"]
                    return costs

                def breakdown(total):
                    return {**base.breakdown_at(total), **additions(total)}

                return Geometry(lambda total: base.bytes_at(total) + sum(additions(total).values()),
                                base.reserve, base.minimum_slots, breakdown)

            def native_protocol_payload(count):
                if self._retention_native_runtime is None:
                    return None
                from .retention_resources import native_sync_fixed_payload, native_sync_prefix_payload

                bound = max(self.comm.native_state_slots_highwater, count)
                return {"fixed": native_sync_fixed_payload(world=tp,
                            ram=retention_policy is not None and retention_policy.tier == "ram",
                            slots=bound, fd_limit=self._retention_native_runtime["hard_nofile"],
                            host_bytes=native_host_bytes),
                        "prefix": native_sync_prefix_payload(world=tp, slots=bound)}

            def session_host_workspace(count):
                from tensorfold.cuda.geometry import multi_static_domains

                total = tp * multi_static_domains(text, count, each, mtp)[1]
                total += tp * status_workspace(count, tp)[1] if tp > 1 else 0
                if self.position_profile is not None:
                    from .history_placement import coordinate_source_host_bytes

                    total += tp * coordinate_source_host_bytes(max(row_envelope(count), rows0))
                if dspark_plan is not None and graphs and proposal_graphs_enabled:
                    total += tp * graph_host_bytes(count, min(int(depth), dspark_plan.config.block), tp, int(max_len))
                if dflash_plan is not None:
                    total += tp * dflash_resources(count, selected_context)["host_serving_bytes"]
                native = native_protocol_payload(count)
                if native is not None:
                    total += native["fixed"]["global_native_fixed_payload_bytes"]
                    total += native["prefix"]["global_native_prefix_payload_bytes"]
                return total

            if dspark_plan is not None and dspark_graph_host:
                host_capacity["host_workspace_bytes"] -= dspark_graph_host
            adaptive_capacity = {"session_geometry": session_geometry, "session_ceiling": session_ceiling,
                                 "session_host_workspace": session_host_workspace}
        if retention_header is not None:
            host_capacity["host_workspace_bytes"] = (
                host_capacity.get("host_workspace_bytes", 0) + retention_header.global_host_bytes)
        if media_resources is not None:
            # Shared host obligation is charged once on every rank's same-host
            # admission. Child virtual address ceilings are not physical RSS.
            host_capacity["host_workspace_bytes"] = host_capacity.get("host_workspace_bytes", 0) + (
                media_resources["shared_host_payload_bound"])
            host_capacity["host_extra_staging"] = host_capacity.get("host_extra_staging", 0) + (
                media_resources["vision_loading_host_payload_bound"])
        self.capacity_plan = admit(model_dir, max_len, context_explicit, torch,
                                   capacity_geometry(geometry, model_dir, vision and not media_requested, rank, workspace),
                                   weight_rule,
                                   rank=rank, world=tp,
                                   **({"capture_device_baseline": True} if sm70 and adaptive_sessions else {}),
                                   gather=gather, extra_files=extra_files(model_dir) if exl3 else (),
                                   **automatic_capacity,
                                   **numeric_capacity,
                                   **rope_capacity, **host_capacity, **adaptive_capacity,
                                   **({"preallocated_device_bytes": nvfp4_preallocated
                                       + (owner_preparation.committed_bytes() or 0 if owner_preparation is not None else 0)}
                                      if owner_preparation is not None or nvfp4_source else {}))
        if retention_header is not None:
            self.capacity_plan["retention_header"] = retention_header.record()
            self.capacity_plan["retention_store"] = {"use_libuv": self._store_use_libuv,
                "process_fixed": True, "selection": "adaptive-unique-profile",
                "evidence": "source-selected; native/resource qualification pending",
                "origin": self.comm.store_origin.record()}
        if media_resources is not None:
            self.capacity_plan["media_profile"] = {"identity": self.media_profile.identity,
                "resources": media_resources, "encoder_rank": self.media_profile.execution.encoder_rank,
                "numerics": self._media_numeric_scope.record(), "qualification": "pending actual V100 model/media gates"}
        if owner_preparation is not None:
            self.capacity_plan["owner_preparation"] = owner_preparation.record()
        if nvfp4_source:
            self.capacity_plan["sm70_weight_preparation"] = self._sm70_weight_preparation
            self.capacity_plan["sm70_weight_manifest"] = self._sm70_weight_manifest_identity
            self.capacity_plan["sm70_weight_plan"] = self._sm70_weight_plan
        if adaptive_sessions:
            streams = self.capacity_plan["session_capacity"]["resident_capacity"]
            resident_windows = streams
            self.streams = self.kv_full_windows = streams
            if self._retention_native_runtime is not None:
                # Probe counts must not inflate the persistent reservation.
                # Reserve the selected count before warmup can create keys.
                self.comm.note_native_state_slots(streams)
                self._native_protocol_resources = native_protocol_payload(streams)
                self.capacity_plan["native_protocol"] = {
                    **self._native_protocol_resources, "abi": self._retention_native_runtime}
            status_device, status_host = status_workspace(streams, tp) if tp > 1 else (0, 0)
            if dspark_plan is not None:
                dspark_cache_copies = streams
                main_rows = row_envelope(streams)
        self._retention_device_resources = None
        if retention_header is not None:
            from .retention_resources import retention_device_resources

            self._retention_device_resources = retention_device_resources(
                session_geometry(text, streams).breakdown_at(self.capacity_plan["cache_slots"]),
                policy=retention_policy, slots=streams, keep=KEEP, points=POINTS, mtp=mtp,
                hidden_width=int(text.get("hc_count", 4)) * int(text["hidden_size"]),
                media_inputs_enabled=media_requested)
            self.capacity_plan["retention_device_resources"] = self._retention_device_resources
        execution_rows = row_envelope(streams) if concurrent else execution_rows
        self.execution_rows = execution_rows
        self.capacity_plan["execution_envelope"] = {
            "target_rows": execution_rows if concurrent else max(8, each),
            "max_proposals": self.depth, "prefill_tokens": rows0,
            "method": self.method_geometry.method,
            "evidence": "source-bounded-unqualified" if adaptive_sessions else "existing-profile"}
        self.capacity_plan["speculation"] = self.method_geometry.receipt()
        if dflash_plan is not None:
            self.capacity_plan["dflash2"] = {"resources": dflash_resources(streams, selected_context),
                "pairing": self._dflash_pairing_receipt, "qualification": "unqualified-candidate",
                "target_shared_weight_bytes": 0, "target_graphs": False,
                "verification": "standard-rejection-actual-conditional-selector-q"}
        if history_plan is not None:
            self.capacity_plan["indexer_history"] = {
                "mode": "ram", "policy": "lossless-fp16-ram-archive-v1",
                "host_capacity_upper_bound": history_host_limit,
                "host_bytes_per_rank_upper_bound": history_host_per_rank,
                "host_bytes_all_ranks_upper_bound": tp * history_host_per_rank,
                "device_bytes_per_layer": history_plan.device_bytes_per_layer,
                "layers": history_plan.layers,
                "host_archive": "full-window-pageable", "pinned": "bounded-transfer-staging-only",
            }
        if resident_windows is not None:
            self.capacity_plan["kv_full_windows"] = {
                "requested": resident_windows, "reserved": min(streams, resident_windows),
                "stream_slots": streams, "context_window": self.capacity_plan["context_window"],
                "cache_slots_each": self.capacity_plan["cache_slots"],
                "auxiliary": self.method_geometry.method if self.method_geometry.method != "none" else None,
                "workspace_bytes": 2 * 2**30, "policy": "resident-full-windows-v2"}
        if dspark_plan is not None:
            self.capacity_plan["dspark"] = {
                "identity": dspark_plan.identity, "source_identity": dspark_plan.source_identity,
                "resident_bytes": dspark_plan.resident_bytes,
                "loading_host_bytes": dspark_plan.loading_host_bytes,
                "cache_row_bytes": dspark_plan.cache_row_bytes,
                "shared_cache_slots": self.capacity_plan["cache_slots"],
                "cache_copies": dspark_cache_copies,
                "initial_other_slots": max(0, streams - dspark_cache_copies) * min(dspark_first, self.capacity_plan["cache_slots"]),
                "workspace_and_cache_bytes": dspark_plan.capacity_bytes(
                    dspark_slots(self.capacity_plan["cache_slots"]),
                    main_rows=main_rows, prompt_rows=rows0, slots=streams),
                "target_graphs_requested": bool(graphs),
                "proposal_graphs": bool(graphs and concurrent and proposal_graphs_enabled),
                "proposal_graph_host_bytes": dspark_graph_host,
                "verification": "position-keyed-first-mismatch",
            }
        if sm70:
            self.capacity_plan["sm70_prefill_gdn_bytes"] = sm70_prefill_bytes
            self.capacity_plan["sm70_status_cohort"] = {"device_bytes": status_device, "host_bytes": status_host}
            if prefill_plan is not None:
                self.capacity_plan["gguf_prefill_experts"] = self._gguf_prefill_receipt
        if expert_session is not None:
            expert_session.pin_future_bytes = self._pin_future_bytes
            expert_session.admit_followers()
        if ram_layout is not None:
            self.capacity_plan["vram_experts"] = {
                "host_bytes": ram_layout.host_bytes, "gpu_bytes": ram_layout.gpu_bytes,
                "slots": ram_layout.slots, "staging_bytes": ram_layout.staging_bytes,
                "loading_bytes": ram_layout.loading_bytes, "plan_device_bytes": scratch_device,
                "plan_host_bytes": scratch_host, "format": ram_layout.format,
                "metadata_device_bytes": ram_layout.metadata_device_bytes,
                "metadata_host_bytes": ram_layout.metadata_host_bytes,
                "control_device_bytes": ram_layout.control_device_bytes,
                "control_host_bytes": ram_layout.control_host_bytes}
        if a4_banks:
            self.capacity_plan["modelopt_a4"] = {"banks": list(a4_banks), "workspace_bytes": a4_workspace,
                                               "input_scale": "direct-dequantization",
                                               "prefetched_scalar_host_bytes": a4_scalar_host,
                                               "resident_loading_host_bytes": host_capacity.get("host_extra_staging", 0)}
        if expert_session is not None:
            self.capacity_plan["expert_math"] = expert_session.math_policy
        self.capacity_plan["rope"] = self.rope.metadata()
        if self.position_profile is not None:
            self.capacity_plan["position_profile"] = {
                "resolved": self.position_profile.receipt(), "identity": self.position_profile.identity,
                "epoch": self.position_profile.epoch, "method_binding": self.position_method_binding,
                "preparation": self._position_prepared.receipt(),
                "effective_context_window": self.capacity_plan["context_window"],
                "vectors": dict(position_vector_ledger)}
            from .history_placement import coordinate_source_host_bytes

            coordinate_rows = max(row_envelope(streams), rows0)
            self.capacity_plan["position_profile"]["coordinate_sources"] = {
                "rows": coordinate_rows, "host_payload_bytes_per_rank": coordinate_source_host_bytes(coordinate_rows),
                "project_pinned_bytes": 0, "gpu_payload": "covered-by-existing-owner-prepare-envelope"}
        if self.rope.rope_type != "default":
            print(f"[tensorfold] static YaRN factor {self.rope.factor:g}, rotary amplitude "
                  f"{self.rope.attention_factor:.8g}, original context {self.rope.native_context}; "
                  "short-context quality may change", flush=True)
        self.prefill_rows, prompt_workspace = (PREFILL_ROWS, 0) if exl3 else (chunk, 0) if chunk else prompt_plan(
            self.capacity_plan, config(model_dir), torch.cuda.get_device_capability(), world=tp, vision=vision,
            fp8=prompt_precision.fp8())
        if prompt_workspace:
            peak = self.capacity_plan["total_bytes_estimate"] / 2**30
            print(f"[tensorfold] {self.prefill_rows}-row idle prompt workspace {prompt_workspace / 2**30:.2f} GiB; "
                  f"planned peak {peak:.2f} GiB at the admitted window", flush=True)
        self.max_len = self.capacity_plan["cache_slots"]
        if history_plan is not None and self.max_len <= int(text.get("indexer_budget", 2048)):
            raise ValueError("RAM indexer history requires pooled sparse attention; the admitted cache must exceed indexer_budget")
        if tp > 1:
            self._same_settings(torch, ids)
        if history_plan is not None:
            from .indexer_admission import prepare as prepare_history
            from .indexer_history import HistorySession

            allocator, placement = prepare_history(history_plan, limit=self.max_len, maximum_owners=history_maximum_owners,
                torch=torch, device=torch.device("cuda", torch.cuda.current_device()), rank=rank, world=tp, gather=gather)
            self._indexer_history_session = HistorySession(history_plan, allocator,
                torch.device("cuda", torch.cuda.current_device()), maximum_owners=history_maximum_owners)
            self.capacity_plan["indexer_history"].update(placement)
        reads = self._startup_reads                              # the n-gram tables' pages, read while the weights load
        if managed:
            self._profile_build.phase("loading_weights")
        w = load(model_dir, mtp=stock_mtp, tp=(rank, tp) if tp > 1 else None,
                 **({"family_manifest": self.family_manifest, "_sm70_plan": self._sm70_import_plan,
                     "_sm70_workspace": self._sm70_weight_workspace,
                     "_sm70_weight_plan": self._sm70_weight_plan} if nvfp4_source else {}),
                 draft_vocab=draft_vocab if stock_mtp else None, ple_on_ssd=ple_on_ssd,
                 table_reads=reads if prefetch and not ple_on_ssd else None, rope=self.rope,
                 **({"position_profile": self.position_profile, "_position_prepared": self._position_prepared}
                    if self.position_profile is not None else {}),
                 **({"_prefill_plan": prefill_plan} if prefill_plan is not None else {}),
                 **({"_draft_ids": ids} if sm70 and ids is not None else {}),
                 **({"mtp_model_dir": self.mtp_model_dir} if self.mtp_model_dir is not None else {}),
                 **({"eos_token_ids": eos_token_ids} if eos_token_ids is not None else {}),
                 **({"expert_loader": expert_session.load_bank} if expert_session is not None else
                    {"vram_experts": vram_experts} if vram_experts is not None else {}),
                 **({"vram_experts_pin": vram_experts_pin, "_pin_future_bytes": self._pin_future_bytes}
                    if vram_experts_pin != "off" else {}))
        self.w = w
        if self.position_profile is not None:
            from ..position_profile import bind_position_profile

            bind_position_profile(w, self.position_profile, mtp_active=stock_mtp, prepared=self._position_prepared)
            w.meta.update(position_profile=self.position_profile, position_profile_identity=self.position_profile.identity,
                          position_profile_epoch=self.position_profile.epoch,
                          position_profile_receipt=self.capacity_plan["position_profile"])
            self._position_prepared = None  # device operands now own the required small vectors
        if not managed:
            self.scheduler = None
        if "mtp_import" in w.meta:
            self.capacity_plan["mtp_import"] = dict(w.meta["mtp_import"])
        self._sm70_import_plan = None  # Headers/prototypes have no serving borrower after materialization.
        if history_placement is not None:
            w.meta.update(history_layout="unique", attention_degree=history_placement.degree)
        if owner_preparation is not None:
            w.meta["owner_workspaces"] = owner_preparation.workspaces
        if self._indexer_history_session is not None:
            w.meta.update(indexer_history="ram", indexer_history_session=self._indexer_history_session)
        if expert_session is not None:
            expert_session.finish_loading()
            w.meta["expert_cache"] = expert_session.cache
        if ram_layout is not None and not automatic_experts:
            cache = w.meta["expert_cache"]
            initial = deepcopy(self.capacity_plan)
            self.capacity_plan["initial_header_pool_admission"] = initial
            pool = self.capacity_plan["vram_experts"]
            pool.update(gpu_bytes=cache.gpu_bytes, slots=cache.capacity,
                        resident_capacity=cache.resident_capacity, safe_lease_capacity=cache.capacity,
                        control_device_bytes=getattr(cache, "control_device_bytes", 0),
                        control_host_bytes=getattr(cache, "control_host_bytes", 0), automatic=False,
                        replacement_bootstrap_gpu_bytes=ram_layout.entry_bytes + 32
                            if ram_layout.format == "exl3" else 0,
                        replacement_bootstrap_host_bytes=64 if ram_layout.format == "exl3" else 0)
            # Header admission retains the conservative maximum-cell GPU
            # bound. Only serving storage changes after load; the original
            # startup/loading forecast remains separately authenticated.
            admitted = initial["vram_experts"]
            delta = cache.gpu_bytes + pool["control_device_bytes"] - admitted["gpu_bytes"] - admitted["control_device_bytes"]
            if delta > 0 or pool["control_host_bytes"] > admitted["control_host_bytes"]:
                raise MemoryError("prepared expert arena exceeds its header admission")
            serving = initial["serving_peak_bytes_estimate"] + delta
            total = max(initial["startup_peak_bytes_estimate"], serving)
            self.capacity_plan.update(weight_bytes_estimate=initial["weight_bytes_estimate"] + delta,
                serving_peak_bytes_estimate=serving, total_bytes_estimate=total,
                full_mapped_working_set_bytes_estimate=total + initial["mapped_table_bytes"],
                estimate_scope="numeric: original conservative header loading forecast and final bounded expert arena")
            authority, pin_receipt = _expert_authority(cache)
            pool["host_pin"] = pin_receipt
            print(f"[tensorfold] RAM experts: {cache.host_bytes / 2**30:.2f} GiB {authority}, "
                  f"{cache.gpu_bytes / 2**30:.2f} GiB GPU pool ({cache.resident_capacity} resident cells, "
                  f"{cache.capacity} experts per lease); cache misses transfer weights", flush=True)
        tables_read = bool(reads)
        waited = time.perf_counter()
        self._drain_startup_reads()                               # raises a table read's error
        waited = time.perf_counter() - waited
        w.comm = self.comm
        if sm70:
            from tensorfold.cuda.sm70.execution_policy import prepare_weights

            prepare_weights(w, graph_contract=unique_graph_contract)
        if self.comm is not None:
            from tensorfold.cuda.comm import agree_eos

            agree_eos(self.comm.all_gather, w.cfg.eos, world=tp, device=w.device)
            self.comm.ready("loading")               # a peer stuck loading is named, not waited on in NCCL
        if dspark_plan is not None:
            from .dspark_runtime import prepare as prepare_dspark

            runtime = prepare_dspark(w, dspark_plan, prefill_rows=self.prefill_rows)
            if runtime.nbytes > dspark_plan.resident_bytes:
                raise MemoryError("loaded DSpark weights exceed their admitted resident plan")
            self.capacity_plan["dspark"]["loaded_bytes"] = runtime.nbytes
        if dflash_plan is not None:
            from .dflash2_runtime import DFlash2Runtime

            if w.cfg.execution_dtype != "float16" or w.cfg.rope_theta != self.rope.theta:
                raise ValueError("loaded target numerics differ from the pre-load DFlash2 pairing")
            runtime = DFlash2Runtime()
            runtime.load(w, dflash_plan, context=selected_context, pairing=self._dflash_pairing,
                         profile_identity=self._dflash_pairing.identity)
            if runtime.nbytes != dflash_plan.resident_bytes:
                raise MemoryError("loaded DFlash2 weights differ from their exact admitted payload")
            self.capacity_plan["dflash2"]["loaded_weight_bytes"] = runtime.nbytes
        if (self.depth > 0 and w.mtp is None and getattr(w, "dspark", None) is None
                and getattr(w, "dflash2", None) is None):
            raise ValueError("this checkpoint has no MTP head, which Flash Next's CUDA engine drafts with: use one "
                             "that has it, or --no-drafts for the serial reference (one token a round)")
        from tensorfold.cuda.markers import resume_points

        # a prompt's message starts to keep states at, or None; the concurrent decoder's passes hold several kept
        # points at no extra pass, so it also keeps the last turn's start (the serial prefill would add a chunk)
        self.points = resume_points(model_dir, turn=concurrent, template=self.chat_template)
        if media_requested:
            from .media_runtime import MediaRuntime

            self.vision = MediaRuntime.create(self, self.media_profile, authority=self._media_authority,
                software_binding=self._media_software_binding, allow_urls=vision_urls,
                register=lambda owner: setattr(self, "vision", owner))
            self._media_software_binding = None  # The warmed runtime owns the complete original software anchor.
            self._media_inputs_enabled = True
        elif vision:
            from tensorfold.vision.qwen_cuda import QwenCudaVision

            self.vision = QwenCudaVision(model_dir, torch.device("cuda", 0),
                                         allow_urls=vision_urls)
            torch.cuda.empty_cache()
            print(f"[tensorfold] vision: image{' and video' if self.vision.videos else ''} input, a "
                  f"{self.vision.weight_bytes / 2**30:.2f} GiB tower with {vision_workspace() / 2**30:.2f} GiB of "
                  f"workspace reserved{'; https URLs allowed' if vision_urls else ''}", flush=True)
        w.meta["media_inputs_enabled"] = self._media_inputs_enabled
        if self.position_profile is not None:
            self._bind_coordinate_profile(w)
            from .input_preparation import install_input_preparation

            install_input_preparation(w)
        if media_requested:
            self.vision.bind(w)
        # ``streams`` > 1: up to that many requests decoded together, every stream's chain in one forward
        self.concurrent = concurrent
        self.multi = None
        if not managed:
            self.scheduler = None
        if self.concurrent:
            from tensorfold.cuda.scheduler import Scheduler

            from .multi import MultiDecoder

            self.e = None
            if managed:
                self._profile_build.phase("allocating_state")
            self.multi = MultiDecoder(w, slots=streams, capacity=self.max_len, logical_context=self.context_window,
                                      depth=self.depth,
                                      confidence=self.confidence, keep=KEEP, points=self.points,
                                      kv_pair=self.kv_pair, share=share, vision=self.vision,
                                      prefill_rows=self.prefill_rows, workspace_bytes=prompt_workspace,
                                      graphs=graphs, graph_budget=graph_request,
                                      graph_header_room=max(0, self.capacity_plan["budget_bytes"]
                                                            - self.capacity_plan["serving_peak_bytes_estimate"]),
                                      vision_workspace_bytes=workspace,
                                      **({"execution_rows": execution_rows} if execution_rows is not None else {}),
                                      **({"memory_live": owner_preparation.live_bytes} if owner_preparation is not None else {}),
                                      **({"full_windows": resident_windows} if resident_windows is not None else {}))
            if dflash_plan is not None:
                planned = self.capacity_plan["dflash2"]["resources"]
                windows = sum(state.dflash2.nbytes for state in self.multi.states)
                frames = sum(buffers.dflash2_features.nbytes for buffers in (self.multi.buf, self.multi.pbuf))
                if windows != planned["state_bytes"] or frames != planned["feature_frame_bytes"]:
                    raise MemoryError("materialized DFlash2 windows/features differ from the admitted extents")
                self.capacity_plan["dflash2"]["materialized"] = {
                    "weight_bytes": runtime.nbytes, "window_bytes": windows, "feature_frame_bytes": frames,
                    "evidence": "actual-local-owned-tensor-extents; device-peak-and-model-qualification-pending"}
            if owner_preparation is not None:
                owner_preparation.adopt(w, self.multi)
            if resident_windows is not None:
                capacities = [st.capacity for st in self.multi.free]
                owned_cache_bytes = sum(st.owned_cache_bytes() for st in self.multi.free)
                local_materialization = [rank, int(self.multi.full_resident), owned_cache_bytes,
                                         int(torch.cuda.memory_allocated()), int(torch.cuda.memory_reserved()), *capacities]
                rank_materializations = gather(local_materialization) if tp > 1 else [local_materialization]
                if any(row[0] != peer or len(row) != 5 + streams or
                       self.multi.full_resident and (row[1] != 1 or any(n != self.max_len for n in row[5:]))
                       for peer, row in enumerate(rank_materializations)):
                    raise RuntimeError("ranks did not materialize every promised full context window")
                self.capacity_plan["kv_full_windows"].update(
                    materialized=self.multi.full_resident, startup_capacities=capacities,
                    allocated_cache_bytes=owned_cache_bytes,
                    ranks=[{"rank": row[0], "materialized": bool(row[1]), "allocated_cache_bytes": row[2],
                            "torch_allocated_bytes": row[3], "torch_reserved_bytes": row[4],
                            "startup_capacities": row[5:]} for row in rank_materializations])
            if self.multi.graph_budget != graph_bytes:   # admission planned exactly what the decoder may capture
                if sm70 and self.multi.graph_record["reason"] is not None and self.multi.graph_budget == 0:
                    from .service_memory import release_uncaptured_graph_plan

                    release_uncaptured_graph_plan(self.capacity_plan, graph_bytes)
                    graph_bytes = 0
                else:
                    raise RuntimeError(f"the startup plan reserved {graph_bytes} bytes for captured rounds, but the "
                                       f"decoder's budget is {self.multi.graph_budget} "
                                       f"({self.multi.graph_record['reason'] or 'capturable'})")
            if automatic_experts:
                self._configure_auto_experts(torch, streams, vision_workspace_bytes=workspace)
            # A managed replacement binds the persistent adapter only after
            # all ranks have loaded, allocated and warmed successfully.
            if self._admission_options["admission_policy"] == "fifo":
                from .prompt_plan import POINTS
                from .retained_catalog import make_retention_catalog

                catalog = make_retention_catalog(self.retention_policy, w, self.multi.states,
                    prefix_limit=self.multi.keep, recovery_versions=POINTS,
                    register=lambda owner: setattr(self, "retention_catalog", owner))
                retained_host = self._retention_header_resources.global_host_bytes
                if retained_host > self.capacity_plan["host_steady_workspace_bytes_estimate"]:
                    raise RuntimeError("retention host reservation is absent from the chosen steady header")
                catalog.configure_host_resources(
                    baseline_host_bytes=self.capacity_plan["host_steady_commitment_bytes"] - retained_host,
                    materialized_host_bytes=0,
                    **({"prepaid_native_fixed_payload_bytes": self._native_protocol_resources["fixed"]["global_native_fixed_payload_bytes"],
                        "prepaid_native_prefix_payload_bytes": self._native_protocol_resources["prefix"]["global_native_prefix_payload_bytes"],
                        "native_state_slots": self._native_protocol_resources["fixed"]["maximum_state_slots"],
                        "native_fd_limit": self._native_protocol_resources["fixed"]["nofile_hard_limit"],
                        "native_host_bytes": native_host_bytes}
                       if self._native_protocol_resources is not None else {}))
                catalog.configure_device_resources(**self._retention_device_resources["limits"])
                self.multi.bind_retention_catalog(catalog)
                self.capacity_plan["retention"] = {
                    **catalog.record(), "full_future_session_bytes": catalog.future_claim()}
                self._publish_retention_catalog(catalog)
            if not managed and self.comm is not None:
                from .tp_scheduler import TensorParallelDecoder

                self.tp_decoder = TensorParallelDecoder(self.multi, self)
        else:
            self.e = Engine(w, capacity=self.max_len, max_rows=max(8, self.depth + 1), graphs=graphs,
                            kv_pair=self.kv_pair, prefill_rows=self.prefill_rows, logical_context=self.context_window)
            if automatic_experts:
                self.serial = self.e.twin()
                self._configure_auto_experts(torch, streams, vision_workspace_bytes=workspace)
        if history_plan is not None:
            history_states = list(self.multi.free) if self.multi is not None else [self.e.st]
            if self.serial is not None:
                history_states.append(self.serial.st)
            actual_host = sum(st.host_cache_bytes() for st in history_states)
            actual_device = sum(st.indexer_history_owner.device_nbytes for st in history_states)
            owners = len(history_states)
            allocator_receipt = self._indexer_history_session.allocator.receipt
            valid = (all(st.indexer_history_owner.limit == self.max_len for st in history_states)
                     and actual_host == owners * history_plan.host_bytes(self.max_len)
                     and actual_device == owners * history_plan.device_nbytes
                     and allocator_receipt["charged_bytes"] == actual_host
                     and allocator_receipt["pinned_extent_bytes"] == owners * history_plan.pinned_bytes)
            local = [rank, int(valid), owners, actual_host, actual_device,
                     allocator_receipt["pinned_extent_bytes"]]
            materializations = gather(local) if tp > 1 else [local]
            if any(row[0] != peer or row[1] != 1 for peer, row in enumerate(materializations)):
                raise RuntimeError("RAM indexer startup allocation disagrees with its admitted archive/staging plan")
            self.capacity_plan["indexer_history"].update(materialized=True,
                startup_ranks=[{"rank": row[0], "owners": row[2], "host_backing_bytes": row[3],
                                "device_raw_bytes": row[4], "pinned_staging_bytes": row[5]}
                               for row in materializations], local_allocator=allocator_receipt)
        if self.multi is not None:
            from .rotor_decoder import MTP_FORMATS, SERIAL_FORMATS, select
            runtime = None
            if self.kv_pair.symmetric and self.kv_pair.key_dtype in SERIAL_FORMATS + MTP_FORMATS:
                import hashlib
                import platform
                from importlib.metadata import version
                try:
                    with Path('/proc/driver/nvidia/version').open('rb') as driver:
                        driver_raw = driver.read(32769)
                except OSError:
                    driver_raw = b''
                runtime = dict(torch=torch.__version__, torch_git=torch.version.git_version,
                               CUDA=torch.version.cuda, triton_distribution=version('triton'),
                               device=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                               kernel_release=platform.release(),
                               driver_proc_sha256=hashlib.sha256(driver_raw).hexdigest())
            self.multi.rotor_decoder_policy = select(mode=self.kv_pair.key_dtype,
                symmetric=self.kv_pair.symmetric, head_dim=head_dim, context=self.context_window,
                slots=streams, depth=self.depth, confidence=self.confidence, exl3=exl3, graphs=graphs,
                prefetch=prefetch, automatic_experts=automatic_experts, vision=vision, tp=tp,
                yarn_factor=yarn_factor, runtime=runtime)
            self.capacity_plan['rotor_decoder_lookup'] = dict(
                serial=self.multi.rotor_decoder_policy.serial, mtp=self.multi.rotor_decoder_policy.mtp,
                concurrent_requests=False, encoder_changed=False)
        started = time.perf_counter()
        locked = False
        if prefetch and not ple_on_ssd:               # the n-gram tables' pages, read now rather than by requests
            from tensorfold.cuda.host_pin import admit as admit_pins, table_bytes
            from tensorfold.cuda.capacity import unified

            tables = {id(layer.ple.table): layer.ple.table for layer in w.layers if layer.ple is not None}
            for table in tables.values():
                if not tables_read:
                    table.prefetch()                  # complete reads remain reclaimable before pin admission
            room = self.capacity_plan["budget_bytes"] - self.capacity_plan["total_bytes_estimate"]
            future = (max(0, self.capacity_plan["serving_peak_bytes_estimate"] - int(torch.cuda.memory_allocated()))
                      if unified(torch) else 0)
            pins = admit_pins(table_bytes(tables.values()), device_room_bytes=room, future_bytes=future)
            self.capacity_plan["ngram_pin_admission"] = pins
            locked = bool(tables) and pins["admitted"]
            for table in tables.values():
                if pins["admitted"]:
                    # A refused lock rolls back this table's owned pages. Keep
                    # every table result; the last table cannot mask a refusal.
                    locked = table.lock() and locked
            pins["all_selected_tables_locked"] = locked
        read_s = time.perf_counter() - started
        captured = self.e.graphs.warm(self.depth + 1) if self.e is not None and self.e.graphs is not None else 0
        started = time.perf_counter()
        if self.concurrent:
            if managed:
                self._profile_build.phase("warming")
            self.multi.warm()                            # and captures its rounds' graphs
            if (graph_bytes and not self.multi.graph_budget and self.multi.graphs is None
                    and self.multi._failed_graphs is None):
                from .service_memory import release_uncaptured_graph_plan

                release_uncaptured_graph_plan(self.capacity_plan, graph_bytes)
                graph_bytes = 0
            floor_extra = getattr(self.multi, "graph_floor_extra_bytes", 0)
            if graph_request is None:
                from .service_memory import reserve_graph_plan, refine_materialized_graph_plan

                graph_bytes = self.multi.graph_budget
                refinement = self.multi.graph_record.get("observed_graph_admission")
                if refinement is not None:
                    refine_materialized_graph_plan(self.capacity_plan, refinement)
                reserve_graph_plan(self.capacity_plan, graph_bytes + floor_extra)
            elif floor_extra:
                from .service_memory import reserve_graph_plan

                reserve_graph_plan(self.capacity_plan, floor_extra)
            if self.multi.graph_budget != graph_bytes or self.capacity_plan["total_bytes_estimate"] > self.capacity_plan["budget_bytes"]:
                raise RuntimeError("resolved captured-round budget differs from its admitted service plan")
            self.capacity_plan["multi_graphs"] = {"planned_bytes": graph_bytes, **self.multi.graph_record}
        else:
            from .decode import warm

            warm(self.e)
        if self.vision is not None:
            self.vision.warm()
            torch.cuda.empty_cache()
        warm_s = time.perf_counter() - started
        self.eos = tuple(w.cfg.eos)
        self.model_dir = Path(model_dir)
        self.served = 0
        self.cache: list[tuple[list[int], dict]] = []    # (committed ids, what resuming from them needs)
        if not automatic_experts or self.concurrent:
            self.serial = None                            # the serial requests' engine, made on first use
        rule = (f"1 to {self.depth} MTP drafts a round, a chain stops before a later draft under "
                f"{self.confidence:.0%}" if self.depth else "no drafts: the serial reference, one token a round")
        if dspark_plan is not None:
            rule = f"1 to {self.depth} DSpark proposals a round, learned confidence threshold {self.confidence:.0%}"
        graphs_note = ""
        if self.concurrent:
            record = self.multi.graph_record
            graphs_note = (f"{record['verify_keys']} verify and {record['mtp_keys']} MTP step graphs captured "
                           f"({record['bytes'] / 2**20:.0f} MiB of {record['budget_bytes'] / 2**20:.0f}, "
                           f"{record['capture_s']:.1f}s)" if record["captured"] else f"eager ({record['reason']})")
            if record.get("stopped"):
                graphs_note += f"; capture stopped: {record['stopped']}"
        where = (f"up to {streams} streams, each growing to {self.context_window} prompt/reply tokens while memory "
                 f"lasts ({self.multi.memory_gate.room / 2**30:.1f} GiB free for their caches, "
                 f"{self.multi.window_bytes / 2**30:.2f} GiB for one at the full window), {graphs_note}"
                 if self.concurrent else
                 f"{self.context_window}-token prompt/reply window; {self.max_len}-token cache")
        if resident_windows is not None and self.concurrent:
            where = (f"{min(streams, resident_windows)} full {self.context_window}-token prompt/reply windows "
                     f"reserved across {streams} streams; {graphs_note}")
        if ple_on_ssd:
            how = "read from SSD at each lookup"
        elif tables_read:                             # read during the load: the wait after it, then any lock
            how = f"read alongside the weights ({waited:.1f}s after them)" + (
                f", locked in memory in {read_s:.1f}s" if locked else "")
        else:
            how = f"{'locked in memory' if locked else 'read'} in {read_s:.1f}s"
        quantized = " (quantized cache changes output)" if min(self.kv_pair.key.bits, self.kv_pair.value.bits) < 16 else ""
        kv = ("" if self.kv_pair.key_dtype == self.kv_pair.value_dtype == "bf16" else
              f"; KV cache K={self.kv_pair.key_dtype}, V={self.kv_pair.value_dtype}{quantized}")
        print(f"[tensorfold] Flash Next on CUDA: {rule}; {where}{kv}; n-gram tables {how}; {captured} "
              f"decode graphs captured; idle prompt pieces {self.prefill_rows} rows; "
              f"prompt kernels warmed in {warm_s:.1f}s", flush=True)
        if ram_layout is not None:
            from tensorfold.cuda.capacity import host_memory_snapshot

            host = self.capacity_plan["host_after_load"] = host_memory_snapshot()
            available = ("unavailable" if host["mem_available_bytes"] is None else
                         f"{host['mem_available_bytes'] / 2**30:.2f} GiB")
            remaining = ("unlimited or not exposed" if host["cgroup_remaining_bytes"] is None else
                         f"{host['cgroup_remaining_bytes'] / 2**30:.2f} GiB")
            if host["cgroup_read_error"] is not None:
                remaining = f"unavailable ({host['cgroup_read_error']})"
            print(f"[tensorfold] host after load: MemAvailable {available}; cgroup hard-limit room {remaining}; "
                  "advisory snapshot, competing allocations can consume it", flush=True)
        if execution_profile_manifest is not None:
            from .execution_identity import bind_execution_manifest

            self.capacity_plan["execution_profile_binding"] = (
                self._profile_build.bind_execution_manifest(execution_profile_manifest) if managed else
                bind_execution_manifest(self, execution_profile_manifest))
            self.multi.execution_manifest_installer = self._install_execution_manifest
        # Public FIFO configuration changes admission semantics. Synthetic
        # warmup must finish on every rank before installing those semantics.
        if self.concurrent and not managed and (self.comm is None or self.rank == 0):
            self.scheduler = Scheduler(self.tp_decoder if self.comm is not None else self.multi,
                                       max_streams=streams, **self._admission_options)
        if self.scheduler is not None and not managed:
            self.scheduler.start()            # warmup's synthetic streams belong solely to the constructor

    def _install_execution_manifest(self, relative_name):
        """Worker-boundary registration confined to the initial evidence root."""
        from .execution_identity import bind_execution_manifest

        root = getattr(self, "_execution_profile_root", None)
        if root is None:
            raise ValueError("execution manifests require the opt-in startup manifest boundary")
        receipt = bind_execution_manifest(self, relative_name, rebind=True, manifest_root=root)
        self.capacity_plan["execution_profile_binding"] = receipt
        return receipt

    def _drain_startup_reads(self) -> None:
        """Observe every accepted read; retain any interrupted pending owner."""
        reads = self._startup_reads
        primary = None
        prior = None
        try:
            finish((future.result for future in reads))
        except BaseException as error:
            primary = error
            prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error), BaseException.__suppress_context__.__get__(error))
        errors = []
        try:
            if all((future.done() for future in reads)):
                reads.clear()
        except BaseException as error:
            errors.append(error)
        if primary is not None:
            if errors:
                errors.extend((root for root in prior[:2] if root is not None and root is not primary))
            else:
                BaseException.__cause__.__set__(primary, prior[0])
                BaseException.__context__.__set__(primary, prior[1])
                BaseException.__suppress_context__.__set__(primary, prior[2])
        raise_failures(primary, errors)

    def _rollback_startup(self) -> None:
        """Read borrowers retire before plans, files or the communicator."""
        primary = None
        try:
            self._drain_startup_reads()
        except BaseException as error:
            primary = error
        if self._startup_reads:
            if primary is not None:
                raise_failures(primary, [])
            raise RuntimeError('unfinished startup read owners remain; engine retained')
        if primary is None:
            self.close(abort=True)
        else:
            finish((lambda: raise_failures(primary, []), lambda: self.close(abort=True)))

    def _send_shutdown(self) -> None:
        """Rank0's existing stop frame, only while admission/callers are quiescent."""
        if self.tp > 1 and self.rank == 0 and (not self._shutdown_sent):
            if getattr(self, "tp_decoder", None) is not None:
                self.tp_decoder.shutdown()
                self._shutdown_sent = True
                return
            text = json.dumps({'stop': True})
            for peer in range(1, self.tp):
                self.comm.store.set(self._key(self.served, peer), text)
            self._shutdown_sent = True

    def _configure_auto_experts(self, torch, streams: int, *, vision_workspace_bytes: int = 0) -> None:
        """Finish fixed device state, then spend its remaining bytes on experts.

        Runs before Scheduler starts or any warmup borrows a cache lease. Full
        context growth, a layer's resize overlap and every kept/in-flight
        snapshot remain available; the cache is fixed throughout serving. An
        explicit --kv-full-windows shares that many full windows across slots,
        with no lone-stream escape from the selected growth budget.
        """
        from tensorfold.cuda.capacity import cuda_limit_bytes

        from ..ram_experts import auto_pool_bytes, reserved_growth
        from .prompt_plan import POINTS

        cache = self.w.meta["expert_cache"]
        headroom = self.expert_headroom_bytes
        decoder = self.multi if streams > 1 else self.e
        states = self.multi.free if streams > 1 else [self.e.st, self.serial.st]
        each = [st.cache_bytes(st.limit) - st.cache_bytes() for st in states]
        full_growth = sum(each)
        growth, windows = reserved_growth(each, self.kv_full_windows)
        # kept states, plus the POINTS (CUTS inside a piece, one at its end) a filling stream's pass copies out first
        retained = (self.multi.keep + streams * POINTS) if streams > 1 else KEEP_SERIAL + 1
        from .service_memory import future_service

        service = future_service(self.w, states, streams=streams, retained=retained, growth_bytes=growth,
                                 vision_workspace_bytes=vision_workspace_bytes)
        overlap = service["copy_overlap_bytes"]
        snapshot, snapshots = service["snapshot_bytes_each"], service["snapshot_bytes"]
        transient = service["transient_bytes"]
        if self.w.x3 is not None:
            from .exl3_mm import X3

            scratch = self.w.x3
            projections = [u for u in scratch.users if isinstance(u, X3) and not u.head]
            if projections:
                # These grow-only workspaces otherwise appear during the first
                # prompt, after the expert pool has already occupied the device.
                scratch.prefill._grow("w", max(u.k * u.n for u in projections), self.w.device)
                scratch.prefill._grow("xh", decoder.pbuf.rows * max(u.k * (self.w.cfg.streams
                                      if self.w.mtp is not None and u is self.w.mtp.fc_h else 1) for u in projections),
                                      self.w.device)
                scratch.prefill.hadamard(self.w.device)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = map(int, torch.cuda.mem_get_info())
        accounted = int(torch.cuda.memory_allocated())
        limit = cuda_limit_bytes()
        # The configured cap is absolute live allocation, rather than another
        # free-memory allowance. Fixed service state has already consumed it.
        room = free if limit is None else max(0, min(free, limit - accounted))
        control = 32 if hasattr(cache, "control_device_bytes") else 0
        decision = auto_pool_bytes(room, cache.entry_bytes, cache.score_history_entries, control_bytes=control,
                                   growth_bytes=growth + overlap, snapshot_bytes=snapshots, workspace_bytes=transient,
                                   headroom_bytes=headroom)
        cache.configure_before_use(decision["gpu_bytes"])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        final_free, final_total = map(int, torch.cuda.mem_get_info())
        final_allocated = int(torch.cuda.memory_allocated())
        final_spare = int(torch.cuda.memory_reserved()) - final_allocated
        future = headroom + growth + overlap + snapshots + transient
        if (final_total != total or final_free < headroom
                or final_free + final_spare < future
                or (limit is not None and final_allocated + future > limit)):
            raise MemoryError("automatic expert pool cannot retain its declared full service workspace")
        if streams > 1:
            from tensorfold.cuda.memory_gate import MemoryGate, torch_live

            # Auto is discrete-only. Read raw device free bytes so the selected
            # headroom is charged once and cannot change with the environment.
            raw_live = torch_live(torch, lambda runtime: int(runtime.cuda.mem_get_info()[0]))
            # _grow already charges its one-layer copy overlap in fits(extra).
            self.multi.memory_gate = MemoryGate(growth + overlap, reserve=0,
                                                live=lambda: max(0, raw_live() - headroom),
                                                allow_lone_growth=self.kv_full_windows is None)
        if self.kv_full_windows is not None:
            decision["policy"] = "remaining-device-bytes-after-fixed-service-state-shared-kv-v1"
        authority, pin_receipt = _expert_authority(cache)
        decision.update(physical_total_bytes=total, fixed_allocated_bytes_before_pool=accounted,
                        host_pin=pin_receipt,
                        physical_free_bytes_before_pool=free, cuda_memory_limit_bytes=limit,
                        allocated_bytes_after_pool=final_allocated,
                        physical_free_bytes_after_pool=final_free, kept_snapshot_count=retained,
                        allocator_spare_bytes_after_pool=final_spare, vision_workspace_bytes=vision_workspace_bytes,
                        snapshot_bytes_each=snapshot, full_context_growth_bytes=full_growth,
                        reserved_context_growth_bytes=growth, kv_full_windows_requested=self.kv_full_windows,
                        kv_full_windows_reserved=windows, kv_stream_slots=len(states),
                        layer_resize_overlap_bytes=overlap, prompt_transient_bytes=transient,
                        prefill_rows=decoder.pbuf.rows, shared_attention_routing_recurrent_buffers_already_allocated=True,
                        lazy_EXL3_prompt_workspace_materialized=self.w.x3 is not None,
                        pool_configured_before_first_lease=True)
        initial_plan = deepcopy(self.capacity_plan)
        self.capacity_plan["initial_minimum_pool_admission"] = initial_plan
        self.capacity_plan["vram_experts_auto"] = decision
        pool = self.capacity_plan["vram_experts"]
        pool.update(gpu_bytes=cache.gpu_bytes, slots=cache.capacity,
                    resident_capacity=cache.resident_capacity, safe_lease_capacity=cache.capacity,
                    control_device_bytes=getattr(cache, "control_device_bytes", 0),
                    control_host_bytes=getattr(cache, "control_host_bytes", 0), automatic=True)
        initial_pool = initial_plan["vram_experts"]
        pool_delta = cache.gpu_bytes + pool["control_device_bytes"] - initial_pool["gpu_bytes"] - initial_pool["control_device_bytes"]
        final_weight_estimate = initial_plan["weight_bytes_estimate"] + pool_delta
        final_serving_estimate = decision["allocated_bytes_after_pool"] + growth + overlap + snapshots + transient
        self.capacity_plan.update(weight_bytes_estimate=final_weight_estimate,
            startup_peak_bytes_estimate=max(initial_plan["startup_peak_bytes_estimate"], decision["allocated_bytes_after_pool"]),
            serving_peak_bytes_estimate=final_serving_estimate,
            total_bytes_estimate=max(initial_plan["startup_peak_bytes_estimate"], final_serving_estimate),
            cache_workspace_bytes_estimate=final_serving_estimate - final_weight_estimate,
            full_mapped_working_set_bytes_estimate=max(initial_plan["startup_peak_bytes_estimate"], final_serving_estimate)
                                                   + initial_plan["mapped_table_bytes"],
            estimate_scope="auto: measured fixed device allocation plus explicit future service bounds; initial header loading forecast retained separately")
        reservation = (f"{streams} full-context slots" if self.kv_full_windows is None else
                       f"{windows} full-context windows shared by {streams} slots (waits and request errors possible)")
        print(f"[tensorfold] RAM experts auto: {cache.host_bytes / 2**30:.2f} GiB {authority}, "
              f"{cache.gpu_bytes / 2**30:.2f} GiB GPU pool ({cache.resident_capacity} resident cells, "
              f"{cache.capacity} experts per lease); "
              f"{reservation} and {retained} snapshots reserved, {headroom / 2**20:g} MiB headroom", flush=True)

    @property
    def kv_dtype(self) -> str:
        """Legacy symmetric cache name; mixed callers use kv_pair."""

        if not self.kv_pair.symmetric:
            raise ValueError("mixed KV engine has no single dtype; use kv_pair")
        return self.kv_pair.key_dtype

    def _bind_coordinate_profile(self, weights) -> None:
        """Issue one resident-model namespace, before bound State construction.

        The owning engine binds actual loaded weights and effective numerics.
        An activation nonce conservatively forbids reuse after A-to-B-to-A;
        this namespace is neither a content nor a qualification certificate.
        Existing startup CPU phases own publication, failure and key retirement.
        """
        import hashlib
        import secrets
        from dataclasses import asdict
        import torch
        from .execution_identity import _ManifestRefusal, _fatal, _phase, _retire_store, digest

        local = {}

        def issue():
            if (weights is not self.w or weights.positions is None or self.position_profile is None
                    or weights.positions.profile.identity != self.position_profile.identity):
                raise ValueError("coordinate activation requires adopted numerical position operands")
            managed = getattr(self, "_profile_build", None)
            candidate = None if managed is None else managed.candidate
            authority = ({"kind": "prepared-serving-profile", "identity": candidate.identity,
                          "activation_epoch": managed.sequence} if candidate is not None else None)
            pairing = getattr(self, "_dflash_pairing", None)
            if authority is None and pairing is not None:
                authority = {"kind": "model-content-authority", "identity": pairing.authority.model_prefix,
                             "pairing_identity": pairing.identity}
            # Config's immutable tuples have their specified JSON-array encoding;
            # no string/default coercion is permitted for unknown runtime objects.
            config_record = json.loads(json.dumps(asdict(weights.cfg), allow_nan=False))
            contract = {"rank": self.rank, "model_config": config_record,
                        "provider": self.execution_provider, "around_one": weights.around_one,
                        "position_profile": self.capacity_plan["position_profile"],
                        "kv_identity": self.kv_pair.identity, "method": self.method_geometry.receipt(),
                        "context_window": self.context_window, "sm70_policy": self.sm70_profile,
                        "nccl_graph_policy": (None if getattr(self, "_nccl_graph_inputs", None) is None else
                                              self._nccl_graph_inputs["effective"]),
                        "family_manifest": getattr(self, "_sm70_weight_manifest_identity", None),
                        "torch": str(torch.__version__), "torch_git": torch.version.git_version,
                        "cuda": torch.version.cuda, "authority": authority,
                        "prompt_fp8": prompt_precision.fp8(), "local_vision_owner": getattr(self, "vision", None) is not None,
                        "media_inputs_enabled": self._media_inputs_enabled,
                        "media_profile": None if getattr(self, "media_profile", None) is None else self.media_profile.identity,
                        "media_source": None if getattr(self, "_media_authority", None) is None else self._media_authority.model_prefix,
                        "media_numerics": None if getattr(self, "_media_numeric_scope", None) is None else self._media_numeric_scope.record(),
                        "media_software": (self.vision.software_binding.record()
                            if getattr(getattr(self, "vision", None), "is_media_runtime", False) else None)}
            local.update(contract_identity=digest(contract), authority=authority)
            return {"rank": self.rank, "contract": local["contract_identity"],
                    "nonce": secrets.token_hex(32) if self.rank == 0 else None}

        channel = "tensorfold/flashnext/coordinate-profile"
        stages = ("issue", "agreement")

        def retire():
            if self.comm is not None:
                try:
                    _retire_store(self, channel, stages=stages)
                except BaseException as error:
                    _fatal(self, error)
                    raise

        def operation():
            try:
                issued = _phase(self, channel, "issue", issue)

                def agree():
                    if (len(issued) != self.tp or any(value["rank"] != rank for rank, value in enumerate(issued))
                            or any(value["nonce"] is not None for value in issued[1:])
                            or not isinstance(issued[0]["nonce"], str) or len(issued[0]["nonce"]) != 64
                            or any(c not in "0123456789abcdef" for c in issued[0]["nonce"])):
                        raise ValueError("coordinate activation issuance is malformed")
                    identity = hashlib.sha256(json.dumps({"version": 1, "activation": issued[0]["nonce"],
                        "rank_contracts": [value["contract"] for value in issued]},
                        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    # Publication is part of the voted phase: even a Python
                    # allocation/store failure must contain every participant.
                    binding = {"identity": identity, "scope": "resident-activation",
                        "content_certificate": False, "qualification_certificate": False,
                        "local_contract_identity": local["contract_identity"], "trusted_authority": local["authority"]}
                    self.coordinate_profile_binding = binding
                    weights.meta["coordinate_profile_identity"] = identity
                    weights.meta["coordinate_profile_binding"] = binding
                    return identity
                identities = _phase(self, channel, "agreement", agree)
                if any(value != identities[0] for value in identities):
                    error = RuntimeError("coordinate activation identities disagree")
                    _fatal(self, error)
                    raise error
            except _ManifestRefusal as error:
                rollback(self, error, retire)
            retire()
            return identities[0]

        construction = None if self.comm is None else getattr(self.comm, "_profile_construction", None)
        if construction is not None and construction.active:
            construction.protocol("coordinate-profile", operation, refusal=_ManifestRefusal)
        else:
            operation()

    def _same_settings(self, torch, ids) -> None:
        """Every rank agrees on execution, sampling and cache identities before payload loading."""

        import hashlib
        import os
        import struct

        draft_digest = (int.from_bytes(hashlib.sha256(ids.tobytes()).digest()[:8], "big") & (2**63 - 1)
                        if ids is not None else -1)
        rope_digest = int.from_bytes(hashlib.sha256(json.dumps(self.rope.metadata(), sort_keys=True).encode()).digest()[:8],
                                     "big") & (2**63 - 1)
        draft_path = getattr(self, "mtp_model_dir", None)
        native_policy = None
        expert_policy = proposal_policy = context_policy = attention_policy = prefix_policy = None
        if getattr(self, "execution_provider", "cuda") == "sm70":
            from tensorfold.cuda.sm70 import family_ops, routing_native
            from . import padded_linears
            from .gguf_weights import GGUFExperts, GGUFConcat
            from .dspark_proposal_graphs import _ENABLED as proposal_enabled
            from .dspark_context_graphs import _ENABLED as context_enabled

            expert_policy = [GGUFExperts.implementation, GGUFExperts.packed_rows, GGUFConcat.implementation]
            proposal_policy = bool(proposal_enabled)
            context_policy = bool(context_enabled)
            attention_policy = getattr(self, "_sm70_attention_identity", None)
            from .prefix_policy import identity as prefix_identity

            prefix_policy = prefix_identity(getattr(self, "sm70_profile", None))

            native_policy = [family_ops._NATIVE_FAMILY, family_ops._NATIVE_ATTENTION, padded_linears._ENABLED,
                             routing_native._NATIVE_LOGITS]
        policy = ["stored-basis-native64-v1", getattr(self, "execution_provider", "cuda"),
                  str(draft_path.resolve()) if draft_path is not None else None,
                  getattr(self, "_sm70_weight_manifest_identity", None),
                  self.streams > 1 and os.environ.get("TENSORFOLD_MTP_COPY", "0") == "1",
                  getattr(self, "_graph_request", None), native_policy,
                  getattr(self, "_gguf_prefill_receipt", None),
                  getattr(getattr(self, "_dspark_plan", None), "identity", None),
                  getattr(getattr(self, "_dflash_pairing", None), "identity", None),
                  expert_policy, proposal_policy, context_policy, attention_policy,
                  getattr(self, "sm70_profile", None), prefix_policy, getattr(self, "kv_full_windows", None)]
        if getattr(self, "position_profile", None) is not None:
            policy.append(["position-profile", self.position_profile.identity, self.position_profile.epoch,
                           self.position_method_binding])
        admission = getattr(self, "_admission_options", None)
        retention = getattr(self, "retention_policy", None)
        if admission is not None and admission["admission_policy"] == "fifo":
            policy.append(["fifo-retention-service-policy", admission,
                           None if retention is None else retention.record(),
                           {"store_use_libuv": self._store_use_libuv}])
        if getattr(self, "_sm70_kv_policy", None) == "stored-rotor-basis-fp16-fp32acc-v1":
            policy.append(["sm70-kv-working-policy", self._sm70_kv_policy])
        if getattr(self, "indexer_history", "gpu") == "ram":
            from .service_memory import RAM_GRAPH_ADMISSION_VERSION

            history_plan = self._indexer_history_plan
            policy.append(["indexer-history", "lossless-fp16-ram-archive-v1",
                           [getattr(history_plan, field) for field in
                            ("layers", "index_dim", "ratio", "prefill_rows", "decode_rows", "page_size")],
                           RAM_GRAPH_ADMISSION_VERSION])
        if getattr(self, "sm70_profile", None) == "source-packed-graphs-trial":
            from .service_memory import GPU_UNIQUE_GRAPH_ADMISSION_VERSION

            policy.append(["unique-gpu-graph-admission", GPU_UNIQUE_GRAPH_ADMISSION_VERSION])
        policy_digest = int.from_bytes(hashlib.sha256(json.dumps(policy).encode()).digest()[:8], "big") & (2**63 - 1)
        confidence_bits = struct.unpack("<q", struct.pack("<d", self.confidence))[0]
        mine = torch.tensor([self.depth, confidence_bits, self.max_len,
                             self.streams, int(self.graphs_enabled), self.prefill_rows,
                             len(ids) if ids is not None else -1, draft_digest, self.kv_pair.handshake, policy_digest,
                             rope_digest, int(prompt_precision.fp8())], dtype=torch.int64, device="cuda")
        both = torch.empty((self.tp * mine.numel(),), dtype=torch.int64, device="cuda")
        self.comm.all_gather(mine, both)
        both = both.view(self.tp, -1).cpu()
        for peer in range(1, self.tp):
            prompt_precision.same_on_ranks(int(both[0, -1]), int(both[peer, -1]))
            if not torch.equal(both[0], both[peer]):
                raise RuntimeError("the tensor-parallel ranks were started with different settings "
                                   "(drafts, confidence, context, parallel streams, graphs, prompt rows, "
                                   "draft vocabulary, copy drafts, KV cache, execution provider, RoPE): "
                                   f"rank 0 {both[0].tolist()}, rank {peer} {both[peer].tolist()}")

    def _key(self, n: int, peer: int | None = None) -> str:
        key = f"tensorfold/flashnext/request/{n}"
        # One consumable frame per follower: deleting rank1's frame must not
        # hide the request or shutdown from ranks2..7. Retain the two-rank key.
        if getattr(self, "tp", 2) > 2:
            return key + f"/rank/{getattr(self, 'rank', 0) if peer is None else peer}"
        return key

    def shutdown(self) -> None:
        """Rank 0: tell every peer to leave ``follow``."""
        if self.tp > 1 and self.rank == 0:
            operation = threading.Lock()
            with operation:
                primary = None
                try:
                    self._begin_request(operation)
                    with self._lifecycle:
                        self._closing = True
                        self._lifecycle.notify_all()
                    self._send_shutdown()
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    self._retire_request(operation, primary)

    def _share(self, prompt: list[int], max_tokens: int, sampling, draft: bool, cached: int, constraint=None,
               stop_eos: bool = True, token_limit: bool = False) -> tuple:
        from tensorfold.engine.grammar import pack

        points = getattr(self, "points", None)
        body = {"prompt": prompt, "max_tokens": max_tokens, "draft": bool(draft), "cached": int(cached),
                "points": list(points(prompt)) if draft and points is not None else [],
                "stop_eos": bool(stop_eos),
                "token_limit": bool(token_limit),
                "sampling": None if sampling is None else [int(sampling.seed), float(sampling.temperature),
                                                           int(sampling.top_k), float(sampling.top_p),
                                                           float(sampling.min_p)],
                "grammar": pack(constraint)}                 # all peers walk and mask the same rows
        text = json.dumps(body)
        for peer in range(1, getattr(self, "tp", 2)):
            self.comm.store.set(self._key(self.served, peer), text)
        return self._unpack(text)

    def _receive(self) -> tuple | None:
        from torch.distributed import DistNetworkError
        from tensorfold.cuda.store_wait import timed_out
        key = self._key(self.served)
        while True:
            try:
                self.comm.store.wait([key], timedelta(hours=1))
                break
            except DistNetworkError:
                with self._lifecycle:
                    self._abort_comm = True
                print(f'[tensorfold] rank 0 closed the connection; rank {getattr(self, "rank", 1)} stops', flush=True)
                return None
            except Exception as error:
                if not timed_out(error, [key], timedelta(hours=1)):
                    raise
                with self._lifecycle:
                    if self._closing:
                        return None
        text = self.comm.store.get(key).decode()
        self.comm.store.delete_key(key)
        return self._unpack(text)

    @staticmethod
    def _unpack(text: str) -> tuple | None:
        from tensorfold.engine.exact_sampling import Sampling

        body = json.loads(text)
        if body.get("stop"):
            return None
        if type(body.get("token_limit", False)) is not bool:
            raise ValueError("a shared token-prefix policy flag must be boolean")
        s = body["sampling"]
        return (body["prompt"], body["max_tokens"], None if s is None else Sampling(*s),
                body["draft"], body["cached"], body.get("grammar") or [], body.get("stop_eos", True),
                body.get("points", []), body.get("token_limit", False))

    @property
    def context_window(self) -> int:
        """Prompt and reply capacity after reserving speculative scratch positions."""

        plan = getattr(self, "capacity_plan", None)
        return plan["context_window"] if plan is not None else max(0, self.max_len - self.depth - 1)

    def _limit(self, prompt: list[int], max_tokens: int) -> int:
        room = self.context_window - len(prompt)
        if room < 1:
            raise ValueError(f"a prompt of {len(prompt)} tokens leaves no room in the {self.context_window}-token context")
        return max(1, min(max_tokens, room))

    def _resume(self, prompt: list[int]):
        """The longest kept state the prompt extends (with at least one new token), or None."""

        best = None
        for ids, snap in self.cache:
            if len(ids) < len(prompt) and prompt[:len(ids)] == ids and (best is None or len(ids) > len(best[0])):
                best = (ids, snap)
        return best

    def _start_from(self, hit) -> None:
        """Before a prefill: resuming overwrites the cache rows past the kept prefix, so the states that extend it go; a fresh prompt overwrites them all."""

        if hit is None:
            self.cache = []
        else:
            n = len(hit[0])
            self.cache = [c for c in self.cache if len(c[0]) < n and hit[0][:len(c[0])] == c[0]] + [hit]

    def _remember(self, ids: list[int], snap: dict) -> None:
        self.cache = [c for c in self.cache if c[0] != ids][-(KEEP_SERIAL - 1):] + [(ids, snap)]

    @property
    def supports_logprobs(self) -> bool:
        return self.tp == 1

    def _serial(self, prompt: list[int], max_tokens: int, sampling, on_tokens, constraint=None,
                stop_eos: bool = True, probabilities=None, token_limit=None) -> dict[str, Any]:
        """One token a round from a fresh prefill in the serial engine's own state (no drafts, no kept states)."""

        import torch

        from .decode import prefill, serial_decode
        from .speculation import collective_prefix_limit
        from tensorfold.cuda.streams import prefix_limit

        if token_limit is not None:
            token_limit = collective_prefix_limit(token_limit, self.comm)

        if self.serial is None:
            self.serial = self.e.twin()
        t0 = time.perf_counter()
        first = prefill(self.serial, prompt, sampling, mtp=False, constraint=constraint, probabilities=probabilities)
        torch.cuda.synchronize()
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": 0, "drafts": False}
        keep, stop = prefix_limit(token_limit, [first], [])
        if not keep:
            return stats
        if (on_tokens is not None and on_tokens([first])) or stop or (stop_eos and first in self.eos) or max_tokens <= 1:
            return stats
        res = serial_decode(self.serial, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                            constraint=constraint, probabilities=probabilities, token_limit=token_limit)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def _decode(self, prompt: list[int], max_tokens: int, sampling, on_tokens, hit, constraint=None,
                stop_eos: bool = True, probabilities=None, points=None, token_limit=None) -> dict[str, Any]:
        import torch

        from .decode import entry_end, mtp_decode, prefill, serial_decode
        from .speculation import collective_prefix_limit
        from tensorfold.cuda.streams import prefix_limit

        if token_limit is not None:
            token_limit = collective_prefix_limit(token_limit, self.comm)

        t0 = time.perf_counter()
        self._start_from(hit)
        from tensorfold.cuda.markers import MIN_GAP

        end, cached = entry_end(prompt), len(hit[0]) if hit else 0
        markers = getattr(self, "points", None)
        if points is None:
            points = markers(prompt) if markers is not None else []
        stops = [p for p in points if cached + MIN_GAP <= p < end]
        def keep(p, snap, tail):
            self._remember(list(prompt[:p]), {"state": snap, "tail": tail})
        first = prefill(self.e, prompt, sampling, resume=hit[1] if hit else None, constraint=constraint,
                        probabilities=probabilities, keep_at=end, stops=stops, keep=keep)
        # the state one token before the prompt's end, so the same prompt or a next turn resumes from it
        self._remember(list(prompt[:end]), self.e.kept)
        torch.cuda.synchronize()
        stats: dict[str, Any] = {"prefill_s": round(time.perf_counter() - t0, 4), "cached": len(hit[0]) if hit else 0,
                                 "drafts": self.depth > 0}
        keep, stop = prefix_limit(token_limit, [first], [])
        if not keep:
            return stats
        if (on_tokens is not None and on_tokens([first])) or stop or (stop_eos and first in self.eos) or max_tokens <= 1:
            return stats
        if self.depth > 0:
            res = mtp_decode(self.e, first, max_tokens, sampling, depth=self.depth, confidence=self.confidence,
                             stop_eos=stop_eos, on_tokens=on_tokens, constraint=constraint, probabilities=probabilities,
                             token_limit=token_limit)
            stats.update(drafted=res.drafted, accepted=res.accepted, min_rows=min(res.widths, default=0))
        else:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens,
                                constraint=constraint, probabilities=probabilities, token_limit=token_limit)
        stats.update(decode_s=round(res.seconds, 4), rounds=res.rounds, decode_tps=round(res.tokens_per_second, 2))
        return stats

    def _begin_request(self, request):
        """Borrow this request's decoder until its caller retires.

        Concurrent acceptance is decided by the captured Scheduler's submit
        lock; a close racing before that submission may reject it, but cannot
        redirect it to serial execution. The serial decoder permits one call
        at a time because its mutable state belongs to that call.
        """

        owner = threading.get_ident()
        with self._lifecycle:
            self._prune_requests()
            if self._closing or self._closed:
                raise RuntimeError("Flash Next engine is closing or closed; new requests are rejected")
            scheduler = self.scheduler
            if scheduler is None and self._calls:
                raise RuntimeError("the serial Flash Next decoder already owns an active request")
            self._calls[request] = owner
            return scheduler

    def _prune_requests(self) -> None:
        """Retire cancelled journals whose native scope lock has exited.

            The caller holds the lifecycle lock. A with-owned native lock releases
            even when cancellation interrupts Python retirement. Its unlocked
            state proves that caller can enqueue no further request work; device
            and worker fences still precede resource cleanup.
            """
        if getattr(self, '_calls', None):
            for request in tuple(self._calls):
                if not request.locked():
                    if getattr(self, '_receiving', False) is request:
                        self._abort_comm = True
                        self._receiving = False
                    del self._calls[request]

    def _end_request(self, request, *, failed: bool=False) -> None:
        with self._lifecycle:
            if failed and request in self._calls:
                self._abort_comm = True
            if getattr(self, '_receiving', False) is request:
                self._receiving = False
            self._calls.pop(request, None)
            self._lifecycle.notify_all()

    def _retire_request(self, request, primary, *, retired=False) -> None:
        """Report cancellation while completing its idempotent owner retirement."""
        try:
            self._end_request(request, failed=primary is not None and not retired)
        except BaseException as cleanup:
            if isinstance(cleanup, KeyboardInterrupt):
                try:
                        self._end_request(request, failed=primary is not None and not retired)
                except BaseException as retry:
                    raise primary if primary is not None else cleanup from retry
            if primary is not None:
                raise primary from cleanup
            raise

    def _publish_retention_catalog(self, catalog) -> None:
        """Publish only the constructed, decoder-bound notification target."""
        catalog.check_health()
        with self._lifecycle:
            if (self.retention_catalog is not catalog or self._published_retention_catalog is not None
                    or self._closing or self._closed):
                raise RuntimeError("retention notifications require the current successfully bound catalog")
            self._published_retention_catalog = catalog

    def _withdraw_retention_catalog(self, catalog) -> None:
        """After request/claim drain, stop new callbacks and retire existing ones."""
        if catalog.has_retained:
            raise RuntimeError("protected retained claims must drain before notification withdrawal")
        with self._lifecycle:
            published = self._published_retention_catalog
            if published is not None and published is not catalog:
                raise RuntimeError("retention cleanup differs from the published catalog")
            leases = {lease: owner for lease, owner in self._catalog_callback_leases.items() if lease.locked()}
            if threading.get_ident() in leases.values():
                raise RuntimeError("a catalog notification cannot retire its own model owner")
            self._published_retention_catalog = None
        # No new callback can register after withdrawal. Each native one-shot
        # lease unlocks only after its callback exits, even if journal cleanup
        # failed. Wait outside the lifecycle lock so callbacks can retire.
        for lease in leases:
            with lease:
                pass
        with self._lifecycle:
            self._catalog_callback_leases = {}

    def _forward_retention_reference(self, committed, principal, sequence, *, generation, reason=None):
        lease = threading.Lock()

        def retire():
            with self._lifecycle:
                self._catalog_callback_leases.pop(lease, None)

        with lease:
            try:
                with self._lifecycle:
                    catalog = self._published_retention_catalog
                    if catalog is None:
                        return  # Logical records outlive a drained/unavailable computed profile.
                    self._catalog_callback_leases[lease] = threading.get_ident()
                catalog.check_health()
                if committed:
                    catalog.logical_reference_committed(principal, sequence, generation=generation)
                else:
                    catalog.logical_reference_unavailable(principal, sequence, generation=generation, reason=reason)
            except BaseException as primary:
                rollback(self, primary, retire)
            else:
                retire()

    def _retention_reference_committed(self, principal, sequence, *, generation):
        self._forward_retention_reference(True, principal, sequence, generation=generation)

    def _retention_reference_unavailable(self, principal, sequence, *, generation, reason):
        self._forward_retention_reference(False, principal, sequence, generation=generation, reason=reason)

    def bind_api_record_store(self, store) -> None:
        """Bind standard logical-record publication to this computed-state owner.

        Callbacks enqueue lifecycle events; the existing GPU actor applies them
        to every rank. The SQL record's lifetime is independent of model pages.
        """
        catalog = self._published_retention_catalog
        if catalog is None or self.retention_policy is None:
            raise ValueError("API computed retention requires the configured resident retention catalog")
        catalog.check_health()
        store.bind_reference_listeners(unavailable=self._retention_reference_unavailable,
                                       committed=self._retention_reference_committed)

    def close(self, *, abort: bool=False) -> None:
        """Reject new work, drain accepted calls and fence owned resources.

            Fixed weights and decoder buffers remain owned by this object until
            it is released. Plans and the expert pool close only after requests
            and the scheduler have quiesced. Failed cleanup retains those owners;
            another close retries it while new requests remain rejected.
            """
        if type(abort) is not bool:
            raise ValueError('typed communicator abort policy required')
        lifecycle = getattr(self, '_lifecycle', None)
        if lifecycle is None:
            self._close_resources()
            return
        lease = threading.Lock()
        with lease:
            try:
                with lifecycle:
                    scheduler = getattr(self, 'scheduler', None)
                    self._prune_requests()
                    calls = getattr(self, '_calls', None)
                    if calls is not None and threading.get_ident() in calls.values() or threading.current_thread() is getattr(scheduler, 'thread', None):
                        raise RuntimeError('Flash Next engine cannot close from an active request or its decoding worker')
                    self._closing = True
                    self._abort_comm = getattr(self, '_abort_comm', False) or abort
                    if getattr(self, '_receiving', False):
                        raise RuntimeError('rank1 follow still owns a store receive; ask rank0 to shutdown or depart, join follow, then retry close; resources retained')
                    lifecycle.notify_all()
                    while getattr(self, '_close_running', False):
                        if not self._close_running.locked():
                            self._close_running = False
                            break
                        lifecycle.wait(0.1)
                    if getattr(self, '_closed', False):
                        return
                    self._close_running = lease
                if scheduler is not None:
                    scheduler.close()
                    with lifecycle:
                        self.scheduler = None
                with lifecycle:
                    self._prune_requests()
                    while getattr(self, '_calls', None):
                        lifecycle.wait(0.1)
                        self._prune_requests()
                if not getattr(self, '_abort_comm', False) and getattr(self, 'comm', None) is not None:
                    self._send_shutdown()
                self._close_resources()
                with lifecycle:
                    self._closed = True
            finally:
                with lifecycle:
                    if getattr(self, '_close_running', None) is lease:
                        self._close_running = False
                    lifecycle.notify_all()

    def _close_resources(self, *, preserve_transport: bool = False) -> None:
        """After request drain, keep owners intact until their device fence."""
        catalog = getattr(self, "retention_catalog", None)
        if catalog is not None:
            self._withdraw_retention_catalog(catalog)
            policy = self.retention_policy
            catalog.close(timeout=None if policy is None else policy.lifetime_s)
            self.retention_catalog = None
        self._position_prepared = None
        seen_graphs = set()
        for decoder in (getattr(self, "multi", None), getattr(self, "e", None), getattr(self, "serial", None)):
            for name in ("graphs", "_failed_graphs"):
                graphs = getattr(decoder, name, None)
                if graphs is not None and id(graphs) not in seen_graphs:
                    graphs.close()
                    seen_graphs.add(id(graphs))
        preparation = getattr(self, "_owner_preparation", None)
        if preparation is not None:
            # Captured communicator borrowers and raw State leases retire
            # before either the communicator or failed-slot storage changes.
            preparation.close()
            decoder = getattr(self, "multi", None)
            for name in ("buf", "mbuf", "pbuf"):
                buffers = getattr(decoder, name, None)
                if buffers is not None:
                    buffers.owner_graphs = None
                    buffers.history_owner = None
            weights = getattr(self, "w", None)
            if weights is not None:
                weights.meta.pop("owner_workspaces", None)
            self._owner_preparation = None
        startup_origin = getattr(self, "_startup_store_origin", None)
        if startup_origin is not None and startup_origin.adopted_by is None:
            startup_origin.retire()  # An untransferred metadata-stage listener has no native communicator borrower.
        self._startup_store = self._startup_store_origin = None
        weights = getattr(self, "w", None)
        if weights is not None:
            from .input_preparation import KEY

            context = weights.meta.get(KEY)
            if context is not None:
                context.close()  # Scheduler/graph drain precedes mailbox-slot retirement and transport reuse.
        comm = getattr(self, 'comm', None)
        if comm is not None and not preserve_transport:
            comm.close(abort=getattr(self, '_abort_comm', False))
            self.comm = None
        session = getattr(self, "expert_session", None)
        if session is not None:
            session.close(abort=getattr(self, "_abort_comm", False))
            self.expert_session = None
        weights = getattr(self, 'w', None)
        if weights is not None:
            import torch
            torch.cuda.synchronize(weights.device)
        vision = getattr(self, "vision", None)
        if getattr(vision, "is_media_runtime", False):
            vision.close()
            self.vision = None
        numeric = getattr(self, "_media_numeric_scope", None)
        if numeric is not None:
            # The request/scheduler drain and this device fence precede all
            # process-global restoration, including partial startup failures.
            def retire_media_math():
                import torch

                torch.cuda.synchronize(None if weights is None else weights.device)
            numeric.close(retire=retire_media_math)
            self._media_numeric_scope = None
        multi = getattr(self, "multi", None)
        if multi is not None:
            multi.recover_failed_slots()
        history = getattr(self, "_indexer_history_session", None)
        if history is not None:
            # Graphs/requests are retired and failed-slot recovery is complete;
            # full host archives outlive every accepted D2H transfer and reset.
            if weights is not None:
                torch.cuda.synchronize(weights.device)
            history.close(fenced=True)
            self._indexer_history_session = None
        dspark = getattr(weights, "dspark", None)
        if dspark is not None:
            dspark.close()
        dflash = getattr(weights, "dflash2", None)
        if dflash is not None:
            # Requests/native readers are retired and the device fence above
            # completed. Discard unresolved failed-frame feature leases before
            # releasing each live window, then the draft weight owner.
            dflash.retire_after_fence(fenced=True)
            dflash.close()
        seen = set()
        for decoder in (getattr(self, 'multi', None), getattr(self, 'e', None), getattr(self, 'serial', None)):
            for name in ('buf', 'mbuf'):
                buffer = getattr(decoder, name, None)
                plans = getattr(buffer, 'hc_plans', None)
                if plans is not None and id(plans) not in seen:
                    plans.close()
                    seen.add(id(plans))
        cache = None if weights is None else weights.meta.get('expert_cache')
        if cache is not None:
            cache.close()
        tables = None if weights is None else weights.meta.get('ple_tables')
        if tables is not None:
            tables.close(lambda: torch.cuda.synchronize(weights.device))
        prepared_workspace = getattr(self, "_sm70_weight_workspace", None)
        if prepared_workspace is not None:
            import torch

            torch.cuda.synchronize(self._sm70_weight_device)
            self._sm70_weight_workspace = None
        self._sm70_import_plan = self._sm70_weight_plan = None
        self.__dict__.pop("_prepared_family_plan", None)

    def generate(self, prompt: list[int], max_tokens: int, sampling,
                 on_tokens: Callable[[list[int]], bool | None], draft: bool = True, constraint=None,
                 stop_eos: bool = True, background: bool = False, probabilities=None, *, vision=None,
                 progress=None, token_limit=None) -> dict[str, Any]:
        """``draft=False``: one token a round, no MTP drafts; ``background``: last, yielding lanes to waiting ones;
        ``progress``: the prompt's prefill progress under ``--parallel`` (``Scheduler.submit``), else unused."""

        catalog = getattr(self, "retention_catalog", None)
        if constraint is not None and catalog is not None:
            from .retained_host import RetentionIntent
            from .retention_resources import retained_constraint_reason

            intent = RetentionIntent.current()
            reason = retained_constraint_reason(catalog, logical_retained=intent is not None and intent.target is not None)
            if reason is not None:
                raise ValueError(reason)
        request = threading.Lock()
        error_retired = threading.Event() if getattr(self, "serving_profiles", None) is not None else None
        with request:
            primary = None
            try:
                scheduler = self._begin_request(request)
                max_tokens = self._limit(prompt, max_tokens)
                if token_limit is not None and not callable(token_limit):
                    raise ValueError("token_limit must be a pure callable returning a prefix/stop decision")
                if probabilities is not None and not self.supports_logprobs:
                    raise ValueError("logprobs are supported on one GPU only")
                if probabilities is not None and constraint is not None:
                    raise ValueError("logprobs do not support structured output")
                if vision is not None and self.vision is None:
                    raise ValueError("image inputs require starting this server with --vision")
                if vision is not None and background:
                    raise ValueError("image requests cannot yield a background lane")
                if scheduler is not None:
                    grammar = {} if constraint is None else {"constraint": constraint}
                    return scheduler.submit(list(prompt), max_tokens, sampling, draft, on_tokens, stop_eos=stop_eos,
                                            **grammar, **({"background": True} if background else {}),
                                            probabilities=probabilities,
                                            **({"token_limit": token_limit} if token_limit is not None else {}),
                                            **({"vision": vision} if vision is not None else {}),
                                            **({"profile_epoch": self.serving_profiles.gate.current().epoch, "_error_retired": error_retired}
                                        if getattr(self, "serving_profiles", None) is not None else {}),
                                     **({"progress": progress} if progress is not None else {}))
                reuse = draft
                if draft:
                    from .prefix_policy import serial_reuse_allowed

                    reuse = serial_reuse_allowed(self.w, sampling, draft=draft,
                                                  position=len(prompt), kv_pair=self.kv_pair, vision=vision)
                hit = self._resume(prompt) if reuse else None
                points = None
                if self.tp > 1:                      # every peer decodes the leader's exact request
                    prompt, max_tokens, sampling, draft, _, _, stop_eos, points, _ = self._share(
                        prompt, max_tokens, sampling, draft, len(hit[0]) if hit else 0, constraint, stop_eos,
                        token_limit is not None)
                    self.served += 1
                    emit = on_tokens
                    on_tokens = lambda new: (emit(new), False)[1]       # noqa: E731  peers retain collective ordering
                if not draft:
                    return self._serial(prompt, max_tokens, sampling, on_tokens, constraint, stop_eos,
                                        probabilities=probabilities, token_limit=token_limit)
                return self._decode(prompt, max_tokens, sampling, on_tokens, hit, constraint, stop_eos,
                                    probabilities=probabilities, points=points, token_limit=token_limit)
            except BaseException as error:
                primary = error
                raise
            finally:
                self._retire_request(request, primary, retired=error_retired is not None and error_retired.is_set())

    def score_labels(self, prompt_ids, label_ids) -> tuple[list[float], float]:
        """Score caller-prepared token sequences through a fresh, journaled one-token generation."""

        import math
        import operator

        from tensorfold.engine.exact_sampling import Sampling
        from tensorfold.engine.probabilities import LabelProbabilities

        if not self.supports_logprobs:
            raise ValueError("decision labels are scored on one GPU only")
        def token_ids(values, name):
            result = []
            for token in values:
                if isinstance(token, bool):
                    raise ValueError(f"{name} requires integer token IDs")
                try:
                    value = operator.index(token)
                except TypeError as error:
                    raise ValueError(f"{name} requires integer token IDs") from error
                if not 0 <= value < self.w.cfg.vocab:
                    raise ValueError(f"{name} is outside the vocabulary")
                result.append(value)
            if not result:
                raise ValueError(f"empty {name}")
            return result

        prompt = token_ids(prompt_ids, "prompt")
        labels = token_ids(label_ids, "decision label")
        self._limit(prompt, 1)
        probe = LabelProbabilities(labels, start=len(prompt))
        self.generate(prompt, 1, Sampling(seed=0, temperature=0.0), lambda new: None, draft=False,
                      probabilities=probe)
        if probe.label_logits is None or probe.logsumexp is None:
            raise ValueError("the prompt's last position was not scored")
        if not math.isfinite(probe.logsumexp) or not all(math.isfinite(value) for value in probe.label_logits):
            raise ValueError("label scoring produced a non-finite logit")
        return probe.label_logits, probe.logsumexp

    def follow(self) -> None:
        """Peer rank: decode every request rank 0 serves, until rank 0 stops."""
        with self._lifecycle:
            if self._closing or self._closed:
                raise RuntimeError('Flash Next engine is closing or closed; new requests are rejected')
        if getattr(self, "tp_decoder", None) is not None:
            self._follow_concurrent()
            return
        while True:
            operation = threading.Lock()
            with operation:
                primary = None
                try:
                    with self._lifecycle:
                        if self._closing:
                            return
                        self._begin_request(operation)
                        self._receiving = operation
                    try:
                        request = self._receive()
                    finally:
                        with self._lifecycle:
                            if self._receiving is operation:
                                self._receiving = False
                            self._lifecycle.notify_all()
                    if request is None:
                        return
                    prompt, max_tokens, sampling, draft, cached, packed, stop_eos, points, limited = request
                    constraint = None
                    if packed:
                        from tensorfold.engine import grammar
                        constraint = grammar.compiler(self, self.model_dir, self.eos).follow(packed)
                    self.served += 1
                    hit = None
                    if draft and cached:
                        hit = next(((ids, snap) for ids, snap in self.cache if len(ids) == cached and prompt[:cached] == ids), None)
                        if hit is None:
                            raise RuntimeError(f'rank {self.rank} has no kept state for the {cached} tokens rank 0 resumes from')
                    try:
                        if draft:
                            self._decode(prompt, max_tokens, sampling, None, hit, constraint, stop_eos, points=points,
                                         token_limit=True if limited else None)
                        else:
                            self._serial(prompt, max_tokens, sampling, None, constraint, stop_eos,
                                         token_limit=True if limited else None)
                    except ValueError as exc:
                        print(f'[tensorfold] request {self.served} failed on tensor-parallel rank {self.rank}: {exc}', flush=True)
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    self._retire_request(operation, primary)

    def _follow_concurrent(self) -> None:
        """Mirror rank zero's Scheduler operations under the existing receive lifetime lease."""
        while True:
            operation = threading.Lock()
            with operation:
                primary = None
                try:
                    with self._lifecycle:
                        if self._closing:
                            return
                        self._begin_request(operation)
                        self._receiving = operation
                    try:
                        body = self.tp_decoder.receive()
                    finally:
                        with self._lifecycle:
                            if self._receiving is operation:
                                self._receiving = False
                            self._lifecycle.notify_all()
                    if body is None:
                        return
                    self.tp_decoder.apply(body)
                except BaseException as error:
                    primary = error
                    raise
                finally:
                    self._retire_request(operation, primary)
            body = None
