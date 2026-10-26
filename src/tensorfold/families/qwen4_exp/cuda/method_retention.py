"""Method payload at an explicitly quiescent, byte-preserving retention boundary.

The common catalog owns source sealing, host request/replay dependencies,
transfer bounds and all-rank publication. Core State inventory owns target/MTP
KV, indexer/proposal bases and protected core versions. This adapter borrows
only actual method views. It never archives a producer lease or GPU proof.
"""

from dataclasses import dataclass
import math

import torch

from tensorfold.cuda.sampling import validate_policy
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import ACCEPT, PROPOSAL, RESIDUAL
from .media_mtp import MTPBoundary, validate_snapshot as validate_media_snapshot
from .position_coordinates import copy_coordinates, issue_coordinates
from .retained_parts import PartSpec, TensorPart, part_specs

SCHEMA = "qwen4exp-method-retention-v1"
_PREFIX = "method."
_RNG = "position-token-splitmix64-midpoint52-v1"


@dataclass(frozen=True)
class MethodInventory:
    metadata: dict
    parts: tuple[TensorPart, ...]
    host: dict
    future_bytes: int


def _integer(value, name, maximum=(1 << 63) - 1):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"retained {name} must be a bounded native nonnegative integer")
    return value


def _hc_element_bytes(state):
    if state.execution_dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("retained HC requires the allocated family execution dtype")
    return 2


def _frontiers(state):
    position = _integer(state.pos, "target frontier", state.limit)
    mtp_len = _integer(state.mtp_len, "MTP frontier", state.limit)
    drafted = _integer(state.mtp_drafted, "MTP tentative rows", mtp_len)
    generation = _integer(state.slot_generation, "slot generation")
    return position, mtp_len, drafted, generation


def _sampling(value, position):
    validate_policy([position], value)
    if value is None:
        return None
    if type(value) is not Sampling:
        raise ValueError("retained draws require the actual keyed Sampling contract")
    return {name: getattr(value, name) for name in ("seed", "temperature", "top_k", "top_p", "min_p")}


def _draws(stream, state, position):
    if stream is None:
        return None
    for name in ("out", "context", "drafts"):
        values = getattr(stream, name)
        if type(values) is not list or any(type(value) is not int or value < 0 for value in values):
            raise ValueError("retained method tokens require canonical native IDs")
    bound = _integer(state.compact_proposal_rows, "proposal row bound")
    if len(stream.drafts) > bound:
        raise ValueError("retained already-drawn proposals exceed their admitted row bound")
    anchor = None
    if stream.out:
        if len(stream.context) not in (position, position + 1):
            raise ValueError("retained output and cached target frontiers have no supported pending-anchor relation")
        if len(stream.context) == position + 1:
            if stream.context[-1] != stream.out[-1]:
                raise ValueError("retained pending anchor differs from canonical output")
            anchor = stream.out[-1]
    return {"sampling": _sampling(stream.sampling, position),
            "target_position": position, "pending_anchor": anchor,
            "drawn_proposals": list(stream.drafts), "output_tokens": len(stream.out),
            "canonical_tokens": len(stream.context), "draft": stream.draft,
            "rng": {"algorithm": _RNG, "domains": [PROPOSAL, ACCEPT, RESIDUAL]}}


def _validate_draws(draws, state, position):
    if draws is None:
        return
    keys = {"sampling", "target_position", "pending_anchor", "drawn_proposals", "output_tokens",
            "canonical_tokens", "draft", "rng"}
    if (type(draws) is not dict or draws.keys() != keys or type(draws["draft"]) is not bool
            or draws["target_position"] != position or type(draws["target_position"]) is not int
            or draws["rng"] != {"algorithm": _RNG, "domains": [PROPOSAL, ACCEPT, RESIDUAL]}):
        raise ValueError("retained method draw schema or RNG identity is incompatible")
    _integer(draws["output_tokens"], "output token count")
    canonical = _integer(draws["canonical_tokens"], "canonical token count")
    if draws["output_tokens"] and canonical not in (position, position + 1):
        raise ValueError("retained output/cached frontiers have no supported anchor relation")
    proposals = draws["drawn_proposals"]
    if (type(proposals) is not list or len(proposals) > state.compact_proposal_rows
            or any(type(token) is not int or token < 0 for token in proposals)):
        raise ValueError("retained method proposal IDs exceed their admitted shape")
    anchor = draws["pending_anchor"]
    if (anchor is not None and (type(anchor) is not int or anchor < 0 or canonical != position + 1)
            or draws["output_tokens"] and (anchor is not None) != (canonical == position + 1)):
        raise ValueError("retained method pending anchor differs from its cached frontier")
    sampling = draws["sampling"]
    if sampling is not None:
        if (type(sampling) is not dict or sampling.keys() != {"seed", "temperature", "top_k", "top_p", "min_p"}
                or type(sampling["seed"]) is not int or type(sampling["top_k"]) is not int or sampling["top_k"] < 0
                or any(type(sampling[name]) not in (int, float) for name in ("temperature", "top_p", "min_p"))):
            raise ValueError("retained sampling fields require their exact native scalar types")
        validate_policy([position], Sampling(**sampling))


def _segments(end, available, capacity):
    first = end - available
    offset = first % capacity
    rows = min(available, capacity - offset)
    result = [(offset, rows, first)] if rows else []
    if available > rows:
        result.append((0, available - rows, first + rows))
    return result


def _dflash_metadata(state):
    window = getattr(state, "dflash2", None)
    if window is None:
        return None
    window.check()  # unresolved FeatureFrame ownership is never serialized
    if type(window.enabled) is not bool:
        raise ValueError("retained DFlash participation must be an explicit boolean")
    if window.generation != state.slot_generation:
        raise ValueError("retained DFlash window belongs to another slot generation")
    end = _integer(window.end, "DFlash frontier", window.context)
    available = _integer(window.available, "DFlash valid rows", window.capacity)
    if available != min(window.capacity, end) or window.enabled and end != state.pos:
        raise ValueError("retained DFlash conditioning does not cover its required committed frontier")
    return {"config": window.config.identity, "profile": window.profile_identity,
            "context": window.context, "capacity": window.capacity,
            "shape": list(window.kv.shape), "end": end, "available": available,
            "enabled": window.enabled, "generation": window.generation}


def _boundary_metadata(state, pending_hc):
    boundary = getattr(state, "mtp_boundary", None)
    if boundary is not None:
        if type(boundary) is not MTPBoundary or boundary.generation != state.slot_generation:
            raise ValueError("retained deferred HC belongs to another slot generation")
        if (boundary.row != state.pos - 1 or state.media_mtp_position is None
                or getattr(state, "media_inputs_enabled", False) is not True
                or boundary.position.identity != state.media_mtp_position.identity
                or boundary.position.epoch != state.media_mtp_position.epoch
                or boundary.coordinates.dtype != torch.int64 or boundary.coordinates.shape != (1, 3)
                or boundary.coordinates.device != boundary.hidden.device):
            raise ValueError("retained deferred HC coordinates differ from the current row/position policy")
        if pending_hc is not None and pending_hc is not boundary.hidden:
            raise ValueError("retained method has two distinct deferred HC owners")
        boundary.coordinate_record.proof.validate(boundary.coordinates, boundary.position,
            profile_identity=state.media_profile_identity, generation=state.slot_generation,
            media_identity=state.media_identity, row_start=boundary.row, row_count=1)
        pending_hc = boundary.hidden
    if pending_hc is None:
        return None, None
    if boundary is None and (state.media_profile_identity is not None or getattr(state, "image_positions", None) is not None):
        raise ValueError("media deferred HC requires its actual issued coordinate rows")
    if (not isinstance(pending_hc, torch.Tensor) or pending_hc.shape != (1, state.mtp_hidden_width)
            or pending_hc.dtype != state.execution_dtype or pending_hc.device != state.pos_dev.device
            or pending_hc.requires_grad or not pending_hc.is_contiguous()
            or state.mtp_hidden_width <= 0 or state.pos <= 0 or state.mtp_len - state.mtp_drafted != state.pos - 1):
        raise ValueError("retained deferred HC must be the actual bounded row at the MTP/target frontier")
    metadata = {"row": state.pos - 1, "kind": "text-hc" if boundary is None else "media-hc",
                "profile": state.media_profile_identity, "media": state.media_identity,
                "position": None if boundary is None else boundary.position.identity,
                "epoch": None if boundary is None else boundary.position.epoch}
    return metadata, pending_hc


def _live_bytes(state):
    width = _integer(state.mtp_hidden_width, "MTP HC width")
    window = getattr(state, "dflash2", None)
    live = (0 if window is None else window.kv.numel() * window.kv.element_size())
    if width:
        live += width * _hc_element_bytes(state)
        if getattr(state, "media_inputs_enabled", False) is True:
            live += 3 * 8
    return live


def _recovery_versions(state):
    if not _live_bytes(state):
        return 0
    versions = getattr(state, "retention_method_recovery_versions", None)
    return _integer(versions, "admitted method recovery-version bound")


def _future_bytes(state):
    live = _live_bytes(state)
    if not live:
        return 0
    versions = _recovery_versions(state)
    if versions >= ((1 << 63) - 1) // live:
        raise ValueError("retained future method bytes exceed signed64 resource accounting")
    return live * (1 + versions)


def _parts(state, metadata, *, hidden=None, coordinates=None):
    parts = []
    df = metadata["dflash"]
    if df is not None:
        window = state.dflash2
        for side in range(window.kv.shape[0]):
            for layer in range(window.kv.shape[1]):
                for piece, (offset, rows, first) in enumerate(_segments(df["end"], df["available"], df["capacity"])):
                    parts.append(TensorPart(f"{_PREFIX}dflash.{side}.{layer}.{piece}",
                        window.kv[side, layer, offset:offset + rows],
                        semantic=("dflash-logical-rows", side, layer, first, rows)))
    boundary = metadata["boundary"]
    if boundary is not None:
        parts.append(TensorPart(_PREFIX + "mtp.hc", hidden, semantic=("deferred-hc", boundary["row"])))
        if boundary["kind"] == "media-hc":
            parts.append(TensorPart(_PREFIX + "mtp.coordinates", coordinates,
                semantic=("deferred-coordinate-rows", boundary["row"], boundary["position"], boundary["epoch"])))
    return tuple(parts)


def _recovery_parts(state, recoveries):
    if type(recoveries) is not tuple or len(recoveries) > _recovery_versions(state):
        raise ValueError("protected method versions exceed their admitted recovery commitment")
    records, parts = [], []
    for index, entry in enumerate(recoveries):
        tail = None
        if type(entry) is tuple and len(entry) == 2:
            entry, tail = entry
        if type(entry) is not dict:
            raise ValueError("protected method recovery requires an actual State snapshot/cut dictionary")
        position = _integer(entry.get("pos"), "protected target frontier", state.limit)
        mtp_len = _integer(entry.get("mtp_len"), "protected MTP frontier", state.limit)
        # The actual State.snapshot/cut contract is committed: mtp_len already
        # excludes chained drafts. Its protected recovery version has no live
        # proposal rows, unlike the current State inventory above.
        if _integer(entry.get("mtp_drafted", 0), "protected MTP tentative rows") != 0:
            raise ValueError("protected State snapshots must describe their committed MTP cut")
        df_payload, df = entry.get("dflash2"), None
        window = getattr(state, "dflash2", None)
        if (df_payload is None) != (window is None):
            raise ValueError("protected snapshot omits or changes its DFlash method payload")
        schema = None
        generation = state.slot_generation
        if df_payload is not None:
            if type(df_payload) is not dict:
                raise ValueError("protected DFlash snapshot must be its typed payload")
            schema = df_payload.get("schema")
            if schema == "dflash2-f16-window-v1":
                if df_payload.keys() != {"schema", "config", "profile", "generation", "end", "available", "kv"}:
                    raise ValueError("protected DFlash snapshot contains unknown method state")
                window.validate_snapshot(df_payload, position)
                end, available = df_payload["end"], df_payload["available"]
                enabled = True
            elif schema == "dflash2-inactive-v1":
                if (df_payload.keys() != {"schema", "config", "profile", "generation", "target_end"}
                        or df_payload["target_end"] != position):
                    raise ValueError("inactive protected DFlash snapshot has an invalid target cut")
                end = available = 0  # the declared inactive snapshot owns no conditioning rows
                enabled = False
            else:
                raise ValueError("unknown protected DFlash snapshot schema")
            generation = _integer(df_payload.get("generation"), "protected DFlash generation")
            df = {"config": df_payload.get("config"), "profile": df_payload.get("profile"),
                  "context": window.context, "capacity": window.capacity, "shape": list(window.kv.shape),
                  "end": end, "available": available, "enabled": enabled, "generation": generation}
        media = entry.get("media_state")
        hidden = coordinates = boundary = None
        delta = 0
        if media is not None:
            validate_media_snapshot(state, media, position, mtp_len)
            generation = media["generation"]
            delta = media["rope_delta"]
            actual = media["mtp_boundary"]
            if actual is not None:
                if tail is not None and tail is not actual["hidden"]:
                    raise ValueError("protected media HC has two distinct payload owners")
                hidden, coordinates = actual["hidden"], actual["coordinates"].tensor
                boundary = {"row": actual["row"], "kind": "media-hc", "profile": media["profile"],
                            "media": media["media"], "position": state.media_mtp_position.identity,
                            "epoch": state.media_mtp_position.epoch}
            elif tail is not None:
                raise ValueError("protected resolved media snapshot has an unexpected deferred HC")
            media = {name: media[name] for name in ("schema", "profile", "media", "generation", "rope_delta", "mtp_enabled")}
        elif state.media_profile_identity is not None:
            raise ValueError("protected media snapshot omitted its continuation and coordinate identity")
        elif tail is not None:
            if (not isinstance(tail, torch.Tensor) or tail.shape != (1, state.mtp_hidden_width)
                    or tail.dtype != state.execution_dtype or tail.device != state.pos_dev.device
                    or position <= 0 or mtp_len != position - 1
                    or getattr(state, "image_positions", None) is not None):
                raise ValueError("protected text HC differs from its actual committed cut")
            hidden = tail
            boundary = {"row": position - 1, "kind": "text-hc", "profile": None,
                        "media": None, "position": None, "epoch": None}
        child = {"schema": SCHEMA, "target": position, "mtp_len": mtp_len, "mtp_drafted": 0,
                 "generation": generation, "boundary": boundary, "dflash": df, "draws": None,
                 "rope_delta": delta, "recovery_versions": _recovery_versions(state), "recoveries": []}
        _validate_metadata(state, child, stream=None, continuation="same-request")
        record = {"method": child, "dflash_schema": schema, "media": media}
        records.append(record)
        prefix = f"{_PREFIX}recovery.{index}."
        if schema == "dflash2-f16-window-v1":
            parts.append(TensorPart(prefix + "dflash.kv", df_payload["kv"], "recovery",
                                    ("protected-dflash-logical-window", index, position)))
        if boundary is not None:
            parts.append(TensorPart(prefix + "mtp.hc", hidden, "recovery", ("protected-hc", index, boundary["row"])))
            if coordinates is not None:
                parts.append(TensorPart(prefix + "mtp.coordinates", coordinates, "recovery",
                    ("protected-coordinates", index, boundary["row"], boundary["position"], boundary["epoch"])))
    return records, tuple(parts)


def method_inventory(state, *, stream=None, pending_hc=None, recoveries=()):
    """Borrow actual method parts after the owning coordinator retires issued work."""
    position, mtp_len, drafted, generation = _frontiers(state)
    if getattr(state, "media_input_span", None) is not None:
        raise ValueError("retained method boundary still owns a live next-input producer span")
    boundary, hidden = _boundary_metadata(state, pending_hc)
    if getattr(state, "dspark_runtime", None) is not None:
        raise ValueError("retention requires the selected method's complete adapter; DSpark is not described here")
    participating = (getattr(state, "dflash2", None) is None
                     and getattr(state, "media_mtp_enabled", False) and (stream is None or stream.draft))
    if participating and mtp_len - drafted != position and boundary is None:
        raise ValueError("retained MTP frontier lacks its actual deferred HC dependency")
    metadata = {"schema": SCHEMA, "target": position, "mtp_len": mtp_len, "mtp_drafted": drafted,
                "generation": generation, "boundary": boundary, "dflash": _dflash_metadata(state),
                "draws": _draws(stream, state, position), "rope_delta": state.rope_delta,
                "recovery_versions": _recovery_versions(state), "recoveries": []}
    coordinates = None if boundary is None or boundary["kind"] != "media-hc" else state.mtp_boundary.coordinates
    parts = _parts(state, metadata, hidden=hidden, coordinates=coordinates)
    metadata["recoveries"], recovery_parts = _recovery_parts(state, recoveries)
    parts += recovery_parts
    _validate_draws(metadata["draws"], state, position)
    part_specs(parts)
    return MethodInventory(metadata, parts, {}, _future_bytes(state))


def _expected_specs(state, metadata):
    specs = []
    df = metadata["dflash"]
    if df is not None:
        for side in range(state.dflash2.kv.shape[0]):
            for layer in range(state.dflash2.kv.shape[1]):
                for piece, (_offset, rows, first) in enumerate(_segments(df["end"], df["available"], df["capacity"])):
                    shape = (rows, *state.dflash2.kv.shape[3:])
                    specs.append(PartSpec(f"{_PREFIX}dflash.{side}.{layer}.{piece}", "method", "float16",
                        shape, math.prod(shape) * 2, ("dflash-logical-rows", side, layer, first, rows)))
    boundary = metadata["boundary"]
    if boundary is not None:
        dtype = str(state.execution_dtype).removeprefix("torch.")
        width = state.mtp_hidden_width
        size = _hc_element_bytes(state)
        specs.append(PartSpec(_PREFIX + "mtp.hc", "method", dtype, (1, width), width * size,
                              ("deferred-hc", boundary["row"])))
        if boundary["kind"] == "media-hc":
            specs.append(PartSpec(_PREFIX + "mtp.coordinates", "method", "int64", (1, 3), 24,
                ("deferred-coordinate-rows", boundary["row"], boundary["position"], boundary["epoch"])))
    return tuple(specs)


def _recovery_specs(state, record, index):
    metadata, schema = record["method"], record["dflash_schema"]
    prefix, specs = f"{_PREFIX}recovery.{index}.", []
    if schema == "dflash2-f16-window-v1":
        shape = (*state.dflash2.kv.shape[:2], metadata["dflash"]["available"], *state.dflash2.kv.shape[3:])
        specs.append(PartSpec(prefix + "dflash.kv", "recovery", "float16", shape, math.prod(shape) * 2,
                              ("protected-dflash-logical-window", index, metadata["target"])))
    boundary = metadata["boundary"]
    if boundary is not None:
        specs.append(PartSpec(prefix + "mtp.hc", "recovery", str(state.execution_dtype).removeprefix("torch."),
            (1, state.mtp_hidden_width), state.mtp_hidden_width * _hc_element_bytes(state),
            ("protected-hc", index, boundary["row"])))
        if boundary["kind"] == "media-hc":
            specs.append(PartSpec(prefix + "mtp.coordinates", "recovery", "int64", (1, 3), 24,
                ("protected-coordinates", index, boundary["row"], boundary["position"], boundary["epoch"])))
    return tuple(specs)


def _validate_metadata(state, metadata, *, stream, continuation):
    if continuation not in ("same-request", "new-turn"):
        raise ValueError("retained continuation must explicitly select same-request or new-turn")
    keys = {"schema", "target", "mtp_len", "mtp_drafted", "generation", "boundary", "dflash", "draws",
            "rope_delta", "recovery_versions", "recoveries"}
    if type(metadata) is not dict or metadata.keys() != keys or metadata["schema"] != SCHEMA:
        raise ValueError("retained method metadata schema is invalid")
    position = _integer(metadata["target"], "target frontier", state.limit)
    mtp_len = _integer(metadata["mtp_len"], "MTP frontier", state.limit)
    drafted = _integer(metadata["mtp_drafted"], "MTP tentative rows", mtp_len)
    _integer(metadata["generation"], "source generation")
    if (_integer(metadata["recovery_versions"], "source method recovery bound") != _recovery_versions(state)):
        raise ValueError("retained method recovery commitment differs from current admitted capacity")
    if type(metadata["recoveries"]) is not list or len(metadata["recoveries"]) > metadata["recovery_versions"]:
        raise ValueError("retained protected method version count exceeds its admitted commitment")
    if type(metadata["rope_delta"]) is not int or not -(1 << 31) <= metadata["rope_delta"] < 1 << 31:
        raise ValueError("retained method continuation delta is outside its representation")
    if getattr(state, "media_input_span", None) is not None:
        raise ValueError("method restore destination still owns a live producer span")
    if getattr(state, "dspark_runtime", None) is not None:
        raise ValueError("method restore cannot substitute an undescribed DSpark state")
    df = metadata["dflash"]
    window = getattr(state, "dflash2", None)
    if (df is None) != (window is None):
        raise ValueError("retained method changes DFlash participation/configuration")
    if df is not None:
        window.check()
        expected = {"config", "profile", "context", "capacity", "shape", "end", "available", "enabled", "generation"}
        if (type(df) is not dict or df.keys() != expected or df["config"] != window.config.identity
                or df["profile"] != window.profile_identity or df["context"] != window.context
                or df["capacity"] != window.capacity or df["shape"] != list(window.kv.shape)
                or type(df["enabled"]) is not bool
                or any(type(df[name]) is not int for name in ("context", "capacity", "generation"))
                or type(df["shape"]) is not list or any(type(value) is not int for value in df["shape"])):
            raise ValueError("retained DFlash profile or allocation geometry differs")
        end = _integer(df["end"], "DFlash frontier", window.context)
        if (_integer(df["available"], "DFlash valid rows", window.capacity) != min(window.capacity, end)
                or df["enabled"] and end != position or df["generation"] != metadata["generation"]):
            raise ValueError("retained DFlash validity/generation differs from its target")
        if continuation == "new-turn" and stream is not None:
            if type(stream.draft) is not bool or stream.draft and end != position:
                raise ValueError("new DFlash request lacks valid conditioning at its retained target frontier")
    boundary = metadata["boundary"]
    if boundary is not None:
        expected = {"row", "kind", "profile", "media", "position", "epoch"}
        if (type(boundary) is not dict or boundary.keys() != expected or type(boundary["row"]) is not int
                or boundary["row"] != position - 1 or mtp_len - drafted != position - 1
                or position == 0 or state.mtp_hidden_width <= 0 or boundary["kind"] not in ("text-hc", "media-hc")
                or boundary["profile"] != state.media_profile_identity or boundary["media"] != state.media_identity):
            raise ValueError("retained deferred HC has an incompatible frontier/profile/media identity")
        if boundary["kind"] == "media-hc":
            policy = state.media_mtp_position
            if (getattr(state, "media_inputs_enabled", False) is not True
                    or policy is None or type(boundary["position"]) is not str or type(boundary["epoch"]) is not int
                    or boundary["position"] != policy.identity or boundary["epoch"] != policy.epoch):
                raise ValueError("retained coordinate rows belong to another positional identity/epoch")
        elif boundary["position"] is not None or boundary["epoch"] is not None:
            raise ValueError("text HC cannot claim an old issued coordinate proof")
    participating = (window is None and getattr(state, "media_mtp_enabled", False)
                     and (stream is None or stream.draft))
    if participating and mtp_len - drafted != position and boundary is None:
        raise ValueError("retained MTP conditioning frontier lacks its actual deferred HC")
    _validate_draws(metadata["draws"], state, position)
    if continuation == "same-request" and metadata["draws"] != _draws(stream, state, position):
        raise ValueError("retained method draws/anchor differ from the preserved request owner")
    _future_bytes(state)


def validate_method_restore(state, metadata, specs, *, stream=None, continuation="same-request"):
    """Validate sealed catalog metadata/specs before tensor allocation or writes."""
    _validate_metadata(state, metadata, stream=stream, continuation=continuation)
    expected = _expected_specs(state, metadata)
    for index, record in enumerate(metadata["recoveries"]):
        if type(record) is not dict or record.keys() != {"method", "dflash_schema", "media"}:
            raise ValueError("protected method record schema is invalid")
        child = record["method"]
        _validate_metadata(state, child, stream=None, continuation="same-request")
        if child["recoveries"] or child["draws"] is not None or child["mtp_drafted"]:
            raise ValueError("protected method versions are committed nonnested cuts")
        df, schema = child["dflash"], record["dflash_schema"]
        if (df is None and schema is not None or df is not None and schema not in (
                "dflash2-f16-window-v1", "dflash2-inactive-v1")
                or df is not None and df["enabled"] != (schema == "dflash2-f16-window-v1")):
            raise ValueError("protected DFlash payload lost its active/inactive role")
        media = record["media"]
        if media is not None:
            if (type(media) is not dict or media.keys() != {
                    "schema", "profile", "media", "generation", "rope_delta", "mtp_enabled"}
                    or media["schema"] != "qwen4exp-media-state-v1" or media["profile"] != state.media_profile_identity
                    or media["media"] != state.media_identity or type(media["mtp_enabled"]) is not bool
                    or media["mtp_enabled"] != state.media_mtp_enabled
                    or type(media["generation"]) is not int or type(media["rope_delta"]) is not int
                    or media["generation"] != child["generation"] or media["rope_delta"] != child["rope_delta"]):
                raise ValueError("protected media continuation differs from its actual method cut")
        elif state.media_profile_identity is not None:
            raise ValueError("protected method version omitted its media continuation")
        expected += _recovery_specs(state, record, index)
    if type(specs) is not tuple or any(type(spec) is not PartSpec for spec in specs) or specs != expected:
        raise ValueError("retained method tensor descriptors are incomplete or incompatible")


def method_restore_resources(state, metadata):
    """Additional phase bytes beyond the full persistent/future method claim."""
    rows = sum(1 for value in [metadata, *(record["method"] for record in metadata["recoveries"])]
               if value["boundary"] is not None and value["boundary"]["kind"] == "media-hc")
    return {"host_bytes": rows * 3 * 8, "device_bytes": rows * 3 * 4}


def method_restore_targets(state, metadata, specs, *, stream=None, continuation="same-request"):
    validate_method_restore(state, metadata, specs, stream=stream, continuation=continuation)
    hidden = coordinates = None
    if metadata["boundary"] is not None:
        hidden = torch.empty((1, state.mtp_hidden_width), dtype=state.execution_dtype, device=state.pos_dev.device)
        if metadata["boundary"]["kind"] == "media-hc":
            coordinates = torch.empty((1, 3), dtype=torch.int64, device="cpu")
    parts = _parts(state, metadata, hidden=hidden, coordinates=coordinates)
    dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16, "int64": torch.int64}
    for index, record in enumerate(metadata["recoveries"]):
        for spec in _recovery_specs(state, record, index):
            device = "cpu" if spec.name.endswith(".coordinates") else state.pos_dev.device
            tensor = torch.empty(spec.shape, dtype=dtypes[spec.dtype], device=device)
            parts += (TensorPart(spec.name, tensor, spec.category, spec.semantic),)
    return parts


def _restore_boundary(state, boundary, hidden, raw):
    if boundary is None or boundary["kind"] != "media-hc":
        return None
    if raw.device.type != "cpu":
        raise ValueError("coordinate authority must be reissued from actual restored CPU rows")
    issued = issue_coordinates(raw, position=state.media_mtp_position,
        profile_identity=state.media_profile_identity, generation=state.slot_generation,
        media_identity=state.media_identity, row_start=boundary["row"], max_rows=1, device=state.pos_dev.device)
    issued = copy_coordinates(issued, source_position=state.media_mtp_position, position=state.media_mtp_position,
                              row_start=boundary["row"], row_count=1, dtype=torch.int64)
    return MTPBoundary(boundary["row"], hidden, issued, state.media_mtp_position, state.slot_generation)


def _restored_recoveries(state, metadata, by_name):
    restored = []
    for index, record in enumerate(metadata["recoveries"]):
        child, prefix = record["method"], f"{_PREFIX}recovery.{index}."
        hidden = by_name.get(prefix + "mtp.hc")
        boundary = _restore_boundary(state, child["boundary"], hidden, by_name.get(prefix + "mtp.coordinates"))
        payload = {"pos": child["target"], "mtp_len": child["mtp_len"], "mtp_drafted": 0, "pending_hc": hidden}
        df, schema = child["dflash"], record["dflash_schema"]
        if df is not None:
            value = {"schema": schema, "config": df["config"], "profile": df["profile"],
                     "generation": state.slot_generation}
            if schema == "dflash2-f16-window-v1":
                value.update(end=df["end"], available=df["available"], kv=by_name[prefix + "dflash.kv"])
            else:
                value["target_end"] = child["target"]
            payload["dflash2"] = value
        if record["media"] is not None:
            media = dict(record["media"], generation=state.slot_generation)
            media["mtp_boundary"] = (None if boundary is None else {
                "row": boundary.row, "hidden": hidden, "coordinates": boundary.coordinate_record})
            payload["media_state"] = media
        restored.append(payload)
    return tuple(restored)


def commit_method_restore(state, metadata, targets, *, stream=None, continuation="same-request"):
    """Publish only after checked byte copies; caller publishes all-rank readiness later."""
    validate_method_restore(state, metadata, part_specs(targets), stream=stream,
                            continuation=continuation)
    by_name = {part.name: part.tensor for part in targets}
    boundary = metadata["boundary"]
    hc = by_name.get(_PREFIX + "mtp.hc")
    restored = _restore_boundary(state, boundary, hc, by_name.get(_PREFIX + "mtp.coordinates"))
    recoveries = _restored_recoveries(state, metadata, by_name)
    df = metadata["dflash"]
    if df is not None:
        window = state.dflash2
        enabled = stream.draft if continuation == "new-turn" and stream is not None else df["enabled"]
        window.end, window.available, window.enabled = df["end"], df["available"], enabled
        window.generation = state.slot_generation
    state.mtp_boundary = restored
    state.set_rope_delta(metadata["rope_delta"])
    return {"pending_hc": hc, "mtp_len": metadata["mtp_len"], "mtp_drafted": metadata["mtp_drafted"],
            "draws": metadata["draws"] if continuation == "same-request" else None, "recoveries": recoveries}
