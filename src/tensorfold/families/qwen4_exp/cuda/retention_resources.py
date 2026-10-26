"""Global host commitments and additional per-rank retained-state allocations.

These are source payload bounds, not an allocator/RSS or V100 certificate.
The header and loaded frontend stages share the same immutable policy record.
The existing capacity owner supplies physical/cgroup admission; it is never
replaced with a process RSS threshold or a new administrator budget.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import struct
import sys

from .retention_policy import RetentionPolicy


def _size(value, name, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= sys.maxsize:
        raise ValueError(f"retention {name} must be a {'nonnegative' if zero else 'positive'} host-sized integer")
    return value


def _total(values):
    result = sum(values)
    return _size(result, "resource sum", zero=True)


@dataclass(frozen=True)
class HeaderResources:
    global_host_backing_bytes: int
    global_host_progress_bytes: int
    per_rank_gpu_overlap_bytes: int
    world: int

    @property
    def global_host_bytes(self):
        return _total((self.global_host_backing_bytes, self.global_host_progress_bytes))

    def record(self):
        return {**asdict(self), "schema": "tensorfold.retention-header-resources-v1",
                "phase": "steady-serving; max with separate loading scratch",
                "evidence": "source-candidate; allocator/device qualification pending"}


@dataclass(frozen=True)
class HostBaseline:
    baseline_host_bytes: int
    materialized_host_bytes: int

    def __post_init__(self):
        _size(self.baseline_host_bytes, "ordinary steady host obligation", zero=True)
        _size(self.materialized_host_bytes, "owned ordinary host materialization", zero=True)
        if self.materialized_host_bytes > self.baseline_host_bytes:
            raise ValueError("ordinary materialized credit exceeds its own declared baseline")


def retention_header_resources(policy, world):
    """Known reservations before model payload; global RAM is not rank-multiplied."""
    _size(world, "rank count")
    if policy is not None and type(policy) is not RetentionPolicy:
        raise ValueError("retention resources require the actual typed policy")
    if policy is None or policy.tier == "gpu":
        return HeaderResources(0, 0, 0, world)
    policy.validate_world(world)
    return HeaderResources(policy.ram_bytes, policy.progress_bytes, policy.overlap_bytes, world)


def retained_recovery_extra(existing_bytes, *, slots, keep, points):
    """Charge private idle versions beyond the existing global KEEP allowance.

    Existing family geometry charges (KEEP + slots*POINTS) versions. An idle
    protected generation may retain KEEP versions independently of the global
    optional prefix list. Reuse the authoritative category's exact width,
    including method-specific snapshots when called for that category.
    """
    for name, value in (("existing recovery bytes", existing_bytes), ("prefix count", keep), ("cut count", points)):
        _size(value, name, zero=True)
    _size(slots, "GPU slot count")
    versions = keep + slots * points
    if versions == 0:
        if existing_bytes:
            raise ValueError("recovery bytes have no admitted version count")
        return 0
    width, remainder = divmod(existing_bytes, versions)
    if remainder:
        raise ValueError("recovery category does not match the actual family version geometry")
    return _total((max(0, slots * keep - keep) * width,))


def retention_device_resources(breakdown, *, policy, slots, keep, points, mtp, hidden_width,
                               media_inputs_enabled=False):
    """Chosen rank geometry -> missing header delta and runtime pool limits.

    Both stages use the SAME actual core/DF snapshot categories. Media HC and
    coordinates reuse the maintained boundary contract, including its explicit
    credit for HC already present in ordinary protected snapshots.
    """
    if type(breakdown) is not dict or type(mtp) is not bool or type(media_inputs_enabled) is not bool:
        raise ValueError("retention resources require the actual chosen geometry and method capabilities")
    if policy is not None and type(policy) is not RetentionPolicy:
        raise ValueError("retained device pools require the actual typed policy or None")
    _size(slots, "GPU slot count")
    _size(keep, "prefix count", zero=True)
    _size(points, "cut count", zero=True)
    _size(hidden_width, "completed HC width", zero=not mtp)
    versions = keep + slots * points
    protected = keep if policy is None else slots * keep
    width = 0
    for name in ("retained_recovery", "dflash_recovery_snapshots"):
        value = _size(breakdown.get(name, 0), name, zero=True)
        if versions == 0:
            if value:
                raise ValueError("snapshot bytes have no source-header versions")
        else:
            unit, remainder = divmod(value, versions)
            if remainder:
                raise ValueError("snapshot category differs from actual version geometry")
            width += unit
    pending = live_extra = coordinate_extra = 0
    if mtp:
        from .media_mtp import boundary_resources

        actual = boundary_resources(hidden_width=hidden_width, slots=slots, keep=keep, points=points)
        pending = actual["live_hc_bytes"]
        if media_inputs_enabled:
            if breakdown.get("media_mtp_boundary", 0) != actual["additional_device_bytes"]:
                raise ValueError("retained media resources lack the actual existing boundary header credit")
            pending += actual["live_coordinate_bytes"]
            coordinate_width = actual["live_coordinate_bytes"] // slots
            width += coordinate_width
            coordinate_extra = max(0, protected - keep) * coordinate_width
        else:
            live_extra = pending
    return {"schema": "tensorfold.retention-device-resources-v1", "scope": "per-rank",
            "header_extra_device_bytes": _total((live_extra, coordinate_extra)),
            "text_live_hc_header_extra_bytes": live_extra,
            "retained_media_coordinate_extra_bytes": coordinate_extra,
            "protected_versions": protected,
            "limits": {"recovery_pool_bytes": _total((protected * width,)),
                       "pending_hc_bytes": pending, "snapshot_bytes": width},
            "evidence": "source-header allocation payload; allocator/device qualification pending"}


@dataclass(frozen=True)
class PresentationBound:
    tokenizer_identity: str
    vocab_size: int
    piece_utf8_bytes: int
    decoded_utf8_bytes_per_token: int
    decoded_utf8_fixed_bytes: int
    working_utf8_bytes_per_token: int
    working_utf8_fixed_bytes: int
    decoder: str


def tokenizer_presentation_bound(tokenizer):
    """Bind the ACTUAL loaded decoder and every lexical/added token.

    tokenizers 0.23.3 ByteLevel maps characters to bytes (or retains a token's
    UTF8 bytes) then uses UTF8 lossy conversion. Each input byte can produce at
    most one three-byte replacement character. Encode-only normalization and
    postprocessing cannot expand decode. Builtin decoder composition follows
    the exact 0.23.3 decode_chain sources; it does not implement decoding again.
    Affine bounds include empty input after Fuse and configured literal Replace
    insertions, rather than extrapolating measured single-token decode maxima.
    """
    import tokenizers
    from tensorfold.token_metadata import tokenizer_identity, tokenizer_vocab_size

    if type(tokenizer) is not tokenizers.Tokenizer or tokenizers.__version__ != "0.23.3":
        raise ValueError("tiered resource promises require the supported tokenizers 0.23.3 frontend")
    decoder = tokenizer.decoder
    state = None if decoder is None else json.loads(decoder.__getstate__())
    vocab = _size(tokenizer_vocab_size(tokenizer), "actual tokenizer vocabulary")
    maximum = 0
    for index in range(vocab):
        piece = tokenizer.id_to_token(index)
        if piece is not None:  # Prepared UNUSED matrix rows have no text.
            maximum = max(maximum, len(piece.encode("utf-8")))
    a, b, count_a, count_b = maximum, 0, 1, 0
    peak_a, peak_b = a, b
    pending = [state]
    while pending:
        operation = pending.pop()
        kind = None if operation is None else operation["type"]
        if kind == "Sequence":
            pending.extend(reversed(operation["decoders"]))
            continue
        if kind is None:  # Tokenizer's no-decoder fallback joins pieces with spaces.
            a, b = a + count_a, b + count_b
        elif kind in ("ByteLevel", "ByteFallback"):
            a, b = 3 * a, 3 * b
            if kind == "ByteLevel":
                count_a, count_b = 0, 1
        elif kind == "Fuse":
            count_a, count_b = 0, 1
        elif kind == "WordPiece":
            a, b = a + count_a, b + count_b
        elif kind == "BPEDecoder":
            if not operation["suffix"]:
                a, b = 2 * a + count_a, 2 * b + count_b
        elif kind == "CTC":
            if operation["cleanup"] and not operation["word_delimiter_token"]:
                a, b = 2 * a + count_a, 2 * b + count_b
        elif kind == "Replace":
            size = len(operation["content"].encode("utf-8"))
            # Nonoverlapping matches, including zero-width boundaries, are
            # bounded by bytes+piece count. Content is inserted literally.
            a, b = (1 + size) * a + size * count_a, (1 + size) * b + size * count_b
        elif kind not in ("Metaspace", "Strip"):
            raise ValueError("resource projection does not recognize this effective builtin decoder")
        _size(a, "decoder affine slope", zero=True)
        _size(b, "decoder affine intercept", zero=True)
        peak_a, peak_b = max(peak_a, a), max(peak_b, b)
    return PresentationBound(tokenizer_identity(tokenizer), vocab, maximum, a, b, peak_a, peak_b,
                             "tokenizers-0.23.3/builtin-affine-v1")


def _python_layout():
    # List over-allocation below is the supported CPython 3.12 list_resize
    # contract, not a universal Python/allocator assumption. These limits apply
    # only to the optional new tier promise; ordinary backends remain unchanged.
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise ValueError("tiered Python payload projection requires the selected CPython 3.12 runtime")
    return {"pointer": struct.calcsize("P"), "list": sys.getsizeof([]), "tuple": sys.getsizeof(()),
            "dict": sys.getsizeof({}), "dict_entry": sys.getsizeof({0: 0}),
            "bytes": sys.getsizeof(b""), "unicode": sys.getsizeof("\U0010ffff"),
            "bytearray": sys.getsizeof(bytearray()), "integer": sys.getsizeof(sys.maxsize)}


def continuation_payload_bytes(*, context, vocab_size, presentation, max_body_bytes, presentation_fixed=0):
    """Named future owned Python/NumPy payloads for ONE logical CPU owner.

    InputClaim independently pays the original full prompt and encoded replay.
    This pays copies/continuation/output through the full supported context,
    including thinking/gating subruns. Shared model/tokenizer tables are already
    loaded and excluded. Runtime allocator arenas/native libraries remain an
    explicit qualification obligation, not hidden inside these byte counts.
    """
    for name, value in (("context", context), ("vocabulary", vocab_size), ("body limit", max_body_bytes)):
        _size(value, name)
    _size(presentation, "presentation bytes per token", zero=True)
    _size(presentation_fixed, "presentation fixed bytes", zero=True)
    layout = _python_layout()
    integer = max(layout["integer"], sys.getsizeof(vocab_size - 1), sys.getsizeof(context))

    def sequence(n):
        return layout["list"] + (n + (n >> 3) + 6) * layout["pointer"] + n * integer

    # Actual Stream lists, CUDA App out, and StreamDecoder.ids. Temporary token
    # slices in decode/stop/copy-proposal work coexist with those owners.
    token_lists = ("stream.out", "stream.context", "stream.committed", "stream.drafts", "stream.stops", "stream.owed",
                   "app.out", "decoder.ids", "decoder.before_ids", "decoder.after_ids", "stop.tail_ids",
                   "copy.query", "copy.starts", "copy.best", "copy.context_slice")
    lists = len(token_lists) * sequence(context)
    # One mailbox node per token is the legal adverse chunking. Each owns a
    # one-item list, the (list,count) tuple and a deque pointer pair.
    from collections import deque

    mailbox = context * (sequence(1) + layout["tuple"] + 2 * layout["pointer"]
                         + integer + sys.getsizeof(deque((None,))))
    # CPU CopyIndex: at most context n-grams, each owns an eight-ID tuple and a
    # start-list; singleton dictionary storage dominates its CPython table per
    # entry. Bulk hashes/starts + sort/hash preparation are separate NumPy u64
    # arrays (six simultaneous arrays in the source update/probe expression).
    from tensorfold.families.qwen3_5.cuda.draft_tree import COPY_MIN_MATCH

    import numpy as np

    array = context * 8 + sys.getsizeof(np.empty(0, dtype=np.uint64))
    copies = context * (layout["dict_entry"] + layout["tuple"]
                        + COPY_MIN_MATCH * (layout["pointer"] + integer) + sequence(1)) + 6 * array
    # Incremental decode keeps before/after/accumulated text; stop/markup/reply
    # splitting can retain old and new presentation strings at the same call.
    # Count actual source string roles rather than assuming one byte/character.
    chars = context * presentation + presentation_fixed
    string = layout["unicode"] + 4 * chars + chars  # PEP393 payload plus possible UTF8 cache.
    text_roles = ("decoder.before", "decoder.after", "decoder.text", "decoder.append_slice", "decoder.new_text",
                  "final.before", "final.decoded", "final.slice", "final.joined", "app.raw", "app.reasoning",
                  "app.answer", "app.answer_raw", "app.delta_reasoning", "app.delta_content", "app.result_text",
                  "stop.decoded", "stop.visible", "markup.input", "markup.normalized", "markup.tail",
                  "markup.fragments", "markup.joined", "tool.argument_source")
    text = len(text_roles) * string
    # Parsed permissive tool outputs and JSON/SSE serialization may coexist
    # with the decoded reply. Use the same conservative node projection as
    # bounded input JSON, plus six-byte JSON escaping of each output codepoint.
    node = max(layout["dict_entry"], layout["unicode"] + 4, layout["integer"])
    reply_graph = chars * node
    reply_wire = 6 * chars + layout["bytes"]
    # Original stop/tool/request fields can remain reachable through an exact
    # suspended/completed callback owner after the HTTP parser returns.
    request_graph = max_body_bytes * node
    return {"continuation_lists": lists, "mailbox": mailbox, "copy_index": copies,
            "presentation_strings": text, "reply_graph_and_wire": reply_graph + reply_wire,
            "request_graph": request_graph, "input_prompt": 0,
            "token_list_roles": token_lists, "text_roles": text_roles,
            "project_payload_bytes": _total((lists, mailbox, copies, text, reply_graph, reply_wire, request_graph))}


def retained_constraint_reason(catalog, *, logical_retained):
    if type(logical_retained) is not bool:
        raise ValueError("retention eligibility must come from the trusted logical operation")
    policy = catalog.policy
    if policy is not None and (policy.tier == "ram" or logical_retained):
        return "structured output with retained/tiered execution has no admitted native matcher CPU-memory envelope"
    return None


def finalize_retention_resources(catalog, *, tokenizer, max_pending_requests,
                                max_waiting_input_bytes, max_http_connections, output_queue_tokens=None,
                                max_http_body_bytes=None, logical_api_limits=None):
    """Freeze actual endpoint obligations before handlers/accepted generation.

    Physical admission is a current source-candidate check, not an exclusive OS
    grant: concurrent unrelated consumers may still exhaust the host. Existing
    OS reserve/cgroup authorities apply without rank-multiplying global caps.
    """
    from tensorfold.cuda.capacity import admit_host_commitment
    from tensorfold.server.request_body import LIMIT

    catalog.check_health()
    baseline = catalog.host_resources
    if type(baseline) is not HostBaseline:
        raise ValueError("FIFO host admission requires the actual ordinary steady header baseline")
    for name, value in (("pending request limit", max_pending_requests),
                        ("waiting input bytes", max_waiting_input_bytes), ("HTTP connection limit", max_http_connections)):
        _size(value, name)
    if output_queue_tokens is not None:
        _size(output_queue_tokens, "output queue tokens")
    body_bytes = LIMIT if max_http_body_bytes is None else max(LIMIT, _size(max_http_body_bytes, "HTTP route body bound"))
    bound = tokenizer_presentation_bound(tokenizer)
    context = catalog.states[0].limit
    terms = continuation_payload_bytes(context=context, vocab_size=bound.vocab_size,
        presentation=bound.working_utf8_bytes_per_token, presentation_fixed=bound.working_utf8_fixed_bytes,
        max_body_bytes=body_bytes)
    if output_queue_tokens is not None:
        terms["mailbox"] = terms["mailbox"] * min(context, output_queue_tokens) // context
        terms["project_payload_bytes"] = _total(value for name, value in terms.items()
                                                if name not in ("project_payload_bytes", "input_prompt", "token_list_roles", "text_roles"))
    header = retention_header_resources(catalog.policy, catalog.world)
    sessions = 0 if catalog.policy is None else catalog.policy.max_sessions
    # Replicated Stream state exists on every rank; detok/HTTP are leader-only.
    per_owner = terms["project_payload_bytes"] + (catalog.world - 1) * (
        terms["continuation_lists"] + terms["copy_index"])
    model_cpu = (max_pending_requests + sessions) * per_owner
    # App.run's logical RequestLimit ends before the HTTP handler serializes
    # and writes its final response. Slow completed sockets and new model jobs
    # may therefore coexist; the real HTTP connection cap owns that tail.
    completed_http = max_http_connections * (terms["presentation_strings"] + terms["reply_graph_and_wire"]
        + terms["continuation_lists"] // len(terms["token_list_roles"]))
    cpu = model_cpu + completed_http
    layout = _python_layout()
    # A JSON graph has no more nodes than encoded bytes. The singleton table
    # bound covers a dictionary/list/scalar node, including its pointer cell.
    # This deliberately conservative wire bound is independent of token shape.
    node = max(layout["dict_entry"], layout["unicode"] + 4, layout["integer"])
    transport = max_http_connections * body_bytes * node
    # The backing cap already pays encoded manifest bytes. Pay additional
    # parsed graph/object headers for all admitted protected generations.
    backed = (0 if catalog.policy is None or catalog.policy.tier != "ram" else min(
        sessions + max_pending_requests + catalog.prefix_limit,
        catalog.policy.ram_bytes // catalog.minimum_future_bytes))
    # Published and replacement parsed generations may coexist; encoding,
    # fragments and the joined input are separate source-owned wire buffers.
    manifests = backed * catalog._manifest_bytes * (2 * node + 3)
    if logical_api_limits is None:
        import inspect
        from tensorfold.server.responses import Store

        # The existing lazy default cache still owns deep-copied standard
        # response/message graphs. Read its actual constructor contract, not a
        # second hard-coded 256MiB default or a throwaway Store instance.
        signature = inspect.signature(Store)
        cached = _size(signature.parameters["max_bytes"].default, "default Responses cache bytes")
        logical_api = (1 + max_http_connections) * cached * node
        logical_receipt = {"mode": "existing bounded in-memory Responses cache", "encoded_cache_bytes": cached}
    else:
        from tensorfold.server.api_records import ApiRecordLimits

        if type(logical_api_limits) is not ApiRecordLimits:
            raise ValueError("logical API host resources require the actual parsed ApiRecordLimits")
        # GET/list materializations and active response operations have
        # separately bounded ownership. Parse temporaries coexist with their
        # checked trees; encoded history is not the incoming body cap.
        logical_api = ((logical_api_limits.max_active_operations + max_http_connections)
                       * logical_api_limits.max_materialized_bytes
                       + max_http_connections * (logical_api_limits.max_history_bytes
                           + logical_api_limits.max_operation_bytes) * node)
        logical_receipt = {"mode": "configured standard SQL records", "limits": asdict(logical_api_limits),
                           "native_sqlite_allocator": "qualification pending; not an RSS certificate"}
    fixed = header.global_host_bytes
    total = _total((baseline.baseline_host_bytes, fixed, max_waiting_input_bytes, cpu, transport, manifests, logical_api))
    materialized = baseline.materialized_host_bytes
    if catalog._buffer is not None and catalog._buffer.is_pinned():
        materialized += catalog.world * catalog._buffer.nbytes
    # Finalize at startup, before any owned payload/accepted request exists.
    # Local RAM object counts are not a global observation and never provide a
    # rank-multiplied credit. Only equal fixed buffers have that proof here.
    if catalog.has_retained:
        raise ValueError("endpoint retention resources must finalize before request admission")
    record = {"schema": "tensorfold.retention-host-envelope-v1", "context_tokens": context,
        "world": catalog.world, "max_pending_requests": max_pending_requests,
        "max_http_connections": max_http_connections, "output_queue_tokens": output_queue_tokens,
        "max_http_body_bytes": body_bytes,
        "tokenizer": asdict(bound), "header": header.record(), "continuation": terms,
        "global_future_cpu_payload_bytes": cpu, "global_waiting_input_bytes": max_waiting_input_bytes,
        "global_model_cpu_projection_bytes": model_cpu, "global_completed_http_projection_bytes": completed_http,
        "global_transport_projection_bytes": transport, "global_manifest_projection_bytes": manifests,
        "ordinary_host_baseline": asdict(baseline), "global_logical_api_projection_bytes": logical_api,
        "logical_api": logical_receipt,
        "max_backed_generations": backed,
        "global_host_commitment_bytes": total, "materialized_owned_bytes": materialized,
        "allocator_evidence": "CPython object payload projection; allocator arenas/native RSS unqualified",
        "native_constraints": ("ordinary unretained matcher residency/allocator qualification pending" if catalog.policy is None
                               else "retained/tiered combination refused before compilation"),
        "evidence": catalog.evidence}
    identity = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    old = getattr(catalog, "host_resource_envelope", None)
    if old is not None and old["identity"] != identity:
        raise ValueError("the active retention endpoint resource envelope is immutable")
    authority = admit_host_commitment(total, materialized_owned_bytes=materialized)
    result = {**record, "identity": identity, "authority": authority}
    catalog.host_resource_envelope = result
    return result
