"""Named CUDA HTTP frontend phases, separate from the model/RPC authority.

These requested object/vector layouts do not authorize an endpoint or certify
allocator retention/RSS. The caller supplies actual route limits and keeps the
captured tokenizer/runtime immutable. Model and other caller owners stay paid.
"""

import sys


def parsed_text_utf8_bound(body_bytes):
    """UTF8 extent of a successfully extracted string from accepted JSON bytes.

    json.loads(bytes) also accepts UTF16/32. BMP UTF16 can expand from two
    source bytes to three UTF8 bytes; UTF8, surrogate pairs and UTF32 have
    smaller ratios. JSON escapes, syntax and BOMs only consume source budget.
    Lone-surrogate extraction failures remain a separate error phase.
    """
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _size

    size = _size(body_bytes, "JSON text source bytes", zero=True)
    return _size(3 * size // 2, "extracted JSON text UTF8 bytes", zero=True)


def live_tool_gate_input_bound(body_bytes, *, lead, tail):
    """A validated string definition's suffix plus the captured template marks."""
    from tensorfold.families.qwen4_exp.cuda.retention_resources import _total

    if type(lead) is not str or type(tail) is not str:
        raise ValueError("tool gate bounds require the actual captured string marks")
    marks = _total((len(lead.encode("utf-8")), len(tail.encode("utf-8"))))
    return {"schema": "tensorfold.live-tool-gate-input-v1",
            "utf8_bytes": _total((parsed_text_utf8_bound(body_bytes), marks)),
            "template_mark_bytes": marks,
            "precondition": "new input definition has a validated string name; historical output is separate",
            "scope": "source UTF8 extent; normalizer/codec/RPC/caller overlap remain separately required"}


def metadata_decode_payload(presentation, policy, *, body_bytes):
    """One /detokenize owner, whose ID budget is its wire rather than context.

    Parsing retires before decode. The validated pointer-list copy and parsed
    request survive decode/conversion and final response serialization. Native
    returned strings coexist with Python conversion only in the library's
    conversion phase; later JSON owns the Python string, not that Rust String.
    """
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        _append_list_payload, _json_one_shot_payload, _json_parse_payload,
        _json_tree_payload, _size, _total,
    )
    from tensorfold.families.qwen4_exp.cuda.tokenizer_payload import bytelevel_decode_payload

    size = _size(body_bytes, "metadata decode route body")
    native = bytelevel_decode_payload(presentation, encoded_id_bytes=size)
    parsed = _json_parse_payload(size, policy)
    input_graph = _json_tree_payload(size, policy)
    validated_ids = _append_list_payload(native["input_ids"])
    wrapper = sys.getsizeof({"prompt": None})
    steady = _total((input_graph, validated_ids, wrapper))
    decoded = native["decoded_utf8_bytes"]
    python_result = 80 + 5 * decoded
    # Default ASCII dumps adds exactly these fixed characters around content.
    wire = _size(native["json_escaped_content_bytes"] + len('{"prompt": ""}'),
                 "detokenize response wire")
    serialized = _json_one_shot_payload(wire, policy)
    phases = {
        "read_and_parse": parsed["total_bytes"],
        "native_decode": _total((steady, native["native_heap_requested_bytes"])),
        "result_to_python": _total((steady, native["result_to_python_payload_bytes"])),
        "result_to_json": _total((steady, python_result, serialized["total_bytes"])),
    }
    return {"schema": "tensorfold.http-metadata-decode-payload-v1", "body_bytes": size,
            "validated_id_pointer_list_bytes": validated_ids, "input_graph_bytes": input_graph,
            "response_wrapper_bytes": wrapper, "python_result_bytes": python_result,
            "response_wire_bytes": wire, "native_decode": native, "json_output": serialized,
            "phase_bytes": phases, "total_bytes": max(phases.values()),
            "separate_obligations": ("model/RPC and other active/retained callers",
                "HTTP framing and runtime-managed allocator/C-stack/RSS",
                "post-exhaustion retirement and reusable-process qualification")}
