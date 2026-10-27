"""Named CUDA HTTP frontend phases, separate from the model/RPC authority.

These requested object/vector layouts do not authorize an endpoint or certify
allocator retention/RSS. The caller supplies actual route limits and keeps the
captured tokenizer/runtime immutable. Model and other caller owners stay paid.
"""

import sys


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
