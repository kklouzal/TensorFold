"""Selected ByteLevel decoder heap requests, separate from allocator/RSS.

The pinned tokenizers/core/PyO3 artifact uses Rust b940084d7 RawVec growth:
live capacity <= max(minimum, 2*n), old+new <= 3*n+minimum. Native phase
lifetimes follow tokenizers v0.23.3 byte_level.rs and tokenizer/mod.rs; only
the maximum phase is charged. Existing Python owners remain separate.
"""

import struct
import sys


_ARTIFACTS = (
    ("tokenizers-native", 11326992, "4b1458a379296762a034d3bc89299c0c35b4eefb0577acca922f481b262c65a1"),
    ("tokenizers-sbom", 151305, "6faa23324b2b53b4a8da9b451a675610cc45e1a23e8c5bb0740c080064e1ba0d"),
)


def _size(value, name):
    if type(value) is not int or not 0 <= value <= sys.maxsize:
        raise ValueError(f"ByteLevel {name} must be a nonnegative host-sized integer")
    return value


def bytelevel_payload_policy():
    """Bind the actually loaded artifact before selecting its capacity rules.

    The DSO digest pins compiled Rust1.99.0/b940084d7 and PyO3.29.2 code;
    the SBOM identifies Python0.23.3/core0.23.2. A version label alone does
    not authorize these equations. The caller keeps runtime files immutable.
    """
    import importlib
    from importlib.metadata import distribution
    from pathlib import Path

    from tensorfold.engine.model_identity import capture_model_identity
    from .execution_identity import _small

    if (sys.platform != "linux" or struct.calcsize("P") != 8
            or sys.implementation.name != "cpython" or sys.version_info[:3] != (3, 12, 3)):
        raise ValueError("ByteLevel native payload requires the selected Linux64 artifact")
    package = distribution("tokenizers")
    if package.version != "0.23.3":
        raise ValueError("ByteLevel native payload requires tokenizers0.23.3")
    native = Path(importlib.import_module("tokenizers.tokenizers").__file__).resolve(strict=True)
    sbom = Path(package.locate_file(
        "tokenizers-0.23.3.dist-info/sboms/tokenizers-python.cyclonedx.json")).resolve(strict=True)
    mapped = set()
    for line in _small("/proc/self/maps", 4 << 20).decode().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and native.name in fields[5]:
            if fields[5].endswith(" (deleted)"):
                raise ValueError("ByteLevel native payload cannot use a retired decoder artifact")
            mapped.add(Path(fields[5]).resolve(strict=True))
    if mapped != {native}:
        raise ValueError("ByteLevel native payload requires its actual mapped decoder artifact")
    paths = {"tokenizers-native": native, "tokenizers-sbom": sbom}
    captured = capture_model_identity(paths, runtime_identity={"policy": "bytelevel-native-payload-v1"},
        authorize=lambda _name, path: path.resolve(strict=True), max_files=2,
        max_file_bytes=max(row[1] for row in _ARTIFACTS), max_total_bytes=sum(row[1] for row in _ARTIFACTS))
    expected = {name: (size, digest) for name, size, digest in _ARTIFACTS}
    if any((item.size, item.sha256) != expected[item.logical_name] for item in captured.files):
        raise ValueError("ByteLevel native payload has no proof for the actual decoder/toolchain artifact")
    captured.verify_unchanged()
    return {"identity": captured.model_prefix,
        "files": tuple((item.logical_name, item.size, item.sha256) for item in captured.files),
        "rust": "1.99.0/b940084d7", "pyo3": "0.29.2", "tokenizers_core": "0.23.2",
        "source": "https://github.com/rust-lang/rust/blob/b940084d7/library/alloc/src/raw_vec/mod.rs",
        "scope": "requested native heap layouts; allocator usable bytes/retention/RSS and fixed stack separate"}


def _ratio_ceiling(value, ratio, name):
    if (type(ratio) is not tuple or len(ratio) != 2 or type(ratio[0]) is not int
            or type(ratio[1]) is not int or not 0 <= ratio[0] <= sys.maxsize
            or not 1 <= ratio[1] <= sys.maxsize):
        raise ValueError(f"ByteLevel {name} requires an exact nonnegative rational bound")
    return (value * ratio[0] + ratio[1] - 1) // ratio[1]


def bytelevel_decode_payload(bound, *, context=None, encoded_id_bytes=None):
    """Bound one synchronous decode, with exactly one source of ID limits.

    ``context`` bounds model-decoder IDs. HTTP detokenize has no model-context
    cap: its decimal-ID wire budget instead supplies n<=(R+1)//2 and the
    enumerated lexical/decoded/escaped-byte ratios. Inputs, existing Python
    outputs and wrapper fields are not credited against these native phases.
    Python conversion assumes the separately bound selected CPython layout.
    """
    if (context is None) == (encoded_id_bytes is None):
        raise ValueError("ByteLevel decode needs exactly one context-ID or encoded-ID-wire budget")
    if (bound.decoder != "tokenizers-0.23.3/bytelevel-concat-v2"
            or bound.decoded_utf8_fixed_bytes != 0 or bound.working_utf8_fixed_bytes != 0):
        raise ValueError("native payload proof requires the bound plain ByteLevel decoder")
    piece = _size(bound.piece_utf8_bytes, "lexical piece bytes")
    decoded = _size(bound.decoded_utf8_bytes_per_token, "decoded piece bytes")
    if context is not None:
        n = _size(context, "context ID count")
        lexical, output = n * piece, n * decoded
        escaped = 6 * output
        budget = {"context_ids": n}
    else:
        wire = _size(encoded_id_bytes, "encoded ID wire bytes")
        n = (wire + 1) // 2
        lexical = min(n * piece, _ratio_ceiling(wire + 1, bound.lexical_id_wire_ratio, "lexical ID-wire ratio"))
        output = min(n * decoded, _ratio_ceiling(wire + 1, bound.decoded_id_wire_ratio, "decoded ID-wire ratio"))
        escaped = min(6 * output, _ratio_ceiling(wire + 1, bound.escaped_id_wire_ratio, "escaped ID-wire ratio"))
        budget = {"encoded_id_bytes": wire, "decimal_id_weight_bytes": wire + 1}
    raw, one = min(lexical, output), min(piece, decoded, lexical, output)
    def capacity(count, minimum):
        return max(minimum, 2 * count) if count else 0
    def realloc(count, minimum):
        return 3 * count + minimum if count else 0
    phases = {
        "token_lookup_clone_and_header_growth": 4 * n + 24 * realloc(n, 4) + lexical,
        "piece_and_flat_bytes": 4 * n + 24 * capacity(n, 4) + lexical + capacity(one, 8) + realloc(raw, 8),
        "lossy_and_to_string": 4 * n + capacity(raw, 8) + realloc(output, 8) + output + 24,
        "decoder_join": 4 * n + 2 * output + 24,
    }
    conversion = output + 160 + 8 * output
    total = _size(max(phases.values()), "native phase payload")
    _size(conversion, "native result/Python conversion payload")
    _size(escaped + 2, "escaped JSON string payload")
    return {"schema": "tensorfold.bytelevel-native-decode-payload-v1", "budget": budget,
        "input_ids": n, "lexical_bytes": lexical, "raw_bytelevel_bytes": raw,
        "decoded_utf8_bytes": output, "phase_heap_requested_bytes": phases,
        "native_heap_requested_bytes": total, "result_to_python_payload_bytes": conversion,
        "json_escaped_content_bytes": escaped, "json_escaped_string_bytes": escaped + 2,
        "policy": bytelevel_payload_policy(),
        "evidence": "source-owned requested layouts; Python owners/allocator/RSS/fixed stack separate"}
