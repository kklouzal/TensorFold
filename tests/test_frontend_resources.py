"""HTTP frontend composition keeps metadata independent of model context."""

import json
import sys

import pytest

pytest.importorskip("torch")

from tokenizers import Tokenizer, decoders, models

from tensorfold.families.qwen4_exp.cuda.retention_resources import (
    _json_payload_policy, tokenizer_presentation_bound,
)
from tensorfold.server.frontend_resources import (
    live_tool_gate_input_bound, metadata_decode_payload, parsed_text_utf8_bound,
)


@pytest.fixture(scope="module")
def frontend():
    tokenizer = Tokenizer(models.BPE({"a": 0, "b": 1, "Ã": 2, "©": 3, "Ġ": 4, "\\": 5}, []))
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer, tokenizer_presentation_bound(tokenizer), _json_payload_policy()


@pytest.mark.parametrize("ids", [[], [0], [2, 3, 4, 5], [0, 1] * 128])
def test_actual_decoded_reply_fits_wire_and_simultaneous_python_owners(frontend, ids):
    tokenizer, presentation, policy = frontend
    body = {"tokens": ids}
    encoded = json.dumps(body).encode()
    record = metadata_decode_payload(presentation, policy, body_bytes=len(encoded))
    copied = list(ids)
    reply = {"prompt": tokenizer.decode(copied, skip_special_tokens=False)}
    assert len(json.dumps(reply).encode()) <= record["response_wire_bytes"]
    assert sys.getsizeof(copied) <= record["validated_id_pointer_list_bytes"]
    assert sys.getsizeof(reply["prompt"]) <= record["python_result_bytes"]
    assert record["phase_bytes"]["result_to_json"] >= (
        record["input_graph_bytes"] + sys.getsizeof(copied) + sys.getsizeof(reply)
        + sys.getsizeof(reply["prompt"]) + record["json_output"]["total_bytes"])


def test_metadata_wire_budget_can_exceed_the_full_native_model_window(frontend):
    _tokenizer, presentation, policy = frontend
    record = metadata_decode_payload(presentation, policy, body_bytes=1 << 20)
    assert record["native_decode"]["input_ids"] > 262144
    assert record["native_decode"]["budget"] == {
        "encoded_id_bytes": 1 << 20, "decimal_id_weight_bytes": (1 << 20) + 1}
    assert "context_ids" not in record["native_decode"]["budget"]


def test_metadata_record_keeps_qualification_obligations_separate(frontend):
    _tokenizer, presentation, policy = frontend
    record = metadata_decode_payload(presentation, policy, body_bytes=1 << 20)
    assert record["total_bytes"] == max(record["phase_bytes"].values())
    assert any("model/RPC" in text for text in record["separate_obligations"])
    assert any("post-exhaustion" in text for text in record["separate_obligations"])


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "utf-16-le", "utf-16-be", "utf-32", "utf-32-le", "utf-32-be"])
def test_accepted_json_encodings_keep_a_real_utf8_expansion_budget(encoding):
    text = "\u0800" * 1024
    wire = json.dumps({"name": text}, ensure_ascii=False).encode(encoding)
    parsed = json.loads(wire)["name"]
    assert len(parsed.encode("utf-8")) <= parsed_text_utf8_bound(len(wire))
    if encoding.startswith("utf-16"):
        assert len(parsed.encode("utf-8")) > len(wire)


def test_live_gate_keeps_actual_template_marks_beyond_the_string_wire_budget():
    wire = json.dumps({"name": "\u0800" * 1024}, ensure_ascii=False).encode("utf-16")
    name = json.loads(wire)["name"]
    lead, tail = "\n<function=", ">"
    bound = live_tool_gate_input_bound(len(wire), lead=lead, tail=tail)
    for seen in (0, 1, 1023, 1024):
        assert len((lead + name[seen:] + tail).encode()) <= bound["utf8_bytes"]
    assert bound["template_mark_bytes"] == 12


def test_lone_surrogate_json_acceptance_does_not_claim_successful_native_extraction():
    parsed = json.loads(b'{"name":"\\ud800"}')["name"]
    with pytest.raises(UnicodeEncodeError):
        parsed.encode("utf-8")
