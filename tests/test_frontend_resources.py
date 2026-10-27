"""HTTP frontend composition keeps metadata independent of model context."""

import json
import sys

import pytest

pytest.importorskip("torch")

from tokenizers import Tokenizer, decoders, models

from tensorfold.families.qwen4_exp.cuda.retention_resources import (
    _json_payload_policy, tokenizer_presentation_bound,
)
from tensorfold.server.frontend_resources import metadata_decode_payload


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
