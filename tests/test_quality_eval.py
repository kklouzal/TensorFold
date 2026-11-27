import json

import pytest

from tools.quality_eval import quality_eval


@pytest.mark.parametrize("serial,drafted", [(False, 2), (True, 0), (True, 2)])
def test_teacher_decoding_mode(tmp_path, monkeypatch, serial, drafted):
    corpus = tmp_path / "tokens.json"
    corpus.write_text(json.dumps({
        "schema": quality_eval.SCHEMA_TOKENS,
        "documents": [{"id": "sample", "kind": "text", "prompt": [1, 2], "continuation": [3, 4]}],
    }))
    out = tmp_path / "scores.json"
    calls = []

    def fake_post(url, body):
        calls.append((url, body))
        return {
            "prompt": [{"id": 2, "position": 1, "logprob": -1.0, "top": []}],
            "continuation": [
                {"id": 3, "position": 2, "logprob": -1.0, "top": [], "origin": "prompt-end"},
                {"id": 4, "position": 3, "logprob": -2.0, "top": [], "origin": [0, 1]},
            ],
            "stats": {"drafted": drafted},
        }

    monkeypatch.setattr(quality_eval, "_post", fake_post)
    args = ["score", "--tokens", str(corpus), "--out", str(out)]
    if serial:
        args += ["--serial-decode"]
    if serial and drafted:
        with pytest.raises(SystemExit, match="reported drafts during serial scoring"):
            quality_eval.main(args)
        assert not out.exists()
    else:
        quality_eval.main(args)
        report = json.loads(out.read_text())
        assert report.get("decode_mode") == ("serial" if serial else None)
        doc, = report["runs"][0]["documents"]
        assert [row["id"] for row in doc["prefill"]] == [2, 3]
        assert [row["id"] for row in doc["decode"]] == [4]
    _, body = calls[0]
    assert ("draft" in body) == serial
    if serial:
        assert body["draft"] is False


def test_serial_quality_rejects_backend_without_draft_control(tmp_path):
    with pytest.raises(SystemExit, match="requires the TensorFold scoring backend"):
        quality_eval.main(["score", "--backend", "vllm", "--serial-decode",
                           "--tokens", str(tmp_path / "unused.json"), "--out", str(tmp_path / "scores.json")])


def test_quality_compare_requires_matching_decode_modes(tmp_path):
    ref, candidate = tmp_path / "ref.json", tmp_path / "candidate.json"
    metadata = {"tokens_sha256": "same-corpus", "continuation_limit": 32}
    ref.write_text(json.dumps(metadata))
    candidate.write_text(json.dumps({**metadata, "decode_mode": "serial"}))
    with pytest.raises(SystemExit, match="different decoding modes"):
        quality_eval.main(["compare", str(ref), str(candidate)])
