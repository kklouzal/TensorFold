"""Frozen corpus allocation/publication contracts; dataset/tokenizer gates run on real sources."""
import stat

import pytest

from tensorfold.offline_quantization.text_corpus import _write_new, balanced_counts


def test_full_mix_is_balanced_subject_to_unique_training_capacity():
    assert balanced_counts({"math": 400, "code": 400, "prose": 400}, 1024) == {
        "math": 342, "code": 341, "prose": 341}
    assert balanced_counts({"math": 500, "code": 320, "prose": 500}, 1024) == {
        "math": 352, "code": 320, "prose": 352}
    with pytest.raises(ValueError, match="insufficient"):
        balanced_counts({"math": 1, "code": 1, "prose": 1}, 4)


def test_artifact_publication_is_complete_readonly_and_never_clobbers(tmp_path):
    target = tmp_path / "corpus.jsonl"
    _write_new(target, b'{"id":"public/train/1"}\n')
    original = target.read_bytes()
    assert stat.S_IMODE(target.stat().st_mode) == 0o444
    with pytest.raises(FileExistsError):
        _write_new(target, b'other data')
    assert target.read_bytes() == original
    assert sorted(path.name for path in tmp_path.iterdir()) == ["corpus.jsonl"]


def test_mbpp_excludes_exact_reserved_prompts_and_solutions_without_fuzzy_rules():
    from tensorfold.offline_quantization.text_corpus import _partition_mbpp
    values = [
        (1, "task A", "def a(): return 1"),
        (511, "task B", "def b(): return 2"),
        (601, " task A\n", "different solution"),
        (602, "different prompt", " def b(): return 2\n"),
        (603, "task  A", "another solution"),
        (604, "Task A", "yet another solution"),
    ]
    records = [(index, {"task_id": task, "text": prompt, "code": code}, str(index))
               for index, (task, prompt, code) in enumerate(values)]
    train, reserved, excluded, _ = _partition_mbpp(records)
    assert [row["task_id"] for _, row, _ in train] == [603, 604]
    assert [row["train_id"] for row in excluded] == [601, 602]
    assert excluded[0]["reserved_prompt_ids"] == [1]
    assert excluded[0]["reserved_solution_ids"] == []
    assert excluded[1]["reserved_prompt_ids"] == []
    assert excluded[1]["reserved_solution_ids"] == [511]
    assert len(reserved) == 2


@pytest.mark.parametrize("field,value", [("url", "https://example.invalid/other"),
                                         ("revision", "other"), ("split", "test"), ("sha256", "0" * 64)])
def test_wikitext_caller_claim_cannot_replace_frozen_acquisition(tmp_path, field, value):
    import json
    from tensorfold.offline_quantization.text_corpus import _wiki_lineage, WIKI_REV, WIKI_URL, WIKI_PARQUET_SHA
    source = {"url": WIKI_URL, "revision": WIKI_REV, "split": "train", "sha256": WIKI_PARQUET_SHA}
    source[field] = value
    (tmp_path / "wikitext-acquisition.json").write_text(json.dumps(source))
    (tmp_path / "wikitext-normalization.json").write_text("{}")
    with pytest.raises(ValueError, match="lineage"):
        _wiki_lineage(tmp_path)
