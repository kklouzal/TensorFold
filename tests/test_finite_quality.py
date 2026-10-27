"""Independent fixed-answer oracles and bounded optional quality-phase controls."""
import gc
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest

from tensorfold.server.request_limits import RequestLimit
from tools import finite_concurrency as finite
from tools import finite_quality as quality
from tools import v100_backend_probe as probe
from tests.test_v100_backend_probe import cli


def test_fixed_cases_have_independent_answers_and_bounded_inputs():
    cases = {name: (prompt, answer) for name, prompt, answer in quality.CASES}
    assert len(cases) == 6 and all(len(prompt.encode("utf-8")) <= 4096 for prompt, _ in cases.values())
    assert cases["math"][1] == str(17 * 24 - 135 + 48)
    table = {("Cedar", "RED"): "P4N7", ("Birch", "GREEN"): "T8K2", ("Aster", "RED"): "W3J6",
             ("Cedar", "GREEN"): "R8Q2", ("Dune", "GREEN"): "M9L1", ("Birch", "RED"): "H5V4"}
    assert cases["retrieval"][1] == table["Cedar", "GREEN"]
    solutions = [row for row in itertools.product((False, True), repeat=3)
                 if sum(row) == 1 and (not row[0] or row[1]) and not row[2]]
    assert solutions == [(False, True, False)] and cases["logic"][1] == "ABC"[solutions[0].index(True)]
    words = ["cobalt", "amber", "jade", "silver"]
    assert cases["instruction"][1] == "/".join(words[index].upper() for index in (1, 3))
    assert json.loads(cases["json"][1]) == {"item": "tape", "remaining": 9 + 4 - 6,
                                          "labels": sorted(("green", "blue"))}
    # Execute only the fixed reviewed program, never model-produced code.
    code = cases["code-output"][0].split("\n", 1)[1]
    actual = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert actual.stdout.strip() == cases["code-output"][1]
    assert quality.OUTPUT_TOKENS == 320 and quality.control_payload_bytes() > sum(
        len(part.encode("utf-8")) for case in quality.CASES for part in case)


@pytest.mark.parametrize("case_id,prompt,expected", quality.CASES)
def test_oracle_accepts_expected_answer_and_retains_objective_differences(case_id, prompt, expected):
    assert quality.assess(case_id, " \n" + expected + "\t")["passed"]
    wrong = quality.assess(case_id, expected + " extra")
    assert not wrong["passed"] and wrong["assessment_complete"]
    assert wrong["expected"] == expected and wrong["difference"]["offset"] == len(expected)
    assert wrong["difference"]["observed_slice"] == " extra"
    assert not quality.assess(case_id, "")["passed"]


def test_quality_flag_preserves_default_workload_and_requires_existing_funded_envelope(tmp_path):
    assert not probe.arguments(cli(tmp_path)).quality_spotchecks
    args = probe.arguments(cli(tmp_path) + ["--concurrency-grid", "--tokens", "320", "--quality-spotchecks"])
    assert args.prompt == probe.DEFAULT_PROMPT and args.tokens == 320 and finite.GRID == (1, 2, 4, 8, 16, 24)
    for extra in (["--quality-spotchecks"], ["--concurrency-grid", "--tokens", "256", "--quality-spotchecks"]):
        with pytest.raises(SystemExit):
            probe.arguments(cli(tmp_path) + extra)


def prepared_cases():
    return [(name, [index], {"rendered": prompt, "messages": [{"role": "user", "content": prompt}],
                            "token_ids": [index]}) for index, (name, prompt, _) in enumerate(quality.CASES)]


def quality_report():
    return {"quality": {"model_identity": {"profile": "independent-reference"},
                        "dataset_sha256": hashlib.sha256(json.dumps(quality.CASES, separators=(",", ":")).encode()).hexdigest(),
                        "helper_sha256": hashlib.sha256(Path(quality.__file__).read_bytes()).hexdigest(),
                        "completed_requests": 0, "failed_requests": 0, "passed": False}}


@pytest.mark.parametrize("wrong", [False, True])
def test_quality_phase_uses_same_owner_persists_all_answers_and_retires_previous_graph(tmp_path, monkeypatch, wrong):
    args = SimpleNamespace(out=tmp_path, serial_reference=True)
    report, limit, references, calls = quality_report(), RequestLimit(24), [], []

    class IDs(list):
        pass

    def invoke(_engine, _tokenizer, tokens, *, count, draft, limit, result):
        gc.collect()
        assert all(reference() is None for reference in references)
        assert count == 320 and limit.maximum == 24
        case_id = quality.CASES[tokens[0]][0]
        with limit.request():
            assert limit.used == 1
            answer = quality.CASES[tokens[0]][2]
            ids = IDs([tokens[0]])
            references.append(weakref.ref(ids))
            result.update(text="wrong" if wrong and case_id == "math" else answer,
                          token_ids=ids, stats={"cached": int(not draft)}, complete=True)
            calls.append((case_id, draft))

    monkeypatch.setattr(probe, "generate_reply", invoke)
    try:
        if wrong:
            with pytest.raises(ValueError, match="all raw answers"):
                probe.run_quality_spotchecks(object(), object(), args, report, limit, prepared_cases(), quality)
        else:
            probe.run_quality_spotchecks(object(), object(), args, report, limit, prepared_cases(), quality)
        assert len(calls) == 12 and limit.used == 0
        assert report["quality"]["completed_requests"] == 12
        assert report["quality"]["failed_requests"] == (2 if wrong else 0)
        assert report["quality"]["passed"] is not wrong
        for case_id, draft in calls:
            mode = "mtp-target" if draft else "serial-reference"
            row = json.loads((tmp_path / f"quality-{case_id}-{mode}.json").read_text())
            assert row["token_ids"] and row["source_prompt_sha256"] and row["prompt_sha256"]
            assert "excluded from performance" in row["measurement_role"]
            assert row["quality"]["passed"] is (not wrong or case_id != "math")
        assert all(reference() is None for reference in references)
    finally:
        limit.close()


def test_quality_generation_failure_preserves_raw_partial_answer_and_does_not_continue(tmp_path, monkeypatch):
    primary, report, limit = RuntimeError("generation failed"), quality_report(), RequestLimit(24)
    calls = []
    def invoke(*_args, limit, result, **_kwargs):
        with limit.request():
            calls.append(1)
            result.update(token_ids=[7], text="partial", complete=False)
            raise primary
    monkeypatch.setattr(probe, "generate_reply", invoke)
    try:
        with pytest.raises(RuntimeError) as caught:
            probe.run_quality_spotchecks(object(), object(), SimpleNamespace(out=tmp_path, serial_reference=False),
                                        report, limit, prepared_cases(), quality)
        assert caught.value is primary and calls == [1] and limit.used == 0
        row = json.loads((tmp_path / "quality-math-mtp-target.json").read_text())
        assert row["text"] == "partial" and row["token_ids"] == [7] and not row["quality"]["passed"]
    finally:
        limit.close()


def test_quality_inputs_and_static_controls_are_funded_before_admission_or_generation(tmp_path, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import retention_resources as resources

    args = probe.arguments(cli(tmp_path) + ["--concurrency-grid", "--tokens", "320", "--quality-spotchecks"])
    prompts, report = [], {"tokenizer_identity": "prepared", "native_flags": {}, "prepared_model_dir": "/prepared"}
    engine = SimpleNamespace(sm70_profile="reference", retention_catalog=object(),
        scheduler=SimpleNamespace(configure_requests=lambda _limit: pytest.fail("configured before admission")))
    def prepare(_engine, _tokenizer, value):
        prompts.append(value.prompt)
        return [len(prompts)], {"rendered": value.prompt, "messages": [{"content": value.prompt}]}
    primary = MemoryError("physical host admission refused")
    def refuse(_catalog, *, caller, **_kwargs):
        assert len(prompts) == 8 and prompts[:2] == [args.prompt, finite.SUSTAINED_PROMPT]
        assert caller.max_concurrent_requests == caller.retained_results == 24
        assert caller.max_output_tokens == 320 and caller.max_input_tokens == 4096
        assert caller.control_payload_bytes > finite.control_payload_bytes(320) + quality.control_payload_bytes()
        assert report["quality"]["prepared_input_owners_total"] == 8
        raise primary
    monkeypatch.setattr(probe, "prepared_prompt", prepare)
    monkeypatch.setattr(resources, "finalize_backend_resources", refuse)
    monkeypatch.setattr(probe, "generate_reply", lambda *_args, **_kwargs: pytest.fail("generated before admission"))
    with pytest.raises(MemoryError) as caught:
        probe.run_concurrency_leader(engine, object(), args, report)
    assert caught.value is primary
