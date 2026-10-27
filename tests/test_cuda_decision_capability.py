"""Unsupported decision scoring rejects before expanding reused input prompts."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.profile_binding import ProfileGate
from tensorfold.cuda.server import App
from tensorfold.cuda.turns import Turns
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.server.decisions import DecisionError, prompts_for
from tensorfold.server.errors import RequestError
from tests.test_decisions import _choice


class Capability:
    supports_logprobs = FlashNextEngine.supports_logprobs
    decision_label_refusal = FlashNextEngine.decision_label_refusal
    score_labels = FlashNextEngine.score_labels

    def __init__(self, tp):
        self.tp = tp


def rejected_app():
    app = object.__new__(App)
    app.engine = Capability(8)
    app.template = SimpleNamespace(render=lambda *a, **kw: pytest.fail("unsupported backend rendered a prompt"))
    app.tok = SimpleNamespace(encode=lambda *a, **kw: pytest.fail("unsupported backend encoded a prompt"))
    return app


def test_authoritative_tp8_refusal_precedes_every_question_materialization():
    app = rejected_app()
    body = _choice()
    body["input"] *= 1024
    body["questions"] = [{**body["questions"][0], "id": f"q{i}"} for i in range(128)]
    with pytest.raises(ValueError) as backend:
        app.engine.score_labels([1], [2])
    with pytest.raises(RequestError) as refused:
        app.decisions(body)
    assert str(refused.value) == f"question 'q0': {backend.value}"


@pytest.mark.parametrize("change", ["unknown", "empty-input", "duplicate-id", "invalid-kind", "invalid-options"])
def test_lightweight_schema_errors_keep_existing_refusals(change):
    body = _choice()
    if change == "unknown":
        body["unknown"] = 1
    elif change == "empty-input":
        body["input"] = " "
    elif change == "duplicate-id":
        body["questions"].append(deepcopy(body["questions"][0]))
    elif change == "invalid-kind":
        body["questions"][0]["type"] = "unknown"
    else:
        body["questions"][0]["options"] = []
    with pytest.raises(DecisionError) as original:
        prompts_for(body, lambda *a: pytest.fail("invalid shape rendered"), lambda *a: pytest.fail("invalid shape encoded"))
    with pytest.raises(RequestError) as current:
        rejected_app().decisions(body)
    assert str(current.value) == str(original.value)


@pytest.mark.parametrize("advertised", [False, True])
def test_supported_or_unadvertised_backends_keep_the_existing_scoring_path(advertised):
    app = object.__new__(App)
    app.engine = Capability(1) if advertised else SimpleNamespace()
    seen = []
    app.engine.score_labels = lambda prompt, labels: (seen.append((prompt, labels)) or ([0., 2.], 2.))
    app.template = SimpleNamespace(render=lambda messages, **kw: messages[0]["content"] + "\n")
    app.tok = SimpleNamespace(encode=lambda text, **kw: SimpleNamespace(ids=list(map(ord, text))))
    app.context_window, app.turns = 0, Turns()
    result = app.decisions(_choice())
    assert result["answers"]["team"]["choice"] == "technical" and len(seen) == 1


def test_capability_query_uses_the_same_existing_profile_lease_as_the_operation():
    app, gate = rejected_app(), ProfileGate()
    observed = []

    class Leased(Capability):
        @property
        def decision_label_refusal(self):
            observed.append(gate.current())
            assert gate.borrowers == 1 and observed[-1] is not None
            return super().decision_label_refusal

    app.engine = Leased(8)
    app.decisions = gate.wrap(app.decisions)
    with pytest.raises(RequestError, match="one GPU only"):
        app.decisions(_choice())
    assert len(observed) == 1 and gate.borrowers == 0 and gate.current() is None
