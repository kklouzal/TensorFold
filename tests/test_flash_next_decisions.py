"""/v1/decisions on Flash Next CUDA: the label collector and selected serial question scoring (no GPU needed)."""

from tensorfold.cuda import server
from tensorfold.engine.probabilities import LabelProbabilities
from tests.test_cuda_server_errors import app_for


def test_label_probabilities_keep_the_prompt_end_only():
    probe = LabelProbabilities([5, 9], start=12)
    assert probe.top == 0 and probe.labels == [5, 9]
    probe.add_labels(11, [0.0, 0.0], 0.0)                 # a position before the prompt's end: ignored
    assert probe.label_logits is None
    probe.add_labels(12, [1.5, -2.0], 3.25)
    assert probe.label_logits == [1.5, -2.0] and probe.logsumexp == 3.25


def _body():
    choice = {"type": "choice", "options": [{"name": "a"}, {"name": "b"}]}
    return {"input": "x", "questions": [{"id": "first", "question": "One?", **choice},
                                        {"id": "second", "question": "Two?", **choice}]}


class _Grouped:
    def __init__(self):
        self.calls = []

    def score_labels_many(self, items):
        raise AssertionError("unqualified automatic bulk path selected")

    def score_labels(self, prompt, labels):
        self.calls.append(list(prompt))
        return [2.0, 0.0], 2.2


class _OneByOne:
    def __init__(self):
        self.calls = 0

    def score_labels(self, prompt, labels):
        self.calls += 1
        return [0.0, 2.0], 2.2


def test_cuda_decisions_use_serial_scoring_when_provider_also_has_a_bulk_method(tmp_path):
    app = app_for(tmp_path, server.App)
    app.engine = _Grouped()
    out = app.decisions(_body())
    assert len(app.engine.calls) == 2 and all(app.engine.calls)
    assert out["answers"]["first"]["choice"] == "a" and out["answers"]["second"]["choice"] == "a"


def test_cuda_decisions_without_the_grouped_path_score_one_by_one(tmp_path):
    app = app_for(tmp_path, server.App)
    app.engine = _OneByOne()
    out = app.decisions(_body())
    assert app.engine.calls == 2
    assert out["answers"]["first"]["choice"] == "b"
