"""Configured decision admission spans preparation, scoring and result construction."""
import http.client
import json
from types import SimpleNamespace

import pytest

pytest.importorskip('tokenizers')
pytest.importorskip('jinja2')

from tensorfold.cuda.server import App
from tensorfold.server.errors import CapacityError
from tests.test_cuda_admission import http_server
from tests.test_cuda_server_errors import TextTokenizer

from cuda_http_fixtures import model_directory


@pytest.fixture
def decision_model_dir(tmp_path):
    return model_directory(tmp_path)


def body():
    return {'input': 'controlled', 'questions': [
        {'id': 'a', 'type': 'choice', 'question': 'controlled?', 'options': [{'name': 'yes'}, {'name': 'no'}]}]}


def app_for(directory):
    engine = SimpleNamespace(scheduler=None)
    app = App(engine, directory, 'controlled', max_pending_requests=1)
    observed = []
    class Tokenizer(TextTokenizer):
        def encode(self, *args, **kwargs):
            observed.append(('prepare', app.pending_requests))
            return super().encode(*args, **kwargs)
    app.tok = Tokenizer()
    def score(prompt, labels):
        observed.append(('score', app.pending_requests))
        return [1.0, 0.0], 1.2
    engine.score_labels = score
    return app, observed


def test_decision_caller_scope_covers_preparation_scoring_and_returns_after_retirement(decision_model_dir):
    app, observed = app_for(decision_model_dir)
    result = app.decisions(body())
    assert result['answers']['a']['choice'] == 'yes'
    assert {phase for phase, _ in observed} == {'prepare', 'score'}
    assert all(used == 1 for _, used in observed)
    assert app.pending_requests == 0


def test_decision_capacity_refuses_before_any_preparation_and_http_reports_503(decision_model_dir):
    app, observed = app_for(decision_model_dir)
    with app._request_limit.request():
        with pytest.raises(CapacityError):
            app.decisions(body())
        assert observed == []
        with http_server(app) as port:
            connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
            try:
                connection.request('POST', '/v1/decisions', json.dumps(body()), {'Content-Type': 'application/json'})
                response = connection.getresponse()
                assert response.status == 503
                assert 'capacity' in json.loads(response.read())['error']['message']
                assert observed == []
            finally:
                connection.close()
    assert app.pending_requests == 0


def test_decision_caller_interruption_preserves_primary_and_releases_retired_scope(decision_model_dir):
    app, _ = app_for(decision_model_dir)
    primary = KeyboardInterrupt('controlled scoring interruption')
    def score(prompt, labels):
        assert app.pending_requests == 1
        raise primary
    app.engine.score_labels = score
    with pytest.raises(KeyboardInterrupt) as caught:
        app.decisions(body())
    assert caught.value is primary
    assert app.pending_requests == 0
