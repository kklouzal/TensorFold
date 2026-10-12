"""A CUDA request that waits for tokens (queued, or its long prompt filling) ticks its caller every TICK_S.

Nothing was written while a prompt filled, so a client with an inactivity timeout gave up on a long cold prefill;
and a client that left was only noticed at the first decoded round, so the GPU went on filling a prompt nobody would
read. Ticks write an SSE comment, check the client, and a request whose client left stops even mid-prompt, keeping
the state its passes committed so that a retry of the same prompt resumes there.
"""

import json
import threading
import time
from types import SimpleNamespace

import pytest

from tensorfold.cleanup import finish
from tensorfold.cuda import scheduler as scheduling
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream


class Filling:
    """A decoder whose prompts never finish filling (a pass a round) until they are cancelled."""

    def __init__(self):
        self.filling, self.streams, self.admitted, self.cancelled = [], {}, [], []

    def live(self):
        return len(self.filling)

    def admit(self, s):
        self.admitted.append(s.prompt)
        self.filling.append(s)

    def round(self):
        time.sleep(0.005)
        return []

    def cancel(self, streams):
        for s in streams:
            self.filling.remove(s)
            s.done = True
        self.cancelled += [s.prompt for s in streams]
        return streams

    def finish(self, done):
        pass

    def drop(self):
        live, self.filling = self.filling, []
        return live


@pytest.fixture
def ticking(monkeypatch):
    monkeypatch.setattr(scheduling, "TICK_S", 0.02)
    owners = []

    def start(decoder, **kw):
        owners.append(Scheduler(decoder, **kw).start())
        return owners[-1]

    yield start
    finish([owner.close for owner in owners])


def leaves_after(n, ticks):
    def emit(new):
        ticks.append(list(new))
        return len(ticks) >= n                      # True: the client has gone
    return emit


def test_a_filling_prompt_ticks_and_stops_when_its_client_leaves(ticking):
    dec = Filling()
    sched = ticking(dec, max_streams=2)
    ticks = []
    stats = sched.submit([1, 2, 3], 4, None, True, leaves_after(3, ticks))
    assert ticks == [[], [], []]                    # only ticks: no tokens were ever decoded
    assert dec.cancelled == [[1, 2, 3]] and dec.filling == [] and isinstance(stats, dict)


def test_a_queued_request_whose_client_left_is_never_admitted(ticking):
    dec = Filling()
    sched = ticking(dec, max_streams=1)
    first_ticks, second_ticks, replies = [], [], []
    first = threading.Thread(target=lambda: replies.append(sched.submit([1], 4, None, True,
                                                                        leaves_after(6, first_ticks))))
    first.start()
    while not dec.admitted:
        time.sleep(0.005)
    replies.append(sched.submit([2], 4, None, True, leaves_after(1, second_ticks)))   # queued behind the first
    first.join(5)
    assert dec.admitted == [[1]] and dec.cancelled == [[1]] and len(replies) == 2


def test_a_cancelled_prompt_keeps_the_state_its_passes_committed():
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    def decoder(*streams, at=700, mtp=True):
        dec = object.__new__(MultiDecoder)
        dec.filling, dec.passed = list(streams), {s.sid: 2 for s in streams}
        dec.fills = {s.sid: [None, mtp, at, [], "tail" if mtp else None] for s in streams}
        dec.kept_calls = []
        dec._remember = lambda ids, st, snap, tail: dec.kept_calls.append((len(ids), snap, tail))
        return dec

    def stream(draft=True, sid=4, cached=0):
        st = SimpleNamespace(mtp_len=700, image_positions=None, snapshot=lambda: {"pos": 700, "rec": "R"})
        return Stream(list(range(2_000)), 5, None, draft=draft, sid=sid, st=st, cached=cached)

    s = stream()
    dec = decoder(s)
    assert dec.cancel([s]) == [s] and s.done and s.finished > 0
    assert dec.kept_calls == [(700, {"pos": 700, "rec": "R", "mtp_len": 699}, "tail")]   # as a piece-end point
    assert dec.filling == [] and dec.fills == {} and dec.passed == {}
    s = stream()
    plain = decoder(s, mtp=False)
    plain.cancel([s])
    assert plain.kept_calls == [(700, {"pos": 700, "rec": "R", "mtp_len": 700}, None)]
    # too little filled; a serial (undrafted) stream; resumed at 700 and cancelled before a pass (its kept entry stands)
    for s, at in [(stream(), 100), (stream(draft=False), 700), (stream(cached=700), 700)]:
        dec = decoder(s, at=at)
        dec.cancel([s])
        assert dec.kept_calls == [] and s.done
    # a state that fails its check is not kept: that stream ends with the error and its slot keeps nothing (finish frees
    # it); the next cancelled stream still stops and keeps its state
    bad, good = stream(sid=4), stream(sid=5)
    dec = decoder(bad, good)
    dec.kept = [([1] * 300, bad.st, {}, None), ([2] * 300, good.st, {}, None)]
    refused = ValueError("a KV snapshot needs matching ordered formats, working policy and validated status")

    def remember(ids, st, snap, tail):
        if st is bad.st:
            raise refused
        dec.kept_calls.append((len(ids), snap, tail))

    dec._remember = remember
    assert dec.cancel([bad, good]) == [bad, good] and bad.done and good.done
    assert bad.error is refused and good.error is None
    assert [k[1] for k in dec.kept] == [good.st]
    assert dec.kept_calls == [(700, {"pos": 700, "rec": "R", "mtp_len": 699}, "tail")]
    assert dec.filling == [] and dec.fills == {} and dec.passed == {}


def test_a_tick_is_an_sse_comment_and_the_reply_is_unchanged(tmp_path):
    from tests.test_cuda_admission import http_server
    from tests.test_cuda_server_errors import EOS, HI, app_for, events, request

    class Ticking:
        eos = (EOS,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            assert on_tokens([]) is False                # a tick: the client is still there
            on_tokens([ord(c) for c in "Hi"])
            on_tokens([EOS])
            return {"generated": 3}

    app = app_for(tmp_path)
    app.engine = Ticking()
    with http_server(app) as port:
        status, _, text = request(port, {"messages": HI, "stream": True})
        plain = request(port, {"messages": HI})
    assert status == 200 and ": keepalive\n\n" in text
    frames = [f for f in events(text) if f != "[DONE]"]
    assert "".join(f["choices"][0]["delta"].get("content") or "" for f in frames) == "Hi"
    assert plain[0] == 200 and json.loads(plain[2])["choices"][0]["message"]["content"] == "Hi"


@pytest.mark.parametrize("route, body", [
    ("/v1/messages", {"model": "fake-cuda", "max_tokens": 16, "messages": [{"role": "user", "content": "Hi"}],
                      "stream": True}),
    ("/v1/responses", {"model": "fake-cuda", "input": "Hi", "stream": True}),
])
def test_a_translated_stream_passes_the_keepalive_on(tmp_path, route, body):
    """The Messages and Responses streams are the chat handler's, translated: its keepalive reaches their client."""

    import http.client

    from tests.test_cuda_admission import http_server
    from tests.test_cuda_server_errors import EOS, app_for

    ticked = []

    class Ticking:
        eos = (EOS,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            ticked.append(on_tokens([]))
            on_tokens([ord(c) for c in "Hi"])
            on_tokens([EOS])
            return {"generated": 3}

    app = app_for(tmp_path)
    app.engine = Ticking()
    with http_server(app) as port:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            connection.request("POST", route, json.dumps(body).encode(), {"Content-Type": "application/json"})
            response = connection.getresponse()
            status, text = response.status, response.read().decode()
        finally:
            connection.close()
    assert status == 200 and ticked == [False]
    blocks = text.split("\n\n")
    assert ": keepalive" in blocks                           # a comment of its own, before the reply's text
    data = [json.loads(line[6:]) for block in blocks for line in block.split("\n") if line.startswith("data: ")]
    assert "Hi" in json.dumps(data)
    assert blocks.index(": keepalive") < next(i for i, b in enumerate(blocks) if "Hi" in b)


def test_a_tick_whose_write_fails_stops_the_request_and_raises_after_generate(tmp_path):
    """As for a delta: the failure never raises into the engine (or the scheduler's submit); True stops the request and
    the reply raises it once ``generate`` returns."""

    from tests.test_cuda_server_errors import EOS, HI, app_for

    broken, seen = RuntimeError("keepalive write failed"), []

    class Ticking:
        eos = (EOS,)

        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            seen.append(on_tokens([]))
            return {"generated": 0}

    def emit(delta):
        if delta is None:
            raise broken
        return True

    app = app_for(tmp_path)
    app.engine = Ticking()
    with pytest.raises(RuntimeError) as caught:
        app.run({"messages": HI, "stream": True}, True, emit)
    assert caught.value is broken and seen == [True]
