"""Gemma target packet/dispatch integration against the retained public sampler."""

import weakref
from contextlib import nullcontext

import numpy as np
import pytest
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.sampling import sample_rows
from tensorfold.families.gemma4.cuda.sampling import greedy_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Gemma target packet requires CUDA")


def _raise_retired(primary, prior_cause, prior_context, prior_suppress, statuses):
    """Retain body/native/exit roots after the selected context has retired."""
    if primary is None:
        return
    try:
        errors = [error for error in statuses if error is not None]
        BaseException.__cause__.__set__(primary, prior_cause)
        BaseException.__context__.__set__(primary, prior_context)
        BaseException.__suppress_context__.__set__(primary, prior_suppress)
        raise_failures(primary, errors)
    except BaseException as transport:
        if transport is primary:
            raise
        raise primary from transport


class _OwnedContext:
    """Preserve the library lifecycle and capture body roots before its exit.

    This is the qualified MoE owning boundary, local to these test fixtures so
    neither an unselected benchmark nor private artifacts become dependencies.
    It performs no stream switch, graph capture, settlement or retry itself.
    """
    __slots__ = ('_cause', '_context', '_enter', '_exit', '_manager', '_statuses', '_suppress')

    def __init__(self, context):
        self._manager = context
        self._enter, self._exit = type(context).__enter__, type(context).__exit__
        self._statuses = [None]
        self._cause = BaseException.__cause__.__get__
        self._context = BaseException.__context__.__get__
        self._suppress = BaseException.__suppress_context__.__get__

    def __enter__(self):
        return self._enter(self._manager)

    def __exit__(self, error_type, primary, traceback):
        if primary is None:
            return self._exit(self._manager, error_type, primary, traceback)
        prior_cause = self._cause(primary)
        prior_context = self._context(primary)
        prior_suppress = self._suppress(primary)
        try:
            result = self._exit(self._manager, error_type, primary, traceback)
        except BaseException as error:
            if error is not primary:
                self._statuses[0] = error
            try:
                _raise_retired(primary, prior_cause, prior_context, prior_suppress, self._statuses)
            except BaseException as transport:
                if transport is not primary:
                    raise primary from transport
                raise
        else:
            BaseException.__cause__.__set__(primary, prior_cause)
            BaseException.__context__.__set__(primary, prior_context)
            BaseException.__suppress_context__.__set__(primary, prior_suppress)
            return result


class _LibraryContext:
    """Observe our adapter's calls without imitating CUDA context internals."""

    def __init__(self, *, enter_error=None, exit_error=None, suppress=False):
        self.enter_error, self.exit_error, self.suppress = enter_error, exit_error, suppress
        self.entries, self.exits = 0, []
        self.value = object()

    def __enter__(self):
        self.entries += 1
        if self.enter_error is not None:
            raise self.enter_error
        return self.value

    def __exit__(self, error_type, primary, traceback):
        self.exits.append((error_type, primary, traceback))
        if self.exit_error is not None:
            if primary is not None:
                BaseException.__cause__.__set__(primary, None)
                BaseException.__context__.__set__(primary, None)
            raise self.exit_error
        return self.suppress


def _roots(error):
    if isinstance(error, BaseExceptionGroup):
        return [root for nested in error.exceptions for root in _roots(nested)]
    return [error]


def test_owned_fixture_context_preserves_library_protocol():
    context = _LibraryContext()
    context.__enter__ = context.__exit__ = None  # Special methods are type-bound.
    with _OwnedContext(context) as value:
        assert value is context.value
    assert context.entries == 1 and context.exits == [(None, None, None)]
    primary = RuntimeError("explicit suppression")
    suppressing = _LibraryContext(suppress=True)
    with _OwnedContext(suppressing):
        raise primary
    assert suppressing.exits[0][:2] == (RuntimeError, primary)
    entering = _LibraryContext(enter_error=primary)
    with pytest.raises(RuntimeError) as caught, _OwnedContext(entering):
        pytest.fail("failed context entry reached the body")
    assert caught.value is primary and entering.entries == 1 and entering.exits == []
    cleanup = OSError("exit without body failure")
    exiting = _LibraryContext(exit_error=cleanup)
    with pytest.raises(OSError) as caught, _OwnedContext(exiting):
        pass
    assert caught.value is cleanup and exiting.exits == [(None, None, None)]


@pytest.mark.parametrize("error_type", [RuntimeError, MemoryError, KeyboardInterrupt])
def test_owned_fixture_context_retains_body_native_and_exit_roots(error_type):
    primary, cause, context, cleanup = error_type("body"), OSError("prior"), KeyError("context"), OSError("exit")
    primary.__cause__, primary.__context__ = cause, context
    manager = _LibraryContext(exit_error=cleanup)
    with pytest.raises(error_type) as caught, _OwnedContext(manager):
        raise primary
    assert caught.value is primary and manager.entries == 1 and len(manager.exits) == 1
    assert manager.exits[0][:2] == (error_type, primary) and manager.exits[0][2] is not None
    assert all(any(root is expected for root in _roots(primary.__cause__)) for expected in (cause, context, cleanup))


def test_owned_fixture_context_diagnostic_exhaustion_retains_raw_statuses(monkeypatch):
    primary, prior, cleanup, allocation = RuntimeError("body"), KeyError("prior"), OSError("exit"), MemoryError()
    primary.__cause__ = prior
    manager = _LibraryContext(exit_error=cleanup)

    def exhaust(*_args):
        raise allocation

    monkeypatch.setitem(globals(), "_raise_retired", exhaust)
    with pytest.raises(RuntimeError) as caught, _OwnedContext(manager):
        raise primary
    assert caught.value is primary and primary.__cause__ is allocation
    frame = primary.__traceback__
    retained = False
    while frame is not None:
        values = frame.tb_frame.f_locals
        if values.get("self") is not None and values.get("self") is not manager:
            owner = values["self"]
            if isinstance(owner, _OwnedContext) and values.get("prior_cause") is prior:
                retained = owner._statuses[0] is cleanup
        frame = frame.tb_next
    assert retained and manager.entries == 1 and len(manager.exits) == 1


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_original_dtype_strided_rows_match_legacy_and_independent_columns(dtype):
    limit = torch.finfo(dtype)
    values = np.tile(np.linspace(-1, 1, 17), (4, 1))
    values[0, 3] = values[0, 8] = 2
    values[1] = 0.0
    values[1, ::2] = -0.0
    values[2] = -np.inf
    values[2, 7] = -limit.tiny
    values[3] = -limit.max
    values[3, 3:5] = limit.max
    if dtype == torch.float64:
        # A narrowing cast before selection would lose the adjacent winner;
        # a narrowing cast after selection would misclassify these finite rows.
        values[3, 3:5] = [1e300, np.nextafter(1e300, np.inf)]
    storage = torch.zeros((4, 34), dtype=dtype, device="cuda")
    logits = storage[:, ::2]
    logits.copy_(torch.tensor(values, dtype=dtype, device="cuda"))
    original = logits.contiguous().view(torch.uint8).clone()
    expected = np.argmax(logits.double().cpu().numpy(), axis=-1).tolist()
    positions = [31, 32, 33, (1 << 64) - 1]
    assert greedy_rows(logits, positions, None) == sample_rows(logits, positions, None) == expected
    assert torch.equal(logits.contiguous().view(torch.uint8), original)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_whole_packet_errors_keep_legacy_type_message_and_global_precedence(bad, reverse, dtype):
    logits = torch.ones((3, 17), dtype=dtype, device="cuda")
    logits[0] = -float("inf")
    if bad == -float("inf"):
        logits[2] = bad
        message = "a complete vocabulary row needs at least one finite score"
    else:
        logits[2, 9] = bad
        message = "sampling scores contain NaN or positive infinity"
    if reverse:
        logits = logits.flip(0)
    for sampler in (sample_rows, greedy_rows):
        with pytest.raises(ValueError) as caught:
            sampler(logits, [17, 18, 19], None)
        assert type(caught.value) is ValueError and caught.value.args == (message,)


def test_empty_rows_validate_before_return_and_do_not_reduce(monkeypatch):
    logits = torch.empty((0, 17), device="cuda")
    monkeypatch.setattr(torch.Tensor, "max", lambda *args, **kwargs: pytest.fail("empty packet was reduced"))
    assert greedy_rows(logits, [], None) == sample_rows(logits, [], None) == []
    for sampler in (sample_rows, greedy_rows):
        with pytest.raises(ValueError, match="one position per row"):
            sampler(logits, [0], None)


@pytest.mark.parametrize("nondefault", [False, True])
def test_readback_completes_producer_stream_and_retains_no_score_owner(nondefault):
    stream = torch.cuda.Stream() if nondefault else None
    with _OwnedContext(torch.cuda.stream(stream) if stream is not None else nullcontext()):
        logits = torch.zeros((4, 4099), device="cuda")
        logits[:, 7].fill_(9)
        done = torch.cuda.Event()
        done.record()
        reference = weakref.ref(logits)
        ids = greedy_rows(logits, [11, 12, 13, 14], None)
        assert ids == [7] * 4 and done.query()
        del logits
        assert reference() is None


def test_readback_failure_keeps_original_error_and_live_input_owner(monkeypatch):
    logits = torch.ones((2, 17), device="cuda")
    reference = weakref.ref(logits)
    primary = RuntimeError("target packet copy failed")

    def failed_copy(packet, *args, **kwargs):
        assert reference() is logits and packet.dtype == torch.int64
        raise primary

    monkeypatch.setattr(torch.Tensor, "cpu", failed_copy)
    with pytest.raises(RuntimeError) as caught:
        greedy_rows(logits, [3, 4], None)
    assert caught.value is primary


def test_borrowed_graph_output_replays_new_rows_without_token_memo():
    source = torch.zeros((3, 17), device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with _OwnedContext(torch.cuda.stream(stream)):
        warm = source + 0
    torch.cuda.current_stream().wait_stream(stream)
    del warm
    graph = torch.cuda.CUDAGraph()
    with _OwnedContext(torch.cuda.graph(graph)):
        logits = source + 0
    for column in (2, 13, 5):
        source.zero_()
        source[:, column].fill_(1)
        graph.replay()
        assert greedy_rows(logits, [9, 10, 11], None) == sample_rows(logits, [9, 10, 11], None) == [column] * 3
