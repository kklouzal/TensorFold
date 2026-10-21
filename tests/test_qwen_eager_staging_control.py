"""Execute the private ownership/key protocol without importing CUDA kernels."""

from __future__ import annotations

import ast
import gc
import time
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.cleanup import rollback

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "src/tensorfold/families/qwen3_5/cuda/eager_staging.py"
DECODE = ROOT / "src/tensorfold/families/qwen3_5/cuda/decode.py"


class Bag:
    def __init__(self, **fields):
        self.__dict__.update(fields)


class Tensor:
    def __init__(self, address, shape=(512, 1, 128)):
        self.address, self.shape = address, shape
        self.dtype, self.device = "bf16", "cuda:0"

    def data_ptr(self):
        return self.address

    def stride(self):
        return (128, 128, 1)


class Stream:
    def __init__(self):
        self.waits, self.error = 0, None

    def synchronize(self):
        self.waits += 1
        if self.error is not None:
            raise self.error


@pytest.fixture
def protocol():
    stream = Stream()
    cuda = NS(default_stream=lambda device: stream, current_stream=lambda device: cuda.current)
    cuda.current = stream
    runtime = NS(cuda=cuda)
    attention = NS(CHUNK=512, slots=lambda p, w: -(-(p + w) // 512))
    tree = ast.parse(SOURCE.read_bytes())
    body = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
            or isinstance(node, ast.ImportFrom) and node.module == "__future__"]
    scope = {"__name__": __name__, "weakref": weakref, "dataclass": dataclass, "torch": runtime,
             "attention": attention, "rollback": rollback}
    exec(compile(ast.Module(body, []), str(SOURCE), "exec"), scope)  # noqa: S102 - actual owned protocol, imports supplied above.
    model = Bag(config=NS(conv_kernel=4, heads=2, kv_heads=1), norm=NS(device="cuda:0"))
    origin = Tensor(16, (32,))
    state = Bag(pos=504, kv_owner=object(), kv=[(Tensor(1024), Tensor(2048))])
    owner = scope["_RequestStaging"](model)
    return NS(owner=owner, model=model, state=state, origin=origin, stream=stream, cuda=cuda,
              parents=list(range(-1, 7)), softmax=(0,))


def begin(p, *, parents=None, chain=True):
    return p.owner.begin(p.model, p.state, p.parents if parents is None else parents,
                         chain, p.softmax, p.origin)


def capture(p):
    assert begin(p) is None
    p.owner.capture(p.model, p.state, p.parents, p.softmax, p.origin,
                    object(), NS(streams=object()), {}, object())
    entry = p.owner.entry
    p.owner.completed()
    return entry


def test_only_completed_exact_key_reuses_one_entry(protocol):
    p = protocol
    first = capture(p)
    p.state.pos = 503  # floor unchanged, ceil unchanged for this width.
    assert begin(p) is first
    with pytest.raises(RuntimeError, match="completed"):
        begin(p)
    p.owner.completed()
    p.owner.close()
    assert p.owner.entry is None and p.stream.waits == 0
    p.owner.close()
    with pytest.raises(RuntimeError, match="open"):
        begin(p)


@pytest.mark.parametrize("position", [505, 512, 513, 1016, 1017, 1024])
def test_both_attention_boundary_quantities_invalidate(protocol, position):
    p = protocol
    capture(p)
    p.state.pos = position
    assert begin(p) is None and p.owner.entry is None
    p.owner.close()
    assert p.stream.waits == 1


@pytest.mark.parametrize("change", ["owner", "view", "address", "shape", "dtype", "origin", "geometry", "width", "tree", "stream"])
def test_changed_authority_or_layout_evicts_before_replacement(protocol, change):
    p = protocol
    capture(p)
    if change == "owner":
        p.state.kv_owner = object()
    elif change == "view":
        p.state.kv[0] = Tensor(1024), p.state.kv[0][1]
    elif change == "address":
        p.state.kv[0][0].address = 4096
    elif change == "shape":
        p.state.kv[0][0].shape = (256, 1, 128)
    elif change == "dtype":
        p.state.kv[0][0].dtype = "float32"
    elif change == "origin":
        p.origin = Tensor(16, (32,))
    elif change == "geometry":
        p.model.config.conv_kernel = 1
    elif change == "width":
        p.parents = [-1]
    elif change == "tree":
        p.parents = [-1, 0, 0, 1, 1, 2, 2, 3]
    elif change == "stream":
        p.cuda.current = Stream()
    assert begin(p, chain=change != "tree") is None and p.owner.entry is None
    p.owner.close()
    assert p.cuda.current.waits == 1


def test_weak_cache_identity_does_not_keep_relocated_arena_view(protocol):
    p = protocol
    entry = capture(p)
    reference = weakref.ref(p.state.kv[0][0])
    p.state.kv[0] = Tensor(1024), p.state.kv[0][1]
    gc.collect()
    assert reference() is None and entry.caches[0][0][0]() is None
    assert begin(p) is None
    p.owner.close()


def test_private_owner_is_bound_to_actual_live_model(protocol):
    p = protocol
    model_reference = weakref.ref(p.model)
    other = Bag(config=p.model.config, norm=p.model.norm)
    with pytest.raises(ValueError, match="another model"):
        p.owner.begin(other, p.state, p.parents, True, p.softmax, p.origin)
    p.model = None
    gc.collect()
    assert model_reference() is None
    p.owner.close()


def test_failed_pending_retirement_keeps_owner_and_primary(protocol):
    p = protocol
    primary, cleanup = ValueError("forward failed"), OSError("stream failed")
    p.stream.error = cleanup
    with pytest.raises(ValueError) as caught, p.owner:
        begin(p)
        raise primary
    assert caught.value is primary and primary.__cause__ is cleanup
    assert primary._tensorfold_retained_owners == [p.owner]
    assert p.owner.pending and not p.owner.closed
    p.stream.error = None
    p.owner.close()
    assert not p.owner.pending and p.owner.closed and p.stream.waits == 2


@pytest.mark.parametrize("fail_verify", [False, True])
def test_actual_decode_acknowledges_only_successful_verify_fence(protocol, fail_verify):
    p = protocol
    failure, events = OSError("verify fence failed"), []

    def synchronize():
        events.append("fence")
        if fail_verify and events.count("fence") == 2:
            raise failure

    def forward(w, tokens, parents, state, *, capture_taps, _request_staging):
        assert _request_staging is p.owner
        p.owner.begin(w, state, parents, True, (), None)
        events.append("forward")
        return None, None

    tree = ast.parse(DECODE.read_bytes())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in {"draft_decode", "DecodeResult", "_paths", "next_copy_rows"}]
    scope = {"__name__": __name__, "dataclass": dataclass, "field": field,
             "torch": NS(cuda=NS(synchronize=synchronize)), "time": time,
             "tree_forward": forward, "_tokens": lambda tokens, device: tokens,
             "sample_rows": lambda *args: [2], "commit": lambda *args, **kwargs: None,
             "_paths": lambda parents: ([0], True)}
    future = ast.ImportFrom("__future__", [ast.alias("annotations")], 0)
    exec(compile(ast.fix_missing_locations(ast.Module([future, *selected], [])), str(DECODE), "exec"), scope)  # noqa: S102 - actual decoder control flow with owned boundaries replaced.
    p.model.config.eos = ()
    if fail_verify:
        with pytest.raises(OSError) as caught, p.owner:
            scope["draft_decode"](p.model, p.state, [1], 1, 2, None, None,
                                   allow_copy=False, inplace=True, _staging=p.owner)
        assert caught.value is failure and p.stream.waits == 1
    else:
        with p.owner:
            result = scope["draft_decode"](p.model, p.state, [1], 1, 2, None, None,
                                            allow_copy=False, inplace=True, _staging=p.owner)
        assert result.tokens == [1, 2] and p.stream.waits == 0
    assert events == (["fence", "forward", "fence"] if fail_verify else ["fence", "forward", "fence", "fence"])
