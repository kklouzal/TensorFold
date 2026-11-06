"""CUDA-hidden contract: private query shapes cannot change public dispatch."""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / "src/tensorfold/families/gemma4/cuda/attention.py"
STATE = ROOT / "src/tensorfold/families/gemma4/cuda/state.py"


def functions():
    return {node.name: node for node in ast.parse(ATTENTION.read_text()).body
            if isinstance(node, ast.FunctionDef)}


@pytest.mark.parametrize("name", ["_queries", "_global_chunk", "_global"])
def test_private_query_rows_is_appended_default_16(name):
    args = functions()[name].args
    assert args.args[-1].arg == "QUERY_ROWS"
    assert ast.literal_eval(args.defaults[-1]) == 16
    assert not args.kwonlyargs and args.vararg is None and args.kwarg is None


def test_public_full_dispatch_cannot_select_private_query_rows():
    fn = functions()["full"]
    assert "QUERY_ROWS" not in {arg.arg for arg in fn.args.args + fn.args.kwonlyargs}
    calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call)]
    global_calls = [node for node in calls if isinstance(node.func, ast.Subscript)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "_global"]
    assert len(global_calls) == 2
    assert all("QUERY_ROWS" not in {kw.arg for kw in call.keywords} for call in global_calls)
    tiles = next(node.value for node in fn.body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "tiles" for target in node.targets))
    assert {node.id for node in ast.walk(tiles) if isinstance(node, ast.Name)} == {"rows", "g", "QUERY_TILE"}
    value = next(node.value for node in ast.parse(STATE.read_text()).body if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == "QUERY_TILE" for target in node.targets))
    assert ast.literal_eval(value) == 16


def test_only_global_forwards_the_private_shape():
    fns = functions()
    calls = [node for node in ast.walk(fns["_global"]) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id in {"_queries", "_global_chunk"}]
    assert [call.func.id for call in calls].count("_queries") == 1
    assert [call.func.id for call in calls].count("_global_chunk") == 2
    for call in calls:
        forwarded = [kw.value for kw in call.keywords if kw.arg == "QUERY_ROWS"]
        assert len(forwarded) == 1 and isinstance(forwarded[0], ast.Name)
        assert forwarded[0].id == "QUERY_ROWS"
    sliding = fns["_sliding"]
    call = next(node for node in ast.walk(sliding) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "_queries")
    assert "QUERY_ROWS" not in {kw.arg for kw in call.keywords}
    assert not any(isinstance(node, ast.Name) and node.id == "QUERY_ROWS"
                   for node in ast.walk(fns["_sliding_chunk"]))


def gpu_test_helper(name, torch):
    """Execute the actual test helper alone, without importing the CUDA suite."""
    path = ROOT / "tests/test_gemma_cuda_attention.py"
    node = next(node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)
                and node.name == name)
    namespace = {"torch": torch, "DEV": "cuda"}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("phase", ["begin", "body", "end", None])
def test_actual_capture_helper_preserves_failure_roots_without_cuda_cleanup(phase):
    # Fake only the public CUDA boundary. The helper's actual source and control
    # flow execute unchanged, then an outer transport retains the native roots.
    from types import SimpleNamespace

    calls = []
    primary = RuntimeError("injected native " + str(phase))
    cause = ValueError("original cause")
    context = LookupError("original context")
    primary.__cause__ = cause
    primary.__context__ = context
    primary.__suppress_context__ = False

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            calls.append(("wait", self.name, other.name))

    previous = Stream("previous")
    owned = Stream("owned")

    class CUDA:
        @staticmethod
        def current_stream():
            calls.append("current_stream")
            return previous

        @staticmethod
        def Stream():
            calls.append("Stream")
            return owned

        @staticmethod
        def set_stream(stream):
            calls.append(("set_stream", stream.name))

    class Graph:
        @staticmethod
        def capture_begin(*, capture_error_mode):
            assert capture_error_mode == "global"
            calls.append("begin")
            if phase == "begin":
                raise primary

        @staticmethod
        def capture_end():
            calls.append("end")
            if phase == "end":
                raise primary

    graph = Graph()
    output = object()
    owners = {"graph": None, "previous": None, "stream": None, "outputs": None}

    def body():
        calls.append("body")
        if phase == "body":
            raise primary
        return output

    helper = gpu_test_helper("_capture_query_rows", SimpleNamespace(cuda=CUDA()))
    transport = {}
    try:
        helper(graph, body, owners)
    except BaseException as error:
        # Models the qualified outer test boundary, not a test-local recovery
        # mechanism. No project CUDA call occurs while recording these roots.
        transport["primary"] = error
        transport["cause"] = BaseException.__cause__.__get__(error)
        transport["context"] = BaseException.__context__.__get__(error)
        transport["traceback"] = BaseException.__traceback__.__get__(error)
        transport["owners"] = owners
    prefix = ["current_stream", "Stream", ("wait", "owned", "previous"),
              ("set_stream", "owned"), "begin"]
    assert owners["graph"] is graph and owners["previous"] is previous and owners["stream"] is owned
    if phase is None:
        assert transport == {} and owners["outputs"] is output
        assert calls == prefix + ["body", "end", ("set_stream", "previous"), ("wait", "previous", "owned")]
    else:
        assert transport["primary"] is primary and transport["cause"] is cause and transport["context"] is context
        assert primary.__suppress_context__ is False and transport["owners"] is owners
        expected = prefix + ([] if phase == "begin" else ["body"]) + (["end"] if phase == "end" else [])
        assert calls == expected  # no capture end, stream restore or retry after failure
        assert owners["outputs"] is (output if phase == "end" else None)
        traceback = transport["traceback"]
        helper_frame = None
        while traceback is not None:
            if traceback.tb_frame.f_code.co_name == "_capture_query_rows":
                helper_frame = traceback.tb_frame
            traceback = traceback.tb_next
        assert helper_frame is not None and helper_frame.f_locals["held"] is owners
        assert helper_frame.f_locals["body"] is body


@pytest.mark.parametrize("failure_at", [2, 3, None])
def test_actual_private_inputs_preserve_rng_order_and_partial_gpu_owners(failure_at):
    from types import SimpleNamespace

    draws = []
    seeds = []
    cuda_calls = []
    primary = RuntimeError("injected input device transfer")

    class Tensor:
        def __init__(self, expression):
            self.expression = expression

        def __mul__(self, scalar):
            return Tensor(("mul", self.expression, scalar))

        def __add__(self, scalar):
            return Tensor(("add", self.expression, scalar))

        def to(self, kind):
            if kind == "cuda":
                cuda_calls.append(self)
                if len(cuda_calls) == failure_at:
                    raise primary
            return Tensor(("to", self.expression, kind))

    class Generator:
        def __init__(self, *, device):
            assert device == "cpu"

        def manual_seed(self, seed):
            seeds.append(seed)

    def randn(*shape, generator):
        assert isinstance(generator, Generator)
        index = len(draws)
        draws.append(shape)
        return Tensor(("randn", index, shape))

    helper = gpu_test_helper("_query_rows_inputs", SimpleNamespace(
        Generator=Generator, randn=randn, bfloat16="bf16"))
    c = SimpleNamespace(heads=16, kv_heads=8, global_kv_heads=2, head_dim=256, global_head_dim=512)
    owners = {'generator': None, 'discard': None, 'gq': [None] * 4, 'gk': [None] * 4,
              'gqw': [None] * 6, 'gkw': [None] * 6}
    retained = {}
    try:
        helper(c, 7, owners)
    except BaseException as error:
        retained['primary'] = error
        retained['traceback'] = BaseException.__traceback__.__get__(error)
        retained['owners'] = owners
    expected_draws = [(7, 4096), (7, 2048), (7, 2048), (7, 8192), (7, 1024),
                      (256,), (256,), (512,), (512,)]
    assert seeds == [19] and draws == expected_draws[:len(draws)]
    assert owners['gq'][3] is not None  # survives every later injected failure
    assert owners['gq'][3].expression == ("to", ("to", ("mul", ("randn", 3, (7, 8192)), 3.0), "bf16"), "cuda")
    if failure_at is None:
        assert retained == {} and len(cuda_calls) == 4 and draws == expected_draws
        # Preserve two distinct BF16 GPU operations; never fold the learned
        # scale and shift into a CPU fixture or a single rounding boundary.
        assert owners['gqw'][5].expression == ("add", ("mul", ("to", ("to", ("mul",
            ("randn", 7, (512,)), 3.0), "bf16"), "cuda"), 0.3), 1)
        assert owners['gkw'][5] is not None
    else:
        assert retained['primary'] is primary and retained['owners'] is owners
        assert retained['traceback'] is not None and len(cuda_calls) == failure_at
        failed = owners['gk' if failure_at == 2 else 'gqw']
        assert failed[2] is not None and failed[3] is None
        assert len(draws) == (5 if failure_at == 2 else 8)
        if failure_at == 3:
            assert owners['gk'][3] is not None


@pytest.mark.parametrize("phase", ["negative", "divide", "pow", "mask", None])
def test_actual_private_frequency_keeps_gpu_temporaries_and_operation_order(phase):
    from types import SimpleNamespace

    calls = []
    primary = RuntimeError("injected frequency " + str(phase))

    class Tensor:
        def __init__(self, expression):
            self.expression = expression

        def __neg__(self):
            calls.append("negative")
            if phase == "negative":
                raise primary
            return Tensor(("negative", self.expression))

        def __truediv__(self, dim):
            calls.append(("divide", dim))
            if phase == "divide":
                raise primary
            return Tensor(("divide", self.expression, dim))

        def __setitem__(self, key, value):
            calls.append(("mask", key.start, key.stop, key.step, value))
            if phase == "mask":
                raise primary

    def arange(start, stop, step, *, dtype, device):
        calls.append(("arange", start, stop, step, dtype, device))
        return Tensor(("indices", start, stop, step, dtype, device))

    def power(theta, exponent):
        calls.append(("pow", theta))
        if phase == "pow":
            raise primary
        return Tensor(("pow", theta, exponent.expression))

    helper = gpu_test_helper("_query_rows_frequency", SimpleNamespace(
        arange=arange, pow=power, float32="float32"))
    c = SimpleNamespace(global_head_dim=512, global_rotary_fraction=0.25, global_rope_theta=1e6)
    owners = {'indices': None, 'negative': None, 'exponent': None, 'value': None}
    retained = {}
    try:
        helper(c, owners)
    except BaseException as error:
        retained['primary'] = error
        retained['traceback'] = BaseException.__traceback__.__get__(error)
        retained['owners'] = owners
    expected_calls = [("arange", 0, 512, 2, "float32", "cuda"), "negative",
                      ("divide", 512), ("pow", 1e6), ("mask", 64, None, None, 0)]
    completed = {"negative": 1, "divide": 2, "pow": 3, "mask": 4, None: 4}[phase]
    keys = ['indices', 'negative', 'exponent', 'value']
    assert all(owners[key] is not None for key in keys[:completed])
    assert all(owners[key] is None for key in keys[completed:])
    if phase is None:
        assert retained == {} and calls == expected_calls
        assert owners['value'].expression == ("pow", 1e6, ("divide", ("negative",
            ("indices", 0, 512, 2, "float32", "cuda")), 512))
    else:
        assert retained['primary'] is primary and retained['owners'] is owners
        assert retained['traceback'] is not None
        assert calls == expected_calls[:completed + 1]  # failure never retries or invokes CUDA cleanup
