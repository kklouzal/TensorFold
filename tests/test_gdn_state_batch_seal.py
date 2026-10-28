"""Execute the installed Multi target-operation boundary with actual owner sealing."""
import ast
import inspect
import textwrap
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.qwen4_exp.cuda.gdn_state_batch import GDNStateBatch  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402


def boundary(operation):
    tree = ast.parse(textwrap.dedent(inspect.getsource(MultiDecoder._round)))
    body = tree.body[0].body
    selected = next(index for index, node in enumerate(body) if isinstance(node, ast.Try)
                    and "gdn_state_batch.finish" in ast.unparse(node.finalbody))
    assert isinstance(body[selected - 1], ast.Assign)
    assert ast.unparse(body[selected - 1]) == "gdn_primary = None"
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg="self"), ast.arg(arg="segs")],
                         kwonlyargs=[], kw_defaults=[], defaults=[])
    function = ast.FunctionDef(name="target_boundary", args=args,
        body=[body[selected - 1], body[selected]], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"compute": operation, "pieces": [], "target": None, "live": [], "dflash": None}
    exec(compile(module, inspect.getfile(MultiDecoder), "exec"), namespace)
    return namespace["target_boundary"]


def reaches(root, expected, seen=None):
    if root is expected:
        return True
    if root is None:
        return False
    seen = set() if seen is None else seen
    if id(root) in seen:
        return False
    seen.add(id(root))
    children = list(root.exceptions) if isinstance(root, BaseExceptionGroup) else []
    children.extend((root.__cause__, root.__context__))
    return any(reaches(item, expected, seen) for item in children)


@pytest.mark.parametrize("operation_fails", [False, True])
@pytest.mark.parametrize("seal_fails", [False, True])
@pytest.mark.parametrize("cancel", [False, True])
def test_actual_target_try_preserves_operation_and_seal_failure_roots(monkeypatch, operation_fails, seal_fails, cancel):
    primary = KeyboardInterrupt("cancelled target") if cancel else ValueError("failed target")
    prior = RuntimeError("original target cause")
    primary.__cause__ = prior
    secondary = OSError("event record failed")
    owner = object.__new__(GDNStateBatch)
    owner.used, owner.pending, owner.error = True, False, None
    owner.stream, owner.device = 17, torch.device("cuda:0")
    owner.finish_errors = [None]
    calls = []
    def record(_stream):
        calls.append("record")
        if seal_fails:
            raise secondary
    owner.completed = NS(record=record)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda _device: NS(cuda_stream=17))
    decoder = NS(w=NS(meta={}), graphs=None, controller=None,
                 buf=NS(gdn_state_batch=owner, gdn_tables=object(), attn_step=object()))
    def compute(*_args, **_kwargs):
        calls.append("target")
        if operation_fails:
            raise primary
        return object()
    call = boundary(compute)
    expected = primary if operation_fails else secondary if seal_fails else None
    if expected is None:
        call(decoder, [])
    else:
        with pytest.raises(BaseException) as caught:
            call(decoder, [])
        assert caught.value is expected
        if operation_fails:
            assert reaches(caught.value, prior)
        if operation_fails and seal_fails:
            assert reaches(caught.value, secondary)
            assert owner.finish_errors[0] is secondary
    assert calls == ["target", "record"]
    assert decoder.buf.gdn_tables is None and decoder.buf.attn_step is None
    assert owner.error is secondary if seal_fails else owner.error is None
