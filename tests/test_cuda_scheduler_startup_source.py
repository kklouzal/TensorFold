"""Execute the actual constructor startup statements without numerical SDKs."""

import ast
import copy
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cleanup import rollback


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3_5", "qwen3_5_moe")


def publication(family, scheduler_type):
    path = ROOT / "src/tensorfold/families" / family / "cuda/engine.py"
    tree = ast.parse(path.read_text())
    constructor = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    startup_scope = constructor
    if family == "qwen3_5":
        calls = [node for node in ast.walk(constructor) if isinstance(node, ast.Call)
                 and ast.unparse(node.func) == "self._prepare_serving"]
        assert len(calls) == 1
        assert ast.unparse(calls[0]) == "self._prepare_serving(model_dir, streams=streams, keep=keep)"
        startup_scope = next(node for node in ast.walk(tree)
                             if isinstance(node, ast.FunctionDef) and node.name == "_prepare_serving")
    blocks = (node.body for node in ast.walk(startup_scope) if isinstance(getattr(node, "body", None), list))
    candidates = [(body[index], body[index + 1]) for body in blocks for index, node in enumerate(body[:-1])
                  if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                  and isinstance(node.value.func, ast.Name) and node.value.func.id == "Scheduler"]
    assert len(candidates) == 1
    owned, startup = candidates[0]
    assert ast.unparse(owned.targets[0]) == "self.scheduler"
    assert isinstance(startup, ast.Try) and ast.unparse(startup.body[0]) == "self.scheduler.start()"
    wrapper = ast.parse("def publish(self, streams):\n    pass\n").body[0]
    wrapper.body = [owned, startup]
    unit = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(unit)
    namespace = {"Scheduler": scheduler_type}
    exec(compile(unit, str(path), "exec"), namespace)
    return namespace["publish"]


class Opaque(BaseException):
    def __str__(self):
        raise AssertionError("foreign error formatted")

    def add_note(self, note):
        raise AssertionError("foreign note hook called")


def test_both_actual_constructor_startup_scopes_publish_before_start_and_preserve_failures():
    for family in FAMILIES:
        for fail_start, fail_close, identical in ((False, False, False), (True, False, False),
                                                 (True, True, False), (True, True, True)):
            calls = []
            owner = SimpleNamespace(multi=object(), scheduler=None)
            primary, cleanup = Opaque(), Opaque()
            if identical:
                cleanup = primary

            class Owned:
                def __init__(self, decoder, *, max_streams):
                    assert decoder is owner.multi and max_streams == 3
                    calls.append("construct")

                def start(self):
                    assert owner.scheduler is self
                    calls.append("start")
                    if fail_start:
                        raise primary

                def close(self):
                    assert owner.scheduler is self
                    calls.append("close")
                    if fail_close:
                        raise cleanup

            invoke = publication(family, Owned)
            try:
                invoke(owner, 3)
            except BaseException as caught:
                assert fail_start and caught is primary
                assert caught.__cause__ is (cleanup if fail_close and not identical else None)
            else:
                assert not fail_start
            assert calls == (["construct", "start", "close"] if fail_start else ["construct", "start"])
            assert isinstance(owner.scheduler, Owned)


def test_actual_flashnext_constructor_cleanup_does_not_self_chain_the_startup_failure():
    path = ROOT / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashNextEngine")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    handlers = [node for node in ast.walk(constructor) if isinstance(node, ast.ExceptHandler) and node.name == "primary"]
    assert len(handlers) == 1
    wrapper = ast.parse("def construct(self):\n    try:\n        raise initial\n    except BaseException as primary:\n        pass\n").body[0]
    wrapper.body[0].handlers = handlers
    unit = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(unit)
    primary, calls = Opaque(), []

    def close():
        calls.append("close")
        raise primary

    namespace = {"initial": primary, "rollback": rollback}
    exec(compile(unit, str(path), "exec"), namespace)
    try:
        namespace["construct"](SimpleNamespace(close=close, _rollback_startup=close))
    except BaseException as caught:
        assert caught is primary and caught.__cause__ is None
    else:
        raise AssertionError("actual FlashNext constructor hid the startup failure")
    assert calls == ["close"]


def test_actual_flashnext_constructor_warms_before_starting_its_published_actor():
    path = ROOT / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashNextEngine")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_initialize")
    calls = [node for node in ast.walk(constructor) if isinstance(node, ast.Call)]
    starts = [node for node in calls if ast.unparse(node.func) == "self.scheduler.start"]
    warmups = [node for node in calls if ast.unparse(node.func) in {"self.multi.warm", "self.vision.warm"}]
    publications = [node for node in ast.walk(constructor) if isinstance(node, ast.Assign)
                    and any(ast.unparse(target) == "self.scheduler" for target in node.targets)
                    and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == "Scheduler"]
    assert len(starts) == len(publications) == 1 and warmups
    assert max(node.lineno for node in warmups) < publications[0].lineno < starts[0].lineno
    adapters = [node for node in calls if ast.unparse(node.func) == "TensorParallelDecoder"]
    assert len(adapters) == 1 and adapters[0].lineno < min(node.lineno for node in warmups)
    public = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    scope = next(node for node in ast.walk(public) if isinstance(node, ast.Try)
                 and any(handler.name == "primary" for handler in node.handlers))
    assert ast.unparse(scope.body[0].value.func) == "self._initialize"
    final = constructor.body[-1]
    assert isinstance(final, ast.If) and ast.unparse(final.test) == "self.scheduler is not None and (not managed)"
    assert ast.unparse(final.body[0]) == "self.scheduler.start()"


@pytest.mark.parametrize("policy", ["legacy", "fifo"])
@pytest.mark.parametrize("rank", [None, 0, 1], ids=["single-device", "tp-leader", "tp-follower"])
def test_actual_flashnext_startup_statements_defer_public_admission_until_warmup(policy, rank):
    path = ROOT / "src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashNextEngine")
    constructor = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    initialized = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_initialize")
    scope = next(node for node in ast.walk(constructor) if isinstance(node, ast.Try)
                 and any(handler.name == "primary" for handler in node.handlers))
    wanted = {"self.multi.warm", "self.vision.warm", "self.scheduler.start"}

    def project(statements):
        selected = []
        for statement in statements:
            if (isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Call)
                    and ast.unparse(statement.value.func) == "Scheduler"):
                selected.append(statement)
            elif (isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
                  and ast.unparse(statement.value.func) in wanted):
                selected.append(statement)
            elif isinstance(statement, ast.If):
                body, alternate = project(statement.body), project(statement.orelse)
                if body or alternate:
                    branch = copy.deepcopy(statement)
                    branch.body, branch.orelse = body or [ast.Pass()], alternate
                    selected.append(branch)
        return selected

    # Keep the production publication/warmup/start branches and rollback,
    # replacing numerical stages with a bounded, independently owned decoder.
    wrapper = ast.parse("def startup(self, streams, managed=False):\n    pass\n").body[0]
    wrapper.body = [ast.Try(body=project(initialized.body), handlers=scope.handlers, orelse=[], finalbody=[])]
    unit = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(unit)
    namespace = {"Scheduler": Scheduler, "rollback": rollback}
    exec(compile(unit, str(path), "exec"), namespace)
    entered, release = threading.Event(), threading.Event()
    stages, outcomes = [], []

    class Decoder:
        warmed = False
        admission_policy = "legacy"

        def configure_admission_policy(self, selected):
            assert self.warmed, "public admission changed the constructor's synthetic request"
            self.admission_policy = selected

        def warm(self):
            assert self.admission_policy == "legacy"
            stages.append("warm")
            entered.set()
            assert release.wait(5), "warmup fixture did not release"
            self.warmed = True

        def live(self):
            assert self.warmed, "actor accessed the constructor's unfinished decoder"
            return 0

        def drop(self):
            return []

    owner = SimpleNamespace(concurrent=True, multi=Decoder(), scheduler=None,
                            comm=None if rank is None else object(), rank=rank,
                            _admission_options={"admission_policy": policy,
                                **({"max_waiting_input_bytes": 4096} if policy == "fifo" else {})},
                            vision=SimpleNamespace(warm=lambda: stages.append("vision")))
    # The source projection retains the public Scheduler branches; numerical
    # TP calls are represented by the same already-owned decoder contract.
    owner.tp_decoder = owner.multi
    owner.close = lambda: owner.scheduler.close() if owner.scheduler is not None else None
    owner._rollback_startup = owner.close

    def construct():
        try:
            namespace["startup"](owner, 2)
            outcomes.append("ready")
        except BaseException as error:
            outcomes.append(error)

    worker = threading.Thread(target=construct)
    worker.start()
    try:
        assert entered.wait(5), outcomes
        assert owner.scheduler is None
    finally:
        release.set()
        worker.join(5)
        assert not worker.is_alive(), "constructor source fixture did not complete"
        if owner.scheduler is not None:
            owner.scheduler.close()
    assert outcomes == ["ready"] and stages == ["warm", "vision"]
    assert (owner.scheduler is not None) == (rank in (None, 0))
    assert owner.multi.admission_policy == (policy if rank in (None, 0) else "legacy")


def test_actual_fifo_decoder_still_requires_public_principal_and_enqueue_ticket():
    pytest.importorskip("torch")
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    decoder = object.__new__(MultiDecoder)
    decoder.admission_policy = "fifo"
    decoder.capacity, decoder.depth = 131, 2
    decoder.w = SimpleNamespace(cfg=SimpleNamespace(vocab=128))
    stream = Stream([0], 1)
    with pytest.raises(AttributeError, match="principal"):
        decoder.admit(stream)
    stream.principal = "startup-contract-test"
    with pytest.raises(ValueError, match="enqueue ticket"):
        decoder.admit(stream)
