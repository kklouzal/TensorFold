"""Execute the actual constructor startup statements without numerical SDKs."""

import ast
import copy
from pathlib import Path
import threading
from types import SimpleNamespace

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cleanup import rollback


ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("qwen3_5", "qwen3_5_moe")


def publication(family, scheduler_type):
    path = ROOT / "src/tensorfold/families" / family / "cuda/engine.py"
    tree = ast.parse(path.read_text())
    constructor = next(node for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    blocks = (node.body for node in ast.walk(constructor) if isinstance(getattr(node, "body", None), list))
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
    assert publications[0].lineno < min(node.lineno for node in warmups)
    assert max(node.lineno for node in warmups) < starts[0].lineno
    public = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    scope = next(node for node in ast.walk(public) if isinstance(node, ast.Try)
                 and any(handler.name == "primary" for handler in node.handlers))
    assert ast.unparse(scope.body[0].value.func) == "self._initialize"
    final = constructor.body[-1]
    assert isinstance(final, ast.If) and ast.unparse(final.test) == "self.scheduler is not None"
    assert ast.unparse(final.body[0]) == "self.scheduler.start()"


def test_actual_flashnext_startup_statements_keep_the_actor_unstarted_during_warmup():
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
    wrapper = ast.parse("def startup(self, streams):\n    pass\n").body[0]
    wrapper.body = [ast.Try(body=project(initialized.body), handlers=scope.handlers, orelse=[], finalbody=[])]
    unit = ast.Module(body=[wrapper], type_ignores=[])
    ast.fix_missing_locations(unit)
    namespace = {"Scheduler": Scheduler, "rollback": rollback}
    exec(compile(unit, str(path), "exec"), namespace)
    entered, release = threading.Event(), threading.Event()
    stages, outcomes = [], []

    class Decoder:
        warmed = False

        def warm(self):
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
                            vision=SimpleNamespace(warm=lambda: stages.append("vision")))
    owner.close = lambda: owner.scheduler.close()
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
        assert entered.wait(5)
        assert owner.scheduler is not None
        assert not owner.scheduler._start_attempted
        assert not owner.scheduler._worker_entered.is_set()
    finally:
        release.set()
        worker.join(5)
        assert not worker.is_alive(), "constructor source fixture did not complete"
        if owner.scheduler is not None:
            owner.scheduler.close()
    assert outcomes == ["ready"] and stages == ["warm", "vision"]
