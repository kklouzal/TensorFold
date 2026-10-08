"""Source-backed host-only fixture contracts; no foreign array execution."""
from __future__ import annotations

import importlib.util
from importlib import metadata
from pathlib import Path
import sys
import importlib
from types import ModuleType, SimpleNamespace

import pytest

from tests.mlx_host_protocol_fakes import cli_protocol, owned_release_class


def test_cli_fixture_exposes_only_startup_protocol_and_scoped_version(monkeypatch):
    with cli_protocol(monkeypatch, 'fixture-version') as core:
        assert core.__version__ == metadata.version('mlx-lm') == 'fixture-version'
        assert core.get_active_memory() == core.set_wired_limit(1) == 0
        core.synchronize()
        core.clear_cache()
        with pytest.raises(AttributeError):
            core.array([1])
        with pytest.raises(AttributeError):
            sys.modules['mlx.nn'].silu(1)
        with pytest.raises(metadata.PackageNotFoundError):
            metadata.version('this-package-is-explicitly-absent-for-fixture-contract')


@pytest.mark.parametrize('path,name', [
    ('tensorfold/families/nemotron_h/model.py', 'NemotronH'),
    ('tensorfold/families/qwen4_exp/runtime.py', 'FlashNext'),
])
def test_pure_release_fixture_executes_owned_method_and_releases_all_recorded_rows(path, name):
    family = owned_release_class(path, name)()
    family.fused = SimpleNamespace(row_states={0: object()}, _last_heads=[object()], last_streams=object())
    if name == 'NemotronH':
        family._last_hidden = object()
        family.release_rounds()
        assert family.fused.row_states == {} and family._last_hidden is None
    else:
        family.mtp_fused = SimpleNamespace(row_states={0: object()}, _last_heads=[object()], last_streams=object())
        family.model = SimpleNamespace(last_streams=object())
        family._streams, family._specs = object(), {0: object()}
        family.release_rounds()
        assert family._streams is None and family._specs == {}
        assert not hasattr(family.model, 'last_streams')
        for decoded in (family.fused, family.mtp_fused):
            assert decoded.row_states == {} and decoded._last_heads == [] and decoded.last_streams is None


def test_owned_method_fixture_rejects_source_escape_before_reading_code():
    with pytest.raises(ValueError, match='escaped'):
        owned_release_class('/etc/passwd', 'untrusted')


def test_protocol_teardown_removes_actual_imported_mx_binding_and_parent_alias(monkeypatch):
    import tensorfold

    name = "tensorfold._owned_host_protocol_dflash_probe"
    assert name not in sys.modules and not hasattr(tensorfold, "_owned_host_protocol_dflash_probe")
    source = Path(tensorfold.__file__).parent / "families/qwen3_5/dflash_head.py"
    original_core = ModuleType("mlx.core")
    original_mlx = ModuleType("mlx")
    original_mlx.core = original_core
    monkeypatch.setitem(sys.modules, "mlx", original_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", original_core)
    borrowed_name = "tensorfold._borrowed_host_protocol_marker"
    borrowed = ModuleType(borrowed_name)
    borrowed.mx = original_core
    monkeypatch.setitem(sys.modules, borrowed_name, borrowed)
    with cli_protocol(monkeypatch) as core:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        setattr(tensorfold, "_owned_host_protocol_dflash_probe", module)
        spec.loader.exec_module(module)
        assert module.mx is core
        assert module.DraftSlot(None).nbytes == 0
    assert name not in sys.modules and not hasattr(tensorfold, "_owned_host_protocol_dflash_probe")
    assert sys.modules.get("mlx.core") is original_core
    assert sys.modules.get(borrowed_name) is borrowed and borrowed.mx is original_core


def test_import_block_sentinel_cannot_mask_primary_scope_failure(monkeypatch):
    name = "tensorfold._borrowed_host_protocol_import_block"
    borrowed = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, borrowed)
    primary = RuntimeError("primary host startup failure")
    with pytest.raises(RuntimeError) as failed:
        with cli_protocol(monkeypatch):
            sys.modules[name] = None
            raise primary
    assert failed.value is primary
    assert sys.modules[name] is None  # an independent replacement stays intact


def test_actual_deepseek_kernel_metadata_imports_and_tears_down_scoped_nn(monkeypatch):
    from tensorfold import families

    before_nn = sys.modules.get("mlx.nn")
    before_linear = sys.modules.get("tensorfold.families.glm5_next.linear")
    with cli_protocol(monkeypatch, "metadata-test"):
        name = families.kernel_version(families.families()["deepseek_v4"], None)
        assert name.startswith("deepseek_v4-v1-")
        if before_linear is None:
            assert sys.modules["tensorfold.families.glm5_next.linear"].nn is sys.modules["mlx.nn"]
        else:
            assert sys.modules["tensorfold.families.glm5_next.linear"] is before_linear
    assert sys.modules.get("mlx.nn") is before_nn
    assert sys.modules.get("tensorfold.families.glm5_next.linear") is before_linear


def test_actual_all_family_kernel_metadata_imports_are_scoped_and_numerical_layers_fail_closed(monkeypatch):
    from tensorfold import families

    before = {name: module for name, module in sys.modules.items() if name.startswith("tensorfold.")}
    with cli_protocol(monkeypatch, "all-family-metadata"):
        nn = sys.modules["mlx.nn"]
        for family in families.families().values():
            package = family.package
            if not hasattr(package, "load"):
                continue
            kernels = importlib.import_module(package.KERNEL_PACKAGE)
            assert kernels.VERSION == package.KERNEL_VERSION == "v1"
            if family.model_type == "qwen3_5":
                model = SimpleNamespace(_tensorfold_lanes=True)
                version = families.kernel_version(family, model)
                assert version.startswith(("qwen-dense-v1-", "qwen3_5-v1-"))
            else:
                names = getattr(package, "MODEL_TYPES", (family.model_type,))
                version = families.kernel_version(family, None)
                assert version.startswith(tuple(f"{name}-v1-" for name in names))
        with pytest.raises(AssertionError, match="cannot construct"):
            nn.QuantizedLinear(4, 8)
        row_module = sys.modules["tensorfold.kernels.nemotron.lightning.v1.rows"]
        if row_module is not before.get(row_module.__name__):
            with pytest.raises(AssertionError, match="cannot construct"):
                row_module.RowLinear(4, 8)
    # Every newly imported module binding the scoped base marker was removed.
    for name, module in list(sys.modules.items()):
        if name.startswith("tensorfold.") and module is not before.get(name) and isinstance(module, ModuleType):
            assert all(value is not nn for value in vars(module).values())
