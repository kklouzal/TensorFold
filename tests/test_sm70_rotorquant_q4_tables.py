"""Generated native tables remain bound to the adopted RotorQuant format."""
import importlib.util
from pathlib import Path

import pytest

from tensorfold.cuda.sm70 import rotorquant_q4_tables as tables
from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref


def test_native_tables_are_exact_authority_materialization():
    tables.verify()
    header = Path(tables.__file__).with_suffix(".cuh").read_text()
    assert ref.descriptor(4, "signed-iso128").codec_id in header
    assert tables._bits(ref.CENTROIDS_4) in header
    assert tables._bits(ref.THRESHOLDS_4) in header
    assert f"0x{ref.SIGN_PAIR_NEGATIVE_MASK:016x}ull" in header


def test_changed_compiler_table_is_rejected(tmp_path):
    path = tmp_path / "tables.cuh"
    path.write_text(tables.render().replace("0xc02ee2bfu", "0xc02ee2beu"))
    with pytest.raises(ValueError, match="canonical format authority"):
        tables.verify(path)


def test_materialization_is_deterministic_and_preserves_q4_descriptor():
    assert tables.render() == tables.render()
    descriptor = ref.descriptor(4, "signed-iso128")
    assert (descriptor.group, descriptor.bits, descriptor.group_bytes, descriptor.scale_dtype,
            descriptor.norm_policy) == (128, 4, 64, "float32", "original-rms")


def test_table_module_does_not_import_accelerator_or_build_at_import(monkeypatch):
    import builtins

    original = builtins.__import__

    def checked(name, *args, **kwargs):
        assert name.split(".")[0] not in ("torch", "triton"), "table generation must remain CPU/stdlib-only"
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked)
    spec = importlib.util.spec_from_file_location("isolated_q4_tables", tables.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.render() == tables.render()
