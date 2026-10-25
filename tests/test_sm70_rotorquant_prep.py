"""Fused Q4 preparation compiler closure and immutable codec authority, without CUDA."""
from types import SimpleNamespace

import pytest

from tensorfold.cuda import build
from tensorfold.cuda.sm70 import family_native, provider, rotorquant_q4_tables, source_cache


@pytest.fixture
def compiler(monkeypatch):
    family_native.extension.cache_clear()
    calls = []
    monkeypatch.setattr(provider, "_native", lambda: SimpleNamespace(__name__="qualified_sm70_fixture"))

    def stage(sources, *, headers):
        calls.append({"sources": sources, "headers": headers})
        return [str(source) for source in sources]

    def load(name, sources, **kwargs):
        calls.append({"name": name, "sources": sources, **kwargs})
        return SimpleNamespace(name=name)

    monkeypatch.setattr(source_cache, "stage_sources", stage)
    monkeypatch.setattr(build, "load", load)
    yield calls
    family_native.extension.cache_clear()


def test_preparation_build_declares_shared_q4_basis_and_exact_table_inputs(compiler):
    module = family_native.extension()
    assert [path.name for path in compiler[0]["sources"]] == ["family_bindings.cpp", "family.cu"]
    assert {path.name for path in compiler[0]["headers"]} == {
        "torch_reduce.cuh", "tensor_contracts.cuh", "rotorquant_q4.cuh", "rotorquant_q4_tables.cuh"
    }
    assert compiler[1]["need"] == (7, 0)
    assert compiler[1]["extra_cuda_cflags"] == ["-O3", "--fmad=false"]
    assert compiler[1]["name"] == module.name
    assert family_native.extension() is module
    assert len(compiler) == 2


def test_changed_shared_basis_uses_a_distinct_compilation_identity(compiler, monkeypatch):
    first = family_native.extension().name
    read = family_native.Path.read_bytes

    def changed(path):
        value = read(path)
        return value + b"\n// distinct declared basis input\n" if path.name == "rotorquant_q4.cuh" else value

    monkeypatch.setattr(family_native.Path, "read_bytes", changed)
    family_native.extension.cache_clear()
    assert family_native.extension().name != first


def test_invalid_generated_tables_refuse_before_compilation(compiler, monkeypatch):
    def refuse():
        raise ValueError("codec authority changed")

    def no_native_setup():
        raise AssertionError("invalid tables reached native preparation")

    monkeypatch.setattr(rotorquant_q4_tables, "verify", refuse)
    monkeypatch.setattr(provider, "_native", no_native_setup)
    with pytest.raises(ValueError, match="authority"):
        family_native.extension()
    assert compiler == []
