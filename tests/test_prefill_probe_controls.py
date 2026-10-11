"""Chunk admission and probe retirement controls; no native model/kernel execution."""

from types import SimpleNamespace

import pytest

from tensorfold.engine import prefill_step
from tests.mlx_host_protocol_fakes import cli_protocol

GIB = 1 << 30


def runtime(core, readings):
    core.get_active_memory = lambda: GIB + readings["fixed"]
    core.get_peak_memory = lambda: GIB + readings["fixed"] + GIB // 2
    core.reset_peak_memory = core.synchronize = core.eval = lambda *args: None
    core.get_cache_memory = lambda: 0
    core.clear_cache = lambda: readings.__setitem__("clears", readings["clears"] + 1)


def test_all_cache_profile_samples_contribute_to_the_resumable_floor(monkeypatch):
    """A first high fixed-cache sample must survive two low samples of the same chunk."""
    readings = {"fixed": 0, "clears": 0}
    samples, probes = [], []
    with cli_protocol(monkeypatch) as core:
        runtime(core, readings)

        def make(step):
            def probe(tokens, **options):
                count = sum(previous == step for previous in samples)
                fixed = 2 * GIB if step == 4096 and count == 0 else 0
                readings["fixed"] = fixed
                samples.append(step)
                probes.append(GIB + fixed + GIB // 2)
                return [SimpleNamespace(state=None, memory_growth=lambda: (fixed, 64))]

            return SimpleNamespace(model=SimpleNamespace(), prefill_prefix=probe,
                                   release_rounds=lambda: readings.__setitem__("fixed", 0))

        monkeypatch.setattr(prefill_step, "CONTEXT_FLOOR", 1000)
        assert prefill_step.choose(make, (4096, 2048), 4 * GIB, [1]) == 2048
    assert samples == [2048] * 3 + [4096] * 3
    assert max(probes) <= 4 * GIB and readings["fixed"] == 0


def test_failed_probe_retires_rounds_and_preserves_primary_with_cleanup_failure(monkeypatch):
    primary, cleanup = RuntimeError("probe failed"), OSError("round retirement failed")
    readings = {"fixed": 0, "clears": 0}
    retired = []
    with cli_protocol(monkeypatch) as core:
        runtime(core, readings)

        def probe(*args, **options):
            raise primary

        def retire():
            retired.append(True)
            raise cleanup

        engine = SimpleNamespace(model=SimpleNamespace(), prefill_prefix=probe, release_rounds=retire)
        with pytest.raises(RuntimeError) as caught:
            prefill_step.choose(lambda step: engine, (4096, 2048), 10 * GIB, [1])
    assert caught.value is primary and primary.__cause__ is cleanup
    assert retired == [True] and readings["clears"] == 2


def test_release_callback_lookup_failure_does_not_hide_the_failed_probe(monkeypatch):
    primary, cleanup = RuntimeError("probe failed"), LookupError("retirement lookup failed")
    readings = {"fixed": 0, "clears": 0}
    with cli_protocol(monkeypatch) as core:
        runtime(core, readings)

        class Engine:
            model = SimpleNamespace()

            def prefill_prefix(self, *args, **options):
                raise primary

            @property
            def release_rounds(self):
                raise cleanup

        with pytest.raises(RuntimeError) as caught:
            prefill_step.choose(lambda step: Engine(), (4096, 2048), 10 * GIB, [1])
    assert caught.value is primary and primary.__cause__ is cleanup
    assert readings["clears"] == 2
