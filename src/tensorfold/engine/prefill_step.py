"""The largest offered prompt chunk that leaves the server's resumable-context floor."""

from __future__ import annotations

from typing import Any, Sequence

# Context a larger chunk must leave room for (or the requested window, if smaller).
CONTEXT_FLOOR = 131072


def choose(make_engine: Any, steps: Sequence[int], budget: int, tokens: Sequence[int], window: int = 0) -> int:
    """Measure eligible chunks; use the server's cache/growth/work terms before increasing the step.

    Larger-probe eligibility scales prior workspace quadratically as a planning
    estimate. MLX's memory limit is a guideline; neither it nor sampled peaks
    proves a hard allocation ceiling. Actual native peak qualification remains
    required. Only a measured chunk leaving the resumable floor replaces a smaller one.
    """
    import mlx.core as mx

    from tensorfold.cleanup import raise_failures
    from tensorfold.engine.family_common import cache_arrays
    from tensorfold.server.memory_budget import PROBE_REPEATS, CacheMemory
    from tensorfold.server.prompt_memory import PromptMemory

    steps = sorted({int(step) for step in steps})
    if not steps or steps[0] <= 0:
        raise ValueError("prefill steps must contain positive token counts")
    small = steps[0]
    if len(steps) == 1:
        return small
    engine = make_engine(small)
    text = [int(token) for token in tokens] or [1000 + i for i in range(small + 64)]
    tighten = getattr(engine.model, "tighten_prefill", None)
    floor = min(int(window) or CONTEXT_FLOOR, CONTEXT_FLOOR)
    memory = PromptMemory(int(budget), engine.model, runtime=mx, overhead_bytes=0)

    def measure(prober: Any, step: int):
        probe = (text * (-(-(step + 64) // len(text))))[:step + 64]
        helds, works, profiles = [], [], []
        for _ in range(PROBE_REPEATS):
            mx.synchronize()
            mx.clear_cache()
            helds.append(int(mx.get_active_memory()))
            mx.reset_peak_memory()
            cache, primary, previous = None, None, None
            try:
                cache = prober.prefill_prefix(probe, cache=None, cached_tokens=0)
                mx.eval(*cache_arrays(cache))
                works.append(max(0, int(mx.get_peak_memory()) - int(mx.get_active_memory())))
                profiles.append(CacheMemory.from_cache(cache))
            except BaseException as error:
                primary = error
                previous = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                            BaseException.__suppress_context__.__get__(error))
            finally:
                cache = None
                cleanup = []
                for owner, name in ((prober, "release_rounds"), (mx, "clear_cache")):
                    try:
                        getattr(owner, name, lambda: None)()
                    except BaseException as error:
                        cleanup.append(error)
                if primary is not None:
                    BaseException.__cause__.__set__(primary, previous[0])
                    BaseException.__context__.__set__(primary, previous[1])
                    BaseException.__suppress_context__.__set__(primary, previous[2])
                raise_failures(primary, cleanup)
        profile = CacheMemory(
            max(item.fixed_bytes for item in profiles), max(item.bytes_per_token for item in profiles),
            max(item.step for item in profiles), max(item.entry_bytes_per_token for item in profiles)
        )
        return max(helds), max(works), profile

    while True:
        best, previous = 0, None
        for step in steps:
            if previous is not None:
                held, work, measured_step = previous
                predicted = held + memory.profile.cache_bytes(step + 64) + work * step * step // (measured_step**2)
                if predicted > memory.budget:
                    break
            held, work, memory.profile = measure(engine if step == small else make_engine(step), step)
            memory.observed_work = work
            if memory.need(floor, held, copies=2) > memory.budget:
                break
            best, previous = step, (held, work, step)
        if best:
            return best
        if tighten is None or not tighten():
            return small


__all__ = ["CONTEXT_FLOOR", "choose"]
