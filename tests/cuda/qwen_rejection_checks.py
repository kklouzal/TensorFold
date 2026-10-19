"""Real companion checks under the standard rejection contract, separate from serial seeded draw order."""


def checkpoint_controls(engine, prompt, sampling, count=48):
    def reply(draft):
        ids = []
        stats = engine.generate(list(prompt), count, sampling, lambda new: ids.extend(new) and False,
                                draft=draft, stop_eos=False)
        assert len(ids) == count
        return ids, stats

    # Cache state belongs to this test's engine; keep the same cache object and its memory-budget owner.
    engine.cache.entries.clear()
    serial, serial_stats = reply(False)
    name = "propose_chain" if sampling is not None and sampling.temperature > 0 else "propose_tree"
    original = getattr(engine.draft, name)
    called = []

    def observed(*args, **kwargs):
        called.append(True)
        return original(*args, **kwargs)

    setattr(engine.draft, name, observed)
    try:
        drafted, draft_stats = reply(True)
        warm, warm_stats = reply(True)
        engine.cache.entries.clear()
        cold, cold_stats = reply(True)
    finally:
        setattr(engine.draft, name, original)
    assert called and draft_stats["drafted"] > 0
    assert drafted == warm == cold
    assert warm_stats["cached"] == len(prompt) - 1 and cold_stats["cached"] == 0
    if sampling is None or sampling.temperature <= 0:
        assert drafted == serial
        assert draft_stats["rounds"] < serial_stats["rounds"]
    return serial, drafted, serial_stats, draft_stats
