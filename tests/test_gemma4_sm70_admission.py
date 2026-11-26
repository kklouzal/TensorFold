from types import SimpleNamespace

import pytest

from tensorfold.cuda.server import App
from tensorfold.families.gemma4.sm70.engine import Gemma4SM70Engine
from tensorfold.families.gemma4.sm70.prefix import PrefixOwner


@pytest.mark.parametrize("block", [1, 3, 16])
@pytest.mark.parametrize("available", [0, 1, 5, 20])
def test_published_context_matches_the_rounded_slot_ledger(block, available):
    engine = Gemma4SM70Engine.__new__(Gemma4SM70Engine)
    engine.block = block
    engine.context_window = 997
    engine.target = SimpleNamespace(alloc=1024, pool=SimpleNamespace(token_bytes=lambda: 13000))
    engine.head = None if block == 1 else SimpleNamespace(token_bytes=lambda: 2600)
    engine.budget = available << 20

    owner = PrefixOwner.__new__(PrefixOwner)
    owner.e = engine
    funded = [rows for rows in range(1, engine.context_window + 1)
              if engine.slot_bytes(owner._cap(rows + block)) <= engine.budget]
    engine.limit = engine._admission_limit()
    assert engine.limit == max(funded, default=0)

    app = App.__new__(App)
    app.engine, app.context_window = engine, engine.context_window
    assert app.max_model_len == engine.limit
