"""The current target binds the private measured-profile drafter without activation."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v4.cuda import dspark_graph, dspark_head, dspark_serving, dspark_weights, speculation
from tensorfold.families.deepseek_v4.cuda.weights import Geometry


def test_head_borrows_existing_window_rope():
    table = torch.zeros((16385, 64), dtype=torch.float32)
    weights = SimpleNamespace(layout=dspark_weights.Layout(), markov_head=torch.zeros((1, 256)), first_expert=0)
    head = dspark_head.DSparkHead(weights, slots=2, device="cpu", window_rope=table)
    assert head.rope is table
    assert head.ring.shape == (2, dspark_weights.STAGES, dspark_weights.RING, dspark_weights.HEAD_DIM)
    with pytest.raises(ValueError, match="FP32 window RoPE"):
        dspark_head.DSparkHead(weights, slots=2, device="cpu", window_rope=table.half())


def test_prepare_binds_existing_target_and_does_not_install_owner(monkeypatch):
    device = torch.device("cpu")
    geometry = Geometry()
    target = SimpleNamespace(g=geometry, w=SimpleNamespace(embed_first=16160, layers=[None] * geometry.layers),
                             device=device, rotor_cache=object(), pool=SimpleNamespace(owned={}),
                             embed=object(), window_rope=object())
    engine = SimpleNamespace(target=target, context=131072, yarn_factor=16, _closed=False,
                             streams=3, rank=1, world=8, slots=7,
                             scheduler=None, part=None, graphs=object(), coll=object(), _draft_verify=None)
    weights = SimpleNamespace(
        layout=dspark_weights.Layout(1, 8, 4), stages=[SimpleNamespace(index=i) for i in range(3)],
        markov_head=torch.empty((16160, 256), dtype=torch.bfloat16),
        markov_embed=torch.empty((129280, 256), dtype=torch.bfloat16),
    )
    created = {}

    def build(name):
        def factory(*args, **options):
            created[name] = (args, options)
            return name
        return factory

    monkeypatch.setattr(dspark_head, "DSparkHead", build("head"))
    monkeypatch.setattr(speculation, "Speculation", build("transaction"))
    monkeypatch.setattr(dspark_graph, "SpecShape", build("shape"))
    monkeypatch.setattr(dspark_graph, "FixedVerifyOwner", build("owner"))
    prepare = dspark_serving.prepare_128k_verifier

    assert prepare(engine, weights, bucket=3, depth=5) == "owner"
    assert engine._draft_verify is None
    assert created == {
        "head": ((weights,), {"slots": 7, "comm": engine.coll, "device": device,
                              "vocab_first": 16160, "target_embed": target.embed, "window_rope": target.window_rope}),
        "transaction": ((target,), {}),
        "shape": ((engine, "head", "transaction", 3, 5, 131072), {"use_prepared_embedding": True}),
        "owner": (("shape",), {}),
    }
    created.clear()
    engine.context = 1048576
    with pytest.raises(ValueError, match="idle private 128K"):
        prepare(engine, weights, bucket=3, depth=5)
    engine.context = 131072
    engine.scheduler = object()
    with pytest.raises(ValueError, match="idle private 128K"):
        prepare(engine, weights, bucket=3, depth=5)
    engine.scheduler = None
    with pytest.raises(ValueError, match="explicit supported bucket"):
        prepare(engine, weights, bucket=4, depth=5)
    weights.layout = dspark_weights.Layout(0, 8, 4)
    with pytest.raises(ValueError, match="rank's head"):
        prepare(engine, weights, bucket=3, depth=5)
    assert created == {}
