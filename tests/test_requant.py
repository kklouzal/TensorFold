"""Load-time requantization of BF16 weights (tensorfold.cuda.requant): one scale an output row, E4M3 codes."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda.requant import E4M3_MAX, requant_rows  # noqa: E402


def test_rows_scale_to_the_e4m3_range_and_reconstruct_within_its_rounding():
    gen = torch.Generator().manual_seed(3)
    w = (torch.randn(300, 128, generator=gen) * torch.logspace(-4, 1, 300)[:, None]).bfloat16()
    w[7] = 0                                                   # an all-zero row keeps scale 1 and zero codes
    codes, scales = requant_rows(w, chunk=64)                  # several chunks, one partial
    assert codes.dtype == torch.float8_e4m3fn and codes.shape == w.shape and scales.dtype == torch.float32
    top = w.float().abs().amax(1)
    expect = torch.where(top > 0, top / E4M3_MAX, torch.ones_like(top))
    assert torch.equal(scales, expect) and scales[7] == 1 and not codes[7].float().any()
    assert float(codes.float().abs().amax(1).max()) <= E4M3_MAX
    back = codes.float() * scales[:, None]
    # E4M3 keeps 3 mantissa bits: a normal value rounds within 2^-4 relative; subnormals within half a step
    err = (back - w.float()).abs()
    bound = torch.maximum(w.float().abs() * 2.0 ** -4, scales[:, None] * 2.0 ** -10)
    assert bool((err <= bound).all())


def test_codes_are_round_to_nearest_of_the_scaled_row():
    w = torch.tensor([[1.0, -0.3, 0.0, 448.0 / 3], [2.5, 2.5, -5.0, 1e-3]], dtype=torch.bfloat16)
    codes, scales = requant_rows(w)
    scaled = w.float() / scales[:, None]
    np.testing.assert_array_equal(codes.float().numpy(), scaled.to(torch.float8_e4m3fn).float().numpy())


@pytest.mark.parametrize("bad", [torch.zeros(4), torch.zeros(2, 3, dtype=torch.int32),
                                 torch.tensor([[1.0, float("nan")]]), torch.tensor([[1.0, 2.0], [float("inf"), 1.0]]),
                                 torch.tensor([[1.0, 2.0], [-float("inf"), 1.0]]),
                                 torch.tensor([[1.0, 2.0], [1e4, float("nan")]])])   # NaN beside a larger value
def test_malformed_weights_are_refused(bad):
    with pytest.raises(ValueError):
        requant_rows(bad, chunk=1)                             # the non-finite row in the second chunk


def test_accounting_matches_the_copy_and_refuses_unknown_kinds():
    from tensorfold.cuda.requant import resident_bytes, staging_bytes

    assert resident_bytes("fp8", 1000, 256) == 1024 * 256 + 1024 * 4 == resident_bytes("fp8a8", 1000, 256)
    assert staging_bytes("fp8", 262144, 2816) > resident_bytes("fp8", 262144, 2816)
    for call in (resident_bytes, staging_bytes):
        with pytest.raises(ValueError):
            call("int4", 10, 64)


def test_the_target_policy_is_set_once_and_records_what_it_changed():
    """The CLI's set-once policy: unnamed groups load as stored; the first read fixes it (a different policy then
    raises, the same one is accepted); ``unapplied`` names requested groups no load requantized; ``using`` restores
    the previous state."""

    from tensorfold.cuda import requant

    with requant.using():
        assert requant.target_kind("head") is None and requant.unapplied() == []
        requant.set_targets()                                   # the same (empty) policy after a read: accepted
        with pytest.raises(RuntimeError, match="fixed"):
            requant.set_targets(head="fp8")
    with requant.using(head="fp8"):
        requant.set_targets(gdn="fp8")                          # unread: still open
        requant.set_targets(head="fp8")
        assert requant.unapplied() == ["head"]
        assert requant.target_kind("gdn") is None and requant.target_kind("head") == "fp8"
        with pytest.raises(RuntimeError, match="fixed"):
            requant.set_targets(head="fp8", gdn="fp8")
        with requant.using(gdn="fp8"):                          # nested override, then the outer state again
            assert requant.target_kind("head") is None
        assert requant.target_kind("head") == "fp8" and requant.unapplied() == ["head"]
    with pytest.raises(ValueError, match="groups"):
        requant.set_targets(embed="fp8")
    with pytest.raises(ValueError, match="kind"):
        requant.set_targets(head="int8")
    with pytest.raises(ValueError, match="groups"):
        requant.target_kind("embed")


def test_qwen_loaders_requantize_only_the_head_and_the_gdn_projections_the_policy_names(monkeypatch):
    """The Qwen NVFP4 loader's groups (head: lm_head; gdn: in_proj_qkv / in_proj_z / out_proj; never the router or
    in_proj_a/b), their resident bytes (the FP8 copy, not BF16), their load-time staging, and the copy it makes."""

    from tensorfold.cuda import requant
    from tensorfold.families.qwen3_5.cuda import nvfp4_load

    root = "model.language_model.layers.3."
    gdn = [root + "linear_attn." + p for p in ("in_proj_qkv", "in_proj_z", "out_proj")]
    stored = [root + "linear_attn.in_proj_a", root + "linear_attn.in_proj_b", root + "mlp.gate",
              root + "mlp.shared_expert_gate", "model.language_model.embed_tokens"]
    assert [nvfp4_load.requant_group(s) for s in ["lm_head", *gdn, *stored]] == ["head", *["gdn"] * 3, *[None] * 5]
    info = {"dtype": "BF16", "shape": [8192, 2048]}
    for policy in ({}, {"head": "fp8"}, {"gdn": "fp8"}, {"head": "fp8", "gdn": "fp8"}):
        with requant.using(**policy):
            for stem in ["lm_head", *gdn, *stored]:
                kind = policy.get(nvfp4_load.requant_group(stem) or "")
                size = nvfp4_load.weight_bytes(stem + ".weight", dict(info, shape=list(info["shape"])))[0]
                assert size == (requant.resident_bytes(kind, 8192, 2048) if kind else 8192 * 2048 * 2)
                assert nvfp4_load.requant_staging(stem, info) == (requant.staging_bytes(kind, 8192, 2048)
                                                                  if kind else None)
    monkeypatch.setattr(requant, "requantize", lambda weight, kind: ("copy", kind))   # the GPU copy itself: CUDA tests
    with requant.using(gdn="fp8"):
        assert requant.target_copy("head", None) is None and requant.unapplied() == ["gdn"]
        assert requant.target_copy("gdn", None) == ("copy", "fp8") and requant.unapplied() == []
