"""Independent Transformers tower oracle; CPU tests do not qualify CUDA/model execution."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers.models.gemma4.modeling_gemma4")

from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig  # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import (  # noqa: E402
    Gemma4VisionModel, Gemma4VisionPooler, Gemma4RMSNorm, apply_multidimensional_rope,
)

from tensorfold.families.gemma4.cuda.vision import (  # noqa: E402
    GemmaCudaVision, PreparedGemmaVision, _frame, _rope,
)
from test_gemma_cuda_vision_contracts import configuration, patches  # noqa: E402


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_tower_matches_independent_transformers_spatial_rope_pooling_and_standardization(dtype):
    torch.manual_seed(149)
    raw = configuration()
    config = Gemma4VisionConfig(**raw["vision_config"])
    # These are distinct arithmetic regions: eager BF16 rounds attention scores
    # and probabilities to BF16, whereas SDPA retains its supported intermediate
    # precision. Production uses SDPA, so compare BF16 against that same pinned
    # Transformers policy. FP32 retains the independently materialized oracle.
    config._attn_implementation = "sdpa" if dtype == torch.bfloat16 else "eager"
    reference = Gemma4VisionModel(config).eval()
    # inv_freq is derived in FP32 by the pinned constructor, not a checkpoint
    # weight. A blanket .to(BF16) otherwise quantizes that nonpersistent buffer
    # (0.1 becomes0.10009765625), creating a different rotary arithmetic REGION.
    inverse = reference.encoder.rotary_emb.inv_freq.clone()
    reference.to(dtype=dtype)
    reference.encoder.rotary_emb.register_buffer("inv_freq", inverse, persistent=False)
    reference.encoder.rotary_emb.register_buffer("original_inv_freq", inverse.clone(), persistent=False)
    reference.std_bias.copy_(torch.linspace(-0.4, 0.3, 16))
    reference.std_scale.copy_(torch.linspace(0.5, 1.5, 16))
    projector = torch.nn.Linear(16, 24, bias=False).eval().to(dtype=dtype)
    weights = {"vision_tower." + key: value for key, value in reference.state_dict().items()}
    native = GemmaCudaVision(raw, weights, projector)
    assert reference.encoder.rotary_emb.inv_freq.dtype == native.inverse.dtype == torch.float32
    torch.testing.assert_close(native.inverse, reference.encoder.rotary_emb.inv_freq, rtol=0, atol=0)
    p, xy = patches()
    frame = _frame(p, xy, raw["vision_config"], "image")
    prepared = PreparedGemmaVision((1, 90, 90, 2), (frame,), ((1, 3),), ("test",), ())
    with torch.inference_mode():
        positions = torch.tensor(xy)[None]
        probe = torch.linspace(-2, 2, 16 * 2 * 8).reshape(1, 16, 2, 8).to(dtype)
        cos, sin = reference.encoder.rotary_emb(probe, positions)
        expected_phase = apply_multidimensional_rope(probe, cos, sin, positions)
        torch.testing.assert_close(_rope(probe, positions, native.inverse), expected_phase, rtol=0, atol=0)
        result = reference(torch.tensor(p)[None], pixel_position_ids=torch.tensor(xy)[None]).last_hidden_state
        expected = projector(Gemma4RMSNorm(16, eps=1e-6, with_scale=False)(result))
        actual = native.encode(prepared, prepared.token_ids)
    torch.testing.assert_close(actual.features, expected)
    assert actual.rows == (1, 2) and actual.blocks == ((1, 3),)
    assert torch.isfinite(actual.features).all()


def test_requests_never_alias_features_or_mutate_prepared_arrays():
    torch.manual_seed(151)
    raw = configuration()
    config = Gemma4VisionConfig(**raw["vision_config"])
    model = Gemma4VisionModel(config).eval().float()
    model.std_bias.zero_()
    model.std_scale.fill_(1)
    native = GemmaCudaVision(raw, {"vision_tower." + k: v for k, v in model.state_dict().items()},
                             torch.nn.Linear(16, 24, bias=False).eval())
    p, xy = patches()
    frame = _frame(p, xy, raw["vision_config"], "video")
    prepared = PreparedGemmaVision((1, 91, 91, 2), (frame,), ((1, 3),), (), ("video",))
    one = native.encode(prepared, prepared.token_ids)
    two = native.encode(prepared, prepared.token_ids)
    assert one.features.data_ptr() != two.features.data_ptr()
    torch.testing.assert_close(one.features, two.features, rtol=0, atol=0)
    np.testing.assert_array_equal(frame.pixels, p)
    np.testing.assert_array_equal(frame.positions, xy)
    with pytest.raises(ValueError, match="different prompt"):
        native.encode(prepared, (1, 91, 91, 3))


def test_bf16_pooling_keeps_fp32_expansion_and_standardization_before_cancellation():
    """Retain a sensitive reproducer for the superseded BF16 intermediate path."""
    torch.manual_seed(491)
    raw = configuration()
    config = Gemma4VisionConfig(**raw["vision_config"])
    _, xy = patches()
    positions = torch.tensor(xy)[None]
    padding = positions.eq(-1).all(-1)
    hidden = (torch.randn(1, 16, 16) * 300).to(torch.bfloat16)
    bias = torch.linspace(-7, 7, 16).to(torch.bfloat16)
    scale = torch.linspace(0.1, 1.8, 16).to(torch.bfloat16)
    owner = type("PoolOwner", (), {"config": raw["vision_config"], "dtype": torch.bfloat16,
                                    "weights": {"std_bias": bias, "std_scale": scale}})()
    with torch.inference_mode():
        scaled, valid = Gemma4VisionPooler(config)(hidden, positions, padding, output_length=4)
        scaled = scaled[valid]
        expected = ((scaled - bias.float()) * scale.float()).to(torch.bfloat16)
        actual = GemmaCudaVision._pool(owner, hidden, positions, padding)
        # This deliberately wrong historic expression establishes that the
        # fixture detects rounding before cancellation; it is not an oracle.
        legacy = (scaled.to(torch.bfloat16) - bias) * scale
    assert torch.any(expected != legacy)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
