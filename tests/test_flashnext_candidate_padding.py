"""Pruned draft shards retain fixed graph packets without inventing token IDs."""

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.decode import _validate_gathered_padding  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import candidates  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import CAND  # noqa: E402


def _packet(width, *, rows=3):
    logits = torch.arange(rows * width, dtype=torch.float32).reshape(rows, width) / 7
    mapping = torch.arange(width, dtype=torch.int64) * 11 + 5
    buffer = SimpleNamespace(cand=torch.empty(rows, 2 * CAND + 1),
                             cand_all=torch.empty(rows * (2 * CAND + 1)), world=1)
    owner = SimpleNamespace(comm=SimpleNamespace(all_gather=lambda source, dest: dest.copy_(source.flatten())))
    candidates(owner, buffer, logits, rows, id_map=mapping)
    packet = buffer.cand_all.view(1, rows, 2 * CAND + 1).numpy()
    values = packet[:, :, :CAND].copy()
    tokens = packet[:, :, CAND:2 * CAND].copy().view(np.int32).astype(np.int64)
    return logits, mapping, packet, values, tokens


@pytest.mark.parametrize("width", [1, 2, 7, CAND - 1, CAND, CAND + 1, 257])
def test_actual_producer_preserves_scores_ids_and_complete_normalizer(width):
    logits, mapping, packet, values, tokens = _packet(width)
    kept = min(CAND, width)
    expected, indices = logits.topk(kept, sorted=False)
    np.testing.assert_array_equal(values[0, :, :kept], expected.numpy())
    np.testing.assert_array_equal(tokens[0, :, :kept], mapping[indices].numpy())
    assert np.isneginf(values[:, :, kept:]).all()
    assert (tokens[:, :, kept:] == -1).all()
    np.testing.assert_array_equal(packet[0, :, -1], logits.logsumexp(-1).numpy())
    _validate_gathered_padding(values, tokens, SimpleNamespace(widths=[width]))


def test_unequal_shards_keep_only_original_global_ids():
    parts = [_packet(width)[3:] for width in (1, 2, 7, 31)]
    values = np.concatenate([part[0] for part in parts])
    tokens = np.concatenate([part[1] for part in parts])
    _validate_gathered_padding(values, tokens, SimpleNamespace(widths=[1, 2, 7, 31]))
    with pytest.raises(ValueError, match="without a shard owner"):
        _validate_gathered_padding(values, tokens, None)
    with pytest.raises(ValueError, match="rank count"):
        _validate_gathered_padding(values, tokens, SimpleNamespace(widths=[1]))


@pytest.mark.parametrize("change", ["live_negative", "padding_id", "padding_score", "wrong_width"])
def test_malformed_packet_cannot_hide_as_padding(change):
    _logits, _mapping, _raw, values, tokens = _packet(7)
    width = 7
    if change == "live_negative":
        tokens[0, 1, 3] = -1
    elif change == "padding_id":
        tokens[0, 1, 12] = -2
    elif change == "padding_score":
        values[0, 1, 12] = 0
    else:
        width = 8
    with pytest.raises(ValueError, match="immutable vocabulary shard widths"):
        _validate_gathered_padding(values, tokens, SimpleNamespace(widths=[width]))


def test_complete_packet_still_rejects_negative_real_ids_without_a_plan():
    _logits, _mapping, _raw, values, tokens = _packet(CAND)
    _validate_gathered_padding(values, tokens, None)
    tokens[0, 0, 0] = -1
    with pytest.raises(ValueError, match="nonnegative int32"):
        _validate_gathered_padding(values, tokens, None)
