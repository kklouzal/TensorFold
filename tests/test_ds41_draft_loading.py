"""CPU checks for the current target to FT draft binding."""
from dataclasses import replace
import json
import struct

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v4.cuda import dspark_weights as draft
from tensorfold.families.deepseek_v4.cuda.weights import Geometry


def test_ft_layout_matches_target_vocabulary_owner_and_trained_taps():
    target = Geometry()
    for rank in range(8):
        layout = draft.ft_layout(target, rank=rank, world=8, vocab_first=rank * 16160)
        assert layout.rows(target.vocab) == slice(rank * 16160, (rank + 1) * 16160)
        assert layout.etp == 4 and len(layout.experts) == 64 and len(layout.expert_rows) == 576
    for invalid in (replace(target, dspark_layers=(36, 37, 38)), replace(target, layers=39)):
        with pytest.raises(ValueError, match="complete 1017 target"):
            draft.ft_layout(invalid, rank=0, world=8, vocab_first=0)
    with pytest.raises(ValueError, match="vocabulary ownership"):
        draft.ft_layout(target, rank=1, world=8, vocab_first=0)
    with pytest.raises(ValueError, match="integers"):
        draft.ft_layout(target, rank=True, world=8, vocab_first=16160)


def test_ft_load_uses_actual_fault_checked_typed_loader(monkeypatch, tmp_path):
    expected = object()
    observed = []
    (tmp_path / "ft").mkdir()
    (tmp_path / "official").mkdir()

    def loader(directory, **options):
        observed.append((directory, options))
        return expected

    monkeypatch.setattr(draft, "load", loader)
    device = torch.device("cpu")
    assert draft.load_ft(tmp_path, Geometry(), rank=3, world=8, vocab_first=48480, device=device) is expected
    assert observed == [(tmp_path / "ft", {"layout": draft.Layout(3, 8, 4), "device": device,
                                         "config_dir": tmp_path / "official", "keep_requestwise_fp32": True})]


@pytest.mark.parametrize("shape", ([2], ["2"], [False], [-2]))
def test_draft_safetensors_metadata_requires_integer_extent(tmp_path, shape):
    name = "mtp.0.synthetic"
    filename = "held.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {name: filename}}))
    meta = json.dumps({name: {"dtype": "I8", "shape": shape, "data_offsets": [0, 2]}}).encode()
    (tmp_path / filename).write_bytes(struct.pack("<Q", len(meta)) + meta + bytes([3, 8]))
    if shape == [2]:
        reader = draft.Checkpoint(tmp_path)
        assert reader.shape(name) == (2,) and reader.array(name).tolist() == [3, 8]
    else:
        with pytest.raises(ValueError, match="invalid tensor extent"):
            draft.Checkpoint(tmp_path)


def test_draft_checkpoint_rejects_out_of_bundle_shard(tmp_path):
    local = tmp_path / "model"
    local.mkdir()
    filename = tmp_path / "foreign.safetensors"
    filename.write_bytes(b"before metadata")
    (local / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.0.synthetic": str(filename)}}))
    with pytest.raises(ValueError, match="within the supplied bundle"):
        draft.Checkpoint(local)
