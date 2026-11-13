"""CPU packet/original-path controls; CUDA bit identity is a separate gate."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.forward import candidates
from tensorfold.families.qwen4_exp.cuda.row_sampling import POLICY, RowSamplingScratch, workspace_bytes
from tensorfold.families.qwen4_exp.cuda.state import CAND
from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation


class Buffer:
    pass


class Gather:
    world = 8
    def __init__(self):
        self.calls = 0

    def all_gather(self, send, receive):
        self.calls += 1
        for rank in range(self.world):
            receive.view(self.world, *send.shape)[rank].copy_(send)


def fixture(width, rows, *, selected, role='decode', mapping=None, device='cpu'):
    w = NS(meta={'execution_provider': 'sm70', 'vocab_offset': 320}, comm=Gather(),
           head=NS(n=width), draft_head=None if mapping is None else NS(n=len(mapping)), draft_ids=mapping)
    b = Buffer()
    b.rows, b.owner_role, b.world, b.execution_dtype = rows, role, 8, torch.float16
    b.logits = torch.empty((rows, width), dtype=torch.float32, device=device)
    b.cand = torch.empty((rows, 2 * CAND + 1), dtype=torch.float32, device=device)
    b.cand_all = torch.empty(rows * 8 * (2 * CAND + 1), dtype=torch.float32, device=device)
    b.row_sampling_scratch = None
    if selected:
        scratch = RowSamplingScratch(width, rows, role, device)
        scratch.bind(w, b)
        b.row_sampling_scratch = scratch
    return w, b


@pytest.mark.parametrize('width', [1, 7, 31, 32, 33, 65, 127])
@pytest.mark.parametrize('rows,first', [(1, 0), (3, 1), (7, 2)])
def test_complete_packet_bits_match_original_for_all_row_offsets_ties_and_padding(width, rows, first):
    w, b = fixture(width, rows + first, selected=True)
    refw, ref = fixture(width, rows + first, selected=False)
    values = ((torch.arange(b.logits.numel()) % 17).float() - 8).view_as(b.logits) / 8
    values[:, 0] = -0.0
    if width > 1:
        values[:, 1] = 0.0
    b.logits.copy_(values)
    ref.logits.copy_(values)
    versions = b.cand._version, b.cand_all._version
    original_versions = ref.cand._version, ref.cand_all._version
    candidates(refw, ref, ref.logits[first:first + rows], rows, offset=320)
    candidates(w, b, b.logits[first:first + rows], rows, offset=320)
    assert torch.equal(b.cand[:rows].view(torch.int32), ref.cand[:rows].view(torch.int32))
    assert torch.equal(b.cand_all[:rows * 8 * (2 * CAND + 1)].view(torch.int32),
                       ref.cand_all[:rows * 8 * (2 * CAND + 1)].view(torch.int32))
    assert w.comm.calls == refw.comm.calls == 1
    assert (b.cand._version - versions[0], b.cand_all._version - versions[1]) == (
        ref.cand._version - original_versions[0], ref.cand_all._version - original_versions[1])
    assert b.row_sampling_scratch.score.data_ptr() % 512 == 0
    assert b.row_sampling_scratch.payload_bytes == workspace_bytes(width)


def test_mapped_compact_draft_rows_keep_global_ids_and_original_packet_bits():
    mapping = torch.tensor([991, 47, 82, 13, 712], dtype=torch.int64)
    w, b = fixture(65, 7, selected=True, role='mtp', mapping=mapping)
    refw, ref = fixture(65, 7, selected=False, role='mtp', mapping=mapping)
    values = torch.tensor([[1., 2., 2., 0., -3.], [0., -0., -1., -2., -3.], [-4., 3., 1., 3., 2.]])
    source = b.logits.view(-1)[:35].view(7, 5)
    original = ref.logits.view(-1)[:35].view(7, 5)
    source[2:5].copy_(values)
    original[2:5].copy_(values)
    candidates(refw, ref, original[2:5], 3, id_map=mapping, offset=320)
    candidates(w, b, source[2:5], 3, id_map=mapping, offset=320)
    assert torch.equal(b.cand[:3].view(torch.int32), ref.cand[:3].view(torch.int32))
    assert (b.cand[:3, CAND + 5:2 * CAND].view(torch.int32) == -1).all()


@pytest.mark.parametrize('value', [float('-inf'), float('inf'), float('nan')])
def test_nonfinite_packets_are_not_silently_repaired_or_validated_early(value):
    w, b = fixture(65, 3, selected=True)
    refw, ref = fixture(65, 3, selected=False)
    b.logits.fill_(value)
    ref.logits.fill_(value)
    candidates(refw, ref, ref.logits, 3, offset=320)
    candidates(w, b, b.logits, 3, offset=320)
    assert torch.equal(b.cand.view(torch.int32), ref.cand.view(torch.int32))
    assert w.comm.calls == refw.comm.calls == 1  # Existing later score/status authority remains.


def test_repeated_rows_reuse_all_payload_owners_and_write_packet_views_directly(monkeypatch):
    w, b = fixture(65, 5, selected=True)
    scratch = b.row_sampling_scratch
    original_topk, original_lse = torch.topk, torch.logsumexp
    packets = []
    def topk(scores, *args, out, **kwargs):
        assert scores.shape == (1, 65) and scores.stride() == (65, 1)
        assert scores.data_ptr() == scratch.score.data_ptr()
        packets.append(out[0].data_ptr())
        return original_topk(scores, *args, out=out, **kwargs)
    def lse(scores, *args, out, **kwargs):
        assert out.shape == (1, 1) and out.stride() == (1, 1)
        return original_lse(scores, *args, out=out, **kwargs)
    monkeypatch.setattr(torch, 'topk', topk)
    monkeypatch.setattr(torch, 'logsumexp', lse)
    owners = [t.data_ptr() for t in scratch.tensors]
    for count in (5, 1, 3):
        b.logits.fill_(count)
        candidates(w, b, b.logits[:count], count, offset=320)
        assert owners == [t.data_ptr() for t in scratch.tensors]
    assert packets == [b.cand[row].data_ptr() for count in (5, 1, 3) for row in range(count)]


def test_storage_map_and_role_replacement_are_rejected_before_packet_writes():
    mapping = torch.arange(65, dtype=torch.int64)
    w, b = fixture(65, 2, selected=True, role='mtp', mapping=mapping)
    b.cand.fill_(123)
    mapping[0] = 99
    with pytest.raises(ValueError, match='ID map changed'):
        candidates(w, b, b.logits, 2, id_map=mapping, offset=320)
    assert (b.cand == 123).all() and w.comm.calls == 0
    w2, b2 = fixture(65, 2, selected=True)
    old = b2.logits
    b2.logits = old.clone()
    with pytest.raises(RuntimeError, match='original'):
        candidates(w2, b2, b2.logits, 2, offset=320)
    assert b2.row_sampling_scratch.anchors[0] is old


def test_unowned_input_autograd_and_closed_owner_do_not_fall_back():
    w, b = fixture(65, 2, selected=True)
    with pytest.raises(ValueError, match='retained logical'):
        candidates(w, b, b.logits.clone(), 2, offset=320)
    b.logits.requires_grad_(True)
    with pytest.raises(ValueError, match='autograd'):
        candidates(w, b, b.logits, 2, offset=320)
    b.logits.requires_grad_(False)
    b.row_sampling_scratch.close()
    with pytest.raises(RuntimeError, match='original live'):
        candidates(w, b, b.logits, 2, offset=320)
    assert w.comm.calls == 0


def test_selection_requires_its_explicit_private_policy():
    base = dict(kernel_family='sm70-paged-owner-rn-ex2-p2051-v1', graph_family='uncaptured',
                graph_budget_bytes=0, rn_heads_per_block=12, trial=True)
    assert parse_preparation(base) == parse_preparation({**base, 'row_sampling_scratch': None})
    assert parse_preparation({**base, 'row_sampling_scratch': POLICY})['row_sampling_scratch'] == POLICY
    for invalid in (True, False, 1, 'reuse'):
        with pytest.raises(ValueError, match='row sampling'):
            parse_preparation({**base, 'row_sampling_scratch': invalid})
    with pytest.raises(ValueError, match='private trial'):
        parse_preparation({**base, 'trial': False, 'row_sampling_scratch': POLICY})
