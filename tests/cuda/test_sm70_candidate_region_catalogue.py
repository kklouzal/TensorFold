"""Actual64-key shared-pool packet oracle; excludes State/NCCL/capacity proof."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.candidate_region_catalogue import Catalogue, PROFILE
from tensorfold.families.qwen4_exp.cuda.forward import candidates_local

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason='actual SM70 shared-pool catalogue oracle')


@torch.inference_mode()
def test_exact64_keys_alternating_roles_rows_original_packet_bits():
    width, rows, device = 31040, 32, torch.device('cuda', torch.cuda.current_device())

    def buffer():
        return SimpleNamespace(prefill=False, rows=rows,
            logits=torch.zeros((rows, width), dtype=torch.float32, device=device),
            cand=torch.empty((rows, 65), dtype=torch.float32, device=device),
            input_preparation_context=SimpleNamespace(fail=lambda error: None))

    weights = SimpleNamespace(meta={'sm70_profile': PROFILE, 'execution_provider': 'sm70',
        'execution_dtype': 'float16', 'history_layout': 'unique', 'world': 8, 'vocab_offset': 0,
        'sm70_candidate_region_graphs_trial': True}, device=device, comm=SimpleNamespace(world=8),
        head=SimpleNamespace(n=width), draft_head=None, draft_ids=None,
        draft_filter=torch.arange(width, device=device) % 3 == 1, mtp=object())
    # These are cold ownership/count guards only. No actual State is accessed,
    # and this fixture does not qualify TP8 or native24-window admission.
    dec = SimpleNamespace(w=weights, streams=[], filling=False, kept=[], free=[object() for _ in range(24)],
        round_tables=SimpleNamespace(slots=24), buf=buffer(), mbuf=buffer())
    charges = []
    parent = SimpleNamespace(dec=dec, allocator_backend='native', side=torch.cuda.Stream(device=device),
        admitted=False, capture_observation=lambda: None,
        charge_capture=lambda before, label: charges.append(label))
    owner = Catalogue(parent)
    primary = None
    try:
        owner.prepare()
        assert len(owner.regions) == len(charges) == 64 and len(owner.pools) == 1
        parent.admitted = True  # Resource accounting itself is a separate integrated owner gate.
        torch.manual_seed(941)
        # Every exactR and both roles, in deliberately different replay order
        # from capture. External packet sentinels expose out-of-range writes.
        for r in (32, 1, 17, *range(31, 17, -1), *range(2, 17), 32):
            for role in ('mtp', 'target'):
                buffers, _, mapping = owner.roles[role]
                buffers.logits.copy_(torch.randn_like(buffers.logits))
                buffers.logits[:r, :40] = 4  # Original unsorted top-k boundary ties.
                if role == 'mtp':
                    buffers.logits[:r].masked_fill_(weights.draft_filter, -float('inf'))
                buffers.cand.fill_(123)
                candidates_local(weights, buffers, buffers.logits[:r], r, id_map=mapping, offset=0)
                expected = buffers.cand.clone()
                buffers.cand.fill_(123)
                assert owner.local_packet(weights, buffers, buffers.logits[:r], r, mapping, 0)
                torch.cuda.synchronize(device)
                assert torch.equal(buffers.cand.view(torch.uint8), expected.view(torch.uint8))
        record = owner.record()
        assert all(record['frames'][role][r] > 0 for role in ('target', 'mtp') for r in range(1, 33))
        assert not getattr(dec, '_memory_error', None)
    except BaseException as error:
        primary = error
        raise
    finally:
        from tensorfold.cleanup import raise_failures

        try:
            owner.close()
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
