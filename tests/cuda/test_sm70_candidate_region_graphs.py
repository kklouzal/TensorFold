"""Exact packet bits for the original rowwise finish, outside NCCL/status."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.candidate_region_graphs import CandidateRegion
from tensorfold.families.qwen4_exp.cuda.forward import candidates
from tensorfold.families.qwen4_exp.cuda.state import CAND

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason='actual SM70 candidate capture component')


@pytest.mark.parametrize('role,width,mapped', [('target', 31040, False), ('mtp', 31040, False),
                                             ('generic-pruned', 317, True), ('generic-small', 19, True)])
@torch.inference_mode()
def test_r32_changed_logits_original_packet_bits_and_collective_exclusion(role, width, mapped):
    # Actual target/MTP share the248320/8 full head; MTP masks declared support
    # before this region. Odd/pruned maps are separate generic adverse coverage.
    torch.manual_seed(931)
    rows = 32
    logits = torch.randn((rows, width), device='cuda', dtype=torch.float32)
    packet = torch.empty((rows, 2 * CAND + 1), device='cuda')
    gathered = torch.empty((rows * (2 * CAND + 1),), device='cuda')
    calls = []

    def all_gather(local, output):
        calls.append(1)
        output.copy_(local.reshape(-1))

    weights = SimpleNamespace(meta={'execution_provider': 'sm70'},
                              comm=SimpleNamespace(all_gather=all_gather))
    buffers = SimpleNamespace(prefill=False, logits=logits, cand=packet, cand_all=gathered, world=1)
    id_map = torch.arange(width - 1, -1, -1, device='cuda', dtype=torch.int64) * 3 if mapped else None
    offset = 1000 if mapped else 0
    owner = CandidateRegion(weights, buffers, logits, rows, id_map=id_map, offset=offset)
    primary = None
    try:
        owner.prepare()
        assert not calls  # Warm/capture contain local math only.
        for mode in range(4):
            logits.copy_(torch.randn_like(logits))
            if mode == 1:
                logits[:, :min(40, width)] = 4  # Exact top-k boundary ties.
            elif mode == 2:
                logits[:, ::2] = -float('inf')  # Proposal filtering contract.
            elif mode == 3:
                logits.fill_(-float('inf'))  # Packet construction does not sample/validate.
            if role == 'mtp':
                logits[:, 1::3] = -float('inf')
            candidates(weights, buffers, logits, rows, id_map=id_map, offset=offset)
            expected = packet.clone()
            packet.fill_(123)
            owner.replay()
            torch.cuda.synchronize()
            assert len(calls) == mode + 1
            assert torch.equal(packet.view(torch.uint8), expected.view(torch.uint8))
        assert owner.hits == 4
    except BaseException as error:
        primary = error
        raise
    finally:
        from tensorfold.cleanup import raise_failures

        try:
            owner.close()
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
