"""SM70 device greedy draft choice against the host gathered chooser (decode.choose_gathered).

The device words are queued by ``issue`` and interpreted by ``take``.

Every case compares the device words' interpretation with the original host
rule on the same packet: chosen token, its FP64 confidence (bitwise), the same
error in the same precedence, or the same full-source fallback for a possibly
truncated tie.
"""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.decode import choose_gathered
from tensorfold.families.qwen4_exp.cuda.state import CAND

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="needs an SM70 GPU")

WORLD = 8


def packet(rows, widths, rng, *, ties=False):
    width = 2 * CAND + 1
    host = np.zeros((WORLD, rows, width), dtype=np.float32)
    for rank, size in enumerate(widths):
        kept = min(CAND, size)
        for row in range(rows):
            values = rng.standard_normal(CAND).astype(np.float32) * 4
            if ties:
                values = np.round(values).astype(np.float32)
            values[kept:] = -np.inf
            ids = np.full(CAND, -1, dtype=np.int32)
            ids[:kept] = rank * 1000 + rng.permutation(1000)[:kept]
            host[rank, row, :CAND] = values
            host[rank, row, CAND:2 * CAND] = ids.view(np.float32)
            host[rank, row, 2 * CAND] = np.float32(rng.standard_normal() + 8)
    return host


def device_choice(host, widths, rows):
    from tensorfold.families.qwen4_exp.cuda.draft_pick import DraftPick

    w = NS(meta={"world": WORLD}, device=torch.device("cuda"))
    pick = DraftPick(w, NS(widths=widths), 32)
    cand = torch.from_numpy(host.reshape(-1).copy()).cuda()
    pick.issue(cand, rows)
    try:
        return pick.take(cand, rows)
    except ValueError as error:
        return ("error", str(error))


def host_choice(host, widths, rows):
    w = NS(meta={"world": WORLD, "vocab_offset": 0})
    plan = NS(widths=widths, graph_ids_safe=True)
    cand = torch.from_numpy(host.reshape(-1).copy()).cuda()
    try:
        chosen, probabilities = choose_gathered(w, cand, rows, list(range(rows)), None, with_prob=True, plan=plan)
        return list(zip(chosen, probabilities))
    except ValueError as error:
        if "ambiguous gathered cutoff" in str(error):
            return None
        return ("error", str(error))


@pytest.mark.parametrize("seed", range(40))
@pytest.mark.parametrize("rows", [1, 3, 24])
def test_random_packets_match_host_rule(seed, rows):
    rng = np.random.default_rng(seed)
    widths = [int(rng.choice([20, 32, 5000])) for _ in range(WORLD)]
    host = packet(rows, widths, rng, ties=seed % 2 == 0)
    if seed % 5 == 1:                       # exact maximum tie across ranks: lowest global ID wins
        host[3, 0, 0] = host[6, 0, 0] = np.float32(50)
    if seed % 5 == 2:                       # an incomplete shard's edge reaches the cutoff: full-source repair
        rank = next((r for r, size in enumerate(widths) if size > CAND), None)
        if rank is not None:
            host[rank, 0, :CAND] = np.float32(60)
    expected = host_choice(host, widths, rows)
    assert device_choice(host, widths, rows) == expected


@pytest.mark.parametrize("fault", ["padding_id", "padding_score", "nan", "posinf", "norm_nan", "norm_none",
                                   "duplicate", "masked", "nan_and_padding"])
def test_faults_raise_the_host_error_in_host_precedence(fault):
    rng = np.random.default_rng(7)
    widths = [20] + [5000] * 7
    host = packet(2, widths, rng)
    if fault in ("padding_id", "nan_and_padding"):
        host[0, 1, CAND + 25] = np.array([3], dtype=np.int32).view(np.float32)[0]
    if fault == "padding_score":
        host[0, 1, 25] = np.float32(1)
    if fault in ("nan", "nan_and_padding"):
        host[4, 0, 3] = np.nan
    if fault == "posinf":
        host[4, 1, 3] = np.inf
    if fault == "norm_nan":
        host[2, 0, 2 * CAND] = np.nan
    if fault == "norm_none":
        host[:, 1, 2 * CAND] = -np.inf
    if fault == "duplicate":
        host[5, 1, CAND + 4] = host[6, 1, CAND + 7]
    if fault == "masked":
        host[:, 0, :CAND] = -np.inf
        host[0, 0, 20:CAND] = -np.inf
    expected = host_choice(host, widths, 2)
    assert type(expected) is tuple and expected[0] == "error"
    assert device_choice(host, widths, 2) == expected


def test_take_requires_its_queued_packet_and_rows():
    rng = np.random.default_rng(3)
    widths = [5000] * WORLD
    from tensorfold.families.qwen4_exp.cuda.draft_pick import DraftPick

    pick = DraftPick(NS(meta={"world": WORLD}, device=torch.device("cuda")), NS(widths=widths), 32)
    cand = torch.from_numpy(packet(2, widths, rng).reshape(-1).copy()).cuda()
    other = cand.clone()
    pick.issue(cand, 2)
    assert pick.take(cand, 1) is False                 # a different row count never reads the queued words
    pick.issue(cand, 2)
    assert pick.take(other, 2) is False                # nor a different packet
    pick.issue(cand, 2)
    assert pick.take(cand, 2) == host_choice(packet(2, widths, np.random.default_rng(3)), widths, 2)
    assert pick.take(cand, 2) is False                 # consumed once
