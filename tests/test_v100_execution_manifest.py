"""The tracked SC117 V100 preparation packet and its catalogue rule, CPU only."""

import json
from pathlib import Path

import pytest

from tools.v100_execution_manifest import catalogue

PACKET = Path(__file__).resolve().parents[1] / "deploy/v100/execution-profiles/sc117-tp8etp4-mtp4-p24.preparation.json"


def test_tracked_packet_is_the_rule_catalogue_and_loads_as_preparation():
    from tensorfold.families.qwen4_exp.cuda.exact_graph_policy import DEMAND_POLICY, parse_policy
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation

    packet = json.loads(PACKET.read_text(encoding="utf-8"))
    assert parse_preparation(packet) is not None
    policy = packet["exact_graph_policy"]
    assert policy["policy"] == DEMAND_POLICY
    assert policy["keys"] == catalogue(24, 4) == parse_policy(policy)["keys"]   # sorted as the loader sorts
    assert len(policy["keys"]) == 171


def test_catalogue_rule_keys_fit_depth_plus_one_rows_per_participant():
    keys = catalogue(2, 4)
    assert keys == sorted(keys) and len({tuple(key) for key in keys}) == len(keys)
    assert {tuple(key) for key in keys} == (
        {(role, rows, 1) for rows in range(1, 6) for role in ("verify", "mtp")}
        | {("mtp", 2, 2), ("mtp", 6, 2), ("mtp", 10, 2), ("verify", 6, 2), ("verify", 7, 2), ("verify", 8, 2),
           ("verify", 10, 2)})
    assert all(rows <= 5 * count for _role, rows, count in catalogue(32, 4))
    with pytest.raises(ValueError):
        catalogue(0, 4)
