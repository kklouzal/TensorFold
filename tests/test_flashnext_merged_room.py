"""Flash Next merged TP rounds: the prompt rows a pass keeps beside a round's verify window (prompt_plan.merged_room),
which MultiDecoder._round uses both to decide whether the pass runs alone first and to size the merged pass."""

import pytest

from tensorfold.families.qwen4_exp.cuda.prompt_plan import merged_room

DEPTH = 4                                              # MTP4: (depth + 1) verify rows a live reply


@pytest.mark.parametrize("pass_rows, window, room", [
    (1024, 40, 984),                                   # prod default: 8 replies x 5 rows ride a 1,024-row pass
    (1024, 160, 864),                                  # 32 replies: the widest window still leaves prompt rows
    (160, 160, 0),                                     # quota equal to the window: no prompt row would run
    (128, 160, 0),                                     # quota below it (administered prefill_tokens)
])
def test_room_is_the_quota_beside_the_window(pass_rows, window, room):
    assert merged_room(pass_rows, window) == room


@pytest.mark.parametrize("pass_rows, live", [(1024, 32), (256, 8), (161, 32), (160, 32), (40, 8), (16, 8)])
def test_a_merging_round_always_has_room_for_its_staged_window(pass_rows, live):
    """The round merges only when the widest window leaves room (else the pass runs alone first, so prompts keep
    filling while replies decode); any staged window is at most that wide, so a merging round never starves."""

    widest = (DEPTH + 1) * live
    alone = not merged_room(pass_rows, widest)
    assert alone == (pass_rows <= widest)
    if not alone:
        assert all(merged_room(pass_rows, rows) > 0 for rows in range(live, widest + 1))
