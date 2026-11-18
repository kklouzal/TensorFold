"""mtp_tokens: the picked IDs of a proposal frame staged before they were known."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.mtp import mtp_tokens


class Event:
    def __init__(self, log):
        self.log = log

    def synchronize(self):
        self.log.append("sync")

    def record(self):
        self.log.append("record")


def buffers(rows=4):
    log = []
    host = torch.full((rows,), -1, dtype=torch.int64)
    return NS(rows=rows, ids_host=host, ids_host_array=host.numpy(), ids=torch.full((rows,), -1, dtype=torch.int64),
              staged=Event(log)), log


def test_publishes_one_id_per_row_under_the_buffer_event():
    b, log = buffers()
    mtp_tokens(b, [(None, 0, 1), (None, 1, 2)], [7, 9])
    assert b.ids[:2].tolist() == [7, 9] and b.ids[2:].tolist() == [-1, -1]
    assert log == ["sync", "record"]                   # host rows reused only after the prior staging's copies


@pytest.mark.parametrize("segs,tokens", [([(None, 0, 2)], [7]), ([], []), ([(None, 0, 5)], [1] * 5)])
def test_refuses_a_count_other_than_the_staged_rows(segs, tokens):
    b, log = buffers()
    with pytest.raises(ValueError):
        mtp_tokens(b, segs, tokens)
    assert log == [] and b.ids.tolist() == [-1] * 4
