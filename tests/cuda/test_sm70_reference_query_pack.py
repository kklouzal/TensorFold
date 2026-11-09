"""SM70 exact copy/allocation/stream-lifetime controls, not P3 measurements."""
import pytest
import torch

from tests.test_sm70_reference_query_pack import query_slab, oracle
from tensorfold.families.qwen4_exp.cuda.history_owner import _reference_query_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires the actual pinned SM70 runtime')


@pytest.mark.parametrize('rows', [1, 2, 3, 5, 40, 160, 256])
@pytest.mark.parametrize('group', [0, 4])
def test_actual_sm70_bits_no_allocation_and_next_gather_lifetime(rows, group):
    query = query_slab(rows, 'cuda')
    before = query.view(torch.int16).clone()
    expected = oracle(query, group).view(torch.int16)
    integer_expected = oracle(before, group)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    allocated = torch.cuda.memory_allocated()
    actual = _reference_query_rows(query, group)
    assert torch.cuda.memory_allocated() == allocated
    assert torch.cuda.max_memory_allocated() == allocated
    assert torch.equal(actual.view(torch.int16), expected)
    assert torch.equal(actual.view(torch.int16), integer_expected)
    assert torch.equal(query[group:group + 4].view(torch.int16), before[group:group + 4])
    assert actual.stride() == (3072, 256, 1)
    assert all(actual[row].data_ptr() % 512 == 0 for row in range(rows))
    # Model consumers issue on the producing stream, before its next gather.
    consumed = torch.empty_like(integer_expected)
    consumed.copy_(actual.view(torch.int16))
    query.view(torch.int16).fill_(123)
    assert torch.equal(consumed, expected)
    if rows == 1:
        assert actual.data_ptr() == query[group].data_ptr()
    else:
        assert actual.data_ptr() == query[4 - group].data_ptr()


def test_actual_sm70_explicit_producer_stream_orders_all_views_before_reuse():
    query = query_slab(40, 'cuda')
    expected = oracle(query, 4).view(torch.int16)
    producer = torch.cuda.Stream()
    producer.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(producer):
        packed = _reference_query_rows(query, 4)
        consumed = torch.empty_like(expected)
        consumed.copy_(packed.view(torch.int16))
        query.zero_()  # Same stream's subsequent gather/producer writer.
    torch.cuda.current_stream().wait_stream(producer)
    assert torch.equal(consumed, expected)
