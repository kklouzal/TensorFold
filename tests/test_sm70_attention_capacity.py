"""Native launch storage is independent of the full promised KV window."""

import pytest

from tensorfold.cuda.sm70 import attention_native as native
from tensorfold.cuda.sm70.attention_qualification import REQUIRED_IDENTITY, supports


@pytest.mark.parametrize("capacity", [0, 65, 256, 881, 2731, 8192, 8193, 262144])
def test_retained_allocation_exactly_matches_capped_forecast(capacity):
    owner = native.Workspace(3, 3, 256, capacity, "cpu")
    actual = sum(value.numel() * value.element_size()
                 for value in (owner.scores, owner.probabilities, owner.partials))
    assert actual == native.workspace_bytes(3, 3, 256, capacity)
    assert owner.capacity == min(capacity, 8192)
    assert owner.scores.shape[-1] == (min(capacity, 8192)+31)//32*32


def test_full_kv_admission_no_longer_multiplies_native_workspace_by_future_cache():
    assert native.workspace_bytes(304, 3, 256, 262144) == native.workspace_bytes(304, 3, 256, 8192)
    assert native.workspace_bytes(304, 3, 256, 262144) == 104595456


@pytest.mark.parametrize("capacity", [-1, 8192.0, True])
def test_invalid_capacity_is_rejected_before_allocation(capacity):
    with pytest.raises(ValueError, match="nonnegative integer"):
        native.Workspace(1, 3, 256, capacity, "cpu")


@pytest.mark.parametrize("first,context,allowed", [(65, 8192, True), (65, 8193, False), (8193, 8193, False), (64, 8192, False)])
def test_graph_qualification_still_refuses_outside_true_end_scope(first, context, allowed):
    assert supports(rows=1, heads=3, kv_heads=1, head_dim=256, context=context,
                    first_end=first, identity=REQUIRED_IDENTITY, capability=(7, 0)) is allowed


def test_original_long_torch_fallback_fits_unchanged_two_gib_service_workspace():
    # family_ops.dequantize_int8: the second side retains the first Half side,
    # plus input/add/product Float32 arrays and Float32 group scales. Positions
    # remain live. Raw/mixed formats need no more than this both-int8 case.
    context, dim, qsa_rows, ratio = 262144, 256, 256, 4
    cells, groups = context*dim, context*(dim//32)
    dequant_peak = cells*2 + 3*cells*4 + groups*4 + context*8
    # Pinned ATen SortStable: scores/values/indices/reverse indices plus CUB's
    # key/value alternates total<=40 bytes/block including retained score
    # scratch. SM70 radix metadata fits a deliberately loose512MiB bound.
    blocks = (context+ratio-1)//ratio
    # Only the <128-row full-sort branch needs global radix metadata. Even
    # one histogram per32 items,256 bins and8-byte counters, plus1MiB for
    # pass bins/alignment, stays under the declared loose512MiB allowance.
    metadata_upper = ((127*blocks+31)//32)*256*8 + (1 << 20)
    assert metadata_upper < 512 << 20
    segmented_sort_peak = 40*qsa_rows*blocks + (512 << 20)
    full_sort_peak = 48*127*blocks + (512 << 20)
    # QSA selection returns before attention dequantization starts. Allow128MiB
    # for BLAS workspace, small masks/IDs, h32 and allocator rounding as well.
    bound = max(dequant_peak, segmented_sort_peak, full_sort_peak) + (128 << 20)
    assert bound == 1280 << 20
    assert bound < 2 << 30
