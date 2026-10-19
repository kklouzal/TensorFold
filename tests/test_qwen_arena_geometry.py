"""Arena admission follows retained owners and physical granules, including small two-rank windows and MTP graphs.

These host checks derive cache bytes from declared dimensions and the allocator's admitted 2 MiB granule; actual
driver mappings/lifetimes remain covered by CUDA tests. No CUDA hardware or payload allocation is required here.
"""

import pytest
from cuda_27b_headers import TEXT

from tensorfold.cuda import geometry
from tensorfold.families.qwen3_5_moe.cuda.engine import stream_geometry


def _mapped(window, world):
    side = window * (4 // world) * 256 * 2             # one layer's one BF16 side, from the 27B dimensions
    return 16 * 2 * ((side + (2 << 20) - 1) // (2 << 20)) * (2 << 20)


@pytest.mark.parametrize("world", [1, 2])
@pytest.mark.parametrize("keep", [0, 2, 4])
@pytest.mark.parametrize("window", [0, 1, 1024, 2048, 2049, 262144])
def test_each_retained_lineage_rounds_its_caches_independently(world, keep, window):
    reserve, rows = 12, 12
    plan = geometry.gdn_geometry(TEXT, world, reserve, rows=rows, kept=keep)
    slots = window + reserve
    def scratch(n):
        return rows * (24 // world) * (256 + 2) * ((n + 511) // 512) * 4
    caches = plan.bytes_at(slots) - plan.bytes_at(reserve) - (scratch(slots) - scratch(reserve))
    assert caches == (keep + 1) * _mapped(window, world)
    if world == 2 and window == 1024:
        assert caches == (keep + 1) * (64 << 20)      # keep=2: 192 MiB; actual engine keep=4: 320 MiB


def test_inferred_attention_head_dimension_keeps_the_same_admission():
    text = {**TEXT, "hidden_size": 6144}              # 6144 / 24 heads = the declared 256-wide head
    inferred = {key: value for key, value in text.items() if key != "head_dim"}
    for world in (1, 2):
        for window in (1, 1024, 262144):
            assert geometry.live_kv(inferred, world, window) == _mapped(window, world)
            assert geometry.gdn_geometry(inferred, world, 12).bytes_at(window + 12) == \
                geometry.gdn_geometry(text, world, 12).bytes_at(window + 12)


@pytest.mark.parametrize("mtp", [False, True])
def test_prepared_cache_dimensions_do_not_follow_later_header_mutations(mtp):
    text = {**TEXT, "layer_types": ["linear_attention"] * 48 + ["full_attention"] * 16}
    plan = geometry.gdn_geometry(text, 2, 12, kept=4, mtp=mtp)
    parallel = stream_geometry(text, 3, 4, 2 if mtp else 0)
    before = [(plan.bytes_at(n), parallel.bytes_at(n)) for n in (0, 1036, 262156)]
    text["layer_types"].clear()
    text.update(num_key_value_heads=2, head_dim=128)
    assert [(plan.bytes_at(n), parallel.bytes_at(n)) for n in (0, 1036, 262156)] == before


@pytest.mark.parametrize("mtp", [False, True])
def test_more_retained_prefixes_count_their_recurrent_state_and_mtp_head(mtp):
    small = geometry.gdn_geometry(TEXT, 1, 12, kept=2, mtp=mtp)
    large = geometry.gdn_geometry(TEXT, 1, 12, kept=4, mtp=mtp)
    recurrent = 48 * (48 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 48 * 128) * 2)
    for slots in (0, 1036, 262156):
        expected = 2 * (recurrent + _mapped(max(0, slots - 12), 1))
        if mtp:
            expected += 2 * slots * 4 * 256 * 4       # each additional prefix's ordinary MTP K/V pair
        assert large.bytes_at(slots) - small.bytes_at(slots) == expected


@pytest.mark.parametrize("slots", [1, 1024, 8192, 8193])
def test_single_mtp_graph_has_a_separate_target_arena_and_head_cache(slots):
    keep, rows = 4, 12
    serial = geometry.gdn_geometry(TEXT, 1, 12, rows=rows, kept=keep)
    mtp = geometry.gdn_geometry(TEXT, 1, 12, rows=rows, kept=keep, mtp=True)
    state = 48 * (48 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 48 * 128) * 2)
    # Existing MTP verify workspace, one graph state/arena, kept/current/graph heads, and Cache.view's cat suffix.
    head_side = slots * 4 * 256 * 2
    expected = 32 * rows * 2560 * 4 + state + _mapped(slots, 1) + (keep + 2) * 2 * head_side + head_side
    assert mtp.bytes_at(slots) - serial.bytes_at(slots) == expected


@pytest.mark.parametrize("streams,keep", [(2, 0), (4, 3)])
@pytest.mark.parametrize("slots", [1, 1024, 8193])
def test_parallel_mtp_graph_arena_keeps_its_granules(streams, keep, slots):
    eager = stream_geometry(TEXT, streams, keep, 0)
    mtp = stream_geometry(TEXT, streams, keep, 2)
    state = 48 * (48 * 128 * 128 * 4 + 3 * (2 * 16 * 128 + 48 * 128) * 2)
    expected = state + _mapped(slots, 1) + (streams + keep + 2) * slots * 4 * 256 * 4
    assert mtp.bytes_at(slots) - eager.bytes_at(slots) == expected
