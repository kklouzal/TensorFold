"""Native owner operator contracts without claiming device numerics or speed."""

from types import SimpleNamespace as NS
from pathlib import Path

import pytest

from tensorfold.cuda.sm70 import paged_attention as operator
from tensorfold.cuda.sm70 import paged_attention_descriptor as descriptor


@pytest.mark.parametrize("rows", [0, 1, 32, 96, 128, 257])
def test_workspace_is_bounded_by_rows_not_retained_context(rows):
    assert operator.DESCRIPTOR_WORDS == 16 and operator.DESCRIPTOR_BYTES == 128
    assert operator.workspace_bytes(rows) == rows * (128 + 12 * 2 * 4)
    assert operator.generation_bytes(rows) == rows * 16


@pytest.mark.parametrize("invalid", [-1, True, 3.5, None])
def test_invalid_resource_geometry_is_rejected(invalid):
    with pytest.raises(ValueError, match="integer"):
        operator.workspace_bytes(invalid)
    with pytest.raises(ValueError, match="integer"):
        operator.generation_bytes(invalid)


@pytest.mark.parametrize("field,value", [("group_start", 2), ("group_start", True), ("page_tokens", 0),
                                         ("page_tokens", 129), ("page_tokens", 2**31),
                                         ("key_bits", 16), ("key_codec", 4),
                                         ("key_bits", True), ("scale", 0), ("scale", float("nan")),
                                         ("scale", float("inf")), ("scale", True), ("scale", 1e-100),
                                         ("scale", 1e100), ("scale", 1 << 2000)])
def test_unsupported_controls_fail_before_extension_acquisition(monkeypatch, field, value):
    monkeypatch.setattr(operator, "extension", lambda: pytest.fail("invalid operator controls acquired native code"))
    controls = dict(group_start=0, page_tokens=128, layer=object(), key_bits=8, key_codec=0, scale=1 / 16)
    controls[field] = value
    with pytest.raises(ValueError):
        operator.local_statistics(None, None, None, None, **controls)


@pytest.mark.parametrize("key,value", [((8, 0), (8, 0)), ((4, 10), (4, 10)), ((4, 9), (4, 9))])
def test_four_phases_borrow_outputs_and_keep_global_normalization_between_collectives(monkeypatch, key, value):
    calls = []
    native = NS(**{name: (lambda *args, name=name: calls.append((name, args))) for name in
                   ("local_statistics", "merge_statistics", "normalized_partial", "merge_values")})
    monkeypatch.setattr(operator, "extension", lambda: native)
    query, selection, descriptor, local, gathered, global_stats, values, all_values, output = [object() for _ in range(9)]
    layer = object()
    controls = dict(group_start=4, page_tokens=128, layer=layer)
    assert operator.local_statistics(query, selection, descriptor, local, **controls,
                                      key_bits=key[0], key_codec=key[1], scale=1/16) is local
    assert operator.merge_statistics(gathered, descriptor, global_stats, **controls) is global_stats
    assert operator.normalized_partial(query, selection, descriptor, global_stats, values, **controls,
                                        key_bits=key[0], key_codec=key[1], value_bits=value[0], value_codec=value[1],
                                        scale=1/16) is values
    assert operator.merge_values(all_values, descriptor, output, rank=6, page_tokens=128, layer=layer,
                                  value_bits=value[0], value_codec=value[1]) is output
    assert [call[0] for call in calls] == ["local_statistics", "merge_statistics", "normalized_partial", "merge_values"]
    assert calls[2][1][3] is global_stats and calls[3][1][0] is all_values
    assert calls[0][1][6] is layer and calls[3][1][5] is layer


def test_provisional_or_unsupported_formats_cannot_enter_pv_or_value_merge(monkeypatch):
    monkeypatch.setattr(operator, "extension", lambda: pytest.fail("unsupported format acquired native code"))
    with pytest.raises(ValueError, match="ordinary Q8"):
        operator.normalized_partial(None, None, None, None, None, group_start=0, page_tokens=128, layer=0,
                                     key_bits=8, key_codec=0, value_bits=4, value_codec=5, scale=1/16)
    with pytest.raises(ValueError, match="rank"):
        operator.merge_values(None, None, None, rank=8, page_tokens=128, layer=0, value_bits=8, value_codec=0)
    with pytest.raises(ValueError, match="homogeneous"):
        operator.normalized_partial(None, None, None, None, None, group_start=0, page_tokens=128, layer=0,
                                     key_bits=8, key_codec=0, value_bits=4, value_codec=9, scale=1/16)


def test_native_source_closure_reuses_the_owned_codec_decoder_and_pins_sm70(monkeypatch):
    from tensorfold.cuda import build
    from tensorfold.cuda.sm70 import provider, source_cache

    staged = []
    built = []

    def stage(sources, *, headers, source_root):
        assert all(path.is_file() for path in [*sources, *headers])
        staged.append((sources, headers, source_root))
        return [str(path) for path in sources]

    monkeypatch.setattr(provider, "_native", lambda: NS(__name__="verified-pinned-foundation"))
    monkeypatch.setattr(source_cache, "stage_sources", stage)
    module = NS(descriptor_abi=descriptor.ABI_VERSION, descriptor_words=descriptor.WORDS,
                descriptor_schema_id=descriptor.SCHEMA_ID)
    monkeypatch.setattr(build, "load", lambda *args, **kwargs: built.append((args, kwargs)) or module)
    operator.extension.cache_clear()
    try:
        first = operator.extension()
        assert operator.extension() is first and len(built) == 1
    finally:
        operator.extension.cache_clear()
    assert {path.name for path in staged[0][1]} == {"paged_attention_descriptor.cuh", "packed_kv_decode.cuh",
                                                  "rotorquant_q4.cuh", "rotorquant_q4_tables.cuh"}
    assert built[0][1]["need"] == (7, 0)
    assert built[0][1]["extra_cuda_cflags"] == ["-O3", "--fmad=false"]


def test_maintained_native_descriptor_header_matches_its_single_source():
    header = Path(descriptor.__file__).with_suffix(".cuh")
    assert header.read_text() == descriptor.render_header()
    assert len(set(descriptor.FIELDS)) == descriptor.WORDS


@pytest.mark.parametrize("bad_module", [NS(), NS(descriptor_abi=1, descriptor_words=16,
                                                 descriptor_schema_id="stale-field-order")])
def test_incompatible_compiled_descriptor_schema_is_rejected_before_use(monkeypatch, bad_module):
    from tensorfold.cuda import build
    from tensorfold.cuda.sm70 import provider, source_cache

    monkeypatch.setattr(provider, "_native", lambda: NS(__name__="verified-pinned-foundation"))
    monkeypatch.setattr(source_cache, "stage_sources", lambda sources, **kwargs: [str(path) for path in sources])
    monkeypatch.setattr(build, "load", lambda *args, **kwargs: bad_module)
    operator.extension.cache_clear()
    try:
        with pytest.raises(RuntimeError, match="descriptor ABI"):
            operator.extension()
    finally:
        operator.extension.cache_clear()
