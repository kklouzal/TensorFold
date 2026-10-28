"""Exact preparation-only comparisons; unchanged math is exercised end to end."""

from contextlib import contextmanager
import pytest
import torch
from tensorfold.families.qwen4_exp.cuda.compact_indexer import Cache, Span
from tensorfold.cuda.sm70 import family_ops as ops
from compact_pool_reference import indexed_pool


def make_case(start, count, *, device="cpu", mtp=False):
    device = torch.device(device)
    cache = Cache(
        64,
        128,
        device,
        torch.zeros(2, dtype=torch.int64, device=device),
        torch.tensor([0, -2], dtype=torch.int32, device=device),
        layer=3,
        proposal_rows=32 if mtp else 0,
        mtp=mtp,
    )
    source = (torch.arange(96 * 128).reshape(96, 128).float().remainder(389) / 53 - 3).half().to(device)
    coordinates = torch.stack(
        (torch.arange(96) * 5 + 7, torch.arange(96) * 3 - 41, torch.arange(96) * 2 + (1 << 33)), 1
    ).to(device)
    cache.n = start
    cache.protected.fill_(-99)
    cache.protected_coordinates.fill_(-987)
    prefix = start % 4
    if prefix:
        cache.protected[:prefix].copy_(source[start - prefix : start])
        cache.protected_coordinates[:prefix].copy_(coordinates[start - prefix : start])
    raw, coords = source[start : start + count].clone(), coordinates[start : start + count].clone()
    cache.span = Span(start, raw, coords, None, 0, mtp, (0, 0))
    pooled = torch.full((16, 128), 37.0, dtype=torch.float16, device=device)
    gamma = torch.linspace(0.7, 1.3, 128, device=device)
    frequency = torch.linspace(0.01, 0.2, 32, device=device)
    return cache, pooled, gamma, frequency


def assert_bits(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    assert torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@torch.no_grad()
def test_every_start_modulo_and_input_length_exact_original_pool():
    for mtp in (False, True):
        for start in (0, 1, 2, 3, 19, 31):
            for count in range(1, 33):
                candidate, out, weight, freq = make_case(start, count, mtp=mtp)
                reference, expected, _, _ = make_case(start, count, mtp=mtp)
                assert candidate._pool_slices_supported(candidate.span)
                candidate.pool(out, weight, freq, 1e-6, (11, 11, 10), 0.83)
                indexed_pool(reference, expected, weight, freq, 1e-6, (11, 11, 10), 0.83)
                assert_bits(out, expected)
                assert torch.equal(candidate.status, reference.status)
                assert candidate._retirement_error is None


@torch.no_grad()
def test_cat_raw_and_coordinate_outputs_are_exact_fresh_contiguous(monkeypatch):
    for start, count in ((0, 4), (1, 3), (2, 9), (3, 1), (19, 32)):
        cache, out, w, f = make_case(start, count)
        first, end = start // 4, (start + count) // 4
        positions = torch.arange(first * 4, end * 4)
        old = positions < start
        source = (positions - start).clamp(0, count - 1)
        expected_raw = torch.where(
            old[:, None], cache.protected.index_select(0, positions % 4), cache.span.raw.index_select(0, source)
        )
        member = positions.reshape(-1, 4)[:, 0]
        expected_coords = torch.where(
            (member < start)[:, None],
            cache.protected_coordinates.index_select(0, member % 4),
            cache.span.coordinates.index_select(0, (member - start).clamp(0, count - 1)),
        )
        records = []
        original = torch.cat

        def cat(values, *args, **kwargs):
            value = original(values, *args, **kwargs)
            records.append(value)
            return value

        with monkeypatch.context() as patch:
            patch.setattr(torch, "cat", cat)
            cache.pool(out, w, f, 1e-6, (11, 11, 10), 1.0)
        assert len(records) == 2
        assert_bits(records[0], expected_raw)
        assert_bits(records[1], expected_coords)
        assert all(value.is_contiguous() for value in records)
        assert all(
            value.data_ptr()
            not in (
                cache.protected.data_ptr(),
                cache.span.raw.data_ptr(),
                cache.protected_coordinates.data_ptr(),
                cache.span.coordinates.data_ptr(),
            )
            for value in records
        )


def test_grad_mode_and_grad_sources_retain_original_indexed_domain():
    cache, *_ = make_case(1, 7)
    assert not cache._pool_slices_supported(cache.span)
    with torch.no_grad():
        assert cache._pool_slices_supported(cache.span)
        cache.span.raw.requires_grad_(True)
        assert not cache._pool_slices_supported(cache.span)


@pytest.mark.parametrize("kind", ["strided", "negative", "float32", "wrong_coordinates", "subclass", "empty"])
@torch.no_grad()
def test_unsupported_domains_keep_original_preparation(kind):
    cache, *_ = make_case(1, 7)
    if kind == "strided":
        cache.span.raw = torch.stack((cache.span.raw, cache.span.raw), dim=-1)[..., 0]
    elif kind == "negative":
        cache.span.raw = torch._neg_view(cache.span.raw)
    elif kind == "float32":
        cache.span.raw = cache.span.raw.float()
    elif kind == "wrong_coordinates":
        cache.span.coordinates = cache.span.coordinates[:, :2]
    elif kind == "subclass":

        class Derived(torch.Tensor):
            pass

        cache.span.raw = cache.span.raw.as_subclass(Derived)
    else:
        cache.span.raw = cache.span.raw[:0]
    assert not cache._pool_slices_supported(cache.span)


@torch.no_grad()
def test_reference_failures_empty_raw_and_coordinate_shapes_are_preserved():
    for kind in ("empty", "bad_coordinates"):
        candidate, out, w, f = make_case(1, 7)
        reference, expected, _, _ = make_case(1, 7)
        for cache in (candidate, reference):
            if kind == "empty":
                cache.span.raw = cache.span.raw[:0]
            else:
                cache.span.coordinates = cache.span.coordinates[:0]
        # Empty data ending before the next group retains the original earlyreturn.
        if kind == "empty":
            candidate.pool(out, w, f, 1e-6, (11, 11, 10), 1.0)
            indexed_pool(reference, expected, w, f, 1e-6, (11, 11, 10), 1.0)
            assert_bits(out, expected)
        else:
            with pytest.raises((IndexError, RuntimeError)) as new:
                candidate.pool(out, w, f, 1e-6, (11, 11, 10), 1.0)
            with pytest.raises(type(new.value)):
                indexed_pool(reference, expected, w, f, 1e-6, (11, 11, 10), 1.0)


@torch.no_grad()
def test_rms_failure_precedes_coordinate_preparation_and_retires_same_sources(monkeypatch):
    cache, out, w, f = make_case(1, 7)
    primary = RuntimeError("original RMS failure")
    events = []
    cat = torch.cat

    @contextmanager
    def reading(tensors, *, span):
        assert tuple(tensors) == (cache.protected, cache.protected_coordinates, cache.span.raw, cache.span.coordinates)
        events.append("enter")
        try:
            yield [False]
        finally:
            events.append("retire")

    def fail(*_args, **_kwargs):
        events.append("rms")
        raise primary

    def record(values, *args, **kwargs):
        events.append("cat")
        return cat(values, *args, **kwargs)

    monkeypatch.setattr(cache, "reading", reading)
    monkeypatch.setattr(ops, "rms", fail)
    monkeypatch.setattr(torch, "cat", record)
    with pytest.raises(RuntimeError) as caught:
        cache.pool(out, w, f, 1e-6, (11, 11, 10), 1.0)
    assert caught.value is primary and events == ["enter", "cat", "rms", "retire"]


@torch.no_grad()
def test_subclass_misdeclared_nonempty_span_keeps_original_empty_index_error():
    class GenericSpan(Span):
        @property
        def end(self):
            return self.start + 4

    candidate, out, w, f = make_case(3, 1)
    reference, expected, _, _ = make_case(3, 1)
    for entry in (candidate, reference):
        old = entry.span
        entry.span = GenericSpan(old.start, old.raw[:0], old.coordinates[:0], None, 0, False, (0, 0))
    assert not candidate._pool_slices_supported(candidate.span)
    with pytest.raises((IndexError, RuntimeError)) as new:
        candidate.pool(out, w, f, 1e-6, (11, 11, 10), 1.0)
    with pytest.raises(type(new.value)):
        indexed_pool(reference, expected, w, f, 1e-6, (11, 11, 10), 1.0)
