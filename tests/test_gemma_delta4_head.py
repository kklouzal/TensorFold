"""Experimental tied-table integration controls; CPU stand-ins are not GPU proof."""
from dataclasses import replace
import gc
from types import SimpleNamespace
import weakref

import pytest
import torch

from tensorfold.cuda import b16_delta4 as codec, b16_delta4_vector as vector
from tensorfold.families.gemma4.cuda import capacity, head, nvfp4_weights as loader
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tensorfold.families.gemma4.cuda.forward import Gemma4Target
from tensorfold.families.gemma4.cuda.weights import Config
from tests.test_gemma_ct_loader_lifecycle import fake_ct
from tests.test_gemma_cuda_modelopt_integration import _capacity_boundary, raw_modelopt
from tests.test_gemma_dflash_sym8_fp8 import factory_fixture


@pytest.mark.parametrize('value', [1, None, 'true'])
def test_delta4_flag_is_strict_before_checkpoint_reads(value, tmp_path):
    with pytest.raises(ValueError, match='bool'):
        Gemma4Engine(tmp_path, delta4_head=value)
    with pytest.raises(ValueError, match='bool'):
        capacity.admit(tmp_path, 128, True, delta4_head=value)
    with pytest.raises(ValueError, match='bool'):
        loader.load(tmp_path, delta4_head=value)


@pytest.mark.parametrize('value', [True, 0, 3, 32, 1.0])
def test_reuse_is_a_bounded_constructor_investigation(value, tmp_path):
    with pytest.raises(ValueError, match='1/2/4/8/16'):
        Gemma4Engine(tmp_path, delta4_head=True, delta4_reuse=value)
    with pytest.raises(ValueError, match='1/2/4/8/16'):
        loader.load(tmp_path, delta4_head=True, delta4_reuse=value)


def test_delta4_reuse_cannot_change_default_raw_policy(tmp_path):
    with pytest.raises(ValueError, match='requires delta4_head'):
        Gemma4Engine(tmp_path, delta4_reuse=8)
    with pytest.raises(ValueError, match='requires delta4_head'):
        loader.load(tmp_path, delta4_reuse=8)


def test_unpublished_table_releases_partial_descriptor_borrow_and_preserves_native_primary(monkeypatch):
    validate = codec._validate
    monkeypatch.setattr(codec, '_validate', lambda value, *, cuda: validate(value, cuda=False))
    primary = MemoryError('late table publication')
    native = OSError('original native root')
    primary.__cause__ = native
    descriptors, partial = [], []

    def fail(owner, value):
        try:
            partial.append(weakref.ref(owner))
            raise primary
        finally:
            del owner, value

    def construct():
        packed = codec.pack(torch.ones((2, 64), dtype=torch.bfloat16))
        descriptors.append(weakref.ref(packed))
        try:
            head.Delta4Table(packed, 128)
        finally:
            packed = None  # isolate the unpublished wrapper's borrowed reference

    monkeypatch.setattr(head.Delta4Table, 'layout', property(fset=fail), raising=False)
    caught = None
    try:
        construct()
    except MemoryError as error:
        caught = error
    gc.collect()
    assert caught is primary and primary.__cause__ is native and primary.__traceback__ is not None
    assert descriptors[0]() is None
    assert partial[0]() is not None and object.__getattribute__(partial[0](), '__dict__') == {}


def test_table_returns_raw_embedding_and_target_applies_original_scaling_and_single_softcap(monkeypatch):
    config = replace(Config.from_dict(raw_modelopt()), global_head_dim=512, global_kv_heads=2)
    original = (torch.arange(config.vocab * config.hidden, dtype=torch.float32).reshape(
        config.vocab, config.hidden) / 128).to(torch.bfloat16)
    packed = codec.pack(original)
    validate = codec._validate
    monkeypatch.setattr(codec, '_validate', lambda value, *, cuda: validate(value, cuda=False))
    table = head.Delta4Table(packed, 16384, reuse_rows=8)
    assert not hasattr(table, 'weight') and not hasattr(table, 'incumbent')
    assert table.nbytes() == packed.nbytes and (table.n, table.k) == tuple(original.shape)
    monkeypatch.setattr(vector, 'lookup', lambda ids, value, **kwargs: original[ids.long()])
    raw_logits = torch.tensor([[-100., 0., 100.]], dtype=torch.bfloat16).repeat(17, 1)
    calls = []

    def projection(hidden, value, *, rows):
        calls.append((hidden.shape[0], rows))
        assert hidden.dtype == torch.bfloat16 and hidden.is_contiguous() and value is packed
        return raw_logits

    monkeypatch.setattr(vector, 'matmul_rows', projection)
    target = Gemma4Target(SimpleNamespace(numerics='ct_nvfp4', config=config, norm=torch.ones(config.hidden), embed=table))
    ids = torch.tensor([0, 2, config.vocab - 1])
    assert torch.equal(target.raw_embed_tokens(ids).view(torch.int16), original[ids].view(torch.int16))
    expected_embedding = original[ids] * torch.tensor(config.hidden**.5, dtype=torch.bfloat16)
    assert torch.equal(target.embed_tokens(ids).view(torch.int16), expected_embedding.view(torch.int16))
    actual = target.project_logits(torch.zeros((17, config.hidden), dtype=torch.float32))
    expected_logits = torch.tanh(raw_logits / config.softcap) * config.softcap
    assert torch.equal(actual.view(torch.int16), expected_logits.view(torch.int16))
    assert calls == [(17, 8)]


def fake_delta_cuda(monkeypatch):
    """Exercise real transfer bookkeeping with CPU storage, not CUDA transport."""
    validate, to = codec._validate, torch.Tensor.to
    monkeypatch.setattr(codec, '_validate', lambda packed, *, cuda: validate(packed, cuda=False))

    def copy(value, *args, **kwargs):
        if 'device' in kwargs and torch.device(kwargs['device']).type == 'cuda':
            kwargs['device'] = 'cpu'
        return to(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, 'to', copy)


def test_ct_loader_never_uploads_raw_table_and_releases_cpu_sources_after_transfer(monkeypatch, tmp_path):
    status, retired, reads = fake_ct(monkeypatch, tmp_path)
    fake_delta_cuda(monkeypatch)
    result = loader.load(tmp_path, status=status, delta4_head=True, delta4_reuse=8)
    assert [device for name, device in reads if name.endswith('embed_tokens.weight')] == ['cpu']
    assert type(result.embed) is head.Delta4Table and result.embed.reuse_rows == 8
    assert not hasattr(result.embed, 'weight') and not hasattr(result.embed, 'incumbent')
    receipt = result.consumption_receipt['delta4_head']
    assert receipt['schema'] == codec.SCHEMA and receipt['packed_bytes'] == result.embed.nbytes()
    assert receipt['raw_gpu_table_retained'] is False
    assert retired == ['drain', 'drain', 'close'] and status.transferred and status.safe_to_release
    assert not status.owners
    assert status.test_read_refs['model.language_model.embed_tokens.weight']() is None


def test_ct_failed_delta_transfer_fence_keeps_complete_owners_without_retry(monkeypatch, tmp_path):
    primary = RuntimeError('unknown upload completion')
    primary.__cause__ = OSError('original native cause')
    status, retired, _ = fake_ct(monkeypatch, tmp_path, drain_error=primary)
    fake_delta_cuda(monkeypatch)
    with pytest.raises(RuntimeError) as caught:
        loader.load(tmp_path, status=status, delta4_head=True)
    assert caught.value is primary and retired == ['drain']
    assert status.owners['read_failed'] == [True] and not status.safe_to_release
    transfer = status.owners['head_transfer']
    assert not transfer.safe_to_release and transfer.source is status.owners['head_encoded']
    assert all(field is not None for field in transfer.fields) and not status.transferred
    assert status.owners['head_cpu'] is not None and status.owners['reader'] is not None


def test_delta_admission_reserves_worst_valid_layout_and_cpu_overlap_once(monkeypatch, tmp_path):
    import tensorfold.families as families
    from tensorfold.cuda import capacity as common
    observed = _capacity_boundary(monkeypatch, uma=True)
    config = replace(Config.from_dict(raw_modelopt()), global_head_dim=512, global_kv_heads=2)
    monkeypatch.setattr(families, 'quant_method', lambda _: 'compressed-tensors')
    monkeypatch.setattr(loader, 'validate_headers', lambda *a, **kw: config)
    monkeypatch.setattr(loader, 'ct_tensor_contract', lambda _: ({}, {}))
    baseline = capacity.admit(tmp_path, 128, True)
    candidate = capacity.admit(tmp_path, 128, True, delta4_head=True)
    bounds = candidate['delta4_head']
    peak = (bounds['raw_source_bytes'] + bounds['max_valid_encoded_bytes'] + bounds['pack_temporary_bytes']
            + 2 * (64 << 20) + 2 * 4096)
    assert candidate['weight'].resident - baseline['weight'].resident == (
        bounds['max_valid_encoded_bytes'] - bounds['raw_source_bytes'])
    assert candidate['weight'].staging - baseline['weight'].staging == peak
    assert candidate['host'] >= peak and observed == [peak]
    monkeypatch.setattr(common, 'unified', lambda _: False)
    discrete = capacity.admit(tmp_path, 128, True, delta4_head=True)
    assert discrete['weight'].staging == baseline['weight'].staging and discrete['host'] >= peak


def test_delta_drafter_fp8_copy_is_raw_unscaled_and_matches_existing_quantizer(monkeypatch):
    from tensorfold.cuda.requant import requantize
    from tensorfold.families.gemma4.cuda.dflash import _packed_fp8_head
    table, original, calls, _ = factory_fixture(monkeypatch)
    table.layout = codec.SCHEMA
    monkeypatch.setattr(head, 'Delta4Table', type(table))
    result = _packed_fp8_head(table, softcap=30.)
    expected = requantize(original, 'fp8')
    assert torch.equal(result.linear.w8, expected.w8)
    assert torch.equal(result.linear.cols.view(torch.int32), expected.cols.view(torch.int32))
    assert result.softcap == 30. and not hasattr(table, 'weight')
    assert [item[1] for item in calls if item[0] == 'lookup'] == [[0, 1], [2, 3], [4]]
