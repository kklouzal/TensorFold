"""Private lossless-head admission/ownership/callback contracts, CPU only.

Fake CUDA boundaries prove transport decisions, never GPU numerics/quality.
"""
from types import SimpleNamespace

import pytest
import torch

from gemma_modelopt_metadata_fixture import fixture
from test_gemma_modelopt_weights import fake_load
from tensorfold.cuda import b16_sym8 as sym
from tensorfold.cuda.b16_sym8_layout import header_bounds
from tensorfold.families.gemma4.cuda import modelopt_weights as loader
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine


@pytest.mark.parametrize('flag', [1, None, 'true'])
def test_head_flag_is_strict_before_runtime(flag, tmp_path):
    with pytest.raises(ValueError, match='bool'):
        Gemma4Engine(tmp_path, lossless_head=flag)


def test_unsupported_companion_storage_refused_before_allocation(tmp_path):
    with pytest.raises(ValueError, match='DFlash runs'):
        Gemma4Engine(tmp_path, lossless_head=True, drafter_bits=4)


def test_header_bound_is_not_the_qualified_allcompressed_size():
    raw, headers = fixture()
    cfg = loader.validate_headers(raw, headers)
    baseline = loader.storage_bounds(cfg, headers)
    candidate = loader.storage_bounds(cfg, headers, lossless_head=True)
    shape = (cfg.vocab, cfg.hidden)
    bounds = header_bounds(shape)
    groups = cfg.vocab * cfg.hidden // 32
    expected_max = 36 * groups + 64 * min(groups, 0xffffff)
    assert bounds['max_valid_encoded_bytes'] == expected_max
    assert candidate['resident'] == baseline['resident'] - cfg.vocab * cfg.hidden * 2 + expected_max
    assert candidate['host_staging'] == (baseline['host_staging'] + expected_max
                                        + bounds['pack_temporary_bytes'] + bounds['cpu_finite_chunk_bytes'])
    assert bounds['pack_temporary_bytes'] == sum(bounds['pack_temporary_inventory'].values())
    assert candidate['lossless_head'] == bounds
    assert loader.storage_bounds(cfg, headers, lossless_head=False) == baseline



def test_pinned_numpy_expression_dtypes_match_pack_inventory():
    import numpy as np
    mask = np.array([[True, False]], dtype=np.bool_)
    sign = np.array([[0, 8]], dtype=np.uint8)
    assert np.where(mask[:, :1], 8, sign).dtype == np.uint8
    assert np.cumsum(mask, axis=1).dtype.itemsize == 8
    assert np.nonzero(mask)[0].dtype.itemsize == 8


def prepare_fake_head(monkeypatch, *, transfer_failure=None):
    original_validate = sym._validate
    calls = []
    def transfer(packed, *, device, status):
        original_validate(packed, cuda=False)
        calls.append('head-upload')
        status.started, status.safe_to_release, status.source = True, False, packed
        for slot, tensor in enumerate((packed.palette, packed.codes, packed.metadata, packed.raw)):
            status.fields[slot] = tensor
        if transfer_failure is not None:
            raise transfer_failure
        status.result, status.transferred, status.safe_to_release = packed, True, True
        status.source = None
        status.fields[:] = [None] * 4
        return packed  # CPU stand-in, explicitly no physical CUDA proof
    monkeypatch.setattr(sym, 'to_cuda', transfer)
    monkeypatch.setattr(sym, '_validate', lambda packed, **kwargs: original_validate(packed, cuda=False))
    return calls


def test_loader_reads_head_cpu_only_one_table_for_embedding_and_projection(monkeypatch, tmp_path):
    module, status, retired = fake_load(monkeypatch, tmp_path)
    from tensorfold.cuda import direct_read
    get = direct_read.SafeTensors.get
    reads = []
    def observed(self, name, device='cpu'):
        reads.append((name, str(device)))
        return get(self, name, device)
    monkeypatch.setattr(direct_read.SafeTensors, 'get', observed)
    copied = prepare_fake_head(monkeypatch)
    result = module.load(tmp_path, status=status, lossless_head=True)
    assert type(result.embed) is sym.Sym8Table and not hasattr(result.embed, 'weight')
    assert [device for name, device in reads if name.endswith('embed_tokens.weight')] == ['cpu']
    assert copied == ['head-upload'] and retired == ['drain', 'close']
    assert status.safe_to_release and status.transferred and status.owners == {}
    assert result.consumption_receipt['consumed_count'] == result.consumption_receipt['tensor_count']
    assert result.consumption_receipt['lossless_head']['schema'] == result.embed.layout
    calls = []
    monkeypatch.setattr(sym, 'lookup', lambda ids, packed, **kw: calls.append(('lookup', packed)) or ids)
    monkeypatch.setattr(sym, 'matmul', lambda hidden, packed: calls.append(('project', packed)) or hidden)
    token = torch.tensor([1])
    result.embed.embedding(token)
    result.embed(token)
    assert calls == [('lookup', result.embed.packed), ('project', result.embed.packed)]


def test_unsafe_nested_transfer_prevents_outer_drain_close_and_release(monkeypatch, tmp_path):
    module, status, retired = fake_load(monkeypatch, tmp_path)
    primary = RuntimeError('nested native copy failure')
    primary.__cause__ = ValueError('native cause')
    prepare_fake_head(monkeypatch, transfer_failure=primary)
    with pytest.raises(RuntimeError) as caught:
        module.load(tmp_path, status=status, lossless_head=True)
    assert caught.value is primary
    assert retired == [] and not status.safe_to_release
    nested = status.owners['head_transfer']
    assert nested.started and not nested.safe_to_release
    assert status.owners['head_cpu'] is not None and nested.source is status.owners['head_encoded']
    assert all(value is not None for value in nested.fields)


def test_nonfinite_cpu_head_rejected_before_upload(monkeypatch, tmp_path):
    module, status, retired = fake_load(monkeypatch, tmp_path)
    from tensorfold.cuda import direct_read
    get = direct_read.SafeTensors.get
    def corrupt(self, name, device='cpu'):
        value = None
        try:
            value = get(self, name, device)
            if name.endswith('embed_tokens.weight'):
                value.reshape(-1)[0] = float('nan')
            return value
        finally:
            value = None
    monkeypatch.setattr(direct_read.SafeTensors, 'get', corrupt)
    monkeypatch.setattr(sym, 'to_cuda', lambda *a, **k: pytest.fail('upload after nonfinite input'))
    with pytest.raises(ValueError, match='finite'):
        module.load(tmp_path, status=status, lossless_head=True)
    assert retired == ['close'] and not status.gpu_work_started and status.safe_to_release


def test_target_raw_embedding_scale_and_softcap_use_same_table(monkeypatch):
    from test_gemma_b16_pair_dispatch import target
    raw = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64).to(torch.bfloat16)
    packed = sym.pack(raw)
    monkeypatch.setattr(sym, '_validate', lambda *a, **k: None)
    table = sym.Sym8Table(packed, 4096)
    model = target('modelopt_nvfp4')
    # Reconstruct the real constructor with a typed table; no .weight access.
    weights = SimpleNamespace(config=model.config, numerics='modelopt_nvfp4', norm=torch.ones(64), embed=table)
    from tensorfold.families.gemma4.cuda.forward import Gemma4Target
    model = Gemma4Target(weights)
    monkeypatch.setattr(sym, 'lookup', lambda ids, owned, **kw: torch.nn.functional.embedding(ids, raw))
    logits = torch.ones((1, 128), dtype=torch.bfloat16) * 2
    monkeypatch.setattr(sym, 'matmul', lambda hidden, owned: logits)
    ids = torch.tensor([0, 7])
    assert torch.equal(model.raw_embed_tokens(ids).view(torch.int16), raw[ids].view(torch.int16))
    assert torch.equal(model.embed_tokens(ids), raw[ids] * model._embed_scale)
    assert torch.equal(model.project_logits(torch.zeros(1, 64)),
                       torch.tanh(logits / model.config.softcap) * model.config.softcap)
    assert model.weights.embed is table and model._embedding.__self__ is table


def test_lossless_capacity_threads_exact_bounds_and_existing_host_gate(monkeypatch, tmp_path):
    from tensorfold.cuda import capacity as common
    from tensorfold.checkpoint_metadata import read_config
    import sys
    from tensorfold.families.gemma4.cuda import capacity
    raw, headers = fixture()
    cfg = loader.validate_headers(raw, headers)
    bounds = loader.storage_bounds(cfg, headers, lossless_head=True)
    monkeypatch.setattr(sys.modules[read_config.__module__], 'read_config', lambda _: raw)
    monkeypatch.setattr(common, 'headers', lambda _: headers)
    monkeypatch.setattr(common, 'unified', lambda _: True)
    calls = []
    monkeypatch.setattr(common, 'admit_host_commitment', lambda amount: calls.append(('host', amount)))
    def admit(*args, **kwargs):
        weights, host = kwargs['weight_estimator'](tmp_path)
        calls.append(('common', weights.resident, weights.staging, host))
        return {'context_window': 128}
    monkeypatch.setattr(common, 'admit', admit)
    plan = capacity.admit(tmp_path, 128, True, max_rows=1, lossless_head=True)
    assert calls == [('host', bounds['host_staging']),
                     ('common', bounds['resident'], bounds['device_staging'] + bounds['host_staging'],
                      bounds['host_staging'])]
    assert plan['lossless_head'] == bounds['lossless_head']
