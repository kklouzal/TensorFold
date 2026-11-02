"""ModelOpt DFlash opt-in metadata/admission; no CUDA or model load."""
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest

from gemma_modelopt_metadata_fixture import fixture
from test_gemma_cuda_dflash_metadata import config as draft_raw
from tensorfold.families.gemma4.dflash_metadata import DraftConfig
from tensorfold.families.gemma4.cuda import capacity, modelopt_weights
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tensorfold.families.gemma4.cuda.weights import Config


def modelopt_pair(monkeypatch):
    import torch
    from tensorfold.cuda import capacity as common
    from tensorfold import checkpoint_metadata
    from tensorfold.families.gemma4 import dflash_metadata
    raw, _ = fixture()
    raw = deepcopy(raw)
    raw['num_hidden_layers'] = 2
    raw['layer_types'] = ['sliding_attention', 'full_attention']
    cfg = Config.from_dict(raw)
    experts, plain = modelopt_weights.tensor_contract(cfg)
    headers = {name: {'dtype': 'BF16', 'shape': list(shape)} for name, shape in plain.items()}
    for name, (n, k) in experts.items():
        for suffix, dtype, shape in (('weight', 'U8', [n, k//2]), ('weight_scale', 'F8_E4M3', [n, k//16]),
                                     ('input_scale', 'F32', []), ('weight_scale_2', 'F32', [])):
            headers[name + '.' + suffix] = {'dtype': dtype, 'shape': shape}
    companion_raw = {**draft_raw(), 'num_target_layers': 2, 'vocab_size': cfg.vocab,
                     'num_hidden_layers': 5, 'head_dim': 64, 'block_size': 16, 'sliding_window': 2048,
                     'layer_types': ['sliding_attention'] * 4 + ['full_attention'],
                     'dflash_config': {'target_layer_ids': [1], 'mask_token_id': 4}}
    draft = DraftConfig.from_dict(companion_raw)
    draft_headers = {name: {'dtype': 'BF16', 'shape': list(shape)} for name, shape in draft.shapes().items()}
    monkeypatch.setattr(checkpoint_metadata, 'read_config', lambda _: raw)
    monkeypatch.setattr(common, 'headers', lambda _: headers)
    monkeypatch.setattr(dflash_metadata, 'checked_headers', lambda _: (draft, draft_headers))
    monkeypatch.setattr(common, 'unified', lambda _: False)
    monkeypatch.setattr(common, 'admit_host_commitment', lambda _: None)
    def admit(directory, context, explicit, runtime, geometry, rule, **kwargs):
        assert runtime is torch
        weights, host = kwargs['weight_estimator'](directory)
        return {'context_window': context, 'weights': weights, 'host': host,
                'geometry': geometry, 'sampling_device': kwargs['device_workspace_bytes'],
                'sampling_host': kwargs['host_workspace_bytes']}
    monkeypatch.setattr(common, 'admit', admit)
    return cfg, draft, draft_headers


def test_validated_bf16_opt_in_admits_companion_weight_state_and_rejection_rows(monkeypatch, tmp_path):
    from tensorfold.engine.penalties import bias_workspace

    cfg, draft, headers = modelopt_pair(monkeypatch)
    assert draft.layers == 5 and len(headers) == 58 and draft.tap_layers == (1,)
    serial = capacity.admit(tmp_path, 32, True, max_rows=1)
    drafted = capacity.admit(tmp_path, 32, True, max_rows=16, drafter_dir=tmp_path/'draft', drafter_bits=16)
    expected = sum(2 * math.prod(info['shape']) for info in headers.values())
    assert drafted['weights'].resident - serial['weights'].resident == expected
    assert drafted['geometry'].needed(32) > serial['geometry'].needed(32)
    serial_bias = bias_workspace(1, cfg.vocab)
    drafted_bias = bias_workspace(16, cfg.vocab)
    assert serial['logit_bias_workspace'] == serial_bias
    assert drafted['logit_bias_workspace'] == drafted_bias
    assert serial['sampling_device'] == serial_bias['device_bytes']
    assert serial['sampling_host'] == serial_bias['host_bytes']
    assert drafted['sampling_device'] - drafted_bias['device_bytes'] > 0
    assert drafted['sampling_host'] - drafted_bias['host_bytes'] > 0


def test_mismatched_taps_refused_before_common_allocation(monkeypatch, tmp_path):
    _, draft, headers = modelopt_pair(monkeypatch)
    from tensorfold.families.gemma4 import dflash_metadata
    from tensorfold.cuda import capacity as common
    invalid = SimpleNamespace(**{**draft.__dict__, 'tap_layers': (0,)})
    monkeypatch.setattr(dflash_metadata, 'checked_headers', lambda _: (invalid, headers))
    monkeypatch.setattr(common, 'admit', lambda *a, **k: pytest.fail('allocation after tap mismatch'))
    with pytest.raises(ValueError, match='taps'):
        capacity.admit(tmp_path, 32, True, max_rows=16, drafter_dir=tmp_path/'draft')


def test_modelopt_fp8_admission_and_lossless_width_bound(monkeypatch, tmp_path):
    modelopt_pair(monkeypatch)
    plan = capacity.admit(tmp_path, 32, True, drafter_dir=tmp_path/'draft', drafter_bits=8)
    assert plan['sampling_device'] > 0 and plan['sampling_host'] > 0
    with pytest.raises(ValueError, match='at most16'):
        capacity.admit(tmp_path, 32, True, max_rows=17, lossless_head=True)


def test_engine_allows_explicit_bf16_companion_to_reach_existing_startup(monkeypatch, tmp_path):
    from tensorfold import checkpoint_metadata
    from tensorfold.cuda import precision
    import torch
    raw, _ = fixture()
    monkeypatch.setattr(checkpoint_metadata, 'read_config', lambda _: raw)
    monkeypatch.setattr(precision, 'mode', lambda: precision.CHECKPOINT)
    class StartupBoundary(RuntimeError):
        pass
    def stop(device):
        assert device == 0
        raise StartupBoundary('before physical CUDA work')
    monkeypatch.setattr(torch.cuda, 'set_device', stop)
    with pytest.raises(StartupBoundary):
        Gemma4Engine(tmp_path, drafter='validated-by-capacity-next', drafter_bits=16)


def test_real_dflash_context_cap_returns_before_device_work_at_last_slot():
    from tensorfold.families.gemma4.cuda.dflash import DFlash
    head = object.__new__(DFlash)
    head.config = SimpleNamespace(vocab=262144)
    head.window, head.block = 131072, 16
    # Actual head source caps block length to its remaining logical window.
    # There is no invented target-overflow bug or smaller q-geometry policy.
    assert head.logits(SimpleNamespace(length=131071), 4, 15) is None


def test_real_spark_context_cap_returns_before_conditioning_at_last_slot():
    from tensorfold.families.gemma4.cuda.sparkulator import Sparkulator
    head = object.__new__(Sparkulator)
    head._closed, head._bound = False, object()
    head.cfg, head.block, head.context_limit = SimpleNamespace(vocab=262144), 7, 131072
    state = SimpleNamespace(owner=head, position=131071)
    assert head.propose(state, 4, 6) is None


def test_sym8_fp8_private_head_materialization_chunk_is_admitted(monkeypatch, tmp_path):
    cfg, _, _ = modelopt_pair(monkeypatch)
    raw = capacity.admit(tmp_path, 32, True, max_rows=16, drafter_dir=tmp_path/'draft', drafter_bits=8)
    packed = capacity.admit(tmp_path, 32, True, max_rows=16, drafter_dir=tmp_path/'draft', drafter_bits=8,
                            lossless_head=True)
    from tensorfold.cuda.requant import CHUNK_ROWS
    rows = min(CHUNK_ROWS, cfg.vocab, cfg.max_position)
    # Fixture's DF-source/quantizer bound dominates the original native stage.
    # Add every NEW ID8/rawBF16(2K)/returnedFP8(K)/scale4 owner explicitly.
    assert packed['weights'].staging == raw['weights'].staging + rows * (3 * cfg.hidden + 12)
