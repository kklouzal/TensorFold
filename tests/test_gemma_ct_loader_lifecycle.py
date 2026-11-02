"""CT raw-loader ownership/retirement controls; fake CUDA is not numerical proof."""
from types import SimpleNamespace
import weakref

import pytest
import torch

from gemma_cuda_target_nvfp4_metadata_test import fixture
from tensorfold.families.gemma4.cuda import nvfp4_weights as loader
from tensorfold.families.gemma4.cuda.modelopt_weights import ModelOptLoadStatus


def fake_ct(monkeypatch, tmp_path, *, drain_error=None, native_error=None):
    from tensorfold import checkpoint_metadata
    from tensorfold.cuda import direct_read
    from tensorfold.cuda.nvfp4 import checkpoint, linear
    from tensorfold.families.gemma4.cuda import nvfp4_experts
    raw, headers = fixture()
    (tmp_path / 'model.safetensors').write_bytes(b'CPU transport fixture')
    retired, reads = [], []
    read_refs = {}
    class Source:
        def __init__(self, files, reader=None):
            self.reader = reader or self
            self.where = {name: (tmp_path/'model.safetensors', 0, 1, v['dtype'], v['shape'])
                          for name, v in headers.items()} if files else {}
        def get(self, name, device='cpu'):
            reads.append((name, str(device)))
            v = headers[name]
            result = torch.ones(v['shape'], dtype=torch.float32).to(direct_read.DTYPES[v['dtype']])
            read_refs[name] = weakref.ref(result)
            if native_error is not None and name.endswith('norm.weight') and '.layers.' not in name:
                # CPU stand-ins for unreturned destination and pinned staging
                # owned by Reader._to_device's failed copied.synchronize frame.
                staging = torch.zeros(16, dtype=torch.uint8)
                read_refs['pending_staging'] = weakref.ref(staging)
                def failed_copy_fence():
                    raise native_error
                copied = SimpleNamespace(synchronize=failed_copy_fence)
                copied.synchronize()
            return result
        def close(self):
            retired.append('close')
    def drain(*args):
        retired.append('drain')
        if drain_error is not None:
            raise drain_error
    monkeypatch.setattr(checkpoint_metadata, 'read_config', lambda _: raw)
    monkeypatch.setattr(checkpoint_metadata, 'read_json', lambda *a: None)
    monkeypatch.setattr(direct_read, 'SafeTensors', Source)
    monkeypatch.setattr(checkpoint, 'available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda *a: (12, 1))
    monkeypatch.setattr(torch.cuda, 'synchronize', drain)
    monkeypatch.setattr(loader, '_calibrations', lambda reader, native: {name: (1., 1., 1.) for name in native})
    monkeypatch.setattr(linear.Fp4Linear, 'from_checkpoint', lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(nvfp4_experts, 'load_experts', lambda *a, **k: SimpleNamespace())
    status = ModelOptLoadStatus()
    status.test_read_refs = read_refs
    return status, retired, reads


def test_ct_raw_tied_head_retires_before_return(monkeypatch, tmp_path):
    from tensorfold.families.qwen3_5.cuda.weights import Plain
    status, retired, reads = fake_ct(monkeypatch, tmp_path)
    result = loader.load(tmp_path, status=status)
    assert result.numerics == 'ct_nvfp4' and type(result.embed) is Plain
    assert [device for name, device in reads if name.endswith('embed_tokens.weight')] == ['cuda:0']
    assert retired == ['drain', 'close'] and status.safe_to_release and status.transferred
    assert not status.owners


def test_ct_failed_final_fence_retains_complete_model(monkeypatch, tmp_path):
    primary = RuntimeError('unknown final drain')
    status, retired, _ = fake_ct(monkeypatch, tmp_path, drain_error=primary)
    with pytest.raises(RuntimeError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and retired == ['drain']
    assert not status.safe_to_release and status.owners['result'].embed is status.owners['embed']
    assert all(status.owners['layers']) and not status.transferred


def test_ct_internal_copy_fence_failure_is_terminal_without_outer_retry(monkeypatch, tmp_path):
    primary = RuntimeError('copied.synchronize failure')
    native = OSError('native copy cause')
    primary.__cause__ = native
    drain = AssertionError('outer CUDA retry is forbidden')
    status, retired, _ = fake_ct(monkeypatch, tmp_path, native_error=primary, drain_error=drain)
    with pytest.raises(RuntimeError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and retired == []
    assert not status.safe_to_release and not status.transferred
    assert status.owners['embed'] is not None and status.owners['reader'] is not None
    assert status.owners['read_failed'] == [True]
    assert status.test_read_refs['model.language_model.norm.weight']() is not None
    assert status.test_read_refs['pending_staging']() is not None
    tb = primary.__traceback__
    while tb is not None and tb.tb_frame.f_locals.get('_native_cause') is not native:
        tb = tb.tb_next
    assert tb is not None


def test_ct_safe_layer_publication_failure_releases_retained_frame_fields(monkeypatch, tmp_path):
    import gc
    status, retired, _ = fake_ct(monkeypatch, tmp_path)
    primary = MemoryError('layer publication')
    original = loader.Layer
    class FailedLayer(original):
        def __init__(self, **kwargs):
            try:
                raise primary
            finally:
                kwargs.clear()  # isolate caller aliases, not deliberate callee retention
    monkeypatch.setattr(loader, 'Layer', FailedLayer)
    with pytest.raises(MemoryError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and retired == ['drain', 'close']
    assert status.safe_to_release and not status.owners
    gc.collect()
    assert all(ref() is None for ref in status.test_read_refs.values())


def test_ct_finish_entry_failure_preserves_native_primary_and_roots(monkeypatch, tmp_path):
    from tensorfold.families.gemma4.cuda import modelopt_weights
    primary = ValueError('native read failure')
    native = RuntimeError('original native cause')
    primary.__cause__ = native
    status, retired, _ = fake_ct(monkeypatch, tmp_path)
    original = loader.Layer
    class FailedLayer(original):
        def __init__(self, **kwargs):
            try:
                raise primary
            finally:
                kwargs.clear()
    monkeypatch.setattr(loader, 'Layer', FailedLayer)
    bookkeeping = MemoryError('finish call entry')
    def failed_entry(primary, *args):
        raise bookkeeping
    monkeypatch.setattr(modelopt_weights, '_finish', failed_entry)
    with pytest.raises(ValueError) as caught:
        loader.load(tmp_path, status=status)
    assert caught.value is primary and primary.__cause__ is bookkeeping and retired == ['drain', 'close']
    tb = primary.__traceback__
    while tb is not None and tb.tb_frame.f_locals.get('_native_cause') is not native:
        tb = tb.tb_next
    assert tb is not None
    assert status.safe_to_release and not status.owners
