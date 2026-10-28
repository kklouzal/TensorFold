"""Real host ABI on CPU tensors; target GPU boundaries remain a separate gate."""
import gc
import importlib.util
import json
import os
from pathlib import Path
import weakref

import numpy as np
import pytest
import torch


@pytest.fixture(scope='module')
def native():
    location = os.environ.get('TENSORFOLD_STORAGE_METADATA_MODULE')
    if not location:
        pytest.skip('requires the explicitly compiled owned SM70 provider in the CPU SDK')
    path = Path(json.loads(Path(location).read_text())['module'])
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def old_signature(tensors):
    return tuple((tensor.data_ptr(), tuple(tensor.shape), tensor.dtype, tensor.device) for tensor in tensors)


def test_real_plain_metadata_content_changes_and_equivalent_views(native):
    tensor = torch.arange(12).reshape(3, 4)
    binding = native.storage_bindings((tensor,))
    assert binding.matches((tensor,)) is True
    tensor.fill_(7)
    assert binding.matches((tensor.view(3, 4),)) is True
    assert binding.checks == 2 and not binding.overflowed
    assert native.storage_bindings_host_bytes_per_state == 16384


@pytest.mark.parametrize('mutation', ('set', 'shape', 'dtype', 'stride', 'extent'))
def test_real_metadata_mutations_are_rejected(native, mutation):
    tensor = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    binding = native.storage_bindings((tensor,))
    prior = old_signature((tensor,))
    if mutation == 'set':
        tensor.set_(torch.ones_like(tensor))
    elif mutation == 'shape':
        tensor.resize_(2, 8)
    elif mutation == 'dtype':
        tensor.data = torch.ones_like(tensor, dtype=torch.float64)
    elif mutation == 'stride':
        tensor.as_strided_((4, 4), (1, 4))
        assert old_signature((tensor,)) == prior  # Preserve cold contiguous schema beyond old4fields.
    else:
        tensor.untyped_storage().resize_(tensor.untyped_storage().nbytes() + 64)
    assert binding.matches((tensor,)) is False


def test_inference_tensors_do_not_require_version_counter(native):
    with torch.inference_mode():
        tensor = torch.zeros(3)
        with pytest.raises(RuntimeError, match='version counter'):
            _ = tensor._version
        binding = native.storage_bindings((tensor,))
        assert binding.matches((tensor,)) is True
        tensor.set_(torch.ones(3))
        assert binding.matches((tensor,)) is False


def test_original_storage_survives_quiescent_tensor_rebind(native):
    array = np.arange(16, dtype=np.float32)
    weak = weakref.ref(array)
    tensor = torch.from_numpy(array)
    binding = native.storage_bindings((tensor,))
    tensor.set_(torch.zeros(16))  # Quiescent cut; no GPU consumers here.
    del array
    gc.collect()
    assert weak() is not None and binding.matches((tensor,)) is False
    del binding
    gc.collect()
    assert weak() is None


def test_tensor_subclass_and_oversized_domain_preserve_python_path(native):
    class Other(torch.Tensor):
        pass
    tensor = torch.ones(3)
    binding = native.storage_bindings((tensor,))
    assert native.storage_bindings((tensor.as_subclass(Other),)) is None
    assert binding.matches((tensor.as_subclass(Other),)) is None
    assert native.storage_bindings((tensor,) * 70) is None
    assert binding.matches((tensor,) * 70) is None


def test_later_metadata_failure_is_not_hidden_by_earlier_mismatch(native):
    first = torch.ones(3)
    binding = native.storage_bindings((first, first))
    unsupported = torch.ones(2, 2, 2, 2)
    # Unsupported domains return the authority to the original full predicate.
    assert binding.matches((torch.ones(4), unsupported)) is None
