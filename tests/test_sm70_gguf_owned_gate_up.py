"""Standalone incumbent GU caller contract; actual math needs native SM70 gates."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops


def test_prepared_standalone_gu_calls_only_gu_and_publishes_its_mutations():
    calls = []
    prepared = gguf_ops._OwnedGateUp(SimpleNamespace(experts_owned_gate_up=lambda *args: calls.append(args)), torch.device('cpu'))
    x = torch.empty((1, 2560), dtype=torch.float16)
    picks = torch.empty((1, 11), dtype=torch.int32)
    gate = SimpleNamespace(data=torch.empty(1, dtype=torch.uint8), ggml_type=22, column_offset=0)
    up = SimpleNamespace(data=torch.empty(1, dtype=torch.uint8), ggml_type=22, column_offset=0)
    local = torch.empty((1, 11), dtype=torch.int64)
    activation = torch.empty((11, 160), dtype=torch.float16)
    grouping = torch.empty(1, dtype=torch.int32)
    versions = [tensor._version for tensor in (x, picks, gate.data, up.data, local, activation, grouping)]
    assert prepared.run(x, picks, gate, up, first=256, local=local, activation=activation, grouping=grouping) is activation
    assert len(calls) == 1 and len(calls[0]) == 12
    assert calls[0][8] == 256
    assert [tensor._version for tensor in (x, picks, gate.data, up.data, local, activation, grouping)] == [
        version + int(index >= 4) for index, version in enumerate(versions)]
    with pytest.raises(ValueError, match='first0'):
        prepared.run(x, picks, gate, up, first=True, local=local, activation=activation, grouping=grouping)
