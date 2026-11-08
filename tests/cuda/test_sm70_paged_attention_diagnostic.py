"""Seeded numerical diagnostic, deliberately without a numerical pass threshold.

Run with pytest --junitxml to retain numerical_diagnostic records. Successful
execution validates fixture/protocol only, not native numerical admissibility.
"""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tests.paged_attention_diagnostic import differences, tensor_sha
from tests.cuda.test_sm70_paged_attention import shard, statistics, sm70  # noqa: F401
from tensorfold.cuda.sm70 import paged_attention as operator
from tensorfold.families.qwen4_exp import kv_formats
from tensorfold.families.qwen4_exp.cuda import history_owner as oracle


def cache_for(state, pair):
    cache = NS(k=state.k, v=state.v, ks=state.ks, vs=state.vs,
               page_offsets=state.page_offsets, logical_capacity=512,
               placement=NS(page_tokens=128), capacity=state.k.shape[0],
               key_format=pair.key, value_format=pair.value)
    cache.local_indices = oracle.OwnedKVCache.local_indices.__get__(cache)
    return cache


def source_identity():
    root = Path(__file__).resolve().parents[2]
    names = [__file__, oracle.__file__, operator.__file__,
             root / "src/tensorfold/cuda/sm70/paged_attention.cu",
             root / "src/tensorfold/cuda/sm70/paged_attention_descriptor.cuh",
             root / "src/tensorfold/cuda/sm70/packed_kv_decode.cuh",
             root / "src/tensorfold/cuda/sm70/paged_attention_bindings.cpp",
             root / "src/tensorfold/cuda/sm70/paged_attention_descriptor.py",
             root / "src/tensorfold/cuda/sm70/rotorquant_q4.cuh",
             root / "src/tensorfold/cuda/sm70/rotorquant_q4_tables.cuh",
             oracle.ops.__file__, kv_formats.__file__,
             root / "tests/paged_attention_diagnostic.py"]
    return {str(Path(name).relative_to(root)): hashlib.sha256(Path(name).read_bytes()).hexdigest()
            for name in names}


@pytest.mark.parametrize("format_name", ["int8", "rotorquant-iso4-sm70-v1"])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("group", [0, 4])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("case", ["nonuniform", "cancellation", "sharp"])
def test_seeded_owner_numerical_diagnostic(format_name, degree, group, sparse, case, record_property):
    seed = 117000 + ["nonuniform", "cancellation", "sharp"].index(case)
    generator = torch.Generator().manual_seed(seed)
    key = torch.randn((512, 1, 256), generator=generator) * .5
    value = torch.randn((512, 1, 256), generator=generator)
    query = torch.randn((8, 2, 3, 256), generator=generator) * .5
    if case == "cancellation":
        value += torch.where(torch.arange(512)[:, None, None] % 2 == 0, 16., -16.)
        query[..., :128] += 1
        query[..., 128:] -= 1
    elif case == "sharp":
        key *= 8
        query *= 16
    pair = kv_formats.get_pair(format_name)
    fmt = (pair.key.bits, pair.key.codec)
    states = [shard(degree, rank, group=group, sparse=sparse, key=fmt, value=fmt) for rank in range(4)]
    status, layer = states[0].status, 3
    for state in states:
        if not sparse:
            state.position.fill_(259)
        else:
            # Cross both the native TILE32 and original TILE128 boundaries.
            selection = torch.zeros((8, 2, 262), dtype=torch.int32, device="cuda")
            ids = torch.cat((torch.tensor([-1], device="cuda"), torch.arange(0, 512, 2, device="cuda"),
                             torch.tensor([511, 512, 900], device="cuda")))
            selection[0, :, :260] = ids
            selection[0, :, -2:] = torch.tensor([260, 1], device="cuda")
            state.selection = selection
    packed = [oracle._encode(x.cuda().half(), form, status, layer, error=1)
              for x, form in [(key, pair.key), (value, pair.value)]]
    query = query.cuda().half()
    if pair.key.codec:
        transformed = torch.empty_like(query)
        oracle.ops._rotor_rotate(query.reshape(16, 3, 256), transformed.view(16, 3, 256),
                                 codec=pair.key.codec, status=status, layer=layer, error=1)
    else:
        transformed = oracle.ops.h32(query).half()
    for state in states:
        state.query.copy_(transformed)
        for page, offset in enumerate(state.page_offsets.cpu().tolist()):
            if offset >= 0:
                for data, meta, (source, scales) in [(state.k, state.ks, packed[0]), (state.v, state.vs, packed[1])]:
                    data[offset:offset+128].copy_(source[page*128:(page+1)*128])
                    meta[offset:offset+128].copy_(scales[page*128:(page+1)*128])
    gathered = torch.zeros((8, 2, 12, 2), device="cuda")
    gathered[..., 0].fill_(-torch.inf)
    values = torch.zeros((8, 2, 12, 256), device="cuda")
    for rank, state in enumerate(states):
        gathered[group+rank].copy_(statistics(state))
    for rank, state in enumerate(states):
        operator.merge_statistics(gathered, state.table, state.global_stats, group_start=group, page_tokens=128, layer=state.layer)
        operator.normalized_partial(state.query, state.selection, state.table, state.global_stats, state.values,
                                    group_start=group, page_tokens=128, layer=state.layer, key_bits=fmt[0], key_codec=fmt[1],
                                    value_bits=fmt[0], value_codec=fmt[1], scale=1/16)
        values[group+rank].copy_(state.values)
    # All four owners must be published before the value merge.
    for rank, state in enumerate(states):
        operator.merge_values(values, state.table, state.out, rank=group+rank, page_tokens=128,
                              layer=state.layer, value_bits=fmt[0], value_codec=fmt[1])
    observations = []
    for row in range(int(states[0].live.item())):
        end = int(states[0].position.item()) + row + 1
        ids = states[0].selection[0, row, :260].long() if sparse else torch.arange(end, device="cuda")
        visible = ids < end
        q = transformed[group:group+4, row].reshape(12, 256)
        caches = [cache_for(state, pair) for state in states]
        partials = [oracle.partial_statistics(q, cache, ids, visible, 1/16) for cache in caches]
        local = torch.stack([torch.stack(item, -1) for item in partials])
        maximum, denominator = oracle.merge_statistics(local[..., 0], local[..., 1], status=status, layer=layer)
        original_values = [oracle.normalized_partial(q, cache, ids, visible, 1/16, maximum, denominator) for cache in caches]
        for rank, state in enumerate(states):
            result = original_values[0][rank*3:rank*3+3]
            for peer in range(1, 4):
                result = result + original_values[peer][rank*3:rank*3+3]
            if pair.value.codec:
                output = torch.empty((1, 3, 256), dtype=torch.float16, device="cuda")
                oracle.ops._rotor_rotate(result[None], output, codec=pair.value.codec, inverse=True, status=status, layer=layer)
                output = output[0]
            else:
                output = oracle.ops.h32(result).half()
            observations.append({"row": row, "owner": rank,
                                 "local_statistics": differences(local[rank], state.stats[row]),
                                 "global_statistics": differences(torch.stack((maximum, denominator), -1), state.global_stats[row]),
                                 "normalized_partial": differences(original_values[rank], state.values[row]),
                                 "output": differences(output, state.out[row])})
    for state in states:
        assert state.status.cpu().tolist() == [0, -2]
        assert state.generations.cpu().tolist() == [0, 0]
        if not sparse:
            assert torch.equal(state.out[1], torch.zeros_like(state.out[1]))
    report = {"schema": "sm70-owner-numerical-diagnostic-v1", "diagnostic_only": True,
              "acceptance_threshold": None, "seed": seed, "case": case, "format": format_name,
              "bits_codec": fmt, "format_storage_uid": pair.key.storage_uid,
              "descriptor_abi": operator.ABI_VERSION, "descriptor_schema": operator.SCHEMA_ID,
              "page_offsets": [state.page_offsets.cpu().tolist() for state in states],
              "position": int(states[0].position.item()), "live_rows": int(states[0].live.item()),
              "selection_sha256": tensor_sha(states[0].selection), "query_sha256": tensor_sha(transformed),
              "degree": degree, "group": group, "sparse": sparse,
              "torch": torch.__version__, "device": torch.cuda.get_device_name(), "capability": [7, 0],
              "sources": source_identity(), "raw_input_sha256": [tensor_sha(x) for x in (key, value, query)],
              "packed_sha256": [[tensor_sha(x) for x in item] for item in packed], "observations": observations}
    encoded = json.dumps(report, sort_keys=True, allow_nan=False)
    record_property("numerical_diagnostic", encoded)
    print(encoded)
