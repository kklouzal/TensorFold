"""Preparation-only storage contract for the dense E4B serving adapter.

Only current ModelOpt NVFP4 W4A4 text-core exports are admitted. Header bounds
cover logical owned tensors, not allocator reservation or physical feasibility;
the existing capacity/host policy is mandatory, including during inference.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

from tensorfold.checkpoint_metadata import read_config
from tensorfold.cuda.capacity import headers, itemsize
from tensorfold.cuda.nvfp4.format import LinearMetadata, config_block
from ..dense_tensor_contract import text_tensor_contract
from .e2b_metadata import E2BConfig


@dataclass(frozen=True)
class PackedStorage:
    resident: int
    device_staging: int
    host_staging: int
    projection_shapes: tuple[tuple[int, int], ...]
    native_buffer_bound: int

    def workspace(self, rows):
        # checkpoint.quant4 pads prompt rows to at most128; lane scales to64.
        # Include contiguous BF16 input/output. qmm.split_k caps at8 and
        # checkpoint._lane allocates split scratch only above8.
        padded = math.ceil(rows / 128) * 128
        return max(2*rows*(n+k) + padded*k//2 + padded*k//16
                   for n, k in self.projection_shapes)


def _native_topology(raw, cfg, inventory, specs, metadata, unused_policy):
    """Bound constructor loops, then compare native META shapes before CPU owners.

    META is used only as an allocation-free schema oracle. Its values never
    initialize the real model or escape this preparation scope.
    """
    # This is the original E4B media allocation profile, not a claim to support
    # arbitrary media architectures just because their model_type matches.
    profiles = {
        'vision_config': {'hidden_size': 768, 'intermediate_size': 3072,
                          'num_hidden_layers': 16, 'num_attention_heads': 12,
                          'num_key_value_heads': 12, 'head_dim': 64, 'global_head_dim': 64,
                          'patch_size': 16, 'pooling_kernel_size': 3, 'position_embedding_size': 10240},
        'audio_config': {'hidden_size': 1024, 'num_hidden_layers': 12,
                         'num_attention_heads': 8, 'conv_kernel_size': 5,
                         'output_proj_dims': 1536, 'subsampling_conv_channels': [128, 32]},
    }
    for section, expected in profiles.items():
        for key, value in expected.items():
            actual = raw[section].get(key)
            if type(actual) is not type(value) or actual != value:
                raise ValueError(f"packed E4B native media allocation profile differs: {section}.{key}")
    import torch
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from transformers.initialization import no_init_weights
    from .dense_loader import _CONSTRUCTION, _validate_native_inventory
    model = value = None
    native = {}
    try:
        with _CONSTRUCTION:
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with torch.device('meta'), no_init_weights():
                    config = Gemma4Config.from_dict(raw)
                    config._attn_implementation = 'sdpa'
                    model = Gemma4ForConditionalGeneration(config)
            finally:
                torch.set_default_dtype(previous)
        if any(value.device.type != 'meta' for value in (*model.parameters(), *model.buffers())):
            raise ValueError("native topology preparation must own META tensors only")
        native, _ = _validate_native_inventory(model, cfg, inventory, specs, unused_policy, metadata)
        logical = sum(value.numel()*value.element_size() for name, value in native.items()
                      if name != 'lm_head.weight')
        ephemeral = sum(value.numel()*value.element_size() for name, value in model.named_buffers(remove_duplicate=False)
                        if name not in native)
        head = model.lm_head.weight.numel()*model.lm_head.weight.element_size()
        return logical, ephemeral, head
    finally:
        native.clear()
        model = value = None


def inspect_packed_storage(model_dir: Path) -> PackedStorage:
    """Header/manifest scope check; no payload or real parameter allocation.

    Loader independently revalidates identities when consuming actual bytes.
    """
    from .dense_loader import _manifest, _projection_specs
    raw = read_config(model_dir)
    cfg = E2BConfig.from_checkpoint(raw)
    block = config_block(raw)
    if block is None:
        raise ValueError("packed serving requires declared ModelOpt NVFP4 storage")
    inventory = headers(model_dir)
    manifest = _manifest(model_dir, inventory)
    metadata = LinearMetadata(block)
    specs = _projection_specs(cfg, inventory, metadata)
    contract, active, unused = text_tensor_contract(cfg)
    if (set(specs) != {name.removesuffix('.weight') for name in active}
            or any(spec.kind != 'nvfp4' or spec.activation_bits != 4
                   or spec.reciprocal_globals for spec in specs.values())):
        raise ValueError("packed E4B serving requires all258 active text projections in ModelOpt NVFP4 W4A4")
    packed_names = {name+'.'+suffix for name, spec in specs.items() for suffix, _, _ in spec.parts}
    retained = {name: info for name, info in inventory.items() if name not in packed_names and name not in unused}
    if any(info['dtype'] != 'BF16' for info in retained.values()):
        raise ValueError("packed E4B keeps all non-core tensors in original BF16")
    payload = sum(itemsize(info, name)*math.prod(info['shape']) for name, info in retained.items())
    # Validate complete native topology BEFORE constructing real CPU parameters;
    # missing or mismatched media headers cannot understate placeholder owners.
    native, buffers, head = _native_topology(raw, cfg, inventory, specs, metadata,
                                           manifest['unused_shared_kv_policy'])
    resident, staging = payload, 0
    for spec in specs.values():
        n, k = spec.shape
        padded = math.ceil(n/64)*64
        prepared = padded*k//2 + padded*k//16
        resident += prepared
        source = sum(itemsize({'dtype': dtype}, spec.name)*math.prod(shape)
                     for _, dtype, shape in spec.parts)
        # Source tensors + packed words + two scale layouts; float validation
        # needs two FP32 block-scale arrays and boolean intermediates (12B/block).
        staging = max(staging, source + prepared + padded*k//16 + 12*n*k//16)
    return PackedStorage(resident+buffers, staging, native+head+8*buffers+2*(64 << 20)+8192,
                         tuple(spec.shape for spec in specs.values()), buffers)
