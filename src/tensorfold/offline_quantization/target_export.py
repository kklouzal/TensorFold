"""Single-use native target export into a NEW private, unqualified staging tree.

The trusted caller owns process exclusivity, live admission and the calibrated
owner's retirement. Public ModelOpt export mutates that model: invoking it is a
terminal transition for pre-export oracle use, even when export fails. Failed
stages are retained for diagnosis, never published or silently retried. No
consumer, quality or serving qualification follows from this operation.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import warnings

import torch

from tensorfold.checkpoint_metadata import read_config
from tensorfold.cleanup import raise_failures
from tensorfold.cuda.capacity import headers
from tensorfold.cuda.nvfp4.format import LinearMetadata, config_block
from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES
from tensorfold.file_io import FileStreams
from tensorfold.families.gemma4.cuda.dense_loader import _projection_specs
from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig
from tensorfold.families.gemma4.dense_tensor_contract import text_tensor_contract
from .export_manifest import (
    CHUNK_BYTES, _STANDARD_METADATA, _hash_range, _input,
    _provenance, write_export_manifest,
)
from .target_calibration import (
    CalibratedTarget, _digest, _phase, _recheck_source, _runtime_proof,
)

_DERIVED = {'config.json', 'hf_quant_config.json', 'quantization_config.json',
            'quant_config.json', 'model.safetensors.index.json'}
_CONFIG_DESCRIPTIVE = {'_name_or_path', 'transformers_version', 'quantization_config', 'quantization'}


def _files(root):
    """Private output namespace; never follow provider-created directory links."""
    result = set()
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('export staging contains a provider-created symlink')
        if path.is_file():
            result.add(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ValueError('export staging contains a nonregular entry')
    return result


def _config_semantics(source, exported):
    """Keep source-declared semantics; producer owns added serialization defaults."""
    if source.get('model_type') == 'gemma4':
        if exported.get('model_type') != 'gemma4':
            raise ValueError('export changed source model type')
        from transformers import Gemma4Config
        # The pinned native config translates legacy vision RoPE and per-layer
        # declarations. Compare its effective semantics, not obsolete spellings.
        source = Gemma4Config.from_dict(deepcopy(source)).to_dict()
        exported = Gemma4Config.from_dict(deepcopy(exported)).to_dict()
    for key, value in source.items():
        if key in _CONFIG_DESCRIPTIVE:
            continue
        field = 'dtype' if key == 'torch_dtype' and key not in exported else key
        if field not in exported:
            raise ValueError(f'export removed source config field: {key}')
        actual = exported[field]
        if isinstance(value, dict):
            if not isinstance(actual, dict):
                raise ValueError(f'export changed source config object {key}')
            _config_semantics(value, actual)
        elif _digest(actual) != _digest(value):
            raise ValueError(f'export changed source config semantics: {key}')


def _restore_config_fields(source, exported):
    """Restore declarations omitted by the producer; never override contradictions."""
    for key, value in source.items():
        if key in _CONFIG_DESCRIPTIVE:
            continue
        field = 'dtype' if key == 'torch_dtype' and key not in exported else key
        if field not in exported:
            exported[field] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(exported[field], dict):
            _restore_config_fields(value, exported[field])


def _restore_assets(source, stage, records):
    source_owners = {tuple(item['identity'][:2]) for item in records.values()}
    for name in _files(stage):
        info = (stage / name).stat()
        if (info.st_dev, info.st_ino) in source_owners:
            raise ValueError('provider output aliases immutable source storage')
    assets = {name for name in records if not name.endswith('.safetensors') and name not in _DERIVED}
    # In particular remove an export-synthesized generation_config when absent
    # in SOURCE. Never replace the producer's derived quantized config/sidecars.
    # HF export copies Git attributes too; these are not runtime model assets.
    known = (set(_STANDARD_METADATA) | {'.gitattributes'}) - _DERIVED
    known.update(name for name in _files(stage) if name.endswith('.jinja'))
    for name in sorted(known - assets):
        path = stage / name
        if path.exists():
            path.unlink()
    original_config, config = read_config(source), read_config(stage)
    _restore_config_fields(original_config, config)
    _config_semantics(original_config, config)
    # Preserve producer map/list order, including ordered quantization groups.
    config_bytes = (json.dumps(config, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(config_bytes) > MAX_HEADER_BYTES:
        raise ValueError('restored derived config exceeds checkpoint metadata bound')
    observed = {}
    for name in sorted(assets | {'config.json'}):
        destination = stage / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with _input(source, name, observed) as (stream, before):
            if before.st_size != records[name]['identity'][2] or before.st_size > MAX_HEADER_BYTES:
                raise ValueError('source asset geometry changed')
            # Stage is private; no source output or caller-owned existing tree
            # is overwritten. Native export's generated copy is replaceable.
            scope, primary = FileStreams(max_files=1), None
            try:
                record, _ = scope.open(destination, os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW,
                                       'wb', max_bytes=MAX_HEADER_BYTES)
                output = record.stream
                if name == 'config.json':
                    output.write(config_bytes)
                else:
                    while chunk := stream.read(CHUNK_BYTES):
                        output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            except BaseException as error:
                primary = error
            try:
                errors = scope.drain()
            except BaseException as retirement:
                # A failed drain has no no-throw guarantee. Do not retry an
                # unknown retirement state; keep scope in this native frame.
                retirement_failure = retirement  # survives except-target clearing
                authority = primary if primary is not None else retirement_failure
                try:
                    authority._tensorfold_export_io_scope = scope
                except BaseException as retention:
                    raise_failures(authority, [retirement_failure, retention])
                if primary is not None:
                    raise_failures(primary, [retirement_failure])
                raise
            if not scope.retired:
                if primary is None:
                    primary = RuntimeError('source asset output scope did not retire')
                try:
                    primary._tensorfold_export_io_scope = scope
                except BaseException as retention:
                    # Scope/errors remain frame-owned if a foreign exception
                    # refuses annotations. Never replace the IO primary.
                    raise_failures(primary, [*errors, retention])
            raise_failures(primary, errors)
        with _input(stage, name, {}) as (output, before):
            expected_hash = (hashlib.sha256(config_bytes).hexdigest() if name == 'config.json'
                             else records[name]['sha256'])
            if _hash_range(output, before.st_size, name) != expected_hash:
                raise ValueError(f'authoritative source asset copy differs: {name}')
    return assets


def _inventory(source_inventory, cfg, stage, unused_policy, *, bf16_linears=()):
    """Header-only complete geometry and exact initial recipe, not payload quality."""
    raw = read_config(stage)
    exported_cfg = E2BConfig.from_architecture(raw)
    if exported_cfg != cfg:
        raise ValueError('export changed the source E4B architecture')
    block = config_block(raw)
    if (block is None or block.get('kv_cache_scheme') is not None
            or block.get('transform_config') or block.get('sparsity_config')):
        raise ValueError('export requires declared untransformed NVFP4 and BF16 KV')
    metadata = LinearMetadata(block)
    inventory = headers(stage)
    specs = _projection_specs(cfg, inventory, metadata)
    _, active, unused = text_tensor_contract(cfg)
    if (len(active) != 258 or set(specs) != {name.removesuffix('.weight') for name in active}
            or any(spec.kind != 'nvfp4' or spec.activation_bits != 4 for spec in specs.values())):
        raise ValueError('export must contain all258 selected W4A4 NVFP4 projections')
    expected = set(source_inventory) - active - unused
    if 'lm_head.weight' in inventory:
        raise ValueError('export must retain one canonical BF16 tied token/head authority')
    if unused_policy == 'retained-original-bf16':
        expected.update(unused)
    retained = set(expected)
    for module, spec in specs.items():
        expected.update(module + '.' + suffix for suffix, _, _ in spec.parts)
    if set(inventory) != expected:
        raise ValueError('export differs from complete source/media/packed/shared-KV inventory')
    for name in retained:
        original, actual = source_inventory[name], inventory[name]
        if original['dtype'] != 'BF16' or actual['dtype'] != 'BF16' or actual['shape'] != original['shape']:
            raise ValueError(f'export changed retained BF16 geometry: {name}')
    for module in bf16_linears:
        # Quantization's default Linear scheme must not be applied to norms or
        # embeddings merely because their state name ends in .weight.
        name = ('model.language_model.embed_tokens.weight' if module == 'lm_head' else module+'.weight')
        if name not in inventory or name not in retained:
            raise ValueError(f'export lost a protected native BF16 Linear: {module}')
        metadata.validate(module, {'weight': ('BF16', inventory[name]['shape'])}, checkpoint=True)
    return inventory, unused


def _unused_hashes(source, records, unused):
    # Only the optional54 discarded donor tensors need new payload hashes. The
    # original16GB source file hashes are REUSED, never reread wholesale here.
    from tensorfold.cuda.capacity import SIZES
    from tensorfold.cuda.tensor_file import read_header_stream
    result, observed = {}, {}
    for name in sorted(name for name in records if name.endswith('.safetensors')):
        with _input(source, name, observed) as (stream, _):
            offset, header = read_header_stream(stream, SIZES, label=name)
            for tensor in unused & header.keys():
                begin, end = header[tensor]['data_offsets']
                stream.seek(offset + begin)
                result[tensor] = _hash_range(stream, end - begin, tensor)
    if set(result) != unused:
        raise ValueError('source lacks its complete54 unused shared-KV tensors')
    return result


def export_target(owner, source, stage, *, admit_phase, source_revision,
                  max_shard_size=1_000_000_000,
                  unused_shared_kv_policy='omitted'):
    """Export calibrated live model once; caller MUST close/drain before reload.

    max_shard_size is a writer HINT, not a largest-tensor/host-allocation limit:
    E4B's BF16 PLE tensor alone is approximately5.6GB. The trusted live admission
    callback must charge provider state-dict/writer/single-tensor staging peaks.
    Neither this API nor the manifest publishes a canonical model directory.
    """
    if (not isinstance(owner, CalibratedTarget) or owner.model is None or owner._terminal
            or getattr(owner, 'export_transition', None) is not None
            or not isinstance(owner.receipt, dict)
            or owner.receipt.get('status') != 'calibrated-native-target-not-exported-or-quality-qualified'):
        raise ValueError('healthy live single-use calibrated target owner required')
    if not callable(admit_phase) or type(max_shard_size) is not int or max_shard_size <= 0:
        raise ValueError('trusted live admission and explicit positive shard-byte hint required')
    if unused_shared_kv_policy not in ('omitted', 'retained-original-bf16'):
        raise ValueError('explicit all-omitted or all-retained shared-KV policy required')
    source = Path(source).resolve(strict=True)
    if source != getattr(owner, 'source_root', None) or not getattr(owner, '_runtime_namespace', None):
        raise ValueError('export requires calibrated owner source and original runtime namespace binding')
    requested = Path(stage)
    stage = requested.parent.resolve(strict=True) / requested.name
    if stage.is_relative_to(source) or source.is_relative_to(stage) or os.path.lexists(stage):
        raise ValueError('export requires a NEW private staging directory disjoint from source')
    records = owner.receipt['source_file_hashes']
    runtime = owner.receipt['runtime']
    provenance = _provenance({
        'source_revision': source_revision, 'producer_git_revision': runtime['producer_commit'],
        'source_config_sha256': records['config.json']['sha256'],
        'source_bf16_payloads_sha256': _digest({name: item['sha256'] for name, item in records.items()
                                             if name.endswith('.safetensors')}),
        'source_bf16_payloads_scope': 'SHA256 canonical mapping of WHOLE source safetensors file SHA256, INCLUDING headers; NOT consumed-tensor proof',
        'producer_wheel_sha256': runtime['wheel_sha256'], 'runtime_sha256': _digest(runtime),
        'recipe_sha256': owner.receipt['recipe_sha256'], 'calibration_sha256': _digest(owner.receipt),
    })
    admissions = [_phase('before-target-export', admit_phase)]
    _recheck_source(source, records)
    namespace = owner._runtime_namespace
    if _runtime_proof(namespace) != runtime:
        raise ValueError('export runtime differs from actual calibration runtime')
    cfg = E2BConfig.from_dict(read_config(source))
    original_inventory = headers(source)
    if owner.model.lm_head.weight is not owner.model.get_input_embeddings().weight:
        raise ValueError('calibrated owner lost canonical tied BF16 head')
    _, active, _ = text_tensor_contract(cfg)
    bf16_linears = tuple(name for name, module in owner.model.named_modules()
                         if isinstance(module, torch.nn.Linear) and name+'.weight' not in active)
    stage.mkdir(mode=0o700)
    transition = {'status': 'export-started-model-may-be-mutated', 'stage': str(stage)}
    owner.export_transition = transition
    try:
        from modelopt.torch.export import export_hf_checkpoint
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            export_error = None
            try:
                export_hf_checkpoint(owner.model, dtype=torch.bfloat16, export_dir=stage,
                                     save_modelopt_state=False, max_shard_size=max_shard_size)
            except BaseException as error:
                export_error = error
            try:
                transition['warnings'] = [str(item.message) for item in captured]
            except BaseException as bookkeeping:
                if export_error is not None:
                    raise_failures(export_error, [bookkeeping])
                raise
            if export_error is not None:
                raise export_error
        if captured:
            raise ValueError('producer export warnings require explicit disposition before acceptance')
        admissions.append(_phase('after-target-export', admit_phase))
        _recheck_source(source, records)
        _files(stage)
        assets = _restore_assets(source, stage, records)
        _, unused = _inventory(original_inventory, cfg, stage, unused_shared_kv_policy,
                               bf16_linears=bf16_linears)
        allowed = set(assets) | _DERIVED | {name for name in _files(stage) if name.endswith('.safetensors')}
        if _files(stage) - allowed:
            raise ValueError('provider emitted unbound files outside the source/derived asset inventory')
        if not (stage / 'hf_quant_config.json').is_file():
            raise ValueError('public export did not emit its required quantization sidecar')
        admissions.append(_phase('before-export-manifest', admit_phase))
        _recheck_source(source, records)
        if _runtime_proof(namespace) != runtime:
            raise ValueError('producer runtime changed during target export')
        retained_hashes = (_unused_hashes(source, records, unused)
                           if unused_shared_kv_policy == 'retained-original-bf16' else {})
        # Validate only optional discarded tensor payloads here, not a duplicate
        # pass over all16GB of original or all packed export payloads.
        if retained_hashes:
            exported_hashes = _unused_hashes(stage, {name: None for name in _files(stage)
                                                   if name.endswith('.safetensors')}, unused)
            if exported_hashes != retained_hashes:
                raise ValueError('retained shared-KV payload differs from original BF16 source')
        manifest = write_export_manifest(stage, provenance, unused_shared_kv_policy=unused_shared_kv_policy,
                                         metadata_files=tuple(sorted(assets)))
        admissions.append(_phase('after-export-manifest', admit_phase))
        _recheck_source(source, records)
        if _runtime_proof(namespace) != runtime:
            raise ValueError('producer runtime changed during manifest hashing')
        transition.update(status='exported-staging-UNQUALIFIED', output_published=False)
        return {'schema_version': 1, **transition, 'manifest': manifest,
                'admissions': admissions, 'max_shard_size_hint_bytes': max_shard_size,
                'shard_hint_is_not_single_tensor_or_peak_bound': True,
                'consumer_reload_pending': True, 'quality_pending': True}
    except BaseException as error:
        try:
            transition['status'] = 'export-failed-owner-requires-retirement'
            error._tensorfold_calibrated_target = owner
            error._tensorfold_export_stage = stage
        except BaseException as bookkeeping:
            # owner/stage/transition stay traceback-frame-owned. An exception
            # subclass may reject annotations; that cannot replace its failure.
            raise_failures(error, [bookkeeping])
        raise
