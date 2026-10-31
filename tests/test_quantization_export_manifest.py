"""Small fresh public-format fixture bytes; no neural payloads/ModelOpt/GPU."""
import hashlib
import json
import os
import struct

import pytest

from tensorfold.offline_quantization import export_manifest as export


def sha(value):
    return hashlib.sha256(value).hexdigest()


def safetensor(path, tensors):
    header, payload, offset = {}, bytearray(), 0
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [offset, offset + len(data)]}
        payload.extend(data)
        offset += len(data)
    raw = json.dumps(header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    path.write_bytes(struct.pack('<Q', len(raw)) + raw + payload)


@pytest.fixture
def provenance():
    return {'source_revision': '1' * 40, 'producer_git_revision': '2' * 40,
            **{key: str(index) * 64 for index, key in enumerate(export._PROVENANCE_HASHES, 3)}}


@pytest.fixture
def root(tmp_path):
    (tmp_path / 'config.json').write_bytes(b'{"model_type": "synthetic-export-fixture"}\n')
    (tmp_path / 'hf_quant_config.json').write_bytes(b'{"quantization":{"quant_algo":"NVFP4"}}\n')
    safetensor(tmp_path / 'model.safetensors', {'a': ('U8', [3], b'\x00\xff\x01'),
                                            'b': ('BF16', [2], b'\x80\x3f\x00\x3f')})
    return tmp_path


def build(root, provenance, **kwargs):
    return export.build_export_manifest(root, provenance, unused_shared_kv_policy='omitted', **kwargs)


def test_raw_tensor_hashes_and_raw_binary_metadata_independent_oracles(root, provenance):
    (root / 'tokenizer.model').write_bytes(b'\x00\xff\xfe\x80binary-model')
    (root / 'chat_template-instruct.jinja').write_bytes(b'{{ messages }}\n')
    (root / 'assets').mkdir()
    (root / 'assets' / 'vocab.bin').write_bytes(b'\x00\xc0\xffasset')
    (root / 'merges.txt').write_bytes(b'')
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
    manifest = build(root, provenance, metadata_files=['assets/vocab.bin'])
    assert manifest['schema_version'] == 1 and manifest['status'] == 'UNQUALIFIED'
    assert manifest['tensor_sha256'] == {'a': sha(b'\x00\xff\x01'), 'b': sha(b'\x80\x3f\x00\x3f')}
    assert manifest['metadata_sha256'] == {str(p.relative_to(root)): sha(data)
                                         for p, data in before.items() if p.suffix != '.safetensors'}
    assert manifest['inventory'] == {'tensors': 2, 'shards': 1, 'tensor_payload_bytes': 7}
    assert not manifest['quality_or_runtime_verified']
    assert {p: p.read_bytes() for p in before} == before
    assert not (root / export.MANIFEST_FILE).exists()
    assert manifest == build(root, provenance, metadata_files=['assets/vocab.bin'])


def test_exact_sharded_index_and_empty_tensor(root, provenance):
    (root / 'model.safetensors').unlink()
    safetensor(root / 'first.safetensors', {'a': ('U8', [2], b'ab')})
    safetensor(root / 'second.safetensors', {'empty': ('U8', [0], b'')})
    mapping = {'a': 'first.safetensors', 'empty': 'second.safetensors'}
    index = json.dumps({'weight_map': mapping}).encode()
    (root / 'model.safetensors.index.json').write_bytes(index)
    result = build(root, provenance)
    assert result['tensor_sha256'] == {'a': sha(b'ab'), 'empty': sha(b'')}
    assert result['metadata_sha256']['model.safetensors.index.json'] == sha(index)


@pytest.mark.parametrize('mutation', ['missing-name', 'extra-name', 'wrong-shard', 'missing-file', 'escape'])
def test_index_inventory_and_placement_must_match_headers(root, provenance, mutation):
    (root / 'model.safetensors').unlink()
    safetensor(root / 'one.safetensors', {'a': ('U8', [1], b'a')})
    safetensor(root / 'two.safetensors', {'b': ('U8', [1], b'b')})
    mapping = {'a': 'one.safetensors', 'b': 'two.safetensors'}
    if mutation == 'missing-name':
        mapping = {'a': 'one.safetensors', 'c': 'two.safetensors'}
    elif mutation == 'extra-name':
        mapping['c'] = 'one.safetensors'
    elif mutation == 'wrong-shard':
        mapping = {'a': 'two.safetensors', 'b': 'one.safetensors'}
    elif mutation == 'missing-file':
        mapping['b'] = 'missing.safetensors'
    else:
        mapping['b'] = '../outside.safetensors'
    (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': mapping}))
    with pytest.raises(ValueError):
        build(root, provenance)


def test_physical_shard_alias_rejected_but_distinct_equal_tensors_preserved(root, provenance):
    os.link(root / 'model.safetensors', root / 'alias.safetensors')
    with pytest.raises(ValueError, match='alias'):
        build(root, provenance)
    (root / 'alias.safetensors').unlink()
    safetensor(root / 'model.safetensors', {'a': ('U8', [2], b'xx'), 'b': ('U8', [2], b'xx')})
    assert build(root, provenance)['tensor_sha256'] == {'a': sha(b'xx'), 'b': sha(b'xx')}


def test_authorized_internal_file_alias_and_external_escape(root, provenance, tmp_path):
    (root / 'model.safetensors').rename(root / 'payload.blob')
    (root / 'model.safetensors').symlink_to('payload.blob')
    assert build(root, provenance)['inventory']['tensors'] == 2
    outside = root.parent / (root.name + '-outside.bin')
    outside.write_bytes(b'outside')
    (root / 'tokenizer.model').symlink_to(outside)
    with pytest.raises(ValueError, match='authorized roots'):
        build(root, provenance)


def test_malformed_header_and_duplicate_metadata_keys_rejected(root, provenance):
    (root / 'model.safetensors').write_bytes(b'\xff' * 16)
    with pytest.raises(ValueError):
        build(root, provenance)
    safetensor(root / 'model.safetensors', {'a': ('U8', [1], b'a')})
    (root / 'config.json').write_text('{"x":1,"x":2}')
    with pytest.raises(ValueError, match='duplicate'):
        build(root, provenance)


def test_tensor_mutation_during_hash_fails_without_writing_manifest(root, provenance, monkeypatch):
    original = export._hash_range
    changed = False
    def hash_then_change(stream, count, label):
        nonlocal changed
        result = original(stream, count, label)
        if not changed:
            path = root / 'model.safetensors'
            data = bytearray(path.read_bytes())
            data[-1] ^= 1
            path.write_bytes(data)
            changed = True
        return result
    monkeypatch.setattr(export, '_hash_range', hash_then_change)
    with pytest.raises(ValueError, match='changed'):
        build(root, provenance)
    assert not (root / export.MANIFEST_FILE).exists()


def test_atomic_no_overwrite_preserves_weights_and_refuses_existing_symlink(root, provenance):
    weights = (root / 'model.safetensors').read_bytes()
    result = export.write_export_manifest(root, provenance, unused_shared_kv_policy='retained-original-bf16')
    path = root / export.MANIFEST_FILE
    assert json.loads(path.read_bytes()) == result
    assert (root / 'model.safetensors').read_bytes() == weights
    assert not list(root.glob('.tensorfold-quant-*.tmp'))
    with pytest.raises(FileExistsError):
        export.write_export_manifest(root, provenance, unused_shared_kv_policy='omitted')
    path.unlink()
    path.symlink_to('missing-manifest')
    with pytest.raises(FileExistsError):
        export.write_export_manifest(root, provenance, unused_shared_kv_policy='omitted')
    assert path.is_symlink()


def test_publication_race_never_replaces_competing_manifest(root, provenance, monkeypatch):
    original = os.link
    def race(src, dst, **kwargs):
        (root / export.MANIFEST_FILE).write_bytes(b'other-owner')
        return original(src, dst, **kwargs)
    monkeypatch.setattr(export.os, 'link', race)
    with pytest.raises(FileExistsError):
        export.write_export_manifest(root, provenance, unused_shared_kv_policy='omitted')
    assert (root / export.MANIFEST_FILE).read_bytes() == b'other-owner'
    assert not list(root.glob('.tensorfold-quant-*.tmp'))


def test_post_link_fsync_error_reports_commit_and_keeps_complete_manifest(root, provenance, monkeypatch):
    original = os.fsync
    calls = 0
    def fsync(fd):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError('directory durability failed')
        return original(fd)
    monkeypatch.setattr(export.os, 'fsync', fsync)
    with pytest.raises(OSError, match='durability') as caught:
        export.write_export_manifest(root, provenance, unused_shared_kv_policy='omitted')
    assert json.loads((root / export.MANIFEST_FILE).read_bytes())['status'] == 'UNQUALIFIED'
    assert any('committed' in note for note in caught.value.__notes__)


def test_provenance_and_missing_config_fail_before_tensor_hash(root, provenance, monkeypatch):
    monkeypatch.setattr(export, '_hash_range', lambda *args: pytest.fail('invalid metadata reached payload hashing'))
    bad = dict(provenance)
    bad['calibration_sha256'] = 'bad'
    with pytest.raises(ValueError):
        build(root, bad)
    (root / 'config.json').unlink()
    with pytest.raises(ValueError, match='config.json'):
        build(root, provenance)


def test_valid_index_dot_paths_follow_owned_reader_authority(root, provenance):
    index = {'weight_map': {'a': './model.safetensors', 'b': 'model.safetensors'}}
    (root / 'model.safetensors.index.json').write_text(json.dumps(index))
    result = build(root, provenance)
    assert set(result['tensor_sha256']) == {'a', 'b'}


def test_namespace_added_after_tensor_read_is_rejected(root, provenance, monkeypatch):
    original = export._hash_range
    changed = False

    def add_file(stream, count, label):
        nonlocal changed
        result = original(stream, count, label)
        if not changed:
            safetensor(root / 'unexpected.safetensors', {'c': ('U8', [1], b'c')})
            changed = True
        return result

    monkeypatch.setattr(export, '_hash_range', add_file)
    with pytest.raises(ValueError, match='namespace changed'):
        build(root, provenance)


def test_metadata_replaced_after_parse_cannot_publish_unvalidated_version(root, provenance, monkeypatch):
    original = export._metadata_object
    changed = False

    def parse_then_change(raw, label):
        nonlocal changed
        result = original(raw, label)
        if label == 'config.json' and not changed:
            (root / 'config.json').write_bytes(b'invalid JSON replacement')
            changed = True
        return result

    monkeypatch.setattr(export, '_metadata_object', parse_then_change)
    with pytest.raises(ValueError, match='changed'):
        build(root, provenance)


def test_writer_manifest_roundtrips_through_actual_loader_metadata_gate(root, provenance):
    from tensorfold.cuda.capacity import headers
    from tensorfold.families.gemma4.cuda.dense_loader import _manifest

    (root / 'merges.txt').write_bytes(b'')
    expected = export.write_export_manifest(root, provenance, unused_shared_kv_policy='omitted')
    assert _manifest(root, headers(root)) == expected


def test_completed_temp_namespace_swap_cannot_return_success(root, provenance, monkeypatch):
    original = os.link

    def replace_source(src, dst, **kwargs):
        path = root / src
        path.unlink()
        path.write_bytes(b'wrong-owner')
        return original(src, dst, **kwargs)

    monkeypatch.setattr(export.os, 'link', replace_source)
    with pytest.raises(ValueError, match='completed output owner') as caught:
        export.write_export_manifest(root, provenance, unused_shared_kv_policy='omitted')
    assert any('committed' in note for note in caught.value.__notes__)
    assert (root / export.MANIFEST_FILE).read_bytes() == b'wrong-owner'


def test_successful_link_followed_by_interruption_still_reports_committed(root, provenance, monkeypatch):
    original = os.link

    def interrupt_after_link(src, dst, **kwargs):
        original(src, dst, **kwargs)
        raise InterruptedError('after committed link')

    monkeypatch.setattr(export.os, 'link', interrupt_after_link)
    with pytest.raises(InterruptedError) as caught:
        export.write_export_manifest(root, provenance, unused_shared_kv_policy='omitted')
    assert any('committed' in note for note in caught.value.__notes__)
    assert json.loads((root / export.MANIFEST_FILE).read_bytes())['status'] == 'UNQUALIFIED'


def test_filesystem_surrogate_names_cannot_create_unreadable_json_manifest(root, provenance):
    path = root / os.fsdecode(b'bad\xff.jinja')
    path.write_bytes(b'example')
    with pytest.raises(UnicodeError):
        build(root, provenance)
