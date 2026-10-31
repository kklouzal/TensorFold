"""Hash an existing immutable safetensors export and publish its integrity manifest.

No ModelOpt/runtime/model import, tensor allocation or weight mutation. The
producer owns a private export directory that remains immutable throughout
validation/publication. Observed descriptor/path changes fail; no claim is made
to defeat a privileged writer that restores every namespace observation.
Directory transaction publication and actual GPU quality remain caller-owned.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import secrets

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.capacity import SIZES
from tensorfold.cuda.tensor_file import (
    MAX_HEADER_BYTES, _metadata_object, _regular_stream, checkpoint_path,
    read_header_stream,
)

MANIFEST_FILE = 'tensorfold_quantization_manifest.json'
CHUNK_BYTES = 64 << 20
_STANDARD_METADATA = (
    'config.json', 'hf_quant_config.json', 'quantization_config.json', 'quant_config.json',
    'model.safetensors.index.json',
    'generation_config.json', 'tokenizer.json', 'tokenizer_config.json',
    'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt',
    'tokenizer.model', 'spiece.model', 'sentencepiece.model', 'sentencepiece.bpe.model',
    'vocab.txt', 'processor_config.json', 'preprocessor_config.json',
    'feature_extractor_config.json', 'video_preprocessor_config.json', 'audio_processor_config.json',
    'README.md', 'LICENSE', 'LICENSE.md', 'LICENSE.txt', 'NOTICE', 'NOTICE.md',
)
_PROVENANCE_HASHES = ('source_config_sha256', 'source_bf16_payloads_sha256', 'producer_wheel_sha256',
                      'runtime_sha256', 'recipe_sha256', 'calibration_sha256')


def _identity(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _sha(value, name, lengths=(64,)):
    if type(value) is not str or len(value) not in lengths or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be an immutable lowercase hexadecimal digest/revision')


def _provenance(value):
    if not isinstance(value, dict):
        raise ValueError('export provenance must be a JSON object')
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError('export provenance must be finite bounded JSON metadata') from error
    if len(encoded) > MAX_HEADER_BYTES:
        raise ValueError('export provenance exceeds checkpoint metadata bound')
    result = _metadata_object(encoded, 'export provenance')
    for name in _PROVENANCE_HASHES:
        _sha(result.get(name), name)
    for name in ('source_revision', 'producer_git_revision'):
        _sha(result.get(name), name, (40, 64))
    return result


def _relative(root, path):
    return str(path.relative_to(root))


def _discovery(root, extra_metadata):
    shards = tuple(sorted(_relative(root, p) for p in root.rglob('*.safetensors')))
    metadata = {name for name in _STANDARD_METADATA if os.path.lexists(root / name)}
    metadata.update(_relative(root, p) for p in root.glob('*.jinja'))
    metadata.update(_relative(root, p) for p in (root / 'chat_templates').glob('*.jinja'))
    metadata.update(extra_metadata)
    for name in (*shards, *metadata):
        name.encode('utf-8')  # Same string domain as the owned checkpoint JSON reader.
    if 'config.json' not in metadata:
        raise ValueError('export requires config.json')
    if not shards:
        raise ValueError('export contains no safetensors payload')
    return shards, tuple(sorted(metadata))


@contextmanager
def _input(root, name, observed):
    authorized = checkpoint_path(root, name)
    with _regular_stream(authorized, no_follow=True) as stream:
        before = os.fstat(stream.fileno())
        previous = observed.get(name)
        if previous is not None and previous != (authorized, _identity(before)):
            raise ValueError(f'{name}: export input changed between validated reads')
        yield stream, before
        after = os.fstat(stream.fileno())
        if (_identity(before) != _identity(after) or checkpoint_path(root, name) != authorized
                or _identity(authorized.stat(follow_symlinks=False)) != _identity(before)):
            raise ValueError(f'{name}: export input changed during hashing')
    observed[name] = (authorized, _identity(before))


def _hash_range(stream, count, label):
    digest = hashlib.sha256()
    while count:
        chunk = stream.read(min(CHUNK_BYTES, count))
        if not chunk:
            raise ValueError(f'{label}: export shortened during hashing')
        digest.update(chunk)
        count -= len(chunk)
    return digest.hexdigest()


def _metadata_hash(root, name, observed):
    with _input(root, name, observed) as (stream, before):
        if not 0 <= before.st_size <= MAX_HEADER_BYTES:
            raise ValueError(f'{name}: export metadata size exceeds checkpoint bound')
        parsed = None
        if name in _STANDARD_METADATA and name.endswith('.json'):
            # Existing strict parser, on the SAME opened version being hashed.
            raw = stream.read(before.st_size + 1)
            if len(raw) != before.st_size:
                raise ValueError(f'{name}: export metadata changed during read')
            parsed = _metadata_object(raw, name)
            digest = hashlib.sha256()
            view = memoryview(raw)
            for offset in range(0, len(raw), CHUNK_BYTES):
                digest.update(view[offset:offset + CHUNK_BYTES])
            result = digest.hexdigest()
        else:
            result = _hash_range(stream, before.st_size, name)
    return parsed, result


def _verify(root, discovery, extra_metadata, observed, root_identity):
    info = root.stat()
    if (info.st_dev, info.st_ino) != root_identity:
        raise ValueError('export directory identity changed')
    if _discovery(root, extra_metadata) != discovery:
        raise ValueError('export shard/metadata namespace changed')
    for name, (authorized, identity) in observed.items():
        if checkpoint_path(root, name) != authorized or _identity(authorized.stat(follow_symlinks=False)) != identity:
            raise ValueError(f'{name}: export input changed after hashing')


def _build(root, provenance, unused_shared_kv_policy, extra_metadata):
    root = Path(root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError('export root must be an existing directory')
    provenance = _provenance(provenance)
    if type(unused_shared_kv_policy) is not str or unused_shared_kv_policy not in ('omitted', 'retained-original-bf16'):
        raise ValueError('unused shared KV policy must be omitted or retained-original-bf16')
    if not isinstance(extra_metadata, (tuple, list)) or len(extra_metadata) > 4096:
        raise ValueError('extra metadata must be a bounded explicit path list')
    normalized_metadata = []
    for name in extra_metadata:
        checkpoint_path(root, name)
        normalized = Path(name).as_posix()
        if normalized == MANIFEST_FILE or normalized.endswith('.safetensors'):
            raise ValueError('tensor/manifest paths cannot be metadata assets')
        normalized_metadata.append(normalized)
    extra_metadata = tuple(normalized_metadata)
    root_info = root.stat()
    root_identity = root_info.st_dev, root_info.st_ino
    discovery = _discovery(root, extra_metadata)
    shards, metadata_names = discovery
    observed = {}
    metadata_sha, parsed_index = {}, None
    for name in metadata_names:
        parsed, digest = _metadata_hash(root, name, observed)
        metadata_sha[name] = digest
        if name == 'model.safetensors.index.json':
            parsed_index = parsed
        # Other JSON objects (notably large vocabularies) need no retained copy.
        del parsed
    index = None
    if parsed_index is not None:
        index = parsed_index.get('weight_map')
        if (not isinstance(index, dict) or not index
                or any(type(k) is not str or not k or type(v) is not str or not v for k, v in index.items())):
            raise ValueError('export index needs a nonempty string weight_map')
        for name in index.values():
            checkpoint_path(root, name)
        index = {name: Path(shard).as_posix() for name, shard in index.items()}
        if set(index.values()) != set(shards):
            raise ValueError('export index shard inventory differs from exported payload files')
    tensor_sha, located, seen_files = {}, {}, set()
    total = 0
    for name in shards:
        with _input(root, name, observed) as (stream, before):
            file_id = before.st_dev, before.st_ino
            if file_id in seen_files:
                raise ValueError('export shard paths alias the same physical payload')
            seen_files.add(file_id)
            offset, header = read_header_stream(stream, SIZES, label=name)
            for tensor, entry in header.items():
                if tensor == '__metadata__':
                    continue
                if not tensor or tensor in tensor_sha:
                    raise ValueError('export contains empty or duplicate canonical tensor names')
                begin, end = entry['data_offsets']
                stream.seek(offset + begin)
                tensor_sha[tensor] = _hash_range(stream, end - begin, tensor)
                located[tensor] = name
                total += end - begin
    if index is not None and (set(index) != set(tensor_sha) or any(located[name] != shard for name, shard in index.items())):
        raise ValueError('export index does not exactly locate every canonical tensor')
    if not tensor_sha:
        raise ValueError('export contains no canonical tensor entries')
    _verify(root, discovery, extra_metadata, observed, root_identity)
    manifest = {'schema_version': 1, 'status': 'UNQUALIFIED',
                'tensor_sha256': dict(sorted(tensor_sha.items())), 'metadata_sha256': metadata_sha,
                'unused_shared_kv_policy': unused_shared_kv_policy, 'provenance': provenance,
                'inventory': {'tensors': len(tensor_sha), 'shards': len(shards), 'tensor_payload_bytes': total},
                'quality_or_runtime_verified': False}
    return root, manifest, (discovery, extra_metadata, observed, root_identity)


def build_export_manifest(export_root, provenance, *, unused_shared_kv_policy, metadata_files=()):
    """Hash existing export bytes only; root/content must remain immutable.

    Every indexed/unindexed exported safetensors header name is represented.
    Extra binary metadata assets can be explicitly listed. Provenance is required
    attribution DATA, not proof of runtime/calibration/quality. No file is written.
    """
    return _build(export_root, provenance, unused_shared_kv_policy, metadata_files)[1]


def write_export_manifest(export_root, provenance, *, unused_shared_kv_policy, metadata_files=()):
    """Build once, atomically create WITHOUT replacement on the POSIX TARGET.

    This does not accept caller-supplied hash claims. A same-directory hard link
    provides no-clobber publication; parent owns whole-directory transaction
    publication. A post-link durability/cleanup error reports committed state
    and never deletes a possibly observed manifest.
    """
    from tensorfold._fd_owner import OwnedFD
    if os.name != 'posix':
        raise ValueError('atomic no-overwrite export publication requires the POSIX target')
    destination = Path(export_root).resolve(strict=True) / MANIFEST_FILE
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    root, manifest, snapshot = _build(export_root, provenance, unused_shared_kv_policy, metadata_files)
    raw = (json.dumps(manifest, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    if len(raw) > MAX_HEADER_BYTES:
        raise ValueError('export manifest exceeds checkpoint metadata bound')
    temporary = '.tensorfold-quant-' + secrets.token_hex(16) + '.tmp'
    # Valid closed native owners precede acquisition. All relative publication
    # and cleanup operations borrow the SAME opened directory descriptor.
    directory, output = OwnedFD(), OwnedFD()
    primary, errors, committed, temporary_identity = None, [], False, None
    try:
        directory.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, 0o600)
        info = os.fstat(directory.fileno())
        if (info.st_dev, info.st_ino) != snapshot[-1]:
            raise ValueError('export publication directory changed')
        output.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600, dir_fd=directory.fileno())
        view, written = memoryview(raw), 0
        while written < len(raw):
            count = os.write(output.fileno(), view[written:written + CHUNK_BYTES])
            if count <= 0:
                raise OSError('export manifest write was incomplete')
            written += count
        os.fsync(output.fileno())
        info = os.fstat(output.fileno())
        temporary_identity = info.st_dev, info.st_ino
        named = os.stat(temporary, dir_fd=directory.fileno(), follow_symlinks=False)
        if (named.st_dev, named.st_ino) != temporary_identity:
            raise ValueError('temporary export manifest changed before publication')
        _verify(root, *snapshot)
        os.link(temporary, MANIFEST_FILE, src_dir_fd=directory.fileno(),
                dst_dir_fd=directory.fileno(), follow_symlinks=False)
        committed = True
        named = os.stat(MANIFEST_FILE, dir_fd=directory.fileno(), follow_symlinks=False)
        if (named.st_dev, named.st_ino) != temporary_identity:
            raise ValueError('published export manifest does not name the completed output owner')
        os.fsync(directory.fileno())
    except BaseException as error:
        primary = error
    finally:
        # Even an interrupted post-open return leaves the acquired native owner
        # discoverable here. A failed O_EXCL acquisition never authorizes unlink.
        if not output.closed:
            try:
                info = os.fstat(output.fileno())
                temporary_identity = info.st_dev, info.st_ino
            except BaseException as error:
                errors.append(error)
        if temporary_identity is not None and not committed and not directory.closed:
            try:
                named = os.stat(MANIFEST_FILE, dir_fd=directory.fileno(), follow_symlinks=False)
                committed = (named.st_dev, named.st_ino) == temporary_identity
            except FileNotFoundError:
                pass
            except BaseException as error:
                errors.append(error)
        for _ in range(2):
            if output.closed:
                break
            try:
                output.close()
            except BaseException as error:
                errors.append(error)
        if not output.closed:
            if primary is None:
                primary = RuntimeError('export manifest output owner did not retire')
            primary._tensorfold_export_file_owner = output
        if temporary_identity is not None and output.closed and not directory.closed:
            try:
                info = os.stat(temporary, dir_fd=directory.fileno(), follow_symlinks=False)
                if (info.st_dev, info.st_ino) != temporary_identity:
                    raise ValueError('temporary export manifest path changed; refusing cleanup of another owner')
                os.unlink(temporary, dir_fd=directory.fileno())
                os.fsync(directory.fileno())
            except BaseException as error:
                errors.append(error)
        for _ in range(2):
            if directory.closed:
                break
            try:
                directory.close()
            except BaseException as error:
                errors.append(error)
        if not directory.closed:
            if primary is None:
                primary = RuntimeError('export directory owner did not retire')
            primary._tensorfold_export_directory = directory
        if committed and (primary is not None or errors):
            try:
                BaseException.add_note(primary if primary is not None else errors[0],
                                       f'export manifest committed in the owned directory originally at {destination}; durability/cleanup reported failure')
            except BaseException as error:
                errors.append(error)
        raise_failures(primary, errors)
    return manifest
