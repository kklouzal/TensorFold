"""Resolve local model directories or Hugging Face snapshots, downloading missing weights when requested."""

from __future__ import annotations

import json
import re
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
MOVED_ORG = "Vontra"           # the Hugging Face org TensorFold moved its models out of on 2 Oct 2026
METADATA_FILES = (
    "config.json", "hf_quant_config.json", "quantization_config.json", "quant_config.json",
    "generation_config.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json",
    "tokenizer.json", "chat_template.json", "chat_template.jinja", "additional_chat_templates/*.jinja",
)


def is_repo_id(name: str) -> bool:
    """``owner/name`` that is not an existing local path."""

    return bool(_REPO_ID.match(str(name))) and not Path(str(name)).expanduser().exists()


def cache_names(repo_id: str) -> list[str]:
    """Where ``repo_id`` can be cached: under its own org, then under the org it moved out of."""

    if not repo_id.startswith("TensorFold/"):
        return [repo_id]
    return [repo_id, f"{MOVED_ORG}/{repo_id.split('/', 1)[1]}"]


def cached(repo_id: str, *, cache_dir: Any = None, revision: str | None = None) -> Path | None:
    """Use the cached snapshot, falling back to the newest config-bearing snapshot when refs/main is absent.

    A ``TensorFold/<name>`` id also reads an older ``models--Vontra--<name>`` cache: the org moved, the old
    names redirect on Hugging Face, and caches downloaded before the move kept their old folder names.
    """

    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import LocalEntryNotFoundError

    if cache_dir is None:
        from huggingface_hub import constants

        cache_dir = constants.HF_HUB_CACHE
    for name in cache_names(repo_id):
        try:
            options = {} if revision is None else {"revision": revision}
            return Path(snapshot_download(name, local_files_only=True, cache_dir=cache_dir, **options))
        except LocalEntryNotFoundError as error:
            # Recent Hub versions report an incomplete selected snapshot with
            # its path. Metadata preparation must retain that exact revision,
            # rather than choosing another snapshot by mtime.
            incomplete = getattr(error, "snapshot_path", None)
            if incomplete is not None:
                return Path(incomplete)
        snapshots = Path(cache_dir) / f"models--{name.replace('/', '--')}" / "snapshots"
        if revision is not None:
            continue  # an exact revision never falls through to another cached revision
        found = [s for s in snapshots.glob("*") if (s / "config.json").is_file()] if snapshots.is_dir() else []
        if found:
            return max(found, key=lambda s: s.stat().st_mtime)
    return None


def pull(repo_id: str, *, cache_dir: Any = None, revision: str | None = None) -> Path:
    """Download (or finish downloading) a repo into the cache; returns its snapshot directory."""

    from huggingface_hub import snapshot_download

    print(f"[tensorfold] downloading {repo_id} from Hugging Face", flush=True)
    options = {} if revision is None else {"revision": revision}
    return Path(snapshot_download(repo_id, cache_dir=cache_dir, **options))


def metadata(repo_id: str, *, cache_dir: Any = None, latest: bool = False) -> Path:
    """Complete ancillary metadata before preflight, without downloading weights.

    A cached snapshot pins the revision and (for moved repositories) its actual
    cache owner. Offline Hub mode uses only locally available metadata; it cannot
    discover an optional file that was never downloaded.
    """
    from huggingface_hub import HfApi, constants, snapshot_download

    from tensorfold.checkpoint_metadata import read_json
    from tensorfold.cuda.tensor_file import checkpoint_path

    found = cached(repo_id, cache_dir=cache_dir)
    if constants.HF_HUB_OFFLINE:
        if found is None:
            raise FileNotFoundError(f"{repo_id} has no cached model metadata while Hugging Face is offline")
        read_json(found, "config.json", required=True)
        return found
    options = {"cache_dir": cache_dir, "allow_patterns": list(METADATA_FILES)}
    revision = None
    if found is not None and not latest:
        revision = found.name
        store = found.parent.parent.name
        repo_id = next((name for name in cache_names(repo_id) if store == f"models--{name.replace('/', '--')}"), repo_id)
    # snapshot_download may silently return an old/partial cache after a remote
    # error. Obtain the exact online revision and inventory first, without that
    # recovery policy, so a never-downloaded sidecar cannot mean "absent".
    info = HfApi().model_info(repo_id, revision=revision)
    if not isinstance(info.sha, str) or re.fullmatch(r"[0-9a-f]{40}", info.sha) is None:
        raise ValueError("the Hub metadata response must identify an exact model commit")
    if revision is not None and info.sha != revision:
        raise ValueError("the Hub metadata response differs from the selected cached revision")
    if not isinstance(info.siblings, list):
        raise ValueError("the Hub metadata response must include the model file inventory")  # noqa: TRY004 - remote schema
    filenames = []
    for sibling in info.siblings:
        filename = sibling.rfilename
        if not isinstance(filename, str):
            raise ValueError("the Hub file inventory must contain relative file names")  # noqa: TRY004 - remote schema
        if any(fnmatchcase(filename, pattern) for pattern in METADATA_FILES):
            # Validate names before the downloader can create any local files.
            relative = Path(filename)
            if (relative.is_absolute() or ".." in relative.parts or not filename or "\x00" in filename
                    or filename != relative.as_posix()):
                raise ValueError("the Hub metadata inventory contains an invalid relative file name")
            filenames.append(filename)
    if "config.json" not in filenames:
        raise FileNotFoundError(f"{repo_id}: the selected Hub model revision has no config.json")
    options["revision"] = info.sha
    directory = Path(snapshot_download(repo_id, **options))
    if directory.name != info.sha:
        raise ValueError("the Hub downloaded metadata for a different model revision")
    for filename in filenames:
        # The checkpoint/cache is immutable over validation and loading. Stat
        # follows only this authorized file's managed checkpoint/blob path.
        checkpoint_path(directory, filename).stat()
    read_json(directory, "config.json", required=True)
    return directory


def _cached_weights_complete(snapshot: Path, *, required_files: tuple[str, ...] = ()) -> bool:
    """Require complete weights before serving because Hugging Face also returns partial local snapshots."""

    if not all((snapshot / name).is_file() for name in required_files):
        return False

    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text())["weight_map"]
        except (OSError, ValueError, KeyError, TypeError):
            return False
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        try:
            files = set(weight_map.values())
        except TypeError:
            return False
        return all(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts
                   and (snapshot / name).is_file() for name in files)

    shards = list(snapshot.glob("model-*-of-*.safetensors"))
    if shards:
        matches = [re.fullmatch(r"model-(\d+)-of-(\d+)\.safetensors", path.name) for path in shards]
        if not all(matches):
            return False
        totals = {int(match.group(2)) for match in matches}
        if len(totals) != 1:
            return False
        total = totals.pop()
        return len(shards) == total and {int(match.group(1)) for match in matches} == set(range(1, total + 1))

    return (snapshot / "model.safetensors").is_file()


def resolve(name: str, *, download: bool = True, cache_dir: Any = None,
            required_files: tuple[str, ...] = (), revision: str | None = None) -> Path:
    """A model directory for ``name``: the directory itself, or a repo id's snapshot (downloaded if needed)."""

    path = Path(str(name)).expanduser()
    if path.suffix.lower() == ".gguf" or (path.is_dir() and not (path / "config.json").exists()
                                           and any(path.glob("*.gguf"))):
        from tensorfold.families.qwen4_exp.gguf_assets import prepare

        return prepare(path)
    if path.is_dir():
        return path
    if not is_repo_id(str(name)):
        raise FileNotFoundError(f"{name} is neither a directory nor a Hugging Face repo id (owner/name)")
    options = {} if revision is None else {"revision": revision}
    found = cached(str(name), cache_dir=cache_dir, **options)
    if found is not None and (found / "config.json").is_file() and (
        not download or _cached_weights_complete(found, required_files=required_files)
    ):
        return found
    if not download:
        raise FileNotFoundError(f"{name} is not in the Hugging Face cache; run: tensorfold pull {name}")
    downloaded = pull(str(name), cache_dir=cache_dir, **options)
    if required_files and not _cached_weights_complete(downloaded, required_files=required_files):
        raise FileNotFoundError(f"{name} is missing required files: {', '.join(required_files)}")
    return downloaded


def size_of(directory: Path) -> int:
    """Bytes of the files under ``directory`` (following the cache's symlinks)."""

    return sum(p.stat().st_size for p in Path(directory).rglob("*") if p.is_file())
