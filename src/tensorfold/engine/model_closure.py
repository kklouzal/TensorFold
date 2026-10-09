"""Owned loader trees and exact-file inputs for startup content identities.

The owner selects checkpoint/drafter roots and any externally selected sidecar
before loading. No configuration or persisted receipt selects executable code.
File membership, aliases and targets are verified again after loading. Generated
new files may be added to a successor receipt, but existing inputs cannot change.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
import os
from pathlib import Path
import stat

from tensorfold.cuda.tensor_file import checkpoint_path


def _identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


@dataclass(frozen=True)
class FileSelection:
    logical: str
    alias: Path
    resolved: Path
    alias_identity: tuple[int, ...]
    target_identity: tuple[int, ...]


@dataclass(frozen=True)
class TreeSelection:
    logical: str
    alias: Path
    resolved: Path


class LoaderClosure:
    """Startup-owned bounded, authorized logical-file closure; no hash cache."""

    def __init__(
        self,
        roots: dict[str, Path],
        *,
        extra_files: dict[str, Path] | None = None,
        max_files: int = 65536,
        max_name_bytes: int = 4096,
        exclude_directories: frozenset[str] = frozenset(),
    ):
        if (
            type(roots) is not dict
            or not roots
            or type(max_files) is not int
            or max_files <= 0
            or type(max_name_bytes) is not int
            or max_name_bytes <= 0
            or type(exclude_directories) is not frozenset
            or any(type(name) is not str for name in exclude_directories)
            or len(roots) > max_files
            or extra_files is not None
            and (type(extra_files) is not dict or len(extra_files) > max_files)
        ):
            raise ValueError("explicit bounded loader input roots required")
        self.max_files, self.max_name_bytes = max_files, max_name_bytes
        self.exclude_directories = exclude_directories
        for name in roots:
            self._name(name)
        self.roots = tuple(
            TreeSelection(name, Path(path).absolute(), Path(path).resolve(strict=True)) for name, path in roots.items()
        )
        for tree in self.roots:
            self._name(tree.logical)
            if not tree.resolved.is_dir():
                raise ValueError("loader input tree must be an existing directory")
        self.extra_files = {} if extra_files is None else dict(extra_files)
        self.selections = self._select()
        self._by_logical = {item.logical: item for item in self.selections}

    def _name(self, name):
        if type(name) is not str or not name or len(name.encode("utf-8")) > self.max_name_bytes:
            raise ValueError("loader logical name exceeds its explicit UTF-8 budget")

    def _file(self, logical, alias, resolved):
        self._name(logical)
        info = resolved.stat()
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("loader input must resolve to a regular file")
        return FileSelection(logical, alias, resolved, _identity(alias.lstat()), _identity(info))

    def _select(self):
        selected = {}
        visited = 0
        for tree in self.roots:
            if tree.alias.resolve(strict=True) != tree.resolved:
                raise ValueError("loader input root alias changed")
            pending = [(tree.alias, ())]
            while pending:
                directory, ancestors = pending.pop()
                relative = directory.relative_to(tree.alias)
                resolved = tree.resolved if not relative.parts else checkpoint_path(tree.resolved, str(relative))
                if resolved in ancestors:
                    raise ValueError("loader tree contains a directory alias cycle")
                ancestors = (*ancestors, resolved)
                with os.scandir(directory) as entries:
                    for entry in entries:
                        if entry.name in self.exclude_directories and entry.is_dir(follow_symlinks=False):
                            continue
                        visited += 1
                        if visited > self.max_files:
                            raise ValueError("loader closure exceeds its declared entry-count budget")
                        alias = Path(entry.path).absolute()
                        relative = alias.relative_to(tree.alias)
                        self._name(tree.logical + "/" + relative.as_posix())
                        target = checkpoint_path(tree.resolved, str(relative))
                        info = target.stat()
                        if stat.S_ISDIR(info.st_mode):
                            pending.append((alias, ancestors))
                        else:
                            if len(selected) >= self.max_files:
                                raise ValueError("loader closure exceeds its declared file-count budget")
                            logical = tree.logical + "/" + relative.as_posix()
                            if logical in selected:
                                raise ValueError("duplicate loader logical file")
                            selected[logical] = self._file(logical, alias, target)
        for name, path in self.extra_files.items():
            self._name(name)
            logical = "sidecar/" + name
            alias = Path(path).absolute()
            if logical in selected or len(selected) >= self.max_files:
                raise ValueError("duplicate or excessive loader sidecar input")
            # Exact externally selected sidecars are authorized by the owner,
            # rather than model/index-controlled path strings.
            selected[logical] = self._file(logical, alias, alias.resolve(strict=True))
        return tuple(selected[name] for name in sorted(selected))

    @property
    def files(self):
        return {item.logical: item.alias for item in self.selections}

    def authorize(self, logical, alias):
        item = self._by_logical.get(logical)
        if item is None:
            raise ValueError("unselected loader input")
        if Path(alias).absolute() != item.alias or item.alias.resolve(strict=True) != item.resolved:
            raise ValueError("loader input authority differs from its selected alias")
        return item.resolved

    def verify_unchanged(self, *, allow_additions=False):
        """Detect membership/alias/target changes; caller still owns immutability."""
        if type(allow_additions) is not bool:
            raise ValueError("loader additions policy must be an exact boolean")
        current = {item.logical: item for item in self._select()}
        if (
            not allow_additions
            and len(current) != len(self.selections)
            or any(current.get(item.logical) != item for item in self.selections)
        ):
            raise ValueError("model execution input closure changed during loading")
        return current


def runtime_closure(project_root: Path, *, vision: bool = False):
    """Capture maintained source/provider trees, separately from model data.

    Source and packaged native libraries bind math/tokenizer implementation.
    Existing bytecode is an execution input too: timestamp validation does not
    prove that its code agrees with source. Newly generated cache files may
    enter a successor receipt after loading, with their full hashing cost.
    Optional packages absent from this environment are not invented inputs;
    the loader retains its ordinary supported-dependency checks.
    """
    packages = {
        "mlx": "mlx",
        "mlx-lm": "mlx_lm",
        "numpy": "numpy",
        "transformers": "transformers",
        "tokenizers": "tokenizers",
        "sentencepiece": "sentencepiece",
        "safetensors": "safetensors",
        "huggingface-hub": "huggingface_hub",
    }
    if vision:
        packages.update({"mlx-vlm": "mlx_vlm", "Pillow": "PIL"})
    roots = {"tensorfold-source": Path(project_root)}
    for distribution, package in packages.items():
        try:
            installed = metadata.distribution(distribution)
        except metadata.PackageNotFoundError:
            continue
        for member in (package, package + ".libs"):
            path = Path(installed.locate_file(member))
            if path.is_dir():
                roots["provider-" + member] = path
    return LoaderClosure(roots)
