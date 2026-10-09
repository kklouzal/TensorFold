"""Explicit authorization for mlx-lm's checkpoint-supplied Python module.

An ordinary family recipe selects installed model code. ``model_file`` instead
executes a Python file with the server's privileges; it therefore requires an
explicit caller opt-in. The checkpoint must remain immutable while borrowed.
This check precedes importing mlx-lm and reading any tensor payload.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
from typing import Any

from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, checkpoint_path, read_metadata_json


def _identity(value: os.stat_result) -> tuple[int, ...]:
    return tuple(
        getattr(value, name) for name in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    )


@dataclass(frozen=True)
class ModelCode:
    """Startup-owned authorization, never accepted from checkpoint metadata."""

    model_dir: Path
    config: dict[str, Any]
    states: tuple[tuple[str, str, tuple[int, ...], str], ...]
    custom: bool

    def validate(self) -> None:
        current = authorize_model_code(self.model_dir, trust_model_code=self.custom)
        if current.config != self.config or current.states != self.states:
            raise RuntimeError("model configuration/code changed while the loader borrowed it")


def _state(root: Path, name: str) -> tuple[str, str, tuple[int, ...], str]:
    path = checkpoint_path(root, name)
    descriptor = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("model configuration/code must resolve to a regular file")
        if name == "config.json" and not 0 < before.st_size <= MAX_HEADER_BYTES:
            raise ValueError("model configuration size exceeds the checkpoint metadata protocol limit")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, 1 << 20):
            digest.update(chunk)
        after = os.fstat(descriptor)
    except BaseException as primary:
        try:
            os.close(descriptor)
        except BaseException as secondary:
            if secondary is primary:
                raise primary
            try:
                BaseException.add_note(primary, "model-code descriptor close failed")
            except BaseException as annotation:
                if annotation is primary:
                    raise primary from secondary
                raise primary from BaseExceptionGroup(
                    "model-code close and annotation failures", [secondary, annotation]
                )
            raise primary from secondary
        raise
    else:
        os.close(descriptor)
    if _identity(before) != _identity(after) or _identity(path.stat()) != _identity(after):
        raise RuntimeError("model configuration/code changed during authorization")
    return name, str(path), _identity(after), digest.hexdigest()


def authorize_model_code(model_dir: str | Path, *, trust_model_code: bool = False) -> ModelCode:
    """Reject implicit executable input; admit explicitly trusted in-root code.

    Absolute/parent paths and foreign symlink targets are rejected even when
    trusted. Standard HF snapshots may use their own blob store. An opt-in is
    code authorization, not a sandbox: trusted Python can access the process.
    """
    if type(trust_model_code) is not bool:
        raise ValueError("trust_model_code must be an explicit boolean")
    root = Path(model_dir).resolve(strict=True)
    config_state = _state(root, "config.json")
    config = read_metadata_json(checkpoint_path(root, "config.json"))
    states = [config_state]
    name = config.get("model_file")
    if name is not None:
        if not isinstance(name, str) or not name or "\0" in name:
            raise ValueError("model_file must name a nonempty relative Python file")
        if not trust_model_code:
            raise ValueError(
                "this checkpoint's model_file executes Python code; use a maintained TensorFold "
                "family recipe or explicitly authorize it with --trust-model-code"
            )
        states.append(_state(root, name))
    if _state(root, "config.json") != config_state:
        raise RuntimeError("model configuration changed during authorization")
    # Tokenizer auto_map code shares the explicit trust authorization. Its
    # arbitrary imports also invalidate a cross-restart disk-cache claim.
    return ModelCode(root, config, tuple(states), trust_model_code)


def load_mlx_model(model_dir: str | Path, *, trust_model_code: bool = False, **options: Any) -> Any:
    """Pin the provider's supported model_file override to the authorized value.

    The provider may reread configuration, but its supported ``model_config``
    merge keeps that reread from introducing an unauthorized executable file.
    Trusted custom code still executes from the selected trusted tree: this is
    not a sandbox or proof of immutable arbitrary imports. Postflight detects
    configuration/code mutation, rather than undoing executed trusted code.
    """
    proof = authorize_model_code(model_dir, trust_model_code=trust_model_code)
    if "model_config" in options:
        raise ValueError("model_config overrides require a separate model-code authorization")
    if "tokenizer_config" in options:
        raise ValueError("tokenizer_config overrides require a separate model-code authorization")
    from mlx_lm import load

    try:
        loaded = load(
            str(proof.model_dir),
            model_config={"model_file": proof.config.get("model_file")},
            tokenizer_config={"trust_remote_code": trust_model_code},
            **options,
        )
    except BaseException as primary:
        try:
            proof.validate()
        except BaseException as secondary:
            if secondary is primary:
                raise primary
            try:
                BaseException.add_note(primary, "model configuration/code validation failed while draining loader")
            except BaseException as annotation:
                if annotation is primary:
                    raise primary from secondary
                raise primary from BaseExceptionGroup(
                    "model-code validation and annotation failures", [secondary, annotation]
                )
            raise primary from secondary
        raise
    proof.validate()
    return loaded
