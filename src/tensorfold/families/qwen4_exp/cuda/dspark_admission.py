"""Bounded metadata and shared-slot admission for an explicit DSpark drafter."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace

from .dspark_config import DSparkConfig
from .dspark_weights import LOAD_CHUNK, SOURCE_BYTES, plan_weights


def is_dspark(directory):
    from tensorfold.cuda.tensor_file import read_metadata_json

    file = Path(directory) / "config.json"
    return file.is_file() and read_metadata_json(file).get("architectures") == ["DSparkDraftModel"]


def _stat(path):
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


@dataclass(frozen=True)
class DSparkPlan:
    directory: Path
    config: DSparkConfig
    source_identity: str
    source_stat: tuple
    rank: int
    world: int
    resident_bytes: int
    loading_host_bytes: int
    chunk_bytes: int

    @property
    def identity(self):
        from .dspark_cuda import EXECUTION_ID

        return hashlib.sha256(
            (
                self.config.identity
                + "|"
                + self.source_identity
                + "|"
                + EXECUTION_ID
                + "|pending-safe-hcmean-f16-prefix-marker-v1"
            ).encode()
        ).hexdigest()

    @property
    def cache_row_bytes(self):
        local = self.config.partition(self.rank, self.world)
        return self.config.layers * local["kv_heads"] * self.config.head_dim * 4

    def capacity_bytes(self, total_slots, *, main_rows, prompt_rows, slots):
        from .dspark_cuda import byte_plan
        from .dspark_taps import byte_plan as tap_plan

        if type(total_slots) is not int or total_slots < 0:
            raise ValueError("DSpark admission needs nonnegative shared cache slots")
        if type(slots) is not int or slots < 1:
            raise ValueError("DSpark admission needs a positive sequence count")
        for value in (main_rows, prompt_rows):
            if type(value) is not int or not 1 <= value <= 2048:
                raise ValueError("DSpark tap buffer rows must lie in1..2048")
        longest = max(1, min(self.config.context, total_slots))
        largest = max(main_rows, prompt_rows)
        eager = byte_plan(self.config, tp=(self.rank, self.world), capacity=longest, slots=1, prefill_rows=largest)[
            "eager_transient_bound"
        ]
        main, prompt = tap_plan(self.config, main_rows), tap_plan(self.config, prompt_rows)
        # Only one GGUF prompt/verify frame is submitted at once. Two staged
        # contexts retain a conservative failure/retirement overlap allowance.
        staged = 2 * largest * self.cache_row_bytes
        taps = main.device_bytes + prompt.device_bytes + max(main.transient_bytes, prompt.transient_bytes)
        return total_slots * self.cache_row_bytes + taps + staged + eager

    def check_source(self):
        if _stat(self.directory / "model.safetensors") != self.source_stat:
            raise ValueError("DSpark checkpoint changed after admission")


def inspect(directory, target_text, *, tp=(0, 1), chunk_bytes=LOAD_CHUNK):
    """Strict local plan and actual source digest; no CUDA/weight allocation."""
    from tensorfold.cuda.tensor_file import _regular_stream, read_header

    directory = Path(directory).resolve()
    c = DSparkConfig.read(directory)
    if not isinstance(target_text, dict):
        raise ValueError("DSpark admission requires original target metadata")
    c.check_target(
        SimpleNamespace(
            hidden=target_text.get("hidden_size"),
            vocab=target_text.get("vocab_size"),
            layers=target_text.get("num_hidden_layers"),
            streams=target_text.get("hc_count"),
        )
    )
    rank, world = tp
    c.partition(rank, world)
    file = directory / "model.safetensors"
    before = _stat(file)
    _, header = read_header(file, SOURCE_BYTES)
    plans = plan_weights(c, header, tp=tp)
    minimum = max(math.prod(p.source_shape[1:]) * SOURCE_BYTES[header[p.name]["dtype"]] for p in plans)
    if type(chunk_bytes) is not int or not minimum <= chunk_bytes <= 1 << 30:
        raise ValueError("DSpark loader chunk must fit a complete original source row")
    digest = hashlib.sha256()
    with _regular_stream(file) as source:
        while data := source.read(1 << 20):
            digest.update(data)
    if _stat(file) != before:
        raise ValueError("DSpark checkpoint changed while hashing its source")
    return DSparkPlan(
        directory,
        c,
        digest.hexdigest(),
        before,
        rank,
        world,
        sum(p.nbytes for p in plans),
        3 * chunk_bytes,
        chunk_bytes,
    )
