"""Flash Next proposal geometry shared by admission and execution.

The pending anchor is already emitted, but has not been consumed by the target.
Each verification row consumes one input and can emit one further token. A reply
with K tokens left therefore permits K rows and at most K - 1 proposals. Padding
or tentative cache capacity never extends the logical context commitment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _count(value: int, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class MethodGeometry:
    """Allocated capability; a zero-proposal batch does not remove its resources.

    ``draft_tail_rows`` is additional stock-MTP conditioning occupancy beyond
    committed target rows. Its separate cache and weights remain resident at
    proposal length zero. DSpark and DFlash have their own actual cache geometry
    and must not be budgeted as a thirteenth target attention cache.
    """

    method: str
    maximum_proposals: int

    def __post_init__(self) -> None:
        _count(self.maximum_proposals, "maximum proposals")
        if self.method not in ("none", "mtp", "dspark", "dflash2"):
            raise ValueError("Flash Next execution supports none, stock MTP, DSpark or DFlash2")
        if self.method == "none" and self.maximum_proposals:
            raise ValueError("the none method cannot allocate proposals")
        if self.method == "dflash2" and self.maximum_proposals > 3:
            raise ValueError("the initial four-row DFlash2 block permits at most three proposals")

    @property
    def target_tail_rows(self) -> int:
        return self.maximum_proposals + 1

    @property
    def draft_tail_rows(self) -> int | None:
        # The absorb consumes the pending anchor, then at most P-1 chained
        # draft inputs. At P=0, committed progress still requires maintenance.
        return self.maximum_proposals if self.method == "mtp" else 0 if self.method == "none" else None

    @property
    def stock_mtp_cache_layers(self) -> int:
        return int(self.method == "mtp")

    def target_rows(self, proposals: int) -> int:
        _count(proposals, "proposals")
        if proposals > self.maximum_proposals:
            raise ValueError("proposal length exceeds the allocated method capability")
        return proposals + 1

    def cap_proposals(self, requested: int, *, reply_tokens: int,
                      context_tokens: int, target_rows: int) -> int:
        """Rows allowed after the pending anchor by reply, context and kernels.

        All three limits count *further outputs*, including the correction or
        bonus output. The caller separately handles the no-output case.
        """

        limits = (_count(requested, "requested proposals"), self.maximum_proposals,
                  max(0, _count(reply_tokens, "remaining reply tokens") - 1),
                  max(0, _count(context_tokens, "remaining context tokens") - 1),
                  max(0, _count(target_rows, "available target rows") - 1))
        return min(limits)

    def receipt(self) -> dict:
        receipt = {"method": self.method, "maximum_proposals": self.maximum_proposals,
                "maximum_target_rows": self.target_rows(self.maximum_proposals),
                "target_tail_rows": self.target_tail_rows, "draft_tail_rows": self.draft_tail_rows,
                "stock_mtp_cache_layers": self.stock_mtp_cache_layers,
                "zero_proposal_maintains_conditioning": self.method != "none",
                "sampling": "position-keyed target sample matching",
                "commit": "accepted conditional prefix including pending anchor"}
        if self.method == "dflash2":
            receipt.update(draft_compute_block_rows=4,
                           sampling="standard rejection with actual conditional selector q",
                           draft_cache="bounded-fp16-swa-separate-from-target-kv")
        return receipt


def checkpoint_method(directory: str | Path) -> str:
    """Identify a supplied architecture before target weights or CUDA allocation.

    A related-model DFlash artifact is not a stock MTP head. Flash Next has no
    qualified DFlash pairing or SM70 draft-mask implementation yet; preserving
    that distinction prevents accidental fallback into an incompatible loader.
    """

    from tensorfold.cuda.tensor_file import read_metadata_json

    path = Path(directory) / "config.json"
    if not path.is_file():
        return "mtp"  # the stock loader owns its existing missing-file error
    config = read_metadata_json(path)
    architectures = config.get("architectures")
    if architectures is None:
        architectures = []  # optional in existing stock-MTP configuration files
    if not isinstance(architectures, list) or any(not isinstance(item, str) for item in architectures):
        raise ValueError("drafter architectures must be a list of names")
    if architectures == ["DSparkDraftModel"]:
        return "dspark"  # DSpark's own schema also names its options dflash_config.
    if any("dflash" in name.lower() for name in architectures) or "dflash_config" in config:
        raise ValueError("Flash Next DFlash needs an exact compatible target/draft manifest and an SM70 "
                         "draft-attention provider; a DFlash checkpoint cannot be loaded as stock MTP or DSpark")
    return "mtp"


def collective_prefix_limit(callback, comm):
    """One serial TP request's leader-owned stop decision on every peer.

    The request packet establishes callback presence before any rank chooses
    this path. Error status participates in the same collective, so a tokenizer
    failure cannot leave followers waiting for a missing decision. Reuse this
    bounded frame for the request; no collective is added when no hook exists.
    """

    if callback is None or comm is None:
        return callback
    import torch
    from tensorfold.cuda.streams import prefix_limit

    send = torch.zeros(3, dtype=torch.int64, device=comm.device)
    receive = torch.empty((comm.world, 3), dtype=torch.int64, device=comm.device)

    def limit(new, prefix):
        failure = None
        if comm.rank == 0:
            try:
                keep, stop = prefix_limit(callback, new, prefix)
                send[0], send[1], send[2] = keep, int(stop), 0
            except BaseException as error:
                failure = error
                send.zero_()
                send[2] = 1
        comm.all_gather(send, receive)
        keep, stop, failed = receive[0].tolist()
        if failed:
            if failure is not None:
                raise failure
            raise RuntimeError("rank zero's token-prefix decision failed before state commit")
        return prefix_limit(lambda *_args: (int(keep), bool(stop)), new, prefix)

    return limit
