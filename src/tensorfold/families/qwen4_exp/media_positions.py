"""Compact, immutable Qwen4Exp media geometry with bounded CPU coordinates.

Canonical expanded IDs remain the PLE/hash input. A visual run describes one
complete image or temporal group, not a projected embedding archive. Coordinate
generation reuses the pinned model's exact visual-position implementation; the
caller issues actual range/role/profile proofs before any device narrowing.
"""

from bisect import bisect_right
from dataclasses import dataclass

from .position_profile import MAX_POSITION


@dataclass(frozen=True)
class PositionRun:
    start: int
    end: int
    base: int
    kind: str
    grid: tuple[int, int, int] | None
    item: int | None
    temporal_group: int | None


@dataclass(frozen=True)
class MediaPositionPlan:
    token_ids: tuple[int, ...]
    runs: tuple[PositionRun, ...]
    starts: tuple[int, ...]
    merge: int
    rope_delta: int
    max_visual_rows: int

    @property
    def visual_runs(self):
        return tuple(run for run in self.runs if run.kind != "text")

    def coordinates(self, start, count, *, max_rows):
        """Materialize only an admitted chunk and one bounded visual-unit helper.

        Callers serialize access to their immutable plan. The full visual helper
        may overlap the returned chunk and belongs in the CPU working envelope.
        No result is retained by the plan. Sequence rows and THW bounds are
        checked independently by the later coordinate issuer.
        """
        if (type(start) is not int or start < 0 or type(count) is not int or count <= 0
                or type(max_rows) is not int or max_rows <= 0 or count > max_rows
                or start > len(self.token_ids) - count):
            raise ValueError("media coordinates exceed the admitted chunk or canonical input")
        import torch
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpModel

        # Explicit factory device also covers an enclosing meta/CUDA default.
        result = torch.empty((count, 3), dtype=torch.int64, device="cpu")
        index, end = bisect_right(self.starts, start) - 1, start + count
        while index < len(self.runs):
            run = self.runs[index]
            lo, hi = max(start, run.start), min(end, run.end)
            if lo >= hi:
                break
            if run.kind == "text":
                values = torch.arange(run.base + lo - run.start, run.base + hi - run.start,
                                      dtype=torch.int64, device="cpu")
                result[lo - start:hi - start].copy_(values[:, None].expand(-1, 3))
            else:
                grid = torch.tensor(run.grid, dtype=torch.int64, device="cpu")
                values = Qwen4ExpModel.get_vision_position_ids(
                    None, run.base, grid, spatial_merge_size=self.merge, device="cpu")
                result[lo - start:hi - start].copy_(values[:, lo - run.start:hi - run.start].t())
                del grid
            # Integer CPU copy_ is complete; keep no previous-run scratch alive
            # while the next helper/factory materializes its bounded result.
            del values
            if hi == end:
                break
            index += 1
        return result


def _grids(values, merge, max_visual_rows, max_segments, *, video):
    if type(values) not in (tuple, list) or len(values) > max_segments:
        raise ValueError("media grids exceed their admitted item count")
    result, segments = [], 0
    for raw in values:
        if (type(raw) not in (tuple, list) or len(raw) != 3
                or any(type(v) is not int or not 0 < v < MAX_POSITION for v in raw)):
            raise ValueError("media grids require positive integer THW values")
        t, h, w = raw
        if (not video and t != 1 or h % merge or w % merge
                or h // merge * (w // merge) > max_visual_rows):
            raise ValueError("media grids exceed complete merge-aligned visual units")
        segments += t
        if segments > max_segments:
            raise ValueError("media grids exceed their admitted temporal segment count")
        result.append((t, h, w))
    return tuple(result)


def plan_media_positions(token_ids, *, image_grids, video_grids, image_token_id, video_token_id,
                         vision_start_token_id, vision_end_token_id, vocab_size, merge,
                         max_rows, max_segments, max_visual_rows):
    """Validate expanded placeholders before allocating any coordinate tensor.

    Images have one temporal group; video grids describe their complete clip
    and are consumed one timestamp-separated temporal group at a time. Source
    ordering and timestamps are already fixed by the admitted processor plan.
    """
    if (any(type(v) is not int or v <= 0 for v in (vocab_size, merge, max_rows, max_segments, max_visual_rows))
            or vocab_size > MAX_POSITION or max_rows >= MAX_POSITION or max_visual_rows > max_rows
            or type(token_ids) not in (tuple, list) or not 0 < len(token_ids) <= max_rows
            or any(type(v) is not int or not 0 <= v < vocab_size for v in token_ids)):
        raise ValueError("expanded media input exceeds its declared integer row/vocabulary bounds")
    special = image_token_id, video_token_id, vision_start_token_id, vision_end_token_id
    if any(type(v) is not int or not 0 <= v < vocab_size for v in special) or len(set(special)) != 4:
        raise ValueError("media requires four distinct declared vocabulary token IDs")
    images = _grids(image_grids, merge, max_visual_rows, max_segments, video=False)
    videos = _grids(video_grids, merge, max_visual_rows, max_segments, video=True)
    if len(images) + sum(grid[0] for grid in videos) > max_segments:
        raise ValueError("combined media exceeds its admitted temporal segment count")
    tokens, runs = tuple(token_ids), []
    cursor = next_position = image_index = video_index = temporal = 0
    for begin, kind in enumerate(tokens):
        if begin < cursor or kind not in (image_token_id, video_token_id):
            continue
        if kind == image_token_id:
            if image_index >= len(images):
                raise ValueError("image placeholders have no matching admitted grid")
            grid, item, group, modality = images[image_index], image_index, 0, "image"
            image_index += 1
        else:
            if video_index >= len(videos):
                raise ValueError("video placeholders have no matching admitted grid")
            t, h, w = videos[video_index]
            grid, item, group, modality = (1, h, w), video_index, temporal, "video"
            temporal += 1
            if temporal == t:
                video_index, temporal = video_index + 1, 0
        _, h, w = grid
        end = begin + h // merge * (w // merge)
        if (begin == 0 or end >= len(tokens) or tokens[begin - 1] != vision_start_token_id
                or tokens[end] != vision_end_token_id or any(value != kind for value in tokens[begin:end])):
            raise ValueError("expanded placeholders must match the complete delimited visual grid")
        if begin > cursor:
            runs.append(PositionRun(cursor, begin, next_position, "text", None, None, None))
        base = next_position + begin - cursor
        next_position = base + max(h, w) // merge
        runs.append(PositionRun(begin, end, base, modality, grid, item, group))
        cursor = end
    if image_index != len(images) or video_index != len(videos) or temporal:
        raise ValueError("admitted media grids have no complete matching expanded placeholders")
    if cursor < len(tokens):
        runs.append(PositionRun(cursor, len(tokens), next_position, "text", None, None, None))
    # For positive H/W, max(H,W) <= H*W. Every temporal group has T=1,
    # so next_position <= cursor by induction and every generated coordinate
    # is below len(tokens) < MAX_POSITION. Role-specific THW bounds are still
    # independently checked by the coordinate issuer, before device narrowing.
    return MediaPositionPlan(tokens, tuple(runs), tuple(run.start for run in runs), merge,
                             next_position - cursor, max_visual_rows)
