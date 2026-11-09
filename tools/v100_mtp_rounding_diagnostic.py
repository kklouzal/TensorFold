"""Four-cell tiny MTP rounding investigation; NOT qualification or performance.

Requires installed SM70 software and the canonical tiny test fixtures. Original
None-context oracle/seed/draw order remain controls. No tolerance or quality
acceptance is defined. Import/help perform no Torch/CUDA or evidence operation.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import hashlib
import math
from pathlib import Path
import re

if __package__:
    from .benchmark_output import write_json
    from .worker_lifetime import raise_failures
else:
    from benchmark_output import write_json
    from worker_lifetime import raise_failures

# One finite diagnostic owns everything until positive retirement; never rely
# on a failed pytest traceback to retain native graph/copy borrowers.
_ACTIVE = [None]
COUNTS = ((1, 3, 2, 1, 2, 3), (1, 1, 1, 1, 1, 1), (3, 1, 2, 3, 2, 1), (1, 1, 1, 1, 1, 1))
LABELS = ("embedding", "fc_e", "hc_norm", "fc_h", "attention_projection", "q", "iq", "attention_output", "streams", "pss", "mixed", "logits")


class Unsupported(RuntimeError):
    """The requested cell/probe was not observed; never a synthetic success."""


def require(value, message):
    if not value:
        raise ValueError(message)


def row_plan(slots, counts, *, width):
    require(len(slots) == len(counts) == 6 and all(type(s) is int for s in slots) and sorted(slots) == list(range(6)), "six distinct actual decoder slots required")
    require(type(width) is int and width in (1, 3) and all(type(n) is int and 1 <= n <= width for n in counts), "invalid live row counts")
    canonical = sorted(zip(slots, counts))
    inverse, segment_inverse, logical = [], [], []
    for slot, count in zip(slots, counts):
        lane = next(i for i, (candidate, _) in enumerate(canonical) if candidate == slot)
        inverse.extend(range(lane * width, lane * width + count))
        segment_inverse.append(lane)
        logical.extend((slot, row) for row in range(count))
    return {"slots": list(slots), "counts": list(counts), "width": width,
            "inverse": inverse, "segment_inverse": segment_inverse, "logical_rows": logical,
            "canonical_slots": [slot for slot, _ in canonical],
            "last": [lane * width + count - 1 for lane, (_, count) in enumerate(canonical)]}


def content(value):
    if isinstance(value, dict):
        if {"shape", "dtype", "sha256"} <= set(value):
            return value["shape"], value["dtype"], value["sha256"]
        return {name: content(row) for name, row in value.items()}
    if isinstance(value, list):
        return [content(row) for row in value]
    return value


def first_difference(left, right):
    """Exact content/shape/dtype identities; this is not a tolerance gate."""
    require(set(left) == set(right), "cell observation keys differ")
    order = (*LABELS[:7], "cache", *LABELS[7:])
    for name in order:
        if content(left[name]) != content(right[name]):
            return name
    return None


def classify(cells):
    for left, right, axis in (("A", "A_repeat", "same_eager_not_reproduced"), ("A", "B", "context_extent"),
                              ("B", "C", "canonical_padding_order"), ("C", "D", "graph_state_or_capture")):
        stage = first_difference(cells[left], cells[right])
        if stage is not None:
            return {"axis": axis, "first_observed_retained_stage": stage,
                    "scope": "first retained observation, not proof of first mathematical operation/cause"}
    return {"axis": "no_difference_observed", "scope": "this finite frame only; original failures not waived"}


def classify_strict_frame(cells, control):
    result = classify(cells)
    if result["axis"] == "no_difference_observed" and control.get("strict_failure_observed"):
        stage = control["observable"]
        stage = stage if stage in LABELS else "cache"
        return {"axis": "strict_observable_failure_without_raw_difference", "first_observed_retained_stage": stage,
                "scope": "actual strict comparator failed despite identical raw observations; not parity"}
    return result


def validate_report(report, *, case, image_id, source_commit):
    require(report.get("schema") == "tensorfold.mtp-four-cell-diagnostic-v1" and report.get("case") == case
            and report.get("image_id_declared") == image_id and report.get("source_commit_declared") == source_commit,
            "diagnostic capture identity differs")
    require(report.get("qualification") is False and report.get("P3_valid") is False
            and report.get("completed") is True and report.get("failed") is False
            and report.get("retirement_verified") is True and report.get("error_types") == []
            and report.get("instrumented_strict_failure_observed") is True
            and report.get("original_anchor", {}).get("strict_failure_observed") is True
            and report.get("original_anchor", {}).get("tensor_storage_read") is False
            and report.get("original_input_match") == "UNPROVEN"
            and report.get("experiment_status") == "CONDITIONAL_MECHANISM_DIAGNOSTIC_ONLY", "diagnostic capture incomplete/unreproduced")
    anchor = report["original_anchor"]
    require(anchor.get("source_function") == "_qualify_mtp" and type(anchor.get("source_sha256")) is str
            and re.fullmatch(r"[0-9a-f]{64}", anchor["source_sha256"]) and type(anchor.get("line")) is int and anchor["line"] > 0
            and type(anchor.get("expression")) is str and anchor["expression"].startswith("torch.testing.assert_close("),
            "original independent comparator provenance missing")
    require(report.get("steps"), "no diagnostic frames")
    step = report["steps"][-1]
    require(set(step["cells"]) == {"A", "A_repeat", "B", "C", "D"} and step.get("discrete_inputs") is True,
            "four actual cells/discrete witness required")
    require(step.get("raw_pair_difference") == first_difference(step["cells"]["A"], step["cells"]["D"])
            and step.get("strict_control", {}).get("strict_failure_observed") is True
            and step.get("difference") == classify_strict_frame(step["cells"], step["strict_control"]) and "first_stage_capture" in step,
            "diagnostic derived status differs from actual content identities")
    if (step["difference"].get("axis") == "context_extent"
            and step["difference"].get("first_observed_retained_stage") == "attention_output"):
        validate_arithmetic_probe(step)
    origin = report.get("native_origin", {})
    require(origin.get("missing_required_prefixes") == [] and any(row.get("name", "").startswith("tf_sm70_family_")
            for row in origin.get("mapped_elfs", [])) and not report.get("native_origin_failure_type"),
            "post-use native origin observation incomplete; not a numerical cause waiver")
    return "CONDITIONAL_MECHANISM_DIAGNOSTIC_ONLY"


def validate_arithmetic_probe(step):
    probe = step.get("arithmetic_probe", {})
    require(probe.get("status") == "CONDITIONAL_MECHANISM_DIAGNOSTIC_ONLY"
            and (probe.get("logical_row"), probe.get("slot"), probe.get("end")) == (1, 4, 12)
            and probe.get("device_copy_obligation_bytes") == 20800
            and set(probe.get("cells", {})) == {"A", "B"}, "attention arithmetic probe incomplete/perturbed")
    for cell, keys in (("A", 12), ("B", 32)):
        data = probe["cells"][cell]
        require(data.get("keys") == keys and content(data["retained"]) == content(step["cells"][cell])
                and data["QSA"] == step["first_stage_capture"][cell]
                and set(data["operators"]) == set(AttentionArithmeticProbe.LAYOUT),
                "attention arithmetic probe changed its controlled witness")


@contextmanager
def attributes(changes):
    """Restore all policy/observer bindings independently; no permanent hooks."""
    old = [(owner, name, getattr(owner, name), name in vars(owner)) for owner, name, _ in changes]
    primary, errors = None, []
    try:
        for owner, name, value in changes:
            setattr(owner, name, value)
        yield
    except BaseException as error:
        primary = error
    for owner, name, value, owned in reversed(old):
        try:
            if owned:
                setattr(owner, name, value)
            else:
                delattr(owner, name)
        except BaseException as error:
            errors.append(error)
    raise_failures(primary, errors)


def original_anchor(helper, weights, cache, *, retain=None):
    """Call the unchanged helper FIRST. Inspect scalar/source metadata only.

    Its finally owns graph retirement. Never dereference actual/expected or any
    borrowed Tensor after it closes. No snapshots, synchronization or probes
    are admitted before this genuine original comparator has raised.
    """
    try:
        helper(weights, cache)
    except BaseException as error:
        if retain is not None:
            try:
                retain(error)  # Only AFTER helper finally, metadata/owners only.
            except BaseException as retention:
                raise_failures(error, [retention])
        if not isinstance(error, AssertionError):
            raise
        traceback = error.__traceback__
        while traceback is not None and traceback.tb_frame.f_code is not helper.__code__:
            traceback = traceback.tb_next
        if traceback is None:
            raise Unsupported("original failure has no helper provenance") from error
        path = Path(helper.__code__.co_filename)
        tree = ast.parse(path.read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and node.lineno <= traceback.tb_lineno <= getattr(node, "end_lineno", node.lineno)
                 and ast.unparse(node.func) == "torch.testing.assert_close"]
        if len(calls) != 1 or not all(any(k.arg == key and isinstance(k.value, ast.Constant) and k.value.value == 0
                                          for k in calls[0].keywords) for key in ("rtol", "atol")):
            raise Unsupported("original failure is not an unchanged strict numeric comparator") from error
        expression = ast.unparse(calls[0])
        values = traceback.tb_frame.f_locals
        iteration, counts = values.get("iteration"), values.get("counts")
        require(type(iteration) is int and 0 <= iteration < 4 and isinstance(counts, (list, tuple))
                and len(counts) == 6 and all(type(n) is int and 1 <= n <= 3 for n in counts), "original scalar frame metadata missing")
        message = error.args[0][:16384] if len(error.args) == 1 and type(error.args[0]) is str else None
        return {"strict_failure_observed": True, "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_function": helper.__name__, "line": traceback.tb_lineno, "expression": expression,
                "iteration": iteration, "counts": list(counts), "message": message,
                "tensor_storage_read": False, "original_prestate_bytes": "UNPROVEN"}, error
    raise Unsupported("unchanged original strict helper did not reproduce failure; all alternative cells UNRUN")


def strict_instrumented_pair(torch, actual, expected, actual_buffer, reference_buffer, states, refs, counts):
    """Same original observables and 0/0 comparator, not internal raw-bit SHA.

    Length slices equal the original post-increment cache ranges, without
    committing metadata during this separately instrumented noncommitting cell.
    """
    stage = "logits"
    try:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        stage = "streams"
        torch.testing.assert_close(actual_buffer.streams[:sum(counts)], reference_buffer.streams[:sum(counts)], rtol=0, atol=0)
        for state, copied, count in zip(states, refs, counts):
            end = state.mtp_len + count
            require(copied.mtp_len + count == end, "instrumented cache frontier differs")
            for name in ("k", "v", "ks", "vs"):
                stage = "cache-" + name
                torch.testing.assert_close(getattr(state.mtp_kc, name)[:end], getattr(copied.mtp_kc, name)[:end], rtol=0, atol=0)
            stage = "pooled"
            torch.testing.assert_close(state.mtp_pooled[:end // state.ratio], copied.mtp_pooled[:end // copied.ratio], rtol=0, atol=0)
    except AssertionError as error:
        return {"strict_failure_observed": True, "observable": stage,
                "message": error.args[0][:16384] if len(error.args) == 1 and type(error.args[0]) is str else None}, error
    return {"strict_failure_observed": False}, None


def state_metadata(state):
    require(state.compact_indexer is None and getattr(state, "indexer_history_owner", None) is None
            and getattr(state, "dspark_runtime", None) is None and getattr(state, "dflash2", None) is None
            and state.image_positions is None and not state.rope_delta, "only original plain text tiny State is supported")
    return {name: getattr(state, name) for name in ("pos", "mtp_len", "mtp_drafted", "capacity", "version",
                                                   "slot_generation", "history_generation", "kv_identity")} | {"cur": list(state.cur)}


def state_tensors(state):
    """Inventory initialized semantic fields; optional presence is significant."""
    result = {name: getattr(state, name) for name in ("rec", "conv", "ple_tail", "pos_dev", "mtp_pos")}
    # State explicitly uses None for absent history/status device storage.
    # Keep absence distinct too; never infer a Tensor from a field name.
    for name in ("history_generations", "kv_status"):
        if hasattr(state, name):
            result[name] = getattr(state, name)
    for name in ("ikc", "pooled"):
        result.update({f"{name}.{i}": value for i, value in enumerate(getattr(state, name))})
    for i, cache in enumerate([*state.kc, state.mtp_kc]):
        result.update({f"cache.{i}.{name}": getattr(cache, name) for name in ("k", "v", "ks", "vs")})
    result.update(mtp_ikc=state.mtp_ikc, mtp_pooled=state.mtp_pooled)
    return result


def state_storage(state, torch):
    """Separate copyable Tensors from exact optional-field structure.

    Only the two optional State diagnostic slots accept scalar/None/absence.
    All cache/initialized-state fields must remain actual Tensors. Unsupported
    objects fail before allocation rather than disappearing from the oracle.
    """
    tensors, structure = {}, {}
    values = state_tensors(state)
    for name in ("history_generations", "kv_status"):
        if name not in values:
            structure[name] = {"kind": "absent"}
    for name, value in values.items():
        if torch.is_tensor(value):
            tensors[name], structure[name] = value, {"kind": "tensor"}
        else:
            require(name in ("history_generations", "kv_status"), "required initialized State field is not a Tensor: " + name)
            if value is None:
                structure[name] = {"kind": "none"}
            else:
                require(type(value) in (bool, int, float) and (type(value) is not float or math.isfinite(value)),
                        "unsupported optional State field: " + name)
                structure[name] = {"kind": "scalar", "type": type(value).__name__,
                                   "value": value.hex() if type(value) is float else value}
    return tensors, structure


def copy_plan(extents, budget=64 << 20):
    require(type(budget) is int and budget > 0 and type(extents) in (list, tuple) and extents
            and all(type(n) is int and 0 <= n < 2**63 for n in extents), "invalid diagnostic copy extents")
    total = sum(extents)
    require(0 < total <= budget, "diagnostic host copy obligation exceeds budget")
    return {"host_snapshot_bytes": total, "project_device_snapshot_bytes": 0,
            "scope": "initialized contiguous same-dtype copies; not native allocator/RSS bound"}


class Frozen:
    def __init__(self, states, torch):
        self.states, self.torch = states, torch
        for state in states:
            state.kv_check()
        self.metadata = [state_metadata(s) for s in states]
        inventories = [state_storage(s, torch) for s in states]
        self.sources = [row for row, _ in inventories]
        self.structure = [shape for _, shape in inventories]
        # Announce exact project copy extents BEFORE allocating any snapshots.
        self.bytes = sum(t.numel() * t.element_size() for row in self.sources for t in row.values())
        require(self.bytes > 0, "empty State copy obligation")
        require(self.bytes <= 64 << 20, "tiny diagnostic host-copy capacity exceeded")
        require(all(t.is_contiguous() for row in self.sources for t in row.values()), "State snapshots require initialized contiguous ranges")
        self.saved = [{name: torch.empty_like(tensor, device="cpu") for name, tensor in row.items()} for row in self.sources]
        self.capture()

    def capture(self):
        self.torch.cuda.synchronize()
        for state, expected, shape, source, saved in zip(self.states, self.metadata, self.structure, self.sources, self.saved):
            state.kv_check()
            require(state_metadata(state) == expected, "noncommitting metadata changed")
            actual, actual_shape = state_storage(state, self.torch)
            require(actual_shape == shape, "State optional/storage structure changed")
            require(all(actual[n].data_ptr() == source[n].data_ptr() for n in source), "State storage identity changed")
            for name, tensor in source.items():
                saved[name].copy_(tensor)  # Same-dtype contiguous D2H; no project GPU snapshot.


    def restore(self):
        self.torch.cuda.synchronize()  # Diagnostic boundary, never a P3 interval.
        for state, expected, shape, source, saved in zip(self.states, self.metadata, self.structure, self.sources, self.saved):
            state.kv_check()  # Never erase a poison/error as part of restoration.
            require(state_metadata(state) == expected, "State metadata changed across a noncommitting cell")
            actual, actual_shape = state_storage(state, self.torch)
            require(actual_shape == shape, "State optional/storage structure changed")
            require(all(actual[n].data_ptr() == source[n].data_ptr() for n in source), "State storage identity changed")
            for name, tensor in source.items():
                tensor.copy_(saved[name])


class Output:
    def __init__(self, directory):
        self.directory, self.counter, self.bytes = directory, 0, 0

    def tensor(self, value, label):
        # Read only fully written selected ranges, no full spare buffer dump.
        import numpy as np
        if __package__:
            from .gb10_io import opened_file
        else:
            from gb10_io import opened_file
        require(value.is_contiguous(), "diagnostic copy requires a contiguous initialized range")
        extent = value.numel() * value.element_size()
        require(self.counter < 2048 and self.bytes + extent <= 64 << 20, "diagnostic output capacity exceeded")
        array = value.detach().cpu().numpy()
        self.bytes += extent
        raw = array.tobytes(order="C")
        name = f"{self.counter:04d}-{label}.npy"
        self.counter += 1
        with opened_file(self.directory / name, "xb") as stream:
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            import os
            os.fsync(stream.fileno())
        return {"shape": list(array.shape), "dtype": array.dtype.str, "sha256": hashlib.sha256(raw).hexdigest(), "file": name}


def observations(output, buffer, result, states, counts, *, inverse=None, segment_inverse=None, physical_rows=None):
    rows = sum(counts)
    def select(value, head=False):
        array = value.detach().cpu()
        indices = segment_inverse if head else inverse
        return array if indices is None else array[indices]
    # These dedicated MTP/projection fields are not overwritten by later HC/MoE.
    physical = physical_rows if physical_rows is not None else rows
    tensors = {"embedding": buffer.mtp_e[:physical], "fc_e": buffer.mtp_eo[:physical],
               "hc_norm": buffer.mtp_hn[:physical],
               "fc_h": buffer.mtp_hs[:physical * 4].view(physical, 4, 256),
               "attention_projection": buffer.pa[:physical], "q": buffer.q[:physical], "iq": buffer.iq[:physical],
               "attention_output": buffer.attn_o[:physical], "streams": buffer.streams[:physical], "pss": buffer.pss[:physical], "mixed": buffer.mixed[:physical]}
    record = {name: output.tensor(select(tensor), name) for name, tensor in tensors.items()}
    record["logits"] = output.tensor(select(result, head=True), "logits")
    cache = []
    for state, count in zip(states, counts):
        end = state.mtp_len + count
        row = {name: output.tensor(getattr(state.mtp_kc, name)[:end], "cache-" + name) for name in ("k", "v", "ks", "vs")}
        row["pooled"] = output.tensor(state.mtp_pooled[:end // state.ratio], "cache-pooled")
        cache.append(row)
    record["cache"] = cache
    return record


class StageProbe:
    """Only one indicated operation; fixed byte storage, no copy allocation in capture."""
    def __init__(self, torch, buffer, stage, weights):
        self.torch, self.stage, self.metadata = torch, stage, {}
        capacity = max(getattr(buffer, name).numel() * getattr(buffer, name).element_size()
                       for name in ("mtp_in", "mtp_hs", "pa", "streams", "logits"))
        capacity = max(capacity, buffer.moe.y.numel() * buffer.moe.y.element_size())
        capacity = (capacity + 7) // 8 * 8
        names = (("input", "output", "picks") if stage == "streams" else
                 ("input", "output", "down_input", "down_output") if stage == "mixed" else ("input", "output"))
        self.obligation_bytes = len(names) * capacity
        require(torch.cuda.mem_get_info()[0] >= self.obligation_bytes, "probe device copy obligation unavailable")
        self.storage = {name: torch.empty((capacity,), dtype=torch.uint8, device=weights.device)
                        for name in names}
        self.capacity = capacity

    def begin(self):
        self.metadata.clear()  # One cell must never inherit another cell's witness.

    def record(self, name, tensor, *, factor=1, row_axis=0):
        size = tensor.numel() * tensor.element_size()
        require(size <= self.capacity and tensor.is_contiguous(), "indicated probe output exceeds prepared copy layout")
        target = self.storage[name][:size].view(tensor.dtype).view(tensor.shape)
        target.copy_(tensor)
        self.metadata[name] = (tuple(tensor.shape), tensor.dtype, size, factor, row_axis)

    def flush(self, output, inverse=None):
        require("input" in self.metadata and "output" in self.metadata, "indicated operation observer was not invoked")
        record = {}
        for name, (shape, dtype, size, factor, row_axis) in self.metadata.items():
            value = self.storage[name][:size].view(dtype).view(shape).detach().cpu()
            if row_axis:
                value = value.movedim(row_axis, 0)
            if factor > 1:
                value = value.view(value.shape[0] // factor, factor, *value.shape[1:])
            if inverse is not None:
                value = value[inverse]
            record[name] = output.tensor(value, "probe-" + name)
        return record

    def bindings(self, weights):
        from tensorfold.families.qwen4_exp.cuda import mtp, forward, glue
        if self.stage in ("fc_e", "fc_h", "attention_projection", "logits"):
            owner = mtp if self.stage in ("fc_e", "fc_h") else forward
            selected = {"fc_e": weights.mtp.fc_e, "fc_h": weights.mtp.fc_h,
                        "attention_projection": weights.mtp.layer.attn.proj, "logits": weights.head}[self.stage]
            original = owner._mm
            def observed(x, matrix, xs, out, buffers, **kwargs):
                if matrix is selected:
                    self.record("input", x, factor=4 if self.stage == "fc_h" else 1)
                result = original(x, matrix, xs, out, buffers, **kwargs)
                if matrix is selected:
                    self.record("output", result, factor=4 if self.stage == "fc_h" else 1)
                return result
            return [(owner, "_mm", observed)]
        if self.stage == "hc_norm":
            original = glue.rmsnorm
            def norm(*args, **kwargs):
                selected = args[1] is weights.mtp.norm_h
                if selected:
                    self.record("input", args[0])
                result = original(*args, **kwargs)
                if selected:
                    self.record("output", result[0])
                return result
            return [(glue, "rmsnorm", norm)]
        if self.stage == "mixed":
            down, up = weights.mtp.mixer.down, weights.mtp.mixer.up
            original_down, original_mm = down.partials, forward._mm
            def partial(x):
                self.record("down_input", x)
                result = original_down(x)
                self.record("down_output", result, row_axis=1)
                return result
            def mix_mm(x, matrix, xs, out, buffers, **kwargs):
                if matrix is up:
                    self.record("input", x)
                result = original_mm(x, matrix, xs, out, buffers, **kwargs)
                if matrix is up:
                    self.record("output", result)
                return result
            return [(down, "partials", partial), (forward, "_mm", mix_mm)]
        if self.stage == "streams":
            bank = weights.mtp.layer.moe.experts
            original = bank.forward
            def experts(x, picks, out=None):
                self.record("input", x)
                self.record("picks", picks)
                result = original(x, picks, out=out)
                self.record("output", result)
                return result
            return [(bank, "forward", experts)]
        return []


class SelectionProbe:
    """Copy per-State QSA decisions before shared scratch reuse, for ONE frame."""
    def __init__(self, torch, weights, positions, counts, order, width):
        self.torch, self.positions, self.counts = torch, positions, counts
        self.order, self.width, self.cursor = order, width, 0
        self.budget, self.ratio = weights.cfg.index_budget, weights.cfg.index_ratio
        require(sorted(positions) == list(range(6)) and set(counts) == set(positions)
                and all(type(counts[s]) is int and 1 <= counts[s] <= 3
                        and type(positions[s]) is int and 0 <= positions[s] <= 32 - counts[s] for s in positions),
                "selection probe requires the exact frozen tiny frame")
        self.bytes = 6 * 3 * (2 + self.budget + self.ratio) * 4
        require(torch.cuda.mem_get_info()[0] >= self.bytes, "selection copy obligation unavailable")
        self.flags = torch.empty((6, 3), dtype=torch.int32, device=weights.device)
        self.nk = torch.empty_like(self.flags)
        self.ids = torch.empty((6, 3, self.budget + self.ratio), dtype=torch.int32, device=weights.device)

    def configure(self, order, width):
        self.order, self.width, self.cursor = order, width, 0

    def binding(self):
        from tensorfold.cuda.sm70 import family_ops
        original = family_ops.qsa_rows
        def observed(iq, pooled, pos0, scratch, rows, **kwargs):
            if self.cursor == 6:
                self.cursor = 0  # Each warm/capture call executes all six States.
            slot = self.order[self.cursor]
            self.cursor += 1
            live = self.counts[slot]
            require(rows == (self.width or live), "QSA invocation differs from frozen frame")
            result = original(iq, pooled, pos0, scratch, rows, **kwargs)
            self.flags[slot, :live].copy_(scratch.sparse[:live])
            self.nk[slot, :live].copy_(scratch.nk[:live])
            for row in range(live):
                end = self.positions[slot] + row + 1
                if end // self.ratio > self.budget // self.ratio:
                    # Only the source-defined sparse branch has initialized IDs.
                    count = self.budget + end % self.ratio
                    self.ids[slot, row, :count].view(self.torch.uint8).copy_(scratch.ids[row, :count].view(self.torch.uint8))
            return result
        return [(family_ops, "qsa_rows", observed)]

    def flush(self):
        require(self.cursor == 6, "actual per-State selection observer was not invoked")
        result = []
        for slot in sorted(self.positions):
            live = self.counts[slot]
            flags = self.flags[slot, :live].cpu().tolist()
            lengths = self.nk[slot, :live].cpu().tolist()
            for row, (flag, count) in enumerate(zip(flags, lengths)):
                end = self.positions[slot] + row + 1
                sparse = end // self.ratio > self.budget // self.ratio
                require(flag == int(sparse) and count == (self.budget + end % self.ratio if sparse else end), "QSA discrete count/flag differs")
                selected = (self.ids[slot, row, :count].cpu().tolist() if sparse else list(range(end)))
                require(all(type(n) is int and 0 <= n < end for n in selected)
                        and selected == sorted(set(selected)), "QSA selected positions invalid/noncanonical")
                result.append({"slot": slot, "row": row, "sparse": bool(flag), "count": count,
                               "selected": selected, "dense_representation": "implicit causal prefix" if not sparse else None})
        return result


def _same_tensors(torch, actual, expected, label):
    actual, expected = actual.detach().cpu(), expected.detach().cpu()
    require(actual.shape == expected.shape and actual.dtype == expected.dtype
            and torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)), "discrete input identity differs: " + label)


@contextmanager
def dispatch_scope(mode):
    """Retire the thread-local mode even after an authentic operator error."""
    primary, errors, entered = None, [], False
    try:
        mode.__enter__()
        entered = True
        yield
    except BaseException as error:
        primary = error
    if entered:
        try:
            mode.__exit__(type(primary) if primary else None, primary, primary.__traceback__ if primary else None)
        except BaseException as error:
            errors.append(error)
    raise_failures(primary, errors)


class AttentionArithmeticProbe:
    """One actual eager attention invocation; never execute shadow arithmetic.

    Nine flat allocations make every saved prefix contiguous, including the
    strided borrowed key transpose. Same-device copy_ writes these already-paid
    buffers; CPU serialization follows the existing whole-cell completion.
    """
    LAYOUT = {"query": (384, "float16"), "key": (4096, "float16"),
              "QK": (96, "float32"), "softmax_input": (96, "float32"),
              "softmax_output": (96, "float32"), "P": (96, "float16"),
              "value": (4096, "float16"), "PV": (384, "float32"), "final": (384, "float16")}
    BYTES = 20800

    def __init__(self, torch, state):
        require(state.capacity == 32 and state.mtp_len == 11, "selected slot4 attention frontier differs")
        self.torch, self.state = torch, state
        self.storage, self.metadata = {}, {}
        self.active, self.calls, self.mm_calls, self.softmax_calls = False, 0, 0, 0
        self.mode = None
        self.expected_keys = None
        # Journal self BEFORE allocation through the caller; all partial
        # allocations stay owned until the ordinary diagnostic final fence.
        self.device = state.mtp_kc.k.device

    def allocate(self):
        require(self.torch.cuda.mem_get_info()[0] >= self.BYTES, "attention probe device copy obligation unavailable")
        for name, (count, dtype) in self.LAYOUT.items():
            self.storage[name] = self.torch.empty((count,), dtype=getattr(self.torch, dtype), device=self.device)

    def begin(self, keys):
        require(not self.active and keys in (12, 32), "invalid/reentrant attention probe cell")
        self.metadata.clear()
        self.calls = self.mm_calls = self.softmax_calls = 0
        self.expected_keys = keys

    def record(self, name, tensor):
        count, dtype = self.LAYOUT[name]
        require(name not in self.metadata and self.torch.is_tensor(tensor)
                and tensor.dtype == getattr(self.torch, dtype) and tensor.device == self.device
                and tensor.numel() <= count, "attention probe tensor capacity/type/duplicate differs: " + name)
        shapes = {"query": (3, 128), "key": (128, self.expected_keys),
                  "QK": (3, self.expected_keys), "softmax_input": (3, self.expected_keys),
                  "softmax_output": (3, self.expected_keys), "P": (3, self.expected_keys),
                  "value": (self.expected_keys, 128), "PV": (3, 128), "final": (3, 128)}
        require(tuple(tensor.shape) == shapes[name], "attention probe actual geometry differs: " + name)
        self.storage[name][:tensor.numel()].view(tensor.shape).copy_(tensor)
        self.metadata[name] = tuple(tensor.shape)

    def dispatch(self, function, args, kwargs, softmax):
        if function is not softmax:
            return function(*args, **kwargs)
        require(self.active and self.mm_calls == 1 and self.softmax_calls == 0,
                "unexpected/repeated actual attention softmax")
        # Exact pinned aten schema: _softmax(Tensor self, int dim, bool
        # half_to_float). Preserve positional/keyword passing to the delegate.
        dim = args[1] if len(args) > 1 else kwargs.get("dim")
        half_to_float = args[2] if len(args) > 2 else kwargs.get("half_to_float")
        require(dim == -1 and half_to_float is False, "actual softmax schema differs")
        self.record("softmax_input", args[0] if args else kwargs["self"])
        self.softmax_calls += 1
        result = function(*args, **kwargs)
        self.record("softmax_output", result)
        return result

    def bindings(self, ops, *, mode_type=None, softmax=None):
        if mode_type is None:
            import torch.utils._python_dispatch as dispatch_module
            from torch.utils._python_dispatch import TorchDispatchMode
            mode_type = TorchDispatchMode
            self.dispatch_source = {"path": dispatch_module.__file__,
                                    "sha256": hashlib.sha256(Path(dispatch_module.__file__).read_bytes()).hexdigest()}
        if softmax is None:
            softmax = self.torch.ops.aten._softmax.default
            require(str(softmax._schema) == "aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor",
                    "pinned actual softmax operator schema differs")
        probe = self
        class SelectedMode(mode_type):
            def __torch_dispatch__(self, function, types, args=(), kwargs=None):
                return probe.dispatch(function, args, {} if kwargs is None else kwargs, softmax)
        original_attention, original_mm = ops.attention, ops._matmul32
        def mm(left, right):
            if not self.active:
                return original_mm(left, right)
            require(self.mm_calls < 2, "unexpected extra attention matrix operation")
            qk = self.mm_calls == 0
            self.record("query" if qk else "P", left)
            self.record("key" if qk else "value", right)
            self.mm_calls += 1
            result = original_mm(left, right)
            self.record("QK" if qk else "PV", result)
            return result
        def attention(*args, **kwargs):
            def arg(index, name):
                return args[index] if len(args) > index else kwargs[name]
            kc, vc, pos0, rows = arg(1, "kc"), arg(2, "vc"), arg(3, "pos0"), arg(5, "rows")
            if kc is not self.state.mtp_kc.k:
                return original_attention(*args, **kwargs)
            require(not self.active and self.calls == 0 and vc is self.state.mtp_kc.v
                    and pos0 is self.state.mtp_pos and rows == 1
                    and min(kc.shape[0], kwargs.get("context") or kc.shape[0]) == self.expected_keys,
                    "selected State attention identity/call geometry differs")
            self.calls += 1
            self.active = True
            try:
                self.mode = SelectedMode()  # Retained by the operation journal on uncertain exit.
                with dispatch_scope(self.mode):
                    result = original_attention(*args, **kwargs)
                if self.mm_calls != 2 or self.softmax_calls != 1:
                    raise Unsupported("actual attention operators unobserved")
                self.record("final", result[0])
                return result
            finally:
                self.active = False
        return [(ops, "attention", attention), (ops, "_matmul32", mm)]

    def flush(self, output):
        require(self.calls == 1 and self.mm_calls == 2 and self.softmax_calls == 1
                and set(self.metadata) == set(self.LAYOUT), "attention arithmetic probe incomplete")
        return {name: output.tensor(self.storage[name][:math.prod(shape)].view(shape), "attention-" + name)
                for name, shape in self.metadata.items()}


def execute(args):
    require(_ACTIVE[0] is None, "previous diagnostic still owns resources; fresh process required")
    journal = {"graphs": None, "decoder": None, "frozen": None, "reference": None, "anchor_error": None,
               "raw_anchor_graph": None, "raw_anchor_decoder": None, "raw_anchor_reference": None, "anchor_retirement_proven": False}
    _ACTIVE[0] = journal
    primary, errors, report, torch = None, [], None, None
    try:
        import sys
        import tensorfold
        import torch
        require("site-packages" in Path(tensorfold.__file__).parts, "installed runtime required")
        require(torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0), "actual SM70 required")
        from tensorfold.cuda.sm70.software_binding import issue_source_binding
        software = issue_source_binding(required=True).record()
        # Test fixtures are canonical source inputs, never editable mirrors.
        from test_flashnext_sm70_mtp import _weights_with_stock_mtp_protocol
        from test_sm70_gguf_prepared import dense_fixture
        from test_flashnext_sm70_bound_graphs import _qualify_six_lanes, _qualify_mtp
        from tensorfold.cuda.sm70 import family_ops
        from tensorfold.families.qwen4_exp.cuda import padded_linears, mtp
        from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder, _slot
        from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs
        from tensorfold.families.qwen4_exp.cuda.decode import prefill
        from tensorfold.families.qwen4_exp.cuda.state import Buffers
        from tensorfold.families.qwen4_exp.cuda.forward import _kv_check
        require(torch.version.git_version == "04cf2557f315a350e7eea7ab064288def65db75a", "pinned original framework required")
        output = Output(args.out)
        report = {"schema": "tensorfold.mtp-four-cell-diagnostic-v1", "qualification": False, "P3_valid": False,
                  "image_id_declared": args.image_id, "source_commit_declared": args.source_commit,
                  "actual_runtime": {"torch": torch.__version__, "torch_git": torch.version.git_version,
                                     "cuda": torch.version.cuda, "python": sys.version}, "software": software,
                  "case": args.case, "seed": 183, "steps": [], "completed": False,
                  "source_files": {str(Path(module.__file__)): hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                                   for module in (mtp, family_ops, padded_linears, sys.modules["test_flashnext_sm70_mtp"],
                                                  sys.modules["test_sm70_gguf_prepared"], sys.modules["test_flashnext_sm70_bound_graphs"],
                                                  sys.modules["test_flashnext_sm70_family"])},
                  "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "math_modes": {"fp16_reduced_precision": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                                 "fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation,
                                 "inference_mode": torch.is_inference_mode_enabled()},
                  "retirement_verified": False}
        policy = [(family_ops, "_NATIVE_FAMILY", True), (family_ops, "_NATIVE_ATTENTION", False),
                  (padded_linears, "_ENABLED", args.case == "prepared")]
        with attributes(policy):
            weights = _weights_with_stock_mtp_protocol()  # Own original seed/draw sequence.
            if args.case == "prepared":
                weights = dense_fixture(weights)
                journal["native_work_attempted"] = True
                _qualify_six_lanes(weights, "int8")  # Original preceding ordered work.
            require((weights.cfg.hidden, weights.cfg.streams, weights.cfg.heads, weights.cfg.head_dim, weights.cfg.layers)
                    == (256, 4, 3, 128, 2) and weights.comm is None and not weights.mtp.layer.linear,
                    "only exact original single-GPU attention-MTP fixture supported")
            journal["anchor_weights"] = weights
            journal["native_work_attempted"] = True
            def retain_anchor(error):
                # Keep owner references independently of traceback/frame clearing.
                # This reads no Tensor storage, including retired bound outputs.
                journal["anchor_error"] = error
                trace = error.__traceback__
                while trace is not None:
                    if trace.tb_frame.f_code is _qualify_mtp.__code__:
                        values = trace.tb_frame.f_locals
                        journal["raw_anchor_graph"] = values.get("graphs")
                        journal["raw_anchor_decoder"] = values.get("decoder")
                        journal["raw_anchor_reference"] = values.get("reference")
                        break
                    trace = trace.tb_next
            report["original_anchor"], journal["anchor_error"] = original_anchor(_qualify_mtp, weights, "int8", retain=retain_anchor)
            journal["anchor_retirement_proven"] = True  # Genuine original comparison propagated through its completed finally.
            report["original_input_match"] = "UNPROVEN"
            report["experiment_status"] = "CONDITIONAL_MECHANISM_DIAGNOSTIC_ONLY"
            # Separate, clearly labelled instrumented rerun. Same seed/call
            # order is not claimed to prove original Tensor byte equivalence.
            weights = _weights_with_stock_mtp_protocol()
            if args.case == "prepared":
                weights = dense_fixture(weights)
                _qualify_six_lanes(weights, "int8")
            journal["weights"] = weights
            matrices = {"fc_e": weights.mtp.fc_e, "fc_h": weights.mtp.fc_h, "head": weights.head,
                        "attention_projection": weights.mtp.layer.attn.proj, "attention_out": weights.mtp.layer.attn.o}
            for role, hc in (("attn_hc", weights.mtp.layer.attn_hc), ("mlp_hc", weights.mtp.layer.mlp_hc), ("mixer", weights.mtp.mixer)):
                matrices[role + "_down"], matrices[role + "_up"] = hc.down, hc.up
            report["weight_inputs"] = {name: output.tensor(matrix.weight, "weight-" + name) for name, matrix in matrices.items()}
            for name in ("gate", "up", "down"):
                report["weight_inputs"]["expert_" + name] = output.tensor(getattr(weights.mtp.layer.moe.experts, name), "weight-expert-" + name)
            report["weight_inputs"]["router"] = output.tensor(weights.mtp.layer.moe.router, "weight-router")
            report["weight_inputs"]["norm_h"] = output.tensor(weights.mtp.norm_h, "weight-norm-h")
            report["weight_inputs"]["norm_e"] = output.tensor(weights.mtp.norm_e, "weight-norm-e")
            for role, hc in (("attn_hc", weights.mtp.layer.attn_hc), ("mlp_hc", weights.mtp.layer.mlp_hc), ("mixer", weights.mtp.mixer)):
                report["weight_inputs"][role + "_scale"] = output.tensor(hc.scale, "weight-" + role + "-scale")
            for name in ("q_scale", "k_scale", "iq_scale", "ik_scale"):
                report["weight_inputs"][name] = output.tensor(getattr(weights.mtp.layer.attn, name), "weight-" + name)
            decoder = journal["decoder"] = MultiDecoder(weights, slots=6, capacity=32, depth=2, prefill_rows=8,
                                                         kv_dtype="int8", graphs=False)
            graphs = journal["graphs"] = RoundGraphs(decoder, 1024 * 2**20)
            require(graphs.bound is not None, "actual bound graph implementation required")
            reference = journal["reference"] = Buffers(weights, decoder.buf.rows, 32, moe_prefill=decoder.mbuf.moe.plan.prefill)
            states = list(reversed(decoder.free))
            journal["native_work_attempted"] = True
            for state in states:
                prefill(_slot(weights, state, decoder.buf, decoder.mbuf, decoder.pbuf, 32, 8),
                        [4, 17, 3, 99, 33, 2, 7, 5, 6], None)
            for iteration, counts in enumerate(COUNTS):
                # Same clone-before-randn order as _qualify_mtp. Extra snapshots
                # below do not draw RNG or change operator inputs.
                refs = [state.clone() for state in states]
                inputs = torch.randn((sum(counts), 1024), device=weights.device, dtype=torch.float16) * .1
                extents = [t.numel() * t.element_size() for owners in (states, refs, states)
                           for state in owners for t in state_storage(state, torch)[0].values()]
                obligation = copy_plan(extents)
                journal["frozen"] = frozen = Frozen(states, torch)
                frozen_refs = Frozen(refs, torch)
                final_state = Frozen(states, torch)
                journal["final_state"] = final_state
                journal["frozen_refs"] = frozen_refs
                ids = [[(13 * index + row) % weights.cfg.vocab for row in range(count)] for index, count in enumerate(counts)]
                plan = row_plan([graphs.bound.indices[id(s)] for s in states], counts,
                                width=1 if all(n == 1 for n in counts) else 3)
                step = {"iteration": iteration, "plan": plan, "metadata": frozen.metadata,
                        "input": output.tensor(inputs, "frozen-HC-input"), "ids": ids,
                        "rng_state": output.tensor(torch.cuda.get_rng_state(), "cuda-RNG-state"),
                        "copy_obligation": obligation,
                        "copy_obligation_host_bytes": frozen.bytes + frozen_refs.bytes + final_state.bytes,
                        "project_device_snapshot_bytes": 0, "execution_order": ["A", "D", "A_repeat", "B", "C"],
                        "graph_budget_bytes": 1024 * 2**20, "cells": {}, "discrete_inputs": False}
                step["frozen_state"] = [{name: output.tensor(value, "frozen-State-" + name.replace(".", "-"))
                                          for name, value in row.items()} for row in frozen.saved]
                step["frozen_state_structure"] = frozen.structure
                require(frozen.metadata == frozen_refs.metadata, "original oracle clone metadata differs")
                require(frozen.structure == frozen_refs.structure, "original oracle clone State structure differs")
                for original, copied in zip(frozen.saved, frozen_refs.saved):
                    for name in original:
                        _same_tensors(torch, original[name], copied[name], "oracle clone " + name)
                report["steps"].append(step)
                def windows(owners):
                    at, result = 0, []
                    for state, tokens, count in zip(owners, ids, counts):
                        result.append((state, tokens, inputs[at:at + count]))
                        at += count
                    return result
                expected_ids = torch.tensor([token for window in ids for token in window], dtype=decoder.mbuf.ids.dtype, device="cpu")
                def compact(name, context, *, observe=True):
                    frozen_refs.restore()
                    segs = mtp.mtp_stage(reference, windows(refs))
                    _same_tensors(torch, reference.mtp_in[:sum(counts)], inputs, name + " HC")
                    _same_tensors(torch, reference.ids[:sum(counts)], expected_ids, name + " IDs")
                    result = mtp.mtp_compute(weights, segs, reference, context=context)
                    torch.cuda.synchronize()
                    comparison = result.clone() if name == "A" else result
                    if observe:
                        step["cells"][name] = observations(output, reference, result, refs, counts)
                    return comparison

                def mapped(name, captured, *, observe=True):
                    frozen.restore()
                    segs = mtp.mtp_stage(decoder.mbuf, windows(states))
                    if captured:
                        result = graphs.mtp_step(segs)
                        if result is None:
                            raise Unsupported("D actual graph declined; no eager fallback cell")
                        _kv_check(segs, comm=weights.comm)
                        torch.cuda.synchronize()
                        buffer = graphs.bound.buffers["mtp"]
                        inverse = graphs.bound.row_maps["mtp"][1][:sum(counts)]
                        segments = graphs.bound.segment_maps["mtp"][1][:6]
                        head_inverse = None  # Real forward already restored its six head rows.
                    else:
                        prepared = graphs.bound._prepare("mtp", segs, decoder.mbuf)
                        if prepared is None:
                            raise Unsupported("C bound preparation declined; no cell fabricated")
                        key, lanes, buffer, bound, run, physical_counts, inverse, segments = prepared
                        require(key[1:] == (plan["width"], 256), "actual bound key differs")
                        result = run()
                        _kv_check(bound, comm=weights.comm)
                        torch.cuda.synchronize()
                        head_inverse = plan["segment_inverse"]
                    physical = 6 * plan["width"]
                    require(inverse.cpu().tolist() == plan["inverse"] and segments.cpu().tolist() == plan["segment_inverse"], "actual row maps differ")
                    _same_tensors(torch, buffer.mtp_in[:physical].detach().cpu()[plan["inverse"]], inputs, name + " canonical HC")
                    _same_tensors(torch, decoder.mbuf.ids[:sum(counts)], expected_ids, name + " source IDs")
                    _same_tensors(torch, buffer.ids[:physical].detach().cpu()[plan["inverse"]], expected_ids, name + " canonical IDs")
                    require(buffer.last_host[:6].tolist() == plan["last"], "actual last rows differ")
                    if observe:
                        step["cells"][name] = observations(output, buffer, result, states, counts,
                                                           inverse=plan["inverse"], segment_inverse=head_inverse, physical_rows=physical)
                    return result

                # Preserve original A→D before any alternative-context warmup.
                original_rng_before_A = torch.cuda.get_rng_state()
                expected_comparison = compact("A", None)
                original_rng_after_A = torch.cuda.get_rng_state()
                step["rng_after_A"] = output.tensor(original_rng_after_A, "RNG-after-original-A")
                actual_comparison = mapped("D", True)
                step["strict_control"], strict_error = strict_instrumented_pair(torch, actual_comparison, expected_comparison,
                                            decoder.mbuf, reference, states, refs, counts)
                if strict_error is not None:
                    journal["instrumented_error"] = strict_error
                final_state.capture()
                original_rng_after_D = torch.cuda.get_rng_state()
                step["raw_pair_difference"] = first_difference(step["cells"]["A"], step["cells"]["D"])
                if not step["strict_control"]["strict_failure_observed"]:
                    # Do NOT perturb either scratch/cache trajectory with B/C
                    # before the original strict pair has actually failed.
                    step["difference"] = {"axis": "instrumented_strict_pair_passed", "alternative_cells": "UNRUN"}
                    write_json(args.out / "diagnostic.json", report)
                    for state, count in zip(states, counts):
                        state.set_mtp_len(state.mtp_len + count)
                    continue
                for name, context in (("A_repeat", None), ("B", 256)):
                    torch.cuda.set_rng_state(original_rng_before_A)
                    compact(name, context)
                torch.cuda.set_rng_state(original_rng_before_A)
                mapped("C", False)
                final_state.restore()
                torch.cuda.set_rng_state(original_rng_after_D)
                step["discrete_inputs"] = True
                step["raw_pair_difference"] = first_difference(step["cells"]["A"], step["cells"]["D"])
                step["difference"] = classify_strict_frame(step["cells"], step["strict_control"])
                write_json(args.out / "diagnostic.json", report)
                if step["difference"]["axis"] != "no_difference_observed":
                    report["instrumented_strict_failure_observed"] = True
                    report["anchor_frame_metadata_match"] = (iteration == report["original_anchor"]["iteration"]
                                                               and list(counts) == report["original_anchor"]["counts"])
                    report["first_observed_difference"] = step["difference"]
                    stage = step["difference"]["first_observed_retained_stage"]
                    axis = step["difference"]["axis"]
                    pair = {"same_eager_not_reproduced": ("A", "A_repeat"), "context_extent": ("A", "B"),
                            "canonical_padding_order": ("B", "C"), "graph_state_or_capture": ("C", "D"),
                            "strict_observable_failure_without_raw_difference": ("A", "D")}[axis]
                    if axis == "graph_state_or_capture":
                        # Cached graphs lack probe nodes. Retire first, then use
                        # one fresh same-frame capture, never patch an exec.
                        graphs.close()
                        graphs = journal["graphs"] = RoundGraphs(decoder, 1024 * 2**20)
                    supported = stage in ("fc_e", "fc_h", "hc_norm", "attention_projection", "streams", "mixed", "logits")
                    probe = (StageProbe(torch, decoder.mbuf, stage, weights) if supported else None)
                    selection = None
                    if stage == "attention_output":
                        positions = {slot: state.mtp_len for slot, state in zip(plan["slots"], states)}
                        counts_by_slot = dict(zip(plan["slots"], counts))
                        selection = SelectionProbe(torch, weights, positions, counts_by_slot, plan["slots"], None)
                    journal["probe"], journal["selection"] = probe, selection
                    bindings = probe.bindings(weights) if probe is not None else selection.binding() if selection is not None else []
                    detail = {"stage": stage, "pair": pair, "qualification": False,
                              "device_copy_obligation_bytes": probe.obligation_bytes if probe is not None else selection.bytes if selection is not None else 0,
                              "scope": "only indicated operation/discrete selection; no full-model hook"}
                    step["first_stage_capture"] = detail
                    if bindings:
                        with attributes(bindings):
                            for cell in pair:
                                if probe is not None:
                                    probe.begin()
                                torch.cuda.set_rng_state(original_rng_before_A)
                                canonical = cell in ("C", "D")
                                if selection is not None:
                                    selection.configure(plan["canonical_slots"] if canonical else plan["slots"], plan["width"] if canonical else None)
                                if canonical:
                                    mapped("probe-" + cell, cell == "D", observe=False)
                                else:
                                    compact("probe-" + cell, None if cell in ("A", "A_repeat") else 256, observe=False)
                                detail[cell] = (probe.flush(output, (plan["segment_inverse"] if stage == "logits" else plan["inverse"]) if canonical else None)
                                                if probe is not None else selection.flush())
                        detail["equal"] = content(detail[pair[0]]) == content(detail[pair[1]])
                        detail["interpretation"] = "observed data only; unequal input/discrete decisions prevent arithmetic-cause attribution"
                    else:
                        detail["scope"] = "first retained embedding/preparation/mixer/cache boundary observed; internal arithmetic not isolated"
                    if stage == "attention_output" and axis == "context_extent":
                        require(iteration == 1 and plan["slots"] == [5, 4, 3, 2, 1, 0]
                                and counts == (1, 1, 1, 1, 1, 1) and detail.get("equal") is True,
                                "selected arithmetic probe requires the actual frozen step1/QSA witness")
                        selected = refs[1]
                        arithmetic = AttentionArithmeticProbe(torch, selected)
                        journal["arithmetic_probe"] = arithmetic  # Before first allocation.
                        ar = {"status": "INCOMPLETE", "logical_row": 1, "slot": 4, "end": 12,
                              "device_copy_obligation_bytes": arithmetic.BYTES,
                              "scope": "one actual eager attention call; conditional mechanism data, not original causality",
                              "cells": {}}
                        step["arithmetic_probe"] = ar
                        arithmetic.allocate()
                        arithmetic_bindings = arithmetic.bindings(family_ops)
                        ar["dispatch_source"] = arithmetic.dispatch_source
                        with attributes(selection.binding() + arithmetic_bindings):
                            for cell, context, keys in (("A", None, 12), ("B", 256, 32)):
                                arithmetic.begin(keys)
                                selection.configure(plan["slots"], None)
                                torch.cuda.set_rng_state(original_rng_before_A)
                                result = compact("arithmetic-" + cell, context, observe=False)
                                retained = observations(output, reference, result, refs, counts)
                                qsa = selection.flush()
                                ar["cells"][cell] = {"keys": keys, "retained": retained, "QSA": qsa,
                                                     "operators": arithmetic.flush(output)}
                                if content(retained) != content(step["cells"][cell]) or qsa != detail[cell]:
                                    ar["status"] = "UNSUPPORTED_PERTURBED_WITNESS"
                                    raise Unsupported("attention observers changed the controlled retained/discrete witness")
                        ar["status"] = "CONDITIONAL_MECHANISM_DIAGNOSTIC_ONLY"
                    final_state.restore()
                    torch.cuda.set_rng_state(original_rng_after_D)
                    break
                # Preserve original D trajectory for the next original draw.
                for state, count in zip(states, counts):
                    state.set_mtp_len(state.mtp_len + count)
            if not report.get("instrumented_strict_failure_observed"):
                raise Unsupported("separate instrumented strict failure not reproduced; alternatives UNRUN, original anchor remains failed") from journal["anchor_error"]
            report["completed"] = True
    except BaseException as error:
        primary = error
    if journal["raw_anchor_graph"] is not None and not journal["anchor_retirement_proven"]:
        errors.append(RuntimeError("original helper graph retirement uncertified; owner retained until process exit"))
    graph = journal["graphs"]
    if graph is not None:
        try:
            graph.close()
        except BaseException as error:
            errors.append(error)
    if torch is not None:
        try:
            torch.cuda.synchronize()
        except BaseException as error:
            errors.append(error)
    retirement_verified = not errors
    if journal.get("native_work_attempted"):
        try:
            if __package__:
                from .v100_native_contract_driver import observe_native_elfs
            else:
                from v100_native_contract_driver import observe_native_elfs
            origin = observe_native_elfs(["tf_sm70_family_"])
            if report is not None:
                report["native_origin"] = origin
            require(not origin["missing_required_prefixes"], "required post-use native origin absent")
        except BaseException as error:
            errors.append(error)
            if report is not None:
                report["native_origin_failure_type"] = type.__dict__["__name__"].__get__(type(error))
    try:
        if report is None:
            report = {"schema": "tensorfold.mtp-four-cell-diagnostic-v1", "qualification": False, "P3_valid": False, "completed": False}
        report["retirement_verified"] = retirement_verified
        report["failed"] = primary is not None or bool(errors)
        report["error_types"] = [type.__dict__["__name__"].__get__(type(e)) for e in ([primary] if primary is not None else []) + errors]
        write_json(args.out / "diagnostic.json", report)
    except BaseException as error:
        errors.append(error)
    if primary is None and not errors:
        _ACTIVE[0] = None
    raise_failures(primary, errors)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("reference", "prepared"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args(argv)
    require(re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id) and re.fullmatch(r"[0-9a-f]{40}", args.source_commit), "exact declared image/source required")
    require(args.out.is_absolute() and not args.out.exists() and all(not p.is_symlink() for p in args.out.parents), "fresh absolute owned evidence directory required")
    require(_ACTIVE[0] is None, "previous diagnostic still owns resources; fresh process required")
    args.out.mkdir(mode=0o700)
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
