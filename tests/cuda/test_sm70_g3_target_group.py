"""Source41 native component group controls, not TP8 policy/model adoption.

Run only via the separately authorized installed-package native Host packet.
No runtime/source overlay, monkeypatch, fake world8 communicator, numerical
acceptance tolerance or P3 claim is supplied by these component tests.
"""

import hashlib
import json
import threading
from pathlib import Path

import pytest
import torch

from tensorfold.cuda.sm70 import gdn_chunk64, gdn_chunk64_artifact, gdn_native
from tensorfold.cuda.sm70.owned_family import prepare as prepare_family
from tensorfold.cuda.sm70.rotorquant_compact import ISO_NAME
from tensorfold.families.qwen4_exp.cuda import forward, gdn
from tensorfold.families.qwen4_exp.cuda import gdn_chunk64_prefill as adapter
from tensorfold.families.qwen4_exp.cuda import gdn_prefill_policy as policy
from tensorfold.families.qwen4_exp.cuda.forward import Cut
from tensorfold.families.qwen4_exp.cuda.state import KVNumericError
from tests.g3_packed_native_support import (
    CAPACITY,
    LAYERS,
    SHAPES,
    CutCopies,
    cut_rows,
    load_input,
    make_state,
    observed_event,
    payload_bound,
)
from tools.v100_g3_native_observation import observe as observe_compiler
from tools.v100_g3_native_observation import retained_jit_files
from tools.v100_gdn_chunk64_trial import boundary_errors, half_epilogue_reference, reference, same_storage_bits
from tools.worker_lifetime import Task, drain


def compiler(data):
    return gdn_chunk64.Compiler(
        data["cuda_home"], data["nvcc_sha256"], gdn_chunk64_artifact.OwnedArtifact(**data["artifact"])
    )


def assert_finite(metrics):
    assert metrics["finite"] and metrics["reference_nonfinite_elements"] == 0


def pack(states, lengths):
    segs, cuts, start = [], [], 0
    for state, rows in zip(states, lengths):
        segs.append((state, start, start + rows))
        cuts.extend(Cut(row, at=start) for row in cut_rows(rows))
        start += rows
    return tuple(segs), tuple(cuts)


def source_layers(device, generator):
    layers = []
    for index in range(LAYERS):
        parameter = torch.full((6,), -0.5 if index == 0 else 0.0, device=device)
        layers.append(
            SimpleLayer(
                torch.randn(1280, 4, device=device, dtype=torch.float16, generator=generator) * 0.125,
                parameter,
                torch.linspace(-1, 1, 6, device=device),
                torch.ones(128, device=device, dtype=torch.float16),
                index == 0,
            )
        )
    return layers


class SimpleLayer:
    """GDN-only fixture parameters; no loaded model/comm/adoption claim."""

    def __init__(self, conv, parameter, bias, norm, negative):
        self.conv, self.dt_bias, self.norm = conv, bias, norm
        self.a_log, self.a_neg = (None, parameter) if negative else (parameter, None)


def copy_frontend(prepared, command, captures, layer, start, end):
    for destination, source in zip(captures[layer], command.half.core_operands):
        destination[:, start:end].copy_(source)


def incumbent(states, before, projections, layers, lengths, family, device):
    """The ACTUAL installed default-off _prefill_chain, including its cuts."""
    scratch = gdn.GDNScratch(CAPACITY, device, 2, 6, dtype=torch.float16)
    output = torch.empty(CAPACITY, 768, device=device, dtype=torch.float16)
    sums = torch.empty(CAPACITY, 24, device=device)
    b = type("IncumbentBuffers", (), {})()
    b.gout, b.gxs, b.prefill_gdn_scratch, b.native_family = output, sums, scratch, family
    results = {}
    start = 0
    for state, rows in zip(states, lengths):
        baseline = state.clone()
        baseline.rec[0].copy_(before[state][0])
        baseline.conv.copy_(before[state][1])
        baseline.cur = [0] * LAYERS
        points = tuple(Cut(row) for row in cut_rows(rows))
        for index, g in enumerate(layers):
            b.proj = projections[index : index + 1]
            # Local projection view, actual original history and real native
            # caller. No candidate flag/provider exists on this baseline.
            b.proj = b.proj[:, start : start + rows]
            forward._prefill_chain(g, baseline, index, b, 0, rows, NoneCfg(), points)
            results[state, index] = (
                output[:rows].cpu(),
                sums[:rows].cpu(),
                baseline.rec[baseline.cur[index], index : index + 1].cpu(),
            )
        # Each CPU diagnostic copy positively completed this baseline. Release
        # its State/cut owners before allocating the next independent clone.
        del baseline, points
        start += rows
    return results


class NoneCfg:
    eps, conv_dim = 1e-6, 1280


def one_frame(prepared, states, lengths, layers, projections, event, iteration, codec, name, records, data):
    rows = sum(lengths)
    segs, cuts = pack(states, lengths)
    before = {state: (state.rec[state.cur[0]].cpu(), state.conv.cpu(), tuple(state.cur)) for state in states}
    output = torch.empty(rows, 768, device=prepared.frontend.device, dtype=torch.float16)
    sums = torch.empty(rows, 24, device=prepared.frontend.device)
    captures = [
        tuple(
            torch.empty_like(value[:, :rows])
            for value in (
                prepared.workspace.q,
                prepared.workspace.k,
                prepared.workspace.v,
                prepared.workspace.raw_gate,
                prepared.workspace.beta,
                prepared.workspace.raw_output,
            )
        )
        for _ in range(LAYERS)
    ]
    group = prepared.begin_group(segs, cuts, cut_budget_bytes=adapter.cut_bytes(LAYERS, len(cuts)))
    group.target_started()
    for state, _, _ in segs:
        state.kv_begin()
    initial_waits, initial_records = event.synchronize_calls, event.record_calls
    with CutCopies(
        cuts,
        gpu_budget_bytes=len(cuts) * LAYERS * 6 * 128 * 128 * 4,
        host_budget_bytes=data["observer_host_budget_bytes"],
    ) as copies:
        for layer, g in enumerate(layers):
            for state, start, end in segs:
                group.submit(
                    state,
                    layer,
                    projections[layer, start:end],
                    g.conv,
                    g.dt_bias,
                    g.norm,
                    1e-6,
                    output[start:end],
                    sums[start:end],
                    a_log=g.a_log,
                    a_neg=g.a_neg,
                    at=start,
                )
                slot = layer * prepared.group_segments + group.indices[state]
                copy_frontend(prepared, prepared.group_journal[slot], captures, layer, start, end)
                assert event.synchronize_calls == initial_waits and event.record_calls == initial_records
                assert all(s._kv_pending for s in states)
            # Keep every layer output for its own CPU diagnostic, not the last
            # layer's reused output. These extra D2D copies invalidate P3.
            if layer == 0:
                first_output, first_sums = output.clone(), sums.clone()
        group.seal()
        assert event.record_calls == initial_records + 1 and event.synchronize_calls == initial_waits
        with pytest.raises(RuntimeError, match="whole-frame"):
            group.wait_native_reuse()
        forward._kv_check(segs, comm=None)  # Original real codec/status readback.
        assert all(not state._kv_pending for state in states)
        group.mark_validated()
        group.wait_native_reuse()
    assert event.synchronize_calls == initial_waits + 1
    assert group.retired and prepared.active is None and all(value is None for value in prepared.group_journal)
    assert all(not state.history_native_readers and state.pos == 0 for state in states)
    copies.assert_bits()
    retained = [(cut.rec, cut.conv, cut.rec.cpu(), cut.conv.cpu()) for cut in cuts]
    baseline = incumbent(states, before, projections, layers, lengths, prepared.family, prepared.frontend.device)
    measurements = []
    for layer, g in enumerate(layers):
        q, k, v, gate, beta, raw = captures[layer]
        observed_output, observed_sums = (first_output, first_sums) if layer == 0 else (output, sums)
        for state, start, end in segs:
            expected, expected_state = reference(
                q[:, start:end],
                k[:, start:end],
                v[:, start:end],
                gate[:, start:end],
                beta[:, start:end],
                before[state][0][layer : layer + 1],
            )
            expected_out, expected_sums = half_epilogue_reference(
                expected.half(), projections[layer, start:end], g.norm, 1e-6
            )
            old_out, old_sums, old_state = baseline[state, layer]
            metric = {
                "serial_raw": boundary_errors(raw[:, start:end], expected),
                "serial_output": boundary_errors(observed_output[start:end], expected_out),
                "serial_sums": boundary_errors(observed_sums[start:end], expected_sums),
                "serial_state": boundary_errors(state.rec[state.cur[layer], layer : layer + 1], expected_state),
                "incumbent_output": boundary_errors(observed_output[start:end], old_out),
                "incumbent_sums": boundary_errors(observed_sums[start:end], old_sums),
                "incumbent_state": boundary_errors(state.rec[state.cur[layer], layer : layer + 1], old_state),
            }
            for item in metric.values():
                assert_finite(item)
            measurements.append({"layer": layer, "slot": state.history_slot, "metrics": metric})
            points = [cut for cut in cuts if cut.at == start]
            assert state.cur[layer] == (before[state][2][layer] + len(points) + 1) % 2
            expected_tail = torch.cat((before[state][1][layer], projections[layer, start:end, :1280].cpu()), dim=0)[-3:]
            assert same_storage_bits(state.conv[layer], expected_tail)
            for cut in points:
                tail = torch.cat(
                    (before[state][1][layer], projections[layer, start : start + cut.row, :1280].cpu()), dim=0
                )[-3:]
                assert same_storage_bits(cut.conv[layer], tail)
    assert all(same_storage_bits(rec, rr) and same_storage_bits(conv, cc) for rec, conv, rr, cc in retained)
    records.append(
        {
            "shape": name,
            "codec": codec,
            "iteration": iteration,
            "lengths": list(lengths),
            "layers": LAYERS,
            "actual_cut_copies": len(copies.sources),
            "cuts": len(cuts),
            "pending_preserved": True,
            "observer_clone_bytes": copies.logical_clone_bytes,
            "observer_python_header_bound": copies.python_header_bound,
            "one_piece_event": True,
            "no_segment_or_layer_event_wait": True,
            "cut_transfer_bits": True,
            "compiled_sets": 1,
            "workspace_slabs": 1,
            "measurements": measurements,
            "scope": "Native component/direct group only; no policy/world8/model/P3 acceptance",
            "observer": "Real TorchDispatch copy forwarded exactly once; source cloned after copy on same stream; all clones held through positive completion",
        }
    )


@torch.inference_mode()
def test_source41_real_packed_group_status_handoff_refusals_and_diagnostics(record_property):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.fail("this native gate requires the separately authorized actual SM70 GPU")
    data = load_input()
    selected_compiler = compiler(data)
    with retained_jit_files(selected_compiler):
        assert max(payload_bound(len(lengths))["total"] for lengths in SHAPES.values()) <= data["fixture_budget_bytes"]
        device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            family = prepare_family(device)
            prepared = adapter.prepare(
                CAPACITY, device, workspace_budget_bytes=4936 * CAPACITY + 21, compiler=selected_compiler, family=family
            )
            prepared.configure_groups(LAYERS, 32, host_budget_bytes=data["journal_host_budget_bytes"])
            event = observed_event(prepared)
        # Use the real existing handoff implementation on a genuinely compiled
        # component. This is explicitly NOT bind_weights/adopt/world8 authority.
        request = policy.Request.parse(
            {
                "policy": policy.POLICY,
                "tensor_budget_bytes": 4936 * CAPACITY + 21,
                "journal_host_budget_bytes": data["journal_host_budget_bytes"],
                "cold_host_budget_bytes": 1,
                "warmup_device_budget_bytes": 5752 * CAPACITY + 804656,
                "cuda_home": data["cuda_home"],
                "nvcc_sha256": data["nvcc_sha256"],
                "artifact": data["artifact"],
            }
        )
        capability = policy.Owner(request, {"scope": "component handoff only"})
        capability.prepared = prepared
        records = []

        def actor():
            torch.cuda.set_device(device)
            with torch.inference_mode(), torch.cuda.stream(stream):
                capability.handoff_to_current_thread()
                assert prepared.core.owner is threading.current_thread()
                for codec in ("int8", ISO_NAME):
                    for name, lengths in SHAPES.items():
                        generator = torch.Generator(device=device).manual_seed(20261113 + len(lengths))
                        states = [make_state(device, codec, index, generator) for index in range(len(lengths))]
                        layers = source_layers(device, generator)
                        projections = (
                            torch.randn(LAYERS, CAPACITY, 2060, device=device, dtype=torch.float16, generator=generator)
                            * 0.125
                        )
                        for iteration in range(2):
                            chosen = states if iteration == 0 else list(reversed(states))
                            sizes = lengths if iteration == 0 else tuple(reversed(lengths))
                            one_frame(
                                prepared,
                                chosen,
                                sizes,
                                layers,
                                projections,
                                event,
                                iteration,
                                codec,
                                name,
                                records,
                                data,
                            )
                        # one_frame completed the real event/status retirement and
                        # every CPU diagnostic transfer. Drop all cohort borrowers
                        # BEFORE the next allocation RHS, avoiding two paid cohorts.
                        del chosen, states, layers, projections
                generator = torch.Generator(device=device).manual_seed(20261114)
                states = [make_state(device, ISO_NAME, index, generator) for index in range(2)]
                layers = source_layers(device, generator)
                projections = (
                    torch.randn(LAYERS, CAPACITY, 2060, device=device, dtype=torch.float16, generator=generator) * 0.125
                )
                # Real epoch transition retires an unsubmitted group, making its
                # old frame unusable; no counter is manually reset backwards.
                segs, cuts = pack(states, (2, 3))
                stale = prepared.begin_group(segs, cuts, cut_budget_bytes=adapter.cut_bytes(LAYERS, len(cuts)))
                states[0].begin_sequence_generation()
                with pytest.raises(RuntimeError, match="not live"):
                    stale.authorize(states[0], 0, 0, 2, stale.cuts[0])
                segs, cuts = pack(states, (2, 3))
                with pytest.raises(MemoryError):
                    prepared.begin_group(segs, cuts, cut_budget_bytes=0)
                cuts[0].rec, cuts[0].conv = states[1].rec[0], states[1].conv
                with pytest.raises(ValueError, match="disjoint"):
                    prepared.begin_group(segs, cuts, cut_budget_bytes=adapter.cut_bytes(LAYERS, len(cuts)))
                assert prepared.active is None
                # Controlled real codec status failure; the original checker owns
                # its latch/pending transition, not the diagnostic observer.
                segs, cuts = pack(states, (2, 3))
                failed = prepared.begin_group(segs, cuts, cut_budget_bytes=adapter.cut_bytes(LAYERS, len(cuts)))
                failed.target_started()
                for state, _, _ in segs:
                    state.kv_begin()
                waits = event.synchronize_calls
                for layer, g in enumerate(layers):
                    for state, start, end in segs:
                        failed.submit(
                            state,
                            layer,
                            projections[layer, start:end],
                            g.conv,
                            g.dt_bias,
                            g.norm,
                            1e-6,
                            torch.empty(end - start, 768, device=device, dtype=torch.float16),
                            torch.empty(end - start, 24, device=device),
                            a_log=g.a_log,
                            a_neg=g.a_neg,
                            at=start,
                        )
                failed.seal()
                states[0].kv_status[0].fill_(1)
                try:
                    forward._kv_check(segs, comm=None)
                except KVNumericError as primary:
                    failed.fail(primary)
                    assert event.synchronize_calls == waits
                    assert all(state._history_error is state._kv_error is primary for state in states)
                    assert prepared.active is failed and not failed.validated and not failed.retired
                    with pytest.raises(KVNumericError) as caught:
                        failed.wait_native_reuse()
                    assert caught.value is primary and prepared.active is failed
                else:
                    pytest.fail("actual codec checker ignored the controlled numeric status")

        task = Task(actor, name="g3-native-group-actor")
        primary = None
        try:
            task.start()
        except BaseException as error:  # noqa: BLE001 - drain the owned task before propagating startup failure.
            primary = error
        drain([task], primary=primary)
        assert len(records) == 12
        record_property("g3_actual_compiler_VMA_FDs", json.dumps(observe_compiler(prepared)))
        record_property(
            "g3_packed_native",
            json.dumps(
                {
                    "cases": records,
                    "fixture_bound": payload_bound(32),
                    "image": data["image_id"],
                    "source_uid": data["source_uid"],
                    "native_half_elf": str(Path(prepared.frontend.module.__file__)),
                    "native_half_sha256": hashlib.sha256(
                        Path(prepared.frontend.module.__file__).read_bytes()
                    ).hexdigest(),
                    "native_incumbent_elf": str(Path(gdn_native.extension().__file__)),
                    "native_incumbent_sha256": hashlib.sha256(
                        Path(gdn_native.extension().__file__).read_bytes()
                    ).hexdigest(),
                    "quality_acceptance": False,
                    "performance_qualified": False,
                    "actual_policy_adoption": "NOT EXERCISED; separate realTP8 HTTP/owned-launcher gate required",
                }
            ),
        )
