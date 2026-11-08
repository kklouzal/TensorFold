"""Private supervised SC117 owned-routed qualification over the ordinary CLI/API.

Parent: --gpus 0,1,2,3,4,5,6,7 --out NEW_DIR -- MODEL [ordinary serve options].
No public serving flag selects this policy. Runtime/device/model guards remain
in the existing constructor; this procedure records a bound candidate only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.cohort import FAILURE_FD_ENV, notify_failure

if __package__:
    from .benchmark_output import write_json
else:
    from benchmark_output import write_json



def _prompt_digest(tokens):
    """Compact JSON integer-array SHA, borrowing IDs without a joined copy."""
    digest = hashlib.sha256(b"[")
    for index, token in enumerate(tokens):
        if type(token) is not int or not 0 <= token < (1 << 31):
            raise ValueError("attestation requires admitted bounded token IDs")
        if index:
            digest.update(b",")
        digest.update(str(token).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


def _history_input(path, capacity=None, context=None):
    # This deliberately small DATA schema carries no request bodies or gold.
    # Five names/scalars plus C fixed-width SHA strings fit this ASCII bound.
    from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
    limit = MAX_MESSAGE_BYTES if capacity is None else 1024 + 256 * capacity
    with Path(path).open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("history attestation DATA exceeds its admitted-C byte bound")
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("duplicate history attestation field")
            result[key] = value
        return result
    record = json.loads(raw, object_pairs_hook=pairs)
    if (type(record) is not dict or set(record) != {"C", "N", "R", "L", "prompt_token_ids_sha256"}
            or any(type(record[key]) is not int or not 0 < record[key] <= sys.maxsize for key in ("C", "N", "R", "L"))
            or capacity is not None and record["C"] != capacity
            or context is not None and record["L"] != context or record["N"] + record["R"] != record["L"]):
        raise ValueError("history attestation requires actual C/N/R/L geometry")
    capacity = record["C"]
    limit = 1024 + 256 * capacity
    if len(raw) > limit:
        raise ValueError("history attestation DATA exceeds its admitted-C byte bound")
    hashes = record["prompt_token_ids_sha256"]
    if (type(hashes) is not list or len(hashes) != capacity or any(type(value) is not str
            or len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes)
            or len(set(hashes)) != capacity):
        raise ValueError("history attestation requires C distinct lowercase token-ID SHA256 values")
    return record, hashlib.sha256(raw).hexdigest(), limit


def _history_resources(capacity, input_limit):
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        _json_payload_policy, _json_parse_payload, _json_tree_payload,
        _python_layout, _streamed_json_encoding_payload,
    )
    policy = _json_payload_policy()
    # Fixed identity/header/resource names plus C rows of <=10 bounded scalars
    # and one 64-byte SHA; ASCII indentation is included in these source bounds.
    receipt_limit = 4096 + 1024 * capacity
    # The extra input tree also pays the five fixed object_pairs_hook pairs
    # and partial result dict; the C hash list is borrowed, not copied by the hook.
    per_rank = (_json_parse_payload(input_limit, policy)["total_bytes"]
                + _json_tree_payload(input_limit, policy)
                + _json_tree_payload(receipt_limit, policy)
                + _streamed_json_encoding_payload(receipt_limit, _python_layout())["total_bytes"]
                + policy.frame_work)
    total = 8 * per_rank
    if total > sys.maxsize:
        raise ValueError("history qualification host obligation exceeds the host size domain")
    return {"global_host_obligation_bytes": total, "per_rank_payload_bytes": per_rank,
            "input_bytes_limit": input_limit, "receipt_bytes_limit": receipt_limit,
            "materialized_credit_bytes": 0, "runtime_policy": policy.identity,
            "scope": "source/object payload; native allocator/RSS separate; probe excluded from performance"}


def prepare_history(engine, prepared, output):
    data, input_sha, resources = prepared
    if (len(engine.multi.states) != data["C"] or engine.context_window != data["L"]
            or engine._owned_qualification_host != (resources["global_host_obligation_bytes"], input_sha)):
        raise ValueError("history qualification differs from actual C/context/paid obligation")
    catalog = engine.retention_catalog
    if catalog is None or catalog.host_resources is None:
        raise ValueError("history qualification requires the original immutable FIFO host baseline")
    if catalog.host_resources.baseline_host_bytes < resources["global_host_obligation_bytes"]:
        raise ValueError("history qualification obligation is absent from the host baseline")
    observer = _HistoryAttestation(engine, data, input_sha, output, resources)
    observer.install()
    return resources


def _observer_failure(engine, primary):
    engine._abort_comm = True
    try:
        if not notify_failure():
            raise RuntimeError("history attestation lost its TP supervisor capability")
    except BaseException as cleanup:
        raise_failures(primary, [cleanup])


class _HistoryAttestation:
    """One private worker-boundary witness; never installed on normal serving."""
    def __init__(self, engine, data, input_sha, output, resources):
        self.engine, self.data, self.input_sha = engine, data, input_sha
        self.output, self.resources = Path(output), resources
        self.original = engine.multi.round
        self.installed = False

    def install(self):
        if self.installed:
            raise ValueError("history observer is already installed")
        self.engine.multi.round = self.round
        self.installed = True

    def snapshot(self):
        engine, data = self.engine, self.data
        decoder, tp = engine.multi, engine.tp_decoder
        streams = [stream for stream in decoder.streams.values() if not stream.done]
        if len(streams) != data["C"] or decoder.filling:
            return None
        if any(stream.st is None or stream.st.pos < data["N"] for stream in streams):
            return None
        if len({id(stream.st) for stream in streams}) != data["C"]:
            raise ValueError("history attestation found aliased active States")
        coordinate = engine.w.meta["coordinate_profile_identity"]
        numerical = ("owned-expert-native", engine._owned_expert_plan.identity)
        rows = []
        for stream in streams:
            state = stream.st
            if (not any(state is owned for owned in decoder.states) or getattr(state, "_history_error", None) is not None
                    or state.coordinate_profile_identity != coordinate
                    or numerical not in state.kv_identity
                    or ("coordinate-profile-v1", coordinate) not in state.kv_identity
                    or len(stream.prompt) != data["N"] or stream.count != data["R"]):
                raise ValueError("history attestation owner differs from its prepared identity/geometry")
            rows.append({"wire": tp.wires[id(stream)], "slot": state.history_slot,
                         "slot_generation": state.slot_generation, "ownership_generation": state.history_generation,
                         "cached_length": state.pos, "prompt_tokens": len(stream.prompt),
                         "prompt_token_ids_sha256": _prompt_digest(stream.prompt)})
        if ({row["prompt_token_ids_sha256"] for row in rows} != set(data["prompt_token_ids_sha256"])
                or len({row["wire"] for row in rows}) != data["C"]
                or len({row["slot"] for row in rows}) != data["C"]
                or any(type(row[key]) is not int or row[key] < 0 for row in rows
                       for key in ("wire", "slot", "slot_generation", "ownership_generation", "cached_length"))
                or type(tp.prefix_operation) is not int or tp.prefix_operation < 0):
            raise ValueError("history attestation corpus, wire, slot or generation mismatch")
        return {"schema": "tensorfold.private-owned-history-v1", "rank": engine.rank, "world": engine.tp,
                "prefix_sequence": tp.prefix_operation, "input_sha256": self.input_sha,
                "C": data["C"], "N": data["N"], "R": data["R"], "L": data["L"],
                "coordinate_profile_identity": coordinate, "plan_identity": list(engine._owned_expert_plan.identity),
                "owners": sorted(rows, key=lambda row: row["wire"]), "resources": self.resources,
                "scope": "coherent pre-round logical owner/device-stream order; not wall-clock GPU completion",
                "qualification": "requires all-eight common facts, successful TP ACKs and bound client outputs",
                "run_identity": engine._history_run_identity, "server_instance_id": engine._history_server_instance}

    def round(self):
        try:
            witness = self.snapshot()
        except BaseException as primary:
            _observer_failure(self.engine, primary)
            raise
        # A model failure retains its exact existing propagation and never publishes.
        result = self.original()
        if witness is not None:
            try:
                target = self.output / f"history-rank-{self.engine.rank}.json"
                if target.exists() or target.is_symlink():
                    raise ValueError("history attestation receipt already exists")
                encoded = sum(len(chunk) for chunk in json.JSONEncoder(indent=1, allow_nan=False).iterencode(witness)) + 1
                if encoded > self.resources["receipt_bytes_limit"]:
                    raise ValueError("history receipt exceeds its prepaid output bound")
                write_json(target, witness)
                self.engine.multi.round = self.original  # Restore before any subsequent owner operation.
            except BaseException as primary:
                _observer_failure(self.engine, primary)
                raise
        return result


def bound_receipt(engine):
    import torch
    from tensorfold.families.qwen4_exp.cuda.owned_experts import Plan, POLICY
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts
    from tensorfold.families.qwen4_exp.cuda.position_coordinates import _digest

    w, plan, decoder = engine.w, engine._owned_expert_plan, engine.multi
    if (type(plan) is not Plan or w.meta.get("owned_expert_plan") is not plan
            or decoder is None or decoder.w is not w or engine.tp != 8
            or plan.partition.rank != engine.rank or plan.partition != engine.expert_partition
            or plan.hidden != 2560 or plan.slots != 11 or plan.partition.experts != 512
            or plan.partition.full_width != 640 or w.cfg.layers != 48 or len(w.layers) != 48):
        raise ValueError("qualification requires the actual bound SC117 TP8/ETP4 plan")
    if w.device.type != "cuda" or torch.cuda.get_device_capability(w.device) != (7, 0):
        raise ValueError("qualification requires actual SM70 CUDA weights")
    banks = [layer.moe.experts for layer in w.layers]
    if w.mtp is not None:
        banks.append(w.mtp.layer.moe.experts)
    if (bool(w.mtp is not None) is not plan.mtp or any(
            type(bank) is not GGUFExperts or bank.implementation != POLICY
            or bank._owned_expert_plan is not plan or bank.partition != plan.partition
            or bank.dims != plan.hidden for bank in banks)):
        raise ValueError("actual target/MTP banks differ from the exclusive native policy")
    resources = plan.reconcile(decoder)  # Post-warm boundary verification, not a second admission/allocation.
    if (resources != decoder.owned_expert_resources
            or resources != engine.capacity_plan.get("owned_expert_native")
            or engine.capacity_plan["expert_parallel"]["implementation"] != POLICY):
        raise ValueError("actual native arenas differ from their admitted/materialized receipt")
    numerical = ("owned-expert-native", plan.identity)
    coordinate = w.meta.get("coordinate_profile_identity")
    if not _digest(coordinate):
        raise ValueError("qualification requires the actual bound coordinate/cohort identity")
    coordinate_marker = ("coordinate-profile-v1", coordinate)
    if not decoder.states or any(numerical not in state.kv_identity or
            coordinate_marker not in state.kv_identity or state.coordinate_profile_identity != coordinate
            for state in decoder.states):
        raise ValueError("actual State numerical identity differs from the native plan")
    return {"schema": "tensorfold.private-owned-qualification-v1", "rank": engine.rank, "world": engine.tp,
            "policy": POLICY, "plan_identity": list(plan.identity), "expert_partition": plan.partition.identity(),
            "target_banks": len(w.layers), "mtp_banks": int(plan.mtp), "resources": resources,
            "coordinate_profile_identity": coordinate, "state_count": len(decoder.states),
            "device": str(w.device), "capability": [7, 0],
            "qualification": "private candidate binding only; no full-native, quality or performance claim"}


def publish_receipt(engine, output):
    try:
        from tensorfold import cli
        receipt = bound_receipt(engine)
        obligation = getattr(engine, "_owned_qualification_host", ())
        if obligation:
            receipt["history_host_obligation"] = {"global_bytes": obligation[0], "input_sha256": obligation[1]}
            receipt["run_identity"] = engine._history_run_identity
            receipt["server_instance_id"] = engine._history_server_instance
        receipt["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        receipt["runtime_cli_sha256"] = hashlib.sha256(Path(cli.__file__).read_bytes()).hexdigest()
        target = Path(output) / f"rank-{engine.rank}.json"
        if target.exists() or target.is_symlink():
            raise ValueError("qualification rank receipt already exists")
        write_json(target, receipt)  # Existing atomic file+directory durability owner.
        print(f"[tensorfold] private owned qualification rank={engine.rank}/8 bound; receipt={target.name}", flush=True)
        engine.comm.ready("owned_expert_qualification")  # All eight receipts precede follower/listener publication.
    except BaseException as primary:
        engine._abort_comm = True
        try:
            if not notify_failure():
                raise RuntimeError("qualification lost its TP supervisor capability")
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
        raise


def worker(argv):
    # Validate the borrowed supervisor capability before importing CLI/Torch.
    import fcntl

    value = os.environ.get(FAILURE_FD_ENV, "")
    if not value.isdecimal():
        raise ValueError("private worker requires the supervised failure pipe")
    descriptor = int(value)
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    if (not stat.S_ISFIFO(os.fstat(descriptor).st_mode)
            or flags & os.O_ACCMODE != os.O_WRONLY or not flags & os.O_NONBLOCK):
        raise ValueError("private worker requires the borrowed nonblocking supervisor writer")
    parser = argparse.ArgumentParser(description="Internal supervised worker")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-input", type=Path)
    parser.add_argument("--history-run-id")
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    selected = parser.parse_args(argv)
    tail = selected.serve_args[1:] if selected.serve_args[:1] == ["--"] else selected.serve_args
    receipt_failed = False
    prepared = None

    def before_listener(engine):
        nonlocal receipt_failed
        try:
            if prepared is not None:
                try:
                    from tensorfold.server.stats import INSTANCE_ID
                    engine._history_run_identity = selected.history_run_id
                    engine._history_server_instance = INSTANCE_ID
                    prepare_history(engine, prepared, selected.out)
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
            publish_receipt(engine, selected.out)
        except BaseException:
            receipt_failed = True  # publish_receipt already poisoned/notified before CLI retirement.
            raise
    try:
        if selected.history_input is not None:
            if (type(selected.history_run_id) is not str or len(selected.history_run_id) != 32
                    or any(char not in "0123456789abcdef" for char in selected.history_run_id)):
                raise ValueError("history worker requires its fresh parent execution identity")
            data, input_sha, input_limit = _history_input(selected.history_input)
            prepared = (data, input_sha, _history_resources(data["C"], input_limit))
        from tensorfold import cli
        args = cli.build_parser().parse_args(["serve", *tail])
        if args.tp != 8 or args.expert_tensor_parallel_size != 4 or not 0 <= args.rank < 8:
            raise ValueError("private qualification requires explicit TP8/ETP4 rank ownership")
        obligation = () if prepared is None else (prepared[2]["global_host_obligation_bytes"], prepared[1])
        args._owned_expert_qualification = cli._OwnedExpertQualification(before_listener, obligation)
        return cli.cmd_serve(args)
    except BaseException as primary:
        try:
            if not receipt_failed and not notify_failure():
                raise RuntimeError("qualification lost its TP supervisor capability")
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
        raise



def accept_history(output, client_report):
    """Offline intersection; successful completed replies imply TP finish ACKs."""
    if __package__:
        from . import v100_adaptive_exercise as exercise
    else:
        import v100_adaptive_exercise as exercise
    from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
    def read(path, limit):
        with Path(path).open("rb") as source:
            raw = source.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("qualification artifact exceeds its bounded input envelope")
        return json.loads(raw)
    client = read(client_report, MAX_MESSAGE_BYTES)
    result = client["supplied_long"]
    capacity = result["resident_capacity"]
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("client has no positive resident capacity")
    initial, final = client["initial_stats"], client["final_stats"]
    instance = initial["server"]["instance_id"]
    if (not client.get("complete") or client.get("error") or instance != final["server"]["instance_id"]
            or not result.get("semantic_parity_passed") or not result.get("sse_nonstream_parity_passed")
            or result.get("drafted_total", 0) <= 0):
        raise ValueError("history acceptance requires completed same-service semantic/parity/MTP evidence")
    exercise.qualification.require_idle(final)
    concurrent = result["concurrent"]["results"]
    if len(result["cases"]) != capacity or len(concurrent) != capacity:
        raise ValueError("history acceptance requires exactly C client histories")
    names = set(result["cases"])
    if {row["request"] for row in concurrent} != names:
        raise ValueError("concurrent replies do not identify each supplied history")
    hashes, drafted_total = set(), 0
    for name, pair in result["cases"].items():
        hashes.add(pair["prompt_token_ids_sha256"])
        expected = result["oracles"][name]["expected"]
        replies = [pair["serial"], pair["mtp"], next(row["reply"] for row in concurrent if row["request"] == name)]
        for index, reply in enumerate(replies):
            stats = reply["tensorfold"]
            if (stats.get("drafts") is not (index != 0) or any(type(stats.get(key, 0)) is not int
                    or stats.get(key, 0) < 0 for key in ("drafted", "accepted"))):
                raise ValueError("client actual draft dispatch/counters differ from requested modes")
            if index == 0 and (stats.get("drafted", 0) or stats.get("accepted", 0)):
                raise ValueError("client serial oracle performed draft work")
            if index:
                drafted_total += stats.get("drafted", 0)
            record = exercise.correctness_reply(reply, result["declared_reply_tokens"], required_finish=None)
            exercise.correctness_answer({"id": name, **expected}, record)
            if reply["usage"]["prompt_tokens"] != result["prompt_tokens"]:
                raise ValueError("client prompt geometry differs from its oracle")
            if not exercise.live.compare(replies[0], reply, result["declared_reply_tokens"])["passed"]:
                raise ValueError("client parity evidence differs")
        if (pair["serial"]["tensorfold"].get("cached") != 0
                or pair["serial"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens") != 0):
            raise ValueError("client serial oracle was not cold")
    if drafted_total <= 0 or drafted_total != result["drafted_total"]:
        raise ValueError("client MTP summary does not match positive actual proposal work")
    common, run = None, None
    for rank in range(8):
        receipt = read(Path(output) / f"history-rank-{rank}.json", 4096 + 1024 * capacity)
        startup = read(Path(output) / f"rank-{rank}.json", MAX_MESSAGE_BYTES)
        if (startup.get("schema") != "tensorfold.private-owned-qualification-v1"
                or type(startup.get("rank")) is not int or startup["rank"] != rank or startup.get("world") != 8
                or receipt["server_instance_id"] != startup["server_instance_id"]
                or receipt["schema"] != "tensorfold.private-owned-history-v1" or type(receipt["rank"]) is not int or receipt["rank"] != rank
                or receipt["world"] != 8 or receipt["C"] != capacity
                or receipt["N"] != result["prompt_tokens"] or receipt["R"] != result["declared_reply_tokens"]
                or receipt["L"] != result["context_tokens"] or receipt["N"] + receipt["R"] != receipt["L"]
                or receipt["run_identity"] != startup["run_identity"]
                or receipt["input_sha256"] != startup["history_host_obligation"]["input_sha256"]
                or receipt["coordinate_profile_identity"] != startup["coordinate_profile_identity"]
                or receipt["plan_identity"] != startup["plan_identity"]
                or rank == 0 and receipt["server_instance_id"] != instance):
            raise ValueError("worker receipt is not bound to this prepared execution/client")
        if run is None:
            run = receipt["run_identity"]
        if receipt["run_identity"] != run:
            raise ValueError("worker receipts belong to different executions")
        owners = receipt["owners"]
        if (len(owners) != capacity or len({row["wire"] for row in owners}) != capacity
                or len({row["slot"] for row in owners}) != capacity
                or {row["prompt_token_ids_sha256"] for row in owners} != hashes
                or len(hashes) != capacity or any(
                    any(type(row[key]) is not int or row[key] < 0 for key in
                        ("wire", "slot", "slot_generation", "ownership_generation", "cached_length", "prompt_tokens"))
                    or row["prompt_tokens"] != receipt["N"] or row["cached_length"] < receipt["N"] for row in owners)):
            raise ValueError("worker owners do not match the C actual client histories")
        facts = {key: receipt[key] for key in ("prefix_sequence", "input_sha256", "C", "N", "R", "L",
                 "coordinate_profile_identity", "plan_identity", "owners")}
        if common is not None and facts != common:
            raise ValueError("worker receipts do not share one logical prefix boundary")
        common = facts
    return {"schema": "tensorfold.private-owned-history-acceptance-v1", "passed": True,
            "run_identity": run, "server_instance_id": instance, "facts": common,
            "client_report_sha256": hashlib.sha256(Path(client_report).read_bytes()).hexdigest(),
            "scope": "coherent logical owner/device-stream ordering plus completed client parity; not GPU wall-clock simultaneity"}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--accept-history"]:
        parser = argparse.ArgumentParser(description="Offline private history evidence intersection")
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--client-report", type=Path, required=True)
        selected = parser.parse_args(argv[1:])
        target = selected.out / "history-acceptance.json"
        if target.exists() or target.is_symlink():
            raise ValueError("history acceptance output already exists")
        write_json(target, accept_history(selected.out, selected.client_report))
        return 0
    if argv[:1] == ["--worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history-input", type=Path, help="private C/N/R/L and expected prompt-ID SHA DATA")
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    devices = [value.strip() for value in args.gpus.split(",")]
    if len(devices) != 8 or len(set(devices)) != 8 or not all(devices):
        parser.error("private qualification requires eight distinct physical devices")
    if __package__:
        from . import launch_cuda_tp
    else:
        import launch_cuda_tp
    output = args.out.absolute()
    output.mkdir(mode=0o700)  # Fresh, caller-owned output; never reuse another qualification's receipts.
    prefix = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(output)]
    if args.history_input is not None:
        prefix += ["--history-input", str(args.history_input.absolute()), "--history-run-id", uuid.uuid4().hex]
    prefix += ["--"]
    return launch_cuda_tp.main(["--gpus", args.gpus, "--master-port", str(args.master_port), *args.serve_args],
                               _command_prefix=prefix)


if __name__ == "__main__":
    raise SystemExit(main())
