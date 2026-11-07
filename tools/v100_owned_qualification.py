"""Private supervised SC117 owned-routed qualification over the ordinary CLI/API.

Parent: --gpus 0,1,2,3,4,5,6,7 --out NEW_DIR -- MODEL [ordinary serve options].
No public serving flag selects this policy. Runtime/device/model guards remain
in the existing constructor; this procedure records a bound candidate only.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import stat
import sys

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.cohort import FAILURE_FD_ENV, notify_failure

if __package__:
    from .benchmark_output import write_json
else:
    from benchmark_output import write_json


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
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    selected = parser.parse_args(argv)
    tail = selected.serve_args[1:] if selected.serve_args[:1] == ["--"] else selected.serve_args
    receipt_failed = False

    def before_listener(engine):
        nonlocal receipt_failed
        try:
            publish_receipt(engine, selected.out)
        except BaseException:
            receipt_failed = True  # publish_receipt already poisoned/notified before CLI retirement.
            raise
    try:
        from tensorfold import cli
        args = cli.build_parser().parse_args(["serve", *tail])
        if args.tp != 8 or args.expert_tensor_parallel_size != 4 or not 0 <= args.rank < 8:
            raise ValueError("private qualification requires explicit TP8/ETP4 rank ownership")
        args._owned_expert_qualification = cli._OwnedExpertQualification(before_listener)
        return cli.cmd_serve(args)
    except BaseException as primary:
        try:
            if not receipt_failed and not notify_failure():
                raise RuntimeError("qualification lost its TP supervisor capability")
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("--out", type=Path, required=True)
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
    prefix = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(output), "--"]
    return launch_cuda_tp.main(["--gpus", args.gpus, "--master-port", str(args.master_port), *args.serve_args],
                               _command_prefix=prefix)


if __name__ == "__main__":
    raise SystemExit(main())
