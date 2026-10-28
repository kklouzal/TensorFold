"""Private finite actual-checkpoint expert counters; never model/P3 qualification."""
import argparse
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

from tools import ncu_counter_smoke as smoke

ROWS = (1, 8, 32)
EXPERTS = 512
SECTIONS = ("SpeedOfLight", "MemoryWorkloadAnalysis", "Occupancy")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def fixture(source):
    # Reuse the owned actual-span loader, explicitly replacing its ten-expert
    # test subset. No library or installed-file mutation is involved.
    directory = source / "tests/cuda"
    sys.path[:0] = [str(directory), str(source / "tests")]
    try:
        spec = importlib.util.spec_from_file_location("actual_fixture", directory / "test_sm70_gguf_exact_actual.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        del sys.path[:2]
    module.count = EXPERTS
    return module


def command(source, out, model, wheel, runtime_source, rows):
    argv = [smoke.NCU, "--config-file", "off", "--target-processes", "application-only",
            "--profile-from-start", "off", "--replay-mode", "kernel", "--kernel-name-base", "function",
            "--rename-kernels", "off", "--kernel-name", "regex:.*(expert_gate_up|expert_down80_exact).*",
            "--metrics", ",".join(smoke.METRICS), "--launch-count", "2", "--launch-skip", "0", "--clock-control", "none", "--cache-control", "none"]
    for section in SECTIONS:
        argv += ["--section", section]
    return argv + ["--export", str(out / "experts.ncu-rep"), sys.executable, "-m", "tools.ncu_experts_actual",
                   "--run-operator", "--source", str(source), "--out", str(out), "--model", str(model),
                   "--wheel", str(wheel), "--runtime-source", runtime_source, "--rows", str(rows)]


def parse_report(text):
    header, units, records = None, {}, []
    for row in csv.reader(io.StringIO(text)):
        if "ID" in row and all(metric in row for metric in smoke.METRICS):
            header = row
            continue
        if header is None or len(row) != len(header):
            continue
        record = dict(zip(header, row))
        if not record["ID"].strip():
            units = {metric: record[metric] for metric in smoke.METRICS}
            continue
        # Existing smoke parser independently validates required finite counters.
        encoded = io.StringIO()
        writer = csv.writer(encoded)
        writer.writerows([header, ["" for _ in header], row])
        values, _ = smoke.parse_counter_table(encoded.getvalue())
        if values.get("gpu__time_duration.sum", 0) <= 0 or values.get("sm__cycles_elapsed.avg", 0) <= 0:
            raise ValueError("missing positive duration/cycles")
        records.append({"id": record["ID"], "kernel": record.get("Kernel Name", ""), "counters": values})
    if len(records) != 2 or len({r["id"] for r in records}) != 2:
        raise ValueError("exactly two distinct profiled kernels required")
    if "expert_gate_up" not in records[0]["kernel"] or "expert_down80_exact" not in records[1]["kernel"]:
        raise ValueError("expected actual gate/up then exact down kernels")
    if not all(units.values()):
        raise ValueError("raw hardware counter units required")
    return {"kernels": records, "units": units, "scope": "hardware diagnostics only; no model/P3 qualification"}


def run_operator(args):
    import numpy as np
    import torch
    import tensorfold
    if torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("actual GV100 required")
    if torch.version.git_version != "04cf2557f315a350e7eea7ab064288def65db75a":
        raise RuntimeError("pinned source SDK required")
    if "site-packages" not in Path(tensorfold.__file__).parts:
        raise RuntimeError("normal owned whole-wheel installation required")
    torch.set_num_threads(1)
    loader = fixture(args.source)
    loader.MODEL = args.model
    bank, spans = loader.make(0, 8)
    for span, packed in zip(spans, (bank.gate, bank.up, bank.down)):
        span["retained_sha256"] = hashlib.sha256(packed.data.cpu().numpy().tobytes()).hexdigest()
        span["ggml_type"] = packed.ggml_type
        span["column_offset"] = packed.column_offset
    if (bank.count, bank.width, bank.dims) != (513, 80, 2560):
        raise RuntimeError("all512 routed experts plus shared bank required")
    seed = 9201 + args.rows
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = (torch.randn(args.rows, 2560, generator=generator) * .1).half().cuda()
    # Uniform sampling without replacement within each synthetic topK10 row.
    ids_cpu = torch.stack([torch.randperm(EXPERTS, generator=generator)[:10] for _ in range(args.rows)])
    ids_cpu = torch.cat([ids_cpu, torch.full((args.rows, 1), EXPERTS)], dim=1)
    picks = ids_cpu.cuda()
    expected = bank.forward_reference(x, picks)
    output = torch.empty_like(expected)
    bank.forward_exact(x, picks, output)
    torch.cuda.synchronize()
    before = output.cpu().numpy().copy()
    reference = expected.cpu().numpy().copy()
    np.testing.assert_array_equal(before.view(np.uint32), reference.view(np.uint32))
    (args.out / "maps-before.txt").write_text(smoke.maps())
    def operation():
        # Native experts_exact fully overwrites activation scratch and every
        # routed output; shared placeholders are zero before _write_shared.
        bank.forward_exact(x, picks, output)
        torch.cuda.synchronize()
    smoke.profiled_call(operation, torch.cuda.profiler.start, torch.cuda.profiler.stop)
    after = output.cpu().numpy().copy()
    np.testing.assert_array_equal(after.view(np.uint32), reference.view(np.uint32))
    np.testing.assert_array_equal(after.view(np.uint32), before.view(np.uint32))
    mapped = smoke.maps()
    (args.out / "maps-after.txt").write_text(mapped)
    libraries = {}
    for line in mapped.splitlines():
        name = line.split()[-1]
        if not name.startswith("/") or ".so" not in Path(name).name or name in libraries:
            continue
        path = Path(name)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
        libraries[name] = {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}
    dump(args.out / "loaded-libraries.json", libraries)
    for name, array in (("input", x.cpu().numpy()), ("picks", ids_cpu.numpy()), ("reference", reference), ("before", before), ("after", after)):
        np.save(args.out / (name + ".npy"), array, allow_pickle=False)
    dump(args.out / "application.json", {"status": "PASS strict uint32 output oracle before/after kernel replay",
        "seed": seed, "rows": args.rows, "hidden": 2560, "local_width": 80, "top_k": 10,
        "resident_routed_experts": 512, "shared_slot": 512, "layer": 0, "world": 8, "rank_slice": 3,
        "routing": "synthetic uniform without replacement per row; actual checkpoint weights",
        "model": str(args.model), "weight_spans": spans, "wheel_sha256": smoke.sha(args.wheel),
        "runtime_source": args.runtime_source, "fixture_sha256": smoke.sha(args.source / "tests/cuda/test_sm70_gguf_exact_actual.py"),
        "torch": torch.__version__, "torch_git": torch.version.git_version, "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(), "tensorfold": tensorfold.__file__,
        "jit_root": __import__("os").environ.get("TORCH_EXTENSIONS_DIR"),
        "scope": "stateless expert component only; no attention/NCCL, no P3 qualification"})


def launch(args):
    if args.out.exists():
        raise RuntimeError("fresh evidence directory required")
    args.out.mkdir(parents=True)
    if smoke.sha(smoke.NCU) != smoke.NCU_SHA:
        raise RuntimeError("pinned NCU2025.2.1 ELF required")
    wheel_before = smoke.sha(args.wheel)
    for name, flags in (("ncu-version", ["--version"]), ("ncu-help", ["--help"])):
        with (args.out / (name + ".txt")).open("wb") as log:
            subprocess.run([smoke.NCU, *flags], stdout=log, stderr=subprocess.STDOUT, check=True)
    dump(args.out / "identity.json", {"ncu_sha256": smoke.NCU_SHA, "wheel_sha256": wheel_before,
         "script_sha256": smoke.sha(__file__), "runtime_source": args.runtime_source,
         "tool_source_manifest": str(args.source / "source-hold.json"),
         "sections": SECTIONS, "scope": "diagnostic only; no end-to-end speed claim"})
    for rows in ROWS:
        out = args.out / f"R{rows}"
        out.mkdir()
        argv = command(args.source, out, args.model, args.wheel, args.runtime_source, rows)
        dump(out / "command.json", argv)
        with (out / "stdout.txt").open("wb") as stdout, (out / "stderr.txt").open("wb") as stderr:
            result = subprocess.run(argv, stdout=stdout, stderr=stderr)
        dump(out / "exit.json", {"profile_exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"R{rows} profile failed; full logs retained")
        argv = smoke.import_command(out / "experts.ncu-rep", out / "counters.csv")
        dump(out / "import-command.json", argv)
        with (out / "import-stdout.txt").open("wb") as stdout, (out / "import-stderr.txt").open("wb") as stderr:
            imported = subprocess.run(argv, stdout=stdout, stderr=stderr)
        dump(out / "import-exit.json", {"exit_code": imported.returncode})
        if imported.returncode or smoke.sha(smoke.NCU) != smoke.NCU_SHA:
            raise RuntimeError("pinned raw report import failed")
        csv_path = out / "counters.csv"
        if csv_path.stat().st_size > 32 << 20:
            raise ValueError("bounded component report CSV exceeds32MiB")
        dump(out / "counters.json", parse_report(csv_path.read_text()))
    if smoke.sha(args.wheel) != wheel_before:
        raise RuntimeError("whole wheel changed during diagnostics")
    dump(args.out / "result.json", {"status": "PASS", "rows": ROWS, "scope": "hardware counters only; no P3/model qualification"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--runtime-source", required=True)
    parser.add_argument("--run-operator", action="store_true")
    parser.add_argument("--rows", type=int, choices=ROWS)
    args = parser.parse_args()
    if args.run_operator and args.rows is None:
        parser.error("--run-operator requires --rows")
    (run_operator if args.run_operator else launch)(args)


if __name__ == "__main__":
    main()
