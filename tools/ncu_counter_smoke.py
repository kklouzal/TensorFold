"""Private finite NCU permission smoke; no Engine, NCCL, or host tuning."""
import argparse
import csv
import io
import math
import hashlib
import json
from pathlib import Path
import subprocess
import sys

NCU = "/opt/nvidia/nsight-compute/2025.2.1/ncu"
NCU_SHA = "a44ff2c735c4dcf66c78f3f430155805d6a0b1f74f26331f98ecec03bfbf3f1a"
METRICS = ("gpu__time_duration.sum", "sm__cycles_elapsed.avg", "dram__bytes_read.sum")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def command(script, out):
    return [NCU, "--config-file", "off", "--target-processes", "application-only",
            "--profile-from-start", "off", "--replay-mode", "kernel",
            "--kernel-name-base", "function", "--rename-kernels", "off",
            "--kernel-name", "regex:.*elementwise_kernel.*", "--launch-count", "1",
            "--launch-skip", "0", "--clock-control", "none", "--cache-control", "none",
            "--metrics", ",".join(METRICS), "--csv", "--log-file", str(out / "ncu.csv"),
            "--export", str(out / "smoke.ncu-rep"), sys.executable, str(script),
            "--run-smoke", "--out", str(out)]


def profiled_call(operation, start, stop):
    from tensorfold.cleanup import raise_failures

    secondary = [None]  # Allocate outcome storage before acquiring profiler state.
    primary, result, started = None, None, False
    try:
        start()
        started = True
        result = operation()
    except BaseException as error:
        primary = error
    if started:
        try:
            stop()
        except BaseException as error:
            secondary[0] = error
    if primary is not None:
        raise_failures(primary, secondary)
    if secondary[0] is not None:
        raise secondary[0]
    return result


def maps():
    return Path("/proc/self/maps").read_text()


def run_smoke(out):
    import numpy as np
    import torch
    import tensorfold

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        raise RuntimeError("actual SM70 is required; CPU/skip is not permission proof")
    if torch.version.git_version != "04cf2557f315a350e7eea7ab064288def65db75a":
        raise RuntimeError("pinned source SDK required")
    if "site-packages" not in Path(tensorfold.__file__).parts:
        raise RuntimeError("normally installed owned wheel required")
    torch.set_num_threads(1)
    x = torch.arange(4096, dtype=torch.float32, device="cuda")
    y = x * 2
    output = torch.empty_like(x)
    torch.add(x, y, out=output)
    torch.cuda.synchronize()
    expected = np.arange(4096, dtype=np.float32) * np.float32(3)
    before = output.cpu().numpy().copy()
    np.testing.assert_array_equal(before, expected)
    (out / "maps-before.txt").write_text(maps())
    def operation():
        torch.add(x, y, out=output)
        torch.cuda.synchronize()
    profiled_call(operation, torch.cuda.profiler.start, torch.cuda.profiler.stop)
    after = output.cpu().numpy().copy()
    np.testing.assert_array_equal(after, expected)
    np.testing.assert_array_equal(after.view(np.uint32), before.view(np.uint32))
    (out / "maps-after.txt").write_text(maps())
    record = {"status": "PASS stock-Torch stateless vector-add replay output oracle",
              "torch": torch.__version__, "torch_git": torch.version.git_version,
              "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(),
              "capability": torch.cuda.get_device_capability(), "elements": 4096,
              "dtype": "float32", "expected_sha256": hashlib.sha256(expected.tobytes()).hexdigest(),
              "before_sha256": hashlib.sha256(before.tobytes()).hexdigest(),
              "after_sha256": hashlib.sha256(after.tobytes()).hexdigest(),
              "scope": "counter permission/component only; no TensorFold model or throughput qualification"}
    (out / "application.json").write_text(json.dumps(record, indent=2) + "\n")



def parse_counters(text):
    header, values = None, {}
    for row in csv.reader(io.StringIO(text)):
        if "Metric Name" in row and "Metric Value" in row:
            header = row
            continue
        if header is None or len(row) != len(header):
            continue
        name = row[header.index("Metric Name")]
        if name not in METRICS:
            continue
        value = float(row[header.index("Metric Value")].replace(",", ""))
        if not math.isfinite(value) or value < 0:
            raise ValueError("invalid NCU hardware counter")
        values[name] = value
    return values

def launch(out):
    if out.exists():
        raise RuntimeError("fresh evidence destination required; preserve prior runs")
    out.mkdir(parents=True)
    if sha(NCU) != NCU_SHA:
        raise RuntimeError("immutable NCU2025.2.1 ELF differs")
    for name, args in (("version", ["--version"]), ("help", ["--help"])):
        with (out / (name + ".txt")).open("wb") as stream:
            subprocess.run([NCU, *args], stdout=stream, stderr=subprocess.STDOUT, check=True)
    argv = command(Path(__file__).resolve(), out)
    (out / "command.json").write_text(json.dumps({"argv": argv, "ncu_sha256": NCU_SHA,
        "script_sha256": sha(__file__), "mode": "stateless kernel replay", "host_tuning": "none"}, indent=2) + "\n")
    with (out / "stdout.txt").open("wb") as stdout, (out / "stderr.txt").open("wb") as stderr:
        result = subprocess.run(argv, stdout=stdout, stderr=stderr)
    text = (out / "ncu.csv").read_text(errors="replace") if (out / "ncu.csv").exists() else ""
    counters = parse_counters(text)
    available = [metric for metric in METRICS if metric in counters]
    passed = (result.returncode == 0 and len(available) == len(METRICS)
              and counters.get("sm__cycles_elapsed.avg", 0) > 0
              and counters.get("gpu__time_duration.sum", 0) > 0
              and (out / "smoke.ncu-rep").is_file() and (out / "application.json").is_file()
              and "ERR_NVGPUCTRPERM" not in text and "==ERROR==" not in text)
    record = {"status": "PASS" if passed else "FAILED or counter permission unsupported",
              "exit_code": result.returncode, "requested_metrics": METRICS,
              "metrics_present": available, "counters": counters, "ncu_elf_sha256": NCU_SHA,
              "scope": "actual permission/replay smoke only; inspect raw values; no model/P3 claim"}
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    if not passed:
        raise RuntimeError("NCU counter smoke did not produce all required counters and output oracle")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-smoke", action="store_true")
    args = parser.parse_args()
    if args.run_smoke:
        run_smoke(args.out)
    else:
        launch(args.out)


if __name__ == "__main__":
    main()
