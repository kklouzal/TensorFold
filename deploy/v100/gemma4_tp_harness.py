"""Gemma 4 on eight V100s for Harness: tools/launch_cuda_tp.py with every rank started through the Harness
compatibility entrypoint (deploy/gb10/tensorfold_harness.py), so /v1/tokenize and the Harness model metadata are
served. Arguments are launch_cuda_tp.py's: ``--gpus 0,...,7 -- MODEL serve-args...``."""
import os
import sys

SOURCE = os.environ.get("TENSORFOLD_SOURCE", "/src")
sys.path.insert(0, os.path.join(SOURCE, "tools"))
import launch_cuda_tp  # noqa: E402

raise SystemExit(launch_cuda_tp.main(sys.argv[1:], _command_prefix=[
    sys.executable, os.path.join(SOURCE, "deploy", "gb10", "tensorfold_harness.py"), "serve"]))
