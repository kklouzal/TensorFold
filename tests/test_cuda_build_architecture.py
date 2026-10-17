"""Independent processes retain their own target at Torch's real build/import boundary (no compiler/GPU)."""

import os
import subprocess
import sys
import time

import pytest

pytestmark = pytest.mark.torch

SCRIPT = r'''
from pathlib import Path
import json, sys, time
import torch
from torch.utils import cpp_extension as ext
from tensorfold.cuda import build
root, target = Path(sys.argv[1]), int(sys.argv[2])
torch.cuda.get_device_capability = lambda: (8, 9) if target == 89 else (12, 0)
build._toolkit = lambda: []
def compile(**options):
    path = Path(options['build_directory']) / (options['name'] + '.target')
    path.write_text(options['extra_cuda_cflags'][-1])
def imported(name, directory, *args):
    if target == 89:
        (root / 'first-import').write_text('ready')
        deadline = time.monotonic() + 20
        while not (root / 'second-import').exists():
            if time.monotonic() >= deadline:
                raise TimeoutError('second target did not reach its import')
            time.sleep(.01)
    else:
        (root / 'second-import').write_text('ready')
    return (Path(directory) / (name + '.target')).read_text()
ext._write_ninja_file_and_build_library = compile
ext._import_module_from_library = imported
result = build.load(name='tf_architecture_probe', sources=[str(root / 'source.cpp')],
                    build_directory=str(root / 'build'), with_cuda=False, verbose=False)
assert result.endswith('code=sm_' + str(target)), result
print('ARCHITECTURE-PASS', target, flush=True)
'''


def test_two_process_imports_cannot_observe_the_other_targets_artifact(tmp_path):
    """The first rank pauses after unlocking; a second target builds before the first imports."""

    pytest.importorskip("torch.utils.cpp_extension")
    (tmp_path / "source.cpp").write_text("// controlled build-boundary fixture\n")
    (tmp_path / "build").mkdir()
    env = dict(os.environ, MAX_JOBS="1")
    processes = []
    try:
        first = subprocess.Popen([sys.executable, "-c", SCRIPT, str(tmp_path), "89"], env=env,
                                 stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(first)
        deadline = time.monotonic() + 30
        while not (tmp_path / "first-import").exists():
            if first.poll() is not None:
                stdout, stderr = first.communicate()
                pytest.fail(f"first build failed before import: {stdout}\n{stderr}")
            assert time.monotonic() < deadline, "first target did not reach the import boundary"
            time.sleep(.01)
        second = subprocess.Popen([sys.executable, "-c", SCRIPT, str(tmp_path), "120"], env=env,
                                  stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        processes.append(second)
        for process, target in ((second, "120"), (first, "89")):
            stdout, stderr = process.communicate(timeout=30)
            assert process.returncode == 0, stdout + stderr
            assert "ARCHITECTURE-PASS " + target in stdout
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=10)
            process.stdout.close()
            process.stderr.close()
