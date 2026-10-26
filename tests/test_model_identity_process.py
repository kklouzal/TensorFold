"""Actual cold-fork receipt rejection; no warm fork or accelerator execution."""

import multiprocessing
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

import pytest

from tensorfold.engine import model_identity


def _inherited_receipt_child(receipt, files):
    options = dict(runtime_identity={'loader': 'pid-control'},
                   authorize=lambda _name, path: path.resolve(strict=True),
                   max_file_bytes=1024, max_total_bytes=1024,
                   reuse=receipt, reuse_is_immutable=True)
    try:
        with patch.object(model_identity, 'canonical_runtime_json', side_effect=AssertionError('runtime before PID')), \
                patch.object(model_identity, 'FileStreams', side_effect=AssertionError('files before PID')):
            for operation in (
                receipt.verify_unchanged,
                lambda: model_identity.capture_model_identity(files, **options),
                lambda: model_identity.capture_shared_model_identity(files, rank=0, world=1,
                    exchange=lambda _stage, operation: [operation()], **options),
            ):
                try:
                    operation()
                except ValueError as error:
                    if 'not issued in this process' not in str(error):
                        os._exit(2)
                else:
                    os._exit(1)
    except BaseException:
        os._exit(2)
    os._exit(0)


def _cold_fork_control():
    from test_sm70_software_binding_runtime import _exercise_fork_control

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / 'model-input'
        path.write_bytes(b'owned immutable model input')
        files = {'input': path}
        receipt = model_identity.capture_model_identity(files, runtime_identity={'loader': 'pid-control'},
            authorize=lambda _name, value: value.resolve(strict=True), max_file_bytes=1024, max_total_bytes=1024)
        process = multiprocessing.get_context('fork').Process(target=_inherited_receipt_child, args=(receipt, files))
        if len(list(Path('/proc/self/task').iterdir())) != 1:
            raise RuntimeError('real model-identity fork control requires exactly one native thread')
        _exercise_fork_control(process)
        print('native_threads=1; inherited model receipt refused before runtime/FD work; child retired')


@pytest.mark.skipif(sys.platform != 'linux', reason='requires the existing owned Linux session/fork harness')
def test_inherited_model_identity_is_refused_in_an_owned_cold_fork():
    from tensorfold.cuda.expert_launch import supervise

    environment = dict(os.environ)
    environment.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       BLIS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    deadline = time.monotonic() + 30
    status = supervise([[sys.executable, str(Path(__file__).resolve()), '--cold-fork']], [environment],
                       grace=0, failure_probe=lambda: time.monotonic() >= deadline)
    assert status == 0


if __name__ == '__main__':
    _cold_fork_control()
