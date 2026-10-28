"""Exact existing standalone numerical controls, one isolated process per phase."""
import argparse
import hashlib
import json
from pathlib import Path

import pytest


class Outcomes:
    def __init__(self):
        self.passed = self.skipped = 0
    def pytest_runtest_logreport(self, report):
        if report.when == 'call' and report.passed:
            self.passed += 1
        if report.skipped:
            self.skipped += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('pre', 'post'))
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from tensorfold.cuda.sm70.dp4a_experts import extension, CONTRACT
    module = extension()
    outcomes = Outcomes()
    command = ['-s', '-c', '/dev/null', '-o', 'pythonpath=/qualification/tests /qualification/tests/cuda /qualification',
               '-p', 'no:cacheprovider', '-x', '-q', '--junitxml=' + str(args.out / ('numeric-' + args.phase + '.xml')),
               '/qualification/tests/cuda/test_sm70_dp4a_profile_cuda.py']
    code = int(pytest.main(command, plugins=[outcomes]))
    result = {'exit': code, 'passed': outcomes.passed, 'skipped': outcomes.skipped,
              'complete': code == 0 and outcomes.passed == 20 and outcomes.skipped == 0,
              'module': module.__file__, 'module_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              'contract': CONTRACT, 'scope': 'original quantizer/int64/status/graph controls; uniform-coefficient fixture limits retained'}
    (args.out / ('numeric-' + args.phase + '.json')).write_text(json.dumps(result, indent=2) + '\n')
    raise SystemExit(0 if result['complete'] else code or 1)


if __name__ == '__main__':
    main()
