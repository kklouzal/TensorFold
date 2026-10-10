"""Verification publication must allow an arbitrary unprivileged test UID."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shlex
import stat
import subprocess
import tempfile
import unittest


DOCKERFILE = Path(__file__).resolve().parents[1] / 'deploy/gb10/Dockerfile'


def instructions():
    result, pending = [], ''
    for raw in DOCKERFILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        pending += line[:-1].rstrip() + ' ' if line.endswith('\\') else line
        if not line.endswith('\\'):
            result.append(pending)
            pending = ''
    if pending:
        raise ValueError('unterminated Docker instruction')
    return result


class VerificationPermissions(unittest.TestCase):
    def publication_instruction(self):
        rows = instructions()
        start = rows.index('FROM application AS verification')
        end = next(i for i in range(start + 1, len(rows)) if rows[i].startswith('FROM '))
        stage = rows[start + 1:end]
        publication = [i for i, line in enumerate(stage) if line.startswith('RUN chmod ')]
        self.assertEqual(len(publication), 1)
        index = publication[0]
        operation = shlex.split(stage[index])[1:]
        self.assertEqual(operation, ['chmod', '-R', 'a+rX', '/opt/TensorFold'])
        return rows, stage, index, operation

    def test_publication_covers_all_verification_copies_and_inherited_sources(self):
        rows, stage, index, _ = self.publication_instruction()
        copies = [i for i, line in enumerate(stage) if line.startswith('COPY ')]
        self.assertTrue(copies)
        self.assertGreater(index, max(copies))
        for name in ('tests', 'deploy', 'tools', 'CHANGELOG.md'):
            self.assertIn(f'COPY {name} /opt/TensorFold/{name}', stage[:index])
        self.assertIn('COPY src /opt/TensorFold/src', rows[:rows.index('FROM application AS verification')])
        self.assertLess(index, stage.index('WORKDIR /opt/TensorFold'))
        self.assertFalse(any(line.startswith('RUN chmod ') for line in rows[:rows.index('FROM application AS verification')]))

    @unittest.skipUnless(os.name == 'posix', 'Linux verification image uses POSIX publication modes')
    def test_private_capture_modes_become_public_without_content_or_mtime_changes(self):
        _, _, _, operation = self.publication_instruction()
        with tempfile.TemporaryDirectory(prefix='verification-permissions-') as temporary:
            root = Path(temporary) / 'project'
            root.mkdir(mode=0o700)
            files = []
            for name in ('src', 'tests', 'tools', 'deploy', 'LICENSES'):
                directory = root / name
                directory.mkdir(mode=0o700)
                path = directory / 'source.py'
                path.write_bytes(b'public source fixture\n')
                path.chmod(0o600)
                os.utime(path, ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
                files.append(path)
            changelog = root / 'CHANGELOG.md'
            changelog.write_bytes(b'public changelog fixture\n')
            changelog.chmod(0o600)
            os.utime(changelog, ns=(1_700_000_000_123_456_789, 1_700_000_000_123_456_789))
            files.append(changelog)
            executable = root / 'tools' / 'executable'
            executable.write_bytes(b'public executable fixture\n')
            executable.chmod(0o700)
            files.append(executable)
            before = {path: (hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns)
                      for path in files}
            self.assertTrue(all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in files[:-1]))
            subprocess.run([*operation[:-1], str(root)], check=True, capture_output=True, timeout=10)
            for directory in [root, *(root / name for name in ('src', 'tests', 'tools', 'deploy', 'LICENSES'))]:
                self.assertEqual(stat.S_IMODE(directory.stat().st_mode), 0o755)
            for path in files:
                self.assertEqual((hashlib.sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns), before[path])
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o755 if path == executable else 0o644)
                self.assertEqual(path.stat().st_mode & 0o022, 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
