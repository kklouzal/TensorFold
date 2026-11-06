"""Offline controls for the frozen SDK input staging boundary."""
import hashlib
from http.client import HTTPMessage
import io
import json
import os
import signal
import shutil
from pathlib import Path
import runpy
import tempfile
import time
import unittest
from unittest.mock import patch

FETCH = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'deploy/v100/fetch_source_artifacts.py'))


def artifact(body=b'payload', folder='python-wheels', size=None):
    return dict(folder=folder, name='example.whl', url='https://files.pythonhosted.org/example.whl',
                sha256=hashlib.sha256(body).hexdigest(), bytes=size)


class Response(io.BytesIO):
    status = 200

    def __init__(self, body, length, *, close_error=None):
        super().__init__(body)
        self.headers = HTTPMessage()
        self.headers['Content-Length'] = str(length)
        self.close_error = close_error

    def geturl(self):
        return 'https://files.pythonhosted.org/example.whl'

    def close(self):
        super().close()
        if self.close_error:
            raise self.close_error


class LocalTransport:
    def __init__(self, body=b'payload', length=None, failure=None):
        self.body = body
        self.length = len(body) if length is None else length
        self.failure = failure
        self.calls = []
        self.responses = []

    def open(self, url, method, timeout):
        self.calls.append(method)
        if self.failure:
            raise self.failure
        response = Response(b'' if method == 'HEAD' else self.body, self.length)
        self.responses.append(response)
        return response


class FetchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def stage(self, items=None, transport=None, **kwargs):
        return FETCH['stage'](items or [artifact()], self.root,
                              max_total_bytes=kwargs.pop('max_total_bytes', 100),
                              transport=transport or LocalTransport(), **kwargs)

    def test_real_frozen_plan_is_bootstrap_readable(self):
        directory = Path(__file__).resolve().parents[1] / 'deploy/v100'
        items, inputs = FETCH['load_plan'](directory)
        self.assertEqual(len(items), 197)
        self.assertEqual(sum(i['folder'] == 'build-debs' for i in items), 124)
        self.assertEqual(len(inputs), 8)
        self.assertTrue(all(len(i['sha256']) == 64 for i in inputs.values()))

    def frozen_metadata_copy(self):
        source = Path(__file__).resolve().parents[1] / 'deploy/v100'
        destination = self.root / 'metadata'
        destination.mkdir()
        for pattern in ('source-*.json', 'requirements-source-*-amd64.lock'):
            for path in source.glob(pattern):
                shutil.copyfile(path, destination / path.name)
        return destination

    def test_apt_missing_size_cannot_become_head_derived(self):
        directory = self.frozen_metadata_copy()
        apt_path = directory / 'source-apt-amd64.json'
        apt = json.loads(apt_path.read_text())
        apt['packages'][0]['bytes'] = None
        apt_raw = json.dumps(apt).encode()
        apt_path.write_bytes(apt_raw)
        pins_path = directory / 'source-pins.json'
        pins = json.loads(pins_path.read_text())
        pins['source_apt_bundle_sha256'] = hashlib.sha256(apt_raw).hexdigest()
        pins_path.write_text(json.dumps(pins))
        # The edited APT hash is valid: refusal must come from its size schema.
        with self.assertRaisesRegex(ValueError, 'positive native integer'):
            FETCH['load_plan'](directory)

    def test_interpreter_missing_size_cannot_become_head_derived(self):
        directory = self.frozen_metadata_copy()
        pins_path = directory / 'source-pins.json'
        pins = json.loads(pins_path.read_text())
        pins['lint_python']['bytes'] = None
        pins_path.write_text(json.dumps(pins))
        with self.assertRaisesRegex(ValueError, 'positive native integer'):
            FETCH['load_plan'](directory)

    def test_snapshot_only_changes_explicit_ubuntu_apt_origins(self):
        directory = Path(__file__).resolve().parents[1] / 'deploy/v100'
        items, _ = FETCH['load_plan'](directory)
        selected, policy = FETCH['acquisition_plan'](items, '20261025T000000Z')
        changed = 0
        for original, effective in zip(items, selected):
            self.assertEqual({k: v for k, v in original.items() if k != 'url'},
                             {k: v for k, v in effective.items() if k != 'url'})
            if original['url'] != effective['url']:
                changed += 1
                self.assertEqual(original['folder'], 'build-debs')
                self.assertTrue(effective['url'].startswith(
                    'https://snapshot.ubuntu.com/ubuntu/20261025T000000Z/pool/'))
            else:
                self.assertEqual(original['url'], effective['url'])
        self.assertEqual(changed, 120)
        self.assertEqual(len(policy['artifacts']), changed)
        self.assertTrue(any('nvidia.com' in i['url'] for i in selected))
        # Host alone is insufficient: only /ubuntu/pool/ APT paths are eligible.
        outside = dict(artifact(folder='build-debs'), url='https://archive.ubuntu.com/not-pool/example.whl')
        wheel = dict(artifact(), url='https://archive.ubuntu.com/ubuntu/pool/example.whl')
        for item in (outside, wheel):
            plan, _ = FETCH['acquisition_plan']([item], '20261025T000000Z')
            self.assertEqual(plan, [item])

    def test_snapshot_default_preserves_original_plan(self):
        items = [dict(artifact(folder='build-debs'),
                      url='https://security.ubuntu.com/ubuntu/pool/example.deb')]
        selected, policy = FETCH['acquisition_plan'](items)
        self.assertEqual(selected, items)
        self.assertIsNone(policy)
        selected, _ = FETCH['acquisition_plan'](items, '20261025T000000Z')
        self.assertEqual(selected[0]['url'],
                         'https://snapshot.ubuntu.com/ubuntu/20261025T000000Z/pool/example.deb')

    def test_snapshot_requires_safe_real_utc_timestamp(self):
        for stamp in ('../../bad', '20260229T000000Z', '20261301T000000Z',
                      '20261025T240000Z', '20261025T000060Z', '20261025T000000+0000', True):
            with self.subTest(stamp=stamp), self.assertRaises(ValueError):
                FETCH['acquisition_plan']([artifact()], stamp)
        FETCH['acquisition_plan']([artifact()], '20240229T235959Z')

    def test_cli_snapshot_receipt_keeps_cache_status_and_origin_provenance(self):
        item = dict(artifact(folder='build-debs', size=7),
                    url='https://archive.ubuntu.com/ubuntu/pool/example.whl')
        self.stage([item])
        original_stage = FETCH['stage']
        def cached_stage(items, destination, **kwargs):
            return original_stage(items, destination,
                                  transport=LocalTransport(failure=AssertionError('no network for cache')),
                                  **kwargs)
        identity = {'source-pins.json': {'sha256': 'a' * 64, 'bytes': 10}}
        output = io.StringIO()
        argv = ['fetch_source_artifacts.py', '--destination', str(self.root),
                '--max-total-bytes', '100', '--ubuntu-snapshot', '20261025T000000Z']
        with patch.dict(FETCH['main'].__globals__, {'load_plan': lambda _: ([item], identity),
                                                   'stage': cached_stage}):
            with patch('sys.argv', argv), patch('sys.stdout', output):
                FETCH['main']()
        receipt = json.loads(output.getvalue())
        self.assertEqual(receipt['inputs'], identity)
        self.assertEqual(receipt['artifacts'][0]['result'], 'cache')
        policy = receipt['acquisition_policy']
        self.assertEqual(policy['kind'], 'explicit-ubuntu-snapshot')
        self.assertEqual(policy['timestamp'], '20261025T000000Z')
        change = policy['artifacts'][0]
        self.assertEqual(change['recorded_url'], item['url'])
        self.assertEqual(change['acquisition_url'], receipt['artifacts'][0]['url'])
        self.assertEqual(receipt['artifacts'][0]['sha256'], item['sha256'])
        self.assertEqual(receipt['artifacts'][0]['bytes'], 7)

    def test_download_cache_and_cross_folder_hardlink(self):
        transport = LocalTransport()
        items = [artifact(), artifact(folder='lint-wheels')]
        receipt = self.stage(items, transport)
        self.assertEqual(transport.calls, ['HEAD', 'GET'])
        self.assertEqual(receipt['logical_bytes'], 14)
        first = self.root / 'python-wheels/example.whl'
        second = self.root / 'lint-wheels/example.whl'
        self.assertEqual(first.read_bytes(), b'payload')
        self.assertEqual(first.stat().st_ino, second.stat().st_ino)
        offline = LocalTransport(failure=AssertionError('cache must not access network'))
        receipt = self.stage(items, offline)
        self.assertEqual([i['result'] for i in receipt['artifacts']], ['cache', 'cache'])
        self.assertTrue(all(r.closed for r in transport.responses))

    def test_existing_wrong_bytes_preserved(self):
        folder = self.root / 'python-wheels'
        folder.mkdir()
        target = folder / 'example.whl'
        target.write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.stage()
        self.assertEqual(target.read_bytes(), b'corrupt')

    def test_failure_does_not_publish_and_resume(self):
        for body, length in ((b'badbody', 7), (b'short', 7), (b'payload!', 7)):
            with self.subTest(body=body), self.assertRaises(ValueError):
                self.stage(transport=LocalTransport(body, length))
            self.assertEqual(list(self.root.rglob('.fetch-*')), [])
            self.assertFalse((self.root / 'python-wheels/example.whl').exists())
        self.stage()
        self.assertEqual((self.root / 'python-wheels/example.whl').read_bytes(), b'payload')

    def test_body_is_not_requested_over_resource_cap(self):
        transport = LocalTransport(length=101)
        with self.assertRaisesRegex(ValueError, 'byte cap'):
            self.stage(transport=transport)
        self.assertEqual(transport.calls, ['HEAD'])
        self.assertEqual(list(self.root.rglob('.fetch-*')), [])

    def test_pinned_size_does_not_need_head(self):
        transport = LocalTransport()
        self.stage([artifact(size=7)], transport)
        self.assertEqual(transport.calls, ['GET'])

    def test_unsafe_metadata_and_conflicts(self):
        for update in ({'name': '../escape'}, {'name': 'x%2fy'}, {'url': 'http://example.org/a'},
                       {'url': 'https://user:password@example.org/a'}, {'bytes': True},
                       {'sha256': 'A' * 64}, {'folder': '../outside'}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.stage([dict(artifact(), **update)])
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            self.stage([artifact(), artifact(b'changed', folder='lint-wheels')])
        self.assertEqual(list(self.root.iterdir()), [])

    def test_duplicate_json_and_boolean_schema_refused(self):
        for raw in (b'{"schema":"s","version":true}',
                    b'{"schema":"s","version":1,"version":1}'):
            with self.assertRaises(ValueError):
                FETCH['parse_document'](raw, 's')

    def test_cancel_closes_stream_and_removes_partial(self):
        transport = LocalTransport()
        original_open = transport.open
        def open_cancel(url, method, timeout):
            response = original_open(url, method, timeout)
            if method == 'GET':
                response.read1 = lambda size: (_ for _ in ()).throw(KeyboardInterrupt())
            return response
        transport.open = open_cancel
        with self.assertRaises(KeyboardInterrupt):
            self.stage(transport=transport)
        self.assertTrue(all(r.closed for r in transport.responses))
        self.assertEqual(list(self.root.rglob('.fetch-*')), [])
        self.assertFalse((self.root / 'python-wheels/example.whl').exists())

    def test_cleanup_failure_preserves_primary(self):
        transport = LocalTransport()
        original_open = transport.open
        primary = RuntimeError('body failure')
        cleanup = OSError('close failure')
        def open_failure(url, method, timeout):
            response = original_open(url, method, timeout)
            if method == 'GET':
                response.read1 = lambda size: (_ for _ in ()).throw(primary)
                response.close_error = cleanup
            return response
        transport.open = open_failure
        with self.assertRaises(RuntimeError) as caught:
            self.stage(transport=transport)
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, cleanup)
        self.assertEqual(list(self.root.rglob('.fetch-*')), [])

    def test_signal_deadline_interrupts_supported_sleep(self):
        class SlowTransport:
            def open(self, *args):
                time.sleep(3)
        with self.assertRaises(TimeoutError):
            self.stage(transport=SlowTransport(), total_seconds=0.02)
        self.assertFalse((self.root / 'python-wheels/example.whl').exists())

    def test_publish_race_never_overwrites_winner(self):
        original_link = FETCH['stage'].__globals__['os'].link
        target = self.root / 'python-wheels/example.whl'
        def raced_link(source, destination, **kwargs):
            if Path(destination) == target:
                target.write_bytes(b'hostile')
            return original_link(source, destination, **kwargs)
        with patch.object(FETCH['stage'].__globals__['os'], 'link', raced_link):
            with self.assertRaisesRegex(ValueError, 'SHA256'):
                self.stage()
        self.assertEqual(target.read_bytes(), b'hostile')
        self.assertEqual(list(self.root.rglob('.fetch-*')), [])

    def test_temporary_unlink_failure_reports_owned_leftover(self):
        original = os.unlink
        cleanup = OSError('temporary unlink failure')
        def fail_temporary(path, *args, **kwargs):
            if Path(path).name.startswith('.fetch-'):
                raise cleanup
            return original(path, *args, **kwargs)
        with patch.object(FETCH['stage'].__globals__['os'], 'unlink', fail_temporary):
            with self.assertRaises(ValueError) as caught:
                self.stage(transport=LocalTransport(b'badbody'))
        self.assertIs(caught.exception.__cause__, cleanup)
        leftovers = list(self.root.rglob('.fetch-*'))
        self.assertEqual(len(leftovers), 1)
        self.assertFalse((self.root / 'python-wheels/example.whl').exists())
        leftovers[0].unlink()

    def test_symlink_cache_is_rejected_without_following(self):
        outside = self.root / 'untouched'
        outside.write_bytes(b'payload')
        folder = self.root / 'python-wheels'
        folder.mkdir()
        (folder / 'example.whl').symlink_to(outside)
        with self.assertRaises(OSError):
            self.stage()
        self.assertEqual(outside.read_bytes(), b'payload')

    def test_head_get_size_disagreement_cannot_publish(self):
        transport = LocalTransport()
        original = transport.open
        def disagree(url, method, timeout):
            response = original(url, method, timeout)
            if method == 'GET':
                response.headers.replace_header('Content-Length', '8')
            return response
        transport.open = disagree
        with self.assertRaisesRegex(ValueError, 'differs'):
            self.stage(transport=transport)
        self.assertTrue(all(r.closed for r in transport.responses))
        self.assertEqual(list(self.root.rglob('.fetch-*')), [])
        self.assertFalse((self.root / 'python-wheels/example.whl').exists())

    def test_directory_fsync_and_close_failure_preserves_both(self):
        primary = OSError('directory fsync failure')
        cleanup = OSError('directory close failure')
        original_close = os.close
        closed = []
        def fail_close(fd):
            original_close(fd)
            closed.append(fd)
            raise cleanup
        with patch.object(FETCH['fsync_directory'].__globals__['os'], 'fsync', side_effect=primary):
            with patch.object(FETCH['fsync_directory'].__globals__['os'], 'close', fail_close):
                with self.assertRaises(OSError) as caught:
                    FETCH['fsync_directory'](self.root)
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, cleanup)
        self.assertEqual(len(closed), 1)
        with self.assertRaises(OSError):
            os.fstat(closed[0])

    def test_timer_arming_failure_restores_handler(self):
        handler = signal.getsignal(signal.SIGALRM)
        original = signal.setitimer
        primary = OverflowError('cannot arm')
        def arming(timer, seconds):
            if seconds:
                raise primary
            return original(timer, seconds)
        with patch.object(FETCH['stage'].__globals__['signal'], 'setitimer', arming):
            with self.assertRaises(OverflowError) as caught:
                self.stage()
        self.assertIs(caught.exception, primary)
        self.assertIs(signal.getsignal(signal.SIGALRM), handler)

    def test_fdopen_failure_retires_raw_descriptor(self):
        original = os.fdopen
        seen = []
        primary = MemoryError('fdopen failure')
        def failing(fd, mode):
            seen.append(fd)
            raise primary
        with patch.object(FETCH['stage'].__globals__['os'], 'fdopen', failing):
            with self.assertRaises(MemoryError) as caught:
                self.stage()
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(seen), 1)
        with self.assertRaises(OSError):
            os.fstat(seen[0])
        self.assertEqual(list(self.root.rglob('.fetch-*')), [])
        self.assertIs(os.fdopen, original)

    def test_cache_inode_is_fsynced_before_success(self):
        self.stage()
        target = (self.root / 'python-wheels/example.whl').stat()
        role = (self.root / 'python-wheels').stat()
        original = os.fsync
        seen = []
        def record(fd):
            info = os.fstat(fd)
            seen.append((info.st_dev, info.st_ino, info.st_mode))
            return original(fd)
        with patch.object(FETCH['stage'].__globals__['os'], 'fsync', record):
            self.stage(transport=LocalTransport(failure=AssertionError('network')))
        self.assertIn((target.st_dev, target.st_ino, target.st_mode), seen)
        self.assertIn((role.st_dev, role.st_ino, role.st_mode), seen)

    def test_redirect_refuses_plaintext_and_credentials(self):
        redirect = FETCH['SecureRedirect']()
        for url in ('http://example.org/x', 'https://secret@example.org/x'):
            with self.assertRaises(ValueError):
                redirect.redirect_request(None, None, 302, '', {}, url)


if __name__ == '__main__':
    unittest.main()
