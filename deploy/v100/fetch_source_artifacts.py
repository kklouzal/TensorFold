#!/usr/bin/env python3
"""Stage frozen source SDK inputs; no installation or SDK/device qualification.

Run with a private, single-writer destination and canonical read-only project
source. Successful files survive failure and are fully rehashed on a retry.
The CLI prints a receipt only after every planned artifact verifies. It needs
only Python's standard library, before packaging or the project is installed.
"""
from __future__ import annotations

import argparse
import errno
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tempfile
import time
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

# The neutral cleanup boundary imports no runtime/model dependencies.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from tensorfold.cleanup import raise_failures  # noqa: E402

DOCUMENT_LIMIT = 16 << 20
CHUNK = 1 << 20
MAX_ARTIFACTS = 512
FOLDERS = {'build-debs', 'python-wheels', 'lint-wheels', 'lint-python'}


def sha(value):
    if type(value) is not str or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected full lowercase SHA256')
    return value


def positive(value):
    if type(value) is not int or value <= 0:
        raise ValueError('expected positive native integer byte count')
    return value


def safe_name(value):
    if (type(value) is not str or len(value) > 255
            or re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.%+:~-]*', value) is None
            or '/' in unquote(value) or '\\' in unquote(value)
            or any(c.isspace() or ord(c) < 32 for c in unquote(value))):
        raise ValueError('unsafe artifact basename')
    return value


def https(value, *, redirect=False):
    if type(value) is not str or len(value) > 4096 or any(c.isspace() or ord(c) < 32 for c in value):
        raise ValueError('invalid HTTPS URL')
    parsed = urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username is not None
            or parsed.password is not None or parsed.fragment or (parsed.query and not redirect)
            or parsed.port not in (None, 443)):
        raise ValueError('artifact URL must use credential-free HTTPS')
    return value


def read_bytes(path):
    with open(path, 'rb', opener=lambda p, f: os.open(p, f | os.O_NONBLOCK | os.O_NOFOLLOW)) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= DOCUMENT_LIMIT:
            raise ValueError('metadata must be a bounded regular file')
        raw = stream.read(before.st_size + 1)
        after = os.fstat(stream.fileno())
        fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
        if len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
            raise ValueError('metadata changed during read')
        return raw


def parse_document(raw, schema):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate metadata key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite metadata')
    doc = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    if (type(doc) is not dict or doc.get('schema') != schema
            or type(doc.get('version')) is not int or doc['version'] != 1):
        raise ValueError('unsupported frozen metadata schema/version')
    return doc


def validate_plan(items):
    if not items or len(items) > MAX_ARTIFACTS:
        raise ValueError('artifact count outside staging bound')
    names, urls, hashes, result = {}, {}, {}, []
    for item in items:
        if set(item) != {'folder', 'name', 'url', 'sha256', 'bytes'} or item['folder'] not in FOLDERS:
            raise ValueError('invalid artifact fields/folder')
        safe_name(item['name'])
        https(item['url'])
        sha(item['sha256'])
        if item['bytes'] is not None:
            positive(item['bytes'])
        identity = (item['url'], item['sha256'], item['bytes'])
        # Identical filenames across environments must mean identical payloads.
        if item['name'] in names and names[item['name']] != identity:
            raise ValueError('conflicting artifact basename metadata')
        if item['url'] in urls and urls[item['url']] != (item['sha256'], item['bytes']):
            raise ValueError('conflicting URL identity')
        if item['sha256'] in hashes and hashes[item['sha256']] != item['bytes']:
            raise ValueError('conflicting SHA256 size metadata')
        names[item['name']] = identity
        urls[item['url']] = item['sha256'], item['bytes']
        hashes[item['sha256']] = item['bytes']
        if item not in result:
            result.append(dict(item))
    return result


def load_plan(directory):
    """Read only the fixed producer schemas and prove each URL lock matches."""
    directory = Path(directory)
    inputs = {}
    def raw(name):
        data = read_bytes(directory / name)
        inputs[name] = {'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}
        return data
    pins = parse_document(raw('source-pins.json'), 'tensorfold.v100-source-foundation')
    apt_raw = raw('source-apt-amd64.json')
    if hashlib.sha256(apt_raw).hexdigest() != sha(pins['source_apt_bundle_sha256']):
        raise ValueError('APT receipt differs from source pins')
    apt = parse_document(apt_raw, 'tensorfold.v100-source-apt-bundle')
    if apt['base'] != pins['base'] or type(apt['packages']) is not list or len(apt['packages']) != 124:
        raise ValueError('wrong frozen APT target/count')
    items = []
    for package in apt['packages']:
        if set(package) != {'architecture', 'artifact', 'bytes', 'package', 'sha256', 'url', 'version'}:
            raise ValueError('invalid APT artifact fields')
        items.append(dict(folder='build-debs', name=package['artifact'], url=package['url'],
                          sha256=package['sha256'], bytes=positive(package['bytes'])))
    for stem, schema, folder in (
        ('build', 'build-tools', 'python-wheels'), ('runtime', 'runtime', 'python-wheels'),
        ('lint', 'lint-tools', 'lint-wheels')):
        doc = parse_document(raw('source-' + schema + '-amd64.json'), 'tensorfold.v100-source-' + schema)
        lock = raw('requirements-source-' + stem + '-amd64.lock')
        if hashlib.sha256(lock).hexdigest() != sha(doc['lock_sha256']):
            raise ValueError('Python lock differs from receipt')
        if set(doc) != {'environment', 'lock_sha256', 'packages', 'resolver_report_sha256', 'schema', 'version'}:
            raise ValueError('invalid Python receipt fields')
        environment = doc['environment']
        if any(environment.get(k) != v for k, v in {
                'sys_platform': 'linux', 'platform_machine': 'x86_64',
                'implementation_name': 'cpython', 'python_version': '3.12'}.items()):
            raise ValueError('wrong Python receipt target')
        packages = doc['packages']
        if type(packages) is not dict or not 0 < len(packages) <= MAX_ARTIFACTS:
            raise ValueError('invalid Python package count')
        expected = []
        for name, package in sorted(packages.items()):
            if (set(package) != {'requires_dist', 'sha256', 'url', 'version'}
                    or type(package['version']) is not str or not package['version']
                    or type(package['requires_dist']) is not list):
                raise ValueError('invalid Python package metadata')
            url = https(package['url'])
            if urlsplit(url).hostname != 'files.pythonhosted.org':
                raise ValueError('wheel must use official PyPI host')
            filename = safe_name(urlsplit(url).path.rsplit('/', 1)[-1])
            if not filename.endswith('.whl'):
                raise ValueError('expected wheel input')
            expected.append(name + ' @ ' + url + ' --hash=sha256:' + sha(package['sha256']))
            items.append(dict(folder=folder, name=filename, url=url, sha256=package['sha256'], bytes=None))
        lines = [line for line in lock.decode('utf-8').splitlines() if line and not line.startswith('#')]
        if lines != expected:
            raise ValueError('Python lock package set differs from receipt')
    lint = pins['lint_python']
    if set(lint) != {'version', 'build', 'url', 'bytes', 'sha256'} or (lint['version'], lint['build']) != ('3.10.20', '20260510'):
        raise ValueError('unexpected lint interpreter identity')
    items.append(dict(folder='lint-python', name=urlsplit(lint['url']).path.rsplit('/', 1)[-1],
                      url=lint['url'], sha256=lint['sha256'], bytes=positive(lint['bytes'])))
    return validate_plan(items), inputs


def acquisition_plan(items, ubuntu_snapshot=None):
    """Explicit Ubuntu archive reacquisition; identities and defaults stay frozen.

    The public Ubuntu Snapshot URI is /ubuntu/<UTC timestamp>/pool/... . This
    selects an acquisition origin, never a new package version or hash fallback.
    """
    items = validate_plan(items)
    if ubuntu_snapshot is None:
        return items, None
    if (type(ubuntu_snapshot) is not str
            or re.fullmatch(r'[0-9]{8}T[0-9]{6}Z', ubuntu_snapshot) is None):
        raise ValueError('Ubuntu snapshot must be a real UTC YYYYMMDDTHHMMSSZ timestamp')
    try:
        datetime.strptime(ubuntu_snapshot, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)
    except ValueError as error:
        raise ValueError('Ubuntu snapshot must be a real UTC YYYYMMDDTHHMMSSZ timestamp') from error
    changes, selected = [], []
    for item in items:
        recorded = urlsplit(item['url'])
        effective = dict(item)
        if (item['folder'] == 'build-debs'
                and recorded.hostname in {'archive.ubuntu.com', 'security.ubuntu.com'}
                and recorded.path.startswith('/ubuntu/pool/')):
            effective['url'] = ('https://snapshot.ubuntu.com/ubuntu/' + ubuntu_snapshot
                                + recorded.path[len('/ubuntu'):])
            changes.append({'folder': item['folder'], 'name': item['name'],
                            'recorded_url': item['url'], 'acquisition_url': effective['url']})
        selected.append(effective)
    return validate_plan(selected), {'kind': 'explicit-ubuntu-snapshot',
                                     'timestamp': ubuntu_snapshot, 'artifacts': changes}


class SecureRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        https(newurl, redirect=True)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


class Transport:
    """Only shipping transport: TLS verification and HTTPS-only redirects."""
    def __init__(self):
        self.opener = build_opener(SecureRedirect())

    def open(self, url, method, timeout):
        return self.opener.open(Request(https(url), method=method,
                                       headers={'Accept-Encoding': 'identity'}), timeout=timeout)


def content_length(response):
    https(response.geturl(), redirect=True)
    if response.status != 200 or response.headers.get('Content-Encoding', 'identity') != 'identity':
        raise ValueError('unexpected HTTP status/content encoding')
    values = response.headers.get_all('Content-Length', [])
    if len(values) != 1 or re.fullmatch('[0-9]{1,20}', values[0]) is None:
        raise ValueError('one bounded Content-Length is required')
    return positive(int(values[0]))


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    primary = None
    try:
        os.fsync(fd)
    except BaseException as error:
        primary = error
    retire(primary, [lambda: os.close(fd)])


def durable_directory(path):
    """Publish each newly created directory entry durably, ancestor first."""
    if not path.exists():
        durable_directory(path.parent)
        try:
            path.mkdir()
        except FileExistsError:
            pass
        fsync_directory(path.parent)
    if path.is_symlink() or not path.is_dir():
        raise ValueError('artifact directory must be a real directory')


def identify(path, expected, limit, check):
    with open(path, 'rb', opener=lambda p, f: os.open(p, f | os.O_NOFOLLOW | os.O_NONBLOCK)) as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError('cached artifact must be a bounded regular file')
        if expected['bytes'] is not None and before.st_size != expected['bytes']:
            raise ValueError('cached artifact size mismatch')
        digest, seen = hashlib.sha256(), 0
        while seen < before.st_size:
            check()
            block = stream.read(min(CHUNK, before.st_size - seen))
            if not block:
                raise ValueError('cached artifact truncated')
            digest.update(block)
            seen += len(block)
        after = os.fstat(stream.fileno())
        if (stream.read(1) or any(getattr(before, k) != getattr(after, k) for k in
                               ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
                or digest.hexdigest() != expected['sha256']):
            raise ValueError('cached artifact changed or SHA256 mismatch')
        os.fsync(stream.fileno())
        return seen


def retire(primary, operations):
    errors = []
    for operation in operations:
        try:
            operation()
        except BaseException as error:
            errors.append(error)
    raise_failures(primary, errors)


def stage(items, destination, *, max_total_bytes, timeout_seconds=60, total_seconds=7200,
          transport=None, inputs=None):
    """Stage sequentially; byte cap counts logical bytes, including cache/dedup.

    Native Linux main-thread execution uses monotonic checks and SIGALRM for
    a cooperative deadline. CPython can delay signal handlers during native
    calls, so this does not guarantee a hard DNS/IO or wall-time bound. Callers
    requiring that bound must own an outer process/container deadline and
    verify process retirement before reusing the destination. No retries,
    worker threads or partial payload reuse. The destination and its parents
    must be caller-controlled.
    """
    items = validate_plan(items)
    positive(max_total_bytes)
    for value in (timeout_seconds, total_seconds):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('timeouts must be finite positive seconds')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise ValueError('staging requires exclusive ownership of SIGALRM timer')
    end = time.monotonic() + total_seconds
    def check():
        left = end - time.monotonic()
        if left <= 0:
            raise TimeoutError('artifact staging deadline exceeded')
        return min(timeout_seconds, left)
    def alarm(signum, frame):
        raise TimeoutError('artifact staging deadline exceeded')
    old_handler = signal.signal(signal.SIGALRM, alarm)
    primary, receipt = None, None
    try:
        signal.setitimer(signal.ITIMER_REAL, total_seconds)
        destination = Path(destination)
        durable_directory(destination)
        for folder in sorted(FOLDERS):
            path = destination / folder
            durable_directory(path)
        fsync_directory(destination)
        fsync_directory(destination.parent)
        transport = transport or Transport()
        results, verified, total = [], {}, 0
        for item in items:
            check()
            path = destination / item['folder'] / item['name']
            size = item['bytes']
            source = verified.get(item['sha256'])
            status, final_url = 'cache', item['url']
            if path.exists() or path.is_symlink():
                size = identify(path, item, max_total_bytes - total, check)
                fsync_directory(path.parent)
            else:
                if source is not None:
                    size = source[1]
                elif size is None:
                    response = transport.open(item['url'], 'HEAD', check())
                    primary = None
                    try:
                        size = content_length(response)
                    except BaseException as error:
                        primary = error
                    retire(primary, [response.close])
                positive(size)
                if total + size > max_total_bytes:
                    raise ValueError('cumulative artifact byte cap exceeded')
                temporary = None
                response = None
                stream = None
                primary = None
                try:
                    fd, temporary = tempfile.mkstemp(prefix='.fetch-', dir=path.parent)
                    try:
                        stream = os.fdopen(fd, 'wb')
                    except BaseException as error:
                        retire(error, [lambda: os.close(fd)])
                    if source is not None:
                        stream.close()
                        stream = None
                        os.unlink(temporary)
                        try:
                            os.link(source[0], temporary, follow_symlinks=False)
                        except OSError as error:
                            if error.errno != errno.EXDEV:
                                raise
                            # Cross-filesystem cache directories cannot share an inode.
                            with open(source[0], 'rb') as cached, open(temporary, 'xb') as copied:
                                remaining = size
                                while remaining:
                                    check()
                                    block = cached.read(min(CHUNK, remaining))
                                    if not block:
                                        raise ValueError('deduplication source truncated')
                                    copied.write(block)
                                    remaining -= len(block)
                                copied.flush()
                                os.fsync(copied.fileno())
                        identify(Path(temporary), dict(item, bytes=size), size, check)
                        status = 'deduplicated'
                    else:
                        response = transport.open(item['url'], 'GET', check())
                        if content_length(response) != size:
                            raise ValueError('download Content-Length differs from expected size')
                        final_url = response.geturl().split('?', 1)[0]
                        digest, seen = hashlib.sha256(), 0
                        while seen < size:
                            check()
                            block = response.read1(min(CHUNK, size - seen))
                            if not block:
                                raise ValueError('truncated artifact body')
                            if len(block) > min(CHUNK, size - seen):
                                raise ValueError('oversized artifact body')
                            stream.write(block)
                            digest.update(block)
                            seen += len(block)
                        check()
                        if response.read1(1) or digest.hexdigest() != item['sha256']:
                            raise ValueError('artifact SHA256 mismatch or oversized body')
                        stream.flush()
                        os.fsync(stream.fileno())
                        stream.close()
                        stream = None
                        completed_response, response = response, None
                        completed_response.close()
                        status = 'downloaded'
                    # link is atomic and never replaces a pre-existing destination.
                    try:
                        os.link(temporary, path, follow_symlinks=False)
                    except FileExistsError:
                        identify(path, dict(item, bytes=size), size, check)
                        status = 'cache-race'
                    fsync_directory(path.parent)
                except BaseException as error:
                    primary = error
                operations = []
                if response is not None:
                    operations.append(response.close)
                if stream is not None:
                    operations.append(stream.close)
                if temporary is not None:
                    operations.append(lambda: os.unlink(temporary))
                operations.append(lambda: fsync_directory(path.parent))
                retire(primary, operations)
            total += size
            if total > max_total_bytes:
                raise ValueError('cumulative artifact byte cap exceeded')
            verified[item['sha256']] = path, size
            results.append(dict(item, bytes=size, result=status, final_url=final_url))
        receipt = {'schema': 'tensorfold.v100-source-artifact-staging', 'version': 1,
                'inputs': inputs or {}, 'artifacts': results, 'logical_bytes': total,
                'evidence': 'verified-source-inputs; no SDK or device qualification'}
    except BaseException as error:
        primary = error
    retire(primary, [lambda: signal.setitimer(signal.ITIMER_REAL, 0),
                     lambda: signal.signal(signal.SIGALRM, old_handler)])
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-directory', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--destination', type=Path, required=True, help='private single-writer release input directory')
    parser.add_argument('--max-total-bytes', type=int, required=True, help='cap logical input bytes, including duplicates/cache')
    parser.add_argument('--timeout-seconds', type=float, default=60, help='maximum individual socket operation time')
    parser.add_argument('--total-seconds', type=float, default=7200, help='cooperative checked/SIGALRM deadline; native calls can delay delivery; use an outer process/container deadline and verify retirement for hard bounds')
    parser.add_argument('--ubuntu-snapshot', metavar='YYYYMMDDTHHMMSSZ',
                        help='explicit Ubuntu archive acquisition timestamp; preserves frozen bytes/SHA256, without automatic fallback')
    args = parser.parse_args()
    items, inputs = load_plan(args.source_directory)
    items, policy = acquisition_plan(items, args.ubuntu_snapshot)
    receipt = stage(items, args.destination, max_total_bytes=args.max_total_bytes,
                    timeout_seconds=args.timeout_seconds, total_seconds=args.total_seconds, inputs=inputs)
    if policy is not None:
        receipt['acquisition_policy'] = policy
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    main()
