"""Qualify RAM indexer storage against frozen same-format GPU-history replies.

Uses saved Client exchanges without changing generation budgets or sampling.
This is functional/allocation qualification, not a throughput benchmark or a
full-context payload test. The timeout is socket inactivity, as in Client.
This host qualification uses physical GPUs 0..7 in rank order and rejects
reordered launches; the runtime's UUID-based NUMA remapping is unchanged.
"""

import argparse
from collections import Counter
import copy
import hashlib
import math
import os
from pathlib import Path
import re
import stat
import threading
import time

if __package__:
    from . import v100_rotorquant_live as live
    from .worker_lifetime import Task, drain, raise_failures
else:
    import v100_rotorquant_live as live
    from worker_lifetime import Task, drain, raise_failures

require, integer = live.require, live.integer
FORMATS = ('int8', *live.Q4_FORMATS)


def load_references(directory, kv_dtype):
    """Read the frozen exchange format, rather than load_cases' request corpus."""
    directory, hashes = Path(directory), {}

    def read(name):
        with open(directory / name, 'rb', opener=lambda p, flags: os.open(p, flags | os.O_NOFOLLOW | os.O_NONBLOCK)) as stream:
            require(stat.S_ISREG(os.fstat(stream.fileno()).st_mode), 'ordinary reference file required')
            raw = stream.read((16 << 20) + 1)
        require(len(raw) <= 16 << 20, 'reference exceeds 16 MiB bound')
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return live.protocol.object_json(raw)

    authority = read('authority.json')
    require(authority.get('history') == 'gpu' and authority.get('kv_dtype') == kv_dtype,
            'frozen GPU reference must use the same selected KV format')
    require(re.fullmatch(r'sha256:[0-9a-f]{64}', str(authority.get('image', ''))), 'immutable reference image required')
    seeded = [('seeded-' + name, body) for name, body in live.profiles('', 128)]
    names = [*live.CASES, *(name for name, _ in seeded)]
    require(set(authority.get('results', {})) == set(names), 'all twelve original and nine seeded authorities required')
    rows = []
    for name in names:
        label = name if name.startswith('seeded-') else 'serial-' + name
        request, exchange, result = (read(label + '/' + file) for file in ('request.json', 'exchange.json', 'result.json'))
        body = request.get('body')
        require(request.get('method') == 'POST' and type(body) is dict, 'saved completion request required')
        require(body.get('draft') is False and not body.get('stream') and body.get('return_token_ids') is True,
                'frozen serial request must retain complete token IDs')
        require(integer(body.get('max_tokens'), 1) and body['max_tokens'] <= 4096
                and type(body.get('messages')) is list and body['messages'], 'bounded saved messages/budget required')
        require(not body.get('tools') and not body.get('functions') and body.get('n', 1) == 1,
                'single text/reasoning references required')
        require(exchange.get('complete') is True and exchange.get('status') == 200
                and exchange.get('cancelled_by_client') is False, 'reference exchange did not finish successfully')
        require(live.completion(result, body['max_tokens']) == authority['results'][name],
                'frozen result differs from its authority')
        if name.startswith('seeded-'):
            expected = dict(live.profiles(body['model'], 128))[name.removeprefix('seeded-')]
            require(all(body.get(key) == value for key, value in expected.items()),
                    'seeded sampling/thinking/token budget changed')
        rows.append((name, body, result))
    return rows, authority, hashes


def capacity_proof(stats, *, parallel, context, kv_dtype):
    require(stats.get('object') == 'tensorfold.stats', 'ordinary public stats required')
    model, plan = stats.get('model') or {}, stats.get('capacity_plan') or {}
    require(model.get('kv_dtype') == kv_dtype and kv_dtype in FORMATS, 'service KV format mismatch')
    require(model.get('context_length') == context and plan.get('context_window') == context, 'context mismatch')
    require(stats.get('limits', {}).get('lanes') == parallel, 'service concurrency mismatch')
    windows = plan.get('kv_full_windows') or {}
    require(all(windows.get(key) == parallel for key in ('requested', 'reserved', 'stream_slots')),
            'every lane must reserve a complete equal window')
    capacity = windows.get('cache_slots_each')
    require(windows.get('materialized') is True and windows.get('context_window') == context
            and integer(capacity, context) and windows.get('startup_capacities') == [capacity] * parallel,
            'physically materialized equal full windows required')
    require(integer(windows.get('workspace_bytes'), 2 << 30), 'existing 2 GiB serving workspace must remain protected')
    ranks, cache_bytes = windows.get('ranks'), windows.get('allocated_cache_bytes')
    require(type(ranks) is list and len(ranks) == 8 and integer(cache_bytes, 1), 'eight physical cache rank receipts required')
    for rank, row in enumerate(ranks):
        require(row.get('rank') == rank and row.get('materialized') is True
                and row.get('startup_capacities') == [capacity] * parallel
                and row.get('allocated_cache_bytes') == cache_bytes, 'rank cache allocation mismatch')
        require(integer(row.get('torch_allocated_bytes'), cache_bytes)
                and integer(row.get('torch_reserved_bytes'), row['torch_allocated_bytes']), 'rank physical allocation missing')
    history = plan.get('indexer_history') or {}
    require(history.get('mode') == 'ram' and history.get('policy') == 'lossless-fp16-ram-archive-v1'
            and history.get('materialized') is True and history.get('archive_limit') == capacity,
            'materialized full-window RAM archive required')
    require(history.get('host_archive') == 'full-window-pageable'
            and history.get('pinned') == 'bounded-transfer-staging-only', 'archive/pinned representation mismatch')
    placements, startup = history.get('ranks'), history.get('startup_ranks')
    require(type(placements) is list and type(startup) is list and len(placements) == len(startup) == 8,
            'all eight NUMA/archive startup ranks required')
    require(Counter(row.get('node') for row in placements) == {0: 4, 1: 4}, 'host requires two NUMA groups of four GPUs')
    require(len({tuple(row.get('gpu_uuid_words', [])) for row in placements}) == 8, 'distinct physical GPU identities required')
    for rank, (placement, row) in enumerate(zip(placements, startup, strict=True)):
        words = placement.get('gpu_uuid_words')
        require(type(words) is list and len(words) == 4 and all(integer(word) and word < 2**32 for word in words),
                'complete physical GPU UUID words required')
        require(placement.get('rank') == row.get('rank') == rank and placement.get('socket') == placement.get('node'),
                'ordered local socket/NUMA rank receipt required')
        require(placement.get('node') == rank // 4, 'qualification requires physical GPUs0..7 in fixed rank/NUMA order')
        require(integer(row.get('owners'), parallel) and row['owners'] <= history.get('maximum_owners', 0), 'startup archive owners mismatch')
        for actual, grant in (('host_backing_bytes', 'host_grant_bytes'), ('pinned_staging_bytes', 'pinned_grant_bytes')):
            require(integer(row.get(actual), 1) and row[actual] <= placement.get(grant, 0), 'archive/staging grant exceeded')
        require(row.get('device_raw_bytes') == row['owners'] * history.get('layers', 0) * history.get('device_bytes_per_layer', 0)
                and integer(row['device_raw_bytes'], 1), 'bounded device history allocation mismatch')
    allocator = history.get('local_allocator') or {}
    require(allocator.get('policy') == 'numa-local-private-anon-mbind-verified-v1'
            and allocator.get('charged_bytes') == startup[0]['host_backing_bytes']
            and allocator.get('pinned_extent_bytes') == startup[0]['pinned_staging_bytes'], 'local archive allocation receipt mismatch')
    local = allocator.get('placement') or {}
    require(local == history.get('placement') and local.get('node') == placements[0]['node'], 'local placement identity mismatch')
    node = local['node']
    cpus = set(range(0, 16)) | set(range(32, 48)) if node == 0 else set(range(16, 32)) | set(range(48, 64))
    require(local.get('cpus') and set(local['cpus']) <= cpus and local.get('sockets') == [node], 'local CPU affinity receipt missing')
    allocations = allocator.get('allocations')
    require(type(allocations) is list and len(allocations) == 5 * startup[0]['owners'], 'complete local archive/staging receipts required')
    archives = [row for row in allocations if row.get('tag') == 'indexer-archive']
    require(len(archives) == startup[0]['owners']
            and all(row.get('shape') == [history['layers'], capacity, 128]
                    and row.get('dtype') == 'torch.float16' and row.get('pinned') is False for row in archives),
            'complete pageable FP16 raw-history windows required')
    require(sum(row.get('backing_bytes', 0) for row in allocations) == allocator['charged_bytes'], 'local backing byte sum mismatch')
    for row in allocations:
        require(row.get('placement') == local and row.get('mapping') == 'MAP_PRIVATE|MAP_ANONYMOUS'
                and row.get('bind_mode') == 32770 and row.get('bind_flags') == 0
                and row.get('write_first_touch') is True, 'strict physical NUMA binding receipt required')
        for phase in ('before_pin', 'after_pin') if row.get('pinned') else ('before_pin',):
            proof = row.get(phase) or {}
            require(proof.get('verified') is True and integer(proof.get('pages'), 1)
                    and proof.get('node_pages') == {str(node): proof['pages']}
                    and proof.get('extent_bytes') == proof['pages'] * local['page_size'], 'physical page locality proof missing')
    return {'context': context, 'parallel': parallel, 'kv_dtype': kv_dtype, 'capacity_each': capacity,
            'cache_bytes_per_rank': cache_bytes, 'history': history, 'allocation_source': 'public startup receipts; no full-context payload test'}


def run(args):
    rows, authority, hashes = load_references(args.reference_dir, args.kv_dtype)
    output = args.output_dir
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), 'new or empty output directory required')
    output.mkdir(parents=True, exist_ok=True)
    client = live.Client(args.url, output, timeout=args.timeout, limits=live.protocol.from_arguments(args))
    mode = 'qualified' if args.kv_dtype == 'int8' else 'cold'
    summary = {'complete': False, 'passed': False, 'prefix_reuse': mode, 'checks': [],
               'scope': 'same-format frozen GPU vs RAM functional/allocation qualification; not a benchmark',
               'tool_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'reference_image': authority['image'], 'reference_files_sha256': hashes}

    def check(group, name, reference, result, maximum, *, recovery=False, serial=None):
        receipt = live.prefix_receipt(result, mode, recovery=recovery)
        comparison = live.compare(reference, result, maximum)
        if serial is not None:
            comparison['against_ram_serial'] = live.compare(serial, result, maximum)
            comparison['passed'] &= comparison['against_ram_serial']['passed']
        summary['checks'].append({'group': group, 'case': name, **receipt, **comparison})
        live.write_json(output / 'summary.json', summary)

    primary = None
    try:
        initial = client.call('stats-initial')
        summary['capacity'] = capacity_proof(initial, parallel=args.parallel, context=args.context, kv_dtype=args.kv_dtype)
        require(live.idle(initial), 'initially idle service required')
        names = [initial['model']['id'], *initial['model'].get('aliases', [])]
        require(all(body.get('model') in names for _, body, _ in rows), 'saved model identity is not served; bodies must not be rewritten')
        serials = {}
        for name, original, frozen in rows:
            for drafting in (False, True):
                result = client.call(name + ('-draft' if drafting else '-serial'), {**copy.deepcopy(original), 'draft': drafting})
                if not drafting:
                    serials[name] = result
                check('draft' if drafting else 'serial', name, frozen, result, original['max_tokens'])
        ordered = [row for row in rows if row[0].startswith('seeded-')] + [row for row in rows if not row[0].startswith('seeded-')]
        concurrent = ordered[:args.parallel]
        barrier, peak, poll = threading.Barrier(args.parallel), 0, 0

        def one(index):
            barrier.wait(timeout=args.timeout)
            return client.call(f'concurrent-{index}', {**copy.deepcopy(concurrent[index][1]), 'draft': True})

        deadline, results = time.monotonic() + args.timeout, [None] * args.parallel

        def request(index):
            results[index] = one(index)

        tasks = [Task(lambda index=index: request(index), name=f'archive-http-{index}') for index in range(args.parallel)]
        concurrent_error = None
        try:
            for task in tasks:
                task.start()
            while not all(task.done.is_set() for task in tasks):
                stats = client.call(f'concurrent-stats-{poll}')
                peak = max(peak, stats.get('load', {}).get('requests_in_flight', 0))
                require(time.monotonic() < deadline, 'concurrent acceptance polling deadline exceeded')
                poll += 1
                time.sleep(.1)
        except BaseException as error:
            concurrent_error = error
        finally:
            abort_error = None
            try:
                if concurrent_error is not None:
                    try:
                        barrier.abort()
                    except BaseException as error:
                        abort_error = error
            finally:
                try:
                    drain(tasks, concurrent_error)
                except BaseException as retired:
                    if abort_error is not None:
                        raise_failures(retired, [abort_error])
                    raise
        require(peak >= args.parallel, 'all offered concurrent requests were not observed in flight')
        summary['concurrency_peak_in_flight'] = peak
        for (name, body, frozen), result in zip(concurrent, results, strict=True):
            check('concurrent', name, frozen, result, body['max_tokens'], serial=serials[name])
        name, body, frozen = concurrent[0]
        stream_body = {**copy.deepcopy(body), 'draft': True, 'stream': True, 'stream_options': {'include_usage': True}}
        result = client.call('complete-sse', stream_body)
        check('sse', name, frozen, result, body['max_tokens'])
        before = client.call('stats-before-cancel')
        cancelled = client.call('cancelled-sse', stream_body, cancel=True)
        require(cancelled.get('intentionally_incomplete') is True, 'cancellation must interrupt an active stream')
        deadline, attempt = time.monotonic() + args.cleanup_timeout, 0
        while True:
            recovered = client.call(f'stats-cancel-recovery-{attempt}')
            if live.idle(recovered) and recovered.get('totals', {}).get('cancelled', 0) > before.get('totals', {}).get('cancelled', 0):
                break
            require(time.monotonic() < deadline, 'cancelled request did not retire and increment its counter')
            attempt += 1
            time.sleep(.25)
        health = client.call('health-after-cancel', path='/health')
        require(health.get('ok') is True or health.get('status') == 'ok', 'service unhealthy after cancellation')
        for phase in ('prime', 'reuse'):
            result = client.call('prefix-recovery-' + phase, {**copy.deepcopy(body), 'draft': True})
            check('prefix-recovery', phase, frozen, result, body['max_tokens'], recovery=phase == 'reuse')
        final = client.call('stats-final')
        capacity_proof(final, parallel=args.parallel, context=args.context, kv_dtype=args.kv_dtype)
        require(final.get('server') == initial.get('server') and final['model'] == initial['model'], 'service identity changed')
        require(live.idle(final) and final.get('totals', {}).get('failed') == initial.get('totals', {}).get('failed'), 'service failed requests or did not return idle')
        summary.update(complete=True, passed=all(row['passed'] for row in summary['checks']), cancellation_cleanup=True)
        return 0 if summary['passed'] else 2
    except BaseException as error:
        primary = error
        summary['error'] = {'type': type(error).__name__, 'message': str(error)}
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        try:
            live.write_json(output / 'summary.json', summary)
        except BaseException as publication:
            raise_failures(primary, [publication])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--reference-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--parallel', type=int, required=True)
    parser.add_argument('--context', type=int, required=True)
    parser.add_argument('--kv-dtype', choices=FORMATS, required=True)
    parser.add_argument('--timeout', type=float, default=900, help='per-socket inactivity timeout; concurrent polling also has this deadline')
    parser.add_argument('--cleanup-timeout', type=float, default=60)
    live.protocol.add_arguments(parser)
    args = parser.parse_args(argv)
    require(1 <= args.parallel <= 21 and args.context > 0, 'positive context and concurrency1..21 required')
    require(all(math.isfinite(value) and value > 0 for value in (args.timeout, args.cleanup_timeout)), 'finite positive timeouts required')
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
