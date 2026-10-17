"""Actual family/CUDA App/loopback HTTP model probe; launched only by the owned GPU runner."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import inspect
import json
import os
from pathlib import Path
import platform
import sys
import threading
import time
import urllib.error
import urllib.request

LRU = ('Write a complete Python module implementing an LRU cache class with get, put, delete, resize '
       'and a thread-safe wrapper, with docstrings and a short usage example.')
MODEL = 'heretic-probe'
MAX_REPLY_BYTES = 16 << 20


def parse(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=Path('/model'))
    parser.add_argument('--drafter', type=Path)
    parser.add_argument('--context', type=int, default=4096,
                        help='explicit representative workload context; does not change model defaults')
    parser.add_argument('--streams', type=int, choices=(1, 4), default=1)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--bits', type=int, choices=(16, 4), default=16)
    parser.add_argument('--pin', choices=('off', 'auto', 'on'), default='off')
    parser.add_argument('--kernel', choices=('single-blob',), default='single-blob')
    parser.add_argument('--max-tokens', type=int, default=512)
    parser.add_argument('--requests', type=int, default=3)
    parser.add_argument('--concurrent', action='store_true', help='also run four concurrent HTTP requests with streams4')
    parser.add_argument('--output', type=Path, default=Path('/evidence/model-probe.json'))
    args = parser.parse_args(argv)
    if (args.context <= 0 or args.max_tokens <= 0 or args.requests <= 0
            or not (2 if args.drafter is not None else 1) <= args.rows <= (16 if args.streams == 4 else 128)
            or args.concurrent and args.streams != 4):
        parser.error('positive context/token/request counts, valid draft rows, and streams4 for concurrent requests required')
    return args


def body(max_tokens, *, draft=True, seeded=False):
    return {'model': MODEL, 'messages': [{'role': 'user', 'content': LRU}], 'max_tokens': max_tokens,
            'temperature': .7 if seeded else 0, 'top_k': 20 if seeded else 0,
            'top_p': .95 if seeded else 1, 'seed': 47 if seeded else 7, 'draft': draft,
            'chat_template_kwargs': {'enable_thinking': False}, 'return_token_ids': True}


def reply_tokens(response):
    block = response['tensorfold']
    values = block['token_ids']
    if (not isinstance(values, list) or not values or any(type(value) is not int or value < 0 for value in values)
            or len(values) != response['usage']['completion_tokens']):
        raise ValueError('Response token IDs/count are malformed or no complete token was produced')
    digest = hashlib.sha256(','.join(map(str, values)).encode()).hexdigest()[:12]
    if block['token_sha'] != digest:
        raise ValueError('Response token_sha differs from the complete token IDs')
    if response['choices'][0]['finish_reason'] not in ('stop', 'length'):
        raise ValueError('Model response did not complete normally')
    return values


def validate_stats(stats, completed):
    if stats['object'] != 'tensorfold.stats' or stats['schema_version'] != 1:
        raise ValueError('Unexpected HTTP statistics schema')
    totals, load = stats['totals'], stats['load']
    expected = {reason: sum(row['choices'][0]['finish_reason'] == reason for row in completed)
                for reason in ('stop', 'length', 'tool_calls')}
    for name, value in expected.items():
        if type(totals['finished'][name]) is not int or totals['finished'][name] != value:
            raise ValueError('HTTP finished counters differ from complete captured replies')
    checks = ((totals['requests'], len(completed)),
              (totals['generated_tokens'], sum(row['usage']['completion_tokens'] for row in completed)),
              (totals['failed'], 0), (totals['cancelled'], 0),
              (load['requests_in_flight'], 0), (load['requests_running'], 0), (load['requests_waiting'], 0))
    if any(type(actual) is not int or actual != expected for actual, expected in checks):
        raise ValueError('HTTP statistics differ from completed work or retain active/failed request owners')


def cache_receipt(engine):
    cache = getattr(engine, 'expert_cache', None)
    multi = getattr(engine, 'multi', None)
    prefix = engine.cache if multi is None else multi.cache
    result = {'expert_cache': None, 'prefix_entries': len(prefix.entries),
              'prefix_cache_owner': 'engine.cache' if multi is None else 'engine.multi.cache'}
    if cache is not None:
        with cache._lock:
            values = {key: value for key, value in vars(cache).items()
                      if not key.startswith('_') and type(value) in (int, float, bool, str, type(None))}
            values.update(host_bytes=cache.host_bytes, score_history_entries=cache.score_history_entries,
                          policy_tick=cache._policy.tick, policy_touches=cache._policy.touches)
            values['pin_receipt'] = None if cache.pin_policy is None else cache.pin_policy.receipt()
            result['expert_cache'] = values
    return result


@contextmanager
def kernel_factory_scope(choice):
    """The private engine binds every blob bank before constructor calibration."""
    if choice != 'single-blob':
        raise ValueError('This private helper only measures the explicit single-blob candidate')
    from tensorfold.families.qwen3_5_moe.cuda.nvfp4_engine import QwenNvfp4MoeEngine
    if getattr(QwenNvfp4MoeEngine, '_experimental_cache_layout', None) != 'single-blob-grouped':
        raise RuntimeError('Private single-blob engine source was not loaded')
    yield []


def module_identity():
    result = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if name.startswith('tensorfold') and path is not None:
            source = Path(path)
            if source.is_file():
                with source.open('rb') as file:
                    result[name] = {'path': str(source), 'sha256': hashlib.file_digest(file, 'sha256').hexdigest()}
    return result


class Loop:
    """Own the loop thread and expose every background failure during retirement."""
    def __init__(self, server):
        self.server = server
        self.failure = None
        self.scope = threading.Lock()
        self.entered = threading.Event()
        self.thread = threading.Thread(target=self.run, name='probe-http-loop')

    def run(self):
        with self.scope:
            self.entered.set()
            try:
                self.server.serve_forever(poll_interval=.05)
            except BaseException as error:
                self.failure = error

    def start(self):
        self.thread.start()
        if not self.entered.wait(30):
            raise RuntimeError('HTTP serving thread did not acknowledge startup')

    def _join(self):
        if not self.server.handlers_drained:
            raise RuntimeError('HTTP handlers are not drained; serving owners must remain retained')
        if self.thread.ident is not None:
            with self.scope:
                pass
            self.thread.join()
        if self.failure is not None:
            raise self.failure

    def stop(self):
        from tensorfold.cleanup import finish

        # server_close stops admission even if the loop has not entered yet.
        # The native loop scope retires before engine cleanup can succeed.
        finish((self.server.server_close, self._join))


def run(args):
    from tensorfold import __version__, families
    from tensorfold.cleanup import finish, raise_failures
    from tensorfold.cli import _close_serving
    from tensorfold.cuda import precision, prompt_precision
    from tensorfold.cuda.server import App, Server, make_handler
    import torch
    import triton

    records, prepared = {}, {}
    record_lock = threading.Lock()
    owners = {'engine': None, 'app': None, 'server': None, 'line': None}
    loop = engine = app = None
    primary = cleanup_failure = output_failure = None
    report = {'configuration': vars(args) | {'model': str(args.model), 'drafter': None if args.drafter is None else str(args.drafter),
                                             'output': str(args.output)},
              'workload_note': f'context{args.context} is an explicit representative workload override, not a model default',
              'cache_condition': 'first HTTP request follows constructor calibration; first requested prefix, not cold expert/kernel state; later repetitions reuse the same chat',
              'kernel_note': 'PRIVATE single-blob-grouped engine default, one H2D per miss; unselected for production',
              'records': records, 'prepared_requests': prepared, 'comparisons': {}}
    started = time.perf_counter()
    try:
        precision.set_mode(precision.CHECKPOINT, asked=True)
        prompt_precision.set_fp8(False)
        family = families.detect(args.model)
        families.require_readable(family, families.read_config(args.model), 'cuda')
        options = dict(drafter='' if args.drafter is None else str(args.drafter), context=args.context,
                       context_explicit=True, parallel=args.streams, draft_rows=args.rows, drafter_bits=args.bits,
                       vram_experts='auto', vram_experts_pin=args.pin, no_drafts=False)
        report['family'] = {'module': family.module, 'model_type': family.model_type, 'title': family.title}
        report['factory_options'] = options
        with kernel_factory_scope(args.kernel) as bound_banks:
            engine = family.package.cuda_engine(args.model, **options)
            owners['engine'] = engine
            from tensorfold.cuda.nvfp4.blob_cache_a4 import BlobCachedExpertsA4, BlobExpertCache
            from tensorfold.cuda.nvfp4.blob_grouped_a4 import run as blob_run
            targets = [layer.moe.experts for layer in engine.w.layers]
            if (not isinstance(engine.expert_cache, BlobExpertCache)
                    or engine.expert_layout.get('experimental_variant') != 'single-blob-grouped'
                    or not targets or any(not isinstance(bank, BlobCachedExpertsA4)
                    or bank.cache is not engine.expert_cache
                    or getattr(bank.run, '__func__', None) is not blob_run for bank in targets)):
                raise RuntimeError('Private single-blob source/default did not cover every target bank')
            bound_banks.extend(targets)
        loaded = time.perf_counter()
        report['bound_target_banks'] = len(bound_banks)
        report['kernel_calibration'] = 'selected kernel installed during target loading, before constructor calibration'
        app_class = getattr(family.package, 'CUDA_APP', None) or App
        app = app_class(engine, args.model, MODEL, default_thinking=False, thinking_budget=0,
                        context_window=args.context, max_tokens=args.max_tokens)
        owners['app'] = app
        if 'draft' not in inspect.signature(engine.generate).parameters or type(app).run is not App.run:
            raise TypeError('This probe requires the verified CUDA App HTTP draft switch for same-engine comparisons')
        report['http_draft_switch'] = True
        server = Server(('127.0.0.1', 0), make_handler(app))
        owners['server'] = server
        loop = Loop(server)
        owners['line'] = loop
        loop.start()
        base_url = f'http://127.0.0.1:{server.server_port}'
        prop = torch.cuda.get_device_properties(0)
        report['runtime'] = dict(python=sys.version, platform=platform.platform(), tensorfold=__version__,
                                 torch=torch.__version__, triton=triton.__version__, cuda=torch.version.cuda,
                                 source_revision=os.environ.get('TENSORFOLD_SOURCE_REVISION'),
                                 gpu=prop.name, device_uuid=str(prop.uuid), compute_capability=list(torch.cuda.get_device_capability()),
                                 device_total_bytes=prop.total_memory,
                                 memory_policy={key: os.environ.get(key) for key in
                                 ('TENSORFOLD_MEMORY_RESERVE_GIB', 'TENSORFOLD_CUDA_MEMORY_RESERVE_GIB', 'TENSORFOLD_CUDA_MEMORY_LIMIT_GB')})
        report['capacity_plan'] = engine.capacity_plan
        report['expert_pool_plan'] = getattr(engine, 'expert_pool_plan', None)
        report['expert_layout'] = getattr(engine, 'expert_layout', None)
        report['metadata'] = dict(eos=list(engine.eos), template=app.template.provenance(),
                                  secondary_present=engine.draft is not None, effective_context=app.effective_context_window)
        report['startup_seconds'] = dict(engine=loaded - started, http_ready=time.perf_counter() - started)
        report['cache_before'] = cache_receipt(engine)
        report['source'] = module_identity()
        report['probe_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

        def exchange(label, payload=None, path='/v1/chat/completions'):
            if payload is not None:
                value = app.prepare(payload, True)
                with record_lock:
                    prepared[label] = dict(prompt=value.prompt, max_tokens=value.max_tokens, thinking=value.thinking,
                                           sampling=None if value.sampling is None else asdict(value.sampling))
            headers = {'Content-Type': 'application/json', 'X-Client-Request-Id': 'probe-' + label}
            request = urllib.request.Request(base_url + path,
                       data=None if payload is None else json.dumps(payload).encode(), headers=headers)
            then = time.perf_counter()
            try:
                response = urllib.request.urlopen(request, timeout=1800)
            except urllib.error.HTTPError as error:
                response = error
            with response:
                raw = response.read(MAX_REPLY_BYTES + 1)
                row = dict(path=path, request=payload, status=response.status, headers=dict(response.headers),
                           elapsed_s=time.perf_counter() - then, raw=raw.decode('utf-8'))
            with record_lock:
                records[label] = row
            if len(raw) > MAX_REPLY_BYTES:
                raise ValueError('HTTP reply exceeded the complete-response capture bound')
            result = json.loads(raw)
            row['response'] = result
            if row['status'] != 200:
                raise RuntimeError(f'HTTP probe {label} failed with status{row["status"]}: {result}')
            row['cache_after'] = cache_receipt(engine)
            if payload is not None:
                row['token_ids'] = reply_tokens(result)
            return row

        health = exchange('health', path='/health')
        if health['response'].get('ok') is not True:
            raise ValueError('Loopback health did not confirm readiness')
        greedy = body(args.max_tokens)
        cold_warm = [exchange('greedy-' + str(i), greedy) for i in range(args.requests)]
        serial = exchange('greedy-serial', body(args.max_tokens, draft=False))
        reference = serial['token_ids']
        if any(row['token_ids'] != reference for row in cold_warm):
            raise ValueError('Greedy cold/warm/drafted HTTP replies differ from same-engine serial tokens')
        report['comparisons']['greedy_serial_draft_equal'] = True
        seeded_rows = [exchange('seeded-' + str(i), body(min(args.max_tokens, 32), draft=draft, seeded=True))
                       for i, draft in enumerate((True, False, True))]
        if not seeded_rows[0]['token_ids'] == seeded_rows[1]['token_ids'] == seeded_rows[2]['token_ids']:
            raise ValueError('Keyed seeded HTTP serial/drafted/repeated tokens differ')
        report['comparisons']['seeded_serial_draft_repeat_equal'] = True
        if args.concurrent:
            before = time.perf_counter()
            failures = []
            rows = []
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(exchange, 'concurrent-' + str(i), greedy) for i in range(4)]
                for future in futures:
                    try:
                        rows.append(future.result())
                    except BaseException as error:
                        failures.append(error)
            raise_failures(None, failures)
            elapsed = time.perf_counter() - before
            if any(row['token_ids'] != reference for row in rows):
                raise ValueError('Concurrent HTTP tokens differ from serial reference')
            report['comparisons']['concurrent'] = dict(equal=True, elapsed_s=elapsed,
                aggregate_tokens_s=sum(len(row['token_ids']) for row in rows) / elapsed)
        stats = exchange('stats', path='/v1/stats')['response']
        complete = [row['response'] for row in records.values() if row.get('request') is not None]
        validate_stats(stats, complete)
        report['comparisons']['completed_http_requests'] = len(complete)
        report['cache_after'] = cache_receipt(engine)
        report['elapsed_before_cleanup_s'] = time.perf_counter() - started
    except BaseException as error:
        primary = error
    finally:
        try:
            def retired():
                if owners.get('server') is not None and not owners['server'].handlers_drained:
                    raise RuntimeError('HTTP handlers remain after cleanup')
                if loop is not None and loop.thread.is_alive():
                    raise RuntimeError('HTTP loop remains alive after cleanup')
                if engine is not None and getattr(engine, '_closed', True) is not True:
                    raise RuntimeError('Model engine did not confirm closure')
            finish((lambda: _close_serving(owners, None), retired))
        except BaseException as error:
            cleanup_failure = error
        try:
            report['result'] = 'pass' if primary is cleanup_failure is None else 'fail'
            report['errors'] = [{'type': type(error).__name__, 'message': str(error)}
                                for error in (primary, cleanup_failure) if error is not None]
            report['cleanup'] = {'handlers_drained': None if owners.get('server') is None else owners['server'].handlers_drained,
                                 'loop_alive': False if loop is None else loop.thread.is_alive(),
                                 'engine_closed': None if engine is None else getattr(engine, '_closed', None)}
            args.output.write_text(json.dumps(report, indent=2) + '\n')
        except BaseException as error:
            output_failure = error
        raise_failures(primary, [error for error in (cleanup_failure, output_failure) if error is not None])
    print('MODEL-PROBE-PASS', args.output, flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(run(parse()))
