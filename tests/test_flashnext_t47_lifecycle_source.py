"""Actual engine ownership methods with opaque SDKs and real stdlib workers."""
from __future__ import annotations

import ast
import builtins
import dis
from concurrent.futures import Future
from datetime import timedelta
import importlib.util
from pathlib import Path
import sys
import threading
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/families/qwen4_exp/cuda/engine.py'


def cleanup():
    p = ROOT / 'src/tensorfold/cleanup.py'
    spec = importlib.util.spec_from_file_location('opaque_engine_owned_cleanup', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def subject():
    helpers = cleanup()
    tree = ast.parse(SOURCE.read_bytes())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlashNextEngine')
    scope = dict(threading=threading, json=__import__('json'), Path=Path, timedelta=timedelta,
                 DEPTH=4, CONFIDENCE=.5, MAX_DEPTH=15, KEEP=8, KEEP_SERIAL=4,
                 finish=helpers.finish, rollback=helpers.rollback, raise_failures=helpers.raise_failures,
                 __package__='tensorfold.families.qwen4_exp.cuda')
    exec(compile(ast.fix_missing_locations(ast.Module([ast.ImportFrom('__future__', [ast.alias('annotations')], 0), cls], [])), str(SOURCE), 'exec'), scope)
    return scope['FlashNextEngine']


def roots(error):
    found, pending = set(), [error]
    while pending:
        value = pending.pop()
        if id(value) in found:
            continue
        found.add(id(value))
        pending.extend(v for v in (BaseException.__cause__.__get__(value), BaseException.__context__.__get__(value)) if v is not None)
        if isinstance(value, BaseExceptionGroup):
            pending.extend(value.exceptions)
    return found


class Controls(unittest.TestCase):
    def setUp(self):
        original = builtins.__import__
        def guard(name, *args, **kwargs):
            if name.split('.')[0] in {'torch', 'triton', 'numpy', 'mlx', 'tensorfold', 'cuda', 'cupy', 'ctypes'}:
                raise AssertionError('actual SDK/native imports forbidden: '+name)
            return original(name, *args, **kwargs)
        owner = patch.object(builtins, '__import__', guard)
        owner.start()
        self.addCleanup(owner.stop)
        self.engine = subject()

    def test_constructor_prepublication_open_failure_abort_retains_native_fields(self):
        primary, native, context = KeyboardInterrupt('open'), OSError('native prior'), EOFError('prior context')
        BaseException.__cause__.__set__(primary, native)
        BaseException.__context__.__set__(primary, context)
        primary.__notes__ = 1
        events = []
        def initialize(engine, *args, **kwargs):
            self.assertEqual(engine._startup_reads, [])
            self.assertIsNone(engine.comm)
            engine.comm = NS(close=lambda **kwargs: events.append(('abort', kwargs['abort'])))
            raise primary
        self.engine._initialize = initialize
        try:
            self.engine(Path('opaque-checkpoint'))
        except BaseException as error:
            self.assertIs(error, primary)
            self.assertIn(id(native), roots(error))
            self.assertIn(id(context), roots(error))
        else:
            self.fail('failed startup accepted')
        self.assertEqual(events, [('abort', True)])

    def test_complete_actual_initialize_protects_open_barrier_admit_settings_and_load(self):
        ordinary = builtins.__import__
        for failed_stage in ('open', 'barrier', 'admit', 'settings', 'load'):
            with self.subTest(stage=failed_stage):
                primary, events, owners = KeyboardInterrupt(failed_stage), [], []
                cls = subject()
                full = cls._initialize
                def initialized(owner, *args, **kwargs):
                    owners.append(owner)
                    return full(owner, *args, **kwargs)
                cls._initialize = initialized
                def stage(name):
                    events.append(name)
                    if name == failed_stage:
                        raise primary
                class Comm:
                    def open(owner, *args):
                        self.assertIs(owners[0].comm, owner)
                        stage('open')
                    def barrier(owner):
                        stage('barrier')
                    def close(owner, *, abort):
                        events.append(('close', abort))
                    def all_gather(owner, *args):
                        self.fail('unexpected native gather in opaque startup fixture')
                def admitted(*args, **kwargs):
                    stage('admit')
                    return {'cache_slots': 32, 'budget_bytes': 100000, 'total_bytes_estimate': 1000}
                def loaded(*args, **kwargs):
                    self.assertIs(kwargs['table_reads'], owners[0]._startup_reads)
                    stage('load')
                    self.fail('failure fixture reached actual payload/model setup')
                cls._same_settings = lambda *args: stage('settings')
                side = NS(bits=16, group=1, scale_bytes=0, codec=None)
                pair = NS(key=side, value=side, symmetric=True, row_bytes=lambda *args: 128)
                rope = NS(rope_type='default', native_context=8192, context_limit=8192, metadata=lambda: {})
                text = dict(vocab_size=128, hidden_size=64, num_attention_heads=1, num_key_value_heads=2, head_dim=64)
                providers = {
                    '': NS(rope_parameters=lambda *args: rope),
                    'kv_formats': NS(get_pair=lambda *args: pair),
                    'torch': NS(cuda=NS(set_device=lambda value: None)),
                    'exl3_pack': NS(admission=lambda value: value, extra_files=lambda *args: (), is_exl3=lambda *args: False),
                    'tensorfold.families': NS(quant_method=lambda value: None, read_config=lambda *args: {}),
                    'decode': NS(Engine=object), 'prompt_plan': NS(choose=lambda *args, **kwargs: (16, 0)),
                    'weights': NS(draft_token_ids=lambda *args: self.fail('depth0 draft allocation'), load=loaded),
                    'tensorfold.cuda.capacity': NS(admit=admitted, config=lambda *args: text, gather_ints=lambda *args: None),
                    'tensorfold.cuda.geometry': NS(PREFILL_ROWS=128, indexed_prefill_rows=lambda: 16,
                        gdn_geometry=lambda *args, **kwargs: None, indexed_stream_geometry=lambda *args, **kwargs: None,
                        indexed_weights=lambda *args, **kwargs: None),
                    'tensorfold.cuda.comm': NS(NCCL=Comm),
                    'tensorfold.vision.qwen_cuda': NS(capacity_geometry=lambda value, *args: value,
                        weight_transform=lambda value, *args: value),
                }
                def imported(name, *args, **kwargs):
                    if name in providers:
                        return providers[name]
                    return ordinary(name, *args, **kwargs)
                with patch.object(builtins, '__import__', imported):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        cls(Path('opaque-checkpoint'), depth=0, tp=2, master='host', prefetch=True)
                self.assertIs(caught.exception, primary)
                phases = ['open', 'barrier', 'admit', 'settings', 'load']
                self.assertEqual(events, phases[:phases.index(failed_stage)+1]+[('close', True)])
                self.assertEqual(owners[0]._startup_reads, [])
                self.assertIsNone(owners[0].comm)
                self.assertTrue(owners[0]._closed)

    def test_constructor_drain_waits_accepted_read_before_comm_release(self):
        read, primary = Future(), KeyboardInterrupt('load')
        read.set_running_or_notify_cancel()
        entered, completed = threading.Event(), threading.Event()
        events, failures = [], []
        def initialize(engine, *args, **kwargs):
            engine.comm = NS(close=lambda **kwargs: events.append(('abort', read.done(), kwargs['abort'])))
            engine._startup_reads.append(read)
            entered.set()
            raise primary
        self.engine._initialize = initialize
        def construct():
            try:
                self.engine(Path('opaque-checkpoint'))
            except BaseException as error:
                failures.append(error)
            finally:
                completed.set()
        caller = threading.Thread(target=construct)
        caller.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertFalse(completed.wait(.05))
            self.assertEqual(events, [])
        finally:
            read.set_result(None)
            caller.join(3)
        self.assertFalse(caller.is_alive())
        self.assertEqual(failures, [primary])
        self.assertEqual(events, [('abort', True, True)])

    def test_all_terminal_read_statuses_observed_and_primary_cleanup_retained(self):
        primary, first, second = KeyboardInterrupt('load'), OSError('read1'), EOFError('read2')
        observed, events = [], []
        class Read(Future):
            def result(owner, *args, **kwargs):
                observed.append(owner)
                return super().result(*args, **kwargs)
        reads = [Read(), Read()]
        reads[0].set_exception(first)
        reads[1].set_exception(second)
        def initialize(engine, *args, **kwargs):
            engine.comm = NS(close=lambda **kwargs: events.append(kwargs['abort']))
            engine._startup_reads.extend(reads)
            raise primary
        self.engine._initialize = initialize
        try:
            self.engine(Path('opaque-checkpoint'))
        except BaseException as error:
            self.assertIs(error, primary)
            self.assertTrue({id(first), id(second)} <= roots(error))
        else:
            self.fail('status failure swallowed')
        self.assertEqual(observed, reads)
        self.assertEqual(events, [True])

    def test_interrupted_pending_read_retains_owner_and_never_releases_comm(self):
        primary, interrupt = KeyboardInterrupt('load'), KeyboardInterrupt('read wait')
        events = []
        class Pending(Future):
            def result(owner, *args, **kwargs):
                raise interrupt
        read = Pending()
        def initialize(engine, *args, **kwargs):
            engine.comm = NS(close=lambda **kwargs: events.append('unsafe release'))
            engine._startup_reads.append(read)
            raise primary
        self.engine._initialize = initialize
        try:
            self.engine(Path('opaque-checkpoint'))
        except BaseException as error:
            self.assertIs(error, primary)
            retained = error.__dict__['_tensorfold_retained_owners'][0]
            self.assertEqual(retained._startup_reads, [read])
            self.assertIsNotNone(retained.comm)
            self.assertIn(id(interrupt), roots(error))
        else:
            self.fail('pending read owner forgotten')
        self.assertEqual(events, [])

    def test_partial_scheduler_start_joins_accepted_actor_before_abort(self):
        primary = KeyboardInterrupt('start accepted then failed')
        stop, joined, started = threading.Event(), threading.Event(), threading.Event()
        events = []
        def actor():
            started.set()
            stop.wait(3)
            joined.set()
        worker = threading.Thread(target=actor)
        def close_scheduler():
            stop.set()
            worker.join(3)
            if worker.is_alive():
                raise AssertionError('actor fixture failed to join')
            events.append('joined')
        def initialize(engine, *args, **kwargs):
            engine.scheduler = NS(thread=worker, close=close_scheduler)
            engine.comm = NS(close=lambda **kwargs: events.append(('abort', joined.is_set(), kwargs['abort'])))
            worker.start()
            self.assertTrue(started.wait(3))
            raise primary
        self.engine._initialize = initialize
        try:
            with self.assertRaises(KeyboardInterrupt) as caught:
                self.engine(Path('opaque-checkpoint'))
            self.assertIs(caught.exception, primary)
        finally:
            stop.set()
            worker.join(3)
        self.assertEqual(events, ['joined', ('abort', True, True)])

    def test_failed_comm_retirement_preserves_owner_and_dependent_resources(self):
        primary, retirement = KeyboardInterrupt('initialize'), OSError('native abort')
        events = []
        def release(**kwargs):
            events.append(kwargs['abort'])
            raise retirement
        communicator = NS(close=release)
        def initialize(engine, *args, **kwargs):
            engine.comm = communicator
            engine.w = NS(device='opaque', meta={'expert_cache': NS(close=lambda: self.fail('dependent release'))})
            raise primary
        self.engine._initialize = initialize
        try:
            self.engine(Path('opaque-checkpoint'))
        except BaseException as error:
            self.assertIs(error, primary)
            owner = error.__dict__['_tensorfold_retained_owners'][0]
            self.assertIs(owner.comm, communicator)
            self.assertIsNotNone(owner.w)
            self.assertFalse(owner._closed)
            self.assertIn(id(retirement), roots(error))
        else:
            self.fail('unresolved retirement accepted')
        self.assertEqual(events, [True])

    def test_close_drains_accepted_failed_call_before_abort_and_rejects_nonbool_policy(self):
        self.engine._initialize = lambda *args, **kwargs: None
        engine = self.engine(Path('opaque-checkpoint'))
        events = []
        engine.comm = NS(close=lambda **kwargs: events.append(kwargs['abort']))
        lock = threading.Lock()
        lock.acquire()
        with engine._lifecycle:
            engine._calls[lock] = -1
        completed, failures = threading.Event(), []
        def close():
            try:
                engine.close()
            except BaseException as error:
                failures.append(error)
            finally:
                completed.set()
        caller = threading.Thread(target=close)
        caller.start()
        try:
            self.assertFalse(completed.wait(.05))
            self.assertEqual(events, [])
            engine._end_request(lock, failed=True)
        finally:
            lock.release()
            caller.join(3)
        self.assertFalse(caller.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(events, [True])
        with self.assertRaises(ValueError):
            engine.close(abort=1)

    def test_receive_only_exact_acknowledged_timeout_retries_network_closes(self):
        class StoreError(Exception):
            pass
        class NetworkError(Exception):
            pass
        p = ROOT / 'src/tensorfold/cuda/store_wait.py'
        spec = importlib.util.spec_from_file_location('opaque_store_timeout', p)
        timeout = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(timeout)
        ordinary = builtins.__import__
        def imports(name, *args, **kwargs):
            if name == 'torch.distributed':
                return NS(DistNetworkError=NetworkError, DistStoreError=StoreError)
            if name == 'tensorfold.cuda.store_wait':
                return timeout
            return ordinary(name, *args, **kwargs)
        engine = self.engine.__new__(self.engine)
        engine._lifecycle = threading.Condition()
        engine._closing = False
        engine.served = 7
        engine._key = lambda value: 'request-'+str(value)
        engine._unpack = lambda text: ('unchanged', text)
        for failure, expected_calls in ((StoreError('wait timeout after 3600000ms, keys: /request-7'), 2),
                                        (StoreError('structural store failure'), 1), (ValueError('protocol'), 1),
                                        (NetworkError('rank0 departed'), 1)):
            calls = []
            def wait(keys, duration):
                calls.append((keys, duration))
                if len(calls) == 1:
                    raise failure
            engine.comm = NS(store=NS(wait=wait, get=lambda key: b'original', delete_key=lambda key: None))
            with patch.object(builtins, '__import__', imports):
                if expected_calls == 2:
                    self.assertEqual(engine._receive(), ('unchanged', 'original'))
                elif isinstance(failure, NetworkError):
                    self.assertIsNone(engine._receive())
                else:
                    with self.assertRaises(type(failure)) as caught:
                        engine._receive()
                    self.assertIs(caught.exception, failure)
            self.assertEqual(len(calls), expected_calls)
            self.assertEqual(calls[0], (['request-7'], timedelta(hours=1)))

    def test_idle_follow_close_refuses_fast_retains_borrow_then_peer_stop_and_retry(self):
        entered, released = threading.Event(), threading.Event()
        events, failures = [], []
        class Store:
            def wait(owner, keys, timeout):
                self.assertEqual(timeout, timedelta(hours=1))
                events.append('wait-enter')
                entered.set()
                if not released.wait(3):
                    raise AssertionError('store fixture deadline')
                events.append('wait-retire')
            def get(owner, key):
                return b'peer-stop'
            def delete_key(owner, key):
                events.append('delete')
        communicator = NS(store=Store(), close=lambda **kwargs: events.append(('close', kwargs['abort'])))
        def initialized(owner, *args, **kwargs):
            owner.tp, owner.rank, owner.served = 2, 1, 0
            owner.comm = communicator
        self.engine._initialize = initialized
        engine = self.engine(Path('opaque-checkpoint'))
        engine._unpack = lambda text: None
        ordinary = builtins.__import__
        def imports(name, *args, **kwargs):
            if name == 'torch.distributed':
                return NS(DistNetworkError=type('Network', (Exception,), {}))
            if name == 'tensorfold.cuda.store_wait':
                return NS(timed_out=lambda *args: False)
            return ordinary(name, *args, **kwargs)
        def follow():
            try:
                engine.follow()
            except BaseException as error:
                failures.append(error)
        with patch.object(builtins, '__import__', imports):
            worker = threading.Thread(target=follow)
            worker.start()
            try:
                self.assertTrue(entered.wait(3))
                self.assertTrue(engine._calls)
                with self.assertRaisesRegex(RuntimeError, 'store receive'):
                    engine.close()
                self.assertTrue(engine._closing)
                self.assertFalse(engine._closed)
                self.assertIs(engine.comm, communicator)
                self.assertEqual(events, ['wait-enter'])
            finally:
                released.set()
                worker.join(3)
        self.assertFalse(worker.is_alive())
        self.assertEqual(failures, [])
        self.assertFalse(engine._receiving)
        self.assertEqual(engine._calls, {})
        engine.close()
        self.assertTrue(engine._closed)
        self.assertEqual(events, ['wait-enter', 'wait-retire', 'delete', ('close', False)])

    def test_receive_acknowledged_timeout_after_closing_stops_without_new_wait(self):
        self.engine._initialize = lambda *args, **kwargs: None
        engine = self.engine(Path('opaque-checkpoint'))
        engine.served = 0
        class StoreError(Exception):
            pass
        status = StoreError('acknowledged only')
        calls = []
        def wait(keys, interval):
            calls.append((keys, interval))
            with engine._lifecycle:
                engine._closing = True
            raise status
        engine.comm = NS(store=NS(wait=wait))
        ordinary = builtins.__import__
        def imports(name, *args, **kwargs):
            if name == 'torch.distributed':
                return NS(DistNetworkError=type('Network', (Exception,), {}))
            if name == 'tensorfold.cuda.store_wait':
                return NS(timed_out=lambda error, keys, duration: error is status and duration == timedelta(hours=1))
            return ordinary(name, *args, **kwargs)
        with patch.object(builtins, '__import__', imports):
            self.assertIsNone(engine._receive())
        self.assertEqual(len(calls), 1)

    def test_shutdown_store_borrow_drains_before_close_without_duplicate_stop(self):
        entered, released = threading.Event(), threading.Event()
        events, failures = [], []
        def send(key, body):
            self.assertEqual((key, body), ('tensorfold/flashnext/request/0', '{"stop": true}'))
            entered.set()
            if not released.wait(3):
                raise AssertionError('shutdown fixture deadline')
            events.append('sent-stop')
        def initialized(owner, *args, **kwargs):
            owner.tp, owner.rank, owner.served = 2, 0, 0
            owner.comm = NS(store=NS(set=send), close=lambda **kwargs: events.append(('close', kwargs['abort'])))
        self.engine._initialize = initialized
        engine = self.engine(Path('opaque-checkpoint'))
        def shutdown():
            try:
                engine.shutdown()
            except BaseException as error:
                failures.append(error)
        def close():
            try:
                engine.close()
            except BaseException as error:
                failures.append(error)
        worker, closer = threading.Thread(target=shutdown), threading.Thread(target=close)
        worker.start()
        try:
            self.assertTrue(entered.wait(3))
            self.assertTrue(engine._calls)
            closer.start()
            self.assertEqual(events, [])
        finally:
            released.set()
            worker.join(3)
            if closer.ident is not None:
                closer.join(3)
        self.assertFalse(worker.is_alive() or closer.is_alive())
        self.assertEqual(failures, [])
        self.assertEqual(events, ['sent-stop', ('close', False)])
        self.assertTrue(engine._closed)

    def test_rank_zero_normal_close_sends_existing_frame_before_comm_destroy(self):
        events = []
        def initialized(owner, *args, **kwargs):
            owner.tp, owner.rank, owner.served = 2, 0, 7
            owner.comm = NS(store=NS(set=lambda key, body: events.append(('stop', key, body))),
                            close=lambda **kwargs: events.append(('close', kwargs['abort'])))
        self.engine._initialize = initialized
        engine = self.engine(Path('opaque-checkpoint'))
        engine.close()
        self.assertEqual(events, [('stop', 'tensorfold/flashnext/request/7', '{"stop": true}'), ('close', False)])
        engine.close()
        self.assertEqual(len(events), 2)

    def test_follow_structural_protocol_failures_abort_and_network_departure_remains_graceful(self):
        class NetworkError(Exception):
            pass
        ordinary = builtins.__import__
        def imports(name, *args, **kwargs):
            if name == 'torch.distributed':
                return NS(DistNetworkError=NetworkError)
            if name == 'tensorfold.cuda.store_wait':
                return NS(timed_out=lambda *args: False)
            return ordinary(name, *args, **kwargs)
        for phase, primary, wanted_abort in (('wait', OSError('structural store'), True),
                                              ('unpack', ValueError('protocol'), True),
                                              ('wait', NetworkError('rank0 departed'), True)):
            events = []
            cause, context = EOFError('prior cause'), LookupError('prior context')
            BaseException.__cause__.__set__(primary, cause)
            BaseException.__context__.__set__(primary, context)
            def wait(*args):
                if phase == 'wait':
                    raise primary
            def unpack(text):
                raise primary
            def initialized(owner, *args, **kwargs):
                owner.tp, owner.rank, owner.served = 2, 1, 0
                owner.comm = NS(store=NS(wait=wait, get=lambda key: b'payload', delete_key=lambda key: None),
                                close=lambda **kwargs: events.append(kwargs['abort']))
            self.engine._initialize = initialized
            engine = self.engine(Path('opaque-checkpoint'))
            engine._unpack = unpack
            with patch.object(builtins, '__import__', imports):
                if not isinstance(primary, NetworkError):
                    try:
                        engine.follow()
                    except BaseException as error:
                        self.assertIs(error, primary)
                        self.assertTrue({id(cause), id(context)} <= roots(error))
                    else:
                        self.fail('store/protocol failure swallowed')
                else:
                    self.assertIsNone(engine.follow())
            self.assertFalse(engine._receiving)
            self.assertEqual(engine._calls, {})
            self.assertEqual(engine._abort_comm, wanted_abort)
            engine.close()
            self.assertEqual(events, [wanted_abort])

    def test_actual_unpack_json_failure_is_an_owned_follow_failure(self):
        import json
        events = []
        def initialized(owner, *args, **kwargs):
            owner.tp, owner.rank, owner.served = 2, 1, 0
            owner.comm = NS(store=NS(wait=lambda *args: None, get=lambda key: b'not-json', delete_key=lambda key: None),
                            close=lambda **kwargs: events.append(kwargs['abort']))
        self.engine._initialize = initialized
        engine = self.engine(Path('opaque-checkpoint'))
        ordinary = builtins.__import__
        def imports(name, *args, **kwargs):
            if name == 'torch.distributed':
                return NS(DistNetworkError=type('Network', (Exception,), {}))
            if name == 'tensorfold.cuda.store_wait':
                return NS(timed_out=lambda *args: False)
            if name == 'tensorfold.engine.exact_sampling':
                return NS(Sampling=object)
            return ordinary(name, *args, **kwargs)
        with patch.object(builtins, '__import__', imports):
            with self.assertRaises(json.JSONDecodeError):
                engine.follow()
        self.assertTrue(engine._abort_comm)
        self.assertEqual(engine._calls, {})
        self.assertFalse(engine._receiving)
        engine.close()
        self.assertEqual(events, [True])

    def test_condition_exit_cancellation_after_receive_publication_retires_its_owner(self):
        primary, cause, context = KeyboardInterrupt('publication exit'), EOFError('prior cause'), OSError('prior context')
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        events = []
        def initialized(owner, *args, **kwargs):
            owner.tp, owner.rank, owner.served = 2, 1, 0
            owner.comm = NS(close=lambda **kwargs: events.append(kwargs['abort']))
        self.engine._initialize = initialized
        engine = self.engine(Path('opaque-checkpoint'))
        raw = engine._lifecycle
        class ConditionExit:
            raised = False
            def __enter__(owner):
                return raw.__enter__()
            def __exit__(owner, *args):
                result = raw.__exit__(*args)
                if engine._receiving and not owner.raised:
                    owner.raised = True
                    raise primary
                return result
            def __getattr__(owner, name):
                return getattr(raw, name)
        engine._lifecycle = ConditionExit()
        engine._receive = lambda: self.fail('receive called after failed condition exit')
        try:
            engine.follow()
        except BaseException as error:
            self.assertIs(error, primary)
            self.assertTrue({id(cause), id(context)} <= roots(error))
        else:
            self.fail('publication failure swallowed')
        self.assertEqual(engine._calls, {})
        self.assertFalse(engine._receiving)
        self.assertTrue(engine._abort_comm)
        engine.close()
        self.assertEqual(events, [True])

    def test_actual_condition_exit_call_opcode_cancellation_replays_metal_witness(self):
        primary, events = KeyboardInterrupt('actual condition exitCALL'), []
        def initialized(owner, *args, **kwargs):
            owner.tp, owner.rank, owner.served = 2, 1, 0
            owner.comm = NS(close=lambda **kwargs: events.append(kwargs['abort']))
        self.engine._initialize = initialized
        engine = self.engine(Path('opaque-checkpoint'))
        engine._receive = lambda: self.fail('receive entered after cancellation')
        code = self.engine.follow.__code__
        instructions = list(dis.get_instructions(code))
        publish = next(i for i,n in enumerate(instructions) if n.opname == 'STORE_ATTR' and n.argval == '_receiving')
        call = next(n.offset for n in instructions[publish+1:] if n.opname == 'CALL')
        triggered = []
        def trace(frame, event, arg):
            if frame.f_code is code:
                frame.f_trace_opcodes = True
                if event == 'opcode' and frame.f_lasti == call:
                    self.assertTrue(engine._receiving.locked())
                    triggered.append(call)
                    raise primary
            return trace
        previous = sys.gettrace()
        current_frame = sys._getframe()
        previous_opcodes = current_frame.f_trace_opcodes
        try:
            current_frame.f_trace_opcodes = True
            sys.settrace(trace)
            try:
                engine.follow()
            except BaseException as error:
                self.assertIs(error, primary)
            else:
                self.fail('actual opcode cancellation swallowed')
        finally:
            sys.settrace(previous)
            current_frame.f_trace_opcodes = previous_opcodes
        self.assertEqual(triggered, [call])
        self.assertEqual(engine._calls, {})
        self.assertFalse(engine._receiving)
        self.assertTrue(engine._abort_comm)
        engine.close()
        self.assertEqual(events, [True])

    def test_prune_retired_native_receive_lock_clears_only_its_marker(self):
        self.engine._initialize = lambda *args, **kwargs: None
        engine = self.engine(Path('opaque-checkpoint'))
        abandoned = threading.Lock()
        engine._calls[abandoned] = -1
        engine._receiving = abandoned
        with engine._lifecycle:
            engine._prune_requests()
        self.assertEqual(engine._calls, {})
        self.assertFalse(engine._receiving)
        self.assertTrue(engine._abort_comm)
        active = threading.Lock()
        active.acquire()
        rejected = threading.Lock()
        engine._calls[active] = -1
        engine._receiving = active
        try:
            engine._end_request(rejected, failed=True)
            self.assertIs(engine._receiving, active)
            self.assertIn(active, engine._calls)
        finally:
            engine._end_request(active)
            active.release()

    def test_rejected_overlapping_follow_cannot_clear_live_receive_marker(self):
        entered, released = threading.Event(), threading.Event()
        outcomes = []
        self.engine._initialize = lambda *args, **kwargs: None
        engine = self.engine(Path('opaque-checkpoint'))
        def receive():
            entered.set()
            if not released.wait(3):
                raise AssertionError('receive fixture deadline')
            return None
        engine._receive = receive
        def follow():
            try:
                outcomes.append(engine.follow())
            except BaseException as error:
                outcomes.append(error)
        caller = threading.Thread(target=follow)
        caller.start()
        try:
            self.assertTrue(entered.wait(3))
            marker = engine._receiving
            self.assertTrue(marker.locked())
            with self.assertRaisesRegex(RuntimeError, 'active request'):
                engine.follow()
            self.assertIs(engine._receiving, marker)
            self.assertTrue(marker.locked())
            self.assertIn(marker, engine._calls)
            self.assertFalse(engine._abort_comm)
            with self.assertRaisesRegex(RuntimeError, 'store receive'):
                engine.close()
        finally:
            released.set()
            caller.join(3)
        self.assertFalse(caller.is_alive())
        self.assertEqual(outcomes, [None])
        self.assertFalse(engine._receiving)
        self.assertEqual(engine._calls, {})
        engine.close()



if __name__ == '__main__':
    unittest.main()
