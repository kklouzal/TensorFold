"""Owned tool callbacks cannot retire resources on a damaged Thread.join.

Pure stdlib real threads/processes and opaque source-only qualifier providers;
no Torch/NumPy/MLX/Triton imports or model/container/GPU execution.
"""
import argparse
import ast
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import shared_prefix_load as shared
from tools import worker_lifetime
from tools.worker_lifetime import Task, drain, raise_failures

REPO = Path(__file__).resolve().parents[1]


class Tasks(unittest.TestCase):
    def test_every_actual_callback_finishes_and_child_errors_observed(self):
        errors = [ValueError('first'),OSError('second'),RuntimeError('third')]
        complete = []
        tasks = []
        for index,error in enumerate(errors):
            def run(index=index,error=error):
                complete.append(index)
                raise error
            task = Task(run)
            tasks.append(task)
            task.start()
        with self.assertRaises(ValueError) as got:
            drain(tasks)
        self.assertIs(got.exception,errors[0])
        self.assertEqual(sorted(complete),[0,1,2])
        self.assertTrue(all(task.done.is_set() for task in tasks))
        self.assertEqual(got.exception.__cause__.exceptions,tuple(errors[1:]))

    def test_prior_native_cause_and_all_retired_task_failures_survive_malformed_notes(self):
        primary, native = KeyboardInterrupt('caller'), ValueError('native cause')
        primary.__cause__ = native
        primary.__notes__ = 123
        failures = [OSError('first'), RuntimeError('second'), EOFError('third')]
        tasks = []
        for error in failures:
            def work(error=error):
                raise error
            task = Task(work)
            task.start()
            tasks.append(task)
        with self.assertRaises(KeyboardInterrupt) as caught:
            drain(tasks, primary)
        self.assertIs(caught.exception, primary)
        self.assertTrue(all(task.done.is_set() for task in tasks))
        self.assertEqual(primary.__cause__.exceptions[:4], (native, *failures))
        self.assertIsInstance(primary.__cause__.exceptions[-1], TypeError)

    def test_every_distinct_cleanup_status_retained_with_bounded_diagnostics(self):
        primary, native = RuntimeError('caller'), OSError('native')
        primary.__cause__ = native
        errors = [ValueError(str(i)) for i in range(20)]
        with self.assertRaises(RuntimeError) as caught:
            raise_failures(primary, [primary, *errors, errors[0]])
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__cause__.exceptions, (native, *errors))
        self.assertEqual(len(primary.__notes__), 9)

    def test_same_primary_annotation_fault_never_creates_selfcause(self):
        class Opaque(KeyboardInterrupt):
            def __getattribute__(self, name):
                if name == '__notes__':
                    raise self
                return super().__getattribute__(name)
        primary, other = Opaque(), OSError()
        with self.assertRaises(Opaque) as caught:
            raise_failures(primary, [primary, other])
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__cause__.exceptions, (other,))

    def test_foreign_metaclass_name_hook_is_bypassed_and_note_length_bounded(self):
        reads = []
        class Names(type):
            def __getattribute__(cls, name):
                if name == '__name__':
                    reads.append(name)
                    raise LookupError('foreign class-name hook')
                return super().__getattribute__(name)
        foreign = Names('X' * 200, (OSError,), {})()
        primary = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt) as caught:
            raise_failures(primary, [foreign])
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, foreign)
        self.assertEqual(reads, [])
        self.assertEqual(primary.__notes__, ['owned task cleanup also failed (' + 'X' * 64 + ')'])

    def test_notes_hook_cannot_erase_the_captured_native_cause(self):
        class Erase(KeyboardInterrupt):
            def __getattribute__(self, name):
                if name == '__notes__':
                    BaseException.__cause__.__set__(self, None)
                    BaseException.__context__.__set__(self, None)
                return super().__getattribute__(name)
        primary, native, context, cleanup = Erase(), OSError(), ArithmeticError(), ValueError()
        primary.__cause__, primary.__context__ = native, context
        with self.assertRaises(Erase) as caught:
            raise_failures(primary, [cleanup])
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__cause__.exceptions, (native, context, cleanup))

    def test_group_exhaustion_retains_original_primary_native_fields_and_statuses(self):
        primary, native, context = KeyboardInterrupt(), LookupError(), EOFError()
        errors, allocation = [OSError(), ValueError()], MemoryError('group allocation')
        primary.__cause__, primary.__context__ = native, context
        def fail(*args):
            raise allocation
        with patch.object(worker_lifetime, 'BaseExceptionGroup', fail, create=True):
            try:
                raise_failures(primary, errors)
            except BaseException as actual:
                self.assertIs(actual, primary)
                self.assertIs(actual.__cause__, allocation)
                trace = actual.__traceback__
                frames = []
                while trace is not None:
                    frames.append(trace.tb_frame)
                    trace = trace.tb_next
                trace = allocation.__traceback__
                while trace is not None:
                    frames.append(trace.tb_frame)
                    trace = trace.tb_next
                owned = next(f.f_locals for f in frames if f.f_code.co_name == 'raise_failures')
                self.assertIs(owned['native_cause'], native)
                self.assertIs(owned['native_context'], context)
                self.assertIs(owned['errors'], errors)
                core = next(f.f_locals for f in frames if f.f_code.co_name == '_raise_failures')
                self.assertEqual(core['causes'], [native, context, *errors])
            else:
                self.fail('group exhaustion discarded failure')

    def test_cold_diagnostic_exhaustion_retains_selected_primary_and_all_statuses(self):
        primary, native, context = KeyboardInterrupt(), LookupError(), EOFError()
        errors, allocation = [primary, OSError(), ValueError()], MemoryError('label allocation')
        primary.__cause__, primary.__context__ = native, context
        def fail(*args):
            raise allocation
        with patch.object(worker_lifetime, 'str', SimpleNamespace(__getitem__=fail), create=True):
            try:
                raise_failures(None, errors)
            except BaseException as actual:
                self.assertIs(actual, primary)
                self.assertIs(actual.__cause__, allocation)
                trace = actual.__traceback__
                while trace is not None and trace.tb_frame.f_code.co_name != 'raise_failures':
                    trace = trace.tb_next
                self.assertIsNotNone(trace)
                owned = trace.tb_frame.f_locals
                self.assertIs(owned['native_cause'], native)
                self.assertIs(owned['native_context'], context)
                self.assertIs(owned['errors'], errors)
            else:
                self.fail('cold allocation discarded primary')

    def test_start_interrupt_before_or_after_spawn_never_enters_callback(self):
        for spawned in (False,True):
            with self.subTest(spawned=spawned):
                touched = []
                task = Task(lambda:touched.append('unsafe resource use'))
                native = task.thread.start
                primary = KeyboardInterrupt('owned startup interruption')
                def start():
                    if spawned:
                        native()
                    raise primary
                task.thread.start = start
                with self.assertRaises(KeyboardInterrupt) as got:
                    task.start()
                self.assertIs(got.exception,primary)
                drain([task])
                if spawned:
                    task.thread.join(timeout=2)
                    self.assertFalse(task.thread.is_alive())
                self.assertTrue(task.done.is_set())
                self.assertTrue(task.start_cancelled)
                self.assertEqual(touched,[])

    def test_real_sigint_join_interruption_journal_drains_before_retirement(self):
        # An isolated child keeps the signal/Thread.join fault off test runner.
        script = r"""
import os,signal,threading,time
from tools.worker_lifetime import Task,drain
for delay in (0.01,0.015,0.02):
 release=threading.Event();entered=threading.Event();complete=[]
 def work():
  entered.set();release.wait();complete.append('finished')
 task=Task(work);task.start();assert entered.wait(2)
 alarm=threading.Timer(delay,lambda:os.kill(os.getpid(),signal.SIGINT));alarm.start()
 primary=None
 try:task.thread.join()
 except KeyboardInterrupt as error:primary=error
 assert primary is not None and not task.done.is_set()
 retire=threading.Timer(0.02,release.set);retire.start()
 try:drain([task],primary)
 except KeyboardInterrupt as error:assert error is primary
 else:raise AssertionError('caller interruption lost')
 assert complete==['finished'] and task.done.is_set()
 alarm.join();retire.join()
print('3 actual SIGINT join interruptions drained')
"""
        child = subprocess.run([sys.executable,'-B','-W','error','-c',script],cwd=REPO,
                               capture_output=True,text=True,timeout=10)
        self.assertEqual(child.returncode,0,child.stdout+child.stderr)
        self.assertIn('3 actual SIGINT',child.stdout)

    def test_invalid_notes_preserve_original_after_all_work(self):
        primary = KeyboardInterrupt('opaque primary')
        primary.__notes__ = 123
        with self.assertRaises(KeyboardInterrupt) as got:
            raise_failures(primary,[OSError('owned cleanup')])
        self.assertIs(got.exception,primary)
        self.assertIsInstance(got.exception.__cause__,BaseExceptionGroup)
        self.assertEqual([type(error) for error in got.exception.__cause__.exceptions],[OSError,TypeError])


class SharedPrefix(unittest.TestCase):
    def run_shared(self,failure=None):
        with tempfile.TemporaryDirectory(prefix='owned-shared-tool-') as directory:
            path = Path(directory)/'prompts.json'
            path.write_text(json.dumps({'items':[{'id':i,'messages':[]} for i in range(4)]}))
            calls = []
            def one(base,model,item,max_tokens,temperature,seed,draft):
                calls.append((item['id'],seed,draft))
                if failure is not None and item['id']==0:
                    raise failure
                stamp = time.perf_counter()
                return {'id':item['id'],'sent':stamp,'first':stamp+.01,'last':stamp+.02,
                        'tokens':2,'error':None,'prompt_tokens':1,'finish':'length'}
            with patch.object(shared,'one',one),patch.object(sys,'argv',['tool','http://owned','model',str(path),
                '--concurrency','2','--seed','7']),redirect_stdout(io.StringIO()) as output:
                if failure is None:
                    shared.main()
                    result = json.loads(output.getvalue())
                    self.assertEqual(result['ok'],4)
                    self.assertEqual(result['requests'],4)
                    self.assertEqual(sorted(calls),[(i,7+i,True) for i in range(4)])
                else:
                    with self.assertRaises(type(failure)) as got:
                        shared.main()
                    self.assertIs(got.exception,failure)
                    self.assertEqual(output.getvalue(),'')
            self.assertFalse(any(thread.name.startswith('shared-prefix-request-') for thread in threading.enumerate()))

    def test_actual_main_collects_every_request_seed_and_no_live_worker(self):
        self.run_shared()

    def test_actual_main_child_failure_is_not_partial_success(self):
        self.run_shared(KeyboardInterrupt('owned request failure'))

    def test_memory_diagnostic_checks_child_status_and_timeout(self):
        calls = []
        primary = subprocess.CalledProcessError(7,['nvidia-smi'])
        def run(*args,**kwargs):
            calls.append(kwargs)
            raise primary
        original = shared.subprocess
        with patch.object(shared,'subprocess',SimpleNamespace(run=run)):
            with self.assertRaises(subprocess.CalledProcessError) as got:
                shared.Memory(()).gpu()
        self.assertIs(got.exception,primary)
        self.assertEqual(calls[0]['timeout'],5)
        self.assertIs(calls[0]['check'],True)
        self.assertIs(shared.subprocess,original)


class Qualifier(unittest.TestCase):
    def run_qualifier(self,bad=None):
        tree = ast.parse((REPO/'tools/qualify_logprobs.py').read_text())
        main = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='main')
        with tempfile.TemporaryDirectory(prefix='owned-probability-tool-') as directory:
            root = Path(directory)
            errors = []
            primary = KeyboardInterrupt('opaque request interruption')
            trace,server_owners = [],[]
            class Engine:
                def close(self):
                    trace.append('engine-closed')
            engine = Engine()
            def app(*args,**kwargs):
                if bad=='app':
                    raise primary
                return object()
            class Server:
                def __init__(self,*args):
                    if bad=='constructor':
                        raise primary
                    self.server_port=123
                    self.stop=threading.Event()
                    self.finished=threading.Event()
                    self.handlers_drained=False
                    server_owners.append(self)
                def serve_forever(self,**kwargs):
                    try:
                        self.stop.wait()
                    finally:
                        self.finished.set()
                def server_close(self):
                    self.stop.set()
                    if bad=='drain':
                        raise OSError('owned handler drain failed')
                    self.handlers_drained=True
                    trace.append('server-closed')
            tasks = []
            def task(target,**kwargs):
                result = Task(target,**kwargs)
                tasks.append(result)
                if bad=='start':
                    native=result.thread.start
                    def start():
                        native()
                        raise primary
                    result.thread.start=start
                return result
            def request(*args):
                trace.append('request')
                if bad=='request':
                    raise primary
                return {'choices':[{}],'measurement':{'sha256':'owned','tokens':64}}
            namespace={'argparse':argparse,'Path':Path,'cuda_engine':lambda *a,**k:engine,'App':app,'Server':Server,
                'make_handler':lambda a:object(),'Task':task,'drain':drain,'raise_failures':raise_failures,
                'request':request,'PROMPTS':['owned prompt']*9,'body':lambda *a,**k:{},'json':json,
                'torch':SimpleNamespace(cuda=SimpleNamespace(reset_peak_memory_stats=lambda:None))}
            exec(compile(ast.Module(body=[main],type_ignores=[]),'<actual-qualifier-main>','exec'),namespace)
            argv=['tool','--model',str(root),'--output',str(root/'out.json'),'--label','owned']
            with patch.object(sys,'argv',argv),redirect_stdout(io.StringIO()):
                if bad:
                    with self.assertRaises((KeyboardInterrupt,OSError)) as got:
                        namespace['main']()
                    errors.append(got.exception)
                    if bad!='drain':
                        self.assertIs(got.exception,primary)
                else:
                    namespace['main']()
                    self.assertEqual(trace.count('request'),19)
            for owned in tasks:
                owned.thread.join(timeout=2)
                self.assertTrue(owned.done.is_set())
                self.assertFalse(owned.thread.is_alive())
            if bad=='drain':
                self.assertNotIn('engine-closed',trace)
                self.assertTrue(errors[0].__notes__)
            else:
                self.assertEqual(trace.count('engine-closed'),1)
                if server_owners:
                    self.assertLess(trace.index('server-closed'),trace.index('engine-closed'))

    def test_actual_main_normal_and_failed_acquisitions_close_owned_engine(self):
        for bad in (None,'app','constructor','start','request'):
            with self.subTest(bad=bad):
                self.run_qualifier(bad)

    def test_failed_handler_drain_retains_model_owner(self):
        self.run_qualifier('drain')

    def test_request_primary_survives_once_connection_close_failure(self):
        tree = ast.parse((REPO/'tools/qualify_logprobs.py').read_text())
        request = next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='request')
        primary = KeyboardInterrupt('owned connection operation')
        closes = []
        class Connection:
            def request(self,*args):raise primary
            def close(self):
                closes.append(True)
                raise OSError('owned connection close failed')
        namespace={'http':SimpleNamespace(client=SimpleNamespace(HTTPConnection=lambda *a,**k:Connection())),
                   'time':time,'json':json,'hashlib':hashlib,'raise_failures':raise_failures}
        exec(compile(ast.Module(body=[request],type_ignores=[]),'<actual-qualifier-request>','exec'),namespace)
        with self.assertRaises(KeyboardInterrupt) as got:
            namespace['request'](123,{})
        self.assertIs(got.exception,primary)
        self.assertIsInstance(got.exception.__cause__,OSError)
        self.assertEqual(closes,[True])


if __name__=='__main__':
    unittest.main()
