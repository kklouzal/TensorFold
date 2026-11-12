"""Private five-window scheduler diagnostic grants and bounded trace storage.

Host bytes are an admission obligation, not a native-profiler RSS ceiling.
The pinned SDK caps in-flight CUPTI activity host buffers only; CPU event storage and
CUPTI device/runtime overhead remain library-managed and require the operator's
whole-container envelope plus measured entry headroom. Disk is enforced while
streaming, before excess bytes reach storage. All results are P3-invalid.
"""
from __future__ import annotations

import errno
import inspect
import json
import os
import re
import signal
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path

if __package__:
    from .worker_lifetime import Task, drain, raise_failures
else:
    from worker_lifetime import Task, drain, raise_failures

SCHEMA='tensorfold.scheduler-five-window-diagnostic-v1'
SCOPE='c1-prefill-c1decode2-c8decode2-v1'
SDK_GIT='05adb434bd30d298eb408a72420b0496ee05f8e3'
BLOCK=4<<20
CHUNK=64<<10
SPEC_LIMIT=4096
KINETO_CONFIG_TEXT='VERBOSE_LOG_LEVEL=1\n'
WINDOWS=('c1-cold-prefill-containing-round','c1-warmed-decode-round-1','c1-warmed-decode-round-2',
         'c8-warmed-decode-round-1','c8-warmed-decode-round-2')
ENTRY_PREFIX=b'SCHEDULER_DIAGNOSTIC_ENTRY '


@dataclass(frozen=True)
class Grants:
    host_obligation_bytes: int
    device_headroom_bytes: int
    disk_bytes: int
    cupti_inflight_buffer_bytes: int
    native_log_bytes: int

    @classmethod
    def parse(cls,value):
        fields={'schema','scope','host_obligation_bytes','device_headroom_bytes','disk_bytes','cupti_inflight_buffer_bytes','native_log_bytes'}
        if type(value)is not dict or set(value)!=fields or value['schema']!=SCHEMA or value['scope']!=SCOPE:
            raise ValueError('versioned five-window diagnostic grants required')
        for key in fields-{'schema','scope'}:
            if type(value[key])is not int or not 0<value[key]<2**63:
                raise ValueError('diagnostic grants must be positive bounded integers')
        result=cls(**{key:value[key]for key in fields-{'schema','scope'}})
        if (result.cupti_inflight_buffer_bytes%BLOCK or result.cupti_inflight_buffer_bytes<BLOCK
                or result.cupti_inflight_buffer_bytes//(1<<20)>2**31-1
                or result.host_obligation_bytes<result.cupti_inflight_buffer_bytes+2*CHUNK):
            raise ValueError('host grant must fund integral4MiB CUPTI activity buffers and bounded IO staging')
        return result

    def as_dict(self):
        return {'schema':SCHEMA,'scope':SCOPE,**vars(self)}

    @property
    def kineto_config_mb(self):
        # SDK count=1+configured_size/4MiB. Account that extra block exactly.
        return (self.cupti_inflight_buffer_bytes-BLOCK)//(1<<20)


def read_grants(path):
    with Path(path).open('rb')as source:data=source.read(SPEC_LIMIT+1)
    if len(data)>SPEC_LIMIT:raise ValueError('scheduler diagnostic DATA exceeds its bound')
    def pairs(items):
        result={}
        for key,value in items:
            if key in result:raise ValueError('duplicate scheduler diagnostic field')
            result[key]=value
        return result
    return Grants.parse(json.loads(data,object_pairs_hook=pairs))


def docker_log_options(grants):
    # Docker's max-size is a rotation target, not an exact byte ceiling: one
    # final record/driver framing can overshoot it. Kept file count is bounded.
    # Export below has a separate hard payload quota, and missing initial-config
    # or any capture marker makes rotated/truncated window evidence ineligible.
    return ['--log-driver','local','--log-opt','max-size='+str(grants.native_log_bytes),
            '--log-opt','max-file=1','--log-opt','compress=false']


def capture_native_log(cid,path,grants,*,timeout=60):
    """Bound the retired container's Docker-log export before excess disk IO."""
    if type(cid)is not str or re.fullmatch('[0-9a-f]{64}',cid)is None:
        raise ValueError('native diagnostic log requires one exact container ID')
    process=subprocess.Popen(['docker','logs',cid],stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    task=None;primary=None;errors=[];used=[0]
    def consume():
        with process.stdout as source,Path(path).open('xb',buffering=0)as output:
            while True:
                block=source.read(min(CHUNK,grants.native_log_bytes-used[0]+1))
                if not block:break
                if len(block)>grants.native_log_bytes-used[0]:raise StorageExhausted()
                view=memoryview(block)
                while view:
                    count=output.write(view)
                    if type(count)is not int or count<=0:raise OSError('native diagnostic log made no write progress')
                    used[0]+=count;view=view[count:]
            os.fsync(output.fileno())
    try:
        task=Task(consume,name='scheduler-native-log-sink');task.start()
        status=process.wait(timeout=timeout)
        if status:raise subprocess.CalledProcessError(status,process.args)
    except BaseException as error:
        primary=error
    finally:
        if process.poll()is None:
            try:process.terminate();process.wait(timeout=5)
            except BaseException as error:
                errors.append(error)
                if process.poll()is None:
                    try:process.kill();process.wait(timeout=5)
                    except BaseException as error:errors.append(error)
        if task is not None and task.started:
            try:drain([task])
            except BaseException as error:errors.append(error)
        else:
            try:process.stdout.close()
            except BaseException as error:errors.append(error)
    raise_failures(primary,errors)
    return {'bytes':used[0],'limit_bytes':grants.native_log_bytes,'rotation_target_bytes':grants.native_log_bytes,
            'driver_scope':'local/max-file1; target plus driver framing/final record, not exact daemon-file byte ceiling'}


class StorageExhausted(OSError):
    def __init__(self):
        super().__init__(errno.ENOSPC,'scheduler diagnostic storage grant exhausted')


class TraceStorage:
    """One exclusive five-window sink; metadata and trace bytes share a quota.

    The exporter writes to its own pipe FD path through the SDK's supported
    Python streaming exporter. A fixed-buffer reader drains to a fresh regular
    partial file, rejects surplus before writing, then closes the pipe. Model
    errors, producer errors and sink/cleanup failures are all retained.
    """
    def __init__(self,directory,limit,*,thread=None):
        self.directory=Path(directory)
        self.limit=limit;self.used=0;self.failed=None
        self._thread=threading.current_thread() if thread is None else thread
        self._files={}
        if type(limit)is not int or limit<=0:raise ValueError('positive disk grant required')
        if not self.directory.is_dir() or any(self.directory.iterdir()):
            raise ValueError('diagnostic storage requires a fresh owned empty directory')
        if signal.getsignal(signal.SIGPIPE)!=signal.SIG_IGN:
            raise RuntimeError('bounded pipe export requires the Python worker SIGPIPE-ignore contract')

    def _path(self,name,*,replace=False):
        if type(name)is not str or not name or Path(name).name!=name:
            raise ValueError('diagnostic output must remain inside its owned directory')
        path=self.directory/name
        if (path.is_symlink() or path.with_name(name+'.partial').exists()
                or path.exists() and (not replace or name not in self._files)):
            raise ValueError('diagnostic output cannot overwrite existing evidence')
        return path

    def _write(self,output,data):
        if len(data)>self.limit-self.used:raise StorageExhausted()
        view=memoryview(data)
        while view:
            count=output.write(view)
            if type(count)is not int or count<=0:raise OSError('diagnostic storage write made no progress')
            self.used+=count;view=view[count:]

    def write_json(self,name,value):
        if self.failed is not None:raise self.failed
        target=self._path(name,replace=True);partial=target.with_name(name+'.partial')
        before=self.used;old=self._files.get(name,0)
        try:
            with partial.open('xb',buffering=0)as output:
                for text in json.JSONEncoder(allow_nan=False,separators=(',',':')).iterencode(value):
                    self._write(output,text.encode('utf-8'))
                self._write(output,b'\n');os.fsync(output.fileno())
            os.replace(partial,target)
            self._files[name]=self.used-before;self.used-=old
        except BaseException as error:
            self.failed=error
            raise

    def export(self,name,profiler):
        if self.failed is not None:raise self.failed
        if threading.current_thread()is not self._thread:raise RuntimeError('diagnostic export belongs to its scheduler thread')
        if 'use_python_export'not in inspect.signature(profiler.export_chrome_trace).parameters:
            raise RuntimeError('SDK lacks the supported streaming exporter; defaultC++ temporary-path export is unsupported')
        target=self._path(name);partial=target.with_name(name+'.partial')
        read_fd,write_fd=os.pipe2(os.O_CLOEXEC)
        task=None;primary=None;errors=[]
        def consume():
            try:
                with os.fdopen(read_fd,'rb',buffering=0)as source,partial.open('xb',buffering=0)as output:
                    while True:
                        block=source.read(min(CHUNK,self.limit-self.used+1))
                        if not block:break
                        self._write(output,block)
                    os.fsync(output.fileno())
            except BaseException as error:
                self.failed=error
                raise
        try:
            task=Task(consume,name='scheduler-trace-disk-sink')
            task.start()
            profiler.export_chrome_trace('/proc/self/fd/'+str(write_fd),use_python_export=True)
        except BaseException as error:
            primary=error
        finally:
            try:os.close(write_fd)
            except BaseException as error:errors.append(error)
            if task is not None and task.started:
                try:drain([task])
                except BaseException as error:errors.append(error)
            else:
                try:os.close(read_fd)
                except BaseException as error:errors.append(error)
        if primary is not None or errors or self.failed is not None:
            if self.failed is not None:errors.append(self.failed)
            failure=primary if primary is not None else errors.pop(0)
            self.failed=failure
            raise_failures(failure,errors)
        os.replace(partial,target)
        size=target.stat().st_size;self._files[name]=size
        return size


def cgroup_memory():
    root=Path('/sys/fs/cgroup')
    if not (root/'memory.max').exists():
        entries=[line[3:]for line in Path('/proc/self/cgroup').read_text().splitlines()if line.startswith('0::')]
        if len(entries)!=1 or '..'in Path(entries[0]).parts:raise RuntimeError('diagnostic cgroup-v2 memory authority unavailable')
        root=root/entries[0].lstrip('/')
    result={}
    for key in ('memory.max','memory.current','memory.peak','memory.events'):
        path=root/key
        if path.exists():result[key]=path.read_text().strip()
    limit=result.get('memory.max')
    if limit is None or not limit.isdecimal() or int(limit)<=0:
        raise RuntimeError('scheduler diagnostic requires the explicit whole-container memory envelope')
    return result


def headroom(engine,grants,*,after=False):
    import torch
    free,total=map(int,torch.cuda.mem_get_info(engine.w.device))
    graphs=getattr(engine.multi,'graphs',None)
    spare=getattr(graphs,'spare',None)
    if not callable(spare):
        raise RuntimeError('scheduler diagnostic requires the existing admitted graph service-spare reader')
    service=spare()
    needed=0 if after else grants.device_headroom_bytes
    if type(service)is not int or service<needed or free<needed:
        raise MemoryError('profiler headroom cannot consume the unchanged device floor/full-window/service workspace')
    group=cgroup_memory()
    return {'free_bytes':free,'total_bytes':total,'service_spare_bytes':service,
            'required_headroom_bytes':needed,'cgroup':group,
            'scope':'fresh minimum-free admission after original floor/service promises; not a reservation or peak proof'}


def check_memory_events(before,after):
    def events(value):
        result={k:int(v)for k,v in (line.split()for line in value.get('cgroup',{}).get('memory.events','').splitlines())}
        if not {'oom','oom_kill'}<=set(result)or any(v<0 for v in result.values()):
            raise ValueError('diagnostic cgroup memory-event evidence is unavailable')
        return result
    first,last=events(before),events(after)
    if any(last.get(key,-1)<value for key,value in first.items()):
        raise ValueError('diagnostic cgroup memory-event counters changed identity/reset')
    if any(last.get(key,0)>first.get(key,0)for key in ('oom','oom_kill','oom_group_kill')):
        raise MemoryError('diagnostic whole-container memory envelope exhausted; trace incomplete')


def prepare_backend(torch,engine,grants):
    """Zero model calls: initialize the CUDA-capable SDK backend before readiness.

    The source-versioned host collector still requires positive base-config
    evidence before every actual capture marker; no sleep or assumed poll race
    is treated as success. Pinned Kineto retains its first cpuOnly selection
    for the process lifetime, so a CPU-only warmup would disable every later
    CUDA capture. Use the same configured CPU+CUDA profiler as serving windows.
    """
    before=headroom(engine,grants)
    with profiler(torch,grants):
        pass
    after=headroom(engine,grants,after=True)
    check_memory_events(before,after)
    return {'activities':['CPU','CUDA'],'entry_headroom':before,'exit_headroom':after,
            'scope':'zero model calls; first process-wide Kineto registration includes CUDA'}


def capture_entry(label):
    if label not in WINDOWS:raise ValueError('five-window diagnostic cannot capture an undeclared callback')
    data=ENTRY_PREFIX+json.dumps({'pid':os.getpid(),'window':label},separators=(',',':')).encode()+b'\n'
    if os.write(2,data)!=len(data):raise OSError('diagnostic entry evidence was not published')


def profiler(torch,grants):
    """Pinned SDK config extension; no experimental custom-backend JSON abuse."""
    if torch.version.git_version!=SDK_GIT or torch.version.cuda!='12.9':
        raise RuntimeError('scheduler diagnostic buffer contract requires the pinned05adb/CUDA12.9 SDK')
    config_path=os.environ.get('KINETO_CONFIG')
    if (os.environ.get('KINETO_LOG_LEVEL')!='0' or not config_path
            or Path(config_path).read_text()!=KINETO_CONFIG_TEXT):
        raise RuntimeError('diagnostic requires its pinned explicit Kineto loss-query logging configuration')
    config_module=torch.profiler.profiler
    base=config_module._ProfilerExtensionConfig
    class BufferConfig(base):
        def _to_config_entries(self):
            return {'ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB':str(grants.kineto_config_mb)}
    activity=torch.profiler.ProfilerActivityConfig(profiler_configs=[BufferConfig()])
    return torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
        {torch.profiler.ProfilerActivity.CUDA:activity}],record_shapes=False,profile_memory=False,with_stack=False)


CUDA_ACTIVITY_SCHEMA='tensorfold.native-cuda-profiler-coverage-v1'
CUDA_ACTIVITY_TYPES=('kernel','cuda_runtime','cuda_driver','gpu_memcpy','gpu_memset')


def native_cuda_activity(profile):
    """Read the stopped pinned-SDK native event buffer, not device headers.

    The profiler retains these events through export. This uses the same native
    result API as its Python exporter; no second trace or CUDA operation runs.
    Native event presence is necessary. Exported activity coverage/correlations
    still require post-run analysis before kernel or critical-path claims.
    """
    counts=dict.fromkeys(CUDA_ACTIVITY_TYPES,0)
    total=0
    for event in profile.profiler.kineto_results.trace_activities():
        total+=1
        kind=event.type()
        if kind in counts:counts[kind]+=1
    return {'schema':CUDA_ACTIVITY_SCHEMA,'counts':counts,'native_event_count':total}


def require_cuda_activity(value):
    if (type(value)is not dict or set(value)!={'schema','counts','native_event_count'}
            or value['schema']!=CUDA_ACTIVITY_SCHEMA or type(value['counts'])is not dict
            or set(value['counts'])!=set(CUDA_ACTIVITY_TYPES)
            or type(value['native_event_count'])is not int or value['native_event_count']<0
            or any(type(n)is not int or n<0 for n in value['counts'].values())
            or sum(value['counts'].values())>value['native_event_count']):
        raise ValueError('stopped native CUDA activity evidence is invalid')
    if value['counts']['kernel']==0 or value['counts']['cuda_runtime']+value['counts']['cuda_driver']==0:
        raise RuntimeError('scheduler diagnostic lacks actual CUDA kernel/runtime activity; trace incomplete')
    return value


LOSS_MARKERS=('Exceeded max GPU buffer count','CollectTrace stopped by GPU profiler',
              'CUPTI unavailable; continuing with CPU-only profiling','CUPTI_ERROR',
              'CUDA is not available, disabling CUDA profiling')


def check_native_log(path,*,pid=None):
    """Source-versioned native loss/readiness checks after positive retirement."""
    configured=False;windows=[]
    with Path(path).open('rb')as source:
        while True:
            line=source.readline(CHUNK+1)
            if not line:break
            if len(line)>CHUNK:raise RuntimeError('native log line exceeds diagnostic evidence envelope; trace incomplete')
            if (any(marker.encode()in line for marker in LOSS_MARKERS)
                    or re.search(rb'Dropped [0-9]+ activity records',line)):
                raise RuntimeError('scheduler diagnostic native activity collection failed/exhausted; trace incomplete')
            if pid is not None:
                if ((' '+str(pid)+':').encode()in line and b'ConfigLoader.cpp:'in line
                        and b'Detected base config change'in line):
                    configured=True
                if line.startswith(ENTRY_PREFIX):
                    event=json.loads(line[len(ENTRY_PREFIX):])
                    if event['pid']==pid:
                        if not configured:raise RuntimeError('native dropped-record query logging was not proven before capture')
                        windows.append(event['window'])
    if pid is not None and windows!=list(WINDOWS):
        raise RuntimeError('native log lacks exact five-window collection evidence')
    return {'native_log_checked':True,'logging_config_proven_before_capture':configured,
            'windows':windows,'P3_valid':False}


def collect(directory,grants,*,expected_pid,native_log):
    """Close the five-window evidence against the actual NUMA rank-zero PID.

    This validates owned producer artifacts after positive process retirement;
    it does not turn instrumented latency into performance evidence.
    """
    if type(expected_pid)is not int or expected_pid<=0:
        raise ValueError('scheduler diagnostic requires its actual NUMA rank-zero PID')
    root=Path(directory)
    expected={name+suffix for name in WINDOWS for suffix in ('.json','-receipt.json')}
    expected.update(('inventory.json','prefill-selection.json'))
    paths=list(root.iterdir())
    if ({p.name for p in paths}!=expected or any(p.is_symlink()or not p.is_file()for p in paths)):
        raise ValueError('scheduler diagnostic artifacts are incomplete or outside the fixed five-window closure')
    total=sum(p.stat().st_size for p in paths)
    if total>grants.disk_bytes:
        raise ValueError('scheduler diagnostic exceeded its shared disk grant')
    memory_limit=None;initial_memory=None
    for window in WINDOWS:
        receipt=root/(window+'-receipt.json')
        with receipt.open('rb')as source:raw=source.read((1<<20)+1)
        if len(raw)>1<<20:raise ValueError('scheduler window receipt exceeds metadata envelope')
        row=json.loads(raw)
        if (row.get('process_pid')!=expected_pid or type(row.get('process_pid'))is not int
                or row.get('rank')!=0 or type(row.get('rank'))is not int
                or row.get('label')!=window or row.get('primary')is not None or row.get('post_errors')!=[]
                or row.get('P3_valid')is not False or row.get('diagnostic_grants')!=grants.as_dict()
                or row.get('capture_limit')!=5 or row.get('trace_region')!='incumbent-c1-c8'):
            raise ValueError('scheduler diagnostic actual process/window/resource outcome differs')
        trace=root/(window+'.json')
        if (type(row.get('exported_trace_bytes'))is not int or row['exported_trace_bytes']<=0
                or trace.stat().st_size!=row['exported_trace_bytes']):
            raise ValueError('scheduler diagnostic export missing/truncated')
        require_cuda_activity(row.get('native_cuda_activity'))
        before,after=row.get('entry_headroom'),row.get('exit_headroom')
        for point,needed in ((before,grants.device_headroom_bytes),(after,0)):
            if (type(point)is not dict or point.get('required_headroom_bytes')!=needed
                    or any(type(point.get(k))is not int or point[k]<needed
                           for k in ('free_bytes','service_spare_bytes'))):
                raise ValueError('scheduler diagnostic lacks its unchanged-floor headroom proof')
            limit=point.get('cgroup',{}).get('memory.max')
            if type(limit)is not str or not limit.isdecimal()or int(limit)<=0:
                raise ValueError('scheduler diagnostic lacks its whole-container memory envelope')
            if memory_limit is not None and limit!=memory_limit:
                raise ValueError('scheduler diagnostic memory envelope changed between windows')
            memory_limit=limit
        check_memory_events(before,after)
        if initial_memory is None:initial_memory=before
        check_memory_events(initial_memory,after)
    native=check_native_log(native_log,pid=expected_pid)
    return {'complete':True,'P3_valid':False,'managed_disk_bytes':total,
            'grants':grants.as_dict(),'native':native,'windows':list(WINDOWS),'process_pid':expected_pid}
