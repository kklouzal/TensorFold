"""Private diagnostic grants, active storage containment and source-bound loss checks."""
import json
import threading
from types import SimpleNamespace as NS

import pytest

from tools import v100_scheduler_diagnostic as diagnostic


def spec():
    return {'schema':diagnostic.SCHEMA,'scope':diagnostic.SCOPE,'host_obligation_bytes':16<<20,
            'device_headroom_bytes':8<<20,'disk_bytes':1<<20,'cupti_inflight_buffer_bytes':8<<20,
            'native_log_bytes':1<<20}


@pytest.mark.parametrize('key,value',[('host_obligation_bytes',True),('disk_bytes',0),
                                     ('device_headroom_bytes',-1),('cupti_inflight_buffer_bytes',1),
                                     ('scope','historical-prefill-only')])
def test_typed_grants_cannot_relabel_or_guess_resources(key,value):
    with pytest.raises(ValueError):diagnostic.Grants.parse({**spec(),key:value})


def test_kineto_inflight_count_rounding_is_explicit_not_total_memory_bound():
    value=diagnostic.Grants.parse(spec())
    assert (1+value.kineto_config_mb//4)*diagnostic.BLOCK==value.cupti_inflight_buffer_bytes
    assert value.as_dict()==spec()


class Exporter:
    def __init__(self,data):self.data=data
    def export_chrome_trace(self,path,*,use_python_export=False):
        assert use_python_export
        with open(path,'wb',buffering=0)as stream:
            stream.writelines(self.data[start:start+1024] for start in range(0,len(self.data),1024))


def test_storage_cumulatively_bounds_metadata_and_streamed_export(tmp_path):
    storage=diagnostic.TraceStorage(tmp_path,10000)
    storage.write_json('selection.json',{'status':'pending'})
    storage.write_json('selection.json',{'status':'observed'})
    assert storage.export('window.json',Exporter(b'payload'))==7
    assert storage.used==sum(p.stat().st_size for p in tmp_path.iterdir())
    assert (tmp_path/'window.json').read_bytes()==b'payload'


def test_disk_exhaustion_prevents_excess_writes_drains_reader_and_is_sticky(tmp_path):
    storage=diagnostic.TraceStorage(tmp_path,2048)
    with pytest.raises((BrokenPipeError,diagnostic.StorageExhausted)):
        storage.export('window.json',Exporter(b'x'*(8<<20)))
    assert sum(p.stat().st_size for p in tmp_path.iterdir())<=2048
    assert storage.failed is not None
    assert not any(t.name=='scheduler-trace-disk-sink' and t.is_alive()for t in threading.enumerate())
    with pytest.raises(BaseException):storage.write_json('not-retried.json',{})
    assert not (tmp_path/'window.json').exists()


def test_native_cpp_export_path_is_not_silently_used(tmp_path):
    storage=diagnostic.TraceStorage(tmp_path,10000)
    native=NS(export_chrome_trace=lambda path:pytest.fail('native exporter would add .tmp outside pipe'))
    with pytest.raises(RuntimeError,match='streaming exporter'):
        storage.export('window.json',native)


def test_source_model_and_reader_failures_both_survive(tmp_path):
    storage=diagnostic.TraceStorage(tmp_path,1)
    primary=RuntimeError('producer failure')
    class Failing:
        def export_chrome_trace(self,path,*,use_python_export=False):
            with open(path,'wb',buffering=0)as target:target.write(b'toolong')
            raise primary
    with pytest.raises(RuntimeError)as caught:storage.export('window.json',Failing())
    assert caught.value is primary
    assert storage.failed is not None


def test_headroom_uses_existing_post_floor_service_reader(monkeypatch):
    torch=pytest.importorskip('torch')
    monkeypatch.setattr(torch.cuda,'mem_get_info',lambda device:(100<<20,32<<30))
    monkeypatch.setattr(diagnostic,'cgroup_memory',lambda:{'memory.max':str(1<<30),'memory.peak':'1','memory.events':'oom 0\noom_kill 0'})
    engine=NS(w=NS(device='cuda:0'),multi=NS(graphs=NS(spare=lambda:4<<20)))
    with pytest.raises(MemoryError,match='unchanged device floor'):
        diagnostic.headroom(engine,diagnostic.Grants.parse(spec()))
    engine.multi.graphs.spare=lambda:9<<20
    result=diagnostic.headroom(engine,diagnostic.Grants.parse(spec()))
    assert result['service_spare_bytes']==9<<20 and result['required_headroom_bytes']==8<<20
    engine.multi.graphs.spare=lambda:0
    assert diagnostic.headroom(engine,diagnostic.Grants.parse(spec()),after=True)['required_headroom_bytes']==0


@pytest.mark.parametrize('message',diagnostic.LOSS_MARKERS)
def test_native_early_stop_or_activity_loss_invalidates_files(tmp_path,message):
    path=tmp_path/'log';path.write_text('x'*(diagnostic.CHUNK-20)+message)
    with pytest.raises(RuntimeError,match='incomplete'):diagnostic.check_native_log(path)


def test_actual_sdk_python_cpu_export_is_hard_contained(tmp_path):
    torch=pytest.importorskip('torch')
    if torch.version.git_version!=diagnostic.SDK_GIT:pytest.skip('actual pinned SDK export boundary')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])as profiler:
        value=torch.arange(8)+1
    storage=diagnostic.TraceStorage(tmp_path,1<<20)
    storage.export('actual.json',profiler)
    parsed=json.loads((tmp_path/'actual.json').read_text())
    assert parsed['traceEvents'] and value.tolist()==list(range(1,9))
    assert storage.used<=storage.limit


def test_native_loss_query_readiness_must_precede_actual_capture(tmp_path):
    path=tmp_path/'log'
    entries=''.join('SCHEDULER_DIAGNOSTIC_ENTRY '+json.dumps({'pid':17,'window':w})+'\n'for w in diagnostic.WINDOWS)
    path.write_text(entries)
    with pytest.raises(RuntimeError,match='not proven'):diagnostic.check_native_log(path,pid=17)
    path.write_text('VERBOSE:2026-11-12 00:00:00 17:18 ConfigLoader.cpp:211] Detected base config change\n'+entries)
    assert diagnostic.check_native_log(path,pid=17)['windows']==list(diagnostic.WINDOWS)
    path.write_text('VERBOSE:2026-11-12 00:00:00 19:18 ConfigLoader.cpp:211] Detected base config change\n'+entries)
    with pytest.raises(RuntimeError,match='not proven'):diagnostic.check_native_log(path,pid=17)


def test_model_container_oom_event_cannot_be_ignored_as_profiler_success():
    with pytest.raises(MemoryError,match='exhausted'):
        diagnostic.check_memory_events({'cgroup':{'memory.events':'oom 0\noom_kill 0'}},
                                       {'cgroup':{'memory.events':'oom 1\noom_kill 0'}})


def test_new_tracer_exports_through_quota_and_records_both_headroom_points(tmp_path,monkeypatch):
    from tools import v100_owned_qualification as owned
    grants=diagnostic.Grants.parse(spec())
    reads=[]
    def headroom(engine,grants,*,after=False):
        reads.append(after);return {'cgroup':{'memory.events':'oom 0\noom_kill 0'},'required':0 if after else grants.device_headroom_bytes}
    monkeypatch.setattr(diagnostic,'headroom',headroom)
    class Profile(Exporter):
        def __init__(self):
            super().__init__(b'{"traceEvents":[1]}')
            self.profiler=NS(kineto_results=NS(trace_activities=lambda:[
                NS(type=lambda:'kernel'),NS(type=lambda:'cuda_runtime')]))
        def __enter__(self):return self
        def __exit__(self,*args):return False
    tracer=owned._SchedulerTrace(NS(rank=0,scheduler=NS(thread=threading.current_thread())),tmp_path,
                                factory=Profile,diagnostic_grants=grants)
    result=object()
    assert tracer.capture('c1-warmed-decode-round-1',lambda:result)is result
    row=json.loads((tmp_path/'c1-warmed-decode-round-1-receipt.json').read_text())
    assert row['entry_headroom']['required']==grants.device_headroom_bytes and row['exit_headroom']['required']==0
    assert reads==[False,True] and row['P3_valid']is False and row['diagnostic_grants']==spec()
    assert row['native_cuda_activity']['counts']['kernel']==1


def test_backend_initialization_never_registers_cpu_only_and_preserves_resource_checks(monkeypatch):
    calls=[];torch=object();engine=object();grants=diagnostic.Grants.parse(spec())
    before={'cgroup':{'memory.events':'oom 0\noom_kill 0'}}
    def headroom(given,allowance,*,after=False):
        assert given is engine and allowance is grants
        calls.append(('headroom',after));return before
    class Profile:
        def __enter__(self):calls.append('enter configured CPU+CUDA')
        def __exit__(self,*args):calls.append('exit configured CPU+CUDA')
    def configured(given,allowance):
        assert given is torch and allowance is grants
        calls.append('configured profiler');return Profile()
    monkeypatch.setattr(diagnostic,'headroom',headroom)
    monkeypatch.setattr(diagnostic,'profiler',configured)
    result=diagnostic.prepare_backend(torch,engine,grants)
    assert calls==[('headroom',False),'configured profiler','enter configured CPU+CUDA',
                   'exit configured CPU+CUDA',('headroom',True)]
    assert result['activities']==['CPU','CUDA']
    assert result['entry_headroom']is before and result['exit_headroom']is before


def coverage(*types):
    profile=NS(profiler=NS(kineto_results=NS(trace_activities=lambda:[
        NS(type=lambda kind=kind:kind)for kind in types])))
    return diagnostic.native_cuda_activity(profile)


def test_cuda_coverage_uses_actual_exporter_activity_types_not_device_headers():
    value=coverage('cpu_op','kernel','cuda_driver','gpu_memcpy','user_annotation')
    assert diagnostic.require_cuda_activity(value)is value
    assert value['counts']=={'kernel':1,'cuda_runtime':0,'cuda_driver':1,'gpu_memcpy':1,'gpu_memset':0}
    assert value['native_event_count']==5


@pytest.mark.parametrize('types',[(),('cpu_op','user_annotation'),('cuda_runtime',),('kernel',)])
def test_cpu_only_or_missing_cuda_kernel_runtime_cannot_be_complete(types):
    with pytest.raises(RuntimeError,match='actual CUDA kernel/runtime'):
        diagnostic.require_cuda_activity(coverage(*types))


@pytest.mark.parametrize('model_fails',[False,True])
def test_missing_cuda_coverage_preserves_export_and_original_model_failure(tmp_path,monkeypatch,model_fails):
    from tools import v100_owned_qualification as owned
    grants=diagnostic.Grants.parse(spec())
    monkeypatch.setattr(diagnostic,'headroom',lambda *a,**kw:{'cgroup':{'memory.events':'oom 0\noom_kill 0'}})
    class Profile(Exporter):
        def __init__(self):
            super().__init__(b'{"traceEvents":[{"cat":"cpu_op"}]}')
            self.profiler=NS(kineto_results=NS(trace_activities=lambda:[]))
        def __enter__(self):return self
        def __exit__(self,*args):return False
    tracer=owned._SchedulerTrace(NS(rank=0,scheduler=NS(thread=threading.current_thread())),tmp_path,
                                factory=Profile,diagnostic_grants=grants)
    primary=ValueError('actual model failed');calls=[]
    def model():
        calls.append('model')
        if model_fails:raise primary
    with pytest.raises((ValueError,RuntimeError))as caught:
        tracer.capture(diagnostic.WINDOWS[1],model)
    assert calls==['model']
    if model_fails:assert caught.value is primary
    else:assert 'actual CUDA kernel/runtime'in str(caught.value)
    assert json.loads((tmp_path/(diagnostic.WINDOWS[1]+'.json')).read_text())['traceEvents']
    row=json.loads((tmp_path/(diagnostic.WINDOWS[1]+'-receipt.json')).read_text())
    assert row['native_cuda_activity']['native_event_count']==0
    assert 'RuntimeError'in row['post_errors'] and tracer.failed is caught.value


def test_unfunded_scheduler_observer_refuses_before_backend_or_hooks(tmp_path):
    from tools import v100_owned_qualification as owned
    engine=NS(_owned_qualification_host=None)
    with pytest.raises(ValueError,match='not admitted'):
        owned.prepare_benchmark(engine,{},tmp_path,diagnostic_grants=diagnostic.Grants.parse(spec()))


def test_campaign_forwards_new_typed_diagnostic_separately_from_closed_prefill():
    from tests.test_v100_onecat_campaign import CommitDiagnosticTests, ModelArmTests
    from tools import v100_onecat_campaign as campaign
    plan=campaign.diagnostic_c1_c8_plan(CommitDiagnosticTests().plan())
    settings=ModelArmTests().settings()
    args=campaign.model_arguments(settings,plan,scheduler_diagnostic=diagnostic.Grants.parse(spec()))
    assert args[args.index('--scheduler-diagnostic-grants')+1]=='/operator/scheduler-diagnostic.json'
    assert args[args.index('--scheduler-trace')+1]=='/evidence/qualification/trace'
    assert '--benchmark-client'not in args and '--benchmark-trace-data'not in args
    with pytest.raises(ValueError,match='finite C1/C8'):
        campaign.model_arguments(settings,CommitDiagnosticTests().plan(),scheduler_diagnostic=diagnostic.Grants.parse(spec()))


def test_actual_sdk_exposes_source_verified_inflight_config_extension(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    if torch.version.git_version!=diagnostic.SDK_GIT:pytest.skip('pinned05adb profiler extension')
    config=tmp_path/'kineto.conf';config.write_text(diagnostic.KINETO_CONFIG_TEXT)
    monkeypatch.setenv('KINETO_CONFIG',str(config));monkeypatch.setenv('KINETO_LOG_LEVEL','0')
    value=diagnostic.profiler(torch,diagnostic.Grants.parse(spec()))
    extensions=value.activity_configs[torch.profiler.ProfilerActivity.CUDA].profiler_configs
    assert extensions[-1]._to_config_entries()=={'ACTIVITIES_MAX_GPU_BUFFER_SIZE_MB':'4'}
    # Construct/configure only: this CPU boundary test never enters CUDA tracing.


@pytest.fixture
def closed_windows(tmp_path):
    grants=diagnostic.Grants.parse(spec());root=tmp_path/'trace';root.mkdir()
    for name in ('inventory.json','prefill-selection.json'):(root/name).write_text('{}')
    for window in diagnostic.WINDOWS:
        (root/(window+'.json')).write_text('{}')
        point={'free_bytes':32<<20,'service_spare_bytes':16<<20,
               'required_headroom_bytes':grants.device_headroom_bytes,
               'cgroup':{'memory.max':str(1<<30),'memory.current':'1','memory.peak':'2',
                         'memory.events':'oom 0\noom_kill 0'}}
        row={'process_pid':17,'rank':0,'label':window,'primary':None,'post_errors':[],
             'P3_valid':False,'diagnostic_grants':spec(),'capture_limit':5,
             'trace_region':'incumbent-c1-c8','exported_trace_bytes':2,'entry_headroom':point,
             'exit_headroom':{**point,'required_headroom_bytes':0},
             'native_cuda_activity':coverage('kernel','cuda_runtime')}
        (root/(window+'-receipt.json')).write_text(json.dumps(row))
    log=tmp_path/'native.log'
    log.write_text('VERBOSE:2026-11-12 00:00:00 17:18 ConfigLoader.cpp:211] Detected base config change\n'+
        ''.join('SCHEDULER_DIAGNOSTIC_ENTRY '+json.dumps({'pid':17,'window':w})+'\n'for w in diagnostic.WINDOWS))
    return root,grants,log


def test_complete_collector_binds_exact_windows_to_actual_rank0_pid(closed_windows):
    root,grants,log=closed_windows
    result=diagnostic.collect(root,grants,expected_pid=17,native_log=log)
    assert result['complete']is True and result['P3_valid']is False and result['process_pid']==17
    assert result['managed_disk_bytes']==sum(p.stat().st_size for p in root.iterdir())
    with pytest.raises(ValueError,match='actual process'):
        diagnostic.collect(root,grants,expected_pid=19,native_log=log)


@pytest.mark.parametrize('failure',['missing','cpu-only','bool-count'])
def test_final_collector_requires_positive_actual_cuda_activity(closed_windows,failure):
    root,grants,log=closed_windows
    path=root/(diagnostic.WINDOWS[1]+'-receipt.json');row=json.loads(path.read_text())
    if failure=='missing':del row['native_cuda_activity']
    elif failure=='cpu-only':row['native_cuda_activity']=coverage('cpu_op')
    else:row['native_cuda_activity']['counts']['kernel']=True
    path.write_text(json.dumps(row))
    with pytest.raises((ValueError,RuntimeError),match='CUDA'):
        diagnostic.collect(root,grants,expected_pid=17,native_log=log)


@pytest.mark.parametrize('failure',['missing','extra','partial','bool-pid','truncated','headroom','memory-limit','gap-oom'])
def test_final_collector_refuses_incomplete_or_wrong_region(closed_windows,failure):
    root,grants,log=closed_windows
    path=root/(diagnostic.WINDOWS[1]+'-receipt.json');row=json.loads(path.read_text())
    if failure=='missing':path.unlink()
    elif failure=='extra':(root/'unknown.json').write_text('{}')
    elif failure=='partial':(root/'inflight.partial').write_text('{}')
    elif failure=='bool-pid':row['process_pid']=True
    elif failure=='truncated':(root/(diagnostic.WINDOWS[1]+'.json')).write_text('')
    elif failure=='headroom':row['entry_headroom']['service_spare_bytes']=0
    elif failure=='memory-limit':row['entry_headroom']['cgroup']['memory.max']=str(2<<30)
    else:
        row['entry_headroom']['cgroup']['memory.events']='oom 1\noom_kill 0'
        row['exit_headroom']['cgroup']['memory.events']='oom 1\noom_kill 0'
    if path.exists():path.write_text(json.dumps(row))
    with pytest.raises((ValueError,MemoryError)):
        diagnostic.collect(root,grants,expected_pid=17,native_log=log)


def test_native_log_rotation_or_reordered_windows_cannot_be_complete(closed_windows):
    root,grants,log=closed_windows
    lines=log.read_text().splitlines(keepends=True)
    log.write_text(''.join(lines[1:]))
    with pytest.raises(RuntimeError,match='not proven'):
        diagnostic.collect(root,grants,expected_pid=17,native_log=log)
    log.write_text(''.join([lines[0],*reversed(lines[1:])]))
    with pytest.raises(RuntimeError,match='exact five-window'):
        diagnostic.collect(root,grants,expected_pid=17,native_log=log)


@pytest.mark.parametrize('payload,limit,failed',[(b'actual\n',100,False),(b'x'*8192,17,True)])
def test_native_log_export_is_hard_bounded_and_reaps_its_client(tmp_path,monkeypatch,payload,limit,failed):
    import subprocess
    import sys
    grants=diagnostic.Grants.parse({**spec(),'native_log_bytes':limit})
    popen=subprocess.Popen;children=[]
    def launch(argv,**kwargs):
        assert argv==['docker','logs','a'*64]
        child=popen([sys.executable,'-c','import sys;sys.stdout.buffer.write('+repr(payload)+')'],**kwargs)
        children.append(child);return child
    monkeypatch.setattr(diagnostic.subprocess,'Popen',launch)
    target=tmp_path/'server.log'
    if failed:
        with pytest.raises(BaseException):diagnostic.capture_native_log('a'*64,target,grants)
    else:
        assert diagnostic.capture_native_log('a'*64,target,grants)['bytes']==len(payload)
        assert target.read_bytes()==payload
    assert target.stat().st_size<=limit
    assert all(c.poll()is not None for c in children)
    assert not any(t.name=='scheduler-native-log-sink'and t.is_alive()for t in threading.enumerate())
    options=diagnostic.docker_log_options(grants)
    assert options==['--log-driver','local','--log-opt','max-size='+str(limit),'--log-opt','max-file=1','--log-opt','compress=false']


def test_campaign_diagnostic_helper_mismatch_refuses_before_docker_or_output(tmp_path,monkeypatch):
    import hashlib
    from tools import v100_onecat_campaign as campaign
    frozen=tmp_path/'capture';path=frozen/'source/tools/v100_scheduler_diagnostic.py'
    path.parent.mkdir(parents=True);path.write_bytes(b'other reviewed worker helper\n')
    source={'source_uid':'a'*64,'files':{'tools/v100_scheduler_diagnostic.py':{
        'type':'file','bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}}}
    monkeypatch.setattr(campaign,'frozen',lambda root:source)
    monkeypatch.setattr(campaign,'immutable_image',lambda *a,**k:pytest.fail('mismatch reached Docker'))
    grants=tmp_path/'grants.json';grants.write_text(json.dumps(spec()))
    output=tmp_path/'output'
    with pytest.raises(ValueError,match='frozen worker helper'):
        campaign.model(frozen,output,'image',tmp_path/'missing-settings',tmp_path/'missing-check',
                       1,'1g',1,1,False,scheduler_diagnostic_grants=grants)
    assert not output.exists()
