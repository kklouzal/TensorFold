"""Closed full32 capacity operation and ordinary lifecycle boundaries; no CUDA evidence."""
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from tools import v100_full32_q4_gate as gate
from tools import v100_target_window_teacher as shared
from tests.test_v100_target_window_capacity import proof
from tensorfold.families.qwen4_exp.kv_formats import get_pair


def write_asset(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    raw=json.dumps(value,sort_keys=True).encode();path.write_bytes(raw);path.chmod(0o444)
    return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}


def profile():
    return {'schema':'tensorfold.execution-profile-manifest','version':1,'artifacts':{'model':{'size':1,'sha256':'a'*64}},
        'preparation':{'kernel_family':'sm70-paged-owner-rn-ex2-p2051-v1','graph_family':'uncaptured',
            'graph_budget_bytes':0,'trial':True,'collective_family':'tp8-allgather-v1','rn_heads_per_block':12,
            'native_gdn_capture':True,'native_hc_writeback_capture':True,
            'exact_graph_policy':{'policy':'observed-demand-v1','keys':[list(k)for k in sorted(gate.C40)],'evidence_sha256':'d'*64}}}


@pytest.fixture
def operation(tmp_path):
    checked=tmp_path/'installed'
    wheel={'source_uid':'a'*64,'members':{'tensorfold/example.py':{'bytes':1,'sha256':'1'*64}},
           'fd_owner_source':{'sha256':'2'*64}}
    return {'schema':gate.SCHEMA,'application':{'root':str(tmp_path/'app'),'source_uid':'a'*64},
        'tools':{'root':str(tmp_path/'tools'),'source_uid':'b'*64},'image':'sha256:'+'c'*64,
        'installed_check':{'root':str(checked),'inputs':write_asset(checked/'check-inputs.json',{}),
            'terminal':write_asset(checked/'container-result.json',{})},
        'wheel':write_asset(tmp_path/'wheel.json',wheel),'profile':write_asset(tmp_path/'profile.json',profile()),
        'model':{'gpus':[f'GPU-{i:08x}-0000-0000-0000-000000000000'for i in range(8)],
            'models_root':'/models/hf/hub','target':'target/model.gguf','mtp':'mtp/model.gguf',
            'cache':str(tmp_path/'cache'),'max_waiting_input_bytes':226493088},
        'numa':{'plan':str(tmp_path/'numa-plan.json'),'plan_sha256':'3'*64,'seccomp_directory':str(tmp_path/'numa'),
            'seccomp_manifest_sha256':'4'*64},'resources':{'cpus':64,'memory':'340g','startup_timeout_s':900,'operation_timeout_s':7200},
        'health_baseline':{name:'5'*64 for name in gate.ordinary.HEALTH},'output':str(tmp_path/'output')}


def test_closed_full32_operation_is_separate_from_existing_teacher_and_keeps_all_keys(operation):
    assert gate.validate_operation(operation) is operation
    selected=profile();keys=selected['preparation']['exact_graph_policy']['keys']
    assert len(keys)==40 and gate.C36<=set(map(tuple,keys))
    assert {(role,rows,32)for role in ('verify','mtp')for rows in (32,160)}<=set(map(tuple,keys))
    from tools.v100_owned_model_contract import SERVICE
    assert SERVICE['resident_capacity']==24 and SERVICE['kv_dtype']=='int8'
    argv=gate.serve_argv(operation)
    assert argv[0]=='/models/hf/hub/target/model.gguf'
    assert argv[argv.index('--drafter')+1]=='/models/hf/hub/mtp/model.gguf'
    for name,value in (('--parallel','32'),('--kv-full-windows','32'),('--context','262144'),
            ('--kv-dtype',gate.CODEC),('--mtp-drafts','4'),('--expert-tensor-parallel-size','4')):
        assert argv[argv.index(name)+1]==value
    assert '--no-graphs' not in argv


@pytest.mark.parametrize('fault',['missing_key','extra_key','duplicate','bool_rows','other_codec_profile','version_bool'])
def test_no_capacity_or_catalogue_shortening_is_admitted(fault):
    selected=profile();keys=selected['preparation']['exact_graph_policy']['keys']
    if fault=='missing_key':keys.pop()
    elif fault=='extra_key':keys.append(['verify',161,32])
    elif fault=='duplicate':keys[-1]=keys[0]
    elif fault=='bool_rows':keys[0][1]=True
    elif fault=='version_bool':selected['version']=True
    else:selected['preparation']['native_family']='owned-native-family-v1'
    with pytest.raises(ValueError):gate.profile_preparation(selected)


def go(spec,digest='f'*64,boot='boot'):
    return {'schema':gate.ROOT_GO_SCHEMA,'operation_sha256':digest,'image':spec['image'],
        'application_source_uid':spec['application']['source_uid'],'tools_source_uid':spec['tools']['source_uid'],
        'boot_id':boot,'approved':True}


@pytest.mark.parametrize('field',['image','application_source_uid','tools_source_uid','operation_sha256','boot_id','approved'])
def test_one_exact_root_authorization_cannot_be_reused(operation,field):
    approved=go(operation);approved[field]=1 if field=='approved' else 'changed'
    with pytest.raises(ValueError):gate.authorize(approved,operation,'f'*64,'boot')


def test_ordinary_container_retains_existing_numapassthrough_and_explicit_worker_supervisor(operation,tmp_path):
    argv=gate.container_argv(operation,{'cpuset_cpus':'1-8','cpuset_mems':'0-1'},tmp_path/'operator',tmp_path/'out','f'*64,123.0)
    assert '--privileged' not in argv and '--cap-add' not in argv and '--pid' not in argv
    assert argv[argv.index('--user')+1]=='1000:1000' and argv[argv.index('--network')+1]=='none'
    assert argv[argv.index('--runtime')+1]=='nvidia' and '--read-only' in argv
    assert '/tools/v100_full32_q4_gate.py' in argv and '--supervise' in argv
    assert '/operator/operation.json' in argv and '--operation-sha256' in argv
    assert any(str(tmp_path/'operator') in arg and arg.endswith(':/operator:ro')for arg in argv)
    assert '/models/hf/hub:/models/hf/hub:ro' in argv
    assert '/models/hf/hub:/models:ro' not in argv


@pytest.mark.parametrize('root',['/','/tools','/tools/nested','/operator','/opt','/cache','/tmp/models','/models/../tools','/models:other'])
def test_model_namespace_cannot_shadow_runtime_or_other_mounts(operation,root):
    operation['model']['models_root']=root
    with pytest.raises(ValueError,match='namespace|absolute DATA path'):gate.validate_operation(operation)


def test_supervisor_keeps_actual_child_owner_and_failurechannel(operation,tmp_path,monkeypatch):
    from tools import launch_cuda_tp
    raw=json.dumps(operation,sort_keys=True).encode();path=tmp_path/'operation.json';path.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest();calls=[]
    monkeypatch.setattr(launch_cuda_tp,'main',lambda argv,**kwargs:calls.append((argv,kwargs)) or 0)
    tail=['--gpus',','.join(operation['model']['gpus']),'--',*gate.serve_argv(operation)]
    deadline=time.monotonic()+1
    assert gate.main(['--supervise','--operation',str(path),'--operation-sha256',digest,
        '--operation-deadline-monotonic',str(deadline),'--operation-clock-offset-ns',str(gate._monotonic_offset_ns()),'--operation-boot-id',Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'--',*tail])==0
    assert calls[0][0]==tail
    prefix=calls[0][1]['_command_prefix']
    assert '--worker' in prefix and prefix[-1]=='--' and prefix[prefix.index('--operation-sha256')+1]==digest
    assert 0<calls[0][1]['_retirement_grace']<=1
    assert calls[0][1]['_failure_probe']() is False


@pytest.mark.parametrize('expired',[False,True])
def test_supervisor_deadline_is_not_renewed_and_clock_epoch_is_required(operation,tmp_path,monkeypatch,expired):
    from tools import launch_cuda_tp
    raw=json.dumps(operation).encode();path=tmp_path/'operation.json';path.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest();calls=[]
    monkeypatch.setattr(launch_cuda_tp,'main',lambda *a,**kw:calls.append(kw)or 0)
    deadline=time.monotonic()+(-1 if expired else 1)
    args=['--supervise','--operation',str(path),'--operation-sha256',digest,
        '--operation-deadline-monotonic',str(deadline),'--operation-boot-id',Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'--operation-clock-offset-ns']
    with pytest.raises(ValueError,match='monotonic offset'):gate.main([*args,str(gate._monotonic_offset_ns()+1),'--','model'])
    assert not calls
    if expired:
        with pytest.raises(ValueError,match='expired before rank acquisition'):
            gate.main([*args,str(gate._monotonic_offset_ns()),'--','model'])
        assert not calls
    else:
        gate.main([*args,str(gate._monotonic_offset_ns()),'--','model'])
        assert calls[0]['_retirement_grace']>0
        assert calls[0]['_failure_probe']() is False
        assert calls[0]['_shutdown_probe'](False) is False
        assert calls[0]['_shutdown_probe'](True) is True


def test_deadline_during_real_acquisition_stops_before_next_child_and_retires_first(operation,tmp_path,monkeypatch):
    from tools import launch_cuda_tp
    from tensorfold.cuda import expert_launch
    raw=json.dumps(operation).encode();path=tmp_path/'operation.json';path.write_bytes(raw)
    digest=hashlib.sha256(raw).hexdigest();clock=[time.monotonic()];deadline=clock[0]+1
    monkeypatch.setattr(gate,'time',NS(monotonic=lambda:clock[0]))
    children=[];popen=expert_launch.subprocess.Popen;main=launch_cuda_tp.main
    def acquire(*args,**kwargs):
        child=popen(*args,**kwargs);children.append(child);clock[0]=deadline
        return child
    def launch(tail,**kwargs):
        # Exercise the real supervisor with harmless CPU children. The first
        # acquisition deterministically consumes the remaining grant.
        kwargs['_command_prefix']=[sys.executable,'-B','-c','import time;time.sleep(5)']
        return main(['--gpus','0,1','--','model'],**kwargs)
    monkeypatch.setattr(expert_launch.subprocess,'Popen',acquire)
    monkeypatch.setattr(launch_cuda_tp,'main',launch)
    with pytest.raises(RuntimeError,match='startup cancelled before cohort acquisition'):
        gate.main(['--supervise','--operation',str(path),'--operation-sha256',digest,
            '--operation-deadline-monotonic',str(deadline),'--operation-clock-offset-ns',str(gate._monotonic_offset_ns()),
            '--operation-boot-id',Path('/proc/sys/kernel/random/boot_id').read_text().strip(),'--','model'])
    assert len(children)==1 and children[0].returncode is not None


@pytest.mark.parametrize('outcome',['positive','postcheck_failure','deadline'])
def test_actual_supervised_close_keeps_long_postcheck_and_nonzero_outcome(tmp_path,outcome):
    from tools import launch_cuda_tp
    # Emulate ordinary HTTP's one shutdown interrupt, resource close, then the
    # required postcheck with that handler still active. The postcheck exceeds
    # the short surrogate old grace; it must complete under the paid deadline.
    script=r'''
import sys,signal,time
from pathlib import Path
root=Path(sys.argv[1]);rank=int(sys.argv[sys.argv.index('--rank')+1]);mode=sys.argv[2]
def stop(*args):raise KeyboardInterrupt
signal.signal(signal.SIGTERM,stop)
(root/('ready'+str(rank))).write_text('ready')
if rank==0:
 try:
  while True:time.sleep(.01)
 except KeyboardInterrupt:(root/'closed').write_text('closed')
else:
 while not (root/'closed').exists():time.sleep(.01)
time.sleep(1 if mode=='deadline' else .25)
(root/('audited'+str(rank))).write_text('checked')
raise SystemExit(7 if mode=='postcheck_failure' and rank==0 else 0)
'''
    start=time.monotonic()
    result=launch_cuda_tp.main(['--gpus','0,1','--','model'],
        _command_prefix=[sys.executable,'-B','-c',script,str(tmp_path),outcome],
        _shutdown_probe=lambda stopping:all((tmp_path/('ready'+str(rank))).exists()for rank in range(2)),
        _retirement_grace=.1 if outcome=='deadline' else 2)
    assert time.monotonic()-start<5
    assert (result==0) is (outcome=='positive')
    if outcome=='positive':assert all((tmp_path/('audited'+str(rank))).exists()for rank in range(2))
    elif outcome=='postcheck_failure':assert (tmp_path/'audited0').exists()


@pytest.mark.parametrize('grace',[-1,True,float('nan'),float('inf')])
def test_private_launch_grace_refuses_invalid_bound_before_acquisition(grace):
    from tools import launch_cuda_tp
    with pytest.raises(ValueError,match='retirement grace'):launch_cuda_tp.main([],_retirement_grace=grace)


def valid_rows(output,spec,app,tools):
    for rank in range(8):
        value,retention=proof(4);value['rank']=rank
        for item in value['capacity']['multi_graphs']['keys']:item['accounting']['rank']=rank
        pair=get_pair(gate.CODEC);value['kv_pair']={'key_dtype':pair.key_dtype,'value_dtype':pair.value_dtype,'identity':pair.identity}
        wheel=json.loads(Path(spec['wheel']['path']).read_bytes())
        row={'schema':gate.RANK_SCHEMA,'rank':rank,'grouped_gate_up':{'requested_rows':4,'actual_rows':4},'PID':100+rank,
            'operation_sha256':spec['_operation_sha256'],'capacity':value,'retention':retention,
            'numerical_trial':{'dense':'reference','concat':'reference','native_family':False,
                'prepared_linears':False,'owned_q8_window_prepare':False},
            'startup':{'schema':'tensorfold.private-owned-qualification-v1','state_count':32,'world':8,'rank':rank,
                'target_banks':48,'mtp_banks':1,'capability':[7,0]},
            'process':{'Uid':'1000 1000 1000 1000','Gid':'1000 1000 1000 1000',
                **{name:'0000000000000000'for name in ('CapEff','CapPrm','CapInh','CapAmb')}},
            'installed':{'wheel_members':len(wheel['members']),'wheel_sha256':hashlib.sha256(json.dumps(wheel,sort_keys=True).encode()).hexdigest(),
                'FD_source_sha256':wheel['fd_owner_source']['sha256']},
            'runtime_cli_sha256':app['files']['src/tensorfold/cli.py']['sha256'],
            'producer_sha256':tools['files']['tools/v100_target_window_teacher.py']['sha256'],
            'owned_observer_sha256':tools['files']['tools/v100_owned_qualification.py']['sha256'],
            'observer_sha256':tools['files']['tools/v100_full32_q4_gate.py']['sha256'],'scope':'ordinary capacity only'}
        write_asset(output/'qualification'/f'full32-rank-{rank}.json',row)


@pytest.fixture
def ready(operation,tmp_path):
    app={'files':{'src/tensorfold/cli.py':{'sha256':'6'*64}}}
    tools={'files':{f'tools/{name}.py':{'sha256':str(index)*64}for index,name in enumerate(
        ('v100_target_window_teacher','v100_owned_qualification','v100_full32_q4_gate'),7)}}
    # Use legal hexadecimal identities; fixture counters are not serialized integers.
    for index,key in enumerate(tools['files']):tools['files'][key]['sha256']=format(index+7,'x')*64
    operation={**operation,'_operation_sha256':'f'*64};output=tmp_path/'evidence';valid_rows(output,operation,app,tools)
    return output,operation,app,tools


def test_real_shared_capacity_consumer_requires_all32_and_every_C40key_per_rank(ready):
    output,spec,app,tools=ready
    rows=gate.collect(output,spec,app,tools,shared)
    assert len(rows)==8 and len({r['PID']for r in rows})==8
    assert all(r['receipt']['sha256'] for r in rows)


@pytest.mark.parametrize('fault',['state_missing','short_context','tail','workspace','graph_missing','sample','codec_alias','pid','source','caps'])
def test_missing_capacity_graph_or_origin_can_never_be_a_pass(ready,fault):
    output,spec,app,tools=ready;path=output/'qualification/full32-rank-7.json';row=json.loads(path.read_bytes())
    value=row['capacity']
    if fault=='state_missing':value['states'].pop()
    elif fault=='short_context':value['capacity']['context_window']=260060
    elif fault=='tail':value['physical_capacity']=262144
    elif fault=='workspace':value['capacity']['kv_full_windows']['workspace_bytes']=1<<30
    elif fault=='graph_missing':value['capacity']['multi_graphs']['exact_graph_catalogue']['captured_keys'].pop()
    elif fault=='sample':value['sample']['live_bytes']=0
    elif fault=='codec_alias':
        pair=get_pair('int4');value['kv_pair']={'key_dtype':pair.key_dtype,'value_dtype':pair.value_dtype,'identity':pair.identity}
    elif fault=='pid':row['PID']=100
    elif fault=='source':row['runtime_cli_sha256']='0'*64
    else:row['process']['CapEff']='0000010000000000'
    path.chmod(0o644);path.write_text(json.dumps(row))
    with pytest.raises(ValueError):gate.collect(output,spec,app,tools,shared)


def test_oversized_metadata_is_fatal_before_publication(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'RECEIPT_LIMIT',16)
    with pytest.raises(ValueError,match='funded'):gate._write_rank(tmp_path/'rank.json',{'data':'x'*100})
    assert not (tmp_path/'rank.json').exists()


@pytest.mark.parametrize('fault',[None,'ready','nonzero_close','cleanup','publication'])
def test_ordinary_ready_retirement_retains_failures_and_never_forces_success(operation,tmp_path,monkeypatch,fault):
    raw=json.dumps(operation,sort_keys=True).encode();digest=hashlib.sha256(raw).hexdigest()
    tool_files={f'tools/{name}.py':{'sha256':str(index)*64}for index,name in enumerate(
        ('v100_target_window_teacher','v100_owned_qualification'),7)}
    tool_files['tools/v100_full32_q4_gate.py']={'sha256':gate.campaign.identity(Path(gate.__file__))['sha256']}
    tools={'source_uid':'b'*64,'files':tool_files};app={'source_uid':'a'*64,'files':{'src/tensorfold/cli.py':{'sha256':'6'*64}}}
    monkeypatch.setattr(gate.campaign,'frozen',lambda root:app if root==Path(operation['application']['root']) else tools)
    monkeypatch.setattr(gate.campaign,'immutable_image',lambda image:{'id':image})
    monkeypatch.setattr(gate.campaign,'installed_check_receipt',lambda *a:{'accepted':True})
    monkeypatch.setattr(gate,'_host_shared',lambda *a:shared)
    monkeypatch.setattr(gate,'_verify_tool_helpers',lambda *a:None) # Synthetic lifecycle tree; separate origin controls below.
    Path(operation['tools']['root']).mkdir(parents=True)
    (Path(operation['tools']['root'])/'source.json').write_text(json.dumps(tools))
    bundle={'files':{'manifest.json':json.dumps({'provenance':{'container':{'image':operation['image']}}}).encode()},
        'cpuset_cpus':'1-8','cpuset_mems':'0-1'}
    monkeypatch.setattr(gate.campaign,'_numa_inputs',lambda *a:bundle)
    monkeypatch.setattr(gate.campaign,'_stage_numa',lambda *a:None)
    monkeypatch.setattr(gate.campaign,'verify_source',lambda *a:None)
    monkeypatch.setattr(gate.ordinary,'health',lambda *a:{'unchanged':True})
    cid='a'*64;removed=[];commands=[]
    def command(argv):
        commands.append(argv)
        if argv[:2]==['docker','rm']:removed.append(argv[-1]);return argv[-1]
        return ''
    monkeypatch.setattr(gate.ordinary,'command',command)
    monkeypatch.setattr(gate.campaign,'numa_receipts',lambda *a:{'ranks':[{'pid':100+i}for i in range(8)]})
    output=Path(operation['output'])
    def logged(argv,*args):
        if argv[:2]==['docker','exec'] and gate.campaign._WAIT_READY in argv:
            if fault in ('ready','publication'):raise RuntimeError('original ordinary READY failure')
            (output/'readiness.json').write_text(json.dumps({'stats':{'load':{'requests_in_flight':0,'requests_running':0,
                'requests_waiting':0,'lanes':{}},'totals':{'requests':0,'generated_tokens':0}}}))
        return None
    monkeypatch.setattr(gate.campaign,'logged',logged)
    owners=[]
    class Owner:
        def __init__(self):self.cid=None;self.state=None;self.errors=[];self.retired=False;owners.append(self)
        def create(self,argv,publish):
            self.cid=cid;publish(argv,NS(stdout=(cid+'\n').encode()))
            valid_rows(output,{**operation,'_operation_sha256':digest},app,tools)
        def retire(self,*,wait_timeout_s):
            assert 0<=wait_timeout_s<=operation['resources']['operation_timeout_s']
            self.retired=True;self.state={'Status':'exited','Pid':0,'ExitCode':7 if fault=='nonzero_close' else 0,'OOMKilled':False}
            if fault=='cleanup':self.errors.append(RuntimeError('owned cleanup failed'))
        def exit_code(self,primary):return int(primary is not None or bool(self.errors) or self.state['ExitCode']!=0)
    if fault=='publication':
        original=gate.campaign.publish
        def publish(path,value):
            if path.name=='result.json':raise MemoryError('result publication failed')
            return original(path,value)
        monkeypatch.setattr(gate.campaign,'publish',publish)
    approved=go(operation,digest,Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    if fault=='publication':
        with pytest.raises(RuntimeError,match='original ordinary READY failure'):
            gate.run(operation,approved,digest,raw,owner_factory=Owner)
    else:
        result=gate.run(operation,approved,digest,raw,owner_factory=Owner)
        assert result==(0 if fault is None else 1)
        report=json.loads((output/'result.json').read_bytes())
        assert report['exit_code']==result and report['P3_valid'] is False and report['quality_qualified'] is False
    assert owners[0].retired and owners[0].state['Pid']==0 and removed==[cid]


@pytest.mark.parametrize('mode',['success','return_nonzero','raise','raise_and_post_fail'])
def test_worker_postclose_identity_check_runs_for_every_ordinary_cli_outcome(monkeypatch,mode):
    primary=RuntimeError('original CLI failure');post=MemoryError('post-close identity failure');calls=[]
    def serve(args):
        calls.append('serve')
        if mode.startswith('raise'):raise primary
        return 7 if mode=='return_nonzero' else 0
    def verify(wheel):
        calls.append('postcheck')
        if mode=='raise_and_post_fail':raise post
        return {'actual':'same'}
    monkeypatch.setattr(gate,'_installed_wheel',verify)
    if mode=='success':assert gate._serve_with_postcheck(NS(cmd_serve=serve),object(),{}, {'actual':'same'})==0
    else:
        with pytest.raises(RuntimeError) as caught:gate._serve_with_postcheck(NS(cmd_serve=serve),object(),{}, {'actual':'same'})
        if mode.startswith('raise'):assert caught.value is primary
    assert calls==['serve','postcheck']


def test_imported_helper_origin_is_checked_even_for_equal_bytes(tmp_path,monkeypatch):
    directory=tmp_path/'source/tools';directory.mkdir(parents=True)
    helper=directory/'helper.py';helper.write_text('# fixed helper fixture\n');helper.chmod(0o444)
    own=directory/'v100_full32_q4_gate.py';own.write_text('# fixed gate fixture\n');own.chmod(0o444)
    monkeypatch.setattr(gate,'__file__',str(own))
    paths={'helper.py':helper};monkeypatch.setattr(gate.ordinary,'helper_paths',lambda:paths)
    record={'files':{'tools/'+p.name:{'type':'file',**gate.campaign.identity(p)}for p in (helper,own)}}
    gate._verify_tool_helpers(directory,record)
    foreign=tmp_path/'helper.py';foreign.write_bytes(helper.read_bytes());foreign.chmod(0o444);paths['helper.py']=foreign
    with pytest.raises(ValueError,match='origin'):gate._verify_tool_helpers(directory,record)
    # A later legitimate alias must not hide the active foreign helper.
    monkeypatch.setitem(gate.sys.modules,'full32_legitimate_alias',NS(__file__=str(helper)))
    with pytest.raises(ValueError,match='origin'):gate._verify_tool_helpers(directory,record)
    paths['helper.py']=helper;helper.chmod(0o644)
    with pytest.raises(ValueError,match='origin'):gate._verify_tool_helpers(directory,record)


@pytest.mark.parametrize('alias_first',[False,True])
def test_application_collision_requires_exact_root_member_and_does_not_hide_foreign_tool(tmp_path,monkeypatch,alias_first):
    directory=tmp_path/'tools';directory.mkdir()
    own=directory/'v100_full32_q4_gate.py';own.write_text('# gate\n');own.chmod(0o444)
    helper=directory/'collision.py';helper.write_text('# tool\n');helper.chmod(0o444)
    package=tmp_path/'installed/tensorfold';(package/'cuda').mkdir(parents=True)
    app=package/'cuda/collision.py';app.write_text('# application\n')
    foreign=tmp_path/'collision.py';foreign.write_bytes(helper.read_bytes());foreign.chmod(0o444)
    monkeypatch.setattr(gate,'__file__',str(own));monkeypatch.setattr(gate.ordinary,'helper_paths',lambda:{'collision.py':helper})
    record={'files':{'tools/'+p.name:{'type':'file',**gate.campaign.identity(p)}for p in (own,helper)}}
    authority=(package,{'tensorfold/cuda/collision.py':gate.campaign.identity(app)})
    monkeypatch.setitem(gate.sys.modules,'tensorfold.cuda.collision',NS(__file__=str(app)))
    gate._verify_tool_helpers(directory,record,application=authority)
    with pytest.raises(ValueError,match='authority'):gate._verify_tool_helpers(directory,record)
    with pytest.raises(ValueError,match='origin'):gate._verify_tool_helpers(directory,record,application=(tmp_path/'other',authority[1]))
    for name,path in ([('foreign_tool_alias',foreign),('legitimate_tool_alias',helper)] if not alias_first
                      else [('legitimate_tool_alias',helper),('foreign_tool_alias',foreign)]):
        monkeypatch.setitem(gate.sys.modules,name,NS(__file__=str(path)))
    with pytest.raises(ValueError,match='origin'):gate._verify_tool_helpers(directory,record,application=authority)


def test_actual_capacity_setup_callback_journals_before_notification(tmp_path):
    import ast
    tree=ast.parse(Path(gate.__file__).read_text())
    worker=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='worker')
    callback=next(node for node in worker.body if isinstance(node,ast.FunctionDef) and node.name=='before_listener')
    calls=[];primary=RuntimeError('first capacity validation failure');engine=NS()
    def capacity(*args,**kwargs):calls.append('capacity');raise primary
    def journal(actual,output,error,rows):
        assert actual is engine and actual._abort_comm is True and output==tmp_path and error is primary and rows==160
        calls.append('journal-before-notify');raise error
    namespace={'shared':NS(_runtime_capacity_proof=capacity),'owned':NS(
        _native_family_receipt=lambda e:None,_prepared_m32_receipt=lambda e:None,
        _owned_q8_preparation_receipt=lambda e:None,_publish_qualification_failure=journal),
        'require':gate.require,'CONTRACT':gate.CONTRACT,'preparation':{},'retention_identity_payload':lambda e:{},'output':tmp_path}
    # Execute the actual small callback, not a second failure-transport model;
    # stop at the deliberately failing producer before any Tensor/CUDA query.
    exec(compile(ast.fix_missing_locations(ast.Module(body=[callback],type_ignores=[])),gate.__file__,'exec'),namespace)
    with pytest.raises(RuntimeError) as caught:namespace['before_listener'](engine)
    assert caught.value is primary and calls==['capacity','journal-before-notify']
