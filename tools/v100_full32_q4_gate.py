"""Closed ordinary-API full32 ISO4 capacity/C40 gate, not a teacher or benchmark.

Import/help/validation have no container/GPU effects. A sealed Root GO binds
one exact application image, application source, read-only tool snapshot,
model/profile/NUMA authorities and boot. Existing C8 remains24-Q8. The actual
shared capacity producer checks32 distinct full native State owners per rank;
no populated-history, independent CAP40/VMA, model-quality or P3 claim follows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

if __package__:
    from . import v100_ragged_replay_gate as ordinary
    from . import v100_onecat_campaign as campaign
else:
    import v100_ragged_replay_gate as ordinary
    import v100_onecat_campaign as campaign

require, fields, absolute, asset, read, object_json = (ordinary.require, ordinary.fields,
    ordinary.absolute, ordinary.asset, ordinary.read, ordinary.object_json)
SCHEMA = 'tensorfold.full32-q4-capacity-operation-v1'
RESULT_SCHEMA = 'tensorfold.full32-q4-capacity-result-v1'
RANK_SCHEMA = 'tensorfold.full32-q4-capacity-rank-v1'
CODEC = 'rotorquant-iso4-sm70-v1'
CONTRACT = 'full-window-physical-headroom-v1'
ROOT_GO_SCHEMA = 'tensorfold.full32-q4-capacity-Root-GO-v1'
C36 = frozenset({(role, n*rows, n) for n in (1,2,4,8,16,24)
    for rows in (range(1,6) if n==1 else (1,5)) for role in ('verify','mtp')} |
    {('verify',28,8),('mtp',28,8),*(('mtp',n,n) for n in (3,5,6,7))})
C40 = C36 | frozenset((role,rows,32) for role in ('verify','mtp') for rows in (32,160))
RECEIPT_LIMIT = 16 << 20
OPERATION_LIMIT = 1 << 20


def profile_preparation(profile):
    require(type(profile) is dict and profile.get('schema') == 'tensorfold.execution-profile-manifest'
        and type(profile.get('version')) is int and profile['version'] == 1
        and type(profile.get('artifacts')) is dict and profile['artifacts'],
        'complete original execution profile required')
    preparation = profile.get('preparation')
    require(type(preparation) is dict, 'explicit C40 preparation required')
    policy = preparation.get('exact_graph_policy')
    fields(policy, ('policy','keys','evidence_sha256'))
    require(policy['policy'] == 'observed-demand-v1' and ordinary.SHA.fullmatch(policy['evidence_sha256'])
        and type(policy['keys']) is list and len(policy['keys']) == 40
        and all(type(k) is list and len(k)==3 and type(k[0]) is str
                and all(type(v) is int for v in k[1:]) for k in policy['keys'])
        and {tuple(k) for k in policy['keys']} == C40,
        'retain every C36 key plus full32/E160 verify and MTP keys')
    expected = {'kernel_family':'sm70-paged-owner-rn-ex2-p2051-v1','graph_family':'uncaptured',
        'graph_budget_bytes':0,'trial':True,'collective_family':'tp8-allgather-v1','rn_heads_per_block':12,
        'native_gdn_capture':True,'native_hc_writeback_capture':True,'exact_graph_policy':policy}
    require(json.dumps(preparation,sort_keys=True)==json.dumps(expected,sort_keys=True),
            'capacity gate retains the independent original numerical provider profile')
    return preparation


def validate_operation(value):
    fields(value, ('schema','application','tools','image','installed_check','wheel','profile','model','numa',
                   'resources','health_baseline','output'))
    require(value['schema']==SCHEMA, 'full32 capacity operation version differs')
    for name in ('application','tools'):
        fields(value[name],('root','source_uid'));absolute(value[name]['root'])
        require(type(value[name]['source_uid']) is str and ordinary.SHA.fullmatch(value[name]['source_uid']), 'exact source UID required')
    require(type(value['image']) is str and re.fullmatch('sha256:[0-9a-f]{64}',value['image']), 'exact application image required')
    fields(value['installed_check'],('root','inputs','terminal'))
    root=absolute(value['installed_check']['root'])
    for name,filename in (('inputs','check-inputs.json'),('terminal','container-result.json')):
        asset(value['installed_check'][name],8<<20)
        require(Path(value['installed_check'][name]['path'])==root/filename, 'installed authority path differs')
    wheel=object_json(asset(value['wheel'],16<<20))
    require(wheel.get('source_uid')==value['application']['source_uid'],'wheel/application source authority differs')
    profile_preparation(object_json(asset(value['profile'],16<<20)))
    fields(value['model'],('gpus','models_root','target','mtp','cache','max_waiting_input_bytes'))
    model=value['model']
    require(type(model['gpus']) is list and len(model['gpus'])==len(set(model['gpus']))==8
        and all(type(g) is str and re.fullmatch('GPU-[0-9a-f-]{36}',g) for g in model['gpus']), 'eight exact ordered GPU UUIDs required')
    for name in ('models_root','cache'):absolute(model[name])
    model_root=Path(model['models_root'])
    require(':' not in str(model_root) and '..' not in model_root.parts and all(
        not model_root.is_relative_to(reserved) and not Path(reserved).is_relative_to(model_root)
        for reserved in ('/tools','/operator','/evidence','/cache','/tmp','/opt','/proc','/sys','/dev','/usr','/etc')),
        'model namespace overlaps a runtime mount or protected application path')
    for name in ('target','mtp'):
        require(type(model[name]) is str and not Path(model[name]).is_absolute()
            and '..' not in Path(model[name]).parts and model[name].endswith('.gguf'), 'relative exact GGUF model required')
    require(type(model['max_waiting_input_bytes']) is int and 0<model['max_waiting_input_bytes']<2**63,'explicit unchanged input credit required')
    fields(value['numa'],('plan','plan_sha256','seccomp_directory','seccomp_manifest_sha256'))
    absolute(value['numa']['plan']);absolute(value['numa']['seccomp_directory'])
    require(all(type(value['numa'][k]) is str and ordinary.SHA.fullmatch(value['numa'][k])
                for k in ('plan_sha256','seccomp_manifest_sha256')), 'sealed ordinary NUMA authority required')
    fields(value['resources'],('cpus','memory','startup_timeout_s','operation_timeout_s'))
    resources=value['resources']
    require(type(resources['cpus']) is int and 1<=resources['cpus']<=64
        and type(resources['memory']) is str and re.fullmatch('[1-9][0-9]{0,2}g',resources['memory'])
        and all(type(resources[k]) is int and 1<=resources[k]<=7200 for k in ('startup_timeout_s','operation_timeout_s'))
        and resources['startup_timeout_s']<=resources['operation_timeout_s'], 'bounded actual host resources/timeouts required')
    fields(value['health_baseline'],ordinary.HEALTH)
    require(all(type(v) is str and ordinary.SHA.fullmatch(v) for v in value['health_baseline'].values()), 'health input identity required')
    absolute(value['output'])
    return value


def authorize(go,spec,digest,boot):
    fields(go,('schema','operation_sha256','image','application_source_uid','tools_source_uid','boot_id','approved'))
    require(go['approved'] is True and go == {'schema':ROOT_GO_SCHEMA,'operation_sha256':digest,'image':spec['image'],
        'application_source_uid':spec['application']['source_uid'],'tools_source_uid':spec['tools']['source_uid'],
        'boot_id':boot,'approved':True}, 'one exact boot/image/sources/operation Root GO required')


def serve_argv(spec):
    model=spec['model']
    return [str(Path(model['models_root'])/model['target']),'--backend','cuda','--context','262144','--parallel','32',
        '--kv-full-windows','32','--kv-dtype',CODEC,'--indexer-history','gpu','--expert-tensor-parallel-size','4',
        '--draft-method','mtp','--drafter',str(Path(model['models_root'])/model['mtp']),'--mtp-drafts','4','--mtp-confidence','0.7',
        '--no-prefill-fp8','--no-thinking','--no-update-check','--host','127.0.0.1','--port','8000',
        '--name','providerone-full32-q4-capacity','--max-pending-requests','32','--max-http-connections','33',
        '--max-waiting-input-bytes',str(model['max_waiting_input_bytes']),
        '--execution-profile-manifest','/operator/profile.json']


def _proof_resources():
    """Use the shared actual-Python JSON payload accounting for bounded receipts."""
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        _json_payload_policy,_json_parse_payload,_json_tree_payload,_streamed_json_encoding_payload,_python_layout)
    policy=_json_payload_policy()
    # Source proof is a bounded32-State/8-rank/C40 metadata projection; no tensors
    # or history contents are serialized. Oversize is fatal, never truncation.
    per_rank=(_json_parse_payload(OPERATION_LIMIT,policy)['total_bytes']
        +_json_tree_payload(OPERATION_LIMIT,policy)+2*_json_tree_payload(RECEIPT_LIMIT,policy)
        +_streamed_json_encoding_payload(RECEIPT_LIMIT,_python_layout())['total_bytes']+policy.frame_work)
    require(0<8*per_rank<=sys.maxsize,'capacity observer host obligation overflow')
    return 8*per_rank


def _write_rank(path,record):
    encoded=0
    for chunk in json.JSONEncoder(indent=1,ensure_ascii=True,allow_nan=False).iterencode(record):
        encoded+=len(chunk)
        require(encoded+1<=RECEIPT_LIMIT,'capacity receipt exceeds its funded metadata bound')
    if __package__:from .benchmark_output import write_json
    else:from benchmark_output import write_json
    require(not path.exists() and not path.is_symlink(),'capacity receipts are single-publication files')
    write_json(path,record)


def _installed_wheel(expected):
    import tensorfold
    from tensorfold import _fd_owner
    package=Path(tensorfold.__file__).resolve().parent
    require(not package.is_relative_to(Path('/tools')), 'installed application cannot be a source overlay')
    actual={}
    for path in package.rglob('*'):
        if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
            actual['tensorfold/'+str(path.relative_to(package))]=campaign.identity(path)
    require(actual==expected['members'], 'actual whole installed application differs from selected wheel')
    require(_fd_owner._source_sha256==expected['fd_owner_source']['sha256'], 'actual native FD source binding differs')
    from tensorfold.cuda.sm70 import software_binding
    binding=software_binding.issue_source_binding(required=True);software_binding.verify_source_binding(binding)
    return {'wheel_members':len(actual),'wheel_sha256':hashlib.sha256(json.dumps(expected,sort_keys=True).encode()).hexdigest(),
            'installed_root':str(package),'FD_source_sha256':_fd_owner._source_sha256}


def _verify_tool_helpers(directory,tools,*,application=None):
    """Fresh imports must originate in the exact authenticated frozen tool tree."""
    paths=list(ordinary.helper_paths().items())
    paths.append(('v100_full32_q4_gate.py',Path(__file__)))
    for logical_name,module in tuple(sys.modules.items()):
        filename=getattr(module,'__file__',None)
        if filename and ('tools/'+Path(filename).name) in tools['files']:
            if logical_name.startswith('tensorfold.'):
                require(application is not None,'application collision lacks source authority: '+logical_name)
                root,members=application
                relative=logical_name.replace('.','/')+'.py'
                path=Path(filename)
                expected=members.get(relative)
                require(expected is not None and path.absolute()==root.parent/relative and not path.is_symlink(),
                    'application collision origin differs: '+logical_name)
                raw=read(path.absolute(),8<<20)
                require(len(raw)==expected.get('bytes',expected.get('size'))
                    and hashlib.sha256(raw).hexdigest()==expected['sha256'],
                    'application collision content differs: '+logical_name)
                continue
            paths.append((Path(filename).name,Path(filename)))
    # An alias with the same basename must never replace an active foreign
    # helper: every observed module origin remains independently authoritative.
    for name,path in paths:
        expected=tools['files']['tools/'+name]
        require(path.absolute()==directory/name and expected['type']=='file' and not path.is_symlink()
            and not path.stat().st_mode&0o222,'actual tool helper origin is not the sealed source: '+name)
        raw=read(path.absolute(),8<<20)
        require(len(raw)==expected['bytes'] and hashlib.sha256(raw).hexdigest()==expected['sha256'],
                'actual imported tool helper differs: '+name)


def worker(operation_path,digest,tail):
    raw=read(operation_path,OPERATION_LIMIT)
    require(hashlib.sha256(raw).hexdigest()==digest,'worker operation differs from supervisor authority')
    spec=object_json(raw)
    # Worker fields are already source/seal validated by the host; no host-path
    # file access or arbitrary command is accepted at this boundary.
    require(spec['schema']==SCHEMA, 'worker capacity operation version differs')
    tools=object_json(read('/operator/tools-source.json',16<<20))
    require(tools['source_uid']==spec['tools']['source_uid'] and hashlib.sha256(json.dumps(
        tools['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()==tools['source_uid'],
        'worker tool source authority differs')
    wheel=object_json(read('/operator/wheel.json',16<<20))
    import tensorfold
    package=Path(tensorfold.__file__).resolve().parent
    require(not package.is_relative_to(Path('/tools')),'application cannot originate in tool overlay')
    application=(package,wheel['members'])
    _verify_tool_helpers(Path('/tools'),tools,application=application)
    from tensorfold import cli
    from tensorfold.families.qwen4_exp.cuda.execution_identity import retention_identity_payload
    if __package__:
        from . import v100_target_window_teacher as shared
        from . import v100_owned_qualification as owned
    else:
        import v100_target_window_teacher as shared
        import v100_owned_qualification as owned
    _verify_tool_helpers(Path('/tools'),tools,application=application)
    actual_tail=serve_argv(spec)
    # launch_cuda_tp appends only its fixed coordinate arguments.
    args=cli.build_parser().parse_args(['serve',*tail])
    require(tail==[*actual_tail,'--tp','8','--rank',str(args.rank),'--master','127.0.0.1',
        '--master-port',str(args.master_port)] and 0<=args.rank<8,'worker ordinary serve arguments differ')
    installed=_installed_wheel(wheel)
    preparation=profile_preparation(object_json(read('/operator/profile.json',16<<20)))
    output=Path('/evidence/qualification');output.mkdir(exist_ok=True)
    obligation=_proof_resources()
    def before_listener(engine):
        try:
            from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense,GGUFConcat
            require(GGUFDense.implementation==GGUFConcat.implementation=='reference'
                and owned._native_family_receipt(engine) is None and owned._prepared_m32_receipt(engine) is None
                and owned._owned_q8_preparation_receipt(engine) is None,
                    'capacity gate actual numerical providers differ from the original reference')
            proof=shared._runtime_capacity_proof(engine,{'capacity_contract':CONTRACT,'prepared_execution':preparation},
                retention_identity_payload(engine),slots=32,bits=4)
            retention=retention_identity_payload(engine)
            shared.validate_capacity_proof(proof,retention=retention,slots=32,bits=4)
            shared._validate_capacity_catalogue(proof['capacity'],{'prepared_execution':preparation})
            require(proof['kv_pair']['key_dtype']==proof['kv_pair']['value_dtype']==CODEC,
                    'full32 gate requires exact ISO4 codec, not a bit-count alias')
            startup=owned.bound_receipt(engine)
            require(startup['rank']==engine.rank and startup['world']==8 and startup['state_count']==32,
                    'actual full32 ordinary startup owner binding differs')
            status=Path('/proc/self/status').read_text();wanted={'Uid','Gid','CapEff','CapPrm','CapInh','CapAmb','CapBnd','NoNewPrivs','Seccomp'}
            facts={key:value.strip() for line in status.splitlines() for key,_,value in [line.partition(':')] if key in wanted}
            require(facts['Uid'].split()==facts['Gid'].split()==['1000']*4
                and facts['CapEff']==facts['CapPrm']==facts['CapInh']==facts['CapAmb']=='0000000000000000',
                'ordinary workers must remain nonroot capability-zero')
            from tensorfold import cli as installed_cli
            _write_rank(output/f'full32-rank-{engine.rank}.json',{'schema':RANK_SCHEMA,'rank':engine.rank,'grouped_gate_up':owned._grouped_gate_up_receipt(engine,4),
                'PID':os.getpid(),'operation_sha256':digest,'capacity':proof,'retention':retention,'startup':startup,
                'numerical_trial':{'dense':'reference','concat':'reference','native_family':False,
                    'prepared_linears':False,'owned_q8_window_prepare':False},
                'process':facts,'installed':installed,'runtime_cli_sha256':campaign.identity(Path(installed_cli.__file__))['sha256'],
                'producer_sha256':campaign.identity(Path(shared.__file__))['sha256'],
                'owned_observer_sha256':campaign.identity(Path(owned.__file__))['sha256'],
                'observer_sha256':campaign.identity(Path(__file__))['sha256'],
                'scope':'ordinary Cap0 capacity/C40 allocation proof; no CAP40 map_files, populated-history, quality or P3 proof'})
        except BaseException as primary:
            engine._abort_comm=True
            owned._publish_qualification_failure(engine,output,primary,160)
    args._owned_expert_qualification=cli._OwnedExpertQualification(before_listener,(obligation,digest),execution_rows=160,
        **owned._grouped_gate_up_capability(cli,4))
    return _serve_with_postcheck(cli,args,wheel,installed)


def _serve_with_postcheck(cli,args,wheel,installed):
    """Ordinary CLI owns all resource retirement; always audit after its outcome."""
    result,primary=None,None
    try:
        result=cli.cmd_serve(args)
        if type(result) is not int or result!=0:
            primary=RuntimeError('ordinary model CLI did not retire positively: '+repr(result))
    except BaseException as error:primary=error
    cleanup=[]
    try:require(_installed_wheel(wheel)==installed,'whole installed application changed through close')
    except BaseException as error:cleanup.append(error)
    campaign.raise_failures(primary,cleanup)
    return result


def container_argv(spec,bundle,operator,output,operation_sha):
    # Reuse ordinary UID1000/Cap0/no-network/NUMA/memlock/static environment,
    # replacing only the fixed supervised worker prefix and declared profile.
    adapter={'model':spec['model'],'resources':spec['resources'],'numa':spec['numa'],'image':spec['image']}
    argv=ordinary.container_argv(adapter,absolute(spec['tools']['root']),bundle,operator,output)
    # Ancillary GGUF metadata embeds absolute model paths. Preserve the declared
    # host namespace inside the container for both target and MTP; shortening it
    # would change config.json and invalidate the sealed execution profile.
    root=spec['model']['models_root'];old_mount=root+':/models:ro'
    require(argv.count(old_mount)==1,'ordinary model mount authority differs')
    argv[argv.index(old_mount)]=root+':'+root+':ro'
    marker=argv.index('--entrypoint')
    supervisor=['/opt/venv/bin/python','-B','/tools/launch_cuda_tp.py','--gpus',','.join(spec['model']['gpus']),
        '--numa-plan','/operator/numa/plan.json','--numa-plan-sha256',spec['numa']['plan_sha256'],'--',*serve_argv(spec)]
    # The fixed supervisor entry dispatch below retains launch_cuda_tp's child
    # lifetime/failure-FD behavior while selecting this reviewed worker module.
    supervisor=['/opt/venv/bin/python','-B','/tools/v100_full32_q4_gate.py','--supervise',
        '--operation','/operator/operation.json','--operation-sha256',operation_sha,'--',*supervisor[3:]]
    return argv[:marker]+['--entrypoint',supervisor[0],spec['image'],*supervisor[1:]]


def _host_shared(tool_root,tools):
    # Only after Root authorization + full frozen source verification. The
    # shared validator imports stdlib-only family metadata, never Torch/CUDA.
    location=str(tool_root/'source/src');sys.path.insert(0,location)
    try:
        if __package__:from . import v100_target_window_teacher as shared
        else:import v100_target_window_teacher as shared
        from tensorfold.families.qwen4_exp.cuda import service_memory,execution_identity
        from tensorfold.families.qwen4_exp import kv_formats
        for module in (service_memory,execution_identity,kv_formats):
            path=Path(module.__file__)
            relative='src/tensorfold/'+str(path.relative_to(Path(sys.modules['tensorfold'].__file__).parent))
            require(path.absolute()==tool_root/'source'/relative
                and campaign.identity(path)['sha256']==tools['files'][relative]['sha256'],
                    'host capacity validator imported a different family source')
        require(campaign.identity(Path(shared.__file__))['sha256']==tools['files']['tools/v100_target_window_teacher.py']['sha256'],
                'host shared capacity producer/consumer source differs')
        _verify_tool_helpers(tool_root/'source/tools',tools,application=(tool_root/'source/src/tensorfold',
            {name.removeprefix('src/'):record for name,record in tools['files'].items() if name.startswith('src/tensorfold/')}))
        return shared
    finally:sys.path.pop(0)


def collect(output,spec,app,tools,shared):
    preparation=profile_preparation(object_json(asset(spec['profile'],16<<20)))
    records=[];pids=set()
    for rank in range(8):
        path=output/'qualification'/f'full32-rank-{rank}.json';row=object_json(read(path.absolute(),RECEIPT_LIMIT))
        fields(row,('schema','rank','grouped_gate_up','PID','operation_sha256','capacity','retention','startup','process','installed','numerical_trial',
                    'runtime_cli_sha256','producer_sha256','owned_observer_sha256','observer_sha256','scope'))
        require(row['schema']==RANK_SCHEMA and type(row['rank']) is int and row['rank']==rank
            and type(row['PID']) is int and row['PID']>0 and row['PID'] not in pids,'eight actual distinct worker identities required')
        pids.add(row['PID'])
        require(row['operation_sha256']==spec['_operation_sha256'],'rank operation authority differs')
        require(row['numerical_trial']=={'dense':'reference','concat':'reference','native_family':False,
            'prepared_linears':False,'owned_q8_window_prepare':False},'actual capacity numerical profile differs')
        require(row['runtime_cli_sha256']==app['files']['src/tensorfold/cli.py']['sha256']
            and row['producer_sha256']==tools['files']['tools/v100_target_window_teacher.py']['sha256']
            and row['owned_observer_sha256']==tools['files']['tools/v100_owned_qualification.py']['sha256']
            and row['observer_sha256']==tools['files']['tools/v100_full32_q4_gate.py']['sha256'], 'actual application/producer/observer source binding differs')
        shared.validate_capacity_proof(row['capacity'],retention=row['retention'],slots=32,bits=4)
        shared._validate_capacity_catalogue(row['capacity']['capacity'],{'prepared_execution':preparation})
        require(type(row['grouped_gate_up']) is dict and set(row['grouped_gate_up'])=={'requested_rows','actual_rows'}
            and all(type(v) is int and v==4 for v in row['grouped_gate_up'].values()),
                'actual grouped4 expert provider binding differs')
        require(row['process']['Uid'].split()==row['process']['Gid'].split()==['1000']*4
            and all(row['process'][name]=='0000000000000000' for name in ('CapEff','CapPrm','CapInh','CapAmb')),
                'ordinary Cap0 process evidence differs')
        require(row['startup']['schema']=='tensorfold.private-owned-qualification-v1'
            and row['startup']['target_banks']==48 and row['startup']['mtp_banks']==1
            and row['startup']['capability']==[7,0] and row['capacity']['rank']==rank and row['startup']['state_count']==32
            and row['startup']['world']==8 and row['startup']['rank']==rank
            and row['capacity']['kv_pair']['key_dtype']==row['capacity']['kv_pair']['value_dtype']==CODEC,
                'actual full32 proof/codec/startup ownership differs')
        wheel=object_json(asset(spec['wheel'],16<<20))
        require(row['installed']['wheel_members']==len(wheel['members'])
            and row['installed']['wheel_sha256']==hashlib.sha256(json.dumps(wheel,sort_keys=True).encode()).hexdigest()
            and row['installed']['FD_source_sha256']==wheel['fd_owner_source']['sha256'], 'actual whole application wheel/FD proof differs')
        records.append({'rank':rank,'PID':row['PID'],'receipt':{'file':str(path.relative_to(output)),**campaign.identity(path)}})
    return records


def run(spec,go,operation_sha,operation_raw,*,owner_factory=ordinary.lifetime.OwnedContainer):
    require(hashlib.sha256(operation_raw).hexdigest()==operation_sha and object_json(operation_raw)==spec,
        'exact held operation bytes required before effects')
    authorize(go,spec,operation_sha,Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    app_root,tool_root=(absolute(spec[name]['root']) for name in ('application','tools'))
    app,tools=(campaign.frozen(root) for root in (app_root,tool_root))
    require(app['source_uid']==spec['application']['source_uid'] and tools['source_uid']==spec['tools']['source_uid'], 'actual source snapshot differs')
    require(campaign.identity(Path(__file__))['sha256']==tools['files']['tools/v100_full32_q4_gate.py']['sha256'], 'executing gate differs from reviewed tool source')
    _verify_tool_helpers(tool_root/'source/tools',tools)
    binding=campaign.immutable_image(spec['image']);require(binding['id']==spec['image'],'actual application image differs')
    installed=campaign.installed_check_receipt(absolute(spec['installed_check']['root']),app['source_uid'],spec['image'])
    shared=_host_shared(tool_root,tools)
    bundle=campaign._numa_inputs(tool_root,tools,{'gpus':spec['model']['gpus'],'numa':spec['numa']})
    numa_manifest=object_json(bundle['files']['manifest.json'])
    require(numa_manifest['provenance']['container']['image']==spec['image'], 'ordinary NUMA seccomp must bind this actual image')
    output=absolute(spec['output']);output.mkdir(parents=True,exist_ok=False);operator=output/'operator';operator.mkdir()
    for name,raw in (('profile.json',asset(spec['profile'],16<<20)),('wheel.json',asset(spec['wheel'],16<<20)),
                     ('operation.json',operation_raw),('tools-source.json',read(tool_root/'source.json',16<<20))):
        (operator/name).write_bytes(raw);(operator/name).chmod(0o444)
    # Worker authenticates the same canonical operation bytes as Root; the CLI
    # validates the supplied file digest before this host run begins.
    require(hashlib.sha256((operator/'operation.json').read_bytes()).hexdigest()==operation_sha,'staged operation bytes differ')
    campaign._stage_numa(bundle,operator,output);operator.chmod(0o555)
    owner=owner_factory();primary=None;errors=[];rows=None;numa=None;before=ordinary.health(spec);after=None;started=time.monotonic()
    campaign.publish(output/'inputs.json',{'operation':spec,'operation_sha256':operation_sha,'root_go':go,'installed':installed,
        'application_source_uid':app['source_uid'],'tools_source_uid':tools['source_uid'],'image_binding':binding,'scope':'full32 capacity/C40 fit only; ordinary Cap0, no teacher/P3 claim'})
    try:
        argv=container_argv(spec,bundle,operator,output,operation_sha)
        owner.create(argv,lambda command,result:campaign.publish(output/'create.json',{'argv':command,'cid':result.stdout.decode().strip()}))
        ordinary.command(['docker','start',owner.cid]);deadline=min(started+spec['resources']['startup_timeout_s'],
            started+spec['resources']['operation_timeout_s'])
        while len(list((output/'qualification').glob('full32-rank-*.json')))!=8:
            state=object_json(ordinary.command(['docker','inspect','--format','{{json .State}}',owner.cid]))
            require(state['Running'],'capacity model exited before all eight proof receipts')
            if time.monotonic()>=deadline:raise TimeoutError('full32 startup deadline expired')
            time.sleep(min(1,max(0,deadline-time.monotonic())))
        local={**spec,'_operation_sha256':operation_sha};rows=collect(output,local,app,tools,shared)
        ready=output/'readiness';ready.mkdir();remaining=deadline-time.monotonic();require(remaining>0,'no remaining ordinary READY time')
        campaign.logged(['docker','exec',owner.cid,'/opt/venv/bin/python','-B','-c',campaign._WAIT_READY,str(remaining),'q4'],ready,remaining+5)
        readiness=object_json(read((output/'readiness.json').absolute(),16<<20))['stats']
        require(all(readiness['load'][name]==0 for name in ('requests_in_flight','requests_running','requests_waiting'))
            and all(n==0 for n in readiness['load']['lanes'].values())
            and readiness['totals']['requests']==readiness['totals']['generated_tokens']==0,'ordinary READY must have zero inference requests')
        numa=campaign.numa_receipts(tool_root,tools,output,bundle)
        require([r['pid'] for r in numa['ranks']]==[r['PID'] for r in rows],'actual ready NUMA/PID proof binding differs')
        pids=[r['PID'] for r in rows];caps=output/'capabilities';caps.mkdir()
        campaign.logged(['docker','exec',owner.cid,'/opt/venv/bin/python','-B','-c',ordinary._CAPS,json.dumps(pids)],caps,30)
    except BaseException as error:primary=error
    try:owner.retire()
    except BaseException as error:errors.append(error)
    errors.extend(e for e in owner.errors if e is not None)
    for action in (lambda:campaign.verify_source(app_root/'source',app['files']),
                   lambda:campaign.verify_source(tool_root/'source',tools['files']),
                   lambda:validate_operation(spec),
                   lambda:require(rows is None or collect(output,{**spec,'_operation_sha256':operation_sha},app,tools,shared)==rows,'capacity proof changed through close'),
                   lambda:require(numa is None or campaign.numa_receipts(tool_root,tools,output,bundle)==numa,'actual NUMA receipts changed through close')):
        try:action()
        except BaseException as error:errors.append(error)
    try:after=ordinary.health(spec)
    except BaseException as error:errors.append(error)
    if owner.cid:
        try:
            logs=output/'logs';logs.mkdir()
            campaign.logged(['docker','logs',owner.cid],logs,60)
        except BaseException as error:errors.append(error)
    removed=False
    if owner.cid and owner.state and owner.state.get('Status')=='exited' and owner.state.get('Pid')==0:
        try:
            require(ordinary.command(['docker','rm',owner.cid]).strip()==owner.cid,'owned removal receipt differs');removed=True
        except BaseException as error:errors.append(error)
    if owner.state is not None and owner.state.get('ExitCode') not in (None,0):
        close_error=RuntimeError('ordinary supervisor/worker close returned nonzero: '+str(owner.state['ExitCode']))
        if primary is None:primary=close_error
        else:errors.append(close_error)
    if time.monotonic()>started+spec['resources']['operation_timeout_s']:
        timeout=TimeoutError('full32 operation deadline exceeded; owned retirement completed before reporting')
        if primary is None:primary=timeout
        else:errors.append(timeout)
    failed=owner.exit_code(primary) or bool(errors)
    try:campaign.publish(output/'result.json',{'schema':RESULT_SCHEMA,'cid':owner.cid,'state':owner.state,'exit_code':int(failed),
        'primary':None if primary is None else repr(primary),'cleanup':[repr(e)for e in errors],'receipts':rows,'numa':numa,
        'health_before':before,'health_after':after,'removed':removed,'elapsed_s':time.monotonic()-started,
        'scope':'materialized full32 ISO4/C40 capacity only, not populated-history or quality',
        'P3_valid':False,'quality_qualified':False})
    except BaseException as error:campaign.raise_failures(primary,[*errors,error])
    return int(failed)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--worker',action='store_true');mode.add_argument('--supervise',action='store_true')
    mode.add_argument('--run',action='store_true');mode.add_argument('--validate',action='store_true')
    parser.add_argument('--operation',required=True,type=Path);parser.add_argument('--operation-sha256',required=True)
    parser.add_argument('--root-go',type=Path);parser.add_argument('tail',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    raw=read(args.operation,OPERATION_LIMIT)
    require(hashlib.sha256(raw).hexdigest()==args.operation_sha256,'external operation identity differs')
    spec=object_json(raw)
    if args.worker:return worker(args.operation,args.operation_sha256,args.tail[1:] if args.tail[:1]==['--'] else args.tail)
    if args.supervise:
        if __package__:from . import launch_cuda_tp
        else:import launch_cuda_tp
        prefix=[sys.executable,'-B',str(Path(__file__)),'--worker','--operation',str(args.operation),
            '--operation-sha256',args.operation_sha256,'--']
        tail=args.tail[1:] if args.tail[:1]==['--'] else args.tail
        return launch_cuda_tp.main(tail,_command_prefix=prefix)
    validate_operation(spec)
    if args.validate:print(json.dumps({'validated':True,'schema':SCHEMA,'operation_sha256':args.operation_sha256}));return 0
    require(args.root_go is not None,'run requires separate sealed Root GO')
    return run(spec,object_json(read(args.root_go,1<<20)),args.operation_sha256,raw)


if __name__=='__main__':raise SystemExit(main())
