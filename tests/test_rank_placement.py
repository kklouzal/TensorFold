"""NUMA plan/rank/policy boundary contracts; actual syscalls have native gates."""
import copy
import hashlib
import json

import pytest

from tensorfold.cuda import rank_placement as placement


@pytest.fixture
def hardware(tmp_path,monkeypatch):
    proc=tmp_path/'proc';sysfs=tmp_path/'sys';(proc/'self').mkdir(parents=True)
    (proc/'self/status').write_text('Mems_allowed_list:\t0-1\n')
    (sysfs/'devices/system/node').mkdir(parents=True);(sysfs/'devices/system/node/online').write_text('0-1')
    (sysfs/'devices/system/cpu').mkdir(parents=True);(sysfs/'devices/system/cpu/online').write_text('0-15')
    for cpu in range(16):
        node=0 if cpu%8<4 else 1
        path=sysfs/f'devices/system/cpu/cpu{cpu}/topology';path.mkdir(parents=True)
        (path/'physical_package_id').write_text(str(node));(path/'core_id').write_text(str(cpu%4))
    uuids=['GPU-00000000-0000-0000-0000-'+str(i+1).zfill(12)for i in range(4)]
    for i,uuid in enumerate(uuids):
        bus=f'0000:{i+1:02x}:00.0';node=i//2
        directory=proc/'driver/nvidia/gpus'/bus;directory.mkdir(parents=True)
        (directory/'information').write_text(f'GPU UUID: {uuid}\nBus Location: {bus}\n')
        pci=sysfs/'bus/pci/devices'/bus;pci.mkdir(parents=True)
        (pci/'numa_node').write_text(str(node));(pci/'local_cpulist').write_text('0-3,8-11' if node==0 else '4-7,12-15')
    library=tmp_path/'libnuma.so';library.write_bytes(b'not executable; metadata fixture')
    monkeypatch.setattr(placement,'_libnuma_path',lambda:library)
    monkeypatch.setattr(placement,'_package_version',lambda:{'name':'libnuma1','version':'fixture','architecture':'fixture'})
    monkeypatch.setattr(placement.os,'sched_getaffinity',lambda pid:set(range(16)))
    return uuids,proc,sysfs


def plan(hardware):
    devices,proc,sysfs=hardware
    return placement.build_plan(devices,proc=proc,sysfs=sysfs)


def seal(tmp_path,value):
    path=tmp_path/'plan.json';data=json.dumps(value).encode();path.write_bytes(data)
    return path,hashlib.sha256(data).hexdigest()


def test_gpu_local_physical_cores_keep_smt_siblings_disjoint(hardware,tmp_path):
    value=plan(hardware)
    assert [rank['cpus']for rank in value['ranks']]==[[0,1,8,9],[2,3,10,11],[4,5,12,13],[6,7,14,15]]
    assert value['cpuset_cpus']==list(range(16)) and value['cpuset_mems']==[0,1]
    path,sha=seal(tmp_path,value)
    assert placement.load_plan(path,sha)==value
    assert placement.validate_plan(value,value['ordered_gpu_uuids'],proc=hardware[1],sysfs=hardware[2])==value


@pytest.mark.parametrize('change',['uuid','overlap','topology','union','bool-rank'])
def test_corrupt_structural_plan_refused_before_native_load(hardware,tmp_path,change):
    value=plan(hardware)
    if change=='uuid':value['ordered_gpu_uuids'][0]='0'
    elif change=='overlap':value['ranks'][1]['cores']=copy.deepcopy(value['ranks'][0]['cores'])
    elif change=='topology':value['topology_sha256']='0'*64
    elif change=='union':value['cpuset_cpus']=value['cpuset_cpus'][:-1]
    elif change=='bool-rank':value['ranks'][0]['rank']=False
    path,sha=seal(tmp_path,value)
    with pytest.raises((ValueError,RuntimeError)):
        placement.load_plan(path,sha)


def test_unknown_node_or_unavailable_local_memory_never_uses_node_zero(hardware):
    devices,proc,sysfs=hardware
    pci=sysfs/'bus/pci/devices/0000:03:00.0/numa_node';pci.write_text('-1')
    with pytest.raises(RuntimeError,match='no observable'):
        placement.build_plan(devices,proc=proc,sysfs=sysfs)
    pci.write_text('1');(proc/'self/status').write_text('Mems_allowed_list:\t0\n')
    with pytest.raises(RuntimeError,match='outside allowed'):
        placement.build_plan(devices,proc=proc,sysfs=sysfs)


def test_numeric_ordinals_and_changed_allowances_refuse(hardware):
    value=plan(hardware)
    with pytest.raises(ValueError,match='full GPU UUID'):
        placement.build_plan(['0','1'],proc=hardware[1],sysfs=hardware[2])
    with pytest.raises(ValueError,match='UUID rank order'):
        placement.validate_plan(value,['0','1','2','3'],proc=hardware[1],sysfs=hardware[2])
    old=placement.os.sched_getaffinity
    placement.os.sched_getaffinity=lambda pid:{0}
    try:
        with pytest.raises(RuntimeError,match='allowance'):
            placement.validate_plan(value,proc=hardware[1],sysfs=hardware[2])
    finally:placement.os.sched_getaffinity=old


def receipt(value,stage='ready'):
    rank=value['ranks'][0]
    result={'schema':placement.SCHEMA,'policy':placement.POLICY,'stage':stage,'rank':0,'pid':101,
        **{k:rank[k]for k in ('gpu_uuid','pci_bus_id','node','cpus')},'allowed_mems':[0,1],
        'memory_policy':{'mode':32770,'nodes':[0]},'anonymous':{'bytes':16384,'page_nodes':[0]*4},
        'plan_sha256':'a'*64,'topology_sha256':value['topology_sha256'],'libnuma':value['libnuma'],
        'bootstrap_source':{'path':'/installed/rank_placement.py','bytes':123,'sha256':'b'*64},'scope':'test'}
    if stage!='bootstrap':result.update(cuda_pinned={'bytes':16384,'pinned':True,'page_nodes':[0]*5},
        threads=[{'tid':101,'name':'MainThread','cpus':rank['cpus']}],actual_cuda_gpu_uuid=rank['gpu_uuid'],cuda_device='cuda:0')
    return result


@pytest.mark.parametrize('stage',['bootstrap','post-cuda','ready'])
def test_frozen_host_receipt_validation_is_detached_and_binds_authority(hardware,stage):
    value=plan(hardware);original=receipt(value,stage)
    checked=placement.validate_receipt(original,value,0,stage,plan_sha256='a'*64,source_sha256='b'*64,pid=101)
    assert checked==original and checked is not original


@pytest.mark.parametrize('change',['source','pid','mode','remote-page','thread-escape','uuid','anon-bytes','bool-node','pinned-bytes'])
def test_bad_worker_receipt_cannot_label_unbound_or_remote_as_local(hardware,change):
    value=plan(hardware);record=receipt(value)
    if change=='source':record['bootstrap_source']['sha256']='c'*64
    elif change=='pid':record['pid']=102
    elif change=='mode':record['memory_policy']['mode']=0
    elif change=='remote-page':record['cuda_pinned']['page_nodes'][0]=1
    elif change=='thread-escape':record['threads'][0]['cpus']=[4]
    elif change=='uuid':record['actual_cuda_gpu_uuid']=value['ordered_gpu_uuids'][1]
    elif change=='anon-bytes':record['anonymous']['bytes']=1
    elif change=='bool-node':record['anonymous']['page_nodes'][0]=False
    elif change=='pinned-bytes':record['cuda_pinned']['bytes']=True
    with pytest.raises(ValueError):placement.validate_receipt(record,value,0,'ready',plan_sha256='a'*64,source_sha256='b'*64,pid=101)


def test_page_queries_cover_both_ends_of_unaligned_owned_allocation():
    policy=placement.MemoryPolicy.__new__(placement.MemoryPolicy);addresses=[]
    def query(out,mask,bits,address,flags):
        addresses.append(address.value);out._obj.value=1;return 0
    policy.get=query
    assert policy.page_nodes(0x1080,8192,1)==[1]*3
    assert addresses==[0x1000,0x2000,0x3000]


def test_thread_enumeration_is_bounded_before_retention(monkeypatch):
    from types import SimpleNamespace
    class Entries:
        def __enter__(self):return iter(SimpleNamespace(name=str(i))for i in range(100))
        def __exit__(self,*args):return None
    monkeypatch.setattr(placement,'LIMIT',96)
    monkeypatch.setattr(placement.os,'scandir',lambda path:Entries())
    with pytest.raises(ValueError,match='receipt envelope'):
        placement._thread_ids('/not-read')


def test_common_launcher_keeps_original_exec_command_and_failure_fd(hardware,tmp_path,monkeypatch):
    from tools import launch_cuda_tp
    value=plan(hardware);path,sha=seal(tmp_path,value)
    monkeypatch.setattr(placement,'validate_plan',lambda p,devices:p)
    calls=[]
    def supervise(commands,environments,**kwargs):
        calls.append((commands,environments,kwargs));return 0
    monkeypatch.setattr(launch_cuda_tp,'supervise',supervise)
    assert launch_cuda_tp.main(['--gpus',','.join(value['ordered_gpu_uuids']),'--numa-plan',str(path),
        '--numa-plan-sha256',sha,'--','MODEL'],_command_prefix=['python','worker.py'])==0
    commands,environments,options=calls[0]
    for command,env in zip(commands,environments):
        assert command[:3]==[launch_cuda_tp.sys.executable,'-m','tensorfold.cuda.rank_placement']
        assert 'worker.py' in command and env['NCCL_IGNORE_CPU_AFFINITY']=='0'
        assert env['CUDA_VISIBLE_DEVICES']in value['ordered_gpu_uuids']
    assert options['pass_fds']
    monkeypatch.setenv('NCCL_IGNORE_CPU_AFFINITY','1')
    with pytest.raises(SystemExit):
        launch_cuda_tp.main(['--gpus',','.join(value['ordered_gpu_uuids']),'--numa-plan',str(path),
                            '--numa-plan-sha256',sha,'--','MODEL'])


@pytest.mark.parametrize('flags', [['--numa-plan','','--numa-plan-sha256',''],['--numa-plan',''],['--numa-plan-sha256','']])
def test_explicit_empty_numa_arguments_cannot_fall_back_to_unbound(monkeypatch,flags):
    from tools import launch_cuda_tp
    monkeypatch.setattr(launch_cuda_tp,'supervise',lambda *a,**k:pytest.fail('empty NUMA request spawned unbound ranks'))
    with pytest.raises((ValueError,SystemExit)):
        launch_cuda_tp.main(['--gpus','0,1',*flags,'--','MODEL'])
