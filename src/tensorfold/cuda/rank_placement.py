"""Strict NUMA rank bootstrap, before exec of Python/Torch/CUDA workers.

The plan binds canonical GPU UUIDs to kernel PCI/NUMA topology and partitions
allowed physical cores, keeping all allowed SMT siblings on one rank. Binding
is process/thread policy, not a claim that shared file pages or TP8 traffic are
local. There is no remote-node or unbound fallback. All failures abort the
supervised cohort. This module and its topology dependency import only stdlib.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import mmap
import os
import stat
import sysconfig
import uuid
from pathlib import Path

SCHEMA = 'tensorfold.gpu-rank-numa-v1'
POLICY = 'gpu-local-disjoint-physical-core-static-bind-v1'
RECEIPT_DIR_ENV = 'TENSORFOLD_NUMA_RECEIPT_DIR'
PLAN_ENV='TENSORFOLD_NUMA_PLAN'
PLAN_SHA_ENV='TENSORFOLD_NUMA_PLAN_SHA256'
RANK_ENV='TENSORFOLD_NUMA_RANK'
BIND_STATIC = 32770
LIMIT = 1 << 20


def _gpu_uuid(value):
    return 'GPU-'+str(uuid.UUID(value.removeprefix('GPU-')))


def _ids(text):
    from tensorfold.cuda.numa_memory import _ids as read_ids
    return read_ids(text)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def file_identity(path):
    path = Path(path)
    with path.open('rb') as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('NUMA authority requires a regular file')
        digest = hashlib.sha256()
        left = before.st_size
        while left:
            block = source.read(min(left, 1 << 20))
            if not block:
                raise ValueError('NUMA authority file truncated')
            digest.update(block); left -= len(block)
        if source.read(1):
            raise ValueError('NUMA authority file grew')
        after = os.fstat(source.fileno())
    def stamp(v):
        return (v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    if stamp(before) != stamp(after) or stamp(before) != stamp(path.stat()):
        raise ValueError('NUMA authority changed during read')
    return {'path':str(path.resolve(strict=True)), 'bytes':before.st_size, 'sha256':digest.hexdigest()}


def _allowed_mems(proc):
    for line in (proc/'self/status').read_text().splitlines():
        if line.startswith('Mems_allowed_list:'):
            return _ids(line.split(':',1)[1])
    raise RuntimeError('kernel memory-node allowance is unavailable')


def _libnuma_path():
    multiarch = sysconfig.get_config_var('MULTIARCH')
    if not isinstance(multiarch,str) or '/' in multiarch:
        raise RuntimeError('native libnuma platform authority is unavailable')
    return (Path('/usr/lib')/multiarch/'libnuma.so.1').resolve(strict=True)


def _package_version():
    path=Path('/var/lib/dpkg/status')
    if not path.is_file():
        raise RuntimeError('selected native libnuma package metadata is unavailable')
    for block in path.read_text().split('\n\n'):
        if block.startswith('Package: libnuma1\n'):
            fields=dict(line.split(': ',1) for line in block.splitlines() if ': ' in line and not line.startswith(' '))
            return {'name':'libnuma1','version':fields['Version'],'architecture':fields['Architecture']}
    raise RuntimeError('selected libnuma1 package is not registered')


def _topology(devices, cpus, mems, *, proc, sysfs):
    from tensorfold.cuda.numa_memory import discover_gpu_uuid

    gpus=[]; nodes={}
    online=_ids((sysfs/'devices/system/cpu/online').read_text())
    for gpu_uuid in devices:
        placement=discover_gpu_uuid(gpu_uuid,proc=proc,sysfs=sysfs,affinity=cpus)
        if placement.node not in mems:
            raise RuntimeError('GPU node is outside the declared memory-node allowance')
        eligible=set(placement.cpus)&online
        groups={}
        for cpu in sorted(eligible):
            root=sysfs/f'devices/system/cpu/cpu{cpu}/topology'
            key=(int((root/'physical_package_id').read_text()),int((root/'core_id').read_text()))
            groups.setdefault(key,[]).append(cpu)
        cores=[{'socket':s,'core':c,'cpus':values}for(s,c),values in sorted(groups.items())]
        node={'node':placement.node,'cores':cores}
        if placement.node in nodes and nodes[placement.node]!=node:
            raise RuntimeError('GPUs on one NUMA node disagree about its CPU topology')
        nodes[placement.node]=node
        gpus.append({'gpu_uuid':placement.gpu_uuid,'pci_bus_id':placement.pci_bus_id,'node':placement.node})
    return {'gpus':gpus,'nodes':[nodes[n]for n in sorted(nodes)]}


def build_plan(devices, *, proc=Path('/proc'), sysfs=Path('/sys'), affinity=None, library_path=None):
    if (type(devices) not in (list,tuple) or len(devices) not in (2,4,8)
            or any(type(v) is not str or not v.startswith('GPU-') for v in devices)):
        raise ValueError('NUMA plans require2/4/8 explicit full GPU UUIDs in rank order')
    uuids=[_gpu_uuid(value) for value in devices]
    if len(set(uuids))!=len(uuids):
        raise ValueError('NUMA rank GPU UUIDs must be distinct')
    cpus=set(os.sched_getaffinity(0) if affinity is None else affinity)
    mems=_allowed_mems(proc)
    topology=_topology(uuids,cpus,mems,proc=proc,sysfs=sysfs)
    ranks=[None]*len(uuids)
    for node in topology['nodes']:
        peers=[i for i,gpu in enumerate(topology['gpus'])if gpu['node']==node['node']]
        cores=node['cores']
        if len(cores)<len(peers):
            raise RuntimeError('GPU-local allowed physical cores cannot give every rank a disjoint nonempty group')
        q,extra=divmod(len(cores),len(peers));start=0
        for ordinal,rank in enumerate(peers):
            take=q+(ordinal<extra);assigned=cores[start:start+take];start+=take
            ranks[rank]={'rank':rank,**topology['gpus'][rank], 'cores':assigned,
                         'cpus':sorted(cpu for core in assigned for cpu in core['cpus'])}
    library=file_identity(library_path or _libnuma_path())
    return {'schema':SCHEMA,'policy':POLICY,'ordered_gpu_uuids':uuids,
            'source_allowed_cpus':sorted(cpus),'source_allowed_mems':sorted(mems),
            'topology':topology,'topology_sha256':hashlib.sha256(_canonical(topology)).hexdigest(),
            'cpuset_cpus':sorted({cpu for rank in ranks for cpu in rank['cpus']}),
            'cpuset_mems':sorted({rank['node']for rank in ranks}),
            'ranks':ranks,'libnuma':library,'libnuma_package':_package_version()}


def load_plan(path, expected_sha256):
    path=Path(path)
    if (not path.is_absolute() or path.is_symlink() or type(expected_sha256)is not str
            or len(expected_sha256)!=64 or any(c not in '0123456789abcdef' for c in expected_sha256)):
        raise ValueError('NUMA plan requires its absolute regular path and lowercase SHA256')
    with path.open('rb') as source:
        info=os.fstat(source.fileno())
        if not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=LIMIT:
            raise ValueError('NUMA plan exceeds its regular bounded schema')
        data=source.read(info.st_size+1)
    if len(data)!=info.st_size or hashlib.sha256(data).hexdigest()!=expected_sha256:
        raise ValueError('NUMA plan identity differs')
    def pairs(values):
        result={}
        for key,value in values:
            if key in result:raise ValueError('duplicate NUMA plan key')
            result[key]=value
        return result
    value=json.loads(data,object_pairs_hook=pairs)
    if (type(value)is not dict or value.get('schema')!=SCHEMA or value.get('policy')!=POLICY
            or set(value)!={'schema','policy','ordered_gpu_uuids','source_allowed_cpus','source_allowed_mems',
                           'topology','topology_sha256','cpuset_cpus','cpuset_mems','ranks','libnuma','libnuma_package'}):
        raise ValueError('NUMA plan schema differs')
    if (type(value['ordered_gpu_uuids'])is not list or len(value['ordered_gpu_uuids'])not in(2,4,8)
            or any(type(v)is not str or not v.startswith('GPU-') or _gpu_uuid(v)!=v for v in value['ordered_gpu_uuids'])
            or len(set(value['ordered_gpu_uuids']))!=len(value['ordered_gpu_uuids'])):
        raise ValueError('NUMA plan GPU UUID order is invalid')
    for key in ('source_allowed_cpus','source_allowed_mems','cpuset_cpus','cpuset_mems'):
        values=value[key]
        if (type(values)is not list or not values or any(type(v)is not int or v<0 for v in values)
                or values!=sorted(set(values))):
            raise ValueError('NUMA plan CPU/node mask is invalid')
    if (not set(value['cpuset_cpus'])<=set(value['source_allowed_cpus'])
            or not set(value['cpuset_mems'])<=set(value['source_allowed_mems'])
            or hashlib.sha256(_canonical(value['topology'])).hexdigest()!=value['topology_sha256']
            or type(value['ranks'])is not list or len(value['ranks'])!=len(value['ordered_gpu_uuids'])):
        raise ValueError('NUMA plan topology/mask authority differs')
    seen_cpus=set();seen_cores=set();nodes=set()
    for rank,target in enumerate(value['ranks']):
        if (type(target)is not dict or set(target)!={'rank','gpu_uuid','pci_bus_id','node','cores','cpus'}
                or type(target['rank'])is not int or target['rank']!=rank
                or target['gpu_uuid']!=value['ordered_gpu_uuids'][rank]
                or type(target['node'])is not int or target['node']<0
                or type(target['cores'])is not list or not target['cores']):
            raise ValueError('NUMA per-rank topology is invalid')
        group=[]
        for core in target['cores']:
            if (type(core)is not dict or set(core)!={'socket','core','cpus'}
                    or any(type(core[k])is not int or core[k]<0 for k in ('socket','core'))
                    or type(core['cpus'])is not list or not core['cpus']
                    or any(type(c)is not int or c<0 for c in core['cpus'])
                    or core['cpus']!=sorted(set(core['cpus']))):
                raise ValueError('NUMA physical core group is invalid')
            key=(core['socket'],core['core'])
            if key in seen_cores:raise ValueError('NUMA physical core groups overlap')
            seen_cores.add(key);group.extend(core['cpus'])
        if (type(target['cpus'])is not list or any(type(c)is not int for c in target['cpus'])
                or target['cpus']!=sorted(group) or len(group)!=len(set(group)) or seen_cpus.intersection(group)):
            raise ValueError('NUMA per-rank CPU masks overlap or disagree')
        seen_cpus.update(group);nodes.add(target['node'])
    if sorted(seen_cpus)!=value['cpuset_cpus'] or sorted(nodes)!=value['cpuset_mems']:
        raise ValueError('NUMA Docker mask unions differ from rank assignments')
    return value


def validate_plan(plan, devices=None, *, proc=Path('/proc'), sysfs=Path('/sys')):
    if devices is not None and list(devices)!=plan['ordered_gpu_uuids']:
        raise ValueError('NUMA plan must match exact canonical UUID rank order; numeric ordinals are not inferred')
    allowed=set(os.sched_getaffinity(0));mems=_allowed_mems(proc)
    if not set(plan['cpuset_cpus'])<=allowed or not set(plan['cpuset_mems'])<=mems:
        raise RuntimeError('NUMA plan is outside the actual container CPU/memory-node allowance')
    current=build_plan(plan['ordered_gpu_uuids'],proc=proc,sysfs=sysfs,
                       affinity=set(plan['source_allowed_cpus']))
    # Source allowances are producer provenance; the runtime may narrow the
    # container to the selected union. Every assigned CPU/node was checked.
    current['source_allowed_mems']=plan['source_allowed_mems']
    if _canonical(current)!=_canonical(plan):
        raise RuntimeError('NUMA plan topology/library/core assignment changed')
    return plan


class MemoryPolicy:
    """Actual libnuma syscalls, never environment labels or a node0 fallback."""
    def __init__(self, expected):
        actual=file_identity(_libnuma_path())
        if actual!=expected:raise RuntimeError('NUMA native library differs from the sealed plan')
        self.library=ctypes.CDLL(actual['path'],use_errno=True)
        self.identity=actual
        self.get=self.library.get_mempolicy
        self.get.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong,ctypes.c_void_p,ctypes.c_ulong]
        self.get.restype=ctypes.c_int
        self.set=self.library.set_mempolicy
        self.set.argtypes=[ctypes.c_int,ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong];self.set.restype=ctypes.c_int

    def _mask(self,node):
        width=8*ctypes.sizeof(ctypes.c_ulong);words=node//width+1
        mask=(ctypes.c_ulong*words)();mask[node//width]=1<<(node%width)
        return mask,words*width

    def bind(self,node):
        if type(node)is not int or node<0:raise ValueError('physical NUMA node must be explicit and nonnegative')
        mask,bits=self._mask(node)
        if self.set(BIND_STATIC,mask,bits):raise OSError(ctypes.get_errno(),'strict NUMA set_mempolicy')
        return self.read(node)

    def read(self,node):
        mask,bits=self._mask(node);mode=ctypes.c_int()
        if self.get(ctypes.byref(mode),mask,bits,None,0):raise OSError(ctypes.get_errno(),'NUMA get_mempolicy')
        width=8*ctypes.sizeof(ctypes.c_ulong)
        selected=[i for i in range(bits)if mask[i//width]&(1<<(i%width))]
        if mode.value!=BIND_STATIC or selected!=[node]:raise RuntimeError('effective physical static BIND policy differs')
        return {'mode':mode.value,'nodes':selected}

    def page_nodes(self,address,bytes_,node):
        if type(address)is not int or address<=0 or type(bytes_)is not int or bytes_<=0:
            raise ValueError('owned page query requires its positive live address/extent')
        page=os.sysconf('SC_PAGE_SIZE');nodes=[]
        first=address-address%page;end=(address+bytes_+page-1)//page*page
        for location in range(first,end,page):
            actual=ctypes.c_int(-1)
            if self.get(ctypes.byref(actual),None,0,ctypes.c_void_p(location),3):
                raise OSError(ctypes.get_errno(),'NUMA owned-page location query')
            if actual.value!=node:raise RuntimeError('owned host page is outside the GPU-local node')
            nodes.append(actual.value)
        return nodes

    def anonymous_probe(self,node):
        page=os.sysconf('SC_PAGE_SIZE')
        with mmap.mmap(-1,4*page,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS) as memory:
            for offset in range(0,len(memory),page):memory[offset]=0
            address=ctypes.addressof(ctypes.c_char.from_buffer(memory))
            nodes=self.page_nodes(address,len(memory),node)
        return {'bytes':4*page,'page_nodes':nodes}


def _receipt(path,value):
    payload=_canonical(value)+b'\n'
    if len(payload)>LIMIT:raise ValueError('NUMA receipt exceeds its bound')
    with Path(path).open('xb') as output:
        output.write(payload);output.flush();os.fsync(output.fileno())


def bootstrap(plan,path_sha,rank,command,*,plan_path):
    validate_plan(plan)
    if os.environ.get('NCCL_IGNORE_CPU_AFFINITY','0')!='0':
        raise ValueError('strict NUMA workers require NCCL_IGNORE_CPU_AFFINITY=0')
    if type(rank)is not int or not 0<=rank<len(plan['ranks']) or not command:
        raise ValueError('NUMA bootstrap needs its exact rank and executable argument sequence')
    target=plan['ranks'][rank]
    if os.environ.get('CUDA_VISIBLE_DEVICES')!=target['gpu_uuid']:
        raise ValueError('rank CUDA visibility differs from its planned physical GPU UUID')
    directory=os.environ.get(RECEIPT_DIR_ENV)
    if not directory or not Path(directory).is_absolute() or not Path(directory).is_dir():
        raise ValueError('NUMA bootstrap requires an existing absolute receipt directory')
    os.sched_setaffinity(0,set(target['cpus']))
    if sorted(os.sched_getaffinity(0))!=target['cpus']:raise RuntimeError('strict rank CPU affinity differs')
    policy=MemoryPolicy(plan['libnuma']);readback=policy.bind(target['node'])
    pages=policy.anonymous_probe(target['node'])
    value={'schema':SCHEMA,'policy':POLICY,'rank':rank,'pid':os.getpid(),'gpu_uuid':target['gpu_uuid'],
           'pci_bus_id':target['pci_bus_id'],'node':target['node'],'cpus':target['cpus'],
           'allowed_mems':sorted(_allowed_mems(Path('/proc'))),'memory_policy':readback,'anonymous':pages,
           'plan_sha256':path_sha,'topology_sha256':plan['topology_sha256'],'libnuma':policy.identity,
           'bootstrap_source':file_identity(__file__),'stage':'bootstrap',
           'scope':'worker policy and owned anonymous pages; shared file pages/necessary TP8 traffic excluded'}
    _receipt(Path(directory)/f'rank-{rank}-bootstrap.json',value)
    environment=dict(os.environ)
    environment[PLAN_ENV]=str(Path(plan_path).resolve(strict=True))
    environment[PLAN_SHA_ENV]=path_sha;environment[RANK_ENV]=str(rank)
    os.execvpe(command[0],command,environment)


def validate_receipt(record,plan,rank,stage,*,plan_sha256,source_sha256,pid=None):
    """Pure stdlib host admission of source-bound per-rank placement evidence.

    The host separately verifies live PID/cohort/container ownership. This
    checks the worker's declared source/plan authority and exact policy/page/
    thread invariants; it does not inspect processes or load native libraries.
    """
    if type(rank)is not int or not 0<=rank<len(plan['ranks']) or stage not in ('bootstrap','post-cuda','ready'):
        raise ValueError('NUMA receipt rank/stage is invalid')
    expected=plan['ranks'][rank]
    common={'schema','policy','stage','rank','pid','gpu_uuid','pci_bus_id','node','cpus','allowed_mems',
            'memory_policy','anonymous','plan_sha256','topology_sha256','libnuma','bootstrap_source','scope'}
    fields=common if stage=='bootstrap' else common|{'cuda_pinned','threads','actual_cuda_gpu_uuid','cuda_device'}
    if (type(record)is not dict or set(record)!=fields or record['schema']!=SCHEMA or record['policy']!=POLICY
            or record['stage']!=stage
            or type(record['rank'])is not int or record['rank']!=rank
            or type(record['pid'])is not int or record['pid']<=0 or pid is not None and record['pid']!=pid
            or record['plan_sha256']!=plan_sha256 or record['topology_sha256']!=plan['topology_sha256']
            or record['libnuma']!=plan['libnuma']):
        raise ValueError('NUMA receipt source/plan/rank/cohort identity differs')
    source=record['bootstrap_source']
    if (type(source)is not dict or set(source)!={'path','bytes','sha256'}
            or source['sha256']!=source_sha256 or type(source['bytes'])is not int or source['bytes']<=0):
        raise ValueError('NUMA receipt verifier source differs')
    if (type(record['node'])is not int or type(record['cpus'])is not list
            or any(type(c)is not int for c in record['cpus'])
            or any(record[key]!=expected[key]for key in ('gpu_uuid','pci_bus_id','node','cpus'))):
        raise ValueError('NUMA receipt GPU/local CPU assignment differs')
    memory=record['memory_policy']
    if (type(memory)is not dict or set(memory)!={'mode','nodes'} or type(memory['mode'])is not int
            or type(memory['nodes'])is not list or any(type(n)is not int for n in memory['nodes'])
            or memory!={'mode':BIND_STATIC,'nodes':[expected['node']]}):
        raise ValueError('NUMA receipt did not prove exact physical static BIND')
    allowed=record['allowed_mems']
    if (type(allowed)is not list or any(type(n)is not int or n<0 for n in allowed)
            or allowed!=sorted(set(allowed)) or expected['node']not in allowed):
        raise ValueError('NUMA receipt memory allowance excludes the GPU node')
    anonymous=record['anonymous']
    if (type(anonymous)is not dict or set(anonymous)!={'bytes','page_nodes'}
            or type(anonymous['bytes'])is not int or anonymous['bytes']!=4*os.sysconf('SC_PAGE_SIZE')
            or type(anonymous['page_nodes'])is not list or any(type(n)is not int for n in anonymous['page_nodes'])
            or anonymous['page_nodes']!=[expected['node']]*4):
        raise ValueError('NUMA receipt anonymous allocation is not GPU-local')
    if stage!='bootstrap':
        pinned=record['cuda_pinned']
        if (record['actual_cuda_gpu_uuid']!=expected['gpu_uuid'] or record['cuda_device']not in ('cuda:0','0')
                or type(pinned)is not dict or set(pinned)!={'bytes','pinned','page_nodes'}
                or pinned['pinned']is not True
                or type(pinned['bytes'])is not int or pinned['bytes']!=anonymous['bytes'] or type(pinned['page_nodes'])is not list
                or not 4<=len(pinned['page_nodes'])<=5
                or any(type(n)is not int or n!=expected['node']for n in pinned['page_nodes'])):
            raise ValueError('NUMA receipt CUDA device/pinned allocation differs')
        threads=record['threads'];seen=set()
        if type(threads)is not list or not threads:raise ValueError('NUMA receipt thread audit missing')
        for thread in threads:
            if (type(thread)is not dict or set(thread)!={'tid','name','cpus'} or type(thread['tid'])is not int
                    or thread['tid']<=0 or thread['tid']in seen or type(thread['name'])is not str
                    or type(thread['cpus'])is not list or not thread['cpus']
                    or any(type(c)is not int for c in thread['cpus'])
                    or thread['cpus']!=sorted(set(thread['cpus'])) or not set(thread['cpus'])<=set(expected['cpus'])):
                raise ValueError('NUMA receipt worker/NCCL/helper thread escaped its CPU group')
            seen.add(thread['tid'])
    if len(_canonical(record))>LIMIT:raise ValueError('NUMA receipt exceeds its bound')
    return json.loads(_canonical(record))


def _thread_ids(root):
    # Even the smallest possible emitted record consumes these bytes. More
    # entries cannot fit the existing receipt envelope; this is not a new
    # supported-thread quota. scandir avoids listdir's unbounded intermediate.
    maximum=LIMIT//(len(_canonical({'tid':1,'name':'','cpus':[0]}))+1)
    ids=set()
    with os.scandir(root) as entries:
        for index,entry in enumerate(entries):
            if index>=maximum:raise ValueError('thread enumeration cannot fit the NUMA receipt envelope')
            if not entry.name.isascii() or not entry.name.isdecimal():raise RuntimeError('unexpected proc task identifier')
            ids.add(entry.name)
    return ids


def _thread_affinities(cpus):
    root=Path('/proc/self/task');before=_thread_ids(root)
    result=[];retained_bytes=2
    for tid in sorted(before,key=int):
        # The proc directory pins identity/path lookup, not thread liveness.
        # Exit/status races and a changed thread set fail this bounded snapshot;
        # no signal, affinity mutation or foreign-thread syscall is issued.
        descriptor=os.open(root/tid,os.O_RDONLY|os.O_DIRECTORY|os.O_CLOEXEC)
        try:
            fd=os.open('status',os.O_RDONLY|os.O_CLOEXEC,dir_fd=descriptor)
            with os.fdopen(fd) as stream:status=stream.read(64<<10)
        finally:os.close(descriptor)
        fields=dict(line.split(':',1)for line in status.splitlines()if ':' in line)
        actual=_ids(fields['Cpus_allowed_list'])
        if not actual or not actual<=set(cpus):
            raise RuntimeError('worker/NCCL/helper thread escaped its planned physical core group')
        row={'tid':int(tid),'name':fields['Name'].strip(),'cpus':sorted(actual)}
        retained_bytes+=len(_canonical(row))+1
        if retained_bytes>LIMIT:raise ValueError('thread records cannot fit the NUMA receipt envelope')
        result.append(row)
    if _thread_ids(root)!=before:
        raise RuntimeError('worker thread set changed during the bounded NUMA placement audit')
    return result


def verify_worker_from_env(torch_module,device,*,stage):
    """Startup-only post-CUDA/ready proof; caller invokes before requests.

    No policy is changed here. Pinned/anonymous probes are owned and bounded;
    shared file cache and arbitrary allocator/driver buffers are not certified.
    """
    values=[os.environ.get(key) for key in (PLAN_ENV,PLAN_SHA_ENV,RANK_ENV)]
    if not any(v is not None for v in values):return None  # Other declared legacy regions.
    if not all(type(v)is str and v for v in values) or stage not in ('post-cuda','ready'):
        raise ValueError('NUMA worker requires its complete inherited plan/rank and startup stage')
    path,sha,rank_text=values
    if not rank_text.isdecimal():raise ValueError('NUMA inherited rank is invalid')
    plan=load_plan(path,sha);rank=int(rank_text)
    if not 0<=rank<len(plan['ranks']):raise ValueError('NUMA inherited rank is outside its plan')
    target=plan['ranks'][rank]
    if sorted(os.sched_getaffinity(0))!=target['cpus']:
        raise RuntimeError('worker CPU group changed across exec/Torch/CUDA initialization')
    placement=_topology([target['gpu_uuid']],set(target['cpus']),_allowed_mems(Path('/proc')),
                        proc=Path('/proc'),sysfs=Path('/sys'))['gpus'][0]
    if placement!={key:target[key]for key in ('gpu_uuid','pci_bus_id','node')}:
        raise RuntimeError('worker GPU/PCI/NUMA topology changed')
    policy=MemoryPolicy(plan['libnuma']);readback=policy.read(target['node'])
    props=torch_module.cuda.get_device_properties(device)
    actual_uuid=_gpu_uuid(str(props.uuid))
    if actual_uuid!=target['gpu_uuid']:
        raise RuntimeError('actual CUDA logical device differs from the planned physical GPU')
    anonymous=policy.anonymous_probe(target['node'])
    size=4*os.sysconf('SC_PAGE_SIZE')
    # Retain through the positive CUDA fence before dropping the pinned owner.
    pinned=torch_module.empty(size,dtype=torch_module.uint8,device='cpu',pin_memory=True)
    if not pinned.is_pinned():raise RuntimeError('CUDA host probe is not actually pinned')
    pinned.zero_()
    pinned_nodes=policy.page_nodes(pinned.data_ptr(),pinned.numel()*pinned.element_size(),target['node'])
    torch_module.cuda.synchronize(device)
    threads=_thread_affinities(target['cpus'])
    value={'schema':SCHEMA,'policy':POLICY,'stage':stage,'rank':rank,'pid':os.getpid(),
           'gpu_uuid':target['gpu_uuid'],'pci_bus_id':target['pci_bus_id'],'node':target['node'],
           'cpus':target['cpus'],'allowed_mems':sorted(_allowed_mems(Path('/proc'))),
           'memory_policy':readback,'anonymous':anonymous,'cuda_pinned':{'bytes':size,'pinned':True,'page_nodes':pinned_nodes},
           'threads':threads,'actual_cuda_gpu_uuid':actual_uuid,'cuda_device':str(device),'plan_sha256':sha,'topology_sha256':plan['topology_sha256'],
           'libnuma':policy.identity,'bootstrap_source':file_identity(__file__),
           'scope':'startup-owned allocations and current thread affinities; no shared-file-cache or all-driver-buffer claim'}
    directory=os.environ.get(RECEIPT_DIR_ENV)
    if not directory or not Path(directory).is_absolute():raise ValueError('NUMA worker receipt directory missing')
    _receipt(Path(directory)/f'rank-{rank}-{stage}.json',value)
    return value


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='operation',required=True)
    create=sub.add_parser('plan');create.add_argument('--gpus',required=True);create.add_argument('--output',type=Path,required=True)
    run=sub.add_parser('exec');run.add_argument('--plan',type=Path,required=True);run.add_argument('--sha256',required=True)
    run.add_argument('--rank',type=int,required=True);run.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args(argv)
    if args.operation=='plan':
        value=build_plan(args.gpus.split(','));_receipt(args.output,value)
        print(json.dumps({'path':str(args.output.resolve()),'sha256':file_identity(args.output)['sha256'],
                          'cpuset_cpus':value['cpuset_cpus'],'cpuset_mems':value['cpuset_mems'],'policy':POLICY}))
        return 0
    command=args.command[1:] if args.command[:1]==['--'] else args.command
    try:
        bootstrap(load_plan(args.plan,args.sha256),args.sha256,args.rank,command,plan_path=args.plan)
    except BaseException as primary:
        from .cohort import notify_failure
        try:
            notify_failure()
        except BaseException as secondary:  # noqa: BLE001 - preserve both bootstrap and notification failures.
            from tensorfold.cleanup import raise_failures
            raise_failures(primary,[secondary])
        raise
    return 0


if __name__=='__main__':
    raise SystemExit(main())
