"""Campaign preparation authority and installed-parser producer seam; no CUDA."""
import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

from tools import v100_onecat_campaign as campaign


BASE = {'kernel_family':'sm70-paged-owner-rn-ex2-p2051-v1','graph_family':'uncaptured',
        'graph_budget_bytes':0,'trial':True,'collective_family':'tp8-allgather-v1','rn_heads_per_block':12}
FLAGS = {'native_family':'owned-native-family-v1','prepared_linears':'owned-prepared-m32-v1',
         'prepared_router':'owned-serial-router-v1','coordinate_staging':'owned-host-int64-coordinates-v1'}


def invoke(tmp_path,**selection):
    return campaign.inputs(tmp_path/'source',tmp_path/'output','image',tmp_path/'checked',
        tmp_path/'models','target.gguf','mtp.gguf',tmp_path/'cache','model','q8',1024,128,3,
        '1,2,4,8,16,24',2,'4g',120,**selection)


def request(tmp_path,raw=None):
    path=tmp_path/'preparation.json'
    raw=json.dumps(BASE).encode() if raw is None else raw
    path.write_bytes(raw)
    return path,hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize('flag',list(FLAGS))
def test_full_data_conflicts_with_each_legacy_flag_before_effects(tmp_path,monkeypatch,flag):
    monkeypatch.setattr(campaign,'frozen',lambda *a:pytest.fail('no source/container effects allowed'))
    with pytest.raises(ValueError,match='legacy preparation flags'):
        invoke(tmp_path,preparation=tmp_path/'unused',preparation_sha256='a'*64,**{flag:FLAGS[flag]})


@pytest.mark.parametrize('fault',['missing_path','missing_hash','bad_hash_format','wrong_hash','oversize','symlink'])
def test_request_authority_refuses_before_source_or_container_effects(tmp_path,monkeypatch,fault):
    path,digest=request(tmp_path)
    if fault=='missing_path':path=None
    if fault=='missing_hash':digest=None
    if fault=='bad_hash_format':digest='A'*64
    if fault=='wrong_hash':digest='0'*64
    if fault=='oversize':
        with path.open('wb')as stream:stream.truncate(campaign.EXECUTION_MANIFEST_LIMIT+1)
    if fault=='symlink':
        other=tmp_path/'alias';other.symlink_to(path);path=other
    monkeypatch.setattr(campaign,'frozen',lambda *a:pytest.fail('no source/container effects allowed'))
    with pytest.raises(ValueError):invoke(tmp_path,preparation=path,preparation_sha256=digest)


@pytest.mark.parametrize('changed',[False,True])
def test_host_stages_exact_readonly_authority_and_retains_normalized_receipt(tmp_path,monkeypatch,changed):
    path,digest=request(tmp_path);raw=path.read_bytes();normalized={**BASE,'row_buckets':{}}
    models=tmp_path/'models';models.mkdir()
    for name in ('target.gguf','mtp.gguf'):(models/name).write_bytes(b'fixture')
    monkeypatch.setattr(campaign,'frozen',lambda *a:{'source_uid':'a'*64,'files':{}})
    monkeypatch.setattr(campaign,'immutable_image',lambda *a:{'id':'image','reference':'image'})
    monkeypatch.setattr(campaign,'installed_check_receipt',lambda *a:{'accepted':True})
    monkeypatch.setattr(campaign,'verify_source',lambda *a:None)
    calls=[]
    def produce(argv,output,timeout):
        calls.append(argv)
        (output/'profile.json').write_text(json.dumps({'preparation':normalized}))
        (output/'matrix.json').write_text('{}')
        if changed:path.write_bytes(b'changed')
    monkeypatch.setattr(campaign,'container',produce)
    if changed:
        with pytest.raises(ValueError,match='changed through input production'):
            invoke(tmp_path,preparation=path,preparation_sha256=digest)
        assert not (tmp_path/'output/inputs.json').exists()
    else:
        invoke(tmp_path,preparation=path,preparation_sha256=digest)
        receipt=json.loads((tmp_path/'output/inputs.json').read_bytes())
        assert receipt['preparation_request']=={'path':str(path),'bytes':len(raw),'sha256':digest}
        assert receipt['normalized_preparation']==normalized
        assert receipt['quality_accepted'] is receipt['P3_accepted'] is False
    assert str(path)+':/operator/preparation.json:ro' in calls[0]
    assert calls[0][-2:]==['/operator/preparation.json',digest]
    assert (tmp_path/'output/preparation-request.json').read_bytes()==raw


def producer(tmp_path,monkeypatch,*,data=None,flags=None,wrong_hash=False,model_calls=None):
    """Run the real producer through its manifest/matrix computation.

    Only unrelated model discovery and natural-text matrix preparation use
    bounded fixtures. The installed runtime parser and strict JSON reader are
    real; the unchanged final file-publication loop is outside this contract.
    """
    import torch
    from tensorfold import hub
    from tensorfold.families.qwen4_exp.cuda import execution_identity
    from tools import bench_concurrent
    assert not torch.cuda.is_initialized()
    item=tmp_path/'artifact';item.write_bytes(b'abc');calls=[]if model_calls is None else model_calls
    monkeypatch.setattr(hub,'resolve',lambda *a,**kw:calls.append(a)or tmp_path)
    monkeypatch.setattr(execution_identity,'artifact_files',lambda *a:({'metadata/config.json':item},None))
    monkeypatch.setattr(bench_concurrent,'prepare_matrix',lambda *a:{'profile':a[2],'levels':a[3]})
    monkeypatch.setitem(sys.modules,'bench_concurrent',bench_concurrent)
    monkeypatch.setitem(sys.modules,'v100_onecat_campaign',campaign)
    path,digest=('','')if data is None else request(tmp_path,data)
    args=['producer','target','mtp','model','q8','1024','128','3','1,2,4,8,16,24',
          *(FLAGS[name]if flags and name in flags else ''for name in FLAGS),str(path),'0'*64 if wrong_hash else digest]
    monkeypatch.setattr(sys,'argv',args)
    # The producer adds its fixed mounted tools path; restore it after execution.
    monkeypatch.setattr(sys,'path',list(sys.path))
    tree=ast.parse(campaign._PREPARE_INPUTS)
    publication=next(i for i,node in enumerate(tree.body)if isinstance(node,ast.For)
        and isinstance(node.target,ast.Tuple)and [item.id for item in node.target.elts]==['name','value'])
    namespace={}
    try:exec(compile(ast.Module(body=tree.body[:publication],type_ignores=[]),'actual-campaign-producer','exec'),namespace)
    finally:assert not torch.cuda.is_initialized()
    return namespace,calls


@pytest.mark.parametrize('flags',[None,*({name}for name in FLAGS)])
def test_legacy_producer_default_and_four_independent_flags_are_unchanged(tmp_path,monkeypatch,flags):
    namespace,_=producer(tmp_path,monkeypatch,flags=flags)
    assert namespace['manifest']['preparation']=={**BASE,**({name:FLAGS[name]for name in flags}if flags else {})}


def test_complete_vendor_data_uses_actual_runtime_normalization(tmp_path,monkeypatch):
    selection={**BASE,'prepared_linears':FLAGS['prepared_linears'],'native_gdn_capture':True,
        'dense_projections':{'backend':'cublaslt','geometry':'fixed-m32','roles':['prefill'],'panel_rows':32}}
    namespace,calls=producer(tmp_path,monkeypatch,data=json.dumps(selection).encode())
    assert calls==[('target',)]
    assert namespace['manifest']=={'schema':'tensorfold.execution-profile-manifest','version':1,
        'artifacts':{'metadata/config.json':{'size':3,'sha256':'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'}},
        'preparation':{**selection,'row_buckets':{}}}
    assert namespace['matrix']=={'profile':'q8','levels':[1,2,4,8,16,24]}


@pytest.mark.parametrize('raw,wrong_hash',[(b'{"trial":true,"trial":false}',False),
    (json.dumps({**BASE,'unexpected':True}).encode(),False),
    (json.dumps({**BASE,'g3_prefill':{}}).encode(),False),
    (json.dumps({**BASE,'prepared_linears':FLAGS['prepared_linears'],'dense_projections':{
        'backend':'cublas','geometry':'fixed-m32','roles':['decode'],'panel_rows':32}}).encode(),False),
    (json.dumps(BASE).encode(),True)])
def test_invalid_full_data_never_reaches_model_hashing(tmp_path,monkeypatch,raw,wrong_hash):
    calls=[]
    with pytest.raises(ValueError):producer(tmp_path,monkeypatch,data=raw,wrong_hash=wrong_hash,model_calls=calls)
    assert calls==[]


def test_cli_passes_explicit_preparation_authority_without_changing_legacy_defaults(monkeypatch):
    calls=[];monkeypatch.setattr(campaign,'inputs',lambda *a,**kw:calls.append(kw))
    argv=['inputs','--source','/source','--out','/out','--image','image','--installed-check','/checked',
          '--models','/models','--target','target','--mtp','mtp','--cache','/cache','--name','model']
    campaign.main(argv+['--preparation','/request.json','--preparation-sha256','a'*64])
    assert calls[0]['preparation']==Path('/request.json')and calls[0]['preparation_sha256']=='a'*64
    assert all(calls[0][name]is None for name in FLAGS)
