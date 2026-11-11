"""CPU contracts only: no HTTP/model/GPU effects or invented media-work count."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools import gemma_shared_prefix_media_qualify as tool


def body(responses=False):
    value = {'model':'gemma4-26b-a4b','temperature':0,'draft':True,'return_token_ids':True,
             'chat_template_kwargs':{'enable_thinking':False},'seed':7}
    value.update({'input':[{'role':'user','content':'fixture'}],'max_output_tokens':128,'store':True}
                 if responses else {'messages':[{'role':'user','content':'fixture'}],'max_tokens':128})
    return value


def prepared(request=None,identities=None,ids=None,blocks=None):
    request = body() if request is None else request
    identity = ['gemma-prepared-frame-v1','image','a'*64,None,0,2,[18,768],[18,2],'c'*64,'b'*64]
    return {'schema':'tensorfold.prepared-gemma-media-witness-v1',
            'body_sha256':hashlib.sha256(tool.semantic_bytes(request)).hexdigest(),
            'token_ids':[2,258880,258880,42] if ids is None else ids,
            'blocks':[[1,3]] if blocks is None else blocks,
            'frame_identities':[identity] if identities is None else identities,
            'media_token_ids':[258880,258884],'processor_binding':'b'*64}


def response():
    return {'id':'resp_new','object':'response','created_at':123,'status':'completed','error':None,
            'incomplete_details':None,'model':'gemma4-26b-a4b','store':True,'temperature':0,'max_output_tokens':128,
            'output':[{'id':'msg_new','type':'message','role':'assistant','status':'completed',
                       'content':[{'type':'output_text','text':'red square','annotations':[],'logprobs':[]}]}],
            'usage':{'input_tokens':286,'output_tokens':2,'total_tokens':288,
                     'input_tokens_details':{'cached_tokens':0},'output_tokens_details':{'reasoning_tokens':0}},
            'tensorfold':{'token_ids':[10,106],'token_sha':'gold','rounds':1,'drafted':4,'accepted':1}}


def test_exact_response_semantics_and_ids_retained_while_operational_cache_differs():
    first = response()
    second = copy.deepcopy(first)
    second.update(id='resp_other',created_at=456,completed_at=457)
    second['output'][0]['id'] = 'msg_other'
    second['usage']['input_tokens_details']['cached_tokens'] = 286
    second['tensorfold'].update(decode_s=2.,cached=286,cache_publication={'matched_tokens':315})
    tool.compare(first,second,True)
    assert first['usage']['input_tokens_details']['cached_tokens']==0
    for field in ('ids','text','usage','new-semantic','draft-work'):
        changed = copy.deepcopy(second)
        if field=='ids':
            changed['tensorfold']['token_ids'][0] = 11
        elif field=='text':
            changed['output'][0]['content'][0]['text'] = 'blue circle'
        elif field=='usage':
            changed['usage']['output_tokens_details']['reasoning_tokens'] = 1
        elif field=='new-semantic':
            changed['metadata'] = {'literal':'must remain'}
        else:
            changed['tensorfold']['accepted'] = 0
        with pytest.raises(ValueError):
            tool.compare(first,changed,True)


def test_historical_profile_only_counter_disposition_does_not_drop_quality():
    first,second = response(),response()
    second['tensorfold'].update(rounds=2,drafted=8,accepted=3,min_rows=1,stats_version=2)
    tool.compare(first,second,True,historical=True)
    with pytest.raises(ValueError):
        tool.compare(first,second,True)
    second['tensorfold']['token_sha'] = 'different-token-gold'
    with pytest.raises(ValueError):
        tool.compare(first,second,True,historical=True)


def test_prepared_fields_bind_exact_body_and_native_atomic_ranges():
    value = prepared()
    assert tool.metadata(value,body()) is value
    assert tool.legal_cut(1,value)==(0,)
    assert tool.legal_cut(3,value)==()
    with pytest.raises(ValueError):
        tool.legal_cut(2,value)
    changed = body()
    changed['messages'][0]['content'] = 'changed literal'
    with pytest.raises(ValueError):
        tool.metadata(value,changed)
    wrong = copy.deepcopy(value)
    wrong['blocks'] = [[0,2]]
    with pytest.raises(ValueError):
        tool.metadata(wrong,body())


def test_same_growth_future_changed_identity_contract():
    cold = prepared()
    grow = prepared(ids=[2,258880,258880,42,51])
    future = prepared(identities=[cold['frame_identities'][0],copy.deepcopy(cold['frame_identities'][0])],
                      ids=[2,258880,258880,42,258880,258880,51],blocks=[[1,3],[4,6]])
    future['frame_identities'][1][4] = 1
    changed = prepared()
    changed['frame_identities'][0][8] = 'd'*64
    rows = {name:(body(),value) for name,value in zip(tool.VARIANTS,(cold,grow,future,changed))}
    assert tool.scenario_relations(rows)==0
    broken = copy.deepcopy(rows)
    broken['grow'][1]['frame_identities'][0][3] = '0000000000000000'
    with pytest.raises(ValueError):
        tool.scenario_relations(broken)
    ordinal_only = copy.deepcopy(rows)
    ordinal_only['changed'][1]['frame_identities'] = copy.deepcopy(cold['frame_identities'])
    ordinal_only['changed'][1]['frame_identities'][0][4] = 1
    with pytest.raises(ValueError):
        tool.scenario_relations(ordinal_only)


def test_frozen_reader_refuses_hash_extent_and_file_type(tmp_path):
    p = tmp_path/'fixture.json'
    data = b'{"public":true}'
    p.write_bytes(data)
    ref = {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
    assert tool.ref_json(ref)=={'public':True}
    wrong = {**ref,'sha256':'0'*64}
    with pytest.raises(ValueError):
        tool.ref_bytes(wrong)
    link = tmp_path/'alias'
    link.symlink_to(p)
    with pytest.raises(OSError):
        tool.ref_bytes({**ref,'path':str(link)})


def test_historical_wire_parsed_and_exact_body_oracles(tmp_path):
    request = {'method':'POST','url':'http://127.0.0.1:8889/v1/responses','body':body(True)}
    values = {'request':request,'exchange':{'status':200,'complete':True},'response':response(),'result':response()}
    refs = {}
    for key,value in values.items():
        raw = tool.semantic_bytes(value)
        p = tmp_path/key
        p.write_bytes(raw)
        refs[key] = {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    loaded,route,gold = tool.historical_case('image-responses',refs)
    assert loaded==body(True) and route=='/v1/responses' and gold==response()
    values['result']['usage']['output_tokens'] = 3
    raw = tool.semantic_bytes(values['result'])
    (tmp_path/'result').write_bytes(raw)
    refs['result'] = {'path':str(tmp_path/'result'),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError):
        tool.historical_case('image-responses',refs)


def test_source_contract_covers_both_encoder_sites_and_request_no_warm(tmp_path):
    root = Path(__file__).resolve().parents[1]/'src/tensorfold'
    paths = {'engine':root/'families/gemma4/cuda/engine.py','vision':root/'families/gemma4/cuda/vision.py',
             'server':root/'cuda/server.py'}
    refs = {k:{'path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for k,p in paths.items()}
    proof = tool.source_contract(refs)
    assert proof['encoder_callsites']==['encode_suffix','warm']
    assert proof['work_label']=='SOURCE_DERIVED_NOT_DIRECT_COUNT'
    changed = tmp_path/'server.py'
    changed.write_text('def request(app):\n    app.warm()\n')
    raw = changed.read_bytes()
    refs['server'] = {'path':str(changed),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
    with pytest.raises(ValueError):
        tool.source_contract(refs)


def test_publication_requires_actual_legal_complete_cut_and_zero_pending():
    class Client:
        def tokens(self,*args):
            return [2,258880,258880,42,10]
    value = prepared()
    result = {'message':{'role':'assistant','content':'red','reasoning_content':'','tool_calls':[]},
              'tensorfold':{'cache_publication':{'complete':True,'status':'published','matched_tokens':4,
                                                'requested_tokens':5,'materialized_tokens':5}}}
    proof = tool.publication_work(Client(),'case',body(),result,value,False,'candidate')
    assert proof['pending_frame_ordinals']==() and proof['direct_counter'] is False
    result['tensorfold']['cache_publication']['matched_tokens'] = 2
    with pytest.raises(ValueError):
        tool.publication_work(Client(),'case',body(),result,value,False,'candidate')
    result['tensorfold']['cache_publication']['matched_tokens'] = 1
    with pytest.raises(ValueError):
        tool.publication_work(Client(),'case',body(),result,value,False,'candidate')


def test_actual_request_pair_uses_public_tokenizer_and_both_publication_cuts():
    calls = []
    request = body()
    witness = prepared(request)
    def reply(cached):
        return {'id':'chat_id','object':'chat.completion','created':1,'model':'gemma4-26b-a4b',
                'choices':[{'index':0,'message':{'role':'assistant','content':'red'},'finish_reason':'stop'}],
                'usage':{'prompt_tokens':4,'completion_tokens':1,'total_tokens':5,
                         'prompt_tokens_details':{'cached_tokens':cached}},
                'tensorfold':{'token_ids':[10],'rounds':0,'drafted':0,'accepted':0,
                    'cache_publication':{'complete':True,'status':'published','matched_tokens':4,
                                         'requested_tokens':5,'materialized_tokens':5}}}
    class Client:
        def stats(self,name):
            return {'load':{'requests_in_flight':0,'requests_running':0,'requests_waiting':0}}
        def tokens(self,name,value):
            calls.append(('tokens',name,value))
            return [2,258880,258880,42,10] if 'closed-history' in name else witness['token_ids']
        def call(self,name,path,value):
            calls.append(('call',name,path,copy.deepcopy(value)))
            return json.dumps(reply(1 if name.endswith('first') else 4)).encode(),'tag',{}
    first,row = tool.request_pair(Client(),'image-cold',request,witness,False,'candidate')
    assert first['tensorfold']['token_ids']==[10] and row['cuts']==[1,4]
    assert [x[0] for x in calls]==['tokens','call','tokens','call','tokens']
    assert calls[1][3]==calls[3][3]==request
    assert row['requests'][0]['generation_pending_frames_SOURCE_DERIVED']==(0,)
    assert row['requests'][1]['generation_pending_frames_SOURCE_DERIVED']==()
    assert all(r['publication']['pending_frame_ordinals']==() for r in row['requests'])


@pytest.mark.parametrize('url',['http://127.0.0.1:8888','https://127.0.0.1:18888',
                               'http://localhost:18888','http://user@127.0.0.1:18888'])
def test_endpoint_refuses_before_inputs_or_http(monkeypatch,url):
    def unexpected(*args,**kwargs):
        raise AssertionError('endpoint refusal performed effects')
    monkeypatch.setattr(tool,'load_spec',unexpected)
    monkeypatch.setattr(tool.shared,'Client',unexpected)
    with pytest.raises(ValueError,match='private declared18888'):
        tool.main(['--inputs','unused','--inputs-sha256','0'*64,'--base-url',url,'--arm','candidate','--out','unused'])


def test_stored_get_exact_native_response():
    reply = {'id':'resp_example','output':[{'content':'literal'}]}
    class Client:
        def json(self,name,path):
            assert path=='/v1/responses/resp_example'
            return reply
    tool.stored_get(Client(),'historical',reply)
    class Changed(Client):
        def json(self,name,path):
            return {**reply,'output':[]}
    with pytest.raises(ValueError,match='stored GET'):
        tool.stored_get(Changed(),'historical',reply)


def test_growth_work_uses_current_frame_coordinates():
    old = {'frame_identities':['same'],'blocks':[[1,3]],'token_ids':[0]*12}
    shifted = {'frame_identities':['same'],'blocks':[[5,7]],'token_ids':[0]*12}
    # Old end <=3 would pass the former check, but frame zero is still pending.
    with pytest.raises(ValueError,match='existing media'):
        tool.growth_work('grow',old,shifted,3)
    tool.growth_work('grow',old,shifted,7)
    future = {'frame_identities':['same','new'],'blocks':[[5,7],[9,11]],'token_ids':[0]*12}
    with pytest.raises(ValueError,match='existing media'):
        tool.growth_work('future',old,future,3)
    tool.growth_work('future',old,future,7)
    tool.growth_work('future',old,future,9)
    with pytest.raises(ValueError,match='future media'):
        tool.growth_work('future',old,future,11)


@pytest.mark.parametrize('responses',[False,True])
def test_variant_cannot_lower_original_output_work(responses):
    request = body(responses)
    key = 'max_output_tokens' if responses else 'max_tokens'
    maximum = request[key]
    tool.fixed_body(request,responses=responses,expected_maximum=maximum)
    request[key] = maximum-1
    with pytest.raises(ValueError,match='historical output work'):
        tool.fixed_body(request,responses=responses,expected_maximum=maximum)


@pytest.mark.parametrize('identifier',[None,7,True,[]])
def test_response_message_id_must_be_native_string(identifier):
    value = response()
    value['output'][0]['id'] = identifier
    with pytest.raises(ValueError,match='typed message ID'):
        tool.projection(value,True)


@pytest.mark.parametrize('token',[True,1.0,-1,262144,'10'])
@pytest.mark.parametrize('responses',[False,True])
def test_accepted_ids_preserve_exact_integer_native_domain(token,responses):
    value = response()
    if not responses:
        value['usage']['completion_tokens'] = value['usage'].pop('output_tokens')
    value['tensorfold']['token_ids'][0] = token
    with pytest.raises(ValueError,match='typed accepted token'):
        tool.accepted_ids(value,responses)


@pytest.mark.parametrize('responses',[False,True])
def test_actual_output_cannot_exceed_request_maximum(responses):
    value = response()
    if not responses:
        value['usage']['completion_tokens'] = value['usage'].pop('output_tokens')
        value['usage']['prompt_tokens'] = value['usage'].pop('input_tokens')
        value['choices'] = [{'index':0,'message':{'role':'assistant','content':'red'},'finish_reason':'stop'}]
    tool.accepted_ids(value,responses,2)
    with pytest.raises(ValueError,match='exceeded declared request work'):
        tool.accepted_ids(value,responses,1)


def test_original_chat_completion_oracle_still_owns_typed_finish():
    value = {'usage':{'prompt_tokens':4,'completion_tokens':1,'total_tokens':5},
             'choices':[{'index':0,'message':{'role':'assistant','content':'red'},'finish_reason':'stop'}],
             'tensorfold':{'token_ids':[10]}}
    tool.accepted_ids(value,False,128)
    value['choices'][0]['finish_reason'] = 'invented'
    with pytest.raises(ValueError,match='finish reason'):
        tool.accepted_ids(value,False,128)


def test_actual_image_null_and_video_binary64_pts_are_lossless():
    image = prepared()
    assert tool.metadata(image,body())['frame_identities'][0][3] is None
    image['frame_identities'][0][3] = ''
    with pytest.raises(ValueError,match='image PTS'):
        tool.metadata(image,body())
    video = prepared()
    video['frame_identities'][0][1] = 'video'
    video['frame_identities'][0][3] = '3ff0000000000000'
    video['media_token_ids'][1] = video['media_token_ids'][0]
    # Same placeholder run, with the actual video token selected explicitly.
    video['media_token_ids'][0] = 258881
    assert tool.metadata(video,body())['frame_identities'][0][3]=='3ff0000000000000'
    for invalid in (None,'','00','nothex0000000000'):
        video['frame_identities'][0][3] = invalid
        with pytest.raises(ValueError):
            tool.metadata(video,body())


@pytest.mark.parametrize('historical',[False,True])
def test_added_duration_telemetry_does_not_change_semantic_equality(historical):
    gold = response()
    actual = copy.deepcopy(gold)
    actual['tensorfold'].update(frontend_settlement_s=2.493845500983298,
                               request_wall_s=18.65398689196445)
    raw_actual = copy.deepcopy(actual)
    tool.compare(gold,actual,True,historical=historical)
    assert actual==raw_actual  # Raw timing evidence is never stripped/mutated.
    assert actual['tensorfold']['frontend_settlement_s']==2.493845500983298
    assert actual['tensorfold']['request_wall_s']==18.65398689196445
    for kind in ('content','ids','noncached-usage'):
        changed = copy.deepcopy(actual)
        if kind=='content':
            changed['output'][0]['content'][0]['text'] = 'different semantic result'
        elif kind=='ids':
            changed['tensorfold']['token_ids'][0] += 1
        else:
            changed['usage']['input_tokens'] += 1
        with pytest.raises(ValueError):
            tool.compare(gold,changed,True,historical=historical)
