"""Fixed Gemma image/video replay and shared-prefix media qualification.

Import is inert. Lifecycle/source/image authority belongs to the sole executor.
Encoder work is SOURCE-DERIVED from bound installed code and actual cuts, never
reported as a directly observed encoder counter. No runtime instrumentation.
"""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path
import urllib.parse

if __package__:
    from . import gemma_shared_prefix_qualify as shared
else:
    import gemma_shared_prefix_qualify as shared

CASES = ('image-chat','image-responses','video-chat','video-responses')
VARIANTS = ('cold','grow','future','changed')
CAP = 6 << 20
HISTORICAL_HELPER = 'd68f002b9d4019411bfd6d83b764ae4291b80e53b2789ab434d77acf46718134'
CLIENT_SHA = '3c92fba4cf2ed06e666f5bea7f5281e30446b99b441e5655b4b3ccdec263ac02'
AST_AUTHORITY = {
    'engine': {'_prefill':'e1a86637fef44460dee2c1951949ec5b325de62c72038fcc62575ac1eafaba42',
               '_publish_canonical_locked':'c1579a3b2ff63f4928fb73a4a99b2825d7f2f1239618e6782754b0a6a85da117',
               'generate':'6299ca3b7a590a8e9a268c95b1e39712347d9d2f2d7d139a5b35be4d8064d52b'},
    'vision': {'warm':'d4fb09a8e7c003140da5df14f9fd9c54689f6a93ba0fffe675fd4e9f7f953499',
               'encode':'fc07781afe741a7439ab6393373d4abdb151097e9e526028e079d49a61a00f27',
               'encode_suffix':'b720bb3e657ac4e4b34e749229ccbed38b82311b81a4552d66d4637b5904eedc'}}
DYNAMIC = frozenset(('request_id','prefill_s','decode_s','queue_s','prepare_s','ttft_s','decode_tps','frontend_settlement_s','request_wall_s','cache_publication','cached'))
HISTORICAL_PROFILE_COUNTERS = frozenset(('stats_version','rounds','drafts','drafted','accepted','min_rows'))


def ref_bytes(ref):
    shared.require(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'closed immutable reference required')
    shared.require(type(ref['bytes']) is int and 0<ref['bytes']<=CAP,'bounded reference extent required')
    raw = shared.read_bound(ref['path'],ref['bytes'])
    shared.require(len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'],'frozen reference changed')
    return raw


def ref_json(ref):
    return json.loads(ref_bytes(ref))


def semantic_bytes(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def source_contract(refs):
    """Exact supported source DAG, including BOTH native encoder call sites."""
    shared.require(set(refs)=={'engine','vision','server'},'installed Engine/Vision/server source references required')
    shared.require(sys.version_info[:2]==(3,12),"source AST contract uses admitted CPython3.12 format")
    facts = {}
    for key,classname in (('engine','Gemma4Engine'),('vision','GemmaCudaVision')):
        tree = ast.parse(ref_bytes(refs[key]))
        cls = next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==classname)
        methods = {n.name:n for n in cls.body if isinstance(n,ast.FunctionDef)}
        for name,expected in AST_AUTHORITY[key].items():
            actual = hashlib.sha256(ast.dump(methods[name],include_attributes=False).encode()).hexdigest()
            shared.require(actual==expected,'unsupported installed media source DAG: '+key+'.'+name)
        if key=='vision':
            sites = sorted(name for name,node in methods.items() for call in ast.walk(node)
                           if isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute)
                           and isinstance(call.func.value,ast.Name) and call.func.value.id=='self'
                           and call.func.attr=='_features')
            shared.require(sites==['encode_suffix','warm'],'uncovered native encoder callsite')
            facts['encoder_callsites'] = sites
    # The second _features site is preparation, not a hidden request encoder.
    # Bind the actual normal request owner too; no warm/direct tower call may
    # appear there or in this exact supported Engine.
    for key in ('engine','server'):
        tree = ast.parse(ref_bytes(refs[key]))
        direct = [n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
                  and n.func.attr in ('warm','_features')]
        shared.require(not direct,'uncovered preparation/encoder call from request owner')
    facts.update(method_AST=AST_AUTHORITY,work_label='SOURCE_DERIVED_NOT_DIRECT_COUNT',
                 warm_scope='warm is preparation-only; executor must bind startup completion before requests',
                 request_scope='fixed no-thinking/no-tools nonstream calls; exact Engine generation/publication DAG')
    return facts


def metadata(value, body):
    """Serialized fields of real current PreparedGemmaVision, not invented counts.

    Producer: GemmaVisionProcessor.prepare under its ordinary funded metadata
    owner, then PreparedGemmaVision.token_ids/blocks/frame_identities and
    processor_binding. Image identity[3] stays None; video PTS bytes become hex.
    """
    shared.require(type(value) is dict and set(value)=={'schema','body_sha256','token_ids','blocks','frame_identities','media_token_ids','processor_binding'},'current prepared-media witness required')
    shared.require(value['schema']=='tensorfold.prepared-gemma-media-witness-v1'
                   and value['body_sha256']==hashlib.sha256(semantic_bytes(body)).hexdigest(),'prepared witness belongs to another exact body')
    ids,blocks,identities = value['token_ids'],value['blocks'],value['frame_identities']
    shared.require(type(ids) is list and 0<len(ids)<262144 and all(type(i) is int and 0<=i<262144 for i in ids),'bounded actual prepared token IDs required')
    shared.require(type(blocks) is list and 0<len(blocks)<=len(ids) and len(blocks)==len(identities),'bounded prepared frame/block bijection required')
    previous = 0
    for block,identity in zip(blocks,identities):
        shared.require(type(block) is list and len(block)==2 and all(type(i) is int for i in block)
                       and previous<=block[0]<block[1]<=len(ids),'invalid atomic visual range')
        shared.require(type(identity) is list and len(identity)==10 and identity[0]=='gemma-prepared-frame-v1'
                       and identity[1] in ('image','video') and identity[5]==block[1]-block[0]
                       and identity[9]==value['processor_binding'],'real pixel/PTS/grid/processor identity required')
        if identity[1]=='image':
            shared.require(identity[3] is None,'actual image PTS identity must remain null')
        else:
            shared.require(type(identity[3]) is str and len(identity[3])==16,
                           'actual video binary64 PTS requires eight-byte hex')
            shared.require(len(bytes.fromhex(identity[3]))==8,'actual video PTS byte extent differs')
        previous = block[1]
    shared.require(type(value['media_token_ids']) is list and len(value['media_token_ids'])==2,'actual image/video token identities required')
    from tensorfold.families.gemma4.cuda.vision import media_blocks
    observed = media_blocks(ids,*value['media_token_ids'],[(v[1],v[5]) for v in identities])
    shared.require([list(row) for row in observed]==blocks,'actual media token runs/processor counts differ')
    return value


def legal_cut(cut, prepared):
    shared.require(type(cut) is int and 0<=cut<=len(prepared['token_ids']),'actual cache cut outside prepared context')
    shared.require(not any(a<cut<b for a,b in prepared['blocks']),'cache cut split a bidirectional visual block')
    return tuple(i for i,(a,_b) in enumerate(prepared['blocks']) if a>=cut)


def accepted_ids(reply,responses,maximum=None):
    ids = reply.get('tensorfold',{}).get('token_ids')
    usage = reply.get('usage')
    shared.require(type(usage) is dict and type(ids) is list
                   and len(ids)==usage.get('output_tokens' if responses else 'completion_tokens')
                   and all(type(token) is int and 0<=token<262144 for token in ids),
                   'complete typed accepted token IDs/usage required')
    shared.require(maximum is None or len(ids)<=maximum,'accepted output exceeded declared request work')
    if not responses and maximum is not None:
        # Reuse the original independent Chat oracle, including usage totals,
        # single assistant choice, finish reason and text-only message types.
        if __package__:
            from .v100_rotorquant_live import completion
        else:
            from v100_rotorquant_live import completion
        completion(reply,maximum)
    return ids


def projection(reply,responses,historical=False):
    """Complete semantic object; only declared operational fields differ."""
    accepted_ids(reply,responses)
    value = copy.deepcopy(reply)
    if responses:
        shared.require(value.get('object')=='response' and value.get('status') in ('completed','incomplete')
                       and value.get('error') is None and type(value.get('output')) is list,'valid native Responses object required')
        for key in ('id','created_at','completed_at'):
            value.pop(key,None)
        for item in value['output']:
            shared.require(type(item) is dict and item.get('type')=='message' and item.get('role')=='assistant'
                           and type(item.get('id')) is str,'text media response with typed message ID required')
            item.pop('id',None)
    else:
        # Same complete choice/message/finish oracle as the original reader;
        # IDs and every non-dynamic field remain compared, not text-only gold.
        for key in ('id','created'):
            value.pop(key,None)
    usage = value['usage']
    for key in ('prompt_tokens_details','input_tokens_details'):
        if key in usage:
            usage[key].pop('cached_tokens',None)
    runtime = value.get('tensorfold',{})
    for key in DYNAMIC | (HISTORICAL_PROFILE_COUNTERS if historical else frozenset()):
        runtime.pop(key,None)
    return value


def compare(first, second, responses, *, historical=False):
    shared.require(projection(first,responses,historical)==projection(second,responses,historical),
                   'media accepted IDs/content/finish/usage/semantic object differs')


def historical_case(name, refs):
    shared.require(set(refs)=={'request','exchange','response','result'},'exact four historical files required')
    request,exchange = ref_json(refs['request']),ref_json(refs['exchange'])
    raw = ref_bytes(refs['response'])
    reply = json.loads(raw)
    shared.require(reply==ref_json(refs['result']),'historical wire and parsed result disagree')
    body = request['body']
    route = '/v1/responses' if name.endswith('responses') else '/v1/chat/completions'
    shared.require(request.get('method')=='POST' and urllib.parse.urlsplit(request['url']).path==route
                   and exchange.get('status')==200 and exchange.get('complete') is True,'completed exact historical media exchange required')
    fixed_body(body,responses=name.endswith('responses'))
    return body,route,reply


def fixed_body(body, *, responses, expected_maximum=None):
    shared.require(type(body) is dict and body.get('model')=='gemma4-26b-a4b' and body.get('temperature')==0
                   and body.get('draft') is True and body.get('return_token_ids') is True
                   and not body.get('stream') and body.get('chat_template_kwargs')=={'enable_thinking':False}
                   and not body.get('tools') and not body.get('instructions') and not body.get('previous_response_id'),
                   'fixed deterministic current media request region required')
    shared.require(type(body.get('input' if responses else 'messages')) is list,'explicit full media history required')
    if responses:
        shared.require(body.get('store') is True,'stored native Responses contract required')
    maximum = body['max_output_tokens' if responses else 'max_tokens']
    shared.require(type(maximum) is int and 0<maximum<=128,'bounded original output work required')
    shared.require(expected_maximum is None or maximum==expected_maximum,'variant changed historical output work')


def chat_body(body,responses,output=()):
    if responses:
        from tensorfold.server.responses_translate import messages
        history = messages(body['input']+list(output),allow_video=True)
    else:
        history = body['messages']+list(output)
    return {'model':body['model'],'messages':history,'chat_template_kwargs':body['chat_template_kwargs'],
            'add_generation_prompt':not bool(output)}


def settled(client,name):
    result = client.stats(name)
    load = result['load']
    shared.require(all(load.get(key)==0 for key in ('requests_in_flight','requests_running','requests_waiting')),
                   'exclusive media operation did not retire to idle')
    return result


def prepared_on_service(client,name,body,prepared,responses):
    ids = client.tokens(name,chat_body(body,responses))
    shared.require(ids==prepared['token_ids'],'served tokenizer/processor differs from bound prepared witness')


def publication_work(client,name,body,result,prepared,responses,arm):
    publication = result['tensorfold'].get('cache_publication')
    if arm!='candidate':
        return {'status':'BASELINE_NEW_PUBLICATION_NOT_REQUIRED'}
    shared.require(type(publication) is dict and publication.get('complete') is True
                   and publication.get('status')=='published','candidate canonical media publication incomplete')
    if responses:
        history = result['response']['output']
    else:
        history = [shared.assistant(result)]
    canonical = client.tokens(name+'-closed-history',chat_body(body,responses,history))
    from tensorfold.families.gemma4.cuda.vision import media_blocks
    blocks = media_blocks(canonical,*prepared['media_token_ids'],[(v[1],v[5]) for v in prepared['frame_identities']])
    bound = {**prepared,'token_ids':canonical,'blocks':[list(v) for v in blocks]}
    cut = publication.get('matched_tokens')
    pending = legal_cut(cut,bound)
    shared.require(publication.get('requested_tokens')==len(canonical)
                   and publication.get('materialized_tokens')==len(canonical),'publication/tokenizer frontier differs')
    shared.require(not pending,'canonical publication repeated visual encoder work')
    return {'status':'SOURCE_DERIVED','actual_cut':cut,'atomic_blocks':bound['blocks'],
            'pending_frame_ordinals':pending,'direct_counter':False}


def stored_get(client,name,reply):
    identifier = reply.get('id')
    shared.require(type(identifier) is str and identifier.startswith('resp_') and len(identifier)<=128,'stored response ID missing')
    shared.require(client.json(name+'-get','/v1/responses/'+urllib.parse.quote(identifier,safe=''))==reply,
                   'stored GET differs from posted response')


def request_pair(client,name,body,prepared,responses,arm):
    route = '/v1/responses' if responses else '/v1/chat/completions'
    prepared_on_service(client,name+'-tokens',body,prepared,responses)
    replies,works,cuts = [],[],[]
    for suffix in ('first','repeat'):
        raw,_,_ = client.call(name+'-'+suffix,route,body)
        reply = json.loads(raw)
        normalized = shared.normalize(raw,False,responses)
        usage = reply['usage']
        prompt = usage['input_tokens' if responses else 'prompt_tokens']
        shared.require(prompt==len(prepared['token_ids']),'actual model/prepared prompt lengths differ')
        cut = usage['input_tokens_details' if responses else 'prompt_tokens_details']['cached_tokens']
        pending = legal_cut(cut,prepared)
        accepted_ids(reply,responses,body['max_output_tokens' if responses else 'max_tokens'])
        works.append({'generation_pending_frames_SOURCE_DERIVED':pending,
                      'publication':publication_work(client,name+'-'+suffix,body,normalized,prepared,responses,arm)})
        replies.append(reply)
        cuts.append(cut)
    compare(replies[0],replies[1],responses)
    if arm=='candidate':
        shared.require(cuts[1]>=max(b for _a,b in prepared['blocks']),'repeated media was not fully cached')
    if responses:
        stored_get(client,name,replies[1])
    settled(client,name+'-settled')
    return replies[0],{'name':name,'cuts':cuts,'requests':works,'accepted_ids_equal':True,
                      'work_label':'SOURCE_DERIVED_NOT_DIRECT_COUNT',
                      'profile_cache_fields_compared_separately':True}


def scenario_relations(rows):
    cold,grow,future,changed = (rows[v][1] for v in VARIANTS)
    shared.require(grow['frame_identities']==cold['frame_identities'],'growth changed existing pixel/PTS/grid identity')
    old = grow['frame_identities']
    shared.require(len(future['frame_identities'])>len(old) and future['frame_identities'][:len(old)]==old,
                   'future media must preserve every prior actual frame identity')
    differences = [i for i,(a,b) in enumerate(zip(changed['frame_identities'],cold['frame_identities']))
                   if any(a[k]!=b[k] for k in (2,3,6,7,8))]
    shared.require(len(changed['frame_identities'])==len(cold['frame_identities']) and differences,
                   'changed media needs actual different pixel/PTS/grid identity')
    # Processor/config remain fixed: the identity change is media, not a new
    # unsupported preprocessing profile.
    shared.require(all(v['processor_binding']==cold['processor_binding'] for v in (grow,future,changed)),
                   'media variant changed processor/config authority')
    return differences[0]


def growth_work(variant,old,current,cut):
    # Frame identity equality does not establish equal token coordinates: a
    # current body may shift the same visual frames later in its token tape.
    count = len(old['frame_identities'])
    pending = legal_cut(cut,current)
    shared.require(count<=len(current['blocks'])
                   and cut>=max(b for _a,b in current['blocks'][:count])
                   and not any(i<count for i in pending),'existing media not reused during growth')
    if variant=='future':
        shared.require(count<len(current['blocks']) and count in pending,
                       'new future media was already cached unexpectedly')


def load_spec(path,digest):
    raw = shared.read_bound(path,CAP)
    shared.require(hashlib.sha256(raw).hexdigest()==digest,'media input manifest changed')
    spec = json.loads(raw)
    shared.require(set(spec)=={'schema','binding','historical','historical_profile','scenarios','sources'} and spec['schema']=='tensorfold.gemma-prefix-media-qualification-v1','closed media manifest required')
    binding = spec['binding']
    shared.require(set(binding)=={'global_cache_mode','context','source_uid','image_id','source_revision','origin'},
                   'closed current source/profile binding required')
    shared.require(type(binding['source_uid']) is str and len(binding['source_uid'])==64
                   and all(c in '0123456789abcdef' for c in binding['source_uid'])
                   and type(binding['image_id']) is str and binding['image_id'].startswith('sha256:')
                   and len(binding['image_id'])==71 and type(binding['source_revision']) is str
                   and 0<len(binding['source_revision'])<=256,'full declared image/source identity required')
    origin = ref_json(binding['origin'])
    shared.require(origin.get('source_uid')==binding['source_uid'] and origin.get('declared_image')==binding['image_id']
                   and origin.get('persistent_cache',{}).get('mode')=='none'
                   and origin['persistent_cache'].get('target_owner_matches') is True
                   and origin['persistent_cache'].get('context')==262144,'current owned NONE source/image profile differs')
    shared.require(binding['global_cache_mode']=='none' and binding['context']==262144,'current NONE/full-native profile required')
    shared.require(set(spec['historical'])==set(CASES) and set(spec['scenarios'])==set(CASES),'image/video Chat+Responses cases required')
    historical = spec['historical_profile']
    shared.require(set(historical)=={'source_uid','image_id','global_cache_mode','context','status','helper_sha256'},
                   'complete historical profile provenance required')
    old_status = ref_json(historical['status'])
    shared.require(historical['helper_sha256']==HISTORICAL_HELPER
                   and historical['global_cache_mode']=='kv' and historical['context']==131072
                   and old_status.get('source_uid')==historical['source_uid']
                   and old_status.get('declared_image')==historical['image_id']
                   and old_status.get('persistent_cache',{}).get('mode')=='kv'
                   and old_status['persistent_cache'].get('target_owner_matches') is True,
                   'historical oracle profile is unbound; never assume current equivalence')
    prepared = {}
    for name in CASES:
        original,_,_ = historical_case(name,spec['historical'][name])
        maximum = original['max_output_tokens' if name.endswith('responses') else 'max_tokens']
        shared.require(set(spec['scenarios'][name])==set(VARIANTS),'same/grow/future/changed variants required')
        for variant,row in spec['scenarios'][name].items():
            shared.require(set(row)=={'body','prepared'},'exact body and current prepared witness references required')
            body = ref_json(row['body'])
            fixed_body(body,responses=name.endswith('responses'),expected_maximum=maximum)
            shared.require(body.get('seed')==7,'frozen current seed7 required')
            prepared[name,variant] = body,metadata(ref_json(row['prepared']),body)
    for name in CASES:
        scenario_relations({v:prepared[name,v] for v in VARIANTS})
    return spec,prepared


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',required=True)
    parser.add_argument('--inputs-sha256',required=True)
    parser.add_argument('--base-url',required=True)
    parser.add_argument('--arm',choices=('baseline','candidate'),required=True)
    parser.add_argument('--out',required=True)
    args = parser.parse_args(argv)
    url = urllib.parse.urlsplit(args.base_url)
    shared.require(url.scheme=='http' and url.hostname=='127.0.0.1' and url.port==18888
                   and not url.username and not url.path.strip('/') and not url.query and not url.fragment,'private declared18888 endpoint only')
    helper = Path(inspect.getsourcefile(shared)).read_bytes()
    shared.require(hashlib.sha256(helper).hexdigest()==CLIENT_SHA,'frozen maintained Client/Evidence changed')
    spec,prepared = load_spec(args.inputs,args.inputs_sha256)
    proof = source_contract(spec['sources']) if args.arm=='candidate' else {'status':'baseline_source_not_new_feature_qualified'}
    out = shared.Evidence(args.out)
    client = shared.Client(args.base_url,out,args.arm)
    report = {'schema':'tensorfold.gemma-prefix-media-result-v1','passed':False,'arm':args.arm,'results':[],
              'source_work_contract':proof,'P3_acceptance':False,'encoder_counts_directly_observed':False,
              'external_executor_image_source_binding':spec['binding'],
              'work_assumptions':['startup preparation complete before private endpoint ready',
                  'only fixed no-thinking/no-tools bounded nonstream requests',
                  'prepared witness is from admitted current processor/source and matches actual served IDs'],
              'historical_helper_sha256':HISTORICAL_HELPER,
              'historical_profile_provenance':spec['historical_profile']}
    primary,errors = None,[None,None]
    try:
        before = settled(client,'initial')
        shared.require(before['server']['source_revision']==spec['binding']['source_revision'],'actual server source revision differs')
        for name in CASES:
            sequence = {v:prepared[name,v] for v in VARIANTS}
            changed_at = scenario_relations(sequence)
            body,route,gold = historical_case(name,spec['historical'][name])
            raw,_,_ = client.call(name+'-historical',route,body)
            actual = json.loads(raw)
            accepted_ids(actual,name.endswith('responses'),body['max_output_tokens' if name.endswith('responses') else 'max_tokens'])
            compare(gold,actual,name.endswith('responses'),historical=True)
            if name.endswith('responses'):
                stored_get(client,name+'-historical',actual)
            for variant in VARIANTS:
                body,info = prepared[name,variant]
                first,row = request_pair(client,name+'-'+variant,body,info,name.endswith('responses'),args.arm)
                if variant=='cold':
                    shared.require(row['cuts'][0]<=info['blocks'][0][0],'cold media unexpectedly reused an encoded visual block')
                if args.arm=='candidate' and variant in ('grow','future'):
                    old = sequence['cold' if variant=='grow' else 'grow'][1]
                    growth_work(variant,old,info,row['cuts'][0])
                if variant=='changed':
                    shared.require(row['cuts'][0]<=info['blocks'][changed_at][0],'incompatible changed media reused cached state')
                report['results'].append(row)
        report['final'] = settled(client,'final')
        report['passed'] = True
    except BaseException as error:
        primary = error
        try:
            report['failure'] = {'type':type(error).__name__,'message':str(error)[:1024]}
        except BaseException as diagnostic:
            errors[0] = diagnostic
    try:
        out.save('media-result.json',report)
    except BaseException as publication:
        errors[1] = publication
    shared.finish_errors(primary,errors)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
