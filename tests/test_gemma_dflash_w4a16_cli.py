"""Maintained private W4A16 CLI/server boundaries, no model or CUDA work."""
import json
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli, families
from tensorfold.families import gemma4
from tests.cli_server_fixture import cuda_cli_app, cuda_cli_engine, install_cli_server


def raw(method='compressed-tensors', **changes):
    quant = ({'quant_method':'compressed-tensors','format':'nvfp4-pack-quantized'}
             if method=='compressed-tensors' else {'quant_method':'modelopt','quant_algo':'NVFP4'})
    return {'model_type':'gemma4','text_config':{'enable_moe_block':True},'quantization_config':quant,**changes}


def family():return NS(title=gemma4.TITLE,model_type='gemma4',package=gemma4)


def test_parser_default_explicit_activation_and_help(capsys):
    parser=cli.build_parser()
    assert parser.parse_args(['serve','owner/model']).dflash_head_w4a16 is False
    assert parser.parse_args(['serve','owner/model','--dflash-head-w4a16']).dflash_head_w4a16 is True
    with pytest.raises(SystemExit) as stop:parser.parse_args(['serve','--help'])
    assert stop.value.code==0
    text=capsys.readouterr().out
    assert '--dflash-head-w4a16' in text and 'proposal head' in text and 'default off' in text


@pytest.mark.parametrize('backend,capable',[('mlx',True),('cuda',False)])
def test_unsupported_family_backend_rejected_before_metadata(monkeypatch,backend,capable):
    args=cli.build_parser().parse_args(['serve','owner/model','--dflash-head-w4a16'])
    found=NS(title='other',model_type='other',package=NS(CUDA_DFLASH_HEAD_W4A16=capable))
    monkeypatch.setattr(families,'read_config',lambda *_:pytest.fail('unsupported option read metadata'))
    with pytest.raises(ValueError,match='CT NVFP4 or ModelOpt MoE on CUDA'):cli._check_serve_options(args,found,backend)


@pytest.mark.parametrize('method',['compressed-tensors','modelopt'])
@pytest.mark.parametrize('model_type',['gemma4','gemma4_text'])
def test_native_CT_and_ModelOpt_metadata_admitted_before_model(tmp_path,method,model_type):
    (tmp_path/'config.json').write_text(json.dumps(raw(method,model_type=model_type)))
    args=cli.build_parser().parse_args(['serve',str(tmp_path),'--dflash-head-w4a16','--drafter-bits','8'])
    cli._check_serve_options(args,family(),'cuda',tmp_path)


@pytest.mark.parametrize('metadata',[
    raw(text_config={'enable_moe_block':False,'hidden_size':1536,'num_hidden_layers':35}),
    raw(text_config={'enable_moe_block':1}),raw(model_type='other'),
    raw(quantization_config={'quant_method':'compressed-tensors','format':'int-quantized'}),
    raw(quantization_config={'bits':4,'group_size':64}),
    raw('modelopt',text_config={'enable_moe_block':False}),
])
def test_dense_affine_and_unsupported_metadata_cannot_reach_engine(monkeypatch,tmp_path,metadata):
    monkeypatch.setattr(families,'read_config',lambda *_:metadata)
    args=cli.build_parser().parse_args(['serve',str(tmp_path),'--dflash-head-w4a16','--drafter-bits','8'])
    with pytest.raises(ValueError):cli._check_serve_options(args,family(),'cuda',tmp_path)
    with pytest.raises(ValueError):gemma4.cuda_engine(tmp_path,drafter='owned/draft',drafter_bits=8,dflash_head_w4a16=True)


@pytest.mark.parametrize('value',[None,1,'yes'])
def test_nonboolean_is_rejected_before_metadata(monkeypatch,tmp_path,value):
    monkeypatch.setattr(families,'read_config',lambda *_:pytest.fail('invalid bool read metadata'))
    with pytest.raises(ValueError,match='must be a bool'):gemma4.cuda_engine(tmp_path,dflash_head_w4a16=value)
    args=cli.build_parser().parse_args(['serve',str(tmp_path)]);args.dflash_head_w4a16=value
    with pytest.raises(ValueError,match='must be a boolean'):cli._check_serve_options(args,family(),'cuda',tmp_path)


@pytest.mark.parametrize('method',['compressed-tensors','modelopt'])
@pytest.mark.parametrize('selected',[None,False,True])
def test_actual_family_forwards_only_true_preserving_serving_arguments(monkeypatch,tmp_path,method,selected):
    from tensorfold.cuda import precision
    from tensorfold.families.gemma4.cuda import engine
    monkeypatch.setattr(families,'read_config',lambda *_:raw(method))
    monkeypatch.setattr(precision,'mode',lambda:precision.CHECKPOINT)
    seen=[];monkeypatch.setattr(engine,'Gemma4Engine',lambda *args,**kwargs:seen.append(kwargs) or 'loaded')
    options={} if selected is None else {'dflash_head_w4a16':selected}
    assert gemma4.cuda_engine(tmp_path,drafter='owned/draft',drafter_bits=8,draft_rows=5,context=131072,
        context_explicit=True,vision=True,chat_template_file='owned-google.jinja',**options)=='loaded'
    assert ('dflash_head_w4a16' in seen[0]) is (selected is True)
    assert seen[0].get('dflash_head_w4a16',False) is (selected is True)
    assert seen[0]['depth']==4 and seen[0]['drafter_bits']==8 and seen[0]['context']==131072
    assert seen[0]['vision'] is True and seen[0]['chat_template_file']=='owned-google.jinja'
    for name in ('global_cache','global_trig','dflash_graph','dflash_head_native_a8','delta4_reuse'):
        assert name not in seen[0]


@pytest.mark.parametrize('selected',[False,True])
def test_native_server_app_forwards_only_explicit_true(monkeypatch,tmp_path,request,selected):
    import tensorfold.cuda.server as server
    from tensorfold.cuda import prompt_precision
    seen=[]
    def engine(*args,**kwargs):seen.append(kwargs);return cuda_cli_engine(max_len=8192,w=NS(fast_prefill=False))
    found=NS(title='fixture',model_type='fixture',package=NS(cuda_engine=engine,CUDA_DFLASH_HEAD_W4A16=True,CUDA_DRAFTER_BITS=True))
    monkeypatch.setattr(server,'App',lambda *args,**kwargs:cuda_cli_app(effective_context_window=8192))
    install_cli_server(monkeypatch,request,server)
    monkeypatch.setattr(cli,'_drafter',lambda *args,**kwargs:'owned/draft')
    args=cli.build_parser().parse_args(['serve',str(tmp_path),'--backend','cuda','--drafter-bits','8']
        +(['--dflash-head-w4a16'] if selected else []))
    try:assert cli._serve_cuda(args,found,tmp_path,8192)==0
    finally:prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)
    assert len(seen)==1 and seen[0]['drafter_bits']==8 and seen[0]['drafter']=='owned/draft'
    assert ('dflash_head_w4a16' in seen[0]) is selected
    assert seen[0].get('dflash_head_w4a16',False) is selected


@pytest.mark.parametrize('extra',[[],['--drafter-bits','16'],['--drafter-bits','8','--no-drafts'],
    ['--drafter-bits','8','--draft-method','none']])
def test_disabled_or_wrong_precision_rejected_before_metadata(monkeypatch,extra):
    args=cli.build_parser().parse_args(['serve','owner/model','--dflash-head-w4a16',*extra])
    monkeypatch.setattr(families,'read_config',lambda *_:pytest.fail('unsupported drafting read metadata'))
    expected='--draft-method.*selected proposal method' if '--draft-method' in extra else 'enabled DFlash.*drafter-bits 8'
    with pytest.raises(ValueError,match=expected):cli._check_serve_options(args,family(),'cuda')


@pytest.mark.parametrize('kwargs',[{}, {'drafter':'owned/draft','drafter_bits':16},
    {'drafter':'owned/draft','drafter_bits':8,'no_drafts':True}])
def test_direct_family_rejects_no_companion_or_wrong_precision(monkeypatch,tmp_path,kwargs):
    monkeypatch.setattr(families,'read_config',lambda *_:pytest.fail('invalid companion read metadata'))
    with pytest.raises(ValueError,match='enabled DFlash'):gemma4.cuda_engine(tmp_path,dflash_head_w4a16=True,**kwargs)
