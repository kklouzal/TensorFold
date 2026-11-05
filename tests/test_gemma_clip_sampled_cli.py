"""Clipping selection boundaries; no model or CUDA initialization."""
import ast
import inspect
import textwrap
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli, families
from tensorfold.families import gemma4
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tests.test_gemma_dflash_w4a16_cli import raw, family
from tests.cli_server_fixture import cuda_cli_app, cuda_cli_engine, install_cli_server


def test_parser_default_activation_and_help(capsys):
    parser = cli.build_parser()
    assert parser.parse_args(['serve', 'owned/model']).clip_sampled_verification is False
    assert parser.parse_args(['serve', 'owned/model', '--clip-sampled-verification']).clip_sampled_verification is True
    with pytest.raises(SystemExit) as stopped:
        parser.parse_args(['serve', '--help'])
    assert stopped.value.code == 0
    help_text = ' '.join(capsys.readouterr().out.split())
    assert '--clip-sampled-verification' in help_text and 'full proposal' in help_text


@pytest.mark.parametrize('backend,capable', [('mlx', True), ('cuda', False)])
def test_unsupported_rejected_before_config(monkeypatch, backend, capable):
    args = cli.build_parser().parse_args(['serve', 'owned/model', '--clip-sampled-verification'])
    found = NS(title='other', package=NS(CUDA_CLIP_SAMPLED_VERIFICATION=capable))
    monkeypatch.setattr(families, 'read_config', lambda *_: pytest.fail('read unsupported config'))
    with pytest.raises(ValueError, match='requires Gemma 4'):
        cli._check_serve_options(args, found, backend)


@pytest.mark.parametrize('value', [None, 0, 1, 'yes'])
def test_strict_bool_before_config_or_weights(monkeypatch, tmp_path, value):
    monkeypatch.setattr(families, 'read_config', lambda *_: pytest.fail('read invalid config'))
    with pytest.raises(ValueError, match='must be a bool'):
        gemma4.cuda_engine(tmp_path, clip_sampled_verification=value)
    with pytest.raises(ValueError, match='must be a bool'):
        Gemma4Engine(tmp_path, clip_sampled_verification=value)
    args = cli.build_parser().parse_args(['serve', str(tmp_path)])
    args.clip_sampled_verification = value
    with pytest.raises(ValueError, match='must be a boolean'):
        cli._check_serve_options(args, family(), 'cuda')


@pytest.mark.parametrize('kwargs', [{}, {'drafter': 'owned/draft', 'no_drafts': True}])
def test_disabled_dflash_refused_before_config(monkeypatch, tmp_path, kwargs):
    monkeypatch.setattr(families, 'read_config', lambda *_: pytest.fail('read disabled config'))
    with pytest.raises(ValueError, match='enabled DFlash'):
        gemma4.cuda_engine(tmp_path, clip_sampled_verification=True, **kwargs)
    with pytest.raises(ValueError, match='enabled DFlash'):
        Gemma4Engine(tmp_path, clip_sampled_verification=True, **kwargs)


@pytest.mark.parametrize('method', ['compressed-tensors', 'modelopt', 'affine'])
@pytest.mark.parametrize('selected', [None, False, True])
def test_family_only_true_and_current_profile(monkeypatch, tmp_path, method, selected):
    from tensorfold.cuda import precision
    from tensorfold.families.gemma4.cuda import engine
    metadata = raw(method) if method != 'affine' else raw(quantization_config={'bits': 4, 'group_size': 64})
    monkeypatch.setattr(families, 'read_config', lambda *_: metadata)
    monkeypatch.setattr(precision, 'mode', lambda: precision.CHECKPOINT)
    seen = []
    monkeypatch.setattr(engine, 'Gemma4Engine', lambda *a, **kw: seen.append(kw) or 'loaded')
    option = {} if selected is None else {'clip_sampled_verification': selected}
    if method == 'compressed-tensors':
        assert gemma4.cuda_engine(tmp_path, drafter='owned/draft', drafter_bits=8, draft_rows=5,
            context=131072, context_explicit=True, vision=True, prepare_global_prompt=True,
            delta4_head=True, dflash_head_w4a16=True, **option) == 'loaded'
    else:
        assert gemma4.cuda_engine(tmp_path, drafter='owned/draft', drafter_bits=16, **option) == 'loaded'
    assert ('clip_sampled_verification' in seen[0]) is (selected is True)
    if method == 'compressed-tensors':
        assert seen[0]['depth'] == 4 and seen[0]['context'] == 131072
        assert seen[0]['dflash_head_w4a16'] is True and seen[0]['prepare_global_prompt'] is True


@pytest.mark.parametrize('selected', [None, False])
def test_e2b_omission_preserves_existing_path(monkeypatch, tmp_path, selected):
    from tensorfold.families.gemma4.cuda import e2b_engine
    monkeypatch.setattr(families, 'read_config', lambda *_: raw(text_config={'enable_moe_block': False, 'hidden_size': 1536, 'num_hidden_layers': 35}))
    seen = []
    monkeypatch.setattr(e2b_engine, 'E2BEngine', lambda *a, **kw: seen.append(kw) or 'dense')
    option = {} if selected is None else {'clip_sampled_verification': selected}
    assert gemma4.cuda_engine(tmp_path, **option) == 'dense'
    assert 'clip_sampled_verification' not in seen[0]
    with pytest.raises(ValueError, match='MoE'):
        gemma4.cuda_engine(tmp_path, drafter='owned/draft', clip_sampled_verification=True)


@pytest.mark.parametrize('selected', [False, True])
def test_real_server_forwarding(monkeypatch, tmp_path, request, selected):
    import tensorfold.cuda.server as server
    from tensorfold.cuda import prompt_precision
    seen = []
    def engine(*a, **kw):
        seen.append(kw)
        return cuda_cli_engine(max_len=8192, w=NS(fast_prefill=False))
    found = NS(title='fixture', model_type='fixture', package=NS(cuda_engine=engine, CUDA_CLIP_SAMPLED_VERIFICATION=True))
    monkeypatch.setattr(server, 'App', lambda *a, **kw: cuda_cli_app(effective_context_window=8192))
    install_cli_server(monkeypatch, request, server)
    monkeypatch.setattr(cli, '_drafter', lambda *a, **kw: 'owned/draft')
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--backend', 'cuda'] +
        (['--clip-sampled-verification'] if selected else []))
    try:
        assert cli._serve_cuda(args, found, tmp_path, 8192) == 0
    finally:
        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)
    assert ('clip_sampled_verification' in seen[0]) is selected


def test_actual_preparation_assignment_is_owned_and_dflash_only():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Gemma4Engine.__init__)))
    branches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
        and any(isinstance(item, ast.Attribute) and item.attr == '_clip_sampled_verification_to_room'
                for item in ast.walk(node))]
    assert len(branches) == 1
    code = compile(ast.Module(body=branches, type_ignores=[]), '<actual-clipping-preparation>', 'exec')
    class Existing:
        _clip_sampled_verification_to_room = True
    omitted = Existing()
    exec(code, {'clip_sampled_verification': False, 'kind': 'other', 'self': omitted})
    assert '_clip_sampled_verification_to_room' not in vars(omitted)
    owned = Existing()
    exec(code, {'clip_sampled_verification': True, 'kind': 'dflash', 'self': owned})
    assert vars(owned)['_clip_sampled_verification_to_room'] is True
    with pytest.raises(ValueError, match='DFlash companion'):
        exec(code, {'clip_sampled_verification': True, 'kind': 'sparkulator', 'self': Existing()})


@pytest.mark.parametrize('method', ['compressed-tensors', 'modelopt', 'affine'])
def test_serving_header_admission_and_disabled_drafting(monkeypatch, tmp_path, method):
    metadata = raw(method) if method != 'affine' else raw(quantization_config={'bits': 4, 'group_size': 64})
    monkeypatch.setattr(families, 'read_config', lambda *_: metadata)
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--clip-sampled-verification'])
    cli._check_serve_options(args, family(), 'cuda', tmp_path)
    args.no_drafts = True
    with pytest.raises(ValueError, match='enabled DFlash'):
        cli._check_serve_options(args, family(), 'cuda', tmp_path)
    args.no_drafts = False
    monkeypatch.setattr(families, 'read_config', lambda *_: raw(text_config={'enable_moe_block': False, 'hidden_size': 1536, 'num_hidden_layers': 35}))
    with pytest.raises(ValueError, match='MoE'):
        cli._check_serve_options(args, family(), 'cuda', tmp_path)
