"""Native Delta4 activation over existing CT backend math; CPU CLI boundaries."""

import json
from types import SimpleNamespace as NS

import pytest

from tensorfold import cli, families
from tensorfold.families import gemma4
from tests.cli_server_fixture import cuda_cli_app, cuda_cli_engine, install_cli_server


def raw(**changes):
    return {'model_type': 'gemma4', 'text_config': {'enable_moe_block': True},
            'quantization_config': {'quant_method': 'compressed-tensors', 'format': 'nvfp4-pack-quantized'},
            **changes}


def family():
    return NS(title=gemma4.TITLE, model_type='gemma4', package=gemma4)


def test_parser_default_explicit_activation_and_help(capsys):
    parser = cli.build_parser()
    assert parser.parse_args(['serve', 'owner/model']).delta4_head is False
    assert parser.parse_args(['serve', 'owner/model', '--delta4-head']).delta4_head is True
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(['serve', '--help'])
    assert stop.value.code == 0
    text = capsys.readouterr().out
    assert '--delta4-head' in text and 'CT NVFP4 MoE' in text and 'reuse16' in text
    assert '--delta4-reuse' not in text


@pytest.mark.parametrize('backend,capable', [('mlx', True), ('cuda', False)])
def test_backend_or_family_rejected_before_metadata(monkeypatch, backend, capable):
    args = cli.build_parser().parse_args(['serve', 'owner/model', '--delta4-head'])
    found = NS(title='other', model_type='other', package=NS(CUDA_DELTA4_HEAD=capable))
    monkeypatch.setattr(families, 'read_config', lambda *_: pytest.fail('unsupported metadata read'))
    with pytest.raises(ValueError, match='requires Gemma 4 CT NVFP4 MoE on CUDA'):
        cli._check_serve_options(args, found, backend)


@pytest.mark.parametrize('model_type', ['gemma4', 'gemma4_text'])
def test_actual_ct_metadata_admitted_by_native_serve_preflight(tmp_path, model_type):
    (tmp_path / 'config.json').write_text(json.dumps(raw(model_type=model_type)))
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--delta4-head', '--no-drafts'])
    cli._check_serve_options(args, family(), 'cuda', tmp_path)


@pytest.mark.parametrize('metadata', [
    raw(text_config={'enable_moe_block': False, 'hidden_size': 1536, 'num_hidden_layers': 35}),
    raw(text_config={'enable_moe_block': 1}),
    raw(quantization_config={'quant_method': 'modelopt', 'quant_algo': 'NVFP4'}),
    raw(quantization_config={'quant_method': 'compressed-tensors', 'format': 'int-quantized'}),
    raw(quantization_config={'bits': 4, 'group_size': 64}),
    raw(model_type='other'),
])
def test_unsupported_actual_metadata_refused_before_engine(monkeypatch, tmp_path, metadata):
    monkeypatch.setattr(families, 'read_config', lambda *_: metadata)
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--delta4-head'])
    with pytest.raises(ValueError, match='dense Gemma and other formats'):
        cli._check_serve_options(args, family(), 'cuda', tmp_path)
    with pytest.raises(ValueError, match='dense Gemma and other formats'):
        gemma4.cuda_engine(tmp_path, delta4_head=True)


def test_lossless_conflict_rejected_before_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(families, 'read_config', lambda *_: pytest.fail('conflict read metadata'))
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--delta4-head', '--lossless-head'])
    with pytest.raises(ValueError, match='mutually exclusive'):
        cli._check_serve_options(args, family(), 'cuda', tmp_path)
    with pytest.raises(ValueError, match='mutually exclusive'):
        gemma4.cuda_engine(tmp_path, delta4_head=True, lossless_head=True)


@pytest.mark.parametrize('value', [None, 1, 'yes'])
def test_nonboolean_rejected_without_metadata(monkeypatch, tmp_path, value):
    monkeypatch.setattr(families, 'read_config', lambda *_: pytest.fail('invalid boolean read metadata'))
    with pytest.raises(ValueError, match='must be a bool'):
        gemma4.cuda_engine(tmp_path, delta4_head=value)
    args = cli.build_parser().parse_args(['serve', str(tmp_path)])
    args.delta4_head = value
    with pytest.raises(ValueError, match='must be a boolean'):
        cli._check_serve_options(args, family(), 'cuda', tmp_path)


@pytest.mark.parametrize('selected', [None, False, True])
def test_family_forwards_explicit_flag_and_preserves_production_options(monkeypatch, tmp_path, selected):
    from tensorfold.cuda import precision
    from tensorfold.families.gemma4.cuda import engine as engine_module

    monkeypatch.setattr(families, 'read_config', lambda *_: raw())
    monkeypatch.setattr(precision, 'mode', lambda: precision.CHECKPOINT)
    seen = []
    monkeypatch.setattr(engine_module, 'Gemma4Engine', lambda *a, **k: seen.append(k) or 'loaded')
    options = {} if selected is None else {'delta4_head': selected}
    assert gemma4.cuda_engine(tmp_path, drafter='owned/draft', context=131072, drafter_bits=8,
                             draft_rows=5, vision=True, chat_template_file='owned-google.jinja', **options) == 'loaded'
    assert seen[0].get('delta4_head', False) is (selected is True)
    assert ('delta4_head' in seen[0]) is (selected is True)
    assert seen[0]['depth'] == 4 and seen[0]['drafter_bits'] == 8 and seen[0]['context'] == 131072
    assert seen[0]['vision'] is True and seen[0]['chat_template_file'] == 'owned-google.jinja'
    for name in ('delta4_reuse', 'global_cache', 'prepare_global_prompt', 'dflash_graph'):
        assert name not in seen[0]  # fixed backend defaults, no unrelated activation


@pytest.mark.parametrize('selected', [False, True])
def test_native_cli_app_forwards_only_explicit_true(monkeypatch, tmp_path, request, selected):
    import tensorfold.cuda.server as server
    from tensorfold.cuda import prompt_precision

    seen = []
    def engine(*args, **kwargs):
        seen.append(kwargs)
        return cuda_cli_engine(max_len=8192, w=NS(fast_prefill=False))
    found = NS(title='fixture', model_type='fixture', package=NS(cuda_engine=engine, CUDA_DELTA4_HEAD=True))
    monkeypatch.setattr(server, 'App', lambda *a, **k: cuda_cli_app(effective_context_window=8192))
    install_cli_server(monkeypatch, request, server)
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--backend', 'cuda', '--no-drafts']
                                        + (['--delta4-head'] if selected else []))
    try:
        assert cli._serve_cuda(args, found, tmp_path, 8192) == 0
    finally:
        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)
    assert len(seen) == 1 and ('delta4_head' in seen[0]) is selected
    assert seen[0].get('delta4_head', False) is selected
    assert 'delta4_reuse' not in seen[0]


def test_legacy_sym8_stays_modelopt_only(monkeypatch, tmp_path):
    monkeypatch.setattr(families, 'read_config', lambda *_: raw())
    with pytest.raises(ValueError, match='ModelOpt MoE'):
        gemma4.cuda_engine(tmp_path, lossless_head=True)
