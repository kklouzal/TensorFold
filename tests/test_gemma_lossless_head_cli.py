"""Opt-in Gemma MoE CUDA CLI capability/metadata routing, no model/CUDA work."""
import json
from types import SimpleNamespace

import pytest

from tensorfold import cli, families
from tensorfold.families import gemma4
from tests.cli_server_fixture import cuda_cli_app, cuda_cli_engine, install_cli_server


def raw(*, moe=True, method='modelopt'):
    return {'model_type': 'gemma4', 'text_config': {'enable_moe_block': moe},
            'quantization_config': {'quant_method': method}}


OPTIONS = [('lossless_head', 'CUDA_LOSSLESS_HEAD'),
           ('spark_global_gqa', 'CUDA_SPARK_GLOBAL_GQA'), ('spark_sampled_q', 'CUDA_SPARK_SAMPLED_Q')]


def family():
    return SimpleNamespace(title=gemma4.TITLE, model_type='gemma4', package=gemma4)


def test_store_true_default_and_public_help(capsys):
    parser = cli.build_parser()
    for option, _ in OPTIONS:
        assert getattr(parser.parse_args(['serve', 'owner/model']), option) is False
        assert getattr(parser.parse_args(['serve', 'owner/model', '--' + option.replace('_', '-')]), option) is True
    with pytest.raises(SystemExit) as stop:
        parser.parse_args(['serve', '--help'])
    assert stop.value.code == 0
    help_text = capsys.readouterr().out
    for option, _ in OPTIONS:
        assert '--' + option.replace('_', '-') in help_text
    assert 'ModelOpt MoE on CUDA' in help_text


@pytest.mark.parametrize('option,capability', OPTIONS)
@pytest.mark.parametrize('backend,declared', [('mlx', True), ('cuda', False)])
def test_unsupported_backend_or_family_rejected_before_metadata(monkeypatch, backend, declared, option, capability):
    args = cli.build_parser().parse_args(['serve', 'owner/model', '--' + option.replace('_', '-')])
    package = SimpleNamespace(**{capability: declared})
    found = SimpleNamespace(title='other', model_type='other', package=package)
    monkeypatch.setattr(families, 'read_config', lambda *_: pytest.fail('unsupported metadata read'))
    with pytest.raises(ValueError, match='requires Gemma 4 ModelOpt MoE on CUDA'):
        cli._check_serve_options(args, found, backend)


@pytest.mark.parametrize('option,capability', OPTIONS)
@pytest.mark.parametrize('metadata', [
    raw(moe=False), raw(method='compressed-tensors'), raw(method='mlx'),
    {'model_type': 'gemma4', 'text_config': {'enable_moe_block': 1},
     'quantization_config': {'quant_method': 'modelopt'}},
])
def test_serve_refuses_dense_and_non_modelopt_metadata_before_engine(monkeypatch, tmp_path, metadata, option, capability):
    (tmp_path / 'config.json').write_text(json.dumps(metadata))
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--' + option.replace('_', '-'), '--no-drafts'])
    with pytest.raises(ValueError, match='dense Gemma and other formats'):
        cli._check_serve_options(args, family(), 'cuda', tmp_path)


@pytest.mark.parametrize('option,capability', OPTIONS)
def test_modelopt_moe_serve_admits_explicit_flag(tmp_path, option, capability):
    (tmp_path / 'config.json').write_text(json.dumps(raw()))
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--' + option.replace('_', '-'), '--no-drafts'])
    cli._check_serve_options(args, family(), 'cuda', tmp_path)


@pytest.mark.parametrize('option,capability', OPTIONS)
@pytest.mark.parametrize('flag', [False, True])
def test_cli_forwards_only_explicit_true_to_capable_factory(monkeypatch, tmp_path, request, flag, option, capability):
    import tensorfold.cuda.server as server
    from tensorfold.cuda import prompt_precision

    seen = []
    def engine(*args, **kwargs):
        seen.append(kwargs)
        return cuda_cli_engine(max_len=8192, w=SimpleNamespace(fast_prefill=False))
    found = SimpleNamespace(title='fixture', model_type='fixture',
                            package=SimpleNamespace(cuda_engine=engine, **{capability: True}))
    monkeypatch.setattr(server, 'App', lambda *a, **k: cuda_cli_app(effective_context_window=8192))
    install_cli_server(monkeypatch, request, server)
    args = cli.build_parser().parse_args(['serve', str(tmp_path), '--backend', 'cuda', '--no-drafts']
                                        + (['--' + option.replace('_', '-')] if flag else []))
    try:
        assert cli._serve_cuda(args, found, tmp_path, 8192) == 0
    finally:
        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)
    assert len(seen) == 1
    assert seen[0].get(option, False) is flag
    assert (option in seen[0]) is flag


@pytest.mark.parametrize('option,capability', OPTIONS)
@pytest.mark.parametrize('selected', [None, False, True])
def test_family_routes_flag_to_only_moe_modelopt_engine(monkeypatch, tmp_path, selected, option, capability):
    from tensorfold.cuda import precision
    from tensorfold.families.gemma4.cuda import engine as engine_module

    monkeypatch.setattr(families, 'read_config', lambda *_: raw())
    monkeypatch.setattr(precision, 'mode', lambda: precision.CHECKPOINT)
    seen = []
    monkeypatch.setattr(engine_module, 'Gemma4Engine', lambda *a, **k: seen.append(k) or 'loaded')
    options = {} if selected is None else {option: selected}
    assert gemma4.cuda_engine(tmp_path, **options) == 'loaded'
    assert (option in seen[0]) is (options.get(option) is True)
    if options.get(option) is True:
        assert seen[0][option] is True


@pytest.mark.parametrize('option,capability', OPTIONS)
@pytest.mark.parametrize('hidden,layers', [(1536, 35), (2560, 42), (5376, 60)])
def test_dense_factory_refuses_before_dense_engine_import(monkeypatch, tmp_path, hidden, layers, option, capability):
    value = raw(moe=False)
    value['text_config'].update(hidden_size=hidden, num_hidden_layers=layers)
    monkeypatch.setattr(families, 'read_config', lambda *_: value)
    with pytest.raises(ValueError, match='dense Gemma and other formats'):
        gemma4.cuda_engine(tmp_path, **{option: True})


@pytest.mark.parametrize('option,capability', OPTIONS)
@pytest.mark.parametrize('method', ['compressed-tensors', 'mlx', None])
def test_nonmodelopt_factory_refuses_before_projection_import(monkeypatch, tmp_path, method, option, capability):
    monkeypatch.setattr(families, 'read_config', lambda *_: raw(method=method))
    with pytest.raises(ValueError, match='dense Gemma and other formats'):
        gemma4.cuda_engine(tmp_path, **{option: True})


@pytest.mark.parametrize('option,capability', OPTIONS)
@pytest.mark.parametrize('value', [1, None, 'yes'])
def test_direct_factory_rejects_non_boolean(monkeypatch, tmp_path, value, option, capability):
    monkeypatch.setattr(families, 'read_config', lambda *_: raw())
    with pytest.raises(ValueError, match='must be a bool'):
        gemma4.cuda_engine(tmp_path, **{option: value})
