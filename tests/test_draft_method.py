"""Explicit proposal selection cannot reinterpret a different drafter checkpoint."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest


@pytest.mark.parametrize('method,architecture,expected',[('auto','DSparkDraftModel','dspark_model_dir'),('dspark','DSparkDraftModel','dspark_model_dir'),('auto','StockMTP','mtp_model_dir'),('mtp','StockMTP','mtp_model_dir')])
def test_matching_checkpoint_and_explicit_or_auto_method(tmp_path,monkeypatch,method,architecture,expected):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine
    monkeypatch.setattr(exl3,'is_exl3',lambda _:False)
    monkeypatch.setattr(gguf,'resolve_gguf',lambda _:Path('target.gguf'))
    calls=[]
    monkeypatch.setattr(engine,'FlashNextEngine',lambda directory,**kw:calls.append(kw) or NS(options=kw))
    draft=tmp_path/'draft'
    draft.mkdir()
    (draft/'config.json').write_text(json.dumps({'architectures':[architecture]}))
    result=family.cuda_engine('target',drafter=str(draft),mtp_drafts=3,kv_dtype='fp16',draft_method=method)
    assert result.options[expected]==draft and len(calls)==1


@pytest.mark.parametrize('method,architecture',[('mtp','DSparkDraftModel'),('dspark','StockMTP')])
def test_mismatched_method_refuses_before_engine_ownership(tmp_path,monkeypatch,method,architecture):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine
    monkeypatch.setattr(exl3,'is_exl3',lambda _:False)
    monkeypatch.setattr(gguf,'resolve_gguf',lambda _:Path('target.gguf'))
    monkeypatch.setattr(engine,'FlashNextEngine',lambda *a,**kw:pytest.fail('mismatch acquired engine'))
    draft=tmp_path/'draft'
    draft.mkdir()
    (draft/'config.json').write_text(json.dumps({'architectures':[architecture]}))
    with pytest.raises(ValueError,match='does not match'):
        family.cuda_engine('target',drafter=str(draft),mtp_drafts=3,kv_dtype='fp16',draft_method=method)


@pytest.mark.parametrize('backend,methods',[('mlx',('auto','mtp','dspark')),('cuda',())])
def test_unsupported_backend_or_family_refuses_before_other_options(backend,methods):
    from tensorfold.serve_options import check
    family=NS(package=NS(CUDA_DRAFT_METHODS=methods))
    with pytest.raises(ValueError,match='draft-method'):
        check(NS(draft_method='dspark'),family,backend)


def test_parser_exposes_configurable_method_and_rejects_unknown():
    from tensorfold.cli_args import build_parser
    from collections import defaultdict
    def parser():
        return build_parser(defaultdict(lambda: lambda args: None))
    parsed=parser().parse_args(['serve','target','--draft-method','dspark','--drafter','draft','--mtp-drafts','7'])
    assert (parsed.draft_method,parsed.drafter,parsed.mtp_drafts)==('dspark','draft',7)
    with pytest.raises(SystemExit):
        parser().parse_args(['serve','target','--draft-method','unknown'])
    assert parser().parse_args(['serve','target','--draft-method','dflash2']).draft_method == 'dflash2'


@pytest.mark.parametrize('method', ['auto', 'dflash2'])
def test_dflash2_selection_preserves_one_draft_owner_and_requires_manifest(tmp_path, monkeypatch, method):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine

    monkeypatch.setattr(exl3, 'is_exl3', lambda _: False)
    monkeypatch.setattr(gguf, 'resolve_gguf', lambda _: Path('target.gguf'))
    calls = []
    monkeypatch.setattr(engine, 'FlashNextEngine', lambda directory, **kw: calls.append(kw) or NS(options=kw))
    draft = tmp_path / 'draft'
    draft.mkdir()
    (draft / 'config.json').write_text(json.dumps({'architectures': ['DFlash2DraftModel']}))
    with pytest.raises(ValueError, match='pairing'):
        family.cuda_engine('target', drafter=str(draft), tp=8, mtp_drafts=3,
                           kv_dtype='fp16', draft_method=method)
    assert not calls
    result = family.cuda_engine('target', drafter=str(draft), tp=8, mtp_drafts=3,
                               kv_dtype='fp16', draft_method=method, execution_profile_manifest=tmp_path / 'manifest.json')
    assert result.options['dflash2_model_dir'] == draft
    assert result.options['adaptive_sessions'] is True
    assert not {'mtp_model_dir', 'dspark_model_dir'} & result.options.keys()


@pytest.mark.parametrize("method,refuses", [("mtp",False),("dspark",True)])
def test_non_gguf_included_mtp_remains_selectable(monkeypatch,method,refuses):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine
    monkeypatch.setattr(exl3,"is_exl3",lambda _:False)
    monkeypatch.setattr(gguf,"resolve_gguf",lambda _:None)
    monkeypatch.setattr(family,"has_mtp",lambda _:True)
    calls=[]
    monkeypatch.setattr(engine,"FlashNextEngine",lambda directory,**kw:calls.append(kw) or NS(options=kw))
    if refuses:
        with pytest.raises(ValueError,match="DSpark currently requires"):
            family.cuda_engine("target",draft_method=method,mtp_drafts=3,kv_dtype="fp16")
        assert not calls
    else:
        result=family.cuda_engine("target",draft_method=method,mtp_drafts=3,kv_dtype="fp16")
        assert result.options["depth"]==3 and len(calls)==1


@pytest.mark.parametrize("gguf_target", [False, True])
def test_none_selects_serial_resources_without_acquiring_any_drafter(tmp_path, monkeypatch, gguf_target):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine

    monkeypatch.setattr(exl3, "is_exl3", lambda _: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: Path("target.gguf") if gguf_target else None)
    monkeypatch.setattr(family, "has_mtp", lambda _: pytest.fail("none inspected optional MTP weights"))
    monkeypatch.setattr(engine, "FlashNextEngine", lambda directory, **kw: NS(options=kw))
    result = family.cuda_engine("target", drafter="missing-drafter", draft_method="none", kv_dtype="fp16")
    assert result.options["depth"] == 0
    assert not {"mtp_model_dir", "dspark_model_dir"} & result.options.keys()


@pytest.mark.parametrize("parallel,automatic,ceiling", [(None, False, None), ("auto", False, None),
                                                       (1, False, 1), (47, False, 47), (1, True, None)])
def test_gguf_preserves_automatic_capacity_intent_and_explicit_session_ceilings(monkeypatch, parallel,
                                                                            automatic, ceiling):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine

    monkeypatch.setattr(exl3, "is_exl3", lambda _: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: Path("target.gguf"))
    monkeypatch.setattr(engine, "FlashNextEngine", lambda directory, **kw: NS(options=kw))
    options = {"parallel": parallel}
    if automatic:
        options["parallel_auto"] = True
    result = family.cuda_engine("target", draft_method="none", kv_dtype="fp16", **options)
    assert result.options["adaptive_sessions"] is True
    assert result.options["session_ceiling"] == ceiling


@pytest.mark.parametrize("gguf_target,history,allowed", [(True, "gpu", True), (True, "ram", False),
                                                       (False, "gpu", False)])
def test_full_window_auto_only_selects_implemented_adaptive_gpu_history(monkeypatch, gguf_target, history, allowed):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine

    monkeypatch.setattr(exl3, "is_exl3", lambda _: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: Path("target.gguf") if gguf_target else None)
    monkeypatch.setattr(engine, "FlashNextEngine", lambda directory, **kw: NS(options=kw))
    if allowed:
        result = family.cuda_engine("target", draft_method="none", kv_full_windows="auto", indexer_history=history)
        assert "kv_full_windows" not in result.options
        assert result.options["adaptive_sessions"] is True
    else:
        with pytest.raises(ValueError, match="kv_full_windows auto"):
            family.cuda_engine("target", draft_method="none", kv_full_windows="auto", indexer_history=history)


def test_explicit_mtp_zero_maximum_does_not_silently_disable_selected_method(monkeypatch):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine

    monkeypatch.setattr(exl3, "is_exl3", lambda _: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: Path("target.gguf"))
    monkeypatch.setattr(engine, "FlashNextEngine", lambda *a, **kw: pytest.fail("invalid MTP acquired engine"))
    with pytest.raises(ValueError, match="positive maximum depth"):
        family.cuda_engine("target", draft_method="mtp", mtp_drafts=0)


@pytest.mark.parametrize("parallel", [False, 1.5, -1, 0, "2"])
def test_gguf_ceiling_rejects_noninteger_or_nonpositive_api_counts(monkeypatch, parallel):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf

    monkeypatch.setattr(exl3, "is_exl3", lambda _: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: Path("target.gguf"))
    with pytest.raises(ValueError, match="ceiling"):
        family.cuda_engine("target", draft_method="none", parallel=parallel)


@pytest.mark.parametrize("gguf_target,history,allowed", [(True, "gpu", True), (True, "ram", False),
                                                       (False, "gpu", False)])
def test_execution_manifest_is_forwarded_only_to_supported_resident_provider(monkeypatch, gguf_target,
                                                                           history, allowed):
    import tensorfold.cuda.exl3.format as exl3
    import tensorfold.families.qwen4_exp as family
    import tensorfold.families.qwen4_exp.gguf as gguf
    import tensorfold.families.qwen4_exp.cuda.engine as engine

    monkeypatch.setattr(exl3, "is_exl3", lambda _: False)
    monkeypatch.setattr(gguf, "resolve_gguf", lambda _: Path("target.gguf") if gguf_target else None)
    monkeypatch.setattr(engine, "FlashNextEngine", lambda directory, **kw: NS(options=kw))
    if allowed:
        result = family.cuda_engine("target", draft_method="none", execution_profile_manifest="qualified.json",
                                    indexer_history=history)
        assert result.options["execution_profile_manifest"] == Path("qualified.json")
    else:
        with pytest.raises(ValueError, match="execution_profile_manifest"):
            family.cuda_engine("target", draft_method="none", execution_profile_manifest="qualified.json",
                               indexer_history=history)


@pytest.mark.parametrize("backend,model_type", [("mlx", "qwen4_exp"), ("cuda", "qwen3_5")])
def test_execution_manifest_cli_rejects_unrelated_provider_before_loading(backend, model_type):
    from tensorfold.serve_options import check

    family = NS(package=NS(), model_type=model_type)
    args = NS(execution_profile_manifest=Path("qualified.json"))
    with pytest.raises(ValueError, match="execution-profile-manifest"):
        check(args, family, backend)
