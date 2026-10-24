"""Explicit proposal selection cannot reinterpret a different drafter checkpoint."""
import argparse
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
    draft=tmp_path/'draft';draft.mkdir();(draft/'config.json').write_text(json.dumps({'architectures':[architecture]}))
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
    draft=tmp_path/'draft';draft.mkdir();(draft/'config.json').write_text(json.dumps({'architectures':[architecture]}))
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
