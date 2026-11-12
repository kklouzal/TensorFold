"""Frozen authoritative Flash validator and bounded campaign operator staging."""
from pathlib import Path
import pytest

from tensorfold.cuda.sm70 import flash_v100_artifact as artifact
from tools import v100_onecat_campaign as campaign
from tests.test_flash_v100_artifact import sealed  # noqa: F401 - shared pytest fixture dependency
from tests.test_v100_onecat_campaign import ModelArmTests


@pytest.fixture
def bundle(tmp_path,sealed,monkeypatch):
    exports,data,write=sealed;path,digest=write()
    (exports/'manifest.sha256').write_text(digest+'\n')
    capture=tmp_path/'frozen';package=Path(artifact.__file__).parents[2]
    files={}
    for name in artifact.OWNED:
        original=artifact.owned_path(name);relative=Path('src/tensorfold')/original.relative_to(package)
        target=capture/'source'/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(original.read_bytes());files[str(relative)]={'type':'file',**campaign.identity(target)}
    source={'source_uid':'c'*64,'files':files}
    settings={**ModelArmTests().settings(),'flash_v100_exports':str(exports)}
    profile={'preparation':{'kernel_family':artifact.FAMILY,'flash_tiles':{
        'native_manifest':'/operator/flash-v100/manifest.json','native_manifest_sha256':digest}}}
    original=campaign._flash_closure
    def fixture_closure(directory,library,sha):
        # Use one tiny foreign fixture rather than shipping vendor inputs in
        # ordinary tests; every validator/copy/schema implementation is real.
        library.INPUTS=artifact.INPUTS
        return original(directory,library,sha)
    monkeypatch.setattr(campaign,'_flash_closure',fixture_closure)
    return capture,source,settings,profile,exports


def test_frozen_validator_and_exact_tree_are_published_readonly(bundle,tmp_path,monkeypatch):
    capture,source,settings,profile,exports=bundle
    monkeypatch.setattr(artifact,'validate_manifest',lambda *a:pytest.fail('live validator used by frozen campaign'))
    selected=campaign._flash_exports(capture,source,settings,profile)
    operator=tmp_path/'operator';operator.mkdir()
    record=campaign._stage_flash_exports(selected,operator)
    assert record['identity']['profile_sha256']==selected['identity']['profile_sha256']
    assert record['files']==record['copied_files'] and record['closure_files']==16
    assert not (operator/'flash-v100').stat().st_mode & 0o222
    assert campaign.model_arguments(settings,ModelArmTests().plan())==campaign.model_arguments(
        ModelArmTests().settings(),ModelArmTests().plan())


@pytest.mark.parametrize('kind',['unused','missing','path','extra','symlink','changed','validator'])
def test_invalid_or_unbound_artifact_fails_before_model_launch(bundle,tmp_path,kind):
    capture,source,settings,profile,exports=bundle
    if kind=='unused':profile={}
    elif kind=='missing':settings.pop('flash_v100_exports')
    elif kind=='path':profile['preparation']['flash_tiles']['native_manifest']='/elsewhere/manifest.json'
    elif kind=='extra':(exports/'unlisted').write_text('extra')
    elif kind=='symlink':
        file=exports/'build.json';file.unlink();file.symlink_to(tmp_path/'absent')
    elif kind=='changed':
        file=next((exports/'native').iterdir());file.write_bytes(file.read_bytes()+b'!')
    else:
        file=capture/'source/src/tensorfold/cuda/sm70/flash_v100_artifact.py';file.write_text('unexpected')
    with pytest.raises((ValueError,FileNotFoundError)):
        campaign._flash_exports(capture,source,settings,profile)


def test_source_mutation_between_validation_and_copy_is_preserved_as_failure(bundle,tmp_path):
    capture,source,settings,profile,exports=bundle
    selected=campaign._flash_exports(capture,source,settings,profile)
    file=next((exports/'native').iterdir());file.write_bytes(file.read_bytes()+b'!')
    operator=tmp_path/'operator';operator.mkdir()
    with pytest.raises(ValueError):campaign._stage_flash_exports(selected,operator)


def test_existing_nonflash_campaign_has_no_new_artifact_work(bundle):
    capture,source,settings,profile,exports=bundle
    assert campaign._flash_exports(capture,source,ModelArmTests().settings(),{}) is None
