"""Native vision group scheduling and optional-audio frontend boundaries."""
from types import SimpleNamespace
import numpy as np
import pytest

from tensorfold.vision.gemma4_e2b import GemmaE2BProcessor, _vision_blocks
from tensorfold.vision.audio import AudioInput
from tensorfold.vision.images_http import ImageInputError


def test_native_vision_groups_preserve_contiguous_image_video_and_text_boundaries():
    import torch
    # Native grouping deliberately joins adjacent image/video soft tokens.
    mmtypes = torch.tensor([[0,1,1,2,0,2,2,0,3,3]])
    assert _vision_blocks(mmtypes) == ((1,4),(5,7))
    assert _vision_blocks(torch.zeros(1,4,dtype=torch.long)) == ()


def test_no_audio_profile_requires_no_extractor_and_rejects_direct_audio(monkeypatch):
    component = SimpleNamespace(patch_size=16,pooling_kernel_size=3,max_soft_tokens=280)
    processor = SimpleNamespace(image_token='<image>',video_token='<video>',
                                tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda t:{'<image>':10,'<video>':11}[t]),
                                image_processor=component,video_processor=component)
    raw = {'model_type':'gemma4','vision_config':{'hidden_size':1152},'audio_config':None,
           'image_token_id':10,'video_token_id':11,'audio_token_id':12,
           'text_config':{'vocab_size':128,'max_position_embeddings':262144,'use_bidirectional_attention':'vision'}}
    frontend = GemmaE2BProcessor(raw,processor,lambda **_:None,fingerprint='synthetic-boundary-fixture')
    assert frontend.audio is False and frontend.videos is True
    geometry = frontend.resource_geometry(8192)
    assert geometry['max_audio_samples_per_clip'] == geometry['max_padded_audio_samples'] == 0
    assert geometry['max_bidirectional_visual_rows'] == 4096
    with pytest.raises(ImageInputError, match='no audio input encoder'):
        frontend.prepare('text',[],audios=[AudioInput(np.ones(201,dtype=np.float32))],max_prompt_tokens=8192)

    # No-audio profiles have budget0, so even an empty audio decoder call would
    # violate that decoder's positive-budget contract before any image request.
    monkeypatch.setattr(frontend, 'prepare', lambda *a, **kw: kw['audios'])
    def forbidden(*args, **kwargs):
        raise AssertionError('audio decoder must not run without audio sources')
    monkeypatch.setattr('tensorfold.vision.gemma4_e2b.load_audios', forbidden)
    assert frontend.prepare_messages([{'role':'user','content':'describe'}],lambda _: 'describe',
                                     max_prompt_tokens=8192) == []
