"""Public media gates, preparation lifetime and diagnostic privacy."""
from types import SimpleNamespace

import pytest

from tensorfold.server.errors import RequestError
from tensorfold.server.failure_capture import redact_images
from tensorfold.server.messages import audio_capable, normalize_messages, validate_modalities
from tensorfold.server import prompts
from tensorfold.server.responses_translate import _content, messages
from tensorfold.server.responses import _media_options
from tensorfold.vision.audio import AudioSource
from tensorfold.vision.images import split_images


def audio_message():
    return [{'role': 'user', 'content': [
        {'type': 'text', 'text': 'Describe this recording.'},
        {'type': 'input_audio', 'input_audio': {'data': 'AAAA', 'format': 'wav'}}]}]


def test_audio_requires_loaded_capability_and_preserves_input():
    original = audio_message()
    with pytest.raises(RequestError):
        normalize_messages(original, allow_images=True)
    assert normalize_messages(original, allow_images=True, allow_audio=True) == original
    assert not audio_capable(SimpleNamespace(vision=SimpleNamespace(audio=False)))
    assert audio_capable(SimpleNamespace(vision=SimpleNamespace(audio=True)))
    assert prompts.has_images(original)  # shared existing predicate now includes admitted audio media


def test_audio_markers_render_only_on_audio_template_route():
    marker = [{'role': 'user', 'content': [{'type': 'audio'}]}]
    assert normalize_messages(marker, allow_audio=True, audio_placeholders=True) == marker
    with pytest.raises(RequestError):
        normalize_messages(marker, allow_images=True, allow_audio=True)
    with pytest.raises(RequestError):
        normalize_messages(marker, allow_images=True)


@pytest.mark.parametrize('role', ['assistant', 'system', 'developer', 'tool'])
def test_audio_is_not_reinterpreted_under_another_role(role):
    value = audio_message()
    value[0]['role'] = role
    with pytest.raises(RequestError):
        normalize_messages(value, allow_images=True, allow_audio=True)


def test_split_audio_preserves_order_and_uses_native_source():
    template, sources = split_images(audio_message(), allow_audio=True)
    assert template[0]['content'] == [{'type': 'text', 'text': 'Describe this recording.'}, {'type': 'audio'}]
    assert sources == [AudioSource('AAAA', 'wav')]
    with pytest.raises(ValueError):
        split_images(audio_message())


def test_audio_output_is_not_advertised_by_input_support():
    with pytest.raises(RequestError):
        validate_modalities({'modalities': ['audio']})
    validate_modalities({'modalities': ['text'], 'messages': audio_message()})


def test_responses_audio_has_same_input_shape_and_explicit_gate():
    part = audio_message()[0]['content'][1]
    with pytest.raises(RequestError):
        _content([part])
    assert _content([part], allow_audio=True) == [part]
    assert messages([{'role': 'user', 'content': [part]}], allow_audio=True)[0]['content'] == [part]
    with pytest.raises(RequestError):
        messages([{'role': 'assistant', 'content': [part]}], allow_audio=True)


def test_native_embedded_video_does_not_require_durable_conversation_store():
    app = SimpleNamespace(vision=SimpleNamespace(audio=True, videos=True, responses_video=True))
    assert _media_options(app, SimpleNamespace(durable_conversations=False)) == {
        'allow_video': True, 'allow_audio': True}
    old = SimpleNamespace(vision=SimpleNamespace(is_media_runtime=True, videos=True, media_envelope=object()))
    assert _media_options(old, SimpleNamespace(durable_conversations=False))['allow_video'] is False
    assert _media_options(old, SimpleNamespace(durable_conversations=True))['allow_video'] is True


def test_new_audio_payload_is_never_saved_in_failure_or_request_logs():
    value = {'messages': audio_message()}
    safe = redact_images(value)
    assert safe['messages'][0]['content'][1]['input_audio'] == {'data': '<redacted>', 'format': 'wav'}
    assert value['messages'][0]['content'][1]['input_audio']['data'] == 'AAAA'


def test_bounded_frontend_receives_real_context_and_request_limits():
    seen = {}
    def prepare(value, render, **kwargs):
        seen.update(kwargs)
        assert value == audio_message()
        return SimpleNamespace(token_ids=(2, 3, 4))
    frontend = SimpleNamespace(audio=True, prepare_messages=prepare)
    result = prompts.prepare_images(frontend, audio_message(), str, context_limit=131072)
    assert result.tokens == [2, 3, 4]
    assert seen['max_prompt_tokens'] == 131072
    assert seen['image_limits'] is prompts.DEFAULT_LIMITS


def test_audio_prepare_failure_releases_both_host_admission_guards():
    def fail(*args, **kwargs):
        raise ValueError('malformed waveform')
    frontend = SimpleNamespace(audio=True, prepare_messages=fail)
    with pytest.raises(RequestError, match='malformed waveform'):
        prompts.prepare_images(frontend, audio_message(), str, context_limit=2048)
    assert prompts.MANY_IMAGES.acquire(blocking=False)
    prompts.MANY_IMAGES.release()
    held = []
    try:
        for _ in range(16):
            assert prompts.IMAGE_SLOTS.acquire(blocking=False)
            held.append(True)
        assert not prompts.IMAGE_SLOTS.acquire(blocking=False)
    finally:
        for _ in held:
            prompts.IMAGE_SLOTS.release()


def test_vision_only_frontend_owns_video_preparation_without_claiming_audio(monkeypatch):
    value = [{'role':'user','content':[{'type':'video_url','video_url':{'url':'data:video/mp4;base64,AAAA'}}]}]
    seen = {}
    def prepare(messages, render, **kwargs):
        assert messages is value
        seen.update(kwargs)
        return SimpleNamespace(token_ids=(5,6),blocks=((1,2),))
    def legacy_decoder(*args, **kwargs):
        raise AssertionError('owned frontend must preserve its native video timestamps')
    monkeypatch.setattr('tensorfold.vision.videos.load_videos',legacy_decoder)
    frontend = SimpleNamespace(audio=False,videos=True,prepare_messages=prepare)
    result = prompts.prepare_images(frontend,value,str,context_limit=8192)
    assert result.tokens == [5,6] and result.vision.blocks == ((1,2),)
    assert seen['max_prompt_tokens'] == 8192 and seen['image_limits'] is prompts.DEFAULT_LIMITS
    assert not audio_capable(SimpleNamespace(vision=frontend))
    with pytest.raises(RequestError):
        normalize_messages(audio_message(),allow_images=True,allow_audio=frontend.audio)


@pytest.mark.parametrize('preserve',[False,True])
@pytest.mark.parametrize('decoder_fails',[False,True])
def test_legacy_video_route_forwards_explicit_pts_capability_and_retires_slot(monkeypatch,preserve,decoder_fails):
    value = [{'role':'user','content':[{'type':'video_url','video_url':{'url':'data:video/mp4;base64,AAAA'}}]}]
    seen = {}
    def decode(clips,size,**kwargs):
        seen.update(kwargs)
        if decoder_fails:
            raise ValueError('invalid video timeline')
        return ['validated-video']
    def prepare(text,images,**kwargs):
        assert kwargs['videos'] == ['validated-video']
        return SimpleNamespace(token_ids=(7,8))
    monkeypatch.setattr('tensorfold.vision.videos.load_videos',decode)
    frontend = SimpleNamespace(videos=True,allow_urls=False,preserve_video_timestamps=preserve,
                                video_size=lambda *args: (48,48),prepare=prepare)
    if decoder_fails:
        with pytest.raises(RequestError,match='invalid video timeline'):
            prompts.prepare_images(frontend,value,str,context_limit=8192)
    else:
        assert prompts.prepare_images(frontend,value,str,context_limit=8192).tokens == [7,8]
    assert seen == ({'allow_urls':False,'preserve_timestamps':True} if preserve else {'allow_urls':False})
    assert prompts.IMAGE_SLOTS.acquire(blocking=False)
    prompts.IMAGE_SLOTS.release()


def test_actual_gemma_cuda_responses_video_uses_chat_processor_route_without_audio(monkeypatch):
    from tensorfold.families.gemma4.cuda.vision import GemmaCudaVision
    from tensorfold.server.api_record_media import video_capable
    # Exercise the actual wrapper methods without allocating the GPU-only tower.
    wrapper = object.__new__(GemmaCudaVision)
    wrapper.videos = True
    seen = {}
    def prepare(text,images,**kwargs):
        seen['prepared'] = kwargs
        assert images == [] and kwargs['videos'] == ['validated-pts-video']
        return SimpleNamespace(token_ids=(12,13))
    wrapper.frontend = SimpleNamespace(prepare=prepare,video_size=lambda *args:(48,48))
    def decode(clips,size,**kwargs):
        seen['decode'] = kwargs
        assert len(clips) == 1
        return ['validated-pts-video']
    monkeypatch.setattr('tensorfold.vision.videos.load_videos',decode)
    app = SimpleNamespace(vision=wrapper)
    options = _media_options(app,SimpleNamespace(durable_conversations=False))
    assert video_capable(app) and options == {'allow_video':True,'allow_audio':False}
    original = [{'role':'user','content':[{'type':'input_file','filename':'clip.mp4',
                 'file_data':'data:video/mp4;base64,AAAA'}]}]
    translated = messages(original,**options)
    assert translated[0]['content'] == [{'type':'video_url','video_url':{'url':'data:video/mp4;base64,AAAA'}}]
    normalized = normalize_messages(translated,allow_images=True,allow_audio=False)
    result = prompts.prepare_images(wrapper,normalized,str,context_limit=8192)
    assert result.tokens == [12,13]
    assert seen['decode'] == {'allow_urls':False,'preserve_timestamps':True}
    assert seen['prepared']['max_prompt_tokens'] == 8192
    assert not audio_capable(app)
    with pytest.raises(RequestError):
        messages(audio_message(),**options)
    with pytest.raises(RequestError):
        messages([{'role':'user','content':[{'type':'input_file','filename':'clip.mp4',
                  'file_data':'data:video/mp4;base64,AAAA','file_url':'https://example.com/clip.mp4'}]}],**options)
    assert not video_capable(SimpleNamespace(vision=SimpleNamespace(videos=True)))
