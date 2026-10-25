"""Installed CPU media closure; these controls require the separate source media image."""

from pathlib import Path
import sys
import subprocess
import types

import pytest

PINS = Path("/opt/tensorfold-media-pins")
pytestmark = pytest.mark.skipif(not (PINS / "media-install.json").is_file(), reason="selected media image required")


def producer():
    sys.path.insert(0, "/opt/tensorfold-pins")
    try:
        return readonly_module("installed_media_producer", Path("/opt/tensorfold-pins/media_foundation.py"))
    finally:
        sys.path.pop(0)


def readonly_module(name, path):
    # The supported probe CLI reads the protected source without a bytecode
    # cache. Keep this same-process test from writing into the sealed inventory.
    module = types.ModuleType(name)
    module.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), "exec"), module.__dict__)
    return module


def test_actual_installed_media_inventory_and_cpu_codecs():
    media = producer()
    manifest = media.read_json(PINS / "media-install.json")[0]
    result = media.audit_install(manifest, media.read_json(PINS / "media-pins.json")[0])
    assert result["software_integrity"] == "passed" and result["release_qualified"] is False
    actual = manifest["runtime"]
    assert actual["torchvision_cuda"] == -1
    assert actual["packages"]["torch"] == "2.14.1+tfcu129sm70"
    assert actual["image_backend"] == "torchvision" and actual["decoder_backend"] == "pyav"
    assert {row["name"] for row in actual["decoders"]} >= {"h264", "hevc", "vp9", "rawvideo", "libdav1d"}
    assert "av1" in actual["known_unavailable_cpu_decoders"]
    aliases = {name for row in actual["demuxers"] for name in row["name"].split(",")}
    assert aliases >= {"mov", "matroska", "yuv4mpegpipe"}
    assert "--disable-encoders" in actual["ffmpeg_configuration"]
    assert "--disable-hwaccels" in actual["ffmpeg_configuration"]


def test_official_processors_decoder_and_actual_tiny_fp16_model():
    module = readonly_module("installed_media_probe", PINS / "media_probe.py")
    result = module.probe(PINS)
    assert result["image_class"] == "Qwen2VLImageProcessor" and result["image_backend"] == "torchvision"
    assert result["video_class"] == "Qwen3VLVideoProcessor"
    assert result["decoder_timestamps"] == [0.0, 0.5, 1.0, 1.5]
    assert result["av1_decoder"] == {"provider": "libdav1d", "frames": 24, "bytes": 864000,
                                     "pixel_sha256": "27fffcfe81c650283c83d03ca28f3956162eae1de093a13affb4e99f1f037e97"}
    assert result["tiny_model_type"] == "Qwen4ExpVisionModel" and result["release_qualified"] is False


def test_warm_media_and_base_process_binding_use_the_same_actual_process(monkeypatch):
    import torchvision
    from tensorfold.cuda.sm70 import software_binding as binding
    from tensorfold.cuda.sm70.software_binding import issue_source_binding, verify_source_binding

    media = producer()
    manifest = media.read_json(PINS / "media-install.json")[0]
    assert torchvision.extension._check_cuda_version() == -1 and torchvision.extension._has_ops()
    assert media.audit_install(manifest, media.read_json(PINS / "media-pins.json")[0])["software_integrity"] == "passed"
    authority = issue_source_binding(required=True)
    verify_source_binding(authority)
    assert authority.record()["release_qualified"] is False
    with monkeypatch.context() as changed:
        pid = binding.os.getpid()
        changed.setattr(binding.os, "getpid", lambda: pid + 1)
        changed.setattr(binding, "_protected_location", lambda: pytest.fail("foreign PID reached protected IO"))
        with pytest.raises(binding.SoftwareBindingError, match="another process"):
            verify_source_binding(authority)


def test_provider_metadata_queries_capture_their_own_lazy_mappings_before_returning():
    command = (
        "import sys;sys.path.insert(0,'/opt/tensorfold-pins');import source_foundation as f;"
        "a=f.runtime_identity();b=f.runtime_identity();"
        "changed=[key for key in a if a[key]!=b[key]];"
        "print(changed);sys.exit(bool(changed))"
    )
    result = subprocess.run([sys.executable, "-c", command], check=True, capture_output=True,
                            text=True, timeout=60)
    assert result.stdout.strip() == "[]"


def test_media_audit_refuses_an_actual_provider_mapping_outside_owned_prefix(monkeypatch, tmp_path):
    media = producer()
    original = media.foundation_runtime_identity

    def relocated():
        result = original()
        result["mapped_libraries"] = [str(tmp_path / Path(path).name) if Path(path).name.startswith("libavcodec.so")
                                       else path for path in result["mapped_libraries"]]
        return result

    monkeypatch.setattr(media, "foundation_runtime_identity", relocated)
    with pytest.raises(ValueError, match="owned prefix: libavcodec"):
        media.runtime_inventory()
