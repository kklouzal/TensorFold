# Flash Next CUDA images and video

Run a complete local Flash Next vision checkpoint with `--vision --parallel 2` (or more), on one CUDA GPU.
The fork supports multiple images and video for `qwen4_exp` checkpoints that include the vision tower,
processor files and video token metadata. The GB10 profile uses four slots, a count limit of 50 images,
and separate 16,384-token image and video resize budgets:

```bash
TENSORFOLD_IMAGE_TOKENS=16384 TENSORFOLD_VIDEO_TOKENS=16384 \
  tensorfold serve /models/flash-next --backend cuda --vision --parallel 4 \
  --vision-max-images 50 --name Qwen3.8-Flash-Next --port 8888
```

Use the [GB10 deployment profile](../../deploy/gb10/README.md) to retain its model mounts, INT8 KV,
YaRN-2 context, sampler and other service settings. The command above demonstrates vision options;
its context still follows normal startup admission.

Send OpenAI `image_url` content parts for images, or `video_url` parts for clips, in user messages.
Data URLs work without remote fetching. Public HTTPS URLs require `--vision-urls` and the same URL
safety checks as [image input](../vision.md#send-an-image); local file URLs are refused. Video decoding
requires PyAV (`av`, included in the pinned container). For example, with a local MP4:

```python
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen

clip = base64.b64encode(Path("clip.mp4").read_bytes()).decode()
body = {
    "model": "Qwen3.8-Flash-Next",
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": "Describe the visible action in this clip."},
        {"type": "video_url", "video_url": {"url": "data:video/mp4;base64," + clip}},
    ]}],
    "max_tokens": 256,
}
request = Request("http://127.0.0.1:8888/v1/chat/completions",
                  data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
with urlopen(request) as response:
    print(json.load(response)["choices"][0]["message"])
```

`--vision-max-images` counts images across the entire submitted history. Flash Next's image frontend
allows 10 MiB encoded per image and 64 MiB total, up to 8,192 pixels per dimension,
16,777,216 decoded pixels per image and 134,217,728 total. `TENSORFOLD_IMAGE_TOKENS` sets the shared
image-token cap from 4,096 to 16,384 (default 16,384); each image remains capped at 4,096 visual tokens. More images share that
budget and can receive less detail. The tower processes bounded runs rather than encoding the entire
image history in one call.

Video input accepts up to two clips, 64 MiB encoded per clip and 96 MiB total, with source dimensions
up to 8,192 and duration up to one hour. Frames are sampled at a target of two per second, with at most
256 spread over the clip; at least two decodable frames are required. `TENSORFOLD_VIDEO_TOKENS`
controls the per-clip resize target (default 16,384), rather than a strict token ceiling. CUDA also
checks a separate request limit of 262,144 video patches and encodes bounded frame groups. HTTP body
limits apply after data-URL encoding. Audio is not transcribed.

Expanded image/video rows count toward prompt usage and the context window. They replace embeddings
in every hyperconnection stream; multimodal RoPE follows the checkpoint's sections, and text decode
continues with the media-position offset. Media prompts prefill from the start and are not retained
in the text-prefix cache, since identical placeholder IDs can name different pixels. Grammar and
ignore-EOS settings remain available.

A floating-point tower in the indexed checkpoint is discovered normally. EXL3 packs whose vision tower is a
quantized sidecar need a one-time CPU conversion, stored outside the original model snapshot:

```bash
python -m tensorfold.vision.exl3_convert /models/vision_k6.safetensors /cache/vision-f16-v3.safetensors
TENSORFOLD_VISION_WEIGHTS=/cache/vision-f16-v3.safetensors tensorfold serve /models --vision --parallel 2
```

The converter decodes represented EXL3 weights, transposes matrices and combines split Q/K/V. It records the
source SHA256 and converter/dtype version; repeat conversions reuse a matching artifact and refuse to
replace a mismatched one. Conversion never runs in the serving loader. The FP16 artifact loads into the
existing BF16 CUDA tower; rounding may differ from native quantized vision execution, so compare image
features and quality for the checkpoint in use. The original weights stay unchanged.

Admission counts an external tower separately, including its expanded resident bytes. It reserves 4 GiB of
image workspace by default; `TENSORFOLD_VISION_WORKSPACE_MIB` sets a measured override from 0 to 16384 MiB.
Image/video requests cannot use yieldable background lanes. Distributed vision and serial-only Flash Next
vision serving are not supported by this port. The earlier GB10 deployment exercised multiple images and
MP4 recognition; see its [recorded scope](../../deploy/gb10/README.md). Those checks do not establish
general video understanding or qualify a changed runtime; each replacement needs its own media smoke checks.

The integration is adapted from MiaAI-Lab patch 0008, with its MIT license in `LICENSES/MiaAI-Lab-MIT.txt`.
