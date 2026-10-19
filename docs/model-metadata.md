# Checkpoint metadata and launch overrides

TensorFold resolves model settings during startup. Keep checkpoint files immutable
while loading or serving, including files selected by explicit template or tokenizer flags.
`config.json` describes the architecture and supported
weight representation; tensor headers and scales must agree with that declaration.
Ancillary metadata supplements it. Conflicting representation declarations are
errors, rather than permission to reinterpret weights.

| Setting | Automatic input | Explicit override |
| --- | --- | --- |
| Quantized weights | Model/text quantization blocks and recognized ModelOpt `hf_quant_config.json`, including per-layer schemes and excluded projections; actual storage must match the selected family reader | Choose a checkpoint in a supported format; flags do not reinterpret its stored bytes |
| Dense Qwen CUDA NVFP4 activations | Per-projection declared activation schemes, validated storage/scales and actual GPU capabilities | `--precision full` requests BF16 activations against stored weights; `--precision checkpoint` selects declared supported math |
| Reply EOS | Model, generation and tokenizer EOS declarations, deduplicated in stable order | Repeat `--eos-token-id ID` to replace reply EOS IDs |
| Chat formatting | Checkpoint Jinja or tokenizer-config template, including named `default`/`tool_use` templates | `--chat-template FILE` or `--chat-template-name NAME` |
| CUDA text tokenization | Checkpoint `tokenizer.json` | `--tokenizer FILE` with the same vocabulary model and added tokens; only text handling (normalizer, pre-tokenizer, post-processor, decoder) may differ. Not with `--vision` or several ranks |
| Sampling | `generation_config.json`: `do_sample`, `temperature`, `top_k`, `top_p`, `min_p`, `repetition_penalty` | Corresponding launch flags (and `--presence-penalty`, `--frequency-penalty`), then request fields |
| Reply limit | `generation_config.json` `max_new_tokens`, otherwise 4096 | `--max-tokens`, then the request's reply limit |
| Context/RoPE | Model's supported native window and RoPE metadata, fitted to available memory | `--context` and supported family-specific RoPE flags |

CUDA uses the checkpoint tokenizer's vocabulary, normalizer, special tokens and
postprocessor, or the text handling of an admitted `--tokenizer` file. Saved
padding/truncation settings describe batch preparation; TensorFold disables them
on its private inference tokenizer so short prompts are not padded and longer
prompts are never silently shortened. Token counting sees
the complete sequence. Generation retains the server's explicit context
admission, including after image/video expansion.

Hugging Face metadata is fetched before compatibility checks and large weight
downloads. Online preflight verifies the published file inventory and exact commit;
network, authorization and I/O errors are reported explicitly. A cached snapshot
pins metadata and subsequent weights to that revision; `pull` resolves the latest
published revision first. Metadata fetches include tokenizer and named Jinja files, ModelOpt and
EXL3 sidecars; they do not execute model code. With `HF_HUB_OFFLINE=1`, only files
already present can be used. Finish downloading a snapshot before taking it offline.
Local directories are used as supplied, without remote lookups.

A nonempty declared config/text quantization block takes precedence over legacy EXL3
inventory sidecars. With no declaration, explicit quantization-method sidecars
are detected before capability checks; an unsupported method is refused early.
Unneeded EXL3 tensor inventories are not repeatedly parsed or copied into the
effective configuration.

Explicit zero/false sampling settings remain overrides. `do_sample: false`
selects greedy generation unless a launch/request temperature overrides it.
`max_length` includes the input in Transformers, so TensorFold does not mistake it
for a reply-only limit. Other Transformers generation processors and decoding
algorithms are not automatically implemented by reading their configuration.
Inherited `max_new_tokens` must be a positive integer. Explicit nonpositive
`--max-tokens` values retain the existing request-owner minimum of one reply token.

Quantization targets support literal module names, the `Linear` class and a bounded
`re:` subset: literals, escaped punctuation, `.` and `.*`, with optional `^`/`$`
anchors. Exclusions also support `*`/`?` globs. Other regular-expression operators are refused during preparation.
Checkpoint-controlled patterns never execute a general backtracking regex.

Dense Qwen CUDA also honors an explicit attention-layer layout, including an
exactly declared auxiliary MTP tail without enabling MTP. Its supported text RoPE
is the default unscaled algorithm or static YaRN (declared, or selected with
`--yarn-factor`); other scaling, non-SiLU activation or attention bias is
rejected before loading. Native dtype hints do not implicitly
select lossy activation or KV math.

Tokenizer EOS adds only its specifically declared EOS token, rather than every
special token in its vocabulary. PLE's training EOS is a separate model-owned
setting and is unaffected by a reply-stop override. Rendering, strict tool-format
checks, counting fingerprints and prefix boundaries use the selected template.
Selecting a template does not add an unsupported strict tool-call grammar.
Modern tokenizer configs with `added_tokens_decoder` own their named roles;
otherwise legacy `special_tokens_map.json` replaces the corresponding config
roles. Rendering and reply stopping use the same resolved EOS role.

Automatic precision selection follows the declared numerical contract and actual
GPU capabilities. An activation scale's presence alone does not authorize FP4/FP8
activations in a weight-only projection. A faster, different arithmetic mode is
available through the precision flag; it is not silently substituted. Optional
vision, remote code, GPU topology and lossy KV formats still require their explicit
controls. Thinking remains enabled by default when supported, with `--no-thinking`
and per-request controls; arbitrary Jinja defaults are not inferred as a safe
server-wide reasoning policy.
The new per-projection activation selection applies to dense Qwen's NVFP4 reader;
other family readers retain their documented arithmetic, including Flash Next's
existing W4A16 expert path. Metadata recognition does not add a missing kernel.

Native calibrated activation math requires usable positive FP32 calibration,
inverse and combined scales. Unsupported scale ranges are diagnosed during
preparation; values are not clamped or repaired. Default text-RoPE frequencies
must also remain finite in the actual runtime's FP32 storage.

Use `tensorfold info MODEL` for format/sampling inspection. Startup reports the
effective reply EOS, selected template source/hash and reply-token limit on CUDA.
These identify the bindings actually prepared, rather than rereading files to
describe running state.

[Local installed-image verification](model-metadata-validation.md) covers real
NVFP4 defaults and overrides, automatic context fitting, Flash Next regression
and the dedicated-host zero-VRAM-reserve policy, with execution limits stated.

The schema contracts follow the pinned
[ModelOpt exporter](https://github.com/NVIDIA/Model-Optimizer/blob/0.46.0/modelopt/torch/export/convert_hf_config.py),
[Transformers generation settings](https://huggingface.co/docs/transformers/main_classes/text_generation)
and [chat-template storage/selection](https://huggingface.co/docs/transformers/chat_templating_writing).
Recognizing a schema does not establish execution coverage for every model or GPU.
