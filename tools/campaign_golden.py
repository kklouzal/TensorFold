"""Golden teacher-forced reference: an eval set's token sequences through the pre-quantization BF16 checkpoint with
Hugging Face transformers and float32 compute, written in the campaign_kl dump format.

  campaign_golden.py --model DIR --eval EVAL.json --out GOLDEN.npz --source REPO@REVISION [--top-k 1024]
                     [--chunk 2048] [--compute fp32|bf16]

EVAL.json is campaign_kl's eval set (rendered prompt ids and answer ids); the checkpoint must share that tokenizer
(the NVFP4 export's own BF16 source does). The model class is the checkpoint's ``architectures[0]`` from
transformers, built with its parameters on the meta device and then filled on the device from the checkpoint's own
tensors (load_bf16); weights stay BF16 in memory, computed constants (rotary frequencies, embedding scale) float32.
With ``--compute fp32`` (default) every decoder layer is cast to float32 for its own forward and restored afterwards
(the parameters and buffers return to their original dtypes, so no value changes), and the embeddings, final norm
and head stay float32; cuBLAS float32 matmuls run at "highest" precision (TF32 off).
Hidden states, attention, routing, the KV cache and the logits are therefore float32 throughout: the reference is the
BF16 weights' function, not BF16 activation rounding. One limit: on CUDA (sm80 and newer) SDPA's float32 attention,
when it selects PyTorch's memory-efficient kernel, multiplies with CUTLASS 3xTF32 (OpMultiplyAddFastF32: three TF32
products per float32 product, float32 accumulation, close to float32 accuracy); the TF32 switches do not govern that
kernel. Each prompt is prefilled in --chunk token chunks through the model's cache, then the answer positions are
scored in one forward; the record per position (top-K log-probabilities, log tail mass, forced token, entropy) comes
from campaign_kl.row_stats. Candidates then dump with ``campaign_kl.py dump --reference GOLDEN.npz`` and
``campaign_kl.py gate GOLDEN OLD NEW`` decides.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import time

if __package__:
    from .campaign_kl import file_sha256, row_stats, save_dump
else:
    from campaign_kl import file_sha256, row_stats, save_dump


@contextlib.contextmanager
def parameters_on_meta(torch):
    """Module construction with every parameter moved to the meta device as it is registered (buffers stay real and
    keep the construction dtype): the model's computed constants (rotary frequencies, embedding scales) are made in
    float32 while no parameter memory is held."""

    register = torch.nn.Module.register_parameter

    def on_meta(module, name, param):
        register(module, name, param)
        if param is not None and not param.is_meta:
            module._parameters[name] = torch.nn.Parameter(param.to("meta"), requires_grad=False)

    torch.nn.Module.register_parameter = on_meta
    try:
        yield
    finally:
        torch.nn.Module.register_parameter = register


def load_bf16(torch, transformers, cls, model_dir: Path, device: str):
    """The model built with parameters on the meta device (buffers computed in float32), then every parameter
    materialised on ``device`` straight from the checkpoint's safetensors in its stored dtype (BF16): no CPU copy of
    the whole model (on unified memory a CPU load plus a device move would hold the weights twice; transformers'
    device_map path needs accelerate, absent from the image). Every model parameter must come from the checkpoint
    or be tied to one that did, and every persistent buffer (Gemma4's per-layer ``layer_scalar``) must come from it;
    checkpoint keys without a model tensor fail."""

    from safetensors import safe_open
    from transformers.initialization import no_init_weights

    config = transformers.AutoConfig.from_pretrained(model_dir)
    with no_init_weights(), parameters_on_meta(torch):
        model = cls._from_config(config, dtype=torch.float32, attn_implementation="sdpa")
    model.eval()
    owners = {}
    for module_name, module in model.named_modules(remove_duplicate=False):
        for name in [*module._parameters, *module._buffers]:
            owners[f"{module_name}.{name}" if module_name else name] = (module, name)
    loaded, unexpected = set(), []
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(str(shard), framework="pt", device=str(device)) as f:
            for key in f.keys():
                if key not in owners:
                    unexpected.append(key)
                    continue
                module, name = owners[key]
                tensor = f.get_tensor(key)
                current = module._parameters.get(name, module._buffers.get(name))
                if tuple(tensor.shape) != tuple(current.shape):
                    raise ValueError(f"{key}: checkpoint shape {tuple(tensor.shape)} != model {tuple(current.shape)}")
                if name in module._parameters:
                    module._parameters[name] = torch.nn.Parameter(tensor, requires_grad=False)
                else:
                    module._buffers[name] = tensor
                loaded.add(key)
    if unexpected:
        raise ValueError(f"{len(unexpected)} checkpoint tensors have no model tensor, e.g. {unexpected[:5]}")
    model.tie_weights()
    missing = [name for name, param in model.named_parameters() if param.is_meta]
    missing += [key for key, (module, name) in owners.items() if name in module._buffers
                and name not in module._non_persistent_buffers_set and key not in loaded]
    if missing:
        raise ValueError(f"{len(missing)} model parameters or persistent buffers were not in the checkpoint, "
                         f"e.g. {missing[:5]}")
    for module in model.modules():                   # computed buffers (float32) join the weights on the device
        for name, buffer in module._buffers.items():
            if buffer is not None and buffer.device != torch.device(device):
                module._buffers[name] = buffer.to(device)
    return model, {"loaded_tensors": len(loaded),
                   "parameters": sum(1 for _ in model.parameters()),
                   "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()})}


def float32_layers(torch, model) -> dict:
    """Cast each text decoder layer to float32 around its forward (restoring every parameter and buffer to its own
    dtype afterwards) and keep the remaining text-path tensors float32; returns call counters for the record."""

    tower = ("vision", "audio", "visual", "multi_modal_projector", "embed_vision", "embed_audio")
    layers = [m for name, m in model.named_modules()
              if type(m).__name__.endswith("DecoderLayer") and not any(t in name for t in tower)]
    if not layers:
        raise ValueError(f"no text decoder layers found in {type(model).__name__}")
    counts = {"layers": len(layers), "calls": 0}
    inside = {id(t) for layer in layers for t in (*layer.parameters(), *layer.buffers())}

    def before(module, args):
        module._golden_dtypes = {}
        for tensor_name, tensor in [*module.named_parameters(), *module.named_buffers()]:
            if tensor.is_floating_point() and tensor.dtype != torch.float32:
                module._golden_dtypes[tensor_name] = tensor.dtype
                tensor.data = tensor.data.float()
        counts["calls"] += 1

    def after(module, args, output):
        for tensor_name, tensor in [*module.named_parameters(), *module.named_buffers()]:
            if tensor_name in module._golden_dtypes:
                tensor.data = tensor.data.to(module._golden_dtypes[tensor_name])
        module._golden_dtypes = {}
        return output

    for layer in layers:
        layer.register_forward_pre_hook(before)
        layer.register_forward_hook(after)
    for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
        if id(tensor) not in inside and tensor.is_floating_point() and not any(t in name for t in tower):
            tensor.data = tensor.data.float()
    return counts


def score(torch, model, prompt: list[int], answer: list[int], chunk: int, top_k: int) -> tuple[list[dict], str]:
    """Teacher-forced records for ``answer`` after ``prompt``: prefill all but the last prompt token in chunks, then one
    forward over the last prompt token and every answer token but the last (row j predicts answer[j])."""

    device = next(model.parameters()).device
    ids = list(prompt) + list(answer)
    prefix, tail = ids[:len(prompt) - 1], ids[len(prompt) - 1:len(ids) - 1]
    cache = None
    with torch.inference_mode():
        for start in range(0, len(prefix), chunk):
            piece = torch.tensor([prefix[start:start + chunk]], device=device)
            cache = model(input_ids=piece, past_key_values=cache, use_cache=True, logits_to_keep=1).past_key_values
        logits = model(input_ids=torch.tensor([tail], device=device), past_key_values=cache, use_cache=True,
                       logits_to_keep=0).logits[0]
        if logits.shape[0] != len(answer):
            raise RuntimeError(f"scored {logits.shape[0]} positions for {len(answer)} answer tokens")
        rows = [row_stats(torch, logits[j], top_k, answer[j]) for j in range(len(answer))]
    return rows, str(logits.dtype)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="pre-quantization BF16 checkpoint directory")
    ap.add_argument("--eval", required=True, help="campaign_kl eval set")
    ap.add_argument("--out", required=True)
    ap.add_argument("--source", required=True, help="REPO@REVISION of the checkpoint, recorded in the dump")
    ap.add_argument("--top-k", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=2048, help="prompt prefill chunk (tokens)")
    ap.add_argument("--compute", choices=("fp32", "bf16"), default="fp32")
    ap.add_argument("--device", default="cuda:0", help="torch device for the whole model (cpu for tool tests)")
    args = ap.parse_args()

    import torch
    import transformers

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model_dir = Path(args.model)
    config = json.loads((model_dir / "config.json").read_text())
    architecture = config["architectures"][0]
    started = time.perf_counter()
    model, load_report = load_bf16(torch, transformers, getattr(transformers, architecture), model_dir, args.device)
    loaded_s = time.perf_counter() - started
    counts = float32_layers(torch, model) if args.compute == "fp32" else None
    data = json.loads(Path(args.eval).read_text())
    rows, offsets, dtype = [], [], None
    cuda = torch.cuda.is_available() and args.device.startswith("cuda")
    for item in data["items"]:
        offsets.append(len(rows))
        item_rows, dtype = score(torch, model, item["prompt"], item["answer"], args.chunk, args.top_k)
        rows.extend(item_rows)
        if cuda:
            torch.cuda.empty_cache()
        print(json.dumps({"item": item["name"], "prompt": len(item["prompt"]), "positions": len(item_rows),
                          "elapsed_s": round(time.perf_counter() - started, 1),
                          "max_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2) if cuda else None}),
              flush=True)
    meta = {"eval_sha256": file_sha256(args.eval), "top_k": args.top_k, "logits_dtype": dtype,
            "names": [it["name"] for it in data["items"]], "reference": None, "golden": True,
            "source": args.source, "model_dir": str(model_dir), "architecture": architecture,
            "config_sha256": file_sha256(str(model_dir / "config.json")),
            "precision": {"weights": "bf16", "compute": args.compute, "tf32_matmul": False,
                          "attention": "sdpa (CUDA float32: memory-efficient kernel products in 3xTF32 when selected)",
                          "float32_layer_hooks": counts},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
            "chunk": args.chunk, "device": args.device, "load_s": round(loaded_s, 1), "load": load_report,
            "tool_sha256": file_sha256(__file__)}
    save_dump(args.out, rows, offsets, meta)
    print(json.dumps({"out": args.out, "positions": len(rows), "logits_dtype": dtype, "hooks": counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
