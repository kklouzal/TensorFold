"""Deterministic, fully serialized tiny Flash Next EXL3 loader fixture.

Original native trellis streams are random valid packed weights, not a trained
checkpoint or an approximation of the resident oracle. Both loader paths read
exactly these safetensors bytes. All main/MTP components are present; no network,
CUDA, converter installation or dependency mutation is needed to create it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import torch

from tensorfold.cuda.exl3 import format as fmt

T = "model.language_model."
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
EXPERT_WIDTHS = ((3, 5, 7), (6, 4, 2), (4, 6, 4), (8, 8, 8))


def write_checkpoint(path: Path, *, seed=731, mtp=True):
    path.mkdir(parents=True, exist_ok=True)
    text = {"hidden_size": 256, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
            "vocab_size": 256, "rms_norm_eps": 1e-6, "num_attention_heads": 4, "num_key_value_heads": 2,
            "head_dim": 256, "max_position_embeddings": 512, "linear_num_key_heads": 8,
            "linear_num_value_heads": 24, "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4, "num_experts": 3, "num_experts_per_tok": 2,
            "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128, "hc_count": 4,
            "hc_lowrank": 128, "indexer_n_heads": 4, "indexer_head_dim": 128, "indexer_budget": 2048,
            "indexer_compress_ratio": 4, "ple_layer_ids": [], "eos_token_id": 255,
            "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                                "partial_rotary_factor": .25, "mrope_section": [11, 11, 10]}}
    config = {"model_type": "qwen4_exp", "architectures": ["Qwen4ExpForConditionalGeneration"],
              "text_config": text, "quantization_config": {"quant_method": "exl3", "version": "1.4.2",
                                                           "bits": 2.5, "codebook": "mul1"}}
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    gen = torch.Generator().manual_seed(seed)
    tensors = {}

    def plain(name, shape, scale=.02, *, ones=False):
        value = (torch.ones(shape) if ones else torch.randn(shape, generator=gen) * scale).half()
        tensors[name] = value.contiguous()

    def norm(name, size):
        # Pack norms are centred gamma-1; tiny variation exercises offset detection.
        plain(name, (size,), .015)

    def projection(name, k, n, k2=4):
        tensors[name + ".trellis"] = torch.randint(-32768, 32768, (k//16, n//16, k2*8),
                                                  generator=gen, dtype=torch.int32).short()
        # Both positive/negative, nonuniform original FP16 scale bit patterns.
        tensors[name + ".suh"] = ((torch.rand(k, generator=gen) - .5) * .2).half()
        tensors[name + ".svh"] = ((torch.rand(n, generator=gen) - .5) * .2).half()
        tensors[name + ".mul1"] = torch.tensor(fmt.MARKERS["mul1"] - 2**32, dtype=torch.int32)

    def hc(name, inject):
        plain(name + ".input_mix_weight_down.weight", (128, 1024))
        plain(name + ".input_mix_weight_up.weight", (1024, 128))
        if inject:
            plain(name + ".block_inject_weight.weight", (4, 1024))
        norm(name + ".hc_norm.weight", 1024)

    def experts(base):
        plain(base + ".gate.weight", (3, 256), .05)
        plain(base + ".shared_expert_gate.weight", (1, 256), .05)
        for index, widths in enumerate(EXPERT_WIDTHS):
            name = base + (f".experts.{index}" if index < 3 else ".shared_expert")
            for part, k, n, k2 in zip(PROJECTIONS, (256,256,128), (128,128,256), widths):
                projection(name + "." + part, k, n, k2)

    def attention(base):
        for part,n in (("q_proj",2048),("k_proj",512),("v_proj",512),("indexer.index_qk_proj",640)):
            projection(base + "." + part,256,n)
        projection(base + ".o_proj",1024,256)
        for part,size in (("q_norm",256),("k_norm",256),("indexer.q_layernorm",128),("indexer.k_layernorm",128)):
            norm(base + "." + part + ".weight",size)

    def layer(base, linear):
        hc(base + ".attn_hyper_connection",True)
        hc(base + ".mlp_hyper_connection",True)
        experts(base + ".mlp")
        if linear:
            name = base + ".linear_attn"
            projection(name + ".in_proj_qkv",256,5120)
            projection(name + ".in_proj_z",256,3072)
            plain(name + ".in_proj_b.weight",(24,256))
            plain(name + ".in_proj_a.weight",(24,256))
            plain(name + ".conv1d.weight",(5120,4),.1)
            plain(name + ".A_log",(24,),.2)
            plain(name + ".dt_bias",(24,),.2)
            plain(name + ".norm.weight",(128,),ones=True)
            projection(name + ".out_proj",3072,256)
        else:
            attention(base + ".self_attn")

    plain(T + "embed_tokens.weight",(256,256),.25)
    layer(T + "layers.0",True)
    layer(T + "layers.1",False)
    hc(T + "hyper_connection_mixer",False)
    projection("lm_head",256,256)
    if mtp:
        norm("mtp.pre_fc_norm_embedding.weight",256)
        norm("mtp.pre_fc_norm_hidden.weight",1024)
        projection("mtp.fc_embedding",256,256)
        projection("mtp.fc_hidden",256,256)
        layer("mtp.layers.0",False)
        hc("mtp.hyper_connection_mixer",False)
    dtype_names = {torch.float16:"F16",torch.int16:"I16",torch.int32:"I32"}
    header, chunks, offset = {},[],0
    for name,tensor in tensors.items():
        raw = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype":dtype_names[tensor.dtype],"shape":list(tensor.shape),
                        "data_offsets":[offset,offset+len(raw)]}
        chunks.append(raw)
        offset += len(raw)
    encoded = json.dumps(header,separators=(",",":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    payload = struct.pack("<Q",len(encoded)) + encoded + b"".join(chunks)
    (path / "model.safetensors").write_bytes(payload)
    (path / "model.safetensors.index.json").write_text(json.dumps({"metadata":{"total_size":offset},
        "weight_map":{name:"model.safetensors" for name in tensors}},indent=2)+"\n")
    return {"seed":seed,"tensor_count":len(tensors),"payload_bytes":offset,"file_bytes":len(payload),
            "safetensors_sha256":hashlib.sha256(payload).hexdigest(),"expert_widths_k2":EXPERT_WIDTHS,
            "scope":"valid synthetic original trellises; loader/inference parity, not trained model quality"}
