# SPDX-License-Identifier: Apache-2.0
# Provenance: friend overlay patch 171 (SPDX Apache-2.0), snapshot lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
#   patches/ds41mu_files/local_ds41_mu_topk.py (sha256 aa262b57e2b9c08970b7bd2adf1bfd516a975cddb75c25b6e8ca7b255637d3cb).
# TensorFold changes: vllm.models.deepseek_v41.index_topk -> the vendored package module (module-level import);
#   production enables the flag (VLLM_LOCAL_DS41_MU_TOPK_SEG=1), so callers use this topk unconditionally.
# - Triton 3.2: no Triton code in this module.
"""LOCAL-PATCH 171 (ds41_mu_topk_seg) segmented decode top-k for long score rows.

Patch 87's decode selector runs index_topk (index_topk.py): ONE program per
token row makes three passes over the row's visible scores, so at 128K-480K
visible rows a decode step's top-k is a serial scan on a single SM per row
(and per Full index layer). The fork already ships index_topk_parallel (same
output: segments of 8192 scores histogrammed in parallel, the same
threshold/tie rule, ascending indices, -1 padded); its single-sequence decode
uses it above 16384 keys (attention_ops.select_packed_positions). With the
flag, patch 87's batched decode selector (and patch 170's Full/source layers
and candidate-block top-k) do the same: bit-identical indices, graph-safe
(fixed grid; segments past a row's length exit).

Flag: VLLM_LOCAL_DS41_MU_TOPK_SEG=1. Rows <= 16384 wide keep index_topk.
"""

from .index_topk import index_topk, index_topk_parallel

SEG_MIN = 16384


def topk(out, scores, positions, valid, ratio, k=512):
    if scores.shape[1] > SEG_MIN:
        return index_topk_parallel(out, scores, positions, valid, ratio, k)
    return index_topk(out, scores, positions, valid, ratio, k)
