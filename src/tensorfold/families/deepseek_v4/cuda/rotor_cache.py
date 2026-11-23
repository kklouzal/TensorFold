"""DeepSeek's single K=V latent row in the registered SM70 RotorQuant basis."""

import torch

from tensorfold.cuda.sm70 import rotorquant_q4_native as native
from tensorfold.families.deepseek_v4 import ROTOR_KV_DTYPE
from tensorfold.families.qwen4_exp.cuda.rotorquant_ref import CENTROIDS_4
from tensorfold.families.qwen4_exp.kv_formats import get

DIM = 512
FORMAT = get(ROTOR_KV_DTYPE)
ROW_BYTES = FORMAT.value_bytes(DIM)


def live_rows(step, scratch):
    # The decode graph has one request per token row. Its padding uses a
    # scratch slot even though the graph's device validity words are all true.
    if step.single_token_slots:
        slots = step.req_slot[:step.valid.shape[0]]
        return step.valid & (slots >= 0) & (slots < scratch)
    return step.valid


class RotorCache:
    def __init__(self, device):
        native.extension()  # Build/import before graph capture.
        # The fallback reader borrows this immutable table for the cache and its graph lifetime.
        self.centroids = torch.tensor(CENTROIDS_4, dtype=torch.float32, device=device)
        self.status = torch.tensor([0, -2], dtype=torch.int32, device=device)

    def pack(self, values, valid, layer):
        packed = torch.empty((values.shape[0], ROW_BYTES), dtype=torch.uint8, device=values.device)
        return native.pack_h64(values, valid, packed, status=self.status, layer=layer)

    def rotate(self, values, valid, layer, *, inverse=False):
        source = torch.where(valid[:, None, None], values, 0)
        output = torch.empty_like(source)
        native.rotate(source, output, codec=FORMAT.codec, inverse=inverse, status=self.status, layer=layer,
                      trap_on_error=True)
        return output
