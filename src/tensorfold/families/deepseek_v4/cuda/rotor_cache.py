"""DeepSeek's single K=V latent row in the registered SM70 RotorQuant basis."""

import torch

from tensorfold.cuda.sm70 import rotorquant_q4_native as native
from tensorfold.families.deepseek_v4 import ROTOR_KV_DTYPE
from tensorfold.families.qwen4_exp.kv_formats import get

DIM = 512
FORMAT = get(ROTOR_KV_DTYPE)


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
        self.status = torch.tensor([0, -2], dtype=torch.int32, device=device)

    def pack(self, values, valid, layer):
        rows = values.shape[0]
        # The stock encoder accepts a contiguous prefix. A fixed graph batch may have holes.
        source = torch.where(valid[:, None], values, 0).view(rows, 1, DIM)
        codes = torch.empty(rows, 1, DIM // 2, dtype=torch.uint8, device=values.device)
        scales = torch.empty(rows, 1, DIM // FORMAT.group, dtype=torch.float32, device=values.device)
        native.encode(source, codes, scales, codec=FORMAT.codec, status=self.status, layer=layer,
                      trap_on_error=True)
        return torch.cat((codes[:, 0], scales[:, 0].view(torch.uint8)), dim=1)

    def rotate(self, values, valid, layer, *, inverse=False):
        source = torch.where(valid[:, None, None], values, 0)
        output = torch.empty_like(source)
        native.rotate(source, output, codec=FORMAT.codec, inverse=inverse, status=self.status, layer=layer,
                      trap_on_error=True)
        return output
