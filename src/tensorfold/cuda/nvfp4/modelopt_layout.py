"""Preparation-only ModelOpt expert activation selection; no accelerator imports."""

from .format import LinearMetadata


def activation_banks(infos, text, *, native, mtp=True, checkpoint=None):
    from .host_layout import bank_bases

    return tuple(bank for bank in bank_bases(infos, text, mtp)
                 if a4_bank(infos, bank, text["num_experts"], text["_quantization"], native=native, checkpoint=checkpoint))


def a4_bank(infos, name, count, block, *, native, checkpoint=None):
    """Validate every projection and select a complete declared static-A4 bank.

    ``native`` is resolved once from explicit precision and the actual device.
    Weight-only declarations never become A4 merely because input_scale exists.
    """
    metadata = LinearMetadata(block)
    if metadata.reciprocal:
        raise ValueError("Flash Next ModelOpt banks require direct calibration scales")
    modes = set()
    for expert in range(count):
        for projection in ("gate_proj", "up_proj", "down_proj"):
            base = f"{name}.experts.{expert}.{projection}"
            weight = infos.get(base + ".weight")
            if weight is None:
                return False  # stacked higher-precision MTP is a distinct representation
            if weight["dtype"] != "U8":
                return False  # the explicit higher-precision MTP conversion is not native A4
            parts = {field: (info["dtype"], list(info["shape"])) for field in
                     ("weight", "weight_scale", "weight_scale_2", "input_scale")
                     if (info := infos.get(base + "." + field)) is not None}
            kind, bits = metadata.validate(base, parts, checkpoint=native if checkpoint is None else checkpoint)
            if kind != "nvfp4":
                return False  # BF16/FP8 MTP keeps its original draft-only conversion
            if native and bits == 4 and (weight["shape"][0] % 64 or weight["shape"][1] % 32):
                raise ValueError(f"{base}: native A4 requires complete64-wide projection tiles")
            modes.add(bits)
    if native and len(modes) != 1:
        raise ValueError(f"{name}: mixed routed activation declarations require --precision full")
    return native and modes == {4}


def a4_read_plan(names, banks, count):
    """Partition only the exact original fields of prepared native-A4 banks.

    Matrix fields are read directly through get_rows on the packing device;
    queuing them would duplicate I/O/uploads. Original scalar calibrations are
    consumed through get on CPU. Unknown/non-A4/shared/MTP names keep their
    default CUDA prefetch. Preserve order and never infer format from a suffix.
    """
    if type(count) is not int or count <= 0:
        raise ValueError("native A4 read plan requires a positive expert count")
    device, host = [], []
    digits = len(str(count - 1))
    for name in names:
        bank, separator, tail = name.partition(".experts.")
        expert, dot, projection_field = tail.partition(".")
        matched = (separator and bank in banks and dot and expert.isascii() and expert.isdecimal()
                   and len(expert) <= digits and (expert == "0" or not expert.startswith("0"))
                   and int(expert) < count)
        if matched and projection_field in ("gate_proj.weight", "gate_proj.weight_scale", "up_proj.weight",
                                            "up_proj.weight_scale", "down_proj.weight", "down_proj.weight_scale"):
            continue
        if matched and projection_field in ("gate_proj.input_scale", "gate_proj.weight_scale_2", "up_proj.input_scale",
                                            "up_proj.weight_scale_2", "down_proj.input_scale", "down_proj.weight_scale_2"):
            host.append(name)
        else:
            device.append(name)
    return device, host


def scalar_prefetch_host_bytes(count, *, workers, alignment):
    """Extra logical CPU scalar payload/backing, separate from final arrays.

    At most two layers own6*count F32 scalars each. Their cloned results use
    48*count bytes; simultaneous contiguous raw spans use at most that payload
    plus3 alignment pages per active Reader worker. Tensor/Future metadata and
    allocator backing remain runtime effects requiring integrated peak checks.
    """
    if any(type(n) is not int or n <= 0 for n in (count, workers, alignment)):
        raise ValueError("scalar prefetch count/workers/alignment must be positive integers")
    return 96 * count + 3 * alignment * workers


def plan_workspace_bytes(rows, slots, width, dims):
    """Device peak for one resident-plan buffer, including both SiLU temporaries.

    Persistent: two input code/scale planes, two BF16 gate/up planes and one
    down code/scale plane. SiLU and its BF16 product add four bytes per width.
    """
    if any(type(n) is not int or n <= 0 for n in (rows, slots, width, dims)):
        raise ValueError("native A4 workspace dimensions must be positive integers")
    if width % 64 or dims % 64:
        raise ValueError("native A4 workspace requires complete64-wide projection tiles")
    return rows * slots * (9 * dims // 8 + 137 * width // 16)


def runtime_workspace_bytes(text, *, rows, prompt_rows, mtp_a4=False, cached=False):
    """Additional main buffer + mixed-prompt buffer, or bounded host-cache window.

    Serial twins and concurrent slot engines share these same buffers. MTP
    contributes its separate row buffer only when its own declarations use A4.
    Cached execution processes at most128 pairs per lease; two successive
    windows can overlap their Python-held allocations during reassignment.
    """
    width, dims = text["moe_intermediate_size"], text["hidden_size"]
    if cached:
        return 2 * plan_workspace_bytes(128, 1, width, dims)
    slots = text["num_experts_per_tok"] + 1
    # A reusable [rows,3] row plan has12bytes/row retained; its one-time
    # arange/zero construction overlaps another8bytes/row before publication.
    total_rows = rows + prompt_rows + (rows if mtp_a4 else 0)
    return plan_workspace_bytes(total_rows, slots, width, dims) + 20 * total_rows


def resident_loading_host_bytes(text):
    """Four ordinary CPU calibration arrays; resident payload packing stays GPU.

    Original read staging is charged by ordinary weight loading. No complete
    CPU packed authority or D2H payload is retained for resident A4 banks.
    """
    return 24 * text["num_experts"]


def loading_device_bytes(text):
    """GPU raw/permuted/joined peak for one complete projection's byte packing."""
    return 3 * (9 * text["moe_intermediate_size"] * text["hidden_size"] // 16)


def cached_control_host_bytes(text):
    """Bounded CPU readback/grouping/lease plan for128 rows of routed picks."""
    return 192 * 128 * (text["num_experts_per_tok"] + 1) + 64 * 1024


def expert_group_policy(policies):
    """One preparation-time arithmetic policy for all expert ranks.

    A heterogeneous group retains the supported W4A16 region uniformly; its
    expert assignment must not select activation precision incidentally.
    """
    if (not isinstance(policies, list) or not policies or any(not isinstance(p, list) or len(p) != 2
            or p[0] not in ("checkpoint", "full") or type(p[1]) is not bool for p in policies)):
        raise ValueError("expert rank math capabilities require a mode and native-FP4 boolean")
    modes = {p[0] for p in policies}
    if len(modes) != 1:
        raise ValueError("expert ranks require matching explicit precision modes")
    native = policies[0][0] == "checkpoint" and all(p[1] for p in policies)
    reason = ("all-native" if native else "explicit-full" if policies[0][0] == "full" else
              "mixed-capability-full" if any(p[1] for p in policies) else "non-native-full")
    return native, reason
