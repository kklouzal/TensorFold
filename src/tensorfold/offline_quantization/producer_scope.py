"""Bind a reviewed ModelOpt recipe to the explicit planned projection inventory.

This prepares configuration only. Real calibration must separately prove source
payloads, actual module coverage, retained Local-Hessian statistics, calibrated
activation globals, producer/provider identity and export compatibility.
"""

from __future__ import annotations

from copy import deepcopy
import re

from .plan import NVFP4_RECIPE


def selected_projections(plan: dict) -> tuple[str, ...]:
    """Extract a trusted fresh planner inventory, not authenticate saved JSON.

    The execution coordinator must obtain this dictionary directly from
    plan_nvfp4() on its verified immutable source. Neither schema nor recipe
    strings turn caller-edited inventories into source/exception-map authority.
    """
    if (
        plan.get("schema") != "tensorfold.offline-quantization-plan"
        or plan.get("schema_version") != 1
        or plan.get("recipe", {}).get("name") != NVFP4_RECIPE
    ):
        raise ValueError("a supported explicit offline quantization plan is required")
    recipe = plan["recipe"]
    if (
        recipe.get("numerics") != "W4A4"
        or recipe.get("block_size") != 16
        or recipe.get("weight_scale_algorithm") != "local_hessian"
        or recipe.get("fp8_scale_sweep") is not True
        or recipe.get("batch_amax_fallback") is not False
    ):
        raise ValueError("plan numerical/calibration contract differs from the supported recipe")
    result = []
    for item in plan["tensors"]:
        if item.get("action") != "quantize":
            continue
        name = item["name"]
        shape = item["shape"]
        if (
            not isinstance(name, str)
            or not name.endswith(".weight")
            or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", name)
            or any(not p for p in name.split("."))
            or item.get("source_dtype") != "BF16"
            or not isinstance(shape, list)
            or len(shape) != 2
            or any(type(value) is not int or value < 1 for value in shape)
            or shape[1] % 64
        ):
            raise ValueError("selected projection has an unsupported name, dtype or geometry")
        result.append(name[: -len(".weight")])
    if not result or len(result) != len(set(result)) or len(result) != plan["coverage"]["selected_projections"]:
        raise ValueError("selected projection inventory is empty, duplicated or inconsistent")
    return tuple(result)


def _numerics(preset: dict, selector: str) -> dict:
    entries = preset.get("quant_cfg")
    if not isinstance(entries, list):
        raise ValueError("pinned ModelOpt ordered quantizer configuration is required")
    matches = [entry for entry in entries if entry.get("quantizer_name") == selector]
    if not matches or not isinstance(matches[-1].get("cfg"), dict):
        raise ValueError("ModelOpt preset omits required NVFP4 quantizer attributes")
    cfg = deepcopy(matches[-1]["cfg"])
    sizes = cfg.get("block_sizes", {})
    if not isinstance(sizes, dict):
        raise ValueError("NVFP4 block attributes must be explicit")
    block = sizes.get(-1, sizes.get("-1"))
    if -1 in sizes and "-1" in sizes and sizes[-1] != sizes["-1"]:
        raise ValueError("conflicting quantizer block declarations")
    if (
        cfg.get("num_bits") not in ("e2m1", (2, 1), [2, 1])
        or block != 16
        or sizes.get("scale_bits") not in ("e4m3", (4, 3), [4, 3])
    ):
        raise ValueError("producer preset is not the planned NVFP4 code/scale format")
    expected_block_type = "static" if selector == "*weight_quantizer" else "dynamic"
    if (
        "dynamic" in cfg
        or cfg.get("type", "static") != "static"
        or cfg.get("axis") is not None
        or cfg.get("bias") is not None
        or cfg.get("scale_bits") is not None
        or cfg.get("fake_quant", True) is not True
        or cfg.get("unsigned", False) is not False
        or sizes.get("type", "static") != expected_block_type
        or set(sizes) - {-1, "-1", "type", "scale_bits"}
    ):
        raise ValueError("producer global/block/axis calibration mode differs from the plan")
    # Top-level static calibration and dynamic per16 activation blocks are
    # separate contracts. Never substitute a batch-dependent global range.
    cfg["type"] = "static"
    return cfg


def scoped_local_hessian_config(plan: dict, preset: dict) -> dict:
    """Use actual producer attributes, enabling ONLY exact planned projections.

    For direct ModelOpt quantize(), not AutoQuantize's different search scope.
    Requires fresh trusted planner output as documented by selected_projections;
    saved plans are reports and must not be passed here as execution authority.
    Do not inherit broad wildcard/default-exclusion behavior: native audio has
    reachable inner Linear modules and the assistant has a functional head that
    ordinary Linear hooks do not cover. The executor owns their explicit scope.
    """
    names = selected_projections(plan)
    algorithm = deepcopy(preset.get("algorithm", {}))
    if (
        not isinstance(algorithm, dict)
        or algorithm.get("method") != "local_hessian"
        or algorithm.get("fp8_scale_sweep") is not True
    ):
        raise ValueError("pinned Local-Hessian scale-sweep producer recipe required")
    # Retain documented diagnostic accumulators so execution can reject silent
    # fallback/unobserved modules before claiming successful calibration.
    algorithm["debug"] = True
    algorithm["hessian_input_dtype"] = "float32"
    weight = _numerics(preset, "*weight_quantizer")
    activation = _numerics(preset, "*input_quantizer")
    entries = [{"quantizer_name": "*", "enable": False}]
    for name in names:
        entries.append({"quantizer_name": name + ".weight_quantizer", "enable": True, "cfg": deepcopy(weight)})
        entries.append({"quantizer_name": name + ".input_quantizer", "enable": True, "cfg": deepcopy(activation)})
    return {"quant_cfg": entries, "algorithm": algorithm}
