"""Pure checkpoint quantization schema reconciliation; no filesystem or accelerator imports.

ModelOpt sidecars use the export schema in Model-Optimizer 0.46.0's
``convert_hf_config.py``. Normalization never chooses a runtime precision.
"""

from __future__ import annotations

from copy import deepcopy


METHODS = ("modelopt", "compressed-tensors")


def _object(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object")
    return value


def _strings(value, where):
    if not isinstance(value, list) or any(not isinstance(v, str) or not v for v in value):
        raise ValueError(f"{where}: expected a list of nonempty strings")
    return value


def _merge(left, right, where):
    result = deepcopy(left)
    for key, value in right.items():
        if key not in result:
            result[key] = deepcopy(value)
        elif isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge(result[key], value, f"{where}.{key}")
        elif key == "ignore" and set(_strings(result[key], where)) == set(_strings(value, where)):
            pass
        elif result[key] != value or type(result[key]) is not type(value):
            raise ValueError(f"{where}.{key}: conflicting checkpoint quantization declarations")
    return result


def _algorithm(name, group_size, where):
    """Recognized ModelOpt algorithms; unsupported arithmetic remains explicit."""
    if name in ("NVFP4", "W4A16_NVFP4"):
        gs = 16 if group_size is None else group_size
        if type(gs) is not int or gs <= 0:
            raise ValueError(f"{where}: group_size must be a positive integer")
        weight = {"dynamic": False, "num_bits": 4, "type": "float", "group_size": gs}
        activation = deepcopy(weight) if name == "NVFP4" else None
    elif name == "FP8":
        weight = {"dynamic": False, "num_bits": 8, "type": "float"}
        activation = deepcopy(weight)
    elif name == "MXFP8":
        gs = 32 if group_size is None else group_size
        weight = {"dynamic": False, "num_bits": 8, "type": "float", "group_size": gs}
        activation = deepcopy(weight)
    elif name == "FP8_PB_WO":
        weight = {"dynamic": False, "num_bits": 8, "type": "float", "block_structure": [128, 128]}
        activation = None
    elif name in ("W4A8_NVFP4_FP8", "FP8_PER_CHANNEL_PER_TOKEN"):
        weight = {"dynamic": False, "num_bits": 4 if name.startswith("W4") else 8, "type": "float"}
        if name.startswith("W4"):
            weight["group_size"] = 16 if group_size is None else group_size
        else:
            weight["strategy"] = "channel"
        activation = {"dynamic": False, "num_bits": 8, "type": "float"}
    else:
        raise ValueError(f"{where}: unsupported ModelOpt quant_algo {name!r}")
    return {"weights": weight, "input_activations": activation}


def _scheme(group, where):
    result = deepcopy(_object(group, where))
    for key in ("weights", "input_activations", "output_activations"):
        spec = result.get(key)
        if spec is None:
            continue
        _object(spec, f"{where}.{key}")
        if "num_bits" in spec and (type(spec["num_bits"]) is not int or spec["num_bits"] <= 0):
            raise ValueError(f"{where}.{key}: num_bits must be a positive integer")
        if "dynamic" in spec and not (type(spec["dynamic"]) is bool or spec["dynamic"] == "local"):
            raise ValueError(f"{where}.{key}: dynamic must be a bool or the local quantization mode")
        if "group_size" in spec and (type(spec["group_size"]) is not int or spec["group_size"] < -1
                                    or spec["group_size"] == 0):
            raise ValueError(f"{where}.{key}: invalid group_size")
    if "targets" in result:
        _strings(result["targets"], f"{where}.targets")
    return result


def _block(value, where):
    result = deepcopy(_object(value, where))
    if "quant_method" in result:
        if not isinstance(result["quant_method"], str) or not result["quant_method"]:
            raise ValueError(f"{where}: quant_method must be a nonempty string")
        result["quant_method"] = result["quant_method"].lower()
    if "quant_method" not in result and "bits" in result:
        result["quant_method"] = "mlx"
    method = result.get("quant_method")
    if method not in METHODS:
        return result
    if "exclude_modules" in result:
        excluded = _strings(result.pop("exclude_modules"), f"{where}.exclude_modules")
        result = _merge(result, {"ignore": excluded}, where)
    if "ignore" in result:
        _strings(result["ignore"], f"{where}.ignore")
    groups = result.get("config_groups")
    if groups is not None:
        result["config_groups"] = {name: _scheme(group, f"{where}.config_groups.{name}")
                                   for name, group in _object(groups, f"{where}.config_groups").items()}
    algo = result.get("quant_algo")
    if method == "modelopt" and algo is not None:
        if not isinstance(algo, str) or not algo:
            raise ValueError(f"{where}: quant_algo must be a nonempty string")
        if algo == "MIXED_PRECISION":
            layers = _object(result.get("quantized_layers"), f"{where}.quantized_layers")
            generated = {}
            for i, (name, layer) in enumerate(layers.items()):
                if not isinstance(name, str) or not name:
                    raise ValueError(f"{where}: quantized layer names must be nonempty strings")
                _object(layer, f"{where}.{name}")
                generated[f"layer_{i}"] = dict(_algorithm(layer.get("quant_algo"), layer.get("group_size"),
                                                         f"{where}.{name}"), targets=[name])
        else:
            generated = {"group_0": dict(_algorithm(algo, result.get("group_size"), where), targets=["Linear"])}
        if not groups:
            result["config_groups"] = generated
        elif algo != "MIXED_PRECISION":
            # ModelOpt's algorithm is a declaration, so partial group records
            # can complement it. An explicit null is retained and conflicts
            # with an algorithm that declares activation quantization.
            expected = _algorithm(algo, result.get("group_size"), where)
            result["config_groups"] = {name: _merge(expected, group, f"{where}.{name}")
                                       for name, group in result["config_groups"].items()}
        # Mixed groups retain their own declarations. The prepared resolver
        # cross-checks their per-layer algorithm records.
    return result


def _sidecar(sidecar, where):
    sidecar = _object(sidecar, where)
    producer = _object(sidecar.get("producer"), f"{where}.producer")
    if str(producer.get("name", "")).lower() != "modelopt":
        raise ValueError(f"{where}: hf_quant_config.json requires a ModelOpt producer")
    quant = deepcopy(_object(sidecar.get("quantization"), f"{where}.quantization"))
    quant["quant_method"] = "modelopt"
    quant["producer"] = deepcopy(producer)
    if "kv_cache_quant_algo" in quant:
        kv = quant.pop("kv_cache_quant_algo")
        quant["kv_cache_scheme"] = ({"dynamic": False, "num_bits": 8, "type": "float"}
                                     if kv == "FP8" else kv)
    return _block(quant, where)


def effective_config(raw: dict, sidecar: dict | None = None, *, where: str) -> dict:
    """Return independent canonical metadata, refusing conflicting declarations.

    Quantization aliases at the top and text levels are combined into one
    top-level ``quantization_config``. Other fields retain their exact values.
    An absent sidecar differs from a malformed/present sidecar.
    """
    result = deepcopy(_object(raw, where))
    sources = [result]
    if result.get("text_config") is not None:
        sources.append(_object(result["text_config"], f"{where}.text_config"))
    blocks = []
    for source in sources:
        for key in ("quantization", "quantization_config"):
            if key in source:
                value = source.pop(key)
                if value is not None:
                    blocks.append(_block(value, f"{where}.{key}"))
    if sidecar is not None:
        blocks.append(_sidecar(sidecar, f"{where}/hf_quant_config.json"))
    if blocks:
        block = {}
        for value in blocks:
            block = _merge(block, value, where)
        block = _block(block, where)
        if block.get("quant_method") in METHODS:
            LinearQuantization(block)
        result["quantization_config"] = block
    return result


class _Pattern:
    """Nonrecursive glob matching, O(pattern length × name length) worst case.

    The supported ``re:`` subset is literals/escaped punctuation, . and .*,
    with optional ^/$ anchors. General backtracking regex is not executed on
    checkpoint-controlled expressions. Unsupported operators fail at prepare.
    """

    def __init__(self, pattern):
        regex = pattern.startswith("re:")
        text = pattern[3:] if regex else pattern
        if regex and text.startswith("^"):
            text = text[1:]
        self.tokens = []
        i, end = 0, False
        while i < len(text):
            char = text[i]
            if regex and char == "\\":
                i += 1
                if i == len(text) or text[i].isalnum():
                    raise ValueError("quantization pattern: only escaped punctuation is supported")
                self.tokens.append(("literal", text[i]))
            elif regex and char == "$" and i == len(text) - 1:
                end = True
            elif regex and char == ".":
                if i + 1 < len(text) and text[i + 1] == "*":
                    self.tokens.append(("regex_star", "")); i += 1
                else:
                    self.tokens.append(("regex_any", ""))
            elif (regex and char in "*+?{}()[]|^$") or (not regex and char in "[]"):
                raise ValueError("quantization pattern: unsupported regex/glob operator; use literals, . or .* for re:")
            elif not regex and char in "*?":
                self.tokens.append(("star" if char == "*" else "any", ""))
            else:
                self.tokens.append(("literal", char))
            i += 1
        self.regex_end = regex and end
        if regex and not end:
            self.tokens.append(("star", ""))  # re.match accepts an unmatched suffix.

    def match(self, value):
        return (self._match(value) or self.regex_end and value.endswith("\n") and self._match(value[:-1]))

    def _match(self, value):
        i = j = 0
        star, restart, regex_star = -1, 0, False
        while i < len(value):
            if j < len(self.tokens) and (self.tokens[j][0] == "any"
                                           or self.tokens[j][0] == "regex_any" and value[i] != "\n"
                                           or self.tokens[j] == ("literal", value[i])):
                i += 1; j += 1
            elif j < len(self.tokens) and self.tokens[j][0] in ("star", "regex_star"):
                regex_star = self.tokens[j][0] == "regex_star"
                j += 1; star, restart = j, i
            elif star >= 0:
                if regex_star and value[restart] == "\n":
                    return False
                restart += 1; i, j = restart, star
            else:
                return False
        while j < len(self.tokens) and self.tokens[j][0] in ("star", "regex_star"):
            j += 1
        return j == len(self.tokens)


class LinearQuantization:
    """Preparation-owned module policy; exact targets are indexed, patterns compiled once.

    ModelOpt's per-layer expert-bank targets also cover their descendants.
    Names are never stripped or aliased across text/vision namespaces.
    """

    def __init__(self, block: dict):
        self.method = block.get("quant_method")
        self.exact = {}
        self.patterns = []
        self.defaults = []
        self.ignored = []
        self.ignored_exact = set()
        for pattern in block.get("ignore", ()):
            _strings([pattern], "quantization.ignore")
            if pattern.startswith("re:") or any(c in pattern for c in "*?["):
                self.ignored.append(_Pattern(pattern))
            else:
                self.ignored_exact.add(pattern)
        for name, value in (block.get("config_groups") or {}).items():
            group = _scheme(value, f"quantization.config_groups.{name}")
            for target in group.get("targets", [] if block.get("quant_algo") == "MIXED_PRECISION" else ["Linear"]):
                if target == "Linear":
                    self.defaults.append(group)
                elif target.startswith("re:"):
                    self.patterns.append((_Pattern(target), group))
                else:
                    self.exact.setdefault(target, []).append(group)
        algo = block.get("quant_algo")
        if self.method == "modelopt" and algo:
            declarations = block.get("quantized_layers", {}) if algo == "MIXED_PRECISION" else {
                "Linear": {"quant_algo": algo, "group_size": block.get("group_size")}}
            for name, declaration in declarations.items():
                expected = _algorithm(declaration.get("quant_algo"), declaration.get("group_size"), name)
                if algo == "MIXED_PRECISION":
                    # The per-layer map assigns modules. Untargeted groups
                    # describe a format inventory, not a global Linear rule.
                    self.exact[name] = [_merge(expected, group, name)
                                        for group in self.exact.get(name, ())] or [expected]
                if name == "Linear":
                    candidates = self.defaults + [g for groups in self.exact.values() for g in groups]
                    candidates += [g for _, g in self.patterns]
                else:
                    selected = self.resolve(name)
                    candidates = [] if selected is None else [selected]
                for group in candidates:
                    if not self._agrees(group, expected):
                        raise ValueError(f"{name}: quant_algo conflicts with config_groups weights/activation metadata")

    @staticmethod
    def _agrees(group, expected):
        for field in ("weights", "input_activations"):
            got, want = group.get(field), expected.get(field)
            if (got is None) != (want is None):
                return False
            if got is not None and any(key in got and got[key] != value for key, value in want.items()):
                return False
        return True

    def excluded(self, name: str) -> bool:
        return name in self.ignored_exact or any(pattern.match(name) for pattern in self.ignored)

    def resolve(self, name: str) -> dict | None:
        if self.excluded(name):
            return None
        groups = list(self.exact.get(name, ()))
        if self.method == "modelopt":
            parent = name
            while "." in parent:
                parent = parent.rsplit(".", 1)[0]
                groups.extend(self.exact.get(parent, ()))
        groups += [group for pattern, group in self.patterns if pattern.match(name)]
        if not groups:
            groups = self.defaults
        if not groups:
            return None
        selected = groups[0]
        for group in groups[1:]:
            for key in ("weights", "input_activations", "output_activations", "format"):
                left, right = selected.get(key), group.get(key)
                if isinstance(left, dict) and isinstance(right, dict):
                    _merge(left, right, f"{name}.{key}")
                elif left != right:
                    raise ValueError(f"{name}: conflicting matching quantization groups")
        return selected
