"""Startup-owned alias indexing against the unchanged public affine scanner."""
from __future__ import annotations

import ast
from pathlib import Path
import random
import unittest

from tensorfold import quantization as quant


ROOT = Path(__file__).resolve().parents[1]


def language_operation():
    path = ROOT / "src/tensorfold/families/qwen3_5/__init__.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == "_language_specs")
    node.body = [n for n in node.body if not isinstance(n, ast.ImportFrom)]
    scope = {name: getattr(quant, name) for name in ("_affine_resolver", "quantization_block", "resolve_affine")}
    exec(compile(ast.Module([node], []), str(path), "exec"), scope)
    return scope


def scanner(config):
    """Independent original complete operation, using the public scan per path."""
    result = {"": quant.resolve_affine(config)}
    for path, value in (quant.quantization_block(config) or {}).items():
        if (isinstance(value, dict) or type(value) is bool) and not set(path.split(".")) & {
            "vision_tower", "visual"
        } and not path.endswith("embed_tokens"):
            result[path] = quant.resolve_affine(config, path)
    return result


def outcome(operation, config):
    try:
        return "return", operation(config)
    except (ValueError, TypeError, AttributeError) as error:
        return "error", type(error).__name__, str(error)


class LanguageSpecsContract(unittest.TestCase):
    def test_complete_operation_metadata_and_exception_order(self):
        operation = language_operation()["_language_specs"]
        rng = random.Random(693781)
        values = [False, True, {}, None, 0, 1, "bad", {"bits": 4}, {"bits": 8, "group_size": 32},
                  {"group_size": 128}, {"bits": True}, {"bits": 0}, {"bits": 4, "mode": "mxfp4"}]
        paths = ["model.layers.0.a", "language_model.layers.0.a", "layers.1.b.weight",
                 "model.language_model.layers.0.a.weight", "text_model.layers.0.a", "vision_tower.proj",
                 "model.embed_tokens", "bits.weight", "model.quant_method"]
        for _ in range(4800):
            block = {"bits": rng.choice([2, 4, 8, None, True]), "group_size": rng.choice([32, 64, 128, 0]),
                     "quant_method": rng.choice([None, "mlx", "affine", "exl3"])}
            for _ in range(rng.randrange(8)):
                block[rng.choice(paths)] = rng.choice(values)
            config = {"quantization": block}
            if rng.randrange(2):
                config = {"text_config": config}
            self.assertEqual(outcome(operation, config), outcome(scanner, config), config)

    def test_no_alias_and_excluded_only_keep_original_path(self):
        scope = language_operation()

        def uncalled(config):
            raise AssertionError("excluded/no-alias configuration prepared a map")

        scope["_affine_resolver"] = uncalled
        for block in ({"bits": 4}, {"bits": 4, "visual.proj": False, "model.embed_tokens": True}):
            self.assertEqual(scope["_language_specs"]({"quantization": block}), scanner({"quantization": block}))

    def test_public_calls_and_next_operation_observe_mutation(self):
        operation = language_operation()["_language_specs"]
        config = {"quantization": {"bits": 4, "model.layers.0.a": {"bits": 8}}}
        self.assertEqual(operation(config)["model.layers.0.a"].bits, 8)
        config["quantization"]["model.layers.0.a"]["bits"] = 2
        self.assertEqual(quant.resolve_affine(config, "layers.0.a").bits, 2)
        self.assertEqual(operation(config)["model.layers.0.a"].bits, 2)

    def test_prepared_alias_values_are_owned(self):
        config = {"quantization": {"bits": 4, "layers.0.a": {"bits": 8}}}
        prepared = quant._affine_resolver(config)
        config["quantization"]["layers.0.a"]["bits"] = 2
        self.assertEqual(prepared("model.layers.0.a.weight").bits, 8)
        self.assertEqual(quant.resolve_affine(config, "model.layers.0.a.weight").bits, 2)

    def test_global_error_precedes_alias_errors(self):
        config = {"quantization": {"bits": 0, "layers.0.a": {"bits": True}}}
        self.assertEqual(outcome(language_operation()["_language_specs"], config), outcome(scanner, config))
        self.assertEqual(outcome(scanner, config)[2], "affine weights require 2, 3, 4, 5, 6 or 8 bits")


if __name__ == "__main__":
    unittest.main()
