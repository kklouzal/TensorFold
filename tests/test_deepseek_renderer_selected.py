"""Original-source prompt oracle, exact selected dispatch and input ownership."""
from __future__ import annotations

import ast
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'triton', 'mlx', 'mlx_lm', 'transformers', 'tokenizers'}:
            raise AssertionError('source control must not import SDK')
        return None


@contextmanager
def source_imports():
    """Reject SDK discovery only while this source control owns its imports."""
    with patch.object(sys, 'meta_path', [NoSDK(), *sys.meta_path]), \
            patch.object(sys, 'path', [str(ROOT / 'src'), *sys.path]):
        yield


with source_imports():
    from tensorfold.families.deepseek_v4 import prompts as current
    from tensorfold.families.deepseek_v4 import rendering
    from tensorfold.families.deepseek_v4.vendor import encoding_dsv4 as vendor
    spec = importlib.util.spec_from_file_location('owned_original_prompt_oracle', ROOT / 'tests/fixtures/deepseek_v4/original_prompts.py')
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)


def observations(function, messages, options):
    data, keywords = deepcopy(messages), deepcopy(options)
    try:
        value = ('result', function(data, **keywords))
    except Exception as error:
        value = ('error', type(error), error.args)
    return value, data, keywords


class Selected(unittest.TestCase):
    def setUp(self):
        self.enterContext(source_imports())

    def test_source_import_scope_restores_prior_import_lists_on_failure(self):
        before_meta, before_path = sys.meta_path, sys.path
        with self.assertRaisesRegex(AssertionError, 'source control must not import SDK'):
            with source_imports():
                self.assertIsInstance(sys.meta_path[0], NoSDK)
                sys.meta_path[0].find_spec('torch')
        self.assertIs(sys.meta_path, before_meta)
        self.assertIs(sys.path, before_path)

    def test_exact_original_source_golden_and_current_two_function_math(self):
        self.assertEqual(hashlib.sha256((ROOT / 'tests/fixtures/deepseek_v4/original_prompts.py').read_bytes()).hexdigest(),
                         '430acf50757a0bae1e8e5137326729cc3eae123d72835c2a4567a84af4b513de')
        self.assertEqual(hashlib.sha256((ROOT / 'src/tensorfold/families/deepseek_v4/vendor/encoding_dsv4.py').read_bytes()).hexdigest(),
                         'bdbd57c132a1b3725042323d02b98b9d1df28e5f388f134399555d041f5055e0')
        for at in (1, 2):
            folder = ROOT / 'tests/fixtures/deepseek_v4'
            data = json.loads((folder / f'test_input_{at}.json').read_bytes())
            messages = data['messages'] if at == 1 else data
            options = {'tools': data['tools'], 'thinking': True} if at == 1 else {'thinking': True}
            self.assertEqual(current.render(messages, **options), (folder / f'test_output_{at}.txt').read_text())
        base = ast.parse((ROOT / 'tests/fixtures/deepseek_v4/original_prompts.py').read_bytes())
        now = ast.parse((ROOT / 'src/tensorfold/families/deepseek_v4/prompts.py').read_bytes())
        now.body = [node for node in now.body if not (isinstance(node, ast.ImportFrom) and node.module == 'rendering')]
        body = next(node for node in now.body if isinstance(node, ast.FunctionDef) and node.name == 'render')
        body.body = [node for node in body.body if not (isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == 'encoder'
                     or isinstance(node, ast.If) and ast.unparse(node.test).startswith('len(msgs) >= 64'))]
        body.body[-1].value.func = ast.Name(id='encode_messages', ctx=ast.Load())
        self.assertEqual(ast.dump(now), ast.dump(base))

    def test_only_measured_suffix_class_uses_owned_path_original_direct_elsewhere(self):
        for count, role, expected in [(1, 'assistant', False), (8, 'assistant', False),
                                      (16, 'assistant', False), (63, 'assistant', False),
                                      (64, 'assistant', True), (256, 'assistant', True),
                                      (1024, 'assistant', True), (64, 'latest_reminder', True),
                                      (64, 'tool', False), (64, 'unknown', False)]:
            messages = [{'role': 'user', 'content': 'q'}] + [{'role': role, 'content': 'a'} for _ in range(count - 1)]
            self.assertEqual(rendering._eligible(messages, len(messages)), expected)
        for count in (1, 8, 16, 64, 256):
            messages = [{'role': 'user' if index % 2 == 0 else 'assistant', 'content': 'x'} for index in range(count)]
            with patch.object(current, 'encode_messages', wraps=vendor.encode_messages) as stock, patch.object(current, 'encode_long_messages', side_effect=AssertionError('uncertain/loser branch selected')):
                self.assertEqual(current.render(messages), original.render(messages))
                self.assertEqual(stock.call_count, 1)
        messages = [{'role': 'user', 'content': 'q'}] + [{'role': 'assistant', 'content': 'a'} for _ in range(63)]
        with patch.object(current, 'encode_messages', side_effect=AssertionError('winner used original')), patch.object(current, 'encode_long_messages', wraps=rendering.encode_long_messages) as selected:
            self.assertEqual(current.render(messages), original.render(messages))
            self.assertEqual(selected.call_count, 1)

    def test_seeded_original_byte_error_and_mutation_oracle(self):
        rng = random.Random(20261009)
        for _ in range(800):
            messages = [{'role': rng.choice(('user', 'assistant', 'developer', 'system', 'latest_reminder', 'unknown')),
                         'content': rng.choice(('你好', 'x', '', None)), 'reasoning_content': 'think'}
                        for _ in range(rng.randrange(0, 140))]
            options = {'thinking': rng.choice((True, False)), 'reasoning_effort': rng.choice((None, 'high', 'max'))}
            self.assertEqual(observations(current.render, messages, options), observations(original.render, messages, options))
        for count in (64, 65, 128, 256, 1024):
            for thinking in (False, True):
                for role in ('assistant', 'latest_reminder'):
                    messages = [{'role': 'developer', 'content': 'requirements'}] + [
                        {'role': role, 'content': 'x', 'reasoning_content': 'think'} for _ in range(count - 1)]
                    self.assertEqual(observations(current.render, messages, {'thinking': thinking}),
                                     observations(original.render, messages, {'thinking': thinking}))

    def test_custom_role_callbacks_and_oversized_graph_keep_original(self):
        calls = []
        class Role(str):
            def __eq__(self, other):
                calls.append(other)
                return super().__eq__(other)
        messages = [{'role': Role('user'), 'content': 'q'}] + [{'role': 'assistant', 'content': 'x'} for _ in range(64)]
        results = []
        for function in (original.render, current.render):
            calls.clear()
            results.append((function(deepcopy(messages)), list(calls)))
        self.assertEqual(results[0], results[1])
        messages = [{'role': 'user', 'content': 'q'}] + [{'role': 'assistant', 'content': 'x', 'unused': ['x'] * 20000} for _ in range(63)]
        self.assertFalse(rendering._eligible(messages, len(messages)))

    def test_full_wrapper_modes_and_mit_provenance(self):
        class Records:
            def encode(self, text, *, add_special_tokens):
                self.assertion = add_special_tokens is False
                return list(text.encode('utf-8'))
        messages = [{'role': 'user', 'content': 'q'}] + [{'role': 'assistant', 'content': 'a'} for _ in range(63)]
        for thinking in (False, True):
            for generation in (False, True):
                options = {'enable_thinking': thinking, 'add_generation_prompt': generation}
                self.assertEqual(current.DeepSeekTokenizer(Records()).apply_chat_template(messages, **options),
                                 original.DeepSeekTokenizer(Records()).apply_chat_template(messages, **options))
        self.assertIn('Copyright (c) 2023 DeepSeek.', (ROOT / 'LICENSES/DeepSeek-MIT.txt').read_text())
        self.assertIn('Permission is hereby granted', (ROOT / 'LICENSES/DeepSeek-MIT.txt').read_text())


if __name__ == '__main__':
    unittest.main()
