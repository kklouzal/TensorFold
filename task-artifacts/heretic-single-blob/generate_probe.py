"""Derive a labelled private blob probe from the reviewed frozen HTTP helper."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

EXPECTED = "d75271f476fb9293625ca391d5a3de65dd7b7d5198f5f256b4752126ba3f4e93"


def generate(source, target, proof):
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED:
        raise ValueError("frozen reviewed HTTP helper identity changed")
    original = raw.decode()
    value = original.replace("from types import MethodType\n", "")
    before = "choices=('baseline', 'grouped'), default='baseline'"
    assert value.count(before) == 1
    value = value.replace(before, "choices=('single-blob',), default='single-blob'")
    node = next(n for n in ast.parse(value).body if isinstance(n, ast.FunctionDef) and n.name == "kernel_factory_scope")
    old_scope = ast.get_source_segment(value, node)
    new_scope = '''def kernel_factory_scope(choice):
    """The private engine binds every blob bank before constructor calibration."""
    if choice != 'single-blob':
        raise ValueError('This private helper only measures the explicit single-blob candidate')
    from tensorfold.families.qwen3_5_moe.cuda.nvfp4_engine import QwenNvfp4MoeEngine
    if getattr(QwenNvfp4MoeEngine, '_experimental_cache_layout', None) != 'single-blob-grouped':
        raise RuntimeError('Private single-blob engine source was not loaded')
    yield []'''
    value = value.replace(old_scope, new_scope)
    old_validation = '''            if args.kernel == 'grouped':
                targets = [layer.moe.experts for layer in engine.w.layers]
                if len(targets) != len(bound_banks) or any(left is not right for left, right in zip(targets, bound_banks)):
                    raise RuntimeError('Diagnostic loader binding did not cover exactly the target expert banks')'''
    new_validation = '''            from tensorfold.cuda.nvfp4.blob_cache_a4 import BlobCachedExpertsA4, BlobExpertCache
            from tensorfold.cuda.nvfp4.blob_grouped_a4 import run as blob_run
            targets = [layer.moe.experts for layer in engine.w.layers]
            if (not isinstance(engine.expert_cache, BlobExpertCache)
                    or engine.expert_layout.get('experimental_variant') != 'single-blob-grouped'
                    or not targets or any(not isinstance(bank, BlobCachedExpertsA4)
                    or bank.cache is not engine.expert_cache
                    or getattr(bank.run, '__func__', None) is not blob_run for bank in targets)):
                raise RuntimeError('Private single-blob source/default did not cover every target bank')
            bound_banks.extend(targets)'''
    assert value.count(old_validation) == 1
    value = value.replace(old_validation, new_validation)
    old_note = "'grouped is a diagnostic per-instance target-bank binding, unselected for production'"
    new_note = "'PRIVATE single-blob-grouped engine default, one H2D per miss; unselected for production'"
    assert value.count(old_note) == 1
    value = value.replace(old_note, new_note)
    compile(value, str(target), 'exec')
    old_functions = {n.name: n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    new_functions = {n.name: n for n in ast.parse(value).body if isinstance(n, ast.FunctionDef)}
    unchanged = [name for name in old_functions if name not in ('parse', 'kernel_factory_scope', 'run')]
    assert all(ast.dump(old_functions[name], include_attributes=False) ==
               ast.dump(new_functions[name], include_attributes=False) for name in unchanged)
    old_classes = {n.name: n for n in ast.parse(original).body if isinstance(n, ast.ClassDef)}
    new_classes = {n.name: n for n in ast.parse(value).body if isinstance(n, ast.ClassDef)}
    assert all(ast.dump(node, include_attributes=False) == ast.dump(new_classes[name], include_attributes=False)
               for name, node in old_classes.items())
    # Reverse only the explicit variant checks/labels, retaining all original
    # request, numerical comparison, evidence and lifetime/error behavior.
    restored = value.replace(new_validation, old_validation).replace(new_note, old_note)
    restored_run = next(n for n in ast.parse(restored).body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    assert ast.dump(restored_run, include_attributes=False) == ast.dump(old_functions['run'], include_attributes=False)
    target.write_text(value)
    proof.write_text(json.dumps({'source_sha256': EXPECTED,
        'target_sha256': hashlib.sha256(value.encode()).hexdigest(),
        'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'unchanged_function_ast': unchanged,
        'run_body_equal_after_reversing_only_variant_validation_and_note': True,
        'scope': 'private labelled kernel binding only; all HTTP/token/evidence/cleanup behavior retained'}, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--proof', type=Path, required=True)
    a = p.parse_args()
    generate(a.source, a.target, a.proof)
