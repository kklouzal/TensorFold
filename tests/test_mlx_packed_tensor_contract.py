"""SDK-free source, ownership and integer layout oracles for legal Metal tensors.

These controls never compile or execute Metal and cannot qualify its arithmetic.
The native bit/codegen/current-model/resource gates remain separate.
"""
from __future__ import annotations

import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "src/tensorfold/kernels/qwen/dense/v1"


def source_scope(filename):
    scope = {}
    tree = ast.parse((BASE / filename).read_text())
    if filename == "lane_widen.py":
        stage = source_scope("lane_stage.py")
        scope.update(tensor_source=stage["tensor_source"], scalar_scale_source=stage["scalar_scale_source"])
        tree.body = [n for n in tree.body if not (isinstance(n, ast.ImportFrom) and n.module.startswith("tensorfold."))]
    exec(compile(tree, str(BASE / filename), "exec"), scope)
    return scope


def templates():
    out = {}
    for node in ast.parse((BASE / "lane_qmm.py").read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ("_MAIN_TEMPLATE", "_COOP_TEMPLATE") for t in node.targets):
            exec(compile(ast.Module([node], []), "lane_templates", "exec"), out)
    widen = source_scope("lane_widen.py")
    return ((out["_MAIN_TEMPLATE"], {}), (out["_MAIN_TEMPLATE"], {"tiled": True}),
            (out["_COOP_TEMPLATE"], {"coop": True}), (widen["_NIBBLES_TEMPLATE"], {"nibbles": True}),
            (widen["_NIBBLES_GROUPED_TEMPLATE"], {"nibbles": True, "grouped": True}))


class PackedTensorContracts(unittest.TestCase):
    def test_effective_sources_have_aligned_padded_rows_and_uniform_publication(self):
        api = source_scope("lane_stage.py")
        for original, policy in templates():
            for partials in (False, True):
                actual = api["tensor_source"](original, partials=partials, **policy)
                self.assertNotIn("tensor<device uint4b_format", actual)
                self.assertEqual(actual.count("alignas(128) threadgroup uint packed_stage"), 1)
                self.assertIn("array<int32_t, 2>{1, 256}", actual)
                self.assertEqual("PART[" in actual, partials)
                self.assertEqual("threadgroup float part[" in actual, not partials)
                # The conditional ends before both shared-memory barriers.
                compact = "".join(actual.split())
                self.assertIn("}threadgroup_barrier(mem_flags::mem_threadgroup);if(live_group)", compact)
                self.assertIn("}threadgroup_barrier(mem_flags::mem_threadgroup);}}", compact)
                for line in original.splitlines():
                    # All explicit FP FMA updates and op descriptors retain text.
                    if "= fma(" in line or "matmul2d_descriptor(" in line or "op.run(" in line:
                        normalized = actual.replace("target_i", "i").replace("target_f", "f")
                        self.assertIn(line.strip(), normalized)

    def test_shared_storage_oracle_and_original_slice_intervals(self):
        plan = source_scope("lane_stage.py")["staging_plan"]
        for nt in (16, 32, 64, 96, 128, 256, 272, 512, 1024):
            for sk in range(1, 33):
                for coop in ((False, True) if nt == 64 else (False,)):
                    stages, device, shared = plan(nt, sk, coop=coop)
                    slot = 4096 if coop else nt * 64
                    literal_partials = max(sk - 1, 1) * slot
                    self.assertEqual(device, literal_partials + min(nt, 256) * 128 > 32768)
                    self.assertEqual(shared, stages * min(nt, 256) * 128 + (0 if device else literal_partials))
                    self.assertLessEqual(shared, 32768)
                    self.assertGreaterEqual(stages, 1)
                    for kg in (1, 2, 3, 7, 16, 33):
                        seen = set()
                        for wave in range((sk + stages - 1) // stages):
                            for turn in range((kg + sk - 1) // sk):
                                owners = set()
                                for slice_ in range(sk):
                                    begin, end = slice_ * kg // sk, (slice_ + 1) * kg // sk
                                    g = begin + turn
                                    if slice_ // stages == wave and g < end:
                                        self.assertNotIn(slice_ % stages, owners)
                                        owners.add(slice_ % stages)
                                        self.assertNotIn((slice_, g), seen)
                                        seen.add((slice_, g))
                        expected = {(s, g) for s in range(sk) for g in range(s * kg // sk, (s + 1) * kg // sk)}
                        self.assertEqual(seen, expected)
        # Column cohorts cover the complete original descriptor width, without
        # narrowing NT or repeating any output's per-group FMA update.
        for nt in (16, 96, 256, 272, 512, 1024):
            staged = min(nt, 256)
            writes = [base + f * 16 + j for base in range(0, nt, staged)
                      for f in range(min(staged, nt - base) // 16) for j in range(16)]
            self.assertEqual(writes, list(range(nt)))

    def test_padded_word_writer_and_consumed_quantized_coordinate_oracle(self):
        # A complete SIMD or pair writes each word of a padded row exactly once.
        for nt in (16, 32, 64, 96, 128, 256):
            for threads in ((32, 64) if nt == 64 else (32,)):
                addresses = [i for lane in range(threads) for i in range(lane, nt * 32, threads)]
                self.assertEqual(sorted(addresses), list(range(nt * 32)))
            for gs in (32, 64):
                # Explicit stride256 nibbles gives128-byte rows, even for GS32.
                for col in range(nt):
                    for code in range(gs):
                        bit = col * 256 * 4 + code * 4
                        self.assertEqual(bit // 32, col * 32 + code // 8)
                        self.assertEqual(bit % 32, (code % 8) * 4)
                        self.assertLess(bit // 32, nt * 32)
                        self.assertEqual((col * 128) % 128, 0)

    def test_scalar_packing_preserves_every_bfloat_payload_and_word_offsets(self):
        # Metal is little endian; as_type<ushort> returns the original bits.
        for word in range(65536):
            for partner in (0, 0x8000, 0x7F80, 0xFFFF):
                pair = word | (partner << 16)
                self.assertEqual(pair.to_bytes(4, "little"), word.to_bytes(2, "little") + partner.to_bytes(2, "little"))
        text = (BASE / "simd_qmm.py").read_text()
        self.assertNotIn("(const device uint4*)", text)
        self.assertNotIn("(const device uint2*)", text)
        self.assertNotIn("(const device float2*)", text)
        self.assertNotIn("device float2* xf", text)
        self.assertIn("alignas(16) threadgroup float xs", text)
        for original, policy in templates():
            emitted = source_scope("lane_stage.py")["tensor_source"](original, **policy)
            self.assertNotIn("(const device uint4*)SBt", emitted)
            self.assertNotIn("(const device uint*)SBt", emitted)
        for row in (0, 1, 7):
            for k in (64, 128, 512):
                for j in range(k // 8):
                    self.assertEqual(16 * (row * (k // 8) + j), 2 * (row * k + 8 * j))
        for group in (32, 64):
            wpg = group // 8
            for c in range(32):
                for h in range(wpg // 4):
                    self.assertEqual(4 * ((wpg // 4) * c + h), wpg * c + 4 * h)

    def test_scale_pair_address_widens_before_doubling_valid_native_domain(self):
        # This admitted low-bit region's packed pair index fits signed int32,
        # while its scalar BF16 index must use size_t before multiplication.
        n, k, group, mp, sk = 65532, 1048576, 32, 16, 1
        pair = (k // group - 1) * n + n - 1
        self.assertLessEqual(mp * n, (1 << 31) - 1)
        self.assertLessEqual((k // group) * n, (1 << 31) - 1)
        self.assertLessEqual(n * (k * 2 // 32), (1 << 32) - 1)
        self.assertEqual(pair * 2, 4294705150)
        self.assertGreater(pair * 2, (1 << 31) - 1)
        self.assertLessEqual((pair * 2 + 2) * 2, (1 << 64) - 1)
        self.assertEqual(sk, 1)
        api = source_scope("lane_stage.py")
        for original, policy in templates():
            emitted = api["tensor_source"](original, **policy)
            scalar = "scale_pair_word(sbw + size_t(2) *" if policy.get("coop") else "scale_pair_words4(sbv + size_t(2) *"
            self.assertIn(scalar, emitted)
            self.assertNotIn("sbw + 2 *", emitted)
            self.assertNotIn("sbv + 2 *", emitted)


if __name__ == "__main__":
    unittest.main()
