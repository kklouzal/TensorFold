"""Legal inline packed tensors with compiler-owned alignment and bounded stages.

Quantized rows occupy 128 bytes with a 256-nibble stride. Cohorts borrow these
rows synchronously; each selected K slice retains its original group interval.
When stage plus original partials cannot coexist, operation-owned float32
partials are published and added in the same slice order by a separate kernel.
No device pointer address or immutable public parameter assumption is used.
"""
from __future__ import annotations


def staging_plan(nt: int, sk: int, *, coop: bool = False) -> tuple[int, bool, int]:
    """Return cohort capacity, device-partial policy and actual shared bytes."""

    stage = min(nt, 256) * 128
    # Include the literal template's one unused array slot at SK==1, without
    # depending on compiler elimination to make the declared storage fit.
    slots = max(sk - 1, 1)
    partial = slots * (16 * 64 * 4 if coop else (nt // 16) * 8 * 32 * 4)
    device = partial + stage > 32768
    retained = 0 if device else partial
    count = min(sk, (32768 - retained) // stage)
    return count, device, retained + count * stage


def _replace(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"packed tensor source contract changed: {old[:70]!r}")
    return source.replace(old, new)


def _loop(source: str) -> tuple[str, str, str]:
    start = source.index("  for (int g = g_begin; g < g_end; g++) {")
    opened = source.index("{", start)
    depth, end = 1, opened + 1
    while depth:
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
        end += 1
    return source[:start], source[opened + 1:end - 1], source[end:]


SCALAR_PAIRS = r"""
inline uint scale_pair_word(const device bfloat* p) {
  return uint(as_type<ushort>(p[0])) | (uint(as_type<ushort>(p[1])) << 16);
}
inline uint4 scale_pair_words4(const device bfloat* p) {
  return uint4(scale_pair_word(p), scale_pair_word(p + 2), scale_pair_word(p + 4), scale_pair_word(p + 6));
}
"""


def scalar_scale_source(source: str, *, coop: bool = False) -> str:
    """Preserve pair bits through scalar BF16 reads, including offset views."""
    if coop:
        source = _replace(source, "const device uint* sbw = (const device uint*)SBt;",
                          "const device bfloat* sbw = SBt;")
        return _replace(source, "sbw[g * N + n0 + ecol[i]]",
                        "scale_pair_word(sbw + size_t(2) * (g * N + n0 + ecol[i]))")
    source = _replace(source, "const device uint4* sbv = (const device uint4*)SBt;",
                      "const device bfloat* sbv = SBt;")
    return _replace(source, "sbv[(g * N + n0 + f * 16 + fn) / 4]",
                    "scale_pair_words4(sbv + size_t(2) * (g * N + n0 + f * 16 + fn))")


def tensor_source(source: str, *, coop: bool = False, tiled: bool = False,
                  nibbles: bool = False, grouped: bool = False, partials: bool = False) -> str:
    """Generate only storage/publication changes around unchanged arithmetic.

    The literal templates are project-owned inputs, never compiled directly.
    Every thread executes both barriers in every cohort round. Only complete
    participating SIMD groups/pairs enter an op; empty slices keep initialized
    zero accumulators. All padded words are initialized before tensor access.
    """

    width, owner, local, step = ("64", "slice", "tip", "64") if coop else ("NT", "sg", "lane", "32")
    columns = width if coop or nibbles else "STAGE_COLS"
    gs = "GS" if not nibbles or grouped else "64"
    declaration = (("  constexpr int STAGE_COLS = NT < 256 ? NT : 256;\n" if not coop and not nibbles else "")
                   + f"  alignas(128) threadgroup uint packed_stage[STAGES * {columns} * 32];\n"
                   f"  threadgroup uint* stage = packed_stage + ({owner} % STAGES) * {columns} * 32;\n")
    tensor = ("tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> "
              "b((threadgroup uchar*)stage, " + f"dextents<int32_t, 2>({gs}, {columns}), "
              "array<int32_t, 2>{1, 256});")
    if nibbles:
        stage_size = "(GS / 8)" if grouped else "8"
        old = (f"  threadgroup uint stage_all[SK * NT * {stage_size}];"
               + ("        // per K slice: NT columns x GS nibbles" if grouped
                  else "               // per K slice: NT columns x 64 nibbles")
               + f"\n  threadgroup uint* stage = stage_all + sg * NT * {stage_size};\n")
        source = _replace(source, old, declaration)
        old_tensor = ("tensor<threadgroup uint4b_format, dextents<int32_t, 2>, tensor_inline> "
                      f"b((threadgroup uchar*)stage, dextents<int32_t, 2>({gs}, NT));")
        source = _replace(source, old_tensor, tensor)
        source = source.replace(f"stage[lane * {stage_size} + c]", "stage[lane * 32 + c]")
    else:
        if not coop:
            source = _replace(source, "  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> tB((device uchar*)Wq, dextents<int32_t, 2>(K, N));\n", declaration)
            source = _replace(source, "    auto b = tB.slice(g * GS, n0);", "    " + tensor)
        else:
            source = _replace(source, "  auto a0 = tA.slice(0, 0);\n", declaration + "  auto a0 = tA.slice(0, 0);\n")
            source = _replace(source, "  tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b0((device uchar*)Wq, dextents<int32_t, 2>(GS, 64));",
                              "  " + tensor.replace(" b(", " b0("))
            source = _replace(source, "    tensor<device uint4b_format, dextents<int32_t, 2>, tensor_inline> b(\n"
                              "        (device uchar*)Wq + (int64_t)(threadgroup_position_in_grid.x * KG + g) * (64 * GS / 2), dextents<int32_t, 2>(GS, 64));", "    " + tensor)
    before, body, after = _loop(source)
    if nibbles:
        at = body.index("    simdgroup_barrier(mem_flags::mem_threadgroup);\n    float s[NF][4]")
        loader = body[:at] + f"\n    for (int j = {gs} / 8; j < 32; j++) stage[lane * 32 + j] = 0;\n"
        body = body[at + len("    simdgroup_barrier(mem_flags::mem_threadgroup);\n"):]
        body = _replace(body, "    simdgroup_barrier(mem_flags::mem_threadgroup);   // the op has read the stage before the next group's widening\n", "")
    else:
        shift = "0" if coop else "col_begin"
        packed = (f"((size_t(threadgroup_position_in_grid.x) * KG + g) * {width} + {shift} + col) * (GS / 8) + word"
                  if tiled or coop else "size_t(nn) * (K / 8) + g * (GS / 8) + word")
        loader = (f"\n    for (int i = {local}; i < {columns} * 32; i += {step}) {{\n"
                  f"      const int col = i / 32, word = i % 32, nn = n0 + {shift} + col;\n"
                  f"      stage[i] = {shift} + col < {width} && nn < N && word < GS / 8 ? Wq[{packed}] : 0u;\n"
                  "    }\n")
        if not coop:
            body = _replace(body, "for (int f = 0; f < NF; f++)\n        for (int r = 0; r < 2; r++)",
                            "for (int f = 0; f < min(STAGE_COLS, NT - col_begin) / 16; f++)\n        for (int r = 0; r < 2; r++)")
            body = _replace(body, "            C[t][i] = fma(s[f][j], P[t * NF * 8 + i], fma(bb[f][j], r ? xs1 : xs0, C[t][i]));",
                            "            const int target_f = col_begin / 16 + f, target_i = target_f * 8 + r * 4 + j;\n"
                            "            C[t][target_i] = fma(s[target_f][j], P[t * NF * 8 + i], fma(bb[target_f][j], r ? xs1 : xs0, C[t][target_i]));")
    column_start = "      for (int col_begin = 0; col_begin < NT; col_begin += STAGE_COLS) {\n" if not coop and not nibbles else ""
    column_end = "      }\n" if column_start else ""
    prefix = (f"  for (int wave = 0; wave < (SK + STAGES - 1) / STAGES; wave++) {{\n"
            "    for (int round = 0; round < (KG + SK - 1) / SK; round++) {\n"
            "      const int g = g_begin + round;\n"
            f"      const bool live_group = {owner} / STAGES == wave && g < g_end;\n")
    loop = (prefix + column_start + "      if (live_group) {" + loader + "      }\n"
            + "      threadgroup_barrier(mem_flags::mem_threadgroup);\n"
            + "      if (live_group) {" + body + "      }\n"
            + "      threadgroup_barrier(mem_flags::mem_threadgroup);\n" + column_end
            + "    }\n  }\n")
    if partials:
        if coop:
            after = """  for (int i = 0; i < CAP; i++) {
    const int m = rb + erow[i], n = n0 + ecol[i];
    if (m < M && n < N) PART[(size_t(slice) * M + m) * N + n] = C[i];
  }
"""
        else:
            after = """  for (int t = 0; t < TMR; t++)
    for (int f = 0; f < NF; f++)
      for (int r = 0; r < 2; r++) {
        const int m = rb + t * 16 + fm + 8 * r;
        const int n = n0 + f * 16 + fn;
        if (m < M && n < N)
          for (int j = 0; j < 4; j++) PART[(size_t(sg) * M + m) * N + n + j] = C[t][f * 8 + r * 4 + j];
      }
"""
    return scalar_scale_source(before + loop + after, coop=coop)


ORDERED_REDUCE = r"""
  const uint e = thread_position_in_grid.x;
  if (e >= M * N) return;
  float v = PART[e];
  for (int slice = 1; slice < SK; slice++) v += PART[size_t(slice) * M * N + e];
  Y[e] = static_cast<bfloat>(v);
"""
