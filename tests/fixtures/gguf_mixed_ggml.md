`gguf_mixed_ggml.npz` contains original packed bytes and independently decoded
FP32 bit patterns for Q6_K, IQ3_XXS, IQ3_S, IQ2_S, IQ4_XS and Q2_0. It includes
every IQ codebook index with nonzero scale, signed zero, subnormal/negative/large
finite scales and multiple rows/blocks. GPU tests also cover partial TP columns
and repeated row gathers. These are format controls, not model-quality evidence.

The embedded provenance identifies the compiler, exact original source hashes,
generator hash and flags. Regenerate from unmodified source files at llama.cpp
revision `b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9`:

```bash
python tools/generate_gguf_mixed_fixture.py \
  --quants /path/to/ggml/src/ggml-quants.c \
  --common /path/to/ggml/src/ggml-common.h
```

The generator checks both original source SHA256 values, extracts the six
original function bodies unchanged and compiles them with the original header.
Its conversion shim uses the C compiler's `_Float16` conversion. It never reads
TensorFold's layouts, decoders or integer codebooks to derive expected values.
GGML MIT attribution is retained in `THIRD_PARTY_NOTICES.md` and
`LICENSES/GGML-MIT.txt`.

`gguf_mixed_projection_ggml.npz` is a separate finite FP16 projection corpus.
It retains every original packed quant/sign/local-scale byte, changing only
the block's FP16 global scale. The same unmodified pinned GGML C dequantizers
produce its independent FP32 oracle bits. Regenerate with the command above
plus `--finite-projections --output tests/fixtures/gguf_mixed_projection_ggml.npz`.
The full-range corpus and its FP16-overflow reproducer remain unchanged.

The independent maximum quant multipliers are4096 (Q6_K),1976.25 (IQ3_XXS),
7905 (IQ3_S),988.125 (IQ2_S),4064 (IQ4_XS) and2 (Q2_0). Global scale magnitude
is bounded by2^-20, so the common8192 bound gives coefficients at most2^-7.
For finite FP16 inputs with magnitude at most1 and K2560, FP32 dot products
stay below21 including rounding; SwiGLU/FP16 activation stays below441.
A K80 down projection using these coefficients stays below276. The existing
IQ4_NL fixture's0.001 global scale gives the looser finite down bound4500.
Projection controls assert this input/coefficient domain and retain strict
bitwise comparisons. The full-range FP32 corpus intentionally contains scales
such as65504 that overflow after FP16 coefficient conversion; it cannot serve
as a finite FP16 dot-product oracle without changing that contract.

The generator selects the full-range path by default and the separate projection
path when `--finite-projections` is present. `--output` overrides either path.
Use `--dry-run` to inspect this choice without reading references, compiling or
writing either corpus. Reference hashes are validated during actual generation.
