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
