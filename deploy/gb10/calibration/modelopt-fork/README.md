# ModelOpt Triton 3.9 source candidate

This incremental Git bundle contains TensorFold's local ModelOpt compatibility
commit, based on the upstream revision in `source.json`. It is **not yet GPU
qualified or admitted as a producer image**. The readable patch is generated
from that same commit; edit the canonical fork, then regenerate these artifacts.

The change replaces removed block pointers in the dynamic NVFP4 activation
kernel with masked pointer tensors. Quantization helpers and floating-point
operations are unchanged. The test change covers ragged/strided inputs and moves
an unrelated MX extension getter out of collection and into its own test body.

The original guarded GPU test reached the Hessian pass, then Triton 3.9 rejected
`tl.make_block_ptr`. Upstream HEAD still matched the pinned base. The alternative
CUDA-MX implementation was inspected but is not admitted: its scalar quantizer
reads an uninitialized sign for zero inputs. No installed or upstream code was
patched, and no NVIDIA remote was pushed.

To reconstruct the exact candidate in an upstream clone that contains the base:

```bash
git bundle verify /absolute/path/modelopt-triton39.bundle
git fetch /absolute/path/modelopt-triton39.bundle refs/heads/tensorfold-triton39
git switch --detach a851b028ecbd5f843eb1fc05b93cc2d0abed0de3
```

Verify the bundle and patch SHA256 values against `source.json` first. The commit
uses a dedicated local build-signing key; its public key is recorded there. This
is local signature verification, not a claim of GitHub-verified identity. The
private key remains outside tracked files in the fork's Git metadata.

Required next gates are a clean source wheel, compatible producer image, the
focused native GPU tests under their declared test dependencies, and the full
guarded calibration smoke. Full-model quality and inference performance remain
separate requirements.
