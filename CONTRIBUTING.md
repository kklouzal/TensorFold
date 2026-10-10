# Maintaining this fork

The maintained fork is `https://github.com/kklouzal/TensorFold`, on
`gb10-flash-next`; upstream is `https://github.com/ashhart/TensorFold`.
Start with the [README](README.md), the model's [recipe](docs/recipes/README.md),
and the [container guide](deploy/gb10/README.md). Read applicable `AGENTS.md`
instructions before changing code. Keep unrelated working-tree changes intact.

## Documentation and evidence

Update `README.md` in the same change whenever capabilities, supported models
or platforms, CLI flags, defaults, build inputs, deployment commands, performance
results, or validation status change. Update the corresponding CLI help, recipe,
API documentation, and deployment guide when affected. Keep usage examples
consistent with the actual parser and supported model/backend combinations.

Performance claims must identify the model revision, hardware, runtime pins,
context, parallelism, cache state, workload, comparison order, and uncertainty.
Separate source checks, synthetic tests, and actual installed-container or
full-model results. Record unrun checks as pending; a result for an earlier image
does not qualify a later image. Keep the current evidence scope in
[optimization validation](docs/optimization-validation.md).

For an ongoing task, retain its decisions, source identities, commands, failures,
results, and remaining gates in a persistent ledger. Private experiment packets
under `task-artifacts/` are excluded from ordinary Docker builds; maintained
tests must work from a fresh checkout without those packets.

## Development and verification

Use the backend's actual supported environment: Apple Silicon for MLX, or the
pinned native Linux ARM64/AMD64 container for CUDA. The project requires Python
3.11 or newer; the reproducible CUDA container uses Python 3.12. A local editable
installation is `python -m pip install -e '.[test]'`. This does not provision a
CUDA compiler or reproduce the Docker dependency locks.

Run the checks relevant to the change from the repository root:

```bash
python -m pytest tests/test_deployment_pins.py tests/test_verification_public_permissions.py -q
python -m ruff check deploy/gb10/platform_pins.py tests/test_deployment_pins.py
git diff --check
```

These commands check deployment contracts; select the affected maintained files
for other changes. Ruff is a separate development tool. For CUDA, build
the separate `verification` Docker target and follow the
[CPU and GPU commands](deploy/gb10/README.md). Test changed boundaries and
failure paths as well as normal execution. Numerical and concurrency changes
need independent oracles, affected model/format coverage, and actual target
execution. Measure performance changes with identical quality settings and
bounded resources; do not substitute import success or a microbenchmark for an
affected full-model gate.

Regenerate dependency locks through `deploy/gb10/lock_dependencies.py` from a
recorded matching-platform resolver report. Keep wheel hashes, architecture,
CUDA/PyTorch/Triton pins, and cache identity coherent. Never edit generated
locks or installed/vendor dependencies to make validation pass.

## Pulling upstream changes

Keep fork changes on the maintained branch and merge upstream into a reviewable
integration branch:

```bash
git switch gb10-flash-next
git fetch origin
git pull --ff-only origin gb10-flash-next
git remote add upstream https://github.com/ashhart/TensorFold.git
git fetch upstream
git switch -c integrate-upstream-YYYYMMDD
git merge upstream/main
```

The `remote add` command is needed only once. Check `git remote show upstream`
for its current default branch and replace `main` if it differs. Resolve changes
against the fork's current contracts, update generated inputs and documentation,
and run the affected checks before merging the integration branch into
`gb10-flash-next`. Build a new image from the resulting commit; update deployed
containers by replacing the image rather than modifying their installed code.
