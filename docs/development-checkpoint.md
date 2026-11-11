# Development handoff checkpoint

The retained development branch is `python-0.6`. This checkpoint preserves the
complete pending source, tests and research drafts so development can continue
from a standalone checkout on the V100 host. It is a development checkpoint;
production deployments retain their separately qualified image identities.

## Preserved work

- SM70 owner preparation, Q8 preparation, GDN, routing, HC graph and model
  identity work, including their independent references and controls.
- The private 1Cat expert integration, active-frame preparation, down-projection
  trials and direct raw-IQ2 gate/up prototype.
- Dense prefill alternatives and private B16 codec experiments with their tests.
- The four user-authored root `!` design/research documents. These remain drafts;
  the maintained design authority is the bundle in `ProviderOne Implementation Docs/`.
- Qualification tools and the unfinished offline candidate comparison adapter.

Component receipts belong to their recorded source/image and workload. They do
not qualify this combined checkpoint. Native compilation, full-model quality,
capacity and performance gates remain applicable to each trial before adoption.

## Known unfinished comparison work

The offline candidate adapter in `tools/v100_rotorquant_quality.py` was stopped
with **71 passing and two failing CPU controls**. Its new positive controls
reject the Docker inspect JSON array because a reused reader accepts only JSON
objects. Some refusal controls can consequently fail before the boundary they
intend to exercise. Fix that narrow reader contract and recheck the positive
and refusal controls before relying on the adapter. The existing strict teacher
comparator retains its independent contract.

The raw-IQ2 gate/up prototype also remains unqualified on SM70: its GPU suite was
collected, but native compilation and execution were not completed. The private
B16 experiments are preserved for development, without a serving selection claim.

## Continue development

Use the current procedures in [the V100 deployment guide](../deploy/v100/README.md),
[source foundation guide](../deploy/v100/source-foundation.md) and
[benchmarking guide](benchmarking.md). Build containers on the V100 host with
the maintained parallel-build arguments and persistent native caches. Keep
runtime dependencies and verification containerized.

Before releasing a new image, bind the exact committed source, installed
application/native origins, SDK, devices, checkpoint and selected profile, then
run the affected checks. A clean Git checkout and successful transfer establish
source integrity, not target qualification. Ignored caches, compiled outputs,
credentials, model payloads and active service configuration are outside the
source handoff.
