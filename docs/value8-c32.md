# Private Value8 32-State batch envelope

The existing explicit `preparation.gdn_value8_capture: true` candidate admits
up to32 resident States, at most five rows per State, and at most160 total
verification rows. It still requires `native_gdn_capture`, resident prepared
SM70 kernels, Half Hq2/Hv6/D128 operands and the original exact-target graph
owner. Omission retains the incumbent. Eager batch, prefill and MTP dispatch
remain unchanged; this is an extension of the target captured component's
participant limit, not a new codec or numerical mode.

The CUDA implementation already indexes each State through `blockIdx.z` and
its registered pointer column/starts interval. It contains no24-State array or
batch-wide shared-memory reduction. At the new maximum, the three grids are
`(2,5,32)`, `(16,6,32)` and `(6,5,32)`. Core64-thread CTAs still compute eight
value rows using the same per-State order and Preload1/2/4/5 variants. No CUDA
math/source changes accompany the Python/C++ envelope extension. All1..24
shapes retain their original grids, arithmetic and independent incumbent oracle.

`RoundTables` retains its dynamically sized eight-plane ABI. For S32, width5,
R160 and L linear layers, each int32 device/host mirror contains
`9R+2S+1 = 1505` words, or6020 bytes; each int64 pointer mirror contains
`8LS = 256L` words, or2048L bytes. These tables and all State pointees already
belong to the full32 native-GDN geometry. Registration still checks every
mutable State/output/saved tensor for overlap; actor staging waits for the
previous mirror DMA, retains pointer owners and validates generation, parity,
stream and compact row intervals before replay. Side capture only consumes
those validated fixed views. Existing graph/State retirement owns completion.

The required Q payload at R160 is163,840 bytes (`1024R`). The model candidate
wraps `MultiDecoder.gdn.q`, which is already paid by the existing native-GDN
scratch forecast; it allocates and credits no second Q tensor. A standalone
component caller must fund that Q workspace explicitly. Capacity receipts now
report the actual participant/row/window capacities and retained device/pinned
Table bytes beside reused Q bytes; the numbers are not a second grant.

The existing exact graph catalogue derives its maximum from actual States and
row credit, so its32- and160-row/32-participant keys need no catalogue-rule
change. The operation must still prove the requested catalogue fits, and the
ordinary full32 ISO4 capacity gate remains distinct. No attention codec support,
quality tolerance, closed24-State C8 teacher schema or performance policy is
expanded by this source change.

Focused CPU controls cover32-State table extents, row/workspace exhaustion,
last-unselected-State alias rejection, changed owner order/parity and paid-Q
capture binding. Added native controls compare32-,96-ragged- and160-row frames
against the unchanged incumbent/scalar oracle, changed frames and every
accepted prefix1..5 through capture/replay. Native32 execution, integrated
full32 model/quality and matched performance remain pending. Previous native
passes apply only to their recorded source/envelope and are not relabeled.
