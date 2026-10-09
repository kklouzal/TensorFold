"""Header-derived two-size compact EXL3 arena planning; no CUDA allocation.

Roles come from the maintained load_compact contract: routed IDs0..count-1,
then the named shared expert at IDcount. The planner does not infer roles from
sizes, model names or IDs. Every immutable registered layer must supply them.
Payload budget and total payload+publication-device budget are independent
bounds. Capacity counts logical handles; safe_lease_count bounds any list of
unique experts and may be smaller. No native arithmetic or quantization change.

This specialized REGION has exactly one routed size and one larger shared size.
Reserve one cell for each shared expert; route cells compete under unchanged
aging LFU/LRU/physical-address tie rules. Other distributions, undersized shared
reservation or a reduced safe wave limit retain the existing fixed-cell layout.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ExpertSpec:
    layer: int
    expert: int
    size: int
    shared: bool


@dataclass(frozen=True)
class Cell:
    handle: int
    offset: int
    size: int
    shared_key: tuple[int, int] | None


@dataclass(frozen=True)
class Layout:
    kind: str
    cells: tuple[Cell, ...]
    payload_bytes: int
    publication_device_bytes: int
    publication_host_bytes: int
    safe_lease_count: int
    resident_capacity: int
    payload_budget: int
    device_budget: int
    reason: str


def integer(value, name, minimum=0):
    if type(value) is not int or value < minimum or value > 2**63 - 1:
        raise ValueError(f'{name} must be a bounded integer >= {minimum}')
    return value


def plan(specs, payload_budget, device_budget):
    """Prepare immutable cell topology; allocate nothing on an accelerator."""
    integer(payload_budget, 'payload budget', 1)
    integer(device_budget, 'payload+publication device budget', 1)
    if not isinstance(specs, tuple) or not specs or len(specs) > 1 << 20:
        raise ValueError('bounded nonempty immutable expert registry required')
    keys = set()
    routed, shared = [], []
    layers = {}
    for spec in specs:
        if not isinstance(spec, ExpertSpec) or type(spec.shared) is not bool:
            raise ValueError('typed immutable expert role specification required')
        integer(spec.layer, 'layer')
        integer(spec.expert, 'expert')
        integer(spec.size, 'aligned trellis bytes', 16)
        if spec.size % 16 or (spec.layer,spec.expert) in keys:
            raise ValueError('aligned distinct expert registry required')
        keys.add((spec.layer,spec.expert))
        (shared if spec.shared else routed).append(spec)
        layers.setdefault(spec.layer, []).append(spec)
    for values in layers.values():
        order = sorted(values, key=lambda x:x.expert)
        if [x.expert for x in order] != list(range(len(order))) or len(order) < 2 or \
                [x.shared for x in order] != [False] * (len(order)-1) + [True]:
            raise ValueError('exact maintained routed-then-shared logical registration required')
    largest = max(x.size for x in specs)
    fixed = min(payload_budget // largest, device_budget // (largest+32))
    if fixed < 1 or fixed > 2**31-1:
        raise ValueError('budget cannot admit a bounded original fixed GPU cell')
    small_sizes, large_sizes = {x.size for x in routed}, {x.size for x in shared}
    eligible = len(small_sizes) == len(large_sizes) == 1 and next(iter(small_sizes)) < next(iter(large_sizes))
    reason = 'distribution outside exact two-size routed/shared REGION'
    if eligible:
        small, large = next(iter(small_sizes)), next(iter(large_sizes))
        nlarge = len(shared)
        nsmall = min((payload_budget - nlarge*large)//small,
                     (device_budget - nlarge*(large+32))//(small+32))
        # Do not improve residency by silently reducing the existing wave limit.
        if nsmall >= fixed and nsmall > 0 and nsmall+nlarge <= 2**31-1:
            specs_shared = sorted(shared, key=lambda x:(x.layer,x.expert))
            cells = [Cell(i, i*small, small, None) for i in range(nsmall)]
            base = nsmall*small
            cells += [Cell(nsmall+i, base+i*large, large, (x.layer,x.expert)) for i,x in enumerate(specs_shared)]
            size = nsmall*small+nlarge*large
            return Layout('two-size',tuple(cells),size,len(cells)*32,len(cells)*64,nsmall,len(cells),
                          payload_budget,device_budget,'all shared entries admitted without reducing safe wave capacity')
        reason = 'shared reservation would reduce the original safe wave capacity'
    cells = tuple(Cell(i,i*largest,largest,None) for i in range(fixed))
    return Layout('fixed',cells,fixed*largest,fixed*32,fixed*64,fixed,fixed,payload_budget,device_budget,reason)
