"""Fixed Gemma CT MoE component diagnostic over two original checkpoint banks.

This source/verification-only tool uses explicit synthetic activations/routes.
It is not a model runner, capture utility, tuning policy or serving default.
Original checkpoint authentication/loading, input quantization, compilation,
oracle preparation, graph construction and teardown are outside the component
window and reported separately. The normal checked native entries remain in
every timed transaction. Root's existing device/resource owner must admit the
actor; endpoint samples here are not a whole-system peak guarantee.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from .benchmark_output import write_json
    from .worker_lifetime import raise_failures
else:
    from benchmark_output import write_json
    from worker_lifetime import raise_failures

CONFIGS = ((2, 2), (2, 3), (2, 4), (4, 2), (4, 3), (4, 4))
ARMS = ('W2S2', 'W2S3', 'W2S4', 'W4S2', 'W4S3', 'W4S4')
PUBLIC = 'PUBLIC4x4'
ROWS = (5, 3, 2, 1)
ROUTE_SETS = 16
BLOCKS = 6
EXPERTS, WIDTH, DIMS, SLOTS, CAPACITY_ROWS = 128, 704, 2816, 8, 16


def require(value, message):
    if not value:
        raise ValueError(message)


def positive(value):
    result = int(value)
    if result < 1 or result > 2**63 - 1:
        raise argparse.ArgumentTypeError('value must be a positive signed64 integer')
    return result


def layer_pair(value):
    try:
        values = tuple(int(part) for part in value.split(','))
    except ValueError as error:
        raise argparse.ArgumentTypeError('layers must be two distinct nonnegative integers') from error
    if len(values) != 2 or min(values) < 0 or len(set(values)) != 2:
        raise argparse.ArgumentTypeError('layers must be two distinct nonnegative integers')
    return values


def schedule():
    """Six whole blocks, each bracketed rotation then its bracketed reverse."""
    return tuple((PUBLIC, *ARMS[at:] + ARMS[:at], PUBLIC,
                  PUBLIC, *reversed(ARMS[at:] + ARMS[:at]), PUBLIC) for at in range(BLOCKS))


def routes(rows, index, locality):
    require(rows in ROWS and type(index) is int and 0 <= index < ROUTE_SETS,
            'route request differs from the fixed component protocol')
    require(locality in ('hot', 'scan'), 'locality must be hot or scan')
    return [[slot if locality == 'hot' else (8 * rows * index + 8 * row + slot) % EXPERTS
             for slot in range(SLOTS)] for row in range(rows)]


def pair_members(picks):
    flat = [expert for row in picks for expert in row]
    return sorted(range(len(flat)), key=lambda member: (flat[member], member))


def _raise_retired(primary, prior_cause, prior_context, prior_suppress, statuses):
    """Resource work precedes this guarded use of the adopted transport.

    Raw statuses, native links and caller-owned graph/tensor objects stay in
    traceback frames if cold list/group preparation cannot allocate.
    """
    if primary is None:
        return
    try:
        errors = [error for error in statuses if error is not None]
        BaseException.__cause__.__set__(primary, prior_cause)
        BaseException.__context__.__set__(primary, prior_context)
        BaseException.__suppress_context__.__set__(primary, prior_suppress)
        raise_failures(primary, errors)
    except BaseException as transport:  # Never replace a retained operation root on diagnostic exhaustion.
        if transport is primary:
            raise
        raise primary from transport


class _OwnedContext:
    """Correlate body/exit failures without replacing the library lifecycle.

    Only the selected graph, stream and inference contexts use this boundary.
    Their exact enter/exit calls, exception triple and suppression result stay
    library-owned. Bind callbacks/storage before entry; snapshot a body root's
    native links before exit can replace it. This adapter performs no capture,
    stream switching, device settlement or retry of its own.
    """
    __slots__ = ('_cause', '_context', '_enter', '_exit', '_manager', '_statuses', '_suppress')

    def __init__(self, context):
        self._manager = context
        self._enter, self._exit = type(context).__enter__, type(context).__exit__
        self._statuses = [None]
        self._cause = BaseException.__cause__.__get__
        self._context = BaseException.__context__.__get__
        self._suppress = BaseException.__suppress_context__.__get__

    def __enter__(self):
        return self._enter(self._manager)

    def __exit__(self, error_type, primary, traceback):
        if primary is None:
            return self._exit(self._manager, error_type, primary, traceback)
        prior_cause = self._cause(primary)
        prior_context = self._context(primary)
        prior_suppress = self._suppress(primary)
        try:
            result = self._exit(self._manager, error_type, primary, traceback)
        except BaseException as error:  # noqa: BLE001 - retain body and independent context-exit roots, including interruption.
            if error is not primary:
                self._statuses[0] = error
            try:
                _raise_retired(primary, prior_cause, prior_context, prior_suppress, self._statuses)
            except BaseException as transport:
                if transport is not primary:
                    raise primary from transport
                raise
        else:
            BaseException.__cause__.__set__(primary, prior_cause)
            BaseException.__context__.__set__(primary, prior_context)
            BaseException.__suppress_context__.__set__(primary, prior_suppress)
            return result


def _stream(path):
    from tensorfold.cuda.tensor_file import _regular_stream

    return _regular_stream(path)


def _digest(path, maximum=None):
    value = hashlib.sha256()
    with _stream(path) as source:
        identity = _descriptor_stamp(source)
        size = identity[3]
        require(size > 0 and (maximum is None or size <= maximum), 'file exceeds its hash work bound')
        done = 0
        while done < size:
            raw = source.read(min(1 << 20, size - done))
            require(raw and len(raw) <= size - done, 'short or invalid read while authenticating a file')
            value.update(raw)
            done += len(raw)
        require(source.read(1) == b'' and _descriptor_stamp(source) == identity,
                'file identity/size changed during authentication')
    return value.hexdigest()


def _identity(item):
    return [item.st_dev, item.st_ino, item.st_mode, item.st_size, item.st_mtime_ns, item.st_ctime_ns]


def _stamp(path):
    return _identity(path.stat())


def _descriptor_stamp(source):
    return _identity(os.fstat(source.fileno()))


class _CheckpointBytes:
    """One synchronous ordinary-file read through the existing FD-safe owner.

    No direct-IO/pinned staging pool or retained file resource is introduced.
    This diagnostic setup policy avoids importing the legacy Reader's owning
    with-open/finally-close boundary. SafeTensors still owns header/range/type
    interpretation; the caller uses the original normal bank loader.
    """
    def __init__(self, torch, identities, maximum):
        self.torch = torch
        self.identities = identities
        self.maximum = maximum
        self.requested_bytes = self.read_calls = 0

    def read(self, path, offset, count, device='cpu'):
        from tensorfold.cuda.tensor_file import byte_range

        require(type(count) is int and 0 <= count <= self.maximum - self.requested_bytes,
                'cumulative original-payload read budget is exhausted')
        self.requested_bytes += count
        self.read_calls += 1
        with _stream(path) as source:
            identity = _descriptor_stamp(source)
            require(identity == self.identities[str(Path(path))], 'checkpoint payload identity changed before read')
            byte_range(offset, count, identity[3], path)
            raw = self.torch.empty((count,), dtype=self.torch.uint8, device='cpu')
            view = memoryview(raw.numpy())
            source.seek(offset)
            done = 0
            while done < count:
                copied = source.readinto(view[done:min(count, done + (1 << 20))])
                if type(copied) is not int or not 0 < copied <= count - done:
                    raise OSError('checkpoint read made invalid progress')
                done += copied
            require(identity == _descriptor_stamp(source), 'checkpoint payload changed during read')
        return raw.to(device)

    def close(self):
        # Each read's existing _regular_stream owner has already retired.
        return None


def _authority(args):
    from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, _metadata_object

    # Parse the same immutable bounded bytes that were authenticated, rather
    # than reopening a metadata path between its hash and interpretation.
    with _stream(args.authority) as source:
        identity = _descriptor_stamp(source)
        require(0 < identity[3] <= MAX_HEADER_BYTES, 'checkpoint authority exceeds metadata bound')
        raw = source.read(identity[3] + 1)
        require(len(raw) == identity[3] and _descriptor_stamp(source) == identity,
                'checkpoint authority changed during read')
    require(hashlib.sha256(raw).hexdigest() == args.authority_sha256, 'checkpoint authority SHA differs')
    value = _metadata_object(raw, args.authority)
    return validate_authority(value, args.max_input_bytes)


def validate_authority(value, maximum):
    """Original file identities plus one cumulative input-authentication bound."""
    require(set(value) == {'schema', 'checkpoint_revision', 'files'} and type(value['schema']) is int
            and value['schema'] == 1 and type(value['checkpoint_revision']) is str
            and bool(value['checkpoint_revision']), 'checkpoint authority schema differs')
    files = value['files']
    require(type(files) is dict and files, 'checkpoint authority files are required')
    total = 0
    for name, item in files.items():
        require(type(name) is str and type(item) is dict and set(item) == {'sha256', 'bytes'}
                and type(item['sha256']) is str and len(item['sha256']) == 64
                and all(letter in '0123456789abcdef' for letter in item['sha256'])
                and type(item['bytes']) is int and 0 < item['bytes'] <= maximum,
                'checkpoint authority file record differs')
        total += item['bytes']
    require(total <= maximum, 'checkpoint authentication exceeds declared input work bound')
    return value


def _open_checkpoint(args, torch, authority):
    from tensorfold.checkpoint_metadata import read_config, read_json
    from tensorfold.cuda.direct_read import SafeTensors
    from tensorfold.cuda.tensor_file import checkpoint_path
    from tensorfold.families.gemma4.cuda.nvfp4_weights import validate_headers

    root = args.model.resolve(strict=True)
    raw = read_config(root)
    text = raw.get('text_config', raw)
    require(type(text) is dict and all(type(text.get(name)) is int and text[name] == expected
            for name, expected in (('num_experts', EXPERTS), ('moe_intermediate_size', WIDTH),
                                   ('hidden_size', DIMS), ('num_hidden_layers', 30))),
            'component metadata must declare the fixed Gemma26B geometry before header expansion')
    index = read_json(root, 'model.safetensors.index.json')
    mapping = None if index is None else index.get('weight_map')
    if index is not None:
        require(type(mapping) is dict and mapping and all(type(key) is str and type(value) is str
                for key, value in mapping.items()), 'original checkpoint requires a string weight_map')
    names = ['model.safetensors'] if mapping is None else sorted(set(mapping.values()))
    required = {'config.json', *names}
    if mapping is not None:
        required.add('model.safetensors.index.json')
    require(set(authority['files']) == required, 'authority must exactly cover config/index/original shards')
    identities = {}
    for name in sorted(required):
        path = checkpoint_path(root, name)
        before = _stamp(path)
        expected = authority['files'][name]
        require(before[3] == expected['bytes'] and _digest(path, expected['bytes']) == expected['sha256']
                and _stamp(path) == before, 'authenticated checkpoint file differs: ' + name)
        identities[name] = {'path': str(path), 'stamp': before, **expected}
    require(read_config(root) == raw and read_json(root, 'model.safetensors.index.json') == index,
            'checkpoint metadata changed across authentication')
    reader = SafeTensors([], reader=_CheckpointBytes(torch,
        {row['path']: row['stamp'] for row in identities.values()}, args.max_payload_read_bytes))
    # The borrowed byte adapter retains no file/staging resource. Every shard
    # parse is independently FD-owned, and duplicate/index checks match load().
    for name in names:
        shard = SafeTensors([checkpoint_path(root, name)], reader=reader.reader)
        require(not set(shard.where).intersection(reader.where), 'duplicate checkpoint tensor across shards')
        reader.where.update(shard.where)
    if mapping is not None:
        require(set(reader.where) == set(mapping) and all(reader.where[name][0] == checkpoint_path(root, file)
                for name, file in mapping.items()), 'checkpoint index differs from actual shard inventory')
    headers = {name: {'dtype': value[3], 'shape': value[4]} for name, value in reader.where.items()}
    config = validate_headers(raw, headers, vision=False)
    require((config.experts, config.moe_width, config.hidden) == (EXPERTS, WIDTH, DIMS)
            and all(layer < config.layers for layer in args.layers), 'component requires original Gemma128/704/2816 banks')
    require(all(_stamp(Path(row['path'])) == row['stamp'] for row in identities.values()),
            'checkpoint changed during header admission')
    return reader, config, identities


def _load_banks(reader, config, layers):
    from tensorfold.families.gemma4.cuda.nvfp4_experts import load_experts
    from tensorfold.families.gemma4.cuda.nvfp4_weights import CT_PREFIX, _calibrations, ct_tensor_contract

    native, _ = ct_tensor_contract(config)
    banks, scalar_hashes = [], []
    for layer in layers:
        prefix = f'{CT_PREFIX}layers.{layer}.experts'
        selected = {name: shape for name, shape in native.items() if name.startswith(prefix + '.')}
        require(len(selected) == EXPERTS * 3, 'complete original layer bank is required')
        calibrated = _calibrations(reader, selected)
        digest = hashlib.sha256()
        for name in sorted(selected):
            for suffix in ('weight_global_scale', 'input_global_scale'):
                raw = reader.get(name + '.' + suffix).view(reader.reader.torch.uint8).numpy().tobytes()
                digest.update((name + '.' + suffix + '\0').encode())
                digest.update(raw)
        scalar_hashes.append(digest.hexdigest())
        wrapped = load_experts(reader.get, prefix, count=EXPERTS, width=WIDTH, dims=DIMS,
                               device='cuda:0', calibrated=calibrated)
        banks.append((prefix, wrapped.bank))
    require(len(banks) in (1, 2) and len({prefix for prefix, _bank in banks}) == len(banks),
            'each phase requires distinct original checkpoint bank identities')
    return banks, scalar_hashes


@dataclass
class Owner:
    bank_index: int
    bank: Any
    buf: Any
    x: Any
    weights: Any
    output: Any
    picks: list
    members: list
    inverse: list
    up_base: tuple
    down_args: tuple
    combine_args: tuple
    input_hashes: dict
    expected: tuple | None = None


def _arguments(bank, buf, rows):
    pairs = rows * SLOTS
    scratch = buf.a4_scratch[(bank.dims, bank.width)]
    common = bank.shared_input and rows <= 65535
    projections, count = (1, rows) if common else (2, pairs)
    codes = scratch[0].view(-1)[:projections * count * (bank.dims // 2)].view(projections, count, bank.dims // 2)
    scales = scratch[1].view(-1)[:projections * count * (bank.dims // 16)].view(projections, count, bank.dims // 16)
    dc, ds = (value[:, :pairs] for value in scratch[2:])
    items = buf.plan.items[:min(pairs, bank.count + 1 + pairs // 16)]
    members = buf.plan.members[:pairs]
    fields = bank.fields
    return (codes, scales, fields[0], fields[2], fields[4], fields[5], items, buf.plan.counts,
            members, dc, ds, bank.width, bank.dims, SLOTS, common)


def prepare_owner(torch, bank_index, bank, rows, route_index, locality, seed):
    from tensorfold.cuda.nvfp4.experts_a4 import PlanBuffers

    generator = torch.Generator(device='cpu').manual_seed(seed + rows * 1009 + bank_index * 97 + route_index)
    cpu = (torch.randn((rows, bank.dims), generator=generator) * .3).to(torch.bfloat16)
    cpu[0] = .5
    x = cpu.to('cuda:0')
    weights = torch.linspace(.03125, .812345, SLOTS, dtype=torch.float32, device='cuda')[None].expand(rows, -1).contiguous()
    picks = routes(rows, route_index, locality)
    buf = PlanBuffers(CAPACITY_ROWS, SLOTS, bank.count, bank.dims, 'cuda:0', prefill=True)
    buf.pick[:rows].copy_(torch.tensor(picks, dtype=torch.int32, device='cuda'))
    buf.route(rows)
    bank.run_plan(x, buf, prefill=True)
    base = _arguments(bank, buf, rows)
    members = pair_members(picks)
    require(base[8].cpu().tolist() == members, 'actual full plan differs from independent expert/flat pair order')
    inverse = [0] * len(members)
    for pair, member in enumerate(members):
        inverse[member] = pair
    output = torch.empty_like(x)
    fields = bank.fields
    down = (base[9], base[10], bank.native_down, fields[3], fields[5], base[6], base[7], base[8],
            buf.y[:rows], bank.dims, bank.width, 1, SLOTS, False, 16)
    hashes = {name: hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
              for name, value in (('x', x), ('weights', weights), ('picks', buf.pick[:rows]))}
    return Owner(bank_index, bank, buf, x, weights, output, picks, members, inverse, base, down,
                 (buf.y[:rows], buf.pick[:rows], weights, output), hashes)


def _reference(torch, reader, prefix, owners):
    from tensorfold.cuda.nvfp4 import checkpoint
    from tensorfold.cuda.nvfp4.linear import Fp4Linear
    from tests.nvfp4_a4_inputs import ct_quant_reference

    pending = []
    for owner in owners:
        rows = len(owner.picks)
        pending.append((torch.empty((1, rows * SLOTS, owner.bank.width // 2), dtype=torch.uint8),
                        torch.empty((1, rows * SLOTS, owner.bank.width // 16), dtype=torch.uint8),
                        torch.empty((rows, SLOTS, owner.bank.dims), dtype=torch.bfloat16),
                        [False] * (rows * SLOTS)))
    selected = sorted({expert for owner in owners for row in owner.picks for expert in row})
    for expert in selected:
        projections = []
        raw_down = None
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            name = f'{prefix}.{expert}.{projection}'
            weight = float(reader.get(name + '.weight_global_scale')[0])
            inverse = float(reader.get(name + '.input_global_scale')[0])
            projections.append(Fp4Linear.from_checkpoint(reader.get(name + '.weight_packed', 'cuda:0'),
                reader.get(name + '.weight_scale', 'cuda:0'), 1. / weight, act=1. / inverse, act_inverse=inverse))
            if projection == 'down_proj':
                raw_down = inverse
        gate, up, down = projections
        for owner, (codes, scales, y, written) in zip(owners, pending, strict=True):
            for row, picks in enumerate(owner.picks):
                for slot, selected_expert in enumerate(picks):
                    if selected_expert != expert:
                        continue
                    g = checkpoint.prompt(checkpoint.A4, owner.x[row:row + 1], gate)
                    u = checkpoint.prompt(checkpoint.A4, owner.x[row:row + 1], up)
                    hidden = torch.nn.functional.gelu(g, approximate='tanh') * u
                    c, s = ct_quant_reference(hidden.cpu(), raw_down)
                    pair = owner.inverse[row * SLOTS + slot]
                    codes[0, pair].copy_(c[0])
                    scales[0, pair].copy_(s[0])
                    y[row, slot].copy_(checkpoint.prompt(checkpoint.A4, hidden, down)[0].cpu())
                    written[row * SLOTS + slot] = True
        del gate, up, down, projections
    for owner, (codes, scales, y, written) in zip(owners, pending, strict=True):
        require(all(written), 'independent oracle did not initialize every routed pair')
        actual_y = y.to('cuda:0')
        out = torch.zeros_like(owner.x)
        for row, picks in enumerate(owner.picks):
            for slot in sorted(range(SLOTS), key=lambda selected_slot: picks[selected_slot]):
                term = (actual_y[row, slot].float() * owner.weights[row, slot]).to(torch.bfloat16)
                out[row] = out[row] + term
        require(bool(torch.isfinite(out).all()), 'synthetic diagnostic reference is nonfinite')
        owner.expected = codes, scales, y, out.cpu()


def _same(torch, left, right):
    if left.dtype == torch.bfloat16:
        return torch.equal(left.view(torch.int16), right.view(torch.int16))
    return torch.equal(left, right)


class OutputMismatchError(RuntimeError):
    """Completed CPU copies stay owned if diagnostic publication exhausts."""
    def __init__(self, owner, field, actual, expected):
        super().__init__('component output differs from the independent oracle')
        self.owner, self.field, self.actual, self.expected = owner, field, actual, expected


def validate_owners(torch, owners):
    for index, owner in enumerate(owners):
        require(owner.expected is not None, 'independent expectations were not prepared')
        actual = (owner.up_base[9], owner.up_base[10], owner.down_args[8], owner.output)
        for field, value, expected in zip(('codes', 'scales', 'slots', 'combined'), actual, owner.expected, strict=True):
            copied = value.cpu()
            if not _same(torch, copied, expected):
                raise OutputMismatchError(index, field, copied, expected)


def poison_owners(owners):
    for owner in owners:
        owner.up_base[9].fill_(0xa5)
        owner.up_base[10].fill_(0x5a)
        owner.down_args[8].fill_(float('nan'))
        owner.output.fill_(float('nan'))


def _macro(operations, down, combine, dims):
    for up, args, owner in operations:
        up(*args)
        down(*owner.down_args)
        combine(*owner.combine_args, N=dims, R=SLOTS, RS=SLOTS, BLOCK=256,
                num_warps=4, enable_fp_fusion=False)


def bindings(ext, owners):
    result = {}
    for name in (PUBLIC, *ARMS):
        operations = []
        for owner in owners:
            base = owner.up_base
            if name == PUBLIC:
                entry = ext.fused_up
                args = (*base[:11], base[11], base[12], 1, SLOTS, base[14], False, 16, 1)
            else:
                warps, stages = CONFIGS[ARMS.index(name)]
                entry = ext.gemma_fused_up
                args = (*base, warps, stages)
            operations.append((entry, args, owner))
        result[name] = tuple(operations)
    return result


def _owner_tensors(owner):
    buf = owner.buf
    values = [owner.x, owner.weights, owner.output, buf.pick, buf.y, buf.plan.members, buf.plan.items,
              buf.plan.counts, buf.plan.rank, buf.plan.hist, *buf.a4_scratch[(owner.bank.dims, owner.bank.width)]]
    if hasattr(buf, 'a4_row_plan'):
        values.append(buf.a4_row_plan)
    return values


def retained_bytes(banks, owners):
    # Storage addresses are only temporary live-object alias keys, never
    # persistent identities or an assertion of cross-process physical VMA.
    observed = {}
    for tensor in [*(value for _prefix, bank in banks for value in bank.fields),
                   *(value for owner in owners for value in _owner_tensors(owner))]:
        storage = tensor.untyped_storage()
        observed[(str(tensor.device), storage.data_ptr(), storage.nbytes())] = storage.nbytes()
    return sum(observed.values())


def _capacity_bound(bank_count, rows, owner_bank_count=None):
    from tensorfold.cuda.experts import max_items
    from tensorfold.cuda.nvfp4.host_layout import authority_fields_a4, plan_scratch_bytes_a4

    bank = sum(math.prod(shape) * (1 if dtype == 'U8' else 4)
               for shape, dtype in authority_fields_a4(EXPERTS, WIDTH, DIMS))
    pairs = CAPACITY_ROWS * SLOTS
    # Rank/hist terms conservatively bound both planner policies; this is a
    # requested retained-tensor bound, not CUDA/runtime/graph peak admission.
    plan = 4 * (pairs + 3 * max_items(pairs, EXPERTS) + 2 + pairs + pairs * EXPERTS)
    owner = (4 * pairs + plan + CAPACITY_ROWS * SLOTS * DIMS * 2
             + plan_scratch_bytes_a4(pairs, WIDTH, DIMS) + CAPACITY_ROWS * 3 * 4
             + rows * DIMS * 4 + rows * SLOTS * 4)
    owner_bank_count = bank_count if owner_bank_count is None else owner_bank_count
    return bank_count * bank + owner_bank_count * ROUTE_SETS * owner


def _memory(torch, args):
    def metric(path):
        with _stream(path) as source:
            raw = source.read((1 << 20) + 1)
        require(len(raw) <= 1 << 20, 'resource metadata exceeds its bound')
        value = raw.decode('ascii').strip()
        return None if value == 'max' else int(value)

    with _stream('/proc/meminfo') as source:
        raw = source.read((1 << 20) + 1)
    require(len(raw) <= 1 << 20, 'host memory metadata exceeds its bound')
    available = next(int(line.split()[1]) * 1024 for line in raw.decode('ascii').splitlines()
                     if line.startswith('MemAvailable:'))
    maximum = metric('/sys/fs/cgroup/memory.max')
    used = metric('/sys/fs/cgroup/memory.current')
    require(type(used) is int and used >= 0 and (maximum is None or maximum > 0),
            'cgroup memory metadata is invalid')
    if maximum is not None:
        available = min(available, maximum - used)
    free, total = torch.cuda.mem_get_info()
    require(available >= args.min_host_available_bytes and free >= args.min_device_free_bytes,
            'declared host/device free-memory floor is not satisfied')
    return {'host_available_bytes': available, 'cgroup_max_bytes': maximum, 'cgroup_current_bytes': used,
            'cgroup_peak_bytes': metric('/sys/fs/cgroup/memory.peak'), 'device_free_bytes': free,
            'device_total_bytes': total, 'torch_allocated_bytes': torch.cuda.memory_allocated(),
            'torch_reserved_bytes': torch.cuda.memory_reserved(), 'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
            'torch_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
            'scope': 'Endpoints plus cumulative run allocator/cgroup counters; not a complete GPU/driver/process peak guarantee.'}


def _cell(torch, ext, banks, reader, rows, locality, mode, stream, args, result):
    from tensorfold.families.gemma4.cuda.nvfp4_experts import _combine

    graph_objects = [None] * 7
    statuses = [None] * 8
    owners, graph_info = [], {}
    primary = prior_cause = prior_context = None
    prior_suppress = False
    retirement_started = None
    try:
        selected_banks = banks[:1] if locality == 'hot' else banks
        require(_capacity_bound(len(banks), rows, len(selected_banks)) <= args.retained_device_tensor_budget_bytes,
                'declared retained-tensor capacity budget is insufficient before cell allocation')
        started = time.perf_counter_ns()
        for bank_index, (prefix, bank) in enumerate(selected_banks):
            group = [prepare_owner(torch, bank_index, bank, rows, route_index, locality, args.seed)
                     for route_index in range(ROUTE_SETS)]
            owners.extend(group)
            _reference(torch, reader, prefix, group)
        require(len({id(owner.buf) for owner in owners}) == len(owners) == len(selected_banks) * ROUTE_SETS,
                'independent per-bank/per-route owners are required')
        macro_bindings = bindings(ext, owners)
        grid = (rows, (DIMS + 255) // 256)
        down_call, combine_call = ext.down_plan, _combine[grid]
        calls = {}
        for name in (PUBLIC, *ARMS):
            operations = macro_bindings[name]
            def call(operations=operations):
                _macro(operations, down_call, combine_call, DIMS)
            call()
            stream.synchronize()
            validate_owners(torch, owners)
            calls[name] = call
        result['owner_preparation_ns'] = time.perf_counter_ns() - started
        result['retained_device_tensor_bytes'] = retained_bytes(banks, owners)
        require(retained_bytes(banks, owners) <= args.retained_device_tensor_budget_bytes,
                'actual retained tensor capacity exceeds the declared budget')
        result['memory_before_capture'] = _memory(torch, args)
        started = time.perf_counter_ns()
        if mode == 'graph':
            for index, name in enumerate((PUBLIC, *ARMS)):
                # The public retained-template mode permits direct metadata
                # queries without installing any capture/replay/model hook.
                # Its real host/template footprint is part of this diagnostic.
                graph = torch.cuda.CUDAGraph(keep_graph=True, num_rng_states=0)
                graph_objects[index] = graph
                with _OwnedContext(torch.cuda.graph(graph, stream=stream)):
                    calls[name]()
                graph.instantiate()
                data = graph.get_graph_data()
                graph_info[name] = {'nodes': len(data['nodes']), 'edges': len(data['edges']),
                                    'node_types': [node['node_type'] for node in data['nodes']],
                                    'raw_template_retained': True}
                require(graph_info[name]['nodes'] >= len(owners) * 3,
                        'actual graph node count omits component work')
                require(not any('child' in str(kind).lower() or 'conditional' in str(kind).lower()
                                for kind in graph_info[name]['node_types']),
                        'nested/conditional graph work is outside this flat component protocol')
                graph.replay()
                stream.synchronize()
                validate_owners(torch, owners)
                calls[name] = graph.replay
        result['graph_preparation_ns'] = time.perf_counter_ns() - started
        result['graphs'] = graph_info
        started = time.perf_counter_ns()
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record(stream)
        end.record(stream)
        end.synchronize()  # Create deferred event resources outside the window.
        result['event_preparation_ns'] = time.perf_counter_ns() - started
        result['memory_after_capture'] = _memory(torch, args)
        result['status'] = 'RUNNING'
        result['inputs'] = [{'bank_index': owner.bank_index, 'route_set': index % ROUTE_SETS,
                             'picks': owner.picks, 'members': owner.members, 'sha256': owner.input_hashes}
                            for index, owner in enumerate(owners)]
        for block, order in enumerate(schedule()):
            for position, name in enumerate(order):
                call = calls[name]
                poison_owners(owners)
                stream.synchronize()
                started = time.perf_counter_ns()
                begin.record(stream)
                call()
                end.record(stream)
                end.synchronize()
                elapsed = time.perf_counter_ns() - started
                device_ms = begin.elapsed_time(end)
                require(type(elapsed) is int and elapsed > 0 and math.isfinite(device_ms) and device_ms >= 0,
                        'component clock observation is invalid')
                sample = {'block': block, 'position': position, 'arm': name, 'status': 'PENDING_VALIDATION',
                    'macro_host_settled_ns': elapsed, 'transactions': len(owners),
                    'host_ns_per_transaction': elapsed / len(owners), 'cuda_event_macro_ms': device_ms}
                result['samples'].append(sample)
                validate_owners(torch, owners)
                sample['status'] = 'PASS'
            _memory(torch, args)
        result['owner_count'] = len(owners)
        result['actual_pair_unions'] = [len({expert for owner in owners if owner.bank_index == index
                                          for row in owner.picks for expert in row}) for index in range(len(selected_banks))]
        result['route_set_unions_by_bank'] = [[len({expert for row in owner.picks for expert in row})
                                              for owner in owners if owner.bank_index == index]
                                             for index in range(len(selected_banks))]
        result['memory_after_samples'] = _memory(torch, args)
        for owner in owners:
            actual_hashes = {name: hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                             for name, value in (('x', owner.x), ('weights', owner.weights),
                                                 ('picks', owner.buf.pick[:rows]))}
            require(actual_hashes == owner.input_hashes, 'component mutated immutable prepared input')
        require(len(result['samples']) == 96, 'complete fixed cell sample count required')
        retirement_started = time.perf_counter_ns()
    except BaseException as error:  # noqa: BLE001 - retain operation/cancellation through independent retirement.
        primary = error
        prior_cause = BaseException.__cause__.__get__(error)
        prior_context = BaseException.__context__.__get__(error)
        prior_suppress = BaseException.__suppress_context__.__get__(error)
        result['status'] = 'FAILED'
    # No new cleanup iterator/callback list is allocated after a failure.
    settled = False
    try:
        stream.synchronize()
        settled = True
    except BaseException as error:  # noqa: BLE001 - unproved device completion retains graph/tensor owners in this frame.
        statuses[0] = error
        if primary is None:
            primary = error
            prior_cause = BaseException.__cause__.__get__(error)
            prior_context = BaseException.__context__.__get__(error)
            prior_suppress = BaseException.__suppress_context__.__get__(error)
    if settled:
        index = 0
        while index < 7:
            if graph_objects[index] is not None:
                try:
                    graph_objects[index].reset()
                except BaseException as error:  # noqa: BLE001 - one graph failure cannot skip another retirement.
                    statuses[1 + index] = error
                    if primary is None:
                        primary = error
                        prior_cause = BaseException.__cause__.__get__(error)
                        prior_context = BaseException.__context__.__get__(error)
                        prior_suppress = BaseException.__suppress_context__.__get__(error)
            index += 1
    if primary is not None:
        result['status'] = 'FAILED'
    try:
        _raise_retired(primary, prior_cause, prior_context, prior_suppress, statuses)
    except BaseException as transport:  # Even helper-frame exhaustion retains the owning primary.
        if primary is not None and transport is not primary:
            raise primary from transport
        raise
    result['status'] = 'COMPLETE'
    result['retirement'] = {'stream_settled': settled, 'graph_count': len(graph_info), 'reset_completed': True}
    result['retirement_ns'] = time.perf_counter_ns() - retirement_started
    return result


def _runtime(args):
    import torch

    from tests.nvfp4_a4_inputs import ct_quant_reference

    require(platform.system() == 'Linux' and sys.byteorder == 'little', 'this diagnostic requires native little-endian Linux CUDA')
    require(Path(ct_quant_reference.__code__.co_filename).resolve()
            == Path(__file__).resolve().parents[1] / 'tests/nvfp4_a4_inputs.py',
            'the independent CPU oracle must come from this canonical verification source tree')
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(args.inter_op_threads)
    require(torch.get_num_threads() == args.threads and torch.get_num_interop_threads() == args.inter_op_threads,
            'actual Torch thread getters differ from the declared region')
    workers = os.environ.get('MAX_JOBS')
    require(workers is not None and workers.isdecimal() and 0 < int(workers) <= len(os.sched_getaffinity(0)),
            'bounded compiler MAX_JOBS must be declared by the actor before preparation')
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1
            and torch.cuda.get_device_capability()[0] == 12, 'one selected native SM12.x device required')
    require(torch.cuda.memory.get_allocator_backend() == 'native', 'this fixed component region requires the native allocator')
    require(all(callable(getattr(torch.cuda.CUDAGraph, name, None))
                for name in ('instantiate', 'get_graph_data', 'reset')),
            'pinned supported graph metadata/lifetime APIs required; no alternate graph path is selected')
    return torch


def run(args, report):
    started = time.perf_counter_ns()
    torch = _runtime(args)
    report['runtime_preparation_ns'] = time.perf_counter_ns() - started
    reader = stream = None
    statuses = [None] * 2
    primary = prior_cause = prior_context = None
    prior_suppress = False
    try:
        report['memory_before_authority'] = _memory(torch, args)
        started = time.perf_counter_ns()
        authority = _authority(args)
        report['authority_preparation_ns'] = time.perf_counter_ns() - started
        report['checkpoint_authority'] = authority
        report['runtime'] = {'python': sys.version, 'torch': torch.__version__, 'torch_git': torch.version.git_version,
            'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
            'capability': list(torch.cuda.get_device_capability()), 'threads': torch.get_num_threads(),
            'inter_op_threads': torch.get_num_interop_threads(), 'affinity': sorted(os.sched_getaffinity(0)),
            'allocator': torch.cuda.memory.get_allocator_backend(), 'max_jobs': os.environ.get('MAX_JOBS')}
        require(args.threads <= len(report['runtime']['affinity']), 'declared threads exceed actor CPU affinity')
        require(2 * 428215296 <= args.retained_device_tensor_budget_bytes,
                'declared retained-tensor budget cannot hold two original banks')
        report['memory_before_loading'] = _memory(torch, args)
        stream = torch.cuda.Stream()
        with _OwnedContext(torch.cuda.stream(stream)), _OwnedContext(torch.inference_mode()):
            report['timing_context'] = {'inference_mode': torch.is_inference_mode_enabled(),
                                        'grad_enabled': torch.is_grad_enabled()}
            started = time.perf_counter_ns()
            reader, config, identities = _open_checkpoint(args, torch, authority)
            report['checkpoint_authentication_and_headers_ns'] = time.perf_counter_ns() - started
            report['checkpoint_identities'] = identities
            from tensorfold.cuda.nvfp4.grouped_a4 import _ext

            started = time.perf_counter_ns()
            ext = _ext()  # Naturally needed provider; never force an unused factory for observation.
            report['provider_preparation_ns'] = time.perf_counter_ns() - started
            report['provider'] = {'path': ext.__file__, 'sha256': _digest(Path(ext.__file__)),
                                  'scope': 'Named normal returned module only; real VMA identity belongs to the actor observer.'}
            for locality in ('hot', 'scan'):
                layers = args.layers[:1] if locality == 'hot' else args.layers
                phase = {'locality': locality, 'layers': layers, 'status': 'LOADING'}
                report['bank_loading_phases'].append(phase)
                started = time.perf_counter_ns()
                banks, scalars = _load_banks(reader, config, layers)
                stream.synchronize()
                phase['loading_ns'] = time.perf_counter_ns() - started
                phase['calibration_sha256'] = scalars
                phase['banks'] = [{'layer': layer, 'prefix': prefix, 'count': bank.count, 'width': bank.width,
                                   'dims': bank.dims, 'shared_input': bank.shared_input, 'bytes': bank.nbytes()}
                                  for layer, (prefix, bank) in zip(layers, banks, strict=True)]
                phase['memory_after_loading'] = _memory(torch, args)
                phase['status'] = 'COMPLETE'
                for rows in ROWS:
                    for mode in ('eager', 'graph'):
                        cell = {'rows': rows, 'locality': locality, 'mode': mode, 'status': 'PREPARING',
                                'samples': [], 'retirement_ns': None}
                        report['cells'].append(cell)
                        _cell(torch, ext, banks, reader, rows, locality, mode, stream, args, cell)
                        write_json(args.out, report)
                stream.synchronize()
                del banks
            require(len(report['cells']) == 16, 'all sixteen registered cells required')
            for row in identities.values():
                require(_stamp(Path(row['path'])) == row['stamp'], 'checkpoint identity changed during diagnostic')
            report['checkpoint_identities'] = identities
            report['payload_reads'] = {'requested_bytes': reader.reader.requested_bytes,
                                       'read_calls': reader.reader.read_calls,
                                       'policy': 'Logical attempted bytes, charged cumulatively before acquisition; not physical IO traffic.'}
    except BaseException as error:  # noqa: BLE001 - every operation/interruption root is retained until work retires.
        primary = error
        prior_cause = BaseException.__cause__.__get__(error)
        prior_context = BaseException.__context__.__get__(error)
        prior_suppress = BaseException.__suppress_context__.__get__(error)
    if stream is not None:
        try:
            stream.synchronize()
            report['run_retirement']['stream_settled'] = True
        except BaseException as error:  # noqa: BLE001 - device failure cannot skip independent reader retirement.
            statuses[0] = error
            if primary is None:
                primary = error
                prior_cause = BaseException.__cause__.__get__(error)
                prior_context = BaseException.__context__.__get__(error)
                prior_suppress = BaseException.__suppress_context__.__get__(error)
    if reader is not None:
        try:
            reader.close()
            report['run_retirement']['reader_close_completed'] = True
        except BaseException as error:  # noqa: BLE001 - retain the exact reader/root for diagnosis.
            statuses[1] = error
            if primary is None:
                primary = error
                prior_cause = BaseException.__cause__.__get__(error)
                prior_context = BaseException.__context__.__get__(error)
                prior_suppress = BaseException.__suppress_context__.__get__(error)
    try:
        _raise_retired(primary, prior_cause, prior_context, prior_suppress, statuses)
    except BaseException as transport:  # Preserve the owning root if the transport call cannot allocate.
        if primary is not None and transport is not primary:
            raise primary from transport
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--layers', type=layer_pair, required=True)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--max-input-bytes', type=positive, required=True)
    parser.add_argument('--max-payload-read-bytes', type=positive, required=True)
    parser.add_argument('--retained-device-tensor-budget-bytes', type=positive, required=True)
    parser.add_argument('--min-device-free-bytes', type=positive, required=True)
    parser.add_argument('--min-host-available-bytes', type=positive, required=True)
    parser.add_argument('--threads', type=positive, required=True)
    parser.add_argument('--inter-op-threads', type=positive, required=True)
    parser.add_argument('--seed', type=positive, default=1137)
    parser.add_argument('--synthetic-inputs', action='store_true', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    require(len(args.authority_sha256) == 64 and all(value in '0123456789abcdef' for value in args.authority_sha256),
            'exact checkpoint authority SHA required')
    require(not args.out.exists() and not args.out.is_symlink(), 'component output must be a fresh owning report path')
    report = {'schema': 1, 'status': 'RUNNING', 'region': 'real-original-banks-explicit-synthetic-input-diagnostic',
              'schedule': schedule(), 'rows': ROWS, 'route_sets': ROUTE_SETS, 'blocks': BLOCKS,
              'cell_order': [(rows, locality, mode) for locality in ('hot', 'scan') for rows in ROWS for mode in ('eager', 'graph')],
              'private_configs': CONFIGS, 'reference': PUBLIC, 'seed': args.seed, 'layers': args.layers,
              'core': 'prepared codes/scales -> checked up/GeGLU/down-quant -> down -> ordered combine',
              'outside_window': ['checkpoint authentication/loading', 'input quantization', 'JIT/provider preparation',
                                 'independent expectations', 'poisoning/validation', 'graph/events/setup/teardown'],
              'resource_policy': {name: getattr(args, name) for name in
                  ('max_input_bytes', 'max_payload_read_bytes', 'retained_device_tensor_budget_bytes',
                   'min_device_free_bytes', 'min_host_available_bytes')},
              'bank_loading_phases': [], 'cells': [],
              'run_retirement': {'stream_settled': False, 'reader_close_completed': False},
              'graph_policy': 'Seven keep_graph=True templates/executables per graph cell; explicit instantiate/get_graph_data before timing; no hooks. Metadata/template costs are included in setup/resource records, not claimed production-wrapper-equivalent.',
              'scope': 'Exploratory component diagnostic only; no serving selection, SM121/model/P3 gain or representative router distribution.'}
    write_json(args.out, report)  # Literal schedule and policy precede any timing.
    primary = prior_cause = prior_context = None
    prior_suppress = False
    publication = [None]
    try:
        run(args, report)
    except BaseException as error:  # noqa: BLE001 - publish a failed owning outcome after retirement, then rethrow exact root.
        primary = error
        prior_cause = BaseException.__cause__.__get__(error)
        prior_context = BaseException.__context__.__get__(error)
        prior_suppress = BaseException.__suppress_context__.__get__(error)
    try:
        report['status'] = 'COMPLETE' if primary is None else 'FAILED'
        if isinstance(primary, OutputMismatchError):
            def snapshot(value):
                import torch

                raw = value.contiguous().view(torch.uint8).numpy().tobytes()
                return {'dtype': str(value.dtype), 'shape': list(value.shape),
                        'sha256': hashlib.sha256(raw).hexdigest(), 'raw_little_endian_hex': raw.hex()}
            report['mismatch'] = {'owner': primary.owner, 'field': primary.field,
                                  'actual': snapshot(primary.actual), 'expected': snapshot(primary.expected)}
        write_json(args.out, report)
    except BaseException as error:  # noqa: BLE001 - publication failure cannot replace an operation/retirement root.
        publication[0] = error
        if primary is None:
            primary = error
            prior_cause = BaseException.__cause__.__get__(error)
            prior_context = BaseException.__context__.__get__(error)
            prior_suppress = BaseException.__suppress_context__.__get__(error)
    try:
        _raise_retired(primary, prior_cause, prior_context, prior_suppress, publication)
    except BaseException as transport:  # Raw primary/publication statuses remain in this owning frame.
        if primary is not None and transport is not primary:
            raise primary from transport
        raise
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
