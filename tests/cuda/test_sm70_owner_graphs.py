"""Real TP8 NCCL/native partial captures; full-model qualification is separate."""

from contextlib import contextmanager
import faulthandler
import json
import os
import socket
import sys
import time
import traceback
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cleanup import raise_failures as _raise_failures


def _diagnostic_phase(rank, head4, phase, rows=None):
    """Bounded test-only stderr records; diagnostic I/O cannot replace failure."""
    try:
        print(f"OWNER_GRAPH_PHASE pid={os.getpid()} rank={rank} head4={int(head4)} "
              f"phase={phase} rows={rows}", file=sys.stderr, flush=True)
    except BaseException:
        # O0/C6: this optional diagnostic has an explicit no-escape contract,
        # even for asynchronous BaseException. Worker-body cancellation and the
        # original parent/cohort cleanup remain authoritative and unchanged.
        pass


def _diagnostic_exception(rank, head4, error):
    """No-escape primary breadcrumb: at most 16 frames/16384 trace characters.

    No local values are captured. The original exception object is neither
    changed nor retained by the diagnostic after it returns. Cleanup and its
    existing failure policy still run exactly as before this optional report.
    """
    try:
        _diagnostic_phase(rank, head4, "worker.primary.exception." + type(error).__name__[:64])
        trace = traceback.TracebackException(type(error), error, error.__traceback__,
                                             limit=16, capture_locals=False)
        chunks, remaining, truncated = [], 16384, False
        for chunk in trace.format(chain=True):
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                truncated = True
                break
            chunks.append(chunk)
            remaining -= len(chunk)
            if remaining == 0:
                truncated = True
                break
        text = "".join(chunks)
        if truncated:
            text += "\n[primary traceback truncated at 16384 characters]\n"
        print(f"OWNER_GRAPH_PRIMARY pid={os.getpid()} rank={rank} head4={int(head4)}\n{text}",
              file=sys.stderr, flush=True)
    except BaseException:
        _diagnostic_phase(rank, head4, "worker.primary.breadcrumb.failed")


@contextmanager
def _worker_diagnostics(rank, head4):
    """This fresh spawned process owns one dump, at120s within the600s test.

    sys.stderr remains open/owned by the worker; no file descriptor is closed or
    replaced. Cancel only after the original worker's final cleanup unwinds.
    One nonrepeating schedule bounds dumps even if parent startup itself stalls.
    Setup/cancel/report plumbing is no-escape, including BaseException, so it
    cannot replace an authentic worker/cleanup result. Exceptions from yield's
    body are never caught here; original parent cancellation remains in force.
    """
    armed = False
    try:
        faulthandler.dump_traceback_later(120, repeat=False, file=sys.stderr, exit=False)
        armed = True
        _diagnostic_phase(rank, head4, "dump.armed.120s")
    except BaseException as error:
        _diagnostic_phase(rank, head4, "dump.start.failed."+type(error).__name__)
    try:
        yield
    finally:
        if armed:
            try:
                faulthandler.cancel_dump_traceback_later()
                _diagnostic_phase(rank, head4, "dump.cancelled.after_cleanup")
            except BaseException as error:
                _diagnostic_phase(rank, head4, "dump.cancel.failed."+type(error).__name__)



def _populate_compact_indexer(state, staging, raw, coordinates, weight, frequencies,
                              eps, sections, amplitude, committed):
    """Produce a real prefill span, pool it, then commit its accepted prefix.

    This setup frame is independent of the unchanged four-row native verify
    workspace. Span/staging ownership remains live until State reset/retirement;
    no frontier, generation, position or span field is assigned by the fixture.
    """
    # Continue the caller's original KV frame: never clear earlier sticky errors.
    expected = (state.slot_generation, state.history_generation)
    staging.begin(mtp=False)
    with staging.reading(mtp=False, borrowers=(raw, coordinates)):
        for ordinal, owner in enumerate(state.compact_indexer.entries):
            if owner is not None:
                entry = state.compact_indexer.append(ordinal, state.pos, raw, coordinates,
                    staging, 0, len(raw), mtp=False, expected=expected)
                entry.pool(state.pooled[ordinal], weight, frequencies, eps, sections, amplitude)
        state.set_pos(committed)


def _fixture_future_plan(weights, head4, workspace):
    """Tensor liabilities plus the existing Engine 2GiB workspace allowance.

    Qualification caller only. The allowance is not a library peak bound.
    """
    from tensorfold.families.qwen4_exp.cuda.history_placement import (
        compact_stage_bytes, for_weights, owner_workspace_bytes,
    )
    cfg, pair = weights.cfg, workspace.native_pair
    text = {name: getattr(cfg, name) for name in
            ('heads', 'head_dim', 'index_heads', 'index_dim', 'index_budget', 'index_ratio')}
    text['context'] = 257
    history = for_weights(weights).rank_bytes(0, 257, weights.meta['rank'], pair.row_bytes(1, 256),
        pair.dummy_bytes, index_dim=128, ratio=4, include_mtp=False, proposal_rows=4)
    # No linear layers/MTP/media in this fixture. Source State always owns the
    # generation/status/peer/three position words and PLE tail in addition.
    state = history + 16 + 8 + 8 * 2 * 4 + 3 * 4
    state += (cfg.ple_kernel - 1) * cfg.ngram_size * cfg.streams * cfg.hidden * 2
    values = 133 * 256
    population = (2 * values * 2 + 2 * values + 2 * (values // 32) * 2
                  + 133 * 128 * 2 + 133 * 3 * 8 + 128 * 4 + 4)
    temporary = max(owner_workspace_bytes(text, rows)['temporary_bytes'] for rows in (4, 133))
    return {'persistent': {'state': state, 'reference': workspace.nbytes if head4 else 0,
            'staging': compact_stage_bytes(133, 12), 'population': population},
        # At most three held clones plus one incoming replacement; native out is Half.
        'clone_peak': 4 * 4 * 3 * 256 * 2, 'project_temporary': temporary,
        'workspace': max(2 * 2**30, temporary),
        'workspace_authority': 'Existing Engine full-window 2GiB policy; qualification allowance, not worst-case bound.'}


def _fixture_remaining(plan, done=(), live_clones=0):
    if (type(live_clones) is not int or not 0 <= live_clones <= plan['clone_peak']
            or any(name not in plan['persistent'] for name in done)):
        raise ValueError('fixture phase exceeds its declared ownership plan')
    return (sum(value for name, value in plan['persistent'].items() if name not in done)
            + max(0, plan['clone_peak'] - live_clones) + plan['workspace'])


def _fixture_tensor_bytes(*owners):
    """Current project storage, counted once; host descriptor mirror excluded."""
    seen, total = set(), 0
    for owner in owners:
        values = ((None, owner),) if isinstance(owner, torch.Tensor) else vars(owner).items()
        for name, value in values:
            if name == 'descriptors_host' or not isinstance(value, torch.Tensor):
                continue
            storage = value.untyped_storage()
            key = (str(value.device), storage.data_ptr(), storage.nbytes())
            if key not in seen:
                seen.add(key)
                total += storage.nbytes()
    return total


def _fixture_collective(comm, send, receive, allowed, value, required):
    for index, item in enumerate((int(allowed), value, required)):
        if type(item) is not int or not 0 <= item < 2**63:
            raise ValueError('fixture grant fields must fit nonnegative signed64')
        send[index].fill_(item)
    comm.all_gather(send, receive)
    return receive.cpu().tolist()  # Existing collective completion before allocation.


def _fixture_check(funding, comm, send, receive, phase, done=(), live_clones=0, graph_extra=0, complete=False):
    from tensorfold.cuda.capacity import device_allocation_baseline
    paid = funding['fixed_tensor_bytes'] + funding['graph_charge']
    paid += sum(value for name, value in funding['plan']['persistent'].items() if name in done) + live_clones
    torch.cuda.synchronize()  # Caller is quiescent before the governing sample.
    sample = device_allocation_baseline(torch, preallocated_device_bytes=paid)
    required = 0 if complete else _fixture_remaining(funding['plan'], done, live_clones) + graph_extra
    incremental = paid - funding['fixed_tensor_bytes']
    case_remaining = funding['initial_usable'] - incremental
    available = min(sample['usable_serving_budget_bytes'], max(0, case_remaining))
    allowed = (case_remaining >= 0 and sample['physical_free_bytes'] >= sample['effective_device_floor_bytes']
               and available >= required)
    rows = _fixture_collective(comm, send, receive, allowed, available, required)
    funding['checks'].append({'phase': phase, 'physical_free': sample['physical_free_bytes'],
        'available': available, 'required_future': required, 'preallocated_project': paid,
        'torch_peak_allocated': int(torch.cuda.max_memory_allocated()),
        'torch_peak_reserved': int(torch.cuda.max_memory_reserved())})
    if not all(row[0] for row in rows):
        raise MemoryError('fixture caller grant no longer funds phase: ' + phase)
    return sample


def _fixture_admit(weights, head4, workspace, scratch, comm, send, receive):
    from tensorfold.cuda.capacity import device_allocation_baseline
    from tensorfold.families.qwen4_exp.cuda.service_memory import automatic_graph_bytes
    # Warm this exact preallocated control packet before the governing sample.
    _fixture_collective(comm, send, receive, True, 0, 0)
    fixed = _fixture_tensor_bytes(workspace, scratch, send, receive)
    plan = _fixture_future_plan(weights, head4, workspace)
    torch.cuda.synchronize()
    sample = device_allocation_baseline(torch, preallocated_device_bytes=fixed)
    future = _fixture_remaining(plan)
    local = automatic_graph_bytes(sample['usable_serving_budget_bytes'], future)
    rows = _fixture_collective(comm, send, receive, local > 0, local, future)
    if not all(row[0] for row in rows):
        raise MemoryError('fixture physical grant cannot fund capture and complete future liabilities')
    funding = {'plan': plan, 'fixed_tensor_bytes': fixed, 'graph_charge': 0, 'graph_charge_measured': False, 'checks': [],
        'graph_bytes': min(row[1] for row in rows), 'governing_free': sample['physical_free_bytes'],
        'initial_usable': sample['usable_serving_budget_bytes'],
        'device_floor': sample['effective_device_floor_bytes'], 'absolute_cap': sample['absolute_grant_cap_bytes']}
    before = _fixture_check(funding, comm, send, receive, 'before-capture', graph_extra=funding['graph_bytes'])
    funding['capture_free_before'] = before['physical_free_bytes']
    return funding


def _emit_fixture_funding(rank, head4, funding):
    raw = ('OWNER_FIXTURE_FUNDING ' + json.dumps({'schema': 'tensorfold.owner-fixture-funding-v1',
        'PID': os.getpid(), 'rank': rank, 'head4': bool(head4), **funding,
        'scope': 'Caller allowances and sampled/cached peaks; not complete physical peak bounds.'},
        sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > 4096:
        raise ValueError('fixture funding receipt exceeds4096bytes')
    if os.write(2, raw) != len(raw):
        raise OSError('fixture funding receipt write was incomplete')


def _fixture_case_id(head4, rn):
    return ('owner-rn-h1' if rn else 'owner') + '-head4-' + str(int(head4))


def _configure_rn_fixture(workspace):
    from tensorfold.families.qwen4_exp.cuda.owner_execution import RN_KERNEL_FAMILY
    workspace.configure_native_family(RN_KERNEL_FAMILY, heads_per_block=1)
    if (workspace.native_kernel_family != RN_KERNEL_FAMILY
            or type(workspace.native_heads_per_block) is not int or workspace.native_heads_per_block != 1):
        raise RuntimeError('RN fixture did not adopt its explicit H1 numerical family')


def _check_rn_graph_identity(workspace, graphs):
    from tensorfold.families.qwen4_exp.cuda.owner_execution import RN_KERNEL_FAMILY, RN_GRAPH_FAMILY
    record = graphs.record()
    if (workspace.native_kernel_family != RN_KERNEL_FAMILY
            or type(workspace.native_heads_per_block) is not int or workspace.native_heads_per_block != 1
            or not graphs.prepared or graphs.kernel_family != RN_KERNEL_FAMILY
            or type(graphs.heads_per_block) is not int or graphs.heads_per_block != 1
            or record['kernel_family'] != RN_KERNEL_FAMILY or record['graph_family'] != RN_GRAPH_FAMILY
            or type(record['rn_heads_per_block']) is not int or record['rn_heads_per_block'] != 1):
        raise RuntimeError('prepared RN graph numerical family/head identity differs')


def _fixture_reports(rank, head4, funding, graphs, workspace, weights, attempted, errors, rn=False):
    """Attempt each independent sink once; journal errors without escaping."""
    if funding is not None and not attempted[0]:
        attempted[0] = True  # Publication precedes every fallible callback.
        try:
            _emit_fixture_funding(rank, head4, funding)
        except BaseException as error:
            errors[0] = error
    if graphs is not None and not attempted[1]:
        attempted[1] = True
        try:
            if rn:
                _emit_graph_resource_context(rank, head4, graphs, workspace, weights,
                    reference_required=True, case_id=_fixture_case_id(head4, True))
            else:
                _emit_graph_resource_context(rank, head4, graphs, workspace, weights)
        except BaseException as error:
            errors[1] = error


def _graph_resource_context(rank, head4, graphs, workspace, weights, reference_required=None, case_id=None):
    """Stored Python metadata only: no free-memory query, fence or device work."""
    from tensorfold.families.qwen4_exp.cuda.history_placement import (
        compact_stage_bytes, for_weights,
    )

    record = graphs.record()  # Existing fields/numel metadata; no CUDA query.
    reference_required = head4 if reference_required is None else reference_required
    pair = workspace.native_pair
    layout = for_weights(weights)
    future = {
        'scope': 'Complete later fixture schedule; tensor plans are not physical/allocator bounds.',
        'state': {'logical_context': 257, 'verify_rows': 4, 'target_layers': 12, 'mtp': False,
            'history_tensor_plan_bytes': layout.rank_bytes(0, 257, rank, pair.row_bytes(1, 256),
                pair.dummy_bytes, index_dim=128, ratio=4, include_mtp=False, proposal_rows=4),
            'other': 'PLE tail, position/status/generation scalars and host metadata remain additional.'},
        'reference_owner': {'enabled': bool(reference_required), 'rows': 4, 'native_comm_world': 8,
            'same_layout_existing_tensor_bytes': workspace.nbytes if reference_required else 0,
            'scope': 'Same owner tensor layout; allocator/transport cost is not measured or inferred.'},
        'compact_stage': {'rows': 133, 'caches': 12, 'tensor_plan_bytes': compact_stage_bytes(133, 12)},
        'population': {'kv_half_shapes': [2, 133, 1, 256], 'raw_indexer_half_shape': [133, 128],
            'coordinate_i64_shape': [133, 3], 'indexer_scale_f32': 128, 'frequency_f32': 1,
            'encoded_KV': 'Existing pair encoders, scales and their bounded working intermediates.'},
        'later_work': 'Pool/encode workspace; eager/replay/reference output clones for rows1,3,4,1; reset/stale replay.',
        'unmeasured': 'Allocator/CUDA/NCCL lazy allocations and physical peaks are not bounded by these tensor plans.'}
    return {'schema': 'tensorfold.owner-graph-resource-context-v1', 'PID': os.getpid(), 'rank': rank,
        'head4': bool(head4), **({'case_id': case_id} if case_id is not None else {}), 'graph_record': record, 'workspace_tensor_bytes': workspace.nbytes,
        'capture_transport_world': workspace.native_comm.world, 'vote_world': graphs.comm.world,
        'remaining_capture_rows': [rows for rows in graphs.buckets if rows not in graphs.graphs],
        'prepare_start_free': None, 'prepare_start_free_reason': 'Not retained by existing producer fields.',
        'future_schedule': future, 'scope': 'Stored producer values at the success/failure boundary, not a worst-case bound.'}


def _emit_graph_resource_context(rank, head4, graphs, workspace, weights, reference_required=None, case_id=None):
    if reference_required is None and case_id is None:
        context = _graph_resource_context(rank, head4, graphs, workspace, weights)
    else:
        context = _graph_resource_context(rank, head4, graphs, workspace, weights, reference_required, case_id)
    raw = ('OWNER_GRAPH_RESOURCES ' + json.dumps(context, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')
    if len(raw) > 4096:
        raise ValueError('owner graph diagnostic exceeds its one-line 4096-byte extent')
    # One bounded write preserves a single pipe record across spawned ranks.
    # Short/error writes are explicit diagnostic failures; never retry/replace primary.
    if os.write(2, raw) != len(raw):
        raise OSError('owner graph resource diagnostic write was incomplete')


def _retire_worker_resources(rank, head4, graphs, workspace, reference, head_comm, comm, *, failed):
    """Retire dependency owners before native communicator release, once only.

    Captured NCCL graph references must die before NCCL destroy/abort. A failed
    graph close retains every dependent owner. Workspace failures retain both
    communicators; a failed head release retains its borrowed parent Store.
    Returned False requires terminal process containment, not Python teardown.
    """
    errors = []
    if graphs is not None:
        try:
            _diagnostic_phase(rank, head4, 'graphs.close.before')
            graphs.close()
            if graphs.closed is not True:
                raise RuntimeError('graph owner did not prove retirement')
            _diagnostic_phase(rank, head4, 'graphs.close.after')
        except BaseException as error:
            errors.append(error)
            return errors, False
    for label, owner in (('owner', workspace), ('reference', reference)):
        if owner is not None:
            try:
                _diagnostic_phase(rank, head4, label + '.close.before')
                owner.close()
                if owner.closed is not True:
                    raise RuntimeError('workspace owner did not prove retirement')
                _diagnostic_phase(rank, head4, label + '.close.after')
            except BaseException as error:
                errors.append(error)
    if errors:
        return errors, False
    if head_comm is not None:
        try:
            _diagnostic_phase(rank, head4, 'head_comm.close.before')
            head_comm.close(abort=failed)
            if head_comm.retired is not True:
                raise RuntimeError('head communicator did not prove retirement')
            _diagnostic_phase(rank, head4, 'head_comm.close.after')
        except BaseException as error:
            errors.append(error)
            return errors, False
    try:
        _diagnostic_phase(rank, head4, 'comm.close.before')
        comm.close(abort=failed)
        if comm.retired is not True:
            raise RuntimeError('communicator did not prove retirement')
        _diagnostic_phase(rank, head4, 'comm.close.after')
    except BaseException as error:
        errors.append(error)
        return errors, False
    return errors, True


def _worker_outcome(rank, head4, primary, errors, retired, diagnostics=None):
    try:
        if diagnostics is not None:
            try:
                _raise_failures(primary, diagnostics)
            except BaseException as combined:
                primary = combined
        _raise_failures(primary, errors)
    except BaseException as error:
        if retired:
            raise
        if primary is not None and error is not primary:
            _diagnostic_exception(rank, head4, primary)
        _diagnostic_exception(rank, head4, error)
    finally:
        if not retired:
            _diagnostic_phase(rank, head4, 'terminal.unretired.owners')
            os._exit(1)  # Caller frames retain owners; no destructor or pointer retry.


def _finish_worker(rank, head4, primary, graphs, workspace, reference, staging, head_comm, comm, *, failed, diagnostic=None, report_errors=None):
    # Arguments and the suspended worker frame retain every owner, including
    # staging, without allocating a separate retention tuple during exhaustion.
    try:
        diagnostics = [diagnostic] if diagnostic is not None else []
        if report_errors is not None:
            diagnostics.extend(error for error in report_errors if error is not None)
        diagnostics = diagnostics or None
        errors, retired = _retire_worker_resources(rank, head4, graphs, workspace,
            reference, head_comm, comm, failed=failed)
    except BaseException as error:
        # Includes preparation/list/tuple allocation failure inside retirement.
        try:
            if primary is not None:
                _diagnostic_exception(rank, head4, primary)
            if diagnostic is not None and diagnostic is not error:
                _diagnostic_exception(rank, head4, diagnostic)
            if report_errors is not None:
                for reported in report_errors:
                    if reported is not None and reported is not error and reported is not primary:
                        _diagnostic_exception(rank, head4, reported)
            _diagnostic_exception(rank, head4, error)
        finally:
            _diagnostic_phase(rank, head4, 'terminal.cleanup.preparation.failed')
            os._exit(1)
    else:
        _worker_outcome(rank, head4, primary, errors, retired, diagnostics)


def _worker(rank, port, head4, rn=False):
    with _worker_diagnostics(rank, head4):
        _diagnostic_phase(rank, head4, "imports.before")
        from tensorfold.cuda.comm import NCCL
        from tensorfold.families.qwen4_exp.cuda.attention import AttnScratch
        from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers, _encode
        from tensorfold.families.qwen4_exp.cuda.owner_graphs import OwnerAttentionGraphs
        from tensorfold.families.qwen4_exp.cuda.owner_execution import FULL_COLLECTIVE, HEAD_COLLECTIVE
        from tensorfold.families.qwen4_exp.cuda.state import State
        from tensorfold.families.qwen4_exp.kv_formats import get_pair

        _diagnostic_phase(rank, head4, "imports.after")
        _diagnostic_phase(rank, head4, "threads.before")
        torch.set_num_threads(1)
        _diagnostic_phase(rank, head4, "threads.after")
        _diagnostic_phase(rank, head4, "device.before")
        torch.cuda.set_device(rank)
        _diagnostic_phase(rank, head4, "device.after")
        _diagnostic_phase(rank, head4, "comm.construct.before")
        report_attempted = [False, False]
        report_errors = [None, None]
        comm, head_comm, failed, graphs = NCCL(), None, False, None
        workspace = reference = staging = funding = None
        primary = None
        weights = None
        reference_required = head4 or rn
        _diagnostic_phase(rank, head4, "comm.construct.after")
        try:
            _diagnostic_phase(rank, head4, "comm.open.before")
            comm.open(rank, 8, "127.0.0.1", port, timeout_s=90, key="tf_owner_graph_component")
            _diagnostic_phase(rank, head4, "comm.open.after")
            if head4:
                _diagnostic_phase(rank, head4, "head_comm.open.before")
                head_comm = NCCL()
                head_comm.open(rank % 4, 4, "", 0, timeout_s=90, store=comm.store,
                               key="tf_owner_graph_head/" + str(rank // 4))
                _diagnostic_phase(rank, head4, "head_comm.open.after")
            device = torch.device("cuda", rank)
            cfg = NS(hidden=512, streams=4, conv_kernel=4, conv_dim=1024, nk=2, nv=4, dk=128, dv=128,
                     kv_heads=1, head_dim=256, index_dim=128, index_ratio=4, ple_kernel=4, ngram_size=3,
                     ple_layers=[], eos=(0,), heads=3, index_heads=4, index_budget=8, eps=1e-6,
                     mrope_section=(1, 1, 1), rope_attention_factor=1.0, execution_dtype="float16")
            weights = NS(cfg=cfg, device=device, layers=[NS(index=i, linear=False) for i in range(12)],
                         mtp=None, comm=comm, meta={"world": 8, "rank": rank, "full": NS(kv_heads=2),
                                                   "execution_provider": "sm70", "history_layout": "unique",
                                                   "attention_degree": 1})
            _diagnostic_phase(rank, head4, "workspace.construct.before")
            scratch = AttnScratch(4, 3, 256, 257, device, budget=8, ratio=4, dtype=torch.float16)
            workspace = object.__new__(OwnerBuffers)
            workspace.__init__(weights, 4, scratch)
            _diagnostic_phase(rank, head4, "workspace.construct.after")
            pair = get_pair("int8")
            _diagnostic_phase(rank, head4, "owner.prepare_native.before")
            workspace.prepare_native(pair=pair, page_tokens=128)
            _diagnostic_phase(rank, head4, "owner.prepare_native.after")
            if rn:
                _configure_rn_fixture(workspace)
            _diagnostic_phase(rank, head4, "owner.transport.before")
            workspace.configure_native_transport(HEAD_COLLECTIVE if head4 else FULL_COLLECTIVE, head_comm or comm)
            _diagnostic_phase(rank, head4, "owner.transport.after")
            budget_send = torch.empty((3,), dtype=torch.int64, device=device)
            budget_receive = torch.empty((8, 3), dtype=torch.int64, device=device)
            funding = _fixture_admit(weights, reference_required, workspace, scratch, comm, budget_send, budget_receive)
            funding['case_id'] = _fixture_case_id(head4, rn)
            _diagnostic_phase(rank, head4, "graphs.construct.before")
            graphs = object.__new__(OwnerAttentionGraphs)
            graphs.__init__(workspace, rows=4, budget_bytes=funding['graph_bytes'], comm=comm)
            _diagnostic_phase(rank, head4, "graphs.construct.after")
            # The actual startup capture has no target weights or State cache yet.
            _diagnostic_phase(rank, head4, "graphs.prepare.before")
            graphs.prepare(quiescent=lambda: workspace.native_states is None)
            _diagnostic_phase(rank, head4, "graphs.prepare.after")
            if rn:
                _check_rn_graph_identity(workspace, graphs)
            from tensorfold.cuda.capacity import device_allocation_baseline
            torch.cuda.synchronize(device)
            prepared = device_allocation_baseline(torch, preallocated_device_bytes=funding['fixed_tensor_bytes'])
            funding['graph_charge'] = max(0, funding['capture_free_before'] - prepared['physical_free_bytes'])
            funding['graph_charge_measured'] = True
            preparation_votes = _fixture_collective(comm, budget_send, budget_receive,
                funding['graph_charge'] <= funding['graph_bytes'], funding['graph_charge'], funding['graph_bytes'])
            if not all(row[0] for row in preparation_votes):
                raise MemoryError('complete graph preparation exceeds caller grant')
            _fixture_check(funding, comm, budget_send, budget_receive, 'before-state')
            # The canonical controlled driver declares/stages this inert tool.
            # Observe this worker only, after native capture preparation executed kernels.
            native_evidence = os.environ.get('TENSORFOLD_NATIVE_CONTRACT_CHILD_EVIDENCE')
            if native_evidence:
                from v100_native_contract_driver import record_native_elfs
                record_native_elfs(['tf_sm70_paged_attention_'], rank=rank,
                    tag=_fixture_case_id(head4, rn), evidence_dir=native_evidence)
            _diagnostic_phase(rank, head4, "state.adopt.before")
            state = State(weights, 257, 4, "int8")
            workspace.bind_native_states((state,))
            graphs.validate_adoption()
            if rn:
                _check_rn_graph_identity(workspace, graphs)
            _diagnostic_phase(rank, head4, "state.adopt.after")
            if reference_required:
                _fixture_check(funding, comm, budget_send, budget_receive, 'before-reference', done=('state',))
                _diagnostic_phase(rank, head4, "reference.prepare.before")
                reference = object.__new__(OwnerBuffers)
                reference.__init__(weights, 4, scratch)
                if rn:
                    reference.prepare_native(pair=pair, page_tokens=128)
                    _configure_rn_fixture(reference)
                    reference.configure_native_transport(FULL_COLLECTIVE, comm)
                    reference.bind_native_states((state,))
                else:
                    reference.prepare_native((state,), pair=pair, page_tokens=128)
                _diagnostic_phase(rank, head4, "reference.prepare.after")
            _fixture_check(funding, comm, budget_send, budget_receive, 'before-population', done=('state', 'reference'))
            _diagnostic_phase(rank, head4, "state.populate.before")
            state.kv_begin()
            key = torch.zeros((133, 1, 256), dtype=torch.float16, device=device)
            value = torch.full_like(key, 1 if rank < 4 else 2)
            codes, scales = _encode(key, pair.key, state.kv_status, 0, error=2)
            values, value_scales = _encode(value, pair.value, state.kv_status, 0, error=4)
            for cache in state.kc:
                cache.append(0, codes, values, scales, value_scales)
            from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging, coordinate_rows

            staging = object.__new__(Staging)
            staging.__init__(133, len(state.kc), device, mtp=False)
            raw_indexer = torch.arange(133 * cfg.index_dim, dtype=torch.float32, device=device).reshape(133, cfg.index_dim).half()
            coordinates = coordinate_rows(state, torch.arange(133, dtype=torch.int64, device=device))
            _populate_compact_indexer(state, staging, raw_indexer, coordinates,
                torch.ones(cfg.index_dim, dtype=torch.float32, device=device),
                torch.zeros(1, dtype=torch.float32, device=device), cfg.eps, cfg.mrope_section,
                cfg.rope_attention_factor, 129)
            layer = NS(index=0)
            _diagnostic_phase(rank, head4, "state.populate.after")

            def inputs(rows):
                _diagnostic_phase(rank, head4, "inputs.before", rows)
                bucket = graphs.bucket(rows)
                workspace.clear_native_inputs(bucket)
                workspace.stage_native_descriptors([(state, 0, rows)], layer, 0, rows, bucket_rows=bucket)
                _diagnostic_phase(rank, head4, "inputs.after", rows)

            live_clones = 0
            for rows in (1, 3, 4, 1):
                _fixture_check(funding, comm, budget_send, budget_receive, 'before-replay-' + str(rows),
                    done=('state', 'reference', 'staging', 'population'), live_clones=live_clones)
                inputs(rows)
                _diagnostic_phase(rank, head4, "eager.before", rows)
                eager = graphs.run(rows, graph=False).clone()
                _diagnostic_phase(rank, head4, "eager.after", rows)
                workspace.mark_native_submitted()  # Include the clone, the last borrowed-output reader.
                inputs(rows)
                _diagnostic_phase(rank, head4, "replay.before", rows)
                captured = graphs.run(rows, graph=True).clone()
                _diagnostic_phase(rank, head4, "replay.after", rows)
                workspace.mark_native_submitted()
                assert torch.equal(captured, eager)
                if rn:
                    assert torch.equal(captured.view(torch.int16), eager.view(torch.int16))
                _diagnostic_phase(rank, head4, "eager_replay.compare.after", rows)
                if reference is not None:
                    bucket = reference.native_bucket(rows)
                    reference.clear_native_inputs(bucket)
                    reference.stage_native_descriptors([(state, 0, rows)], layer, 0, rows, bucket_rows=bucket)
                    reference.wait_native_inputs()
                    _diagnostic_phase(rank, head4, "reference.run.before", rows)
                    full = reference.native_run(bucket)[:rows].clone()
                    _diagnostic_phase(rank, head4, "reference.run.after", rows)
                    reference.mark_native_submitted()
                    assert torch.equal(captured, full)
                    if rn:
                        assert torch.equal(captured.view(torch.int16), full.view(torch.int16))
                assert not int(state.kv_status[0])
                _diagnostic_phase(rank, head4, "status.after", rows)
                live_clones = eager.nbytes + captured.nbytes + (full.nbytes if reference is not None else 0)
            # Reset/reuse advances a stable device generation. A previously staged
            # graph must reject its old row descriptor before reading live history.
            _fixture_check(funding, comm, budget_send, budget_receive, 'before-stale-replay',
                done=('state', 'reference', 'staging', 'population'), live_clones=live_clones)
            inputs(3)
            _diagnostic_phase(rank, head4, "stale.reset.before")
            state.reset(weights)
            _diagnostic_phase(rank, head4, "stale.reset.after")
            _diagnostic_phase(rank, head4, "stale.replay.before")
            stale = graphs.run(3, graph=True).clone()
            _diagnostic_phase(rank, head4, "stale.replay.after")
            workspace.mark_native_submitted()
            assert not bool(stale.any()) and int(state.kv_status[0]) & 32
            if rn:
                assert not bool(stale.view(torch.int16).any())
            _diagnostic_phase(rank, head4, "ready.before")
            comm.ready("owner-native-graphs-complete", every=1, timeout=60)
            _diagnostic_phase(rank, head4, "ready.after")
            _fixture_check(funding, comm, budget_send, budget_receive, 'complete',
                done=('state', 'reference', 'staging', 'population'), live_clones=live_clones + stale.nbytes, complete=True)
            _fixture_reports(rank, head4, funding, graphs, workspace, weights, report_attempted, report_errors, rn=rn)
            if report_errors[0] is not None:
                raise report_errors[0]
            if report_errors[1] is not None:
                raise report_errors[1]
        except BaseException as error:
            failed, primary = True, error
            _diagnostic_exception(rank, head4, error)
            _fixture_reports(rank, head4, funding, graphs, workspace, weights, report_attempted, report_errors, rn=rn)
        finally:
            # Successful graph retirement is required even when the body failed
            # after capture. Uncertain owners remain live for process containment.
            _finish_worker(rank, head4, primary, graphs, workspace, reference,
                           staging, head_comm, comm, failed=failed, report_errors=report_errors)



@pytest.mark.parametrize("head4", [False, True])
def test_v100_tp8_owner_partial_graphs_replay_live_and_masked_rows(head4):
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(8)):
        pytest.skip("requires eight real SM70 GPUs and the supported source-built V100 foundation")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(_worker, args=(port, head4), nprocs=8, join=False)
    try:
        deadline = time.monotonic() + 600
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("native owner graph participant did not retire")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)


@pytest.mark.parametrize("head4", [False, True])
def test_v100_tp8_rn_h1_five_phase_graphs_match_rn_full8_reference(head4):
    """New-family graph/transport/lifetime proof; independent math gate is separate."""
    if torch.cuda.device_count() < 8 or any(torch.cuda.get_device_capability(rank) != (7, 0) for rank in range(8)):
        pytest.skip("requires eight real SM70 GPUs and the supported source-built V100 foundation")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context = torch.multiprocessing.spawn(_worker, args=(port, head4, True), nprocs=8, join=False)
    try:
        deadline = time.monotonic() + 600
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("RN owner graph participant did not retire")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.kill()
            process.join(timeout=10)
