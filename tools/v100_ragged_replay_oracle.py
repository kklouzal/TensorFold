"""Controlled short-context oracle on an already loaded, idle native decoder.

Call ``qualify(engine.multi)`` on every TP rank, before admitting requests. The
caller owns model construction, the unchanged graph grant and process cleanup.
No capture, State/weight clone, new math, or numerical tolerance is introduced.
This is a component correctness gate, never full-context or P3 qualification.
"""
from __future__ import annotations

import torch

from tensorfold.families.qwen4_exp.cuda.exact_graphs import ExactStateGraphs
from tensorfold.families.qwen4_exp.cuda.forward import _kv_check, commit, compute, stage
from tensorfold.families.qwen4_exp.cuda.mtp import mtp_compute, mtp_stage

# Startup seeds (4,4,4,4,3,3,3,3). Neither alternate is that layout.
LAYOUTS = ((5, 5, 5, 5, 5, 1, 1, 1), (1, 5, 2, 4, 5, 3, 4, 4))
SEED_LENGTHS = (1, 2, 3, 4, 3, 1, 2, 3)
MTP_SINGLE_ROW_GROUPS = (3, 5, 6, 7)


def snapshot_payload(cfg, *, world=8):
    """Worst-rank live tensor payload from the unsharded authoritative header.

    One result tree holds at most eight short State records, 28/8 local
    vocabulary rows, HC rows, candidates and committed prefixes. The caller
    pays both trees and the transfer peak. Sequential hot single-row probes
    have at most seven State/logit/candidate rows, fourteen HC rows and five valid history
    rows, all bounded by this existing 8-State/28+8-row/10-history envelope.
    Full-window
    caches and weights never enter this calculation or a reference copy.
    """
    from tensorfold.families.qwen4_exp.cuda.state import CAND
    linear = cfg.layer_types.count('linear')
    attention = cfg.layer_types.count('attention') + 1  # stock MTP
    nv, nk = cfg.nv // world, cfg.nk // world
    rec = linear * nv * cfg.dv * cfg.dk * 4
    conv = linear * (cfg.conv_kernel - 1) * (2 * nk * cfg.dk + nv * cfg.dv) * 2
    ple = (cfg.ple_kernel - 1) * cfg.ngram_size * cfg.streams * cfg.hidden * 2
    # Q8 K/V + Half scales; this component deliberately requires Q8 on both sides.
    kv = attention * 10 * (2 * cfg.head_dim + 4 * (cfg.head_dim // 32))
    indexer = attention * (4 * cfg.index_dim * 2 + 4 * 3 * 8 + 2 * cfg.index_dim * 2)
    outputs = 36 * ((cfg.vocab + world - 1) // world) * 4
    outputs += 56 * cfg.streams * cfg.hidden * 2
    outputs += world * 36 * (2 * CAND + 1) * 4 + 8 * 4
    # Three int32 words/State cover target/MTP committed positions plus the
    # separately held live MTP input position in a smaller single-row probe.
    return 8 * (rec + conv + ple + kv + indexer + 3 * 4) + outputs


def _prepare(decoder, label, operation):
    context = decoder.buf.input_preparation_context
    return context.prepare(label, operation)


def _stage(decoder, windows):
    def prepare():
        segs = stage(decoder.w, decoder.buf, windows, late=True)
        if decoder.native_gdn_capture:
            frame = decoder.round_tables.native_frame(segs, decoder.buf, allow_single=True)
            if frame is None:
                raise ValueError('ragged oracle lost fixed native GDN registration')
            decoder.buf.gdn_tables = frame
        return segs
    return _prepare(decoder, 'ragged-oracle-target', prepare)


def _cpu(value):
    return value.detach().cpu()


def _state(state):
    """Only valid short prefixes; shared arena offsets belong to each cache."""
    result = {'pos': state.pos, 'mtp_len': state.mtp_len, 'cur': tuple(state.cur),
              'pos_dev': _cpu(state.pos_dev), 'mtp_pos': _cpu(state.mtp_pos),
              'rec': _cpu(state.rec[state.cur[0]]) if state.cur else _cpu(state.rec),
              'conv': _cpu(state.conv), 'ple_tail': _cpu(state.ple_tail),
              'ple_history': None if state.ple_history is None else state.ple_history.tolist()}
    for index, (cache, length) in enumerate([*( (cache, state.pos) for cache in state.kc),
                                             (state.mtp_kc, state.mtp_len)]):
        for span, (_logical, physical, count) in enumerate(cache._spans(0, length)):
            for name in ('k', 'v', 'ks', 'vs'):
                # Unquantized scale placeholders are not valid per-token rows.
                if name == 'ks' and not cache.key_format.quantized:
                    continue
                if name == 'vs' and not cache.value_format.quantized:
                    continue
                result[f'cache/{index}/{span}/{name}'] = _cpu(getattr(cache, name)[physical:physical + count])
    for index, pool in enumerate([*state.pooled, state.mtp_pooled]):
        length = state.pos if index < len(state.pooled) else state.mtp_len
        result[f'pool/{index}'] = _cpu(pool[:length // state.ratio])
    compact = state.compact_indexer
    compact.wait_native_reuse()
    for index, entry in enumerate(compact.entries):
        if entry is not None:
            length = state.pos if index < len(state.kc) else state.mtp_len
            if entry.n != length:
                raise AssertionError('ragged oracle compact frontier is not committed')
            # Only the incomplete group is valid raw history. Borrow its real
            # producer storage; no GPU snapshot clone or inactive tail compare.
            with entry.reading((entry.raw, entry.coordinates)):
                result[f'indexer/{index}/raw'] = _cpu(entry.raw[:length % state.ratio])
                result[f'indexer/{index}/coordinates'] = _cpu(entry.coordinates[:length % state.ratio])
            result[f'indexer/{index}/length'] = length
    return result


def _equal(actual, expected):
    if actual.keys() != expected.keys():
        raise AssertionError('ragged oracle output fields changed')
    for name, value in actual.items():
        reference = expected[name]
        if isinstance(value, torch.Tensor):
            if (value.shape != reference.shape or value.dtype != reference.dtype
                    or not torch.equal(value, reference)
                    or not torch.equal(value.view(torch.uint8), reference.view(torch.uint8))):
                raise AssertionError(f'ragged oracle mismatch: {name}')
        elif value != reference:
            raise AssertionError(f'ragged oracle mismatch: {name}')


def _runtime_payload(decoder):
    """Structural preflight over actual tensors, before any State mutation."""
    from tensorfold.families.qwen4_exp.cuda.state import CAND
    def nbytes(value):
        return value.numel() * value.element_size()
    state_bytes = []
    for st in decoder.free:
        value = nbytes(st.rec[0]) + nbytes(st.conv) + nbytes(st.ple_tail)
        value += nbytes(st.pos_dev) + 2 * nbytes(st.mtp_pos)
        for cache in (*st.kc, st.mtp_kc):
            rows = sum(count for _logical, _physical, count in cache._spans(0, 10))
            value += rows * cache.pair.row_bytes(1, cache.head_dim)
        for pool in (*st.pooled, st.mtp_pooled):
            value += nbytes(pool[:10 // st.ratio])
        if st.compact_indexer is None:
            raise ValueError('ragged oracle requires actual compact unique indexer owners')
        for entry in st.compact_indexer.entries:
            if entry is not None:
                value += nbytes(entry.raw) + nbytes(entry.coordinates)
        state_bytes.append(value)
    b, mb = decoder.buf, decoder.mbuf
    outputs = nbytes(b.logits[:28]) + nbytes(mb.logits[:8])
    outputs += nbytes(b.streams[:28]) + nbytes(mb.streams[:28])
    outputs += decoder.w.comm.world * 36 * (2 * CAND + 1) * 4 + 8 * 4
    return sum(sorted(state_bytes, reverse=True)[:8]) + outputs


def _seed(decoder, states):
    w, b, mb = decoder.w, decoder.buf, decoder.mbuf
    # Two genuine eager commits establish unequal contexts and mixed GDN banks.
    seed = SEED_LENGTHS[:len(states)]
    for active, lengths in ((states, seed), (states[::2], (1,) * len(states[::2]))):
        seed_windows = [(st, [(19 * i + row + 7) % w.cfg.vocab for row in range(n)])
                        for i, (st, n) in enumerate(zip(active, lengths))]
        segs = _stage(decoder, seed_windows)
        compute(w, segs, b, check=False)
        _kv_check(segs, comm=w.comm, workspace=getattr(decoder, 'kv_cohort', None))
        for st, first, last in segs:
            commit(w, st, b, last - first, last - first, at=first)
        # Populate the genuine MTP context too: the later ragged replay must
        # consume unequal nonempty histories, rather than only empty caches.
        mseg = _prepare(decoder, 'ragged-oracle-mtp-seed', lambda segs=segs, seed_windows=seed_windows: mtp_stage(mb,
            [(st, [(token + 1) % w.cfg.vocab for token in tokens], b.streams[first:last])
             for (st, tokens), (_st, first, last) in zip(seed_windows, segs)]))
        # The no-head MTP branch always performs its cohort _kv_check.
        mtp_compute(w, mseg, mb, logits=False, check=False)
        for st, first, last in mseg:
            st.set_mtp_len(st.mtp_len + last - first)
    if any(st.pos != st.mtp_len or not st.mtp_len for st in states):
        raise AssertionError('ragged oracle did not establish matching nonempty target/MTP seed contexts')
    parity = tuple(tuple(st.cur) for st in states)
    if not any(value != parity[0] for value in parity):
        raise AssertionError('ragged oracle did not establish mixed GDN parity')


def _run(decoder, states, counts, *, captured):
    w, b, mb = decoder.w, decoder.buf, decoder.mbuf
    _seed(decoder, states)
    windows = [(st, [(31 * i + row + 11) % w.cfg.vocab for row in range(n)])
               for i, (st, n) in enumerate(zip(states, counts))]
    segs = _stage(decoder, windows)
    out = decoder.graphs.verify_forward(segs) if captured else compute(w, segs, b, check=False)
    if out is None:
        raise AssertionError('requested ragged verifier fell back to eager')
    _kv_check(segs, comm=w.comm, workspace=getattr(decoder, 'kv_cohort', None))
    result = {'verify/logits': _cpu(out), 'verify/streams': _cpu(b.streams[:sum(counts)])}
    # Preserve the complete rank candidate publication, including integer IDs.
    from tensorfold.families.qwen4_exp.cuda.state import CAND
    words = w.comm.world * sum(counts) * (2 * CAND + 1)
    result['verify/candidates'] = _cpu(b.cand_all[:words].view(torch.int32))
    for i, (st, first, last) in enumerate(segs):
        commit(w, st, b, last - first, min(last - first, 1 + i % 2), at=first)
    mseg = _prepare(decoder, 'ragged-oracle-mtp', lambda: mtp_stage(mb,
        [(st, tokens, b.streams[first:last])
         for (st, tokens), (_st, first, last) in zip(windows, segs)]))
    out = decoder.graphs.mtp_step(mseg) if captured else mtp_compute(w, mseg, mb, check=False)
    if out is None:
        raise AssertionError('requested ragged MTP fell back to eager')
    _kv_check(mseg, comm=w.comm, workspace=getattr(decoder, 'kv_cohort', None))
    result.update({'mtp/logits': _cpu(out), 'mtp/streams': _cpu(mb.streams[:sum(counts)]),
                   'mtp/last': _cpu(mb.last[:len(states)])})
    words = w.comm.world * len(states) * (2 * CAND + 1)
    result['mtp/candidates'] = _cpu(mb.cand_all[:words].view(torch.int32))
    for st, first, last in mseg:
        st.set_mtp_len(st.mtp_len + last - first)
    for i, st in enumerate(states):
        if st.pos > 10 or st.mtp_len > 10:
            raise AssertionError('ragged oracle exceeded its declared short prefix')
        result.update({f'state/{i}/{name}': value for name, value in _state(st).items()})
    return result


def _run_mtp(decoder, states, counts, *, captured):
    """One genuine next-row MTP frame on each unequal nonempty State history."""
    w, b, mb = decoder.w, decoder.buf, decoder.mbuf
    if counts != (1,) * len(states):
        raise ValueError('hot MTP probes require exactly one row per State')
    _seed(decoder, states)
    windows = [(st, [(31 * i + 11) % w.cfg.vocab]) for i, st in enumerate(states)]
    segs = _stage(decoder, windows)
    compute(w, segs, b, check=False)  # Original eager target conditioning, never a new target graph key.
    _kv_check(segs, comm=w.comm, workspace=getattr(decoder, 'kv_cohort', None))
    for st, first, last in segs:
        commit(w, st, b, 1, 1, at=first)
    def positions():
        result = {}
        for i, st in enumerate(states):
            value = _cpu(st.mtp_pos)
            if value.item() != st.mtp_len or st.mtp_len <= 0:
                raise AssertionError('hot MTP input must use its actual live nonzero GPU frontier')
            result[f'mtp/input_position/{i}'] = value
        return result
    result = _prepare(decoder, 'ragged-hot-mtp-positions', positions)
    result['conditioning/streams'] = _cpu(b.streams[:len(states)])
    mseg = _prepare(decoder, 'ragged-oracle-hot-mtp', lambda: mtp_stage(mb,
        [(st, [(tokens[0] + 1) % w.cfg.vocab], b.streams[first:last])
         for (st, tokens), (_st, first, last) in zip(windows, segs)]))
    out = decoder.graphs.mtp_step(mseg) if captured else mtp_compute(w, mseg, mb, check=False)
    if out is None:
        raise AssertionError('requested single-row MTP group fell back to eager')
    _kv_check(mseg, comm=w.comm, workspace=getattr(decoder, 'kv_cohort', None))
    from tensorfold.families.qwen4_exp.cuda.state import CAND
    words = w.comm.world * len(states) * (2 * CAND + 1)
    result.update({'mtp/logits': _cpu(out), 'mtp/streams': _cpu(mb.streams[:len(states)]),
                   'mtp/last': _cpu(mb.last[:len(states)]),
                   'mtp/candidates': _cpu(mb.cand_all[:words].view(torch.int32))})
    for i, (st, _first, _last) in enumerate(mseg):
        st.set_mtp_len(st.mtp_len + 1)
        if not 0 < st.pos == st.mtp_len <= 10:
            raise AssertionError('single-row MTP group lost its matching live short-context frontier')
        result.update({f'state/{i}/{name}': value for name, value in _state(st).items()})
    return result


def qualify(decoder, *, snapshot_payload_bytes):
    """Mutates only empty original slots; success resets them for later admission.

    Failure propagates to the caller's existing process containment. Never reset
    or reuse a failed State to disguise an execution/status/cleanup failure.
    """
    def validate():
        graphs = decoder.graphs
        if (graphs is None or type(graphs.bound) is not ExactStateGraphs
                or graphs.bound.preparing or graphs.failure is not None
                or decoder.streams or decoder.filling or decoder.kept or decoder.held
                or len(decoder.states) != 24 or not decoder.full_resident
                or decoder.full_windows != 24
                or any(st.logical_context != 262144 or st.capacity < 262144 for st in decoder.states)
                or len(decoder.free) < 10 or decoder.mbuf is None
                or decoder.depth < 4 or decoder.w.comm is None or decoder.w.comm.world != 8
                or decoder.w.meta.get('execution_provider') != 'sm70'
                or torch.cuda.get_device_capability(decoder.w.device) != (7, 0)
                or any(st.pos or st.mtp_len for st in decoder.free)):
            raise ValueError('ragged oracle requires idle native TP8 ExactStateGraphs and ten empty slots')
        if (type(snapshot_payload_bytes) is not int or snapshot_payload_bytes <= 0
                or any(st.kv_pair.key.name != 'int8' or st.kv_pair.value.name != 'int8' for st in decoder.states)):
            raise ValueError('ragged oracle requires prepaid snapshot payload and Q8 KV')
        if _runtime_payload(decoder) > snapshot_payload_bytes:
            raise ValueError('actual ragged oracle tensor plan exceeds prepaid header bound')
        required = [('verify', 28, 8), ('mtp', 28, 8), *(('mtp', n, n) for n in MTP_SINGLE_ROW_GROUPS)]
        if not all(key in graphs.bound.captured_keys for key in required):
            raise ValueError('ragged oracle requires both (28,8) roles and all four hot MTP groups captured')
        if (graphs.bound.requested_policy is None
                or graphs.bound.requested_policy['policy'] != 'observed-demand-v1'):
            raise ValueError('ragged oracle requires the explicit observed-demand policy')
        return tuple(decoder.free[:10])
    slots = _prepare(decoder, 'ragged-oracle-admit', validate)
    hits = dict(decoder.graphs.bound.serving_hits)
    bindings = tuple(decoder.graphs.bound.bindings)
    receipts, mtp_receipts = [], []
    plans = [('both', tuple(reversed(slots[2 * i:2 * i + 8])), counts) for i, counts in enumerate(LAYOUTS)]
    plans += [('mtp', tuple(reversed(slots[1:n + 1])), (1,) * n) for n in MTP_SINGLE_ROW_GROUPS]
    with torch.inference_mode():
        for role, states, counts in plans:
            runner = _run if role == 'both' else _run_mtp
            def reset(states=states):
                for st in states:
                    st.reset(decoder.w)  # ordinary new-sequence generation publication
            _prepare(decoder, 'ragged-oracle-reset', reset)
            generations = tuple(st.slot_generation for st in states)
            expected = runner(decoder, states, counts, captured=False)
            _prepare(decoder, 'ragged-oracle-reset', reset)
            def generation_check(states=states, generations=generations):
                if tuple(st.slot_generation for st in states) != tuple(n + 1 for n in generations):
                    raise AssertionError('ragged oracle reset did not advance slot generations')
            _prepare(decoder, 'ragged-oracle-generation', generation_check)
            actual = runner(decoder, states, counts, captured=True)
            if any(sum(v.numel() * v.element_size() for v in result.values()
                       if isinstance(v, torch.Tensor)) > snapshot_payload_bytes for result in (expected, actual)):
                raise AssertionError('ragged oracle exceeded prepaid tensor payload')
            _prepare(decoder, 'ragged-oracle-compare', lambda actual=actual, expected=expected: _equal(actual, expected))
            receipt = {'rows': sum(counts), 'participants': len(states), 'lengths': list(counts),
                       'slots': [st.history_slot for st in states], 'exact': True}
            if role == 'mtp':
                receipt['input_positions'] = [int(actual[f'mtp/input_position/{i}'].item()) for i in range(len(states))]
                mtp_receipts.append(receipt)
            else:
                receipts.append(receipt)
            del expected, actual
            _prepare(decoder, 'ragged-oracle-reset', reset)
        _kv_check([(st, 0, 1) for st in decoder.states], comm=decoder.w.comm,
                  workspace=getattr(decoder, 'kv_cohort', None))
        def complete():
            if (decoder.streams or decoder.filling or decoder.kept or decoder.held
                    or any(st.pos or st.mtp_len for st in decoder.states)):
                raise AssertionError('ragged oracle did not return all States clean and idle')
            if tuple(decoder.graphs.bound.bindings) != bindings:
                raise AssertionError('ragged oracle changed startup graph bindings')
            for role in ('verify', 'mtp'):
                expected_hits = len(LAYOUTS) + (len(MTP_SINGLE_ROW_GROUPS) if role == 'mtp' else 0)
                if decoder.graphs.bound.serving_hits[role] - hits[role] != expected_hits:
                    raise AssertionError('ragged oracle did not replay every captured role')
        _prepare(decoder, 'ragged-oracle-complete', complete)
    return {'schema': 'tensorfold.native-ragged-replay-oracle-v2', 'cases': receipts, 'mtp_cases': mtp_receipts,
            'comparison': 'bit-exact eager versus captured same fixed tokens',
            'scope': 'short-context component; not full-context, quality or P3 qualification',
            'rank': decoder.w.comm.rank, 'snapshot_payload_bytes': snapshot_payload_bytes}
