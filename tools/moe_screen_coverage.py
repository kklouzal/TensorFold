"""Read-only post-screen coverage: distinguish real eager calls from graph work."""
import argparse
import hashlib
import json
from pathlib import Path

if __package__:
    from .benchmark_output import write_json
else:
    from benchmark_output import write_json

ROWS = (30, 31, 32)
MAX_COUNT = (1 << 63) - 1


def count(delta, name):
    value = delta.get(name, 0)
    if type(value) is not int or not 0 <= value <= MAX_COUNT:
        raise ValueError('invalid bounded execution counter: ' + name)
    return value


def validate_case(offered, delta, *, prepared_keys):
    if prepared_keys != {(rows, layer) for rows in ROWS for layer in range(48)}:
        raise ValueError('required144 actual prepared MoE keys are missing')
    required = ('moe_regions.frames', 'moe_regions.hits',
                'moe_regions.rows_30_frames', 'moe_regions.rows_31_frames', 'moe_regions.rows_32_frames')
    if any(name not in delta for name in required):
        raise ValueError('required captured execution counters are unavailable, not zero')
    frames = count(delta, 'moe_regions.frames')
    hits = count(delta, 'moe_regions.hits')
    row_frames = {rows: count(delta, f'moe_regions.rows_{rows}_frames') for rows in ROWS}
    if frames != sum(row_frames.values()) or hits != frames * 48:
        raise ValueError('captured MoE counters do not describe complete qualified exactR cycles')
    eager = count(delta, 'router_blas.target.calls')
    mtp = count(delta, 'router_blas.mtp.calls')
    gdn = count(delta, 'gdn.batch_submitted_layers')
    if offered > 1:
        if not eager and not hits:
            raise ValueError('no actual target eager or qualified captured execution')
        if not mtp or not gdn:
            raise ValueError('independent eager MTP and GDN evidence must remain positive')
        for name in ('owner.prelude_submitted_batches', 'owner.prelude_submitted_mtp_batches'):
            if not count(delta, name):
                raise ValueError('independent native owner batch evidence missing: ' + name)
    return {'offered': offered, 'eager_target_wrapper_calls': eager,
            'captured_moe_replays': hits, 'completed_captured_forwards': frames,
            'captured_forwards_by_exact_rows': row_frames, 'eager_mtp_wrapper_calls': mtp,
            'gdn_submitted_layers': gdn,
            'scope': 'cohort readiness+decode+drain; graph replay is not a fabricated Python router call'}


def validate_prepared_profile(ready, *, hc=False):
    profile = ready.get('sm70_profile')
    if hc:
        boundary = ready.get('capacity_plan', {}).get('multi_graphs', {}).get('moe_regions', {})
        if (boundary.get('graph_contract') != 'mlp-hc-mode3-moe-target-exactR-ordered-pools-v1'
                or boundary.get('funding', {}).get('boundary') != 'MLP-HC-mode3+pure-MoE'):
            raise ValueError('HC/MoE captured boundary seal missing')
    if profile == 'source-router-gdn-shared-fusion-trial':
        flags = ready.get('native_flags', {})
        fusion = ready.get('shared_fusion', {})
        graph = ready.get('capacity_plan', {}).get('multi_graphs', {}).get('moe_regions', {})
        if (flags.get('gguf_experts_implementation') != 'packed_blocks2w'
                or fusion.get('enabled') is not True or fusion.get('banks') != 49
                or fusion.get('public_count') != 513 or fusion.get('slots') != 11
                or fusion.get('target_output') != 'float16' or fusion.get('mtp_output') != 'float32'
                or graph.get('numeric_profile') != profile
                or graph.get('graph_contract') != ('mlp-hc-mode3-moe-target-exactR-ordered-pools-v1' if hc else 'pure-moe-target-exactR-ordered-pools-v1')
                or hc and graph.get('funding', {}).get('boundary') != 'MLP-HC-mode3+pure-MoE'):
            raise ValueError('captured Fusion backend/profile/output contract is not qualified')
        return 'packed_blocks2w'
    if profile != 'source-batch-router-blas-trial':
        raise ValueError('unknown captured numeric profile')
    return 'packed_exact'


def validate_candidate_case(offered, delta):
    values = {}
    for role in ('target', 'mtp'):
        names = [f'candidate_regions.{role}.rows_{rows}' for rows in range(1, 33)]
        if any(name not in delta for name in names):
            raise ValueError('candidate counters unavailable, not zero')
        values[role] = sum(count(delta, name) for name in names)
    if offered > 1 and not all(values.values()):
        raise ValueError('independent target/MTP candidate replay evidence missing')
    return {'offered': offered, 'replays': values,
            'scope': 'host replay submissions; N1 enclosing graphs intentionally retain original local body'}


def validate_candidate_ledger(ready, run):
    descriptor = ready.get('candidate_capture_ledger')
    if not descriptor or descriptor['phase_records'] != 64:
        raise ValueError('required64-phase candidate descriptor missing')
    payload = (run / Path(descriptor['path']).name).read_bytes()
    if len(payload) != descriptor['bytes'] or hashlib.sha256(payload).hexdigest() != descriptor['sha256']:
        raise ValueError('candidate raw capture ledger differs')
    phases = json.loads(payload)['capture_phases']
    keys = {tuple(row['key']) for row in phases}
    if len(phases) != 64 or keys != {('candidate', role, rows) for role in ('target', 'mtp') for rows in range(1, 33)}:
        raise ValueError('candidate64-key catalogue is incomplete')
    graph = ready.get('capacity_plan', {}).get('multi_graphs', {}).get('candidate_regions', {})
    if (graph.get('ready') is not True or graph.get('graphs') != 64
            or graph.get('capture_phase_count') != 64 or graph.get('capture_accounted_phase_count') != 64
            or graph.get('funding', {}).get('future_host_bytes_per_rank') != 1 << 30):
        raise ValueError('candidate admitted cold catalogue proof missing')
    return {'sha256': descriptor['sha256'], 'phase_records': 64}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--hc-moe', action='store_true', help='Require the explicit qualified MLP-HC plus MoE boundary')
    args = parser.parse_args()
    prepared = None
    raw = []
    for rank in range(8):
        ready = json.loads((args.run / f'rank-{rank}-ready.json').read_text())
        backend = validate_prepared_profile(ready, hc=args.hc_moe)
        candidate_ledger = validate_candidate_ledger(ready, args.run)
        descriptor = ready.get('moe_capture_ledger')
        if not descriptor or descriptor['phase_records'] != 145:
            raise ValueError('required cold capture descriptor missing')
        path = args.run / Path(descriptor['path']).name
        payload = path.read_bytes()
        if len(payload) != descriptor['bytes'] or hashlib.sha256(payload).hexdigest() != descriptor['sha256']:
            raise ValueError('raw capture ledger differs from cold checkpoint')
        ledger = json.loads(payload)
        if args.hc_moe and ledger['funding'].get('boundary') != 'MLP-HC-mode3+pure-MoE':
            raise ValueError('cold HC/MoE boundary seal missing')
        phases = ledger['capture_phases']
        keys = {(row['key'][2], row['key'][3]) for row in phases
                if len(row['key']) == 4 and row['key'][:2] == ['moe', 'target']}
        if prepared is not None and keys != prepared:
            raise ValueError('prepared graph keyspaces differ between ranks')
        prepared = keys
        raw.append({'rank': rank, 'sha256': descriptor['sha256'], 'phase_records': len(phases),
                    'qualified_captured_backend': backend, 'candidate_ledger': candidate_ledger})
    results = []
    for offered in (1, 2, 4, 8, 16, 24):
        case = json.loads((args.run / f'fast-mtp-target-n{offered:02}.json').read_text())
        result = validate_case(offered, case['graph_execution']['delta'], prepared_keys=prepared)
        result['candidate'] = validate_candidate_case(offered, case['graph_execution']['delta'])
        results.append(result)
    if not any(row['captured_moe_replays'] for row in results):
        raise ValueError('grid did not execute the requested MoE catalogue')
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, {'scope': 'coverage only, no performance/quality acceptance',
                            'all8_cold_ledgers': raw, 'cases': results, 'passed': True})


if __name__ == '__main__':
    main()
