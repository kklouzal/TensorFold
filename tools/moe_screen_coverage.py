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


def validate_prepared_profile(ready):
    profile = ready.get('sm70_profile')
    if profile == 'source-router-gdn-shared-fusion-trial':
        flags = ready.get('native_flags', {})
        fusion = ready.get('shared_fusion', {})
        graph = ready.get('capacity_plan', {}).get('multi_graphs', {}).get('moe_regions', {})
        if (flags.get('gguf_experts_implementation') != 'packed_blocks2w'
                or fusion.get('enabled') is not True or fusion.get('banks') != 49
                or fusion.get('public_count') != 513 or fusion.get('slots') != 11
                or fusion.get('target_output') != 'float16' or fusion.get('mtp_output') != 'float32'
                or graph.get('numeric_profile') != profile
                or graph.get('graph_contract') != 'pure-moe-target-exactR-ordered-pools-v1'):
            raise ValueError('captured Fusion backend/profile/output contract is not qualified')
        return 'packed_blocks2w'
    if profile != 'source-batch-router-blas-trial':
        raise ValueError('unknown captured numeric profile')
    return 'packed_exact'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepared = None
    raw = []
    for rank in range(8):
        ready = json.loads((args.run / f'rank-{rank}-ready.json').read_text())
        backend = validate_prepared_profile(ready)
        descriptor = ready.get('moe_capture_ledger')
        if not descriptor or descriptor['phase_records'] != 145:
            raise ValueError('required cold capture descriptor missing')
        path = args.run / Path(descriptor['path']).name
        payload = path.read_bytes()
        if len(payload) != descriptor['bytes'] or hashlib.sha256(payload).hexdigest() != descriptor['sha256']:
            raise ValueError('raw capture ledger differs from cold checkpoint')
        ledger = json.loads(payload)
        phases = ledger['capture_phases']
        keys = {(row['key'][2], row['key'][3]) for row in phases
                if len(row['key']) == 4 and row['key'][:2] == ['moe', 'target']}
        if prepared is not None and keys != prepared:
            raise ValueError('prepared graph keyspaces differ between ranks')
        prepared = keys
        raw.append({'rank': rank, 'sha256': descriptor['sha256'], 'phase_records': len(phases),
                    'qualified_captured_backend': backend})
    results = []
    for offered in (1, 2, 4, 8, 16, 24):
        case = json.loads((args.run / f'fast-mtp-target-n{offered:02}.json').read_text())
        results.append(validate_case(offered, case['graph_execution']['delta'], prepared_keys=prepared))
    if not any(row['captured_moe_replays'] for row in results):
        raise ValueError('grid did not execute the requested MoE catalogue')
    if args.output.exists():
        raise FileExistsError(args.output)
    write_json(args.output, {'scope': 'coverage only, no performance/quality acceptance',
                            'all8_cold_ledgers': raw, 'cases': results, 'passed': True})


if __name__ == '__main__':
    main()
