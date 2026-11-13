"""Stdlib host-boundary controls. No container, native model or GPU proof."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import v100_ragged_replay_gate as gate


class GateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = self.make_spec()

    def sealed(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = value if isinstance(value, bytes) else (json.dumps(value) + '\n').encode()
        path.write_bytes(raw)
        path.chmod(0o444)
        return {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}

    def make_spec(self):
        closure = {name: self.sealed(name, b'controlled host helper fixture\n') for name in gate.HELPERS}
        oracle = {'schema': 'tensorfold.native-ragged-replay-input-v2', 'key': [28, 8],
                  'layouts': [[5, 5, 5, 5, 5, 1, 1, 1], [1, 5, 2, 4, 5, 3, 4, 4]],
                  'slots': 10, 'maximum_context_rows': 10, 'comparison': 'bit-exact', 'mtp_single_row_groups': [3, 5, 6, 7]}
        demand = self.sealed('demand.json', {'schema': 'tensorfold.source27.actual-graph-demand-result-v1',
                                           'all8_event_and_histogram_census_identical': True})
        keys = [[role, n * rows, n] for n in (1, 2, 4, 8, 16, 24)
                for rows in (range(1, 6) if n == 1 else (1, 5)) for role in ('verify', 'mtp')]
        keys += [['verify', 28, 8], ['mtp', 28, 8], *[['mtp', n, n] for n in (3, 5, 6, 7)]]
        prep = {'kernel_family': 'sm70-paged-owner-rn-ex2-p2051-v1', 'graph_family': 'uncaptured',
                'graph_budget_bytes': 0, 'trial': True, 'collective_family': 'tp8-allgather-v1',
                'rn_heads_per_block': 12, 'native_gdn_capture': True, 'native_hc_writeback_capture': True,
                'native_family': 'owned-native-family-v1', 'owned_q8_window_prepare': True,
                'prepared_linears': 'owned-prepared-m32-v1', 'owned_q8_window_dispatch': 'grid-v1',
                'exact_graph_policy': {'policy': 'observed-demand-v1', 'keys': keys, 'evidence_sha256': demand['sha256']}}
        return {'schema': gate.SCHEMA, 'output': str(self.root / 'out'), 'source': {'root': str(self.root / 'source'), 'source_uid': 'a' * 64},
            'image': 'sha256:' + 'b' * 64,
            'installed_check': {'root': str(self.root / 'installed'),
                'inputs': self.sealed('installed/check-inputs.json', {'fixture': 'inputs'}),
                'terminal': self.sealed('installed/container-result.json', {'fixture': 'terminal'})},
            'profile': self.sealed('profile.json', {'schema': 'tensorfold.execution-profile-manifest', 'version': 1,
                'artifacts': {'target/model.gguf': {'size': 1, 'sha256': 'c' * 64}}, 'preparation': prep}),
            'oracle': self.sealed('oracle.json', oracle), 'demand': demand,
            'model': {'gpus': [f'GPU-{i:08x}-0000-0000-0000-000000000000' for i in range(8)],
                'models_root': str(self.root / 'models'), 'target': 'target/model.gguf', 'mtp': 'draft/mtp.gguf',
                'cache': str(self.root / 'cache'), 'max_waiting_input_bytes': 226493088},
            'numa': {'plan': str(self.root / 'plan.json'), 'plan_sha256': 'c' * 64,
                     'seccomp_directory': str(self.root / 'numa'), 'seccomp_manifest_sha256': 'd' * 64},
            'resources': {'cpus': 64, 'memory': '340g', 'startup_timeout_s': 900, 'operation_timeout_s': 7200},
            'health_baseline': {name: 'e' * 64 for name in gate.HEALTH}, 'controller_closure': closure}

    def validate(self, spec):
        with patch.object(gate, '__file__', str(self.root / 'v100_ragged_replay_gate.py')), \
                patch.object(gate, 'helper_paths', return_value={name: self.root / name for name in gate.HELPERS}):
            return gate.validate_operation(spec)

    def go(self):
        return {'schema': 'tensorfold.native-ragged-gate-root-go-v1', 'approved': True,
            'operation_sha256': 'f' * 64, 'controller_sha256': self.spec['controller_closure']['v100_ragged_replay_gate.py']['sha256'],
            'source_uid': self.spec['source']['source_uid'], 'image': self.spec['image'],
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}

    def test_closed_operation_accepts_only_bound_readonly_inputs(self):
        self.assertEqual(self.validate(self.spec), self.spec)
        changed = copy.deepcopy(self.spec)
        changed['benchmark_plan'] = 'forbidden'
        with self.assertRaises(ValueError):
            self.validate(changed)
        Path(self.spec['oracle']['path']).chmod(0o644)
        with self.assertRaises(ValueError):
            self.validate(self.spec)

    def test_operation_refuses_changed_identity_and_noninteger_shape(self):
        changed = copy.deepcopy(self.spec)
        changed['profile']['sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            self.validate(changed)
        data = gate.object_json(Path(self.spec['oracle']['path']).read_bytes())
        data['slots'] = 10.0
        changed = copy.deepcopy(self.spec)
        changed['oracle'] = self.sealed('float-oracle.json', data)
        with self.assertRaises(ValueError):
            self.validate(changed)

    def test_unknown_preparation_and_missing_original_key_are_refused(self):
        data = gate.object_json(Path(self.spec['profile']['path']).read_bytes())
        data['preparation']['other_observer'] = True
        changed = copy.deepcopy(self.spec)
        changed['profile'] = self.sealed('bad-profile.json', data)
        with self.assertRaises(ValueError):
            self.validate(changed)
        del data['preparation']['other_observer']
        data['preparation']['exact_graph_policy']['keys'].pop(0)
        changed['profile'] = self.sealed('short-profile.json', data)
        with self.assertRaises(ValueError):
            self.validate(changed)

    def test_foreign_loaded_helper_origin_is_refused(self):
        loaded = {name: self.root / name for name in gate.HELPERS}
        loaded['v100_onecat_campaign.py'] = self.root / 'foreign.py'
        with patch.object(gate, '__file__', str(self.root / 'v100_ragged_replay_gate.py')), \
                patch.object(gate, 'helper_paths', return_value=loaded), \
                self.assertRaisesRegex(ValueError, 'origin'):
            gate.validate_operation(self.spec)

    def test_root_go_binds_every_authority_axis(self):
        value = self.go()
        gate.authorization(value, operation_sha='f' * 64, spec=self.spec, boot_id=value['boot_id'])
        for field in ('approved', 'operation_sha256', 'controller_sha256', 'source_uid', 'image', 'boot_id'):
            with self.subTest(field=field):
                bad = dict(value)
                bad[field] = False if field == 'approved' else 'wrong'
                with self.assertRaises(ValueError):
                    gate.authorization(bad, operation_sha='f' * 64, spec=self.spec, boot_id=value['boot_id'])

    def test_no_effects_before_root_go(self):
        value = self.go()
        value['approved'] = False
        with patch.object(gate.campaign, 'frozen') as frozen:
            with self.assertRaises(ValueError):
                gate.run(self.spec, self.root / 'out', root_go=value, operation_sha='f' * 64)
            frozen.assert_not_called()

    def test_exact_transport_has_full_profile_and_no_observers(self):
        bundle = {'cpuset_cpus': '0-63', 'cpuset_mems': '0,1'}
        argv = gate.container_argv(self.spec, self.root / 'source', bundle, self.root / 'operator', self.root / 'out')
        for flag, expected in (('--user', '1000:1000'), ('--memory', '340g'), ('--memory-swap', '340g'),
                ('--context', '262144'), ('--parallel', '24'), ('--kv-full-windows', '24'), ('--mtp-drafts', '4')):
            self.assertEqual(argv[argv.index(flag) + 1], expected)
        for flag in ('--benchmark-plan', '--teacher-plan', '--graph-use-receipts', '--commit-trace-events',
                     '--scheduler-trace', '--no-graphs', '--no-drafts', '--cap-add', '--privileged'):
            self.assertNotIn(flag, argv)
        self.assertIn('--ragged-replay-input', argv)
        self.assertIn(self.spec['image'], argv)

    def test_duplicate_or_nonfinite_json_fails(self):
        for raw in (b'{"key":1,"key":2}', b'{"x":NaN}'):
            with self.assertRaises(ValueError):
                gate.object_json(raw)

    def test_all_rank_startup_and_oracle_files_are_immutable_through_retirement(self):
        # Synthetic schema inputs exercise only host collection/lifetime checks.
        out = self.root / 'receipt-fixture'
        directory = out / 'qualification'
        directory.mkdir(parents=True)
        files = {path: {'sha256': str(i) * 64} for i, path in enumerate((
            'src/tensorfold/families/qwen4_exp/cuda/engine.py',
            'src/tensorfold/families/qwen4_exp/cuda/multi_graphs.py',
            'src/tensorfold/families/qwen4_exp/cuda/exact_graphs.py',
            'tools/v100_owned_qualification.py', 'src/tensorfold/cli.py'))}
        sources = dict(zip(('engine', 'graph_controller', 'graph_policy', 'observer_tool'),
                           [row['sha256'] for row in list(files.values())[:4]]))
        policy = gate.object_json(Path(self.spec['profile']['path']).read_bytes())['preparation']['exact_graph_policy']
        layouts = gate.object_json(Path(self.spec['oracle']['path']).read_bytes())['layouts']
        for rank in range(8):
            accounting = {'schema': 'tensorfold.native-ragged-accounting-v2', 'rank': rank,
                'catalogue': {**{name: policy['keys'] for name in ('available_keys', 'requested_keys', 'selected_keys', 'captured_keys')},
                    'evidence_sha256': self.spec['demand']['sha256'], 'declined_keys': []},
                'capture_settlement': {'room_bytes': 1, 'budget_bytes': 2},
                'capacity': {'graph_admission': 'AUTO', 'context_window': 262144, 'reserved': 24,
                             'workspace_bytes': 2 * 2**30, 'effective_device_floor_bytes': 4 * 2**30},
                'quiescent_sample': {'physical_free_bytes': 8 * 2**30},
                'actual_states': [{'logical_context': 262144, 'capacity': 262149, 'pos': 0, 'mtp_len': 0}] * 24}
            row = {'schema': 'tensorfold.native-ragged-replay-oracle-v2', 'rank': rank,
                'input_sha256': self.spec['oracle']['sha256'], 'runtime_sources': sources,
                'resources': {'global_host_obligation_bytes': 123},
                'cases': [{'exact': True, 'lengths': counts, 'rows': 28, 'participants': 8, 'slots': list(range(8))} for counts in layouts],
                'mtp_cases': [{'exact': True, 'rows': n, 'participants': n, 'lengths': [1] * n, 'slots': list(range(n)), 'input_positions': [2 + i % 2 for i in range(n)]} for n in (3, 5, 6, 7)],
                'accounting': {'before_probe': accounting, 'after_probe': accounting}}
            (directory / f'ragged-replay-rank-{rank}.json').write_text(json.dumps(row))
            startup = {'rank': rank, 'world': 8, 'state_count': 24, 'capability': [7, 0],
                'tool_sha256': sources['observer_tool'], 'runtime_cli_sha256': files['src/tensorfold/cli.py']['sha256'],
                'history_host_obligation': {'global_bytes': 123, 'input_sha256': self.spec['oracle']['sha256']}}
            (directory / f'rank-{rank}.json').write_text(json.dumps(startup))
        frozen = {'files': files}
        before = gate.collect(out, self.spec, frozen)
        self.assertEqual(len(before), 8)
        self.assertIn('startup', before[0])
        gate.unchanged_receipts(out, self.spec, frozen, before)
        path = directory / 'rank-7.json'
        row = json.loads(path.read_text())
        row['extra_valid_metadata'] = 'changed after publication'
        path.write_text(json.dumps(row))
        with self.assertRaisesRegex(ValueError, 'changed after publication'):
            gate.unchanged_receipts(out, self.spec, frozen, before)

    def run_owned_failure(self, *, publication_failure=False):
        events = []
        class Owner:
            def __init__(self):
                self.cid = None
                self.errors = [None] * 6
                self.state = {'Status': 'exited', 'Pid': 0, 'Running': False, 'ExitCode': 0, 'OOMKilled': False}
            def create(self, _argv, _publish):
                self.cid = '1' * 64
                events.append('acquired')
                raise RuntimeError('publication failed after acquisition')
            def retire(self):
                events.append('retired')
            def exit_code(self, primary):
                return int(primary is not None)
        frozen = {'source_uid': self.spec['source']['source_uid'], 'files': {
            'tools/' + name: self.spec['controller_closure'][name] for name in gate.HELPERS if name != 'v100_ragged_replay_gate.py'}}
        bundle = {'cpuset_cpus': '0-63', 'cpuset_mems': '0,1', 'files': {'manifest.json': json.dumps({
            'provenance': {'container': {'image': self.spec['image']}}}).encode()}}
        def command(argv):
            self.assertEqual(argv[:2], ['docker', 'rm'])
            events.append('remove failed')
            raise RuntimeError('removal failed')
        original_publish = gate.campaign.publish
        def publish(path, value):
            if publication_failure and path.name == 'result.json':
                raise OSError('result publication failed')
            return original_publish(path, value)
        with patch.object(gate.campaign, 'frozen', return_value=frozen), \
                patch.object(gate.campaign, 'immutable_image', return_value={'id': self.spec['image']}), \
                patch.object(gate.campaign, 'installed_check_receipt', return_value={}), \
                patch.object(gate.campaign, '_numa_inputs', return_value=bundle), \
                patch.object(gate.campaign, '_stage_numa'), patch.object(gate.campaign, 'verify_source'), \
                patch.object(gate.campaign, 'logged'), patch.object(gate, 'health', return_value={}), \
                patch.object(gate, 'validate_operation'), patch.object(gate, 'command', side_effect=command), \
                patch.object(gate.campaign, 'publish', side_effect=publish):
            out = self.root / 'failure-out'
            self.spec['output'] = str(out)
            if publication_failure:
                with self.assertRaisesRegex(RuntimeError, 'publication failed after acquisition') as failure:
                    gate.run(self.spec, out, root_go=self.go(), operation_sha='f' * 64, owner_factory=Owner)
                causes = failure.exception.__cause__
                self.assertIsNotNone(causes)
                self.assertIn('removal failed', str(causes.exceptions))
                self.assertIn('result publication failed', str(causes.exceptions))
            else:
                self.assertEqual(gate.run(self.spec, out, root_go=self.go(), operation_sha='f' * 64, owner_factory=Owner), 1)
                result = json.loads((out / 'result.json').read_text())
                self.assertFalse(result['removed'])
                self.assertIn('publication failed', result['primary'])
                self.assertTrue(any('removal failed' in error for error in result['cleanup']))
        self.assertEqual(events, ['acquired', 'retired', 'remove failed'])

    def test_owned_acquisition_failure_still_retires_and_reports_removal_failure(self):
        self.run_owned_failure()

    def test_final_publication_failure_preserves_original_and_cleanup_failures(self):
        self.run_owned_failure(publication_failure=True)

    def test_unsealed_output_cannot_reuse_root_go(self):
        with patch.object(gate.campaign, 'frozen') as frozen:
            with self.assertRaisesRegex(ValueError, 'one-attempt'):
                gate.run(self.spec, self.root / 'different', root_go=self.go(), operation_sha='f' * 64)
            frozen.assert_not_called()



if __name__ == '__main__':
    unittest.main()
