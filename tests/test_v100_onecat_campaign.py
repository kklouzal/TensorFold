"""Campaign artifact boundaries; no Docker, SDK or GPU emulation."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools import v100_onecat_campaign as campaign


class HCExportStagingTests(unittest.TestCase):
    def prepare(self, root):
        from tests.test_bitblas_library import manifest

        frozen = root / "frozen"
        relative = "src/tensorfold/cuda/sm70/bitblas_library.py"
        helper = frozen / "source" / relative
        helper.parent.mkdir(parents=True)
        helper.write_bytes((Path(campaign.__file__).parent.parent / relative).read_bytes())
        helper.chmod(0o444)
        source = {"source_uid": "c" * 64,
                  "files": {relative: {"type": "file", **campaign.identity(helper)}}}
        exports = root / "exports"
        exports.mkdir()
        path, _ = manifest(exports)
        sha = campaign.identity(path)["sha256"]
        settings = {**ModelArmTests().settings(), "bitblas_hc_exports": str(exports)}
        profile = {"preparation": {"dense_projections": {
            "backend": "bitblas-hc", "manifest": "/operator/bitblas-hc/manifest.json",
            "manifest_sha256": sha, "roles": ["decode", "prefill"]}}}
        return frozen, source, settings, profile

    def test_sealed_closure_uses_frozen_stdlib_validator_and_copies_readonly(self):
        from tensorfold.cuda.sm70 import bitblas_library

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen, source, settings, profile = self.prepare(root)
            with patch("ctypes.CDLL", side_effect=AssertionError("native load")), \
                    patch.dict(sys.modules, {"torch": None}), \
                    patch.object(bitblas_library, "hc_manifest", side_effect=AssertionError("live helper")):
                bundle = campaign._hc_exports(frozen, source, settings, profile)
                operator = root / "operator"
                operator.mkdir()
                receipt = campaign._stage_hc_exports(bundle, operator)
            self.assertEqual(receipt["closure_files"], 10)
            self.assertEqual(receipt["files"], receipt["copied_files"])
            self.assertEqual(receipt["validator"]["source_uid"], source["source_uid"])
            self.assertEqual(receipt["payload_bytes"], sum(v["bytes"] for v in receipt["files"].values()))
            for path in (operator / "bitblas-hc").rglob("*"):
                self.assertEqual(path.stat().st_mode & 0o777, 0o555 if path.is_dir() else 0o444)
            self.assertEqual((operator / "bitblas-hc").stat().st_mode & 0o777, 0o555)
            self.assertEqual(campaign.model_arguments(settings, ModelArmTests().plan()),
                             campaign.model_arguments(ModelArmTests().settings(), ModelArmTests().plan()))

    def test_missing_unused_and_wrong_runtime_binding_fail_before_staging(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen, source, settings, profile = self.prepare(root)
            for supplied, declared in [(settings, {}), (ModelArmTests().settings(), profile),
                                      (settings, {"preparation": {"dense_projections": {
                                          **profile["preparation"]["dense_projections"], "manifest": "/host/manifest.json"}}})]:
                with self.subTest(supplied=supplied, declared=declared), self.assertRaises(ValueError):
                    campaign._hc_exports(frozen, source, supplied, declared)
            self.assertIsNone(campaign._hc_exports(frozen, source, ModelArmTests().settings(), {}))

    def test_extra_entries_symlinks_stale_seal_and_changed_validator_are_refused(self):
        for defect in ("extra", "file-link", "directory-link", "root-link", "seal", "validator"):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                frozen, source, settings, profile = self.prepare(root)
                exports = Path(settings["bitblas_hc_exports"])
                if defect == "extra":
                    (exports / "unlisted").write_text("extra")
                elif defect == "file-link":
                    file = exports / "hc-down/library.so"
                    file.rename(root / "outside.so")
                    file.symlink_to(root / "outside.so")
                elif defect == "directory-link":
                    (exports / "hc-down").rename(root / "outside")
                    (exports / "hc-down").symlink_to(root / "outside", target_is_directory=True)
                elif defect == "root-link":
                    (root / "link").symlink_to(exports, target_is_directory=True)
                    settings["bitblas_hc_exports"] = str(root / "link")
                elif defect == "seal":
                    profile["preparation"]["dense_projections"]["manifest_sha256"] = "0" * 64
                else:
                    helper = frozen / "source/src/tensorfold/cuda/sm70/bitblas_library.py"
                    helper.chmod(0o644)
                    helper.write_text("raise AssertionError('changed validator executed')")
                with self.assertRaises(ValueError):
                    campaign._hc_exports(frozen, source, settings, profile)

    def test_source_growth_after_validation_stops_bounded_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen, source, settings, profile = self.prepare(root)
            bundle = campaign._hc_exports(frozen, source, settings, profile)
            with (Path(settings["bitblas_hc_exports"]) / "hc-down/library.so").open("ab") as stream:
                stream.write(b"x")
            operator = root / "operator"
            operator.mkdir()
            with self.assertRaisesRegex(ValueError, "grew during bounded copy"):
                campaign._stage_hc_exports(bundle, operator)

    def test_copied_payload_is_validated_again_before_sealing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen, source, settings, profile = self.prepare(root)
            bundle = campaign._hc_exports(frozen, source, settings, profile)
            operator = root / "operator"
            operator.mkdir()
            validate = campaign._hc_closure

            def corrupt_copy(directory, library, seal):
                if directory == operator / "bitblas-hc":
                    file = directory / "hc-up/library.so"
                    file.chmod(0o644)
                    file.write_bytes(b"x" * file.stat().st_size)
                return validate(directory, library, seal)

            with patch.object(campaign, "_hc_closure", side_effect=corrupt_copy), \
                    self.assertRaisesRegex(ValueError, "artifact bytes changed"):
                campaign._stage_hc_exports(bundle, operator)


class SourceCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "repo"
        self.repo.mkdir()
        subprocess.run(["git", "init", "-q", str(self.repo)], check=True)
        (self.repo / "file.py").write_text("original\n")
        (self.repo / ".gitignore").write_text("task-artifacts/\nsecret\n")
        subprocess.run(["git", "-C", str(self.repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(self.repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "commit", "-qm", "baseline"], check=True)

    def test_capture_binds_dirty_bytes_and_excludes_ignored_private_inputs(self):
        (self.repo / "file.py").write_text("current\n")
        (self.repo / "new.py").write_text("new owned file\n")
        (self.repo / "secret").write_text("excluded\n")
        output = self.root / "capture"
        record = campaign.freeze(self.repo, output)
        self.assertEqual((output / "source/file.py").read_text(), "current\n")
        self.assertEqual(set(record["files"]), {".gitignore", "file.py", "new.py"})
        self.assertIn("file.py", record["status"])
        self.assertEqual(campaign.frozen(output)["source_uid"], record["source_uid"])

    def test_capture_rejects_symlink_to_external_or_mutable_absolute_path(self):
        (self.repo / "link").symlink_to(self.repo / "file.py")
        with self.assertRaisesRegex(ValueError, "escapes"):
            campaign.freeze(self.repo, self.root / "capture")

    def test_manifest_and_payload_mutations_cannot_reuse_source_identity(self):
        output = self.root / "capture"
        campaign.freeze(self.repo, output)
        manifest = json.loads((output / "source.json").read_text())
        manifest["files"]["file.py"]["sha256"] = "0" * 64
        (output / "source.json").write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ValueError, "inventory identity"):
            campaign.frozen(output)

    def test_frozen_build_context_refuses_unlisted_inputs_or_write_permissions(self):
        output = self.root / "capture"
        campaign.freeze(self.repo, output)
        source = output / "source"
        source.chmod(0o755)
        (source / "injected.py").write_text("unlisted build input\n")
        with self.assertRaisesRegex(ValueError, "unlisted"):
            campaign.frozen(output)
        (source / "injected.py").unlink()
        (source / "file.py").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "permissions"):
            campaign.frozen(output)

    def test_symlink_cannot_borrow_an_ignored_target(self):
        (self.repo / "secret").write_text("not captured\n")
        (self.repo / "link").symlink_to("secret")
        with self.assertRaisesRegex(ValueError, "absent from the captured inventory"):
            campaign.freeze(self.repo, self.root / "capture")


class RetirementTests(unittest.TestCase):
    def test_attachment_failure_retires_container_and_retains_primary(self):
        primary = TimeoutError("attached process timed out")
        class Owner:
            cid = "a" * 64
            state = {"Running": False}
            errors = []
            retired = False

            def create(self, argv, publish):
                return self.cid

            def retire(self):
                self.retired = True

            def exit_code(self, error):
                return int(error is not None)

        owner = Owner()
        with tempfile.TemporaryDirectory() as temporary, patch.object(campaign, "OwnedContainer", return_value=owner), \
                patch.object(campaign, "logged", side_effect=primary):
            with self.assertRaises(TimeoutError) as raised:
                campaign.container(["docker", "create"], Path(temporary), 1)
            self.assertIs(raised.exception, primary)
            self.assertTrue(owner.retired)
            record = json.loads((Path(temporary) / "container-result.json").read_text())
            self.assertEqual(record["exit_code"], 1)


class ModelArmTests(unittest.TestCase):
    def settings(self):
        return {"gpus": [str(i) for i in range(8)], "models_root": "/models", "target": "target.gguf",
                "mtp": "mtp.gguf", "profile_manifest": "/operator/profile.json", "matrix_plan": "/operator/matrix.json",
                "cache": "/cache", "gu": "incumbent", "down": "tensorfold", "device_budget_bytes": 117976064,
                "global_host_bytes": 67108864, "max_waiting_input_bytes": 226493088,
                "prefill_rows": 4096, "execution_rows": 120, "mtp_confidence": .7,
                "graphs": True, "iq2_source": "active-frame", "prefill_method": "tensorfold"}

    def plan(self):
        return {"schema": "tensorfold.phase-matrix-plan-v1", "profile": "q8", "admin_cap": 24,
                "context": 262144, "model": "comparison"}

    def test_complete_incumbent_uses_original_chain_and_requested_MTP4_graphs(self):
        argv = campaign.model_arguments(self.settings(), self.plan())
        self.assertIn("--grouped-gate-up-rows", argv)
        self.assertNotIn("--onecat-gu-method", argv)
        self.assertNotIn("--no-graphs", argv)
        self.assertEqual(argv[argv.index("--mtp-drafts") + 1], "4")
        self.assertEqual(argv[argv.index("--execution-rows") + 1], "120")
        self.assertEqual(argv[argv.index("--expert-tensor-parallel-size") + 1], "4")

    def test_validated_numa_settings_reach_supervisor_before_serve_separator(self):
        settings = {**self.settings(), "numa": {
            "plan": "/sealed/plan.json", "plan_sha256": "a" * 64,
            "seccomp_directory": "/sealed/seccomp", "seccomp_manifest_sha256": "b" * 64}}
        argv = campaign.model_arguments(settings, self.plan(), smoke_only=True)
        supervisor, serve = argv[:argv.index("--")], argv[argv.index("--") + 1:]
        self.assertEqual(supervisor[supervisor.index("--numa-plan") + 1], "/operator/numa/plan.json")
        self.assertEqual(supervisor[supervisor.index("--numa-plan-sha256") + 1], "a" * 64)
        self.assertNotIn("--numa-plan", serve)
        self.assertNotIn("--numa-plan-sha256", serve)
        self.assertEqual(serve[serve.index("--mtp-drafts") + 1], "4")
        with self.assertRaisesRegex(ValueError, "complete explicit campaign settings"):
            campaign.model_arguments({**settings, "unknown_setting": True}, self.plan())

    def test_raw_candidate_is_explicit_and_smoke_cannot_enter_timing_observer(self):
        settings = {**self.settings(), "gu": "tensorfold-fp32-epilogue", "down": "onecat-q8-slots",
                    "iq2_source": "raw", "graphs": False, "prefill_method": "candidate"}
        argv = campaign.model_arguments(settings, self.plan(), smoke_only=True)
        self.assertNotIn("--benchmark-plan", argv)
        self.assertIn("--no-graphs", argv)
        self.assertEqual(argv[argv.index("--onecat-iq2-source") + 1], "raw")
        self.assertEqual(argv[argv.index("--onecat-down-method") + 1], "onecat-q8-slots")
        self.assertEqual(argv[argv.index("--onecat-prefill-method") + 1], "candidate")

    def test_incumbent_cannot_silently_select_approximate_down(self):
        with self.assertRaisesRegex(ValueError, "complete TensorFold"):
            campaign.model_arguments({**self.settings(), "down": "onecat-q8-slots"}, self.plan())

    def test_incumbent_prefill_stays_separate_from_explicit_candidate_prefill(self):
        with self.assertRaisesRegex(ValueError, "retain TensorFold prefill"):
            campaign.model_arguments({**self.settings(), "prefill_method": "candidate"}, self.plan())
        argv = campaign.model_arguments({**self.settings(), "gu": "tensorfold", "prefill_method": "candidate"}, self.plan())
        self.assertEqual(argv[argv.index("--onecat-prefill-method") + 1], "candidate")


class GraphUseTests(unittest.TestCase):
    def prepare(self, root, *, graphs=True):
        directory = root / "qualification"
        directory.mkdir()
        files = {"engine": "src/tensorfold/families/qwen4_exp/cuda/engine.py",
                 "observer_tool": "tools/v100_owned_qualification.py"}
        if graphs:
            files.update(graph_controller="src/tensorfold/families/qwen4_exp/cuda/multi_graphs.py",
                         graph_policy="src/tensorfold/families/qwen4_exp/cuda/exact_graphs.py")
        sources = {key: "a" * 64 for key in files}
        for rank in range(8):
            startup = directory / f"rank-{rank}.json"
            startup.write_text(json.dumps({"rank": rank}))
            use = {"requested": graphs, "prepared": graphs, "serving_hits": None, "serving_misses": None}
            if graphs:
                use.update(capture_during_serving=False, serving_hits={"verify": 2, "mtp": 4},
                           serving_misses={"verify": 0, "mtp": 0})
            record = {"schema": "tensorfold.graph-use-rank-v1", "rank": rank, "world": 8,
                      "closed": True, "failed": False, "mtp_depth": 4, "mtp_active": True,
                      "primary_error_type": None, "snapshot_error_type": None, "runtime_sources": sources,
                      "startup_receipt_sha256": campaign.identity(startup)["sha256"], "graph_use": use,
                      "selector": {"gu": "incumbent", "grouped_gate_up_rows": 4,
                                   "down": "tensorfold", "iq2_source": "not-applicable", "prefill_method": "tensorfold"}}
            (directory / f"graph-use-rank-{rank}.json").write_text(json.dumps(record))
        return {"files": {name: {"sha256": "a" * 64} for name in files.values()}}, {
            "graphs": graphs, "gu": "incumbent", "down": "tensorfold", "iq2_source": "active-frame", "prefill_method": "tensorfold"}

    def test_all_eight_actual_serving_roles_are_required_separately_from_startup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, settings = self.prepare(root)
            self.assertEqual(len(campaign.graph_use_receipts(root, source, settings)), 8)
            last = root / "qualification/graph-use-rank-7.json"
            record = json.loads(last.read_text())
            record["graph_use"]["serving_hits"]["mtp"] = 0
            last.write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "both replay"):
                campaign.graph_use_receipts(root, source, settings)

    def test_changed_startup_receipt_cannot_reuse_closed_graph_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, settings = self.prepare(root)
            (root / "qualification/rank-2.json").write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, "source/rank/MTP"):
                campaign.graph_use_receipts(root, source, settings)

    def test_eager_control_is_positive_without_fabricated_replay_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, settings = self.prepare(root, graphs=False)
            records = campaign.graph_use_receipts(root, source, settings)
            self.assertTrue(all(record["graph_use"]["serving_hits"] is None for record in records))


class BindingValidationTests(unittest.TestCase):
    def fixture(self):
        manifest = {'schema': 'tensorfold.execution-profile-manifest', 'version': 1,
                    'artifacts': {'target/a.gguf': {'size': 100, 'sha256': 'a' * 64},
                                  'metadata/tokenizer.json': {'size': 23, 'sha256': 'b' * 64}}}
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                allow_nan=False, separators=(',', ':')).encode()).hexdigest()
        binding = {'manifest_sha256': digest(manifest), 'content_passes': 1, 'verified_input_files': 2,
                   'verified_input_bytes': 123, 'bytes_read_this_binding': 140,
                   'hashes_reused_this_binding': 7, 'shared_immutable_same_host': True}
        return manifest, binding, digest

    def test_formatted_manifest_uses_canonical_identity_and_accepts_valid_alias_reuse(self):
        manifest, binding, digest = self.fixture()
        manifest = json.loads(json.dumps(manifest, indent=4))
        self.assertNotEqual(hashlib.sha256(json.dumps(manifest, indent=4).encode()).hexdigest(), binding['manifest_sha256'])
        result = campaign.validate_execution_binding(binding, manifest, digest=digest)
        self.assertEqual(result['verified_input_bytes'], 123)
        self.assertEqual(result['hashes_reused_this_binding'], 7)
        self.assertEqual(result['bytes_read_this_binding'], 140)

    def test_changed_manifest_pass_count_file_count_or_byte_count_refuses(self):
        manifest, binding, digest = self.fixture()
        for field, value in (('manifest_sha256', 'c' * 64), ('content_passes', 0), ('content_passes', True),
                             ('verified_input_files', 1), ('verified_input_bytes', 122),
                             ('shared_immutable_same_host', False)):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(ValueError, 'verification differs'):
                    campaign.validate_execution_binding({**binding, field: value}, manifest, digest=digest)
        changed = copy.deepcopy(manifest)
        changed['artifacts']['target/a.gguf']['size'] += 1
        with self.assertRaisesRegex(ValueError, 'verification differs'):
            campaign.validate_execution_binding(binding, changed, digest=digest)

    def test_negative_or_boolean_read_reuse_observations_are_not_valid_counts(self):
        manifest, binding, digest = self.fixture()
        for field in ('bytes_read_this_binding', 'hashes_reused_this_binding'):
            for value in (-1, False):
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, 'nonnegative integers'):
                        campaign.validate_execution_binding({**binding, field: value}, manifest, digest=digest)


class CommitDiagnosticTests(unittest.TestCase):
    def plan(self):
        cohorts = []
        for identifier, level, warmup in ((0, 1, True), (1, 1, False), (2, 2, False), (3, 8, False)):
            cohorts.append({'id': identifier, 'level': level, 'warmup': warmup, 'input_tokens': 1024,
                            'tokens': 16 if warmup else 128, 'repetition': 0,
                            'requests': [{'prefix': [identifier, index], 'prompt_sha256': f'p{identifier}-{index}'}
                                         for index in range(level)]})
        return {**ModelArmTests().plan(), 'levels': [1, 2, 8], 'inputs': [1024], 'reps': 3,
                'cohorts': cohorts, 'excluded_levels': [32], 'run_identity': 'a' * 32}

    def test_diagnostic_forwards_finite_markers_without_closed_client_or_teacher(self):
        spec = campaign.commit_diagnostic_spec(1024, True)
        self.assertTrue(spec['diagnostic_only'])
        self.assertFalse(spec['P3_valid'])
        derived = campaign.diagnostic_c1_c8_plan(self.plan())
        argv = campaign.model_arguments(ModelArmTests().settings(), derived,
            diagnostic_commit_events=1024, diagnostic_commit_nvtx=True)
        self.assertEqual(argv[argv.index('--commit-trace-events') + 1], '1024')
        self.assertIn('--commit-trace-nvtx', argv)
        self.assertIn('--benchmark-plan', argv)
        self.assertNotIn('--benchmark-client', argv)
        self.assertNotIn('--teacher-plan', argv)
        smoke = campaign.model_arguments(ModelArmTests().settings(), self.plan(), smoke_only=True,
                                         diagnostic_commit_events=1024)
        self.assertNotIn('--benchmark-plan', smoke)
        self.assertNotIn('--commit-trace-nvtx', smoke)

    def test_representative_c1_c8_plan_preserves_exact_canonical_requests_and_detaches_data(self):
        original = self.plan()
        before = copy.deepcopy(original)
        derived = campaign.diagnostic_c1_c8_plan(original)
        self.assertEqual(original, before)
        self.assertEqual(derived['levels'], [1, 8])
        self.assertEqual(derived['reps'], 1)
        self.assertEqual([row['id'] for row in derived['cohorts']], [0, 1, 3])
        self.assertEqual(derived['cohorts'][2]['requests'], original['cohorts'][3]['requests'])
        derived['cohorts'][2]['requests'][0]['prefix'].append(999)
        self.assertEqual(original, before)
        with self.assertRaisesRegex(ValueError, 'separate finite C1/C8'):
            campaign.model_arguments(ModelArmTests().settings(), original, diagnostic_commit_events=1024)
        original['cohorts'] = [row for row in original['cohorts'] if row['level'] != 8]
        with self.assertRaisesRegex(ValueError, 'complete 1024-input/128-output C1 and C8'):
            campaign.diagnostic_c1_c8_plan(original)

    def test_invalid_event_or_nvtx_options_fail_before_model_or_filesystem_work(self):
        for events, nvtx in ((None, True), (0, False), (-1, False), (True, False)):
            with self.subTest(events=events, nvtx=nvtx), patch.object(campaign, 'frozen') as frozen:
                with self.assertRaises(ValueError):
                    campaign.model(None, None, None, None, None, 1, '1g', 1, 1, True, events, nvtx)
                frozen.assert_not_called()
        with patch.object(campaign, 'model') as selected:
            self.assertEqual(campaign.main(['model', '--source', '/source', '--out', '/out', '--image', 'image',
                '--settings', '/settings.json', '--installed-check', '/checked', '--smoke-only',
                '--diagnostic-commit-events', '4096', '--diagnostic-commit-nvtx']), 0)
            self.assertEqual(selected.call_args.args[-3:], (True, 4096, True))

    def test_trace_result_requires_all_closed_ranks_and_preserves_batch_counts_not_per_token_times(self):
        spec = campaign.commit_diagnostic_spec(16, False)
        source = {'files': {'tools/v100_owned_qualification.py': {'sha256': 'a' * 64}}}
        from tools.v100_commit_trace import COLUMNS
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'qualification').mkdir()
            for rank in range(8):
                record = {'schema': 'tensorfold.commit-batch-diagnostic-v1', 'rank': rank, 'world': 8,
                          'closed': True, 'complete': True, 'failure': None, 'engine_closed': True,
                          'engine_aborted': False, 'diagnostic_only': True, 'P3_valid': False,
                          'columns': list(COLUMNS), 'observer_source_sha256': 'a' * 64,
                          'requested': spec, 'events': [[17, 100, 160, 0, 4, 4]] if rank == 0 else []}
                (root / 'qualification' / f'commit-trace-rank-{rank}.json').write_text(json.dumps(record))
            result = campaign.commit_trace_receipts(root, source, spec)
            self.assertEqual(result[0]['events'], 1)
            self.assertEqual(result[0]['committed_delta'], 4)
            self.assertTrue(all(row['P3_valid'] is False for row in result))
            path = root / 'qualification/commit-trace-rank-7.json'
            invalid = json.loads(path.read_text())
            invalid['P3_valid'] = True
            path.write_text(json.dumps(invalid))
            with self.assertRaisesRegex(ValueError, 'positive bound closed-rank'):
                campaign.commit_trace_receipts(root, source, spec)


class CublasWorkspaceConfigurationTests(unittest.TestCase):
    def test_only_explicit_supported_values_forward_one_whitelisted_environment(self):
        self.assertEqual(campaign.cublas_workspace_configuration("inherit")["environment"], {})
        self.assertEqual(campaign.cublas_workspace_configuration("on")["environment"],
                         {"TORCH_CUBLAS_WORKSPACE_CACHE": "1"})
        self.assertEqual(campaign.cublas_workspace_configuration("off")["environment"],
                         {"TORCH_CUBLAS_WORKSPACE_CACHE": "0"})
        for value in (None, True, 1, "1", "ON", "on\nEVIL=1"):
            with self.subTest(value=value), patch.object(campaign, "frozen") as frozen:
                with self.assertRaisesRegex(ValueError, "inherit, on or off"):
                    campaign.model(None, None, None, None, None, 1, "1g", 1, 1, True,
                                   cublas_workspace_cache=value)
                frozen.assert_not_called()

    def test_cli_preserves_default_and_forwards_explicit_configuration_separately(self):
        argv = ["model", "--source", "/source", "--out", "/out", "--image", "image",
                "--settings", "/settings.json", "--installed-check", "/checked", "--smoke-only"]
        for selection in ("inherit", "on", "off"):
            with self.subTest(selection=selection), patch.object(campaign, "model") as model:
                extra = [] if selection == "inherit" else ["--cublas-workspace-cache", selection]
                self.assertEqual(campaign.main(argv + extra), 0)
                self.assertEqual(model.call_args.kwargs, {"cublas_workspace_cache": selection, "diagnostic_graph_pieces": False})
                self.assertEqual(model.call_args.args[-3:], (True, None, False))


class GraphPieceDiagnosticTests(unittest.TestCase):
    def test_cli_explicit_piece_diagnostic_is_separate_from_workspace_selection(self):
        argv = ["model", "--source", "/source", "--out", "/out", "--image", "image",
                "--settings", "/settings.json", "--installed-check", "/checked", "--smoke-only",
                "--diagnostic-graph-pieces"]
        with patch.object(campaign, "model") as model:
            self.assertEqual(campaign.main(argv), 0)
            self.assertEqual(model.call_args.kwargs,
                {"cublas_workspace_cache": "inherit", "diagnostic_graph_pieces": True})

    def test_non_boolean_selection_fails_before_source_or_container_access(self):
        for value in (1, "1", None):
            with self.subTest(value=value), patch.object(campaign, "frozen") as frozen:
                with self.assertRaisesRegex(ValueError, "must be boolean"):
                    campaign.model(None, None, None, None, None, 1, "1g", 1, 1, True,
                                   diagnostic_graph_pieces=value)
                frozen.assert_not_called()


class FiniteSmokeCoverageTests(unittest.TestCase):
    def run_client(self, *, short_bad=False, continuation_short=False, continuation_drift=False):
        import contextlib
        import io
        requests = []
        publications = []
        def respond(request, timeout):
            body = json.loads(request.data)
            self.assertEqual(timeout, 180)
            requests.append(body)
            if len(requests) <= 2:
                content, ids = ('5' if short_bad else '4'), [19, 248046]
            elif len(requests) <= 4:
                content, ids = 'READY', [44061, 248046]
            else:
                content = 'A controlled continuation fixture.'
                ids = list(range(2 if continuation_short else 16))
                if continuation_drift and body['draft']:
                    ids[-1] += 1
            payload = {'choices': [{'message': {'content': content}}], 'tensorfold': {'token_ids': ids}}
            return io.BytesIO(json.dumps(payload).encode())
        with patch('urllib.request.urlopen', side_effect=respond), patch('sys.argv', ['python', 'fixture-model']), \
                patch.object(Path, 'write_text', side_effect=lambda value: publications.append(json.loads(value))), \
                contextlib.redirect_stdout(io.StringIO()):
            exec(compile(campaign._SMOKE, '<finite smoke client>', 'exec'), {})
        return requests, publications

    def test_short_known_answers_and_long_pair_preserve_draft_and_token_bounds(self):
        requests, publications = self.run_client()
        self.assertEqual(len(requests), 6)
        self.assertEqual([x['draft'] for x in requests], [False, True] * 3)
        self.assertEqual([x['max_tokens'] for x in requests], [16] * 4 + [64] * 2)
        self.assertEqual([len(x) for x in publications], list(range(1, 7)))
        self.assertEqual(requests[4]['messages'], requests[5]['messages'])

    def test_original_known_answer_contract_remains_required(self):
        with self.assertRaisesRegex(RuntimeError, 'known-answer'):
            self.run_client(short_bad=True)

    def test_early_continuation_eos_does_not_claim_refresh_coverage(self):
        with self.assertRaisesRegex(RuntimeError, 'required token window'):
            self.run_client(continuation_short=True)

    def test_long_serial_and_mtp_token_ids_must_match(self):
        with self.assertRaisesRegex(RuntimeError, 'serial/MTP greedy'):
            self.run_client(continuation_drift=True)


class NativeFamilyInputTests(unittest.TestCase):
    def test_cli_passes_explicit_numerical_choice_to_manifest_generator(self):
        argv = ['inputs', '--source', '/source', '--out', '/out', '--image', 'image',
                '--installed-check', '/check', '--models', '/models', '--target', 'target',
                '--mtp', 'mtp', '--cache', '/cache', '--name', 'model']
        for requested in (None, 'owned-native-family-v1'):
            with patch.object(campaign, 'inputs') as selected:
                campaign.main(argv + ([] if requested is None else ['--native-family', requested]))
                self.assertEqual(selected.call_args.kwargs['native_family'], requested)
        with patch.object(campaign, 'inputs'), self.assertRaises(SystemExit):
            campaign.main(argv + ['--native-family', 'optimized'])


if __name__ == "__main__":
    unittest.main()


class SmokeBeforeMatrixTests(unittest.TestCase):
    def test_cli_selects_combined_workflow_without_changing_default(self):
        argv = ["model", "--source", "/source", "--out", "/out", "--image", "image",
                "--settings", "/settings.json", "--installed-check", "/checked"]
        with patch.object(campaign, "model") as model:
            self.assertEqual(campaign.main(argv + ["--smoke-before-matrix"]), 0)
            self.assertIs(model.call_args.kwargs["smoke_before_matrix"], True)
            self.assertIs(model.call_args.args[-3], False)
        with patch.object(campaign, "model") as model:
            self.assertEqual(campaign.main(argv), 0)
            self.assertNotIn("smoke_before_matrix", model.call_args.kwargs)

    def test_invalid_or_instrumented_combinations_fail_before_loading(self):
        cases = [
            {"smoke_before_matrix": 1},
            {"smoke_before_matrix": True, "diagnostic_commit_events": 32},
            {"smoke_before_matrix": True, "scheduler_diagnostic_grants": Path("/unused")},
        ]
        for selection in cases:
            with self.subTest(selection=selection), patch.object(campaign, "frozen") as frozen:
                with self.assertRaisesRegex(ValueError, "smoke-before-matrix"):
                    campaign.model(None, None, None, None, None, 1, "1g", 1, 1, False, **selection)
                frozen.assert_not_called()
        with patch.object(campaign, "frozen") as frozen:
            with self.assertRaisesRegex(ValueError, "matrix mode"):
                campaign.model(None, None, None, None, None, 1, "1g", 1, 1, True,
                               smoke_before_matrix=True)
            frozen.assert_not_called()
