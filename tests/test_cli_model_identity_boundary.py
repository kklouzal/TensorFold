"""Complete owned CLI content-identity path, with explicit SDK-free seams."""

from contextlib import redirect_stdout
from importlib import metadata
import io
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from snapshot_fd_transport import TransportOwner
from tensorfold import cli, engine, families
from tensorfold.engine import model_closure


class Captured(Exception):
    pass


class IdentityStartup(unittest.TestCase):
    def setUp(self):
        # This metadata-only caller control never loads the native extension.
        substitution = patch("tensorfold.file_io._owned_slot", TransportOwner)
        substitution.start()
        self.addCleanup(substitution.stop)

    def execute(self, base, *, loader=None, trust=False, persistence=True, head=None, owner_journal=None,
                mtp_family=None, mtp_drafts=None, no_drafts=True, head_resolver=None, drafter_choice="auto",
                model_flags=(), tokenizer_journal=None):
        model_dir, runtime_dir = Path(base) / "model", Path(base) / "runtime"
        model_dir.mkdir(exist_ok=True)
        runtime_dir.mkdir(exist_ok=True)
        if not (model_dir / "config.json").exists():
            (model_dir / "config.json").write_text('{"model_type":"owned"}')
        if not (model_dir / "tokenizer.json").exists():
            (model_dir / "tokenizer.json").write_text('{"vocab":["a","b"]}')
        if not (model_dir / "model.safetensors").exists():
            (model_dir / "model.safetensors").write_bytes(b"opaque model payload")
        if not (runtime_dir / "runtime.py").exists():
            (runtime_dir / "runtime.py").write_text("value = 1\n")
        args = cli.build_parser().parse_args(
            [
                "serve",
                str(model_dir),
                *(["--no-drafts"] if no_drafts else []),
                "--prompt-cache-gib",
                "0",
                "--snapshot-dir",
                str(Path(base) / "snapshots") if persistence else "none",
                *model_flags,
            ]
        )
        args.trust_model_code = trust
        args.mtp_drafts = mtp_drafts
        args.drafter = drafter_choice
        model = SimpleNamespace(layers=[], release_rounds=lambda: None)
        options_seen, ids = [], []
        owners = {} if owner_journal is None else owner_journal

        def load(path, **options):
            options_seen.append(options)
            if loader:
                loader(Path(path), runtime_dir)
            tokenizer = SimpleNamespace(eos_token_ids=(0,))
            if tokenizer_journal is not None:
                tokenizer_journal.append(tokenizer)
            return model, tokenizer

        package = SimpleNamespace(load=load, MLX_MODEL_FILE=True, __name__="owned.family")
        if mtp_family == "qwen3_5_moe":
            package.__name__ = "tensorfold.families.qwen3_5_moe"
        elif head is not None:
            package.__name__ = "tensorfold.families.nemotron_h"
        family = SimpleNamespace(title="owned metadata", model_type=mtp_family or "owned", package=package)
        core, mlx = ModuleType("mlx.core"), ModuleType("mlx")
        core.__version__ = "owned-host-1"
        mlx.core = core
        lane, plan, choose, prompt, residency, app, nemotron, qwen = [
            ModuleType(name)
            for name in (
                "tensorfold.engine.lane_engine",
                "tensorfold.engine.prefill_plan",
                "tensorfold.engine.prefill_step",
                "tensorfold.server.prompt_memory",
                "tensorfold.server.residency",
                "tensorfold.server.app",
                "tensorfold.families.nemotron_h.model",
                "tensorfold.families.qwen3_5_moe",
            )
        ]

        class Lane:
            prefill_step = 64

        lane.LaneEngine = Lane
        plan.PrefillPlan = lambda step, *unused: SimpleNamespace(name="owned-plan", min_chunk=256)
        plan.message_markers = lambda tokenizer: ((), ())
        choose.choose = lambda *unused: 64
        prompt.probe_tokens = lambda tokenizer: [0]
        residency.wire_resident = lambda *unused: 0

        def capture(*unused, **options):
            ids.append(options["model_id"])
            raise Captured()

        app.ChatApp = capture
        nemotron.find_mtp_head = lambda directory: head
        qwen.mtp_file = head_resolver or (lambda directory: head)
        replacements = {
            module.__name__: module for module in (core, mlx, lane, plan, choose, prompt, residency, app, nemotron, qwen)
        }
        output = io.StringIO()
        with (
            patch.dict(sys.modules, replacements),
            patch.object(engine, "prefill_step", choose, create=True),
            patch.object(families, "kernel_version", return_value="owned-kernels"),
            patch.object(metadata, "version", return_value="owned-provider-1"),
            patch.object(
                model_closure,
                "runtime_closure",
                side_effect=lambda *unused, **kw: model_closure.LoaderClosure({"runtime": runtime_dir}),
            ),
            redirect_stdout(output),
        ):
            try:
                cli._serve_mlx_start(args, family, model_dir, 64, [], 1 << 30, 0.9, owners)
            except Captured:
                pass
        return ids, owners, options_seen, model, output.getvalue()

    def test_model_tokenizer_and_runtime_inputs_have_separate_content_keys(self):
        with tempfile.TemporaryDirectory() as base:
            first = self.execute(base)[0][0]
            same = self.execute(base)[0][0]
            self.assertEqual(first, same)
            (Path(base) / "model/tokenizer.json").write_text('{"vocab":["b","a"]}')
            changed_data = self.execute(base)[0][0]
            self.assertNotEqual(first.split("|", 1)[0], changed_data.split("|", 1)[0])
            (Path(base) / "runtime/runtime.py").write_text("value = 2\n")
            changed_code = self.execute(base)[0][0]
            self.assertEqual(changed_data.split("|", 1)[0], changed_code.split("|", 1)[0])
            self.assertNotEqual(changed_data, changed_code)
            self.assertTrue(first.startswith("model-content-v1:"))

    def test_changed_selected_input_refuses_after_returned_model_is_owned(self):
        with tempfile.TemporaryDirectory() as base:
            owners = {}

            def mutate(path, runtime):
                (path / "tokenizer.json").write_text('{"changed":true}')

            with self.assertRaises(ValueError):
                self.execute(base, loader=mutate, owner_journal=owners)
            self.assertIn("model", owners)
            self.assertIsInstance(owners["model"], SimpleNamespace)

    def test_new_owned_inputs_are_rehashed_with_all_pass_costs_observed(self):
        with tempfile.TemporaryDirectory() as base:

            def generated(path, runtime):
                (path / "generated.safetensors").write_bytes(b"owned derived input")
                (runtime / "new.pyc").write_bytes(b"owned generated bytecode")

            ids, owners, _, model, output = self.execute(base, loader=generated)
            self.assertEqual(len(ids), 1)
            self.assertIs(owners["model"], model)
            for role in ("model", "runtime"):
                self.assertEqual(owners["identity_diagnostics"][role]["passes"], 2)
                self.assertGreater(owners["identity_diagnostics"][role]["bytes_read"], 0)
            self.assertIn("2 passes", output)

    def test_trusted_tokenizer_code_has_process_unique_cache_namespace(self):
        with tempfile.TemporaryDirectory() as base:
            first = self.execute(base, trust=True)[0][0]
            second = self.execute(base, trust=True)[0][0]
            self.assertTrue(first.startswith("custom-code-v1:"))
            self.assertNotEqual(first, second)
            self.assertEqual(first.split(":")[1], second.split(":")[1])

    def test_selected_external_mtp_input_is_bound_and_hashed(self):
        with tempfile.TemporaryDirectory() as base:
            head = Path(base) / "external-mtp.safetensors"
            head.write_bytes(b"owned head 1")
            first, _, options, _, _ = self.execute(base, head=head)
            self.assertEqual(options[0]["mtp_head"], str(head))
            head.write_bytes(b"owned head 2")
            second = self.execute(base, head=head)[0]
            self.assertNotEqual(first[0].split("|", 1)[0], second[0].split("|", 1)[0])

    def test_selected_external_template_binds_bytes_and_operator_selection(self):
        with tempfile.TemporaryDirectory() as base:
            selected = Path(base) / "operator.jinja"
            same_bytes = Path(base) / "other.jinja"
            selected.write_text("operator source one")
            same_bytes.write_text("operator source one")
            providers = []
            flags = ("--chat-template", str(selected))
            first = self.execute(base, model_flags=flags, tokenizer_journal=providers)[0][0]
            self.assertEqual(providers[-1].chat_template, "operator source one")
            self.assertEqual(first, self.execute(base, model_flags=flags)[0][0])

            other = self.execute(base, model_flags=("--chat-template", str(same_bytes)))[0][0]
            self.assertEqual(first.split("|", 1)[0], other.split("|", 1)[0])
            self.assertNotEqual(first.split("|", 1)[1], other.split("|", 1)[1])

            selected.write_text("operator source two")
            changed = self.execute(base, model_flags=flags, tokenizer_journal=providers)[0][0]
            self.assertEqual(providers[-1].chat_template, "operator source two")
            self.assertNotEqual(first.split("|", 1)[0], changed.split("|", 1)[0])
            self.assertEqual(first.split("|", 1)[1], changed.split("|", 1)[1])

    def test_external_template_mutation_refuses_after_returned_model_is_owned(self):
        with tempfile.TemporaryDirectory() as base:
            selected = Path(base) / "operator.jinja"
            selected.write_text("selected bytes")
            owners = {}

            def mutate(path, runtime):
                selected.write_text("changed bytes during loading")

            with self.assertRaises(ValueError):
                self.execute(base, model_flags=("--chat-template", str(selected)), loader=mutate,
                             owner_journal=owners)
            self.assertIn("model", owners)
            self.assertIsInstance(owners["model"], SimpleNamespace)

    def test_named_template_and_explicit_eos_bind_runtime_identity(self):
        with tempfile.TemporaryDirectory() as base:
            model_dir = Path(base) / "model"
            model_dir.mkdir()
            (model_dir / "tokenizer_config.json").write_text(
                '{"chat_template":{"default":"model default","rag":"named source"}}')
            providers = []
            first = self.execute(base, tokenizer_journal=providers)[0][0]
            self.assertEqual(providers[-1].chat_template, "model default")
            self.assertEqual(providers[-1].eos_token_ids, (0,))

            for flags, source, eos in (
                    (("--chat-template-name", "rag"), "named source", (0,)),
                    (("--eos-token-id", "1"), "model default", (1,))):
                with self.subTest(flags=flags):
                    changed = self.execute(base, model_flags=flags, tokenizer_journal=providers)[0][0]
                    self.assertEqual(providers[-1].chat_template, source)
                    self.assertEqual(providers[-1].eos_token_ids, eos)
                    self.assertEqual(first.split("|", 1)[0], changed.split("|", 1)[0])
                    self.assertNotEqual(first.split("|", 1)[1], changed.split("|", 1)[1])

    def test_qwen_external_mtp_is_resolved_once_bound_and_content_hashed(self):
        with tempfile.TemporaryDirectory() as base:
            head = Path(base) / "external-qwen-mtp.safetensors"
            head.write_bytes(b"qwen head 1")
            resolutions = []

            def resolve(directory):
                resolutions.append(directory)
                return head

            first, _, options, _, _ = self.execute(
                base, mtp_family="qwen3_5_moe", mtp_drafts=4, no_drafts=False, head_resolver=resolve,
            )
            self.assertEqual(options[0]["mtp_head"], str(head))
            self.assertEqual(resolutions, [Path(base) / "model"])
            head.write_bytes(b"qwen head 2")
            second = self.execute(
                base, mtp_family="qwen3_5_moe", mtp_drafts=4, no_drafts=False, head_resolver=resolve,
            )[0]
            self.assertNotEqual(first[0].split("|", 1)[0], second[0].split("|", 1)[0])
            self.assertEqual(len(resolutions), 2)

    def test_qwen_mtp_mutation_refuses_after_model_ownership_is_journaled(self):
        with tempfile.TemporaryDirectory() as base:
            head = Path(base) / "external-qwen-mtp.safetensors"
            head.write_bytes(b"qwen selected bytes")
            owners = {}

            def mutate(path, runtime):
                head.write_bytes(b"qwen changed bytes")

            with self.assertRaises(ValueError):
                self.execute(base, mtp_family="qwen3_5_moe", mtp_drafts=4, no_drafts=False,
                             head=head, loader=mutate, owner_journal=owners)
            self.assertIn("model", owners)

    def test_qwen_mtp_absence_is_bound_before_a_late_optional_file_appears(self):
        with tempfile.TemporaryDirectory() as base:
            late = Path(base) / "late-qwen-mtp.safetensors"

            def generated(path, runtime):
                late.write_bytes(b"not a selected loader input")

            _, _, options, _, _ = self.execute(base, mtp_family="qwen3_5_moe", mtp_drafts=4, no_drafts=False,
                                               loader=generated, head_resolver=lambda directory: None)
            self.assertEqual(options[0]["mtp_head"], "")

    def test_qwen_serial_and_zero_depth_do_not_resolve_an_unused_mtp_head(self):
        for no_drafts, depth in ((True, None), (True, 4), (False, 0)):
            with self.subTest(no_drafts=no_drafts, depth=depth), tempfile.TemporaryDirectory() as base:

                def forbidden(directory):
                    self.fail("an unused MTP head was resolved")

                _, _, options, _, _ = self.execute(base, mtp_family="qwen3_5_moe", mtp_drafts=depth,
                                                   no_drafts=no_drafts, head_resolver=forbidden)
                self.assertNotIn("mtp_head", options[0])
                self.assertEqual(options[0]["mtp_drafts"], 0)

    def test_qwen_omitted_mtp_cap_keeps_baseline_load_options_and_no_optional_head(self):
        for choice in ("auto", "none"):
            with self.subTest(drafter=choice), tempfile.TemporaryDirectory() as base:

                def forbidden(directory):
                    self.fail("an unselected optional MTP head was resolved")

                _, _, options, _, _ = self.execute(base, mtp_family="qwen3_5_moe", no_drafts=False,
                                                   head_resolver=forbidden, drafter_choice=choice)
                self.assertNotIn("mtp_head", options[0])
                self.assertNotIn("mtp_drafts", options[0])

    def test_persistence_disabled_keeps_existing_no_hash_load_path(self):
        with tempfile.TemporaryDirectory() as base:
            ids, owners, _, _, output = self.execute(base, persistence=False)
            self.assertTrue(ids[0].startswith(str((Path(base) / "model").resolve())))
            self.assertNotIn("identity_diagnostics", owners)
            self.assertNotIn("snapshot content identity", output)


if __name__ == "__main__":
    unittest.main()
