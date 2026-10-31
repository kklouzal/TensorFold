import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys
import importlib.metadata
import os


from tensorfold.cuda.sm70 import attention_qualification as q


class Qualification(unittest.TestCase):
    def kwargs(self):
        return {"rows": 24, "heads": 3, "kv_heads": 1, "head_dim": 256,
                "context": 8192, "first_end": 8169,
                "identity": dict(q.REQUIRED_IDENTITY), "capability": (7, 0)}

    def test_exact_scope(self):
        for rows in (1, 3, 4, 24):
            kw = self.kwargs()
            kw.update(rows=rows, first_end=8192-rows+1)
            self.assertTrue(q.supports(**kw))

    def test_above_limit_or_out_of_range_declines(self):
        for change in ({"context": 8193}, {"first_end": 8170}, {"first_end": 64},
                       {"rows": 25}, {"rows": True}, {"rows": 0}, {"prompt": True}):
            with self.subTest(change=change):
                self.assertFalse(q.supports(**(self.kwargs() | change)))

    def test_other_tp_and_architecture_decline(self):
        for change in ({"heads": 6}, {"kv_heads": 2}, {"head_dim": 128},
                       {"capability": (8, 0)}):
            with self.subTest(change=change):
                self.assertFalse(q.supports(**(self.kwargs() | change)))

    def test_runtime_identity_is_exact(self):
        for key in q.REQUIRED_IDENTITY:
            kw = self.kwargs()
            kw["identity"].pop(key)
            with self.subTest(key=key):
                self.assertFalse(q.supports(**kw))


class RuntimeIdentity(unittest.TestCase):
    def setUp(self):
        q.runtime_identity.cache_clear()
        self.addCleanup(q.runtime_identity.cache_clear)

    def torch(self, **changes):
        identity = dict(q.REQUIRED_IDENTITY) | changes
        return SimpleNamespace(__version__=identity["torch"],
            version=SimpleNamespace(git_version=identity["torch_git"], cuda=identity["cuda"]),
            backends=SimpleNamespace(cuda=SimpleNamespace(
                preferred_blas_library=lambda: identity["blas_backend"],
                matmul=SimpleNamespace(allow_fp16_reduced_precision_reduction=identity["fp16_reduced_precision"],
                                      allow_fp16_accumulation=identity["fp16_accumulation"]))),
            _C=SimpleNamespace(_cuda_getCublasWorkspaceSize=lambda: identity["workspace_bytes"]),
            cuda=SimpleNamespace(get_device_properties=lambda _device: SimpleNamespace(
                multi_processor_count=identity["sms"])))

    def test_source_build_does_not_require_legacy_wheel_metadata_or_qualify_candidate(self):
        fake = self.torch(torch="2.14.1+tfcu129sm70", cuda="12.9",
                          torch_git="04cf2557f315a350e7eea7ab064288def65db75a")
        with patch.dict(sys.modules, {"torch": fake}), patch.dict(os.environ, {}, clear=True), \
                patch.object(importlib.metadata, "version", side_effect=AssertionError("legacy wheel queried")):
            actual = q.runtime_identity("cuda:0")
        self.assertNotIn("cublas", actual)
        expected = dict(q.REQUIRED_IDENTITY) | {"torch": fake.__version__, "torch_git": fake.version.git_version,
                                               "cuda": fake.version.cuda}
        expected.pop("cublas")
        self.assertEqual(actual, expected)
        self.assertFalse(q.supports(rows=1, heads=3, kv_heads=1, head_dim=256, context=128,
                                    first_end=128, identity=actual, capability=(7, 0)))

    def test_each_legacy_build_discriminator_precedes_package_lookup(self):
        for key, value in (("torch", "other"), ("torch_git", "a" * 40), ("cuda", "12.9")):
            with self.subTest(key=key):
                q.runtime_identity.cache_clear()
                with patch.dict(sys.modules, {"torch": self.torch(**{key: value})}), \
                        patch.object(importlib.metadata, "version", side_effect=AssertionError("wheel queried")):
                    self.assertNotIn("cublas", q.runtime_identity("cuda:0"))

    def test_source_metadata_preparation_and_attention_dispatch_stay_ineligible(self):
        fake = self.torch(torch="2.14.1+tfcu129sm70", cuda="12.9",
                          torch_git="04cf2557f315a350e7eea7ab064288def65db75a")
        fake.cuda.get_device_capability = lambda _device: (7, 0)
        owner = SimpleNamespace(cfg=SimpleNamespace(execution_dtype="float16", heads=3, kv_heads=1,
                                                    head_dim=256), device="cuda:0", meta={})
        with patch.dict(sys.modules, {"torch": fake}), patch.object(q, "active", return_value=True), \
                patch.object(importlib.metadata, "version", side_effect=AssertionError("legacy wheel queried")):
            descriptor = q.metadata_prepare(owner)
            self.assertEqual(descriptor["runtime"], q.runtime_identity("cuda:0"))
            self.assertFalse(q.allows(owner, [127], 1, 128))

    def test_exact_legacy_identity_retains_required_wheel_and_all_settings(self):
        with patch.dict(sys.modules, {"torch": self.torch()}), patch.dict(os.environ, {}, clear=True), \
                patch.object(importlib.metadata, "version", return_value=q.REQUIRED_IDENTITY["cublas"]) as package:
            self.assertEqual(q.runtime_identity("cuda:0"), q.REQUIRED_IDENTITY)
            q.runtime_identity("cuda:0")
            package.assert_called_once_with("nvidia-cublas-cu12")

    def test_missing_legacy_metadata_still_fails_instead_of_claiming_identity(self):
        with patch.dict(sys.modules, {"torch": self.torch()}), \
                patch.object(importlib.metadata, "version", side_effect=importlib.metadata.PackageNotFoundError):
            with self.assertRaises(importlib.metadata.PackageNotFoundError):
                q.runtime_identity("cuda:0")

    def test_ineligible_actual_math_and_workspace_settings_are_preserved(self):
        fake = self.torch(torch="2.14.1+tfcu129sm70", cuda="12.9", blas_backend="actual-other-backend",
                          fp16_reduced_precision=False, fp16_accumulation=True, workspace_bytes=33554432, sms=79)
        with patch.dict(sys.modules, {"torch": fake}), \
                patch.dict(os.environ, {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"}), \
                patch.object(importlib.metadata, "version", side_effect=AssertionError("legacy wheel queried")):
            actual = q.runtime_identity.__wrapped__("cuda:0")
        self.assertEqual({key: actual[key] for key in ("blas_backend", "fp16_reduced_precision", "fp16_accumulation",
                                                       "workspace_config", "workspace_bytes", "sms")},
                         {"blas_backend": "actual-other-backend", "fp16_reduced_precision": False,
                          "fp16_accumulation": True, "workspace_config": ":4096:8",
                          "workspace_bytes": 33554432, "sms": 79})


class OwnerPolicy(unittest.TestCase):
    def owner(self):
        return SimpleNamespace(cfg=SimpleNamespace(execution_dtype="float16", heads=3,
            kv_heads=1, head_dim=256), device="cuda:0", meta={})

    def test_provisional_position_controls_lookup_without_cuda_readback(self):
        w=self.owner()
        fake=SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda device:(7,0)))
        with patch.object(q,"active",return_value=True), patch.object(q,"runtime_identity",return_value=q.REQUIRED_IDENTITY), patch.dict(sys.modules,{"torch":fake}):
            self.assertTrue(q.allows(w,[128],1,256))
            self.assertFalse(q.allows(w,[20],1,256))
            self.assertFalse(q.allows(w,[8192],1,16384))
            self.assertFalse(q.allows(w,[128,20],4,256))
            w.cfg.heads=6
            self.assertFalse(q.allows(w,[128],1,256))

    def test_metadata_is_recorded_and_immutable(self):
        w=self.owner()
        with patch.object(q,"runtime_identity",return_value=q.REQUIRED_IDENTITY):
            descriptor=q.metadata_prepare(w)
            self.assertEqual(q.metadata_prepare(w),descriptor)
            w.meta["attention_qualification"]={}
            with self.assertRaisesRegex(ValueError,"identity changed"):
                q.metadata_prepare(w)

    def test_reference_provider_does_not_query_runtime(self):
        w=self.owner()
        with patch.object(q,"active",return_value=False), patch.object(q,"runtime_identity",side_effect=AssertionError("runtime queried")):
            self.assertTrue(q.allows(w,[0],100,16384))


if __name__ == "__main__":
    unittest.main()
