import unittest
from types import SimpleNamespace
from unittest.mock import patch
import sys


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
