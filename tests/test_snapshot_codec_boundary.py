"""Runtime representation/lazy-read contract under opaque no-SDK metadata."""

from pathlib import Path
from types import SimpleNamespace
import unittest

from tensorfold.engine.snapshot_codec import Codec, DTYPES
from tensorfold.engine.snapshot_payload import SnapshotPayload


class Native:
    def __init__(self, value, dtype):
        self.dtype, self.shape, self.words = dtype, value.shape, value.words


class Host:
    def __init__(self, dtype, shape=(1, 3), words=b"exact fixture words"):
        self.dtype, self.shape, self.words = dtype, shape, words


class CodecControls(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.runtime = SimpleNamespace(array=Native, **{name: name for name, _, _ in DTYPES})
        self.numpy = SimpleNamespace(
            dtype=lambda name: name, ndarray=Host, array=lambda value: Host(value.dtype, value.shape, value.words)
        )
        self.codec = Codec(self.runtime, self.numpy)

    def test_supported_native_writer_codes_and_host_conversion_have_exact_dtype_geometry(self):
        for name, code, _ in DTYPES:
            host = Host(name)
            native = Native(host, name)
            self.assertEqual(self.codec.describe(native), {"dtype": code, "shape": [1, 3]})
            if name != "bfloat16":
                converted = self.codec.native_array(host)
                self.assertEqual(converted.dtype, name)
                self.assertEqual(converted.words, host.words)
                restored = self.codec.host_array(native)
                self.assertEqual(restored.dtype, host.dtype)
                self.assertEqual(restored.words, host.words)
        self.assertIsNone(self.codec.describe(None))
        self.assertIsNone(self.codec.kind(object()))

    def test_foreign_dtype_and_wrong_conversion_refuse_instead_of_lower_precision(self):
        with self.assertRaises(ValueError):
            self.codec.describe(Host("float64"))  # unsupported by the native writer
        value = Host("int64")
        class WrongNative(Native):
            def __init__(self, value, dtype):
                super().__init__(value, "float32")

        self.runtime.array = WrongNative
        with self.assertRaises(ValueError):
            self.codec.native_array(value)
        self.numpy.array = lambda v: Host("float32", v.shape, v.words)
        with self.assertRaises(ValueError):
            self.codec.host_array(Native(value, "int64"))

    def test_complete_lazy_evaluation_precedes_return_and_failure_propagates(self):
        arrays = {"0.keys": Native(Host("float32"), "float32")}
        metadata = {"model": "fixture"}
        self.runtime.load = lambda path, return_metadata: (self.events.append("load") or arrays, metadata)
        self.runtime.eval = lambda values: self.events.append(("eval", values))
        self.assertEqual(self.codec.load(Path("private.safetensors")), (arrays, metadata))
        self.assertEqual(self.events, ["load", ("eval", list(arrays.values()))])
        primary = KeyboardInterrupt("read completion interrupted")

        def fail(values):
            raise primary

        self.runtime.eval = fail
        with self.assertRaises(KeyboardInterrupt) as caught:
            self.codec.load(Path("private.safetensors"))
        self.assertIs(caught.exception, primary)

    def test_native_writer_borrows_array_and_converts_only_explicit_numpy_keys(self):
        host, native = Host("int64"), Native(Host("float32"), "float32")
        payload = SnapshotPayload({"0.history": host, "0.keys": native}, {"format": "2"}, frozenset({"0.history"}))
        calls = []
        self.runtime.save_safetensors = lambda path, arrays, metadata: calls.append((path, arrays, metadata))
        self.codec.write(Path("private.safetensors"), payload)
        self.assertIs(calls[0][1]["0.keys"], native)
        self.assertIs(type(calls[0][1]["0.history"]), Native)
        self.assertEqual(calls[0][1]["0.history"].dtype, "int64")
        self.assertEqual(calls[0][1]["0.history"].words, host.words)
        self.assertIs(payload.arrays["0.history"], host)


if __name__ == "__main__":
    unittest.main()
