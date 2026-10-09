"""CUDA request replay containment over actual native-free Stream state."""
import unittest

from tensorfold.cuda.streams import Stream


class Replay(unittest.TestCase):
    def stream(self, owed, *, count=8):
        emitted = []
        stream = Stream([10], count, emit=lambda ids: emitted.extend(ids), owed=list(owed))
        return stream, emitted

    def test_valid_chunked_replay_and_new_tokens_preserve_order_and_counts(self):
        stream, emitted = self.stream([97, 98])
        stream.take([97])
        self.assertFalse(emitted)
        self.assertEqual(stream.owed, [98])
        stream.take([98, 99])
        self.assertEqual(emitted, [99])
        self.assertEqual(stream.out, [97, 98, 99])
        self.assertEqual(stream.context, [97, 98, 99])
        self.assertEqual(stream.accepted, 1)
        self.assertIsNone(stream.error)

    def test_divergent_prefix_never_emits_tail_or_commits_bad_tokens(self):
        stream, emitted = self.stream([97, 98])
        stream.take([120, 98, 99])
        self.assertFalse(emitted)
        self.assertFalse(stream.out)
        self.assertFalse(stream.context)
        self.assertEqual(stream.owed, [97, 98])
        self.assertTrue(stream.done)
        self.assertIsInstance(stream.error, RuntimeError)
        self.assertEqual(stream.accepted, 0)

    def test_early_eos_or_count_with_owed_tokens_is_failure(self):
        for count, eos in ((8, [97]), (1, [])):
            stream, emitted = self.stream([97, 98], count=count)
            stream.take([97], eos=eos)
            self.assertFalse(emitted)
            self.assertTrue(stream.done)
            self.assertIsInstance(stream.error, RuntimeError)
            self.assertEqual(stream.owed, [98])

    def test_no_owed_normal_eos_and_client_stop_keep_original_emit_api(self):
        stream, emitted = self.stream([])
        stream.take([97], eos=[97])
        self.assertEqual(emitted, [97])
        self.assertTrue(stream.done)
        self.assertIsNone(stream.error)
        stream, emitted = self.stream([])
        stream.emit = lambda ids: (emitted.extend(ids), True)[1]
        stream.take([97, 98])
        self.assertEqual(emitted, [97, 98])
        self.assertTrue(stream.done)
        self.assertIsNone(stream.error)

    def test_callback_failure_preserves_original_exception_identity(self):
        primary = RuntimeError("client failed")
        stream, _ = self.stream([])

        def failure(ids):
            raise primary

        stream.emit = failure
        with self.assertRaises(RuntimeError) as caught:
            stream.take([97])
        self.assertIs(caught.exception, primary)
        self.assertEqual(stream.out, [97])

    def test_normal_terminal_check_keeps_its_original_order_after_callback(self):
        stream, emitted = self.stream([])

        def update(ids):
            emitted.extend(ids)
            stream.count = 1

        stream.emit = update
        stream.take([97])
        self.assertTrue(stream.done)
        self.assertEqual(emitted, [97])

    def test_repeated_preemption_preserves_keyed_sampling_and_full_owed_prefix(self):
        stream, _ = self.stream([97, 98, 99])
        sampling = object()
        stream.sampling = sampling
        stream.take([97, 98])
        continued = stream.continued()
        self.assertIs(continued.sampling, sampling)
        self.assertEqual(continued.prompt, [10])
        self.assertEqual(continued.count, stream.count)
        self.assertEqual(continued.owed, [97, 98, 99])


if __name__ == "__main__":
    unittest.main()
