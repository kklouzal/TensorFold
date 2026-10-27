"""Actual current-SDK SM70 target kernels over all eight logical shard offsets.

This one-device component test proves no TP transport/full-model/throughput
qualification; Root's separate eight-rank model run remains required.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_sm70_target_integer_max_and_acceptance_match_original_full_vocabulary():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires actual SM70")
    if torch.version.git_version != "04cf2557f315a350e7eea7ab064288def65db75a" or torch.version.cuda != "12.9":
        pytest.skip("requires the current owned source SDK; legacy cu126 evidence is separate")
    from tensorfold.cuda.sm70.greedy_native import extension
    from tensorfold.cuda.streams import accept

    native = extension()
    rng = np.random.default_rng(713)
    world = 8
    for rows, width in ((1, 1), (2, 17), (3, 33), (32, 71), (3, 31040)):
        vocabulary = world * width
        for pattern in ("random", "ties", "signed-zero", "subnormal"):
            full = rng.normal(size=(rows, vocabulary)).astype(np.float32)
            if pattern == "ties":
                full.fill(0)
            elif pattern == "signed-zero":
                full.view(np.uint32).reshape(-1)[:] = np.resize(np.asarray([0, 0x80000000], dtype=np.uint32), full.size)
            elif pattern == "subnormal":
                full.view(np.uint32).reshape(-1)[:] = np.resize(np.asarray([0, 1, 0x80000001, 0x007fffff], dtype=np.uint32), full.size)
            ids = np.arange(vocabulary, dtype=np.int64)
            sampled = [int(ids[np.lexsort((ids, -row))[0]]) for row in full]
            frame = torch.empty((world, rows, 2), dtype=torch.int64, device="cuda")
            for rank in range(world):
                shard = torch.from_numpy(full[:, rank * width:(rank + 1) * width].copy()).to("cuda")
                native.target_local(shard, frame[rank], rank * width, vocabulary)
            for partial, room, ended in ((False, rows, False), (True, rows, False),
                                          (False, max(1, rows // 2), False), (False, rows, True)):
                window = [0] + sampled[:-1]
                if partial and rows > 1:
                    window[1] = (window[1] + 1) % vocabulary
                eos = [sampled[max(0, rows // 2 - 1)]] if ended else []
                expected = accept(window, list(range(-1, rows - 1)), sampled, room, eos)
                committed = [window[row] for row in expected[0][1:]] + [expected[1]]
                feedback = torch.empty(rows + 2, dtype=torch.int32, device="cuda")
                native.target_accept(frame, torch.tensor(window, dtype=torch.int32, device="cuda"),
                    torch.tensor(eos, dtype=torch.int32, device="cuda"), feedback, room, vocabulary, True)
                result = feedback.cpu().tolist()
                assert result[:2] == [len(expected[0]), 0]
                assert result[2:2 + len(committed)] == committed and result[2 + len(committed):] == [0] * (rows - len(committed))
    # An invalid rejected row must not disappear behind a mismatch at row1.
    for value, bit in ((float("nan"), 1), (float("inf"), 2), (-float("inf"), 4)):
        frame = torch.empty((world, 3, 2), dtype=torch.int64, device="cuda")
        for rank in range(world):
            shard = torch.zeros((3, 33), dtype=torch.float32, device="cuda")
            if bit == 4:
                shard[2].fill_(value)
            elif rank == world - 1:
                shard[2, 0] = value
            native.target_local(shard, frame[rank], rank * 33, world * 33)
        feedback = torch.empty(5, dtype=torch.int32, device="cuda")
        native.target_accept(frame, torch.tensor([11, 1, 2], dtype=torch.int32, device="cuda"),
            torch.empty(0, dtype=torch.int32, device="cuda"), feedback, 3, world * 33, False)
        assert int(feedback.cpu()[1]) & bit
