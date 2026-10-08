"""Exercise the actual CPU queue/take/close path before CUDA fixture use."""
import pytest

torch = pytest.importorskip("torch")
from tensorfold.cuda.direct_read import ReadAhead  # noqa: E402
from exl3_read_recorder import read_ahead_recorder  # noqa: E402


@pytest.mark.parametrize("failed",[False,True])
def test_recorder_preserves_worker_count_tracks_real_futures_and_joins_threads(failed):
    error = OSError("owned range failed")
    class Bytes:
        def read(self,path,offset,count):
            assert path == "fixture" and count == 4
            if failed and offset == 8:
                raise error
            return torch.arange(offset,offset+count,dtype=torch.uint8)
    records = []
    Recorder = read_ahead_recorder(ReadAhead,records)
    owner = Recorder(reader=Bytes(),threads=1,run=32,gap=0)
    assert owner.threads == 1 and type(owner.threads) is int
    owner.queue([("first","fixture",0,4,None),("second","fixture",8,12,None)])
    assert owner.take("first").tolist() == [0,1,2,3]
    try:
        if failed:
            with pytest.raises(OSError) as caught:
                owner.take("second")
            assert caught.value is error
        else:
            assert owner.take("second").tolist() == [8,9,10,11]
    finally:
        owner.close()
    assert records == [owner] and owner.observed_closed and owner.pool is None
    assert len(owner.observed_futures) == 2 and all(future.done() for future in owner.observed_futures)
    assert owner.observed_threads and all(not worker.is_alive() for worker in owner.observed_threads)
    assert not torch.cuda.is_initialized()
