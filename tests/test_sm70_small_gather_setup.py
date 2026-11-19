"""CPU checks of the opt-in small all-gather's configuration and Store rendezvous (no CUDA is used)."""

import pytest

from tensorfold.cuda import small_gather


@pytest.mark.parametrize("raw,expected", [("", 0), ("  ", 0), ("0", 0), ("8", 8), ("131072", 131072),
                                          (str(small_gather.MAX_CAPACITY), small_gather.MAX_CAPACITY)])
def test_capacity_accepts_zero_and_multiples_of_eight_up_to_the_ceiling(monkeypatch, raw, expected):
    monkeypatch.setenv(small_gather.ENV, raw)
    assert small_gather.configured_capacity() == expected


@pytest.mark.parametrize("raw", [str(small_gather.MAX_CAPACITY + 8), str(1 << 30), "12", "-8", "abc", "8.0"])
def test_capacity_rejects_values_outside_the_contract(monkeypatch, raw):
    monkeypatch.setenv(small_gather.ENV, raw)
    with pytest.raises(ValueError, match=small_gather.ENV):
        small_gather.configured_capacity()


def test_mailbox_bytes_are_two_parities_of_sixteen_byte_lines():
    assert small_gather.mailbox_bytes(8, 131072) == 2 * 8 * (131072 // 8) * 16 == 4 * 1024 * 1024


@pytest.mark.parametrize("world,raw,expected", [(8, "", 1638400), (6, "", 1638336), (7, "", 1638336), (8, "0", 0),
                                                (8, "128", 128), (8, str(small_gather.MAX_BULK), small_gather.MAX_BULK)])
def test_bulk_capacity_defaults_to_160_rows_of_2560_and_accepts_zero(monkeypatch, world, raw, expected):
    monkeypatch.setenv(small_gather.BULK_ENV, raw)
    assert small_gather.configured_bulk(world) == expected


@pytest.mark.parametrize("raw", ["64", "-128", "abc", str(small_gather.MAX_BULK + 128)])
def test_bulk_capacity_rejects_values_outside_the_contract(monkeypatch, raw):
    monkeypatch.setenv(small_gather.BULK_ENV, raw)
    with pytest.raises(ValueError, match=small_gather.BULK_ENV):
        small_gather.configured_bulk(8)


def test_bulk_mailbox_bytes_at_the_default_on_the_v100_cube_mesh():
    # 3 transit regions (non-neighbours per rank): 2 x (8 x 4 x 204800 + 8 x 102400) + flags 2 x 8 x 5 x 4 x 4.
    assert small_gather.bulk_mailbox_bytes(8, 1638400, 3) == 14745600 + 1280
    # wide (FP32 sums): phase-2 slots of 204800 bytes.
    assert small_gather.bulk_mailbox_bytes(8, 1638400, 3, wide=True) == 16384000 + 1280


def test_each_open_has_its_own_store_namespace():
    first = small_gather.store_prefix("tf_nccl_uid", "a" * 32)
    second = small_gather.store_prefix("tf_nccl_uid", "b" * 32)
    assert first.startswith("tf_nccl_uid/small_gather/") and first != second
    for nonce in ("", "a/b", None):
        with pytest.raises(ValueError):
            small_gather.store_prefix("tf_nccl_uid", nonce)


class _Store:
    """Non-blocking stand-in: a missing key fails the test instead of waiting for a timeout."""

    def __init__(self):
        self.values = {}

    def set(self, key, value):
        self.values[key] = value.encode() if isinstance(value, str) else bytes(value)

    def get(self, key):
        return self.values[key]


class _Native:
    released, split = 0, 4

    def __init__(self, rank, world, capacity):
        self.rank = rank

    def pci_bus_id(self):
        return f"0000:{self.rank + 1:02X}:00.0"

    def handle(self):
        return bytes(64)

    def connect(self, *args):
        raise AssertionError("no plan exists when a peer failed")

    def release(self):
        type(self).released += 1


class _Extension:
    SmallGather = _Native


def test_local_setup_failure_publishes_markers_so_peers_return_none_without_waiting(monkeypatch):
    store, nonce, world, capacity = _Store(), "c" * 32, 2, 4096
    prefix = small_gather.store_prefix("key", nonce)
    store.set(prefix + "capacity/1", f"{capacity}/0")  # The peer reached the rendezvous first (no bulk mailbox).

    def broken():
        raise RuntimeError("extension build failed")

    monkeypatch.setattr(small_gather, "extension", broken)
    with pytest.raises(RuntimeError, match="extension build failed"):
        small_gather.open_small_gather(store, "key", 0, world, capacity, nonce)
    assert store.get(prefix + "topology/0") == b"failed"
    assert store.get(prefix + "handle/0") == b""
    assert store.get(prefix + "ok/0") == b"0"

    monkeypatch.setattr(small_gather, "extension", lambda: _Extension)
    monkeypatch.setattr(small_gather, "nvlink_neighbours", lambda bus: ["0000:01:00.0"])
    before = _Native.released
    assert small_gather.open_small_gather(store, "key", 1, world, capacity, nonce) is None
    assert store.get(prefix + "ok/1") == b"0" and _Native.released == before + 1
    # A later open (new nonce) never reads these keys.
    assert not any(key.startswith(small_gather.store_prefix("key", "d" * 32)) for key in store.values)


def test_mismatched_capacity_fails_every_rank_before_native_setup(monkeypatch):
    store, nonce = _Store(), "e" * 32
    prefix = small_gather.store_prefix("key", nonce)
    store.set(prefix + "capacity/1", "8")
    monkeypatch.setattr(small_gather, "extension", lambda: pytest.fail("native setup must not start"))
    with pytest.raises(RuntimeError, match="different"):
        small_gather.open_small_gather(store, "key", 0, 2, 16, nonce)
