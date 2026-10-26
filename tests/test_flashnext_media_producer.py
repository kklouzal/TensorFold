"""CPU protocol/ownership oracles; these do not qualify NCCL or the vision model."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier, Condition
from types import SimpleNamespace as NS

import pytest
import torch

from test_input_preparation import Store
from test_qwen4_exp_media_plan import NAMES, planner, probe
from test_qwen4_exp_position_coordinates import policy
from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers

from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation
from tensorfold.families.qwen4_exp.cuda.media_producer import MediaInputProducer, MediaInputSlice, producer_resources


class Encoder:
    def __init__(self, hidden):
        self.hidden, self.calls = hidden, 0
    def _retire_inputs(self):
        pass
    def encode_device_unit(self, pixels, grid):
        self.calls += 1
        assert pixels.dtype == torch.float16 and torch.all(pixels == .125)
        rows = int(grid.prod()) // 4
        return NS(features=torch.arange(rows * self.hidden).reshape(rows, self.hidden).half() + 37)


class Embedding:
    def __init__(self):
        self.table = torch.arange(64 * 4).reshape(64, 4).half()
        self.reads = []
    def lookup(self, ids, *, copies, out):
        assert copies == 1
        self.reads.extend(ids.tolist())
        out.copy_(self.table.index_select(0, ids))
        return out


def planned():
    tokens = Tokenizer(models.WordLevel({"[UNK]": 0, "hello": 1, **{v: i + 2 for i, v in enumerate(NAMES)}},
                                        unk_token="[UNK]"))
    tokens.pre_tokenizer = pre_tokenizers.Whitespace()
    tokens.add_special_tokens([AddedToken(v, normalized=False, special=True) for v in NAMES])
    image, _, start, end = NAMES
    return planner(tokens, profile_identity="a" * 64)(f"hello {start}{image}{end} hello", (probe("image", 0),),
                                                      max_prompt_tokens=512, max_visual_tokens=None)


def weights(comm=None, *, mtp=False):
    face = Embedding()
    value = NS(device=torch.device("cpu"), cfg=NS(hidden=4, vocab=64), embed=(face,),
               positions=NS(target=policy(), mtp=policy("mtp") if mtp else None), mtp=object() if mtp else None,
               comm=comm, meta={"coordinate_profile_identity": "a" * 64, "media_inputs_enabled": True,
                                "rank": 0 if comm is None else comm.rank, "world": 1 if comm is None else 8})
    install_input_preparation(value)
    return value


def consume(span):
    result = span.values.clone()
    for role in span._leases:
        lease = span.borrow(role, profile_identity=span.profile_identity, generation=span.generation,
                            media_identity=span.media_identity)
        assert lease.rows(span.start, span.token_ids).data_ptr() == span.values.data_ptr()
        lease.complete()
    return result


@pytest.mark.parametrize("mtp", [False, True])
def test_actual_rows_original_ids_coordinates_and_both_consumers_share_one_span(mtp):
    w, plan = weights(mtp=mtp), planned()
    encoder, registered, loaded = Encoder(4), [], []
    producer = MediaInputProducer.create(w, encoder=encoder, encoder_rank=0, rows=64, patches=64,
                                         cache_rows=32, register=registered.append)
    request = MediaInputSlice(plan, 4, 0, len(plan.token_ids))
    def pixels(index, unit):
        loaded.append((index, unit.probe_index))
        return torch.full((24, 1536), .125)
    span, = producer.publish((request,), pixel_loader=pixels)
    expected = w.embed[0].table[torch.tensor(plan.token_ids)].clone()
    unit, = plan.units
    expected[unit.run.start:unit.run.end] = torch.arange(24).reshape(6, 4).half() + 37
    assert registered == [producer] and loaded == [(0, 0)]
    assert span.token_ids == plan.token_ids and encoder.calls == 1
    torch.testing.assert_close(consume(span), expected, rtol=0, atol=0)
    assert 2 not in w.embed[0].reads  # No placeholder-token embedding lookup for visual rows.
    torch.testing.assert_close(span.positions.long(), plan.positions.coordinates(0, len(plan.token_ids), max_rows=64))
    next_span, = producer.publish((replace(request, generation=5),), pixel_loader=pixels)
    assert loaded == [(0, 0)] and encoder.calls == 1  # Bounded whole-unit cache, independent of request generation.
    assert span.closed and next_span.generation == 5
    consume(next_span)
    producer.close()
    assert producer.closed and producer.values is None


def test_partial_spans_retire_before_arena_reuse_and_cache_miss_has_same_values():
    w, plan = weights(), planned()
    producer = MediaInputProducer.create(w, encoder=Encoder(4), encoder_rank=0, rows=64, patches=64,
                                         cache_rows=0, register=lambda owner: None)
    first, = producer.publish((MediaInputSlice(plan, 1, 0, 4),),
                             pixel_loader=lambda *_: torch.full((24, 1536), .125))
    before = first.values.clone()
    with pytest.raises(ValueError, match="consumers"):
        producer.retire()
    assert not first.closed and torch.equal(first.values, before)
    consume(first)
    second, = producer.publish((MediaInputSlice(plan, 1, 4, len(plan.token_ids) - 4),),
                              pixel_loader=lambda *_: torch.full((24, 1536), .125))
    assert first.closed and second.start == 4
    consume(second)
    producer.close()


class Cohort:
    def __init__(self):
        self.store, self.barrier, self.condition = Store(), Barrier(8, timeout=5), Condition()
        self.broadcasts, self.messages, self.calls = {}, {}, []
    def comm(self, rank):
        group = self
        class Comm:
            world = 8
            def __init__(self):
                self.rank, self.store, self.at = rank, group.store, 0
            def broadcast(self, value, root):
                at = self.at
                self.at += 1
                with group.condition:
                    group.broadcasts.setdefault(at, {})[rank] = value
                group.barrier.wait()
                if rank == root:
                    for destination in group.broadcasts[at].values():
                        destination.copy_(value)
                    group.calls.append(("broadcast", root))
                group.barrier.wait()
            def send(self, value, peer):
                with group.condition:
                    group.messages[(rank, peer)] = value.clone()
                    group.calls.append(("send", rank, peer))
                    group.condition.notify_all()
            def recv(self, value, peer):
                with group.condition:
                    if not group.condition.wait_for(lambda: (peer, rank) in group.messages, timeout=5):
                        raise TimeoutError("test media transfer was stranded")
                    value.copy_(group.messages.pop((peer, rank)))
        return Comm()


@pytest.mark.parametrize("encoder_rank", [0, 3])
def test_eight_rank_actual_preparation_contract_and_fixed_encoder_transport(monkeypatch, encoder_rank):
    from tensorfold.cuda import cohort
    group, plan = Cohort(), planned()
    monkeypatch.setattr(cohort, "notify_failure", group.store.abort)
    all_weights = [weights(group.comm(rank)) for rank in range(8)]
    encoder = Encoder(4)
    producers = [MediaInputProducer.create(w, encoder=encoder if rank == encoder_rank else None,
                   encoder_rank=encoder_rank, rows=64, patches=64, cache_rows=16, register=lambda owner: None)
                 for rank, w in enumerate(all_weights)]
    def worker(rank):
        loader = (lambda *_: torch.full((24, 1536), .125)) if rank == 0 else None
        result, = producers[rank].publish((MediaInputSlice(plan, 7, 0, len(plan.token_ids)),), pixel_loader=loader)
        return consume(result)
    with ThreadPoolExecutor(max_workers=8) as pool:
        actual = list(pool.map(worker, range(8)))
    for value in actual[1:]:
        torch.testing.assert_close(value, actual[0], rtol=0, atol=0)
    assert encoder.calls == 1 and ("broadcast", encoder_rank) in group.calls
    assert (("send", 0, encoder_rank) in group.calls) == (encoder_rank != 0)
    assert not group.store.values and not group.messages
    assert all(not w.embed[0].reads for rank, w in enumerate(all_weights) if rank != encoder_rank)
    for producer in producers:
        producer.close()


def test_resource_products_and_capability_reject_before_allocating(monkeypatch):
    for changes in ({"rows": True}, {"patches": 65}, {"cache_rows": -1}, {"encoder_rank": 8}):
        with pytest.raises(ValueError):
            producer_resources(**{**dict(rows=64, patches=64, cache_rows=32, hidden=4, world=8, encoder_rank=3), **changes})
    w = weights()
    w.meta["media_inputs_enabled"] = False
    monkeypatch.setattr(torch, "empty", lambda *_a, **_k: pytest.fail("allocation before media capability"))
    with pytest.raises(ValueError, match="admitted profile"):
        MediaInputProducer.create(w, encoder=Encoder(4), encoder_rank=0, rows=64, patches=64,
                                  cache_rows=32, register=lambda owner: None)


def test_different_rank_slices_abort_before_text_media_or_transport(monkeypatch):
    from tensorfold.cuda import cohort
    group, plan = Cohort(), planned()
    monkeypatch.setattr(cohort, "notify_failure", group.store.abort)
    all_weights = [weights(group.comm(rank)) for rank in range(8)]
    encoder = Encoder(4)
    producers = [MediaInputProducer.create(w, encoder=encoder if rank == 0 else None, encoder_rank=0,
                   rows=64, patches=64, cache_rows=16, register=lambda owner: None) for rank, w in enumerate(all_weights)]
    def worker(rank):
        try:
            producers[rank].publish((MediaInputSlice(plan, 7, 0, 1 if rank == 3 else len(plan.token_ids)),),
                pixel_loader=(lambda *_: pytest.fail("pixels before metadata agreement")) if rank == 0 else None)
        except BaseException as error:
            return error
        pytest.fail("different valid rank slices entered publication")
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(worker, range(8)))
    assert all(isinstance(value, BaseException) for value in outcomes)
    assert group.store.aborted and not group.calls and encoder.calls == 0
    assert all(not w.embed[0].reads for w in all_weights)
    for producer in producers:
        producer.close()


def test_foreign_process_refusal_never_uses_inherited_cohort(monkeypatch):
    import tensorfold.families.qwen4_exp.cuda.media_producer as implementation
    w = weights()
    producer = MediaInputProducer.create(w, encoder=Encoder(4), encoder_rank=0, rows=64,
                                         patches=64, cache_rows=0, register=lambda owner: None)
    producer.pid -= 1
    monkeypatch.setattr(producer.context, "fail", lambda *_: pytest.fail("inherited communicator access"))
    with pytest.raises(ValueError, match="another process"):
        producer.publish((), pixel_loader=None)
    producer.pid = implementation.os.getpid()
    producer.close()
