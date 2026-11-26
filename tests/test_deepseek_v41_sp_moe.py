"""Regression for global routing across sequence shards and local expert banks."""

from types import SimpleNamespace

import pytest


torch = pytest.importorskip("torch")


def _route(logits, bias, top_k, scale):
    # CPU stand-in for the Triton router. Only the ordering and transport after
    # the router are under test. The native driver leaves the real router in place.
    score = torch.nn.functional.softplus(logits).sqrt()
    ids = (score + bias).topk(top_k, dim=-1).indices
    weights = score.gather(-1, ids)
    return weights / weights.sum(-1, keepdim=True) * scale, ids


def _local_bank(ids, weights, first, count):
    owns = (first <= ids) & (ids < first + count)
    return torch.where(owns, ids - first, 0).int(), torch.where(owns, weights, 0)


def _exercise_sp_routes(patch, rank, device, comm=None, *, fp32_sp=False, moved=False):
    from tensorfold.cuda.sm70 import ds41_lattice_moe, onecat_ds41
    from tensorfold.families.deepseek_v4.cuda import target, vision
    from tensorfold.families.deepseek_v4.cuda.kernels import band, mhc, router

    tokens, experts, width = 1025, 384, 4
    geometry = SimpleNamespace(top_k=6, routed_scale=1.5)
    x = torch.zeros(tokens, width, dtype=torch.float16, device=device)
    logits = torch.full((tokens, experts), -8.0, dtype=torch.float32, device=device)
    rows = torch.arange(tokens, device=device)
    chosen = rows[:, None] % 170 + torch.tensor([0, 192, 1, 193, 2, 194], device=device)
    logits[rows[:, None], chosen] = torch.arange(7, 1, -1, dtype=torch.float32, device=device)
    logits[:, [191, 383]] = torch.tensor([-1.0, -2.0], device=device)
    text_bias = torch.zeros(experts, dtype=torch.float32, device=device)
    image_bias = text_bias.clone()
    image_bias[[191, 383]] = torch.tensor([6.0, 5.0], device=device)
    shard = target.Shard.of(tokens, rank, 8)
    first, count = rank // 4 * (experts // 2), experts // 2
    layer = SimpleNamespace(index=0, gate=None, gate_bias=text_bias, exp_gate=None, exp_up=None,
                            exp_down=None, gate_up_type=16, down_type=16, down_left=0, sh_down=None,
                            sh_gu=lambda y: (y, y), expert_groups_changed=moved)

    def local_router(logits, bias, top_k, scale, start, n):
        weights, ids = _route(logits, bias, top_k, scale)
        return _local_bank(ids, weights, start, n)

    if comm is None:
        patch.setattr(router, "route_topk", _route)
        patch.setattr(band, "route_local", local_router)
    patch.setattr(mhc, "fp32_projection", lambda rows, gate: shard.take(logits))
    patch.setattr(onecat_ds41, "shared_swiglu", lambda gate, up: gate)

    def invoke(image):
        retained = fp32_sp and moved and image is None
        # Reference routes all rows, with no sequence sharding or EP filtering.
        # Reuse the shipping router to isolate the subsequent transport and mapping.
        p, ids = vision.route_with_vision(logits, text_bias, image_bias, image, 6, 1.5)
        expected_ids, expected_p = _local_bank(ids, p, first, count)
        _, lo = _local_bank(ids, p, 0, count)
        _, hi = _local_bank(ids, p, count, count)
        assert bool((lo.sum(-1) > 0).all() and (hi.sum(-1) > 0).all())
        # Every destination applies the same routing mask to four ETP partials.
        # Synthetic experts emit their route sum in FP16 to check redistribution.
        parts = [weight.sum(-1) for weight in (lo, hi)]
        if not retained:
            parts = [part.half().float() for part in parts]
        result = (4 * (parts[0] + parts[1])).half()
        calls = []
        sends = []

        def experts(values, local, weights, *_, fragment_down=False, fp32_sp_rows=None):
            assert fragment_down is fp32_sp
            assert fp32_sp_rows == (shard.size * 8 if retained else None)
            calls.append((local, weights))
            routed = weights.sum(-1, keepdim=True).expand_as(values)
            if not retained:
                return routed.half().contiguous()
            send = torch.zeros(fp32_sp_rows, width, dtype=torch.float32, device=device)
            send[:tokens].copy_(routed)
            sends.append(send)
            return send

        patch.setattr(ds41_lattice_moe, "experts", experts)
        if comm is None:
            def gather_rows(part, sh):
                # Other senders deliver their global routes. Preserve the actual
                # sender under test so a premature EP filter changes its shard.
                received = (p if part.dtype.is_floating_point else ids).to(part.dtype).clone()
                sh.take(received).copy_(part)
                return received

            def scatter_sum(part, sh):
                if retained:
                    (send,) = sends
                    assert part is send
                    assert not torch.count_nonzero(part[tokens:])
                    wanted = expected_p.sum(-1, keepdim=True).expand(tokens, width)
                    assert torch.equal(part[:tokens], wanted)
                else:
                    assert not sends
                return sh.take(result[:, None]).expand(sh.rows, width)

            collectives = SimpleNamespace(gather_rows=gather_rows, scatter_sum=scatter_sum)
        else:
            collectives = comm
        owner = SimpleNamespace(c=collectives, g=geometry, local=range(first, first + count),
                                w=SimpleNamespace(compact_expert_down=fp32_sp), lattice=None, fp32_sp=fp32_sp,
                                max_tokens=8192,
                                bias_vl=image_bias[None], _linear=lambda y, w: torch.zeros_like(y))
        actual = target.Target.moe(owner, layer, x, image=image, shard=shard, x_rows=shard.take(x))
        (local, weights), = calls
        assert torch.equal(local, expected_ids), f"incorrect local expert IDs on rank {rank}"
        assert torch.equal(weights, expected_p), f"incorrect local route weights on rank {rank}"
        assert torch.equal(actual, shard.take(result[:, None]).expand(shard.rows, width))

    invoke(None)
    invoke(rows % 3 == 0)  # Exercise both bias vectors on every shard, including the ragged tail.


@pytest.mark.parametrize("rank", [0, 7])
def test_sequence_shard_gathers_before_expert_localization(monkeypatch, rank):
    _exercise_sp_routes(monkeypatch, rank, "cpu")


@pytest.mark.parametrize("moved", (False, True))
def test_retained_sp_reuses_padded_fp32_send_only_for_moved_text_layers(monkeypatch, moved):
    _exercise_sp_routes(monkeypatch, 7, "cpu", fp32_sp=True, moved=moved)


@pytest.mark.parametrize(("moved", "round_tag", "fused"), [
    (False, 0, False), (False, 1, False), (True, None, False), (True, 0, True), (True, 1, True),
])
def test_graph_moe_retains_original_round_flag_only_for_moved_layers(monkeypatch, moved, round_tag, fused):
    from tensorfold.cuda.sm70 import onecat_ds41
    from tensorfold.families.deepseek_v4.cuda import target
    from tensorfold.families.deepseek_v4.cuda.kernels import band, mhc

    row = torch.ones(1, 4, dtype=torch.float16)
    tag = None if round_tag is None else torch.tensor([round_tag])
    step = SimpleNamespace(round_w576_routed=tag, expert_sample=None, single_token_slots=True,
                           pack_window_ring=True)
    layer = SimpleNamespace(gate=None, gate_bias=None, exp_gate=None, exp_up=None, exp_down=None, gate_up_type=16,
                            down_type=16, down_left=0, sh_down=None, sh_gu=lambda x: (x, x),
                            expert_groups_changed=moved)
    monkeypatch.setattr(mhc, "fp32_projection", lambda *_: None)
    monkeypatch.setattr(band, "route_local", lambda *_: (None, None))
    monkeypatch.setattr(band, "add_halves", lambda a, b: a.float() + b.float())
    monkeypatch.setattr(onecat_ds41, "shared_swiglu", lambda _, up: up)
    sends = []

    def experts(x, *args, fragment_down, shared=None, round_routed=None):
        assert fragment_down
        if fused:
            assert round_routed is tag and shared is row
            return x.float() + shared.float()
        assert round_routed is None and shared is None
        return x

    def collective(x, dtype):
        assert dtype == torch.float16
        sends.append(x)
        return x.to(dtype)

    owner = SimpleNamespace(c=SimpleNamespace(sum=collective), g=SimpleNamespace(top_k=6, routed_scale=1.5),
                            local=range(192), w=SimpleNamespace(compact_expert_down=True),
                            decode_moe=SimpleNamespace(experts=experts), _linear=lambda x, _: x)
    actual = target.Target.moe(owner, layer, row, step=step)
    (send,) = sends
    assert send.dtype == torch.float32 and torch.equal(actual, row + row)


def run_native(rank, world, master, port):
    import os

    from tensorfold.cuda import small_gather
    from tensorfold.cuda.cohort import notify_failure
    from tensorfold.cuda.comm import open_comm
    from tensorfold.families.deepseek_v4.cuda.collectives import Collectives

    assert world == 8 and torch.cuda.get_device_capability() == (7, 0)
    torch.cuda.set_device(0)
    os.environ.setdefault(small_gather.ENV, str(small_gather.MAX_CAPACITY))
    comm = open_comm(rank, world, master, port)
    try:
        with pytest.MonkeyPatch.context() as patch, torch.inference_mode():
            _exercise_sp_routes(patch, rank, "cuda", Collectives(comm))
        torch.cuda.synchronize()
    except BaseException as primary:
        try:
            notify_failure()
            comm.close(abort=True)
        except BaseException as cleanup:
            raise primary from cleanup
        raise
    else:
        comm.close()
    print(f"rank {rank}: text and mixed vision global routes passed", flush=True)
