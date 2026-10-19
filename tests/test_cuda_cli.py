"""The CLI's CUDA path: backend choice and argument checks that run before any GPU work (any machine)."""

import argparse
from types import SimpleNamespace

import pytest

from tensorfold import cli
from tests.cli_server_fixture import cuda_cli_app, cuda_cli_engine, install_cli_server


def _family(**members):
    return SimpleNamespace(title="Test family", package=SimpleNamespace(**members))


def test_auto_backend_follows_the_platform(monkeypatch):
    both = _family(load=lambda *a, **k: None, cuda_engine=lambda *a, **k: None)
    monkeypatch.setattr(cli.sys, "platform", "darwin")
    assert cli._backend("auto", both) == "mlx"
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli._backend("auto", both) == "cuda"


def test_a_family_serves_only_the_backends_it_has():
    with pytest.raises(ValueError, match="no CUDA engine"):
        cli._backend("cuda", _family(load=lambda *a, **k: None))
    with pytest.raises(ValueError, match="NVIDIA GPUs only"):
        cli._backend("mlx", _family(cuda_engine=lambda *a, **k: None))


def test_two_gpus_need_a_master_before_anything_loads(tmp_path):
    called = []
    family = _family(cuda_engine=lambda *a, **k: called.append(k))
    args = argparse.Namespace(tp=2, rank=0, master="", master_port=29551, no_drafts=True, drafter="none",
                              mtp_drafts=None, name="", model=str(tmp_path))
    with pytest.raises(ValueError, match="--master"):
        cli._check_serve_options(args, family, "cuda")
    args.tp, args.rank = 1, 1
    with pytest.raises(ValueError, match="--rank 1 needs --tp 2"):
        cli._check_serve_options(args, family, "cuda")
    assert not called


def test_four_gpus_only_for_a_family_that_splits_over_four(tmp_path):
    called = []
    family = _family(cuda_engine=lambda *a, **k: called.append(k))
    args = argparse.Namespace(tp=4, rank=0, master="192.0.2.11", master_port=29551, no_drafts=True, drafter="none",
                              mtp_drafts=None, name="", model=str(tmp_path))
    with pytest.raises(ValueError, match="--tp 4: Test family runs on one or two GPUs"):
        cli._check_serve_options(args, family, "cuda")
    args.tp, args.rank = 2, 3
    with pytest.raises(ValueError, match="--rank 3 needs --tp 4"):
        cli._check_serve_options(args, family, "cuda")
    assert not called

def test_serve_parses_the_cuda_flags():
    args = cli.build_parser().parse_args(["serve", "owner/model", "--tp", "2", "--rank", "1", "--master", "192.0.2.11"])
    assert (args.backend, args.tp, args.rank, args.master, args.master_port) == ("auto", 2, 1, "192.0.2.11", 29551)


@pytest.mark.parametrize("override, expected", [(None, 128), (0, 0), (64, 64)])
def test_cuda_dispatch_keeps_the_resolved_context(tmp_path, monkeypatch, override, expected):
    import json

    from tensorfold import families, hub

    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 128}))
    family = _family(cuda_engine=lambda *a, **k: None)
    family.model_type = "test"
    monkeypatch.setattr(families, "detect", lambda path: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: tmp_path)
    monkeypatch.setattr("faulthandler.register", lambda *a, **k: None)
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, found, path, context: seen.append(context) or 0)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-update-check"]
    if override is not None:
        command += ["--context", str(override)]
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    assert seen == [expected]


def test_a_cuda_start_dumps_its_stacks_on_sigusr1(tmp_path, monkeypatch):
    """`kill -USR1 <pid>` shows where a CUDA start waits, as on the Mac: the handler is in place before any load."""

    import json
    import signal

    from tensorfold import families, hub

    (tmp_path / "config.json").write_text(json.dumps({"max_position_embeddings": 128}))
    family = _family(cuda_engine=lambda *a, **k: None)
    family.model_type = "test"
    monkeypatch.setattr(families, "detect", lambda path: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: tmp_path)
    events = []
    monkeypatch.setattr("faulthandler.register", lambda signum, **k: events.append(("register", signum, k)))
    monkeypatch.setattr(cli, "_serve_cuda", lambda *a: events.append(("serve",)) or 0)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-update-check"]
    assert cli.cmd_serve(cli.build_parser().parse_args(command)) == 0
    assert events == [("register", signal.SIGUSR1, {"all_threads": True}), ("serve",)]


def test_cuda_admission_metadata_does_not_enlarge_the_engine_cache(tmp_path, monkeypatch, request, capsys):
    import tensorfold.cuda.server as server

    made, served = [], []
    engine = cuda_cli_engine(max_len=8192)
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: served.append(k) or
                        cuda_cli_app(effective_context_window=8185))
    install_cli_server(monkeypatch, request, server)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    assert cli._serve_cuda(args, family, tmp_path, 262144) == 0
    assert made[0]["context"] == 262144
    assert made[0]["context_explicit"] is False
    assert served[0]["context_window"] == 262144
    assert "context: 8185" in capsys.readouterr().out


def test_the_prompt_precision_flag_parses_and_help_states_the_default():
    from tensorfold.cuda import prompt_precision

    parser = cli.build_parser()
    assert getattr(parser.parse_args(["serve", "owner/model"]), "prefill_fp8", None) is None   # the default applies
    assert parser.parse_args(["serve", "owner/model", "--prefill-fp8"]).prefill_fp8 is True
    assert parser.parse_args(["serve", "owner/model", "--no-prefill-fp8"]).prefill_fp8 is False
    default = "FP8" if prompt_precision.FP8_BY_DEFAULT else "bf16"         # one constant picks the default
    serve = next(a for a in parser._actions if a.dest == "command").choices["serve"]
    flag = next(a for a in serve._actions if a.dest == "prefill_fp8")
    assert "e4m3" in flag.help and f"Default: {default} activations" in flag.help   # --help states both precisions


@pytest.mark.parametrize("fast,flags,fp8", [(True, [], None), (True, ["--prefill-fp8"], True),
                                            (True, ["--no-prefill-fp8"], False), (False, ["--prefill-fp8"], True),
                                            (False, [], None), (False, ["--no-prefill-fp8"], False)])
def test_the_prompt_precision_is_set_before_loading_and_shown(tmp_path, monkeypatch, request, capsys, fast, flags, fp8):
    """The switch is set before the engine loads (None: the default); a checkpoint without an FP8 prompt kernel
    refuses the flag by name and serves bf16 prompts otherwise."""

    import tensorfold.cuda.server as server
    from tensorfold.cuda import prompt_precision

    asked = prompt_precision.FP8_BY_DEFAULT if fp8 is None else fp8
    seen = []

    def engine(*a, **k):
        seen.append(prompt_precision.fp8())
        return cuda_cli_engine(max_len=8192, w=SimpleNamespace(fast_prefill=fast))

    family = _family(cuda_engine=engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: cuda_cli_app(effective_context_window=8185))
    install_cli_server(monkeypatch, request, server)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"] + flags)
    try:
        if not fast and flags == ["--prefill-fp8"]:
            with pytest.raises(ValueError, match="no FP8 kernel"):
                cli._serve_cuda(args, family, tmp_path, 8192)
        else:
            assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
            shown = "FP8 activations" if asked and fast else "bf16 activations"
            assert f"prompts: {shown}" in capsys.readouterr().out
        assert seen == [asked]
    finally:
        prompt_precision.set_fp8(prompt_precision.FP8_BY_DEFAULT)


def test_serve_parses_the_kv_cache_flag():
    plain = cli.build_parser().parse_args(["serve", "owner/model"])
    assert plain.kv_dtype == "bf16"                        # the cache stays bf16 unless it is asked for
    assert cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "int8"]).kv_dtype == "int8"
    assert cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "int4"]).kv_dtype == "int4"
    assert cli.build_parser().parse_args(["serve", "owner/model", "--mtp-confidence", "0.6"]).mtp_confidence == 0.6
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["serve", "owner/model", "--kv-dtype", "fp8"])


@pytest.mark.torch
def test_kv_dtype_reaches_only_the_families_that_declare_it(tmp_path, monkeypatch):
    """``CUDA_KV_DTYPES`` is the gate: a family that does not list a dtype is refused before anything loads, and
    a family that does gets it through its engine."""

    import json

    from tensorfold.families import glm5_next, qwen3_5, qwen4_exp
    from tensorfold.families.qwen4_exp.cuda import engine as fn_engine

    made = []
    monkeypatch.setattr(fn_engine, "FlashNextEngine", lambda *a, **k: made.append(k) or SimpleNamespace(**k))
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"mtp.fc.weight": "x"}}))

    from tensorfold.families.qwen4_exp.kv_formats import DTYPES
    assert qwen4_exp.CUDA_KV_DTYPES == DTYPES
    assert DTYPES[:3] == ("bf16", "int8", "int4")
    assert qwen4_exp.cuda_engine(tmp_path, kv_dtype="int8").kv_dtype == "int8"
    assert made[-1]["kv_dtype"] == "int8"
    with pytest.raises(ValueError, match="kv-dtype"):
        qwen4_exp.cuda_engine(tmp_path, kv_dtype="fp8")
    for module in (qwen3_5, glm5_next):
        args = argparse.Namespace(tp=1, rank=0, master="", master_port=29551, no_drafts=True, drafter="none",
                                  mtp_drafts=None, name="", model=str(tmp_path), kv_dtype="int8")
        with pytest.raises(ValueError, match="KV cache, not --kv-dtype int8"):
            cli._check_serve_options(args, SimpleNamespace(title=module.TITLE, package=module), "cuda")
    assert not made[1:]


@pytest.mark.parametrize("flags,backend,family,message", [
    (["--kv-dtype", "int8"], "mlx", "qwen4_exp", "MLX path caches keys and values as bf16"),
    (["--kv-dtype", "int4"], "cuda", "qwen3_5", "KV cache, not --kv-dtype int4"),
    (["--kv-dtype", "int8"], "cuda", "nemotron_h", "KV cache, not --kv-dtype int8"),
    (["--mtp-confidence", "0.6"], "mlx", "qwen4_exp", "on MLX has no such rule"),
    (["--mtp-confidence", "0.6"], "cuda", "glm5_next", "on CUDA has no such rule"),
    (["--mtp-confidence", "1.5"], "cuda", "qwen4_exp", "--mtp-confidence must be between 0 and 1"),
    (["--mtp-confidence", "-0.1"], "cuda", "qwen4_exp", "--mtp-confidence must be between 0 and 1"),
    (["--prefill-fp8"], "mlx", "qwen3_5", "Qwen3.8 dense on MLX has none"),
    (["--prefill-fp8"], "cuda", "nemotron_h", "on CUDA has none"),
    (["--prefill-fp8"], "cuda", "glm5_next", "on CUDA has none"),
])
def test_cache_and_confidence_options_are_refused_before_any_download(tmp_path, monkeypatch, flags, backend, family,
                                                                      message):
    """Every family and backend answers ``--kv-dtype``, ``--mtp-confidence`` and ``--prefill-fp8``: served as asked,
    or refused by name before a weight moves; none ignores them."""

    import importlib

    from tensorfold import families, hub

    module = importlib.import_module(f"tensorfold.families.{family}")
    found = SimpleNamespace(title=module.TITLE, package=module, model_type=family)
    monkeypatch.setattr(families, "detect", lambda path: found)
    monkeypatch.setattr(cli, "_backend", lambda choice, fam: backend)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weights were fetched before the refusal"))
    monkeypatch.setattr(families, "require_readable",
                        lambda *a: pytest.fail("the checkpoint was read before the refusal"))
    command = ["serve", str(tmp_path), "--no-update-check"] + flags
    with pytest.raises(ValueError, match=message):
        cli.cmd_serve(cli.build_parser().parse_args(command))


@pytest.mark.parametrize("flags", [["--kv-dtype", "int8"], ["--kv-dtype", "int4", "--mtp-confidence", "0.6"],
                                   ["--mtp-confidence", "0"], ["--mtp-confidence", "1"]])
def test_flash_next_on_cuda_takes_both_options(tmp_path, flags):
    from tensorfold.families import qwen4_exp

    args = cli.build_parser().parse_args(["serve", str(tmp_path)] + flags)
    family = SimpleNamespace(title=qwen4_exp.TITLE, package=qwen4_exp, model_type="qwen4_exp")
    assert cli._check_serve_options(args, family, "cuda") is None


@pytest.mark.torch
def test_no_cuda_engine_serves_one_token_a_round_by_default(tmp_path, monkeypatch):
    """Everything on the lanes: a CUDA engine whose drafter is missing refuses to start rather than decode one token
    a round, and names the fix; --no-drafts (the serial reference) still starts."""

    import json

    from tensorfold.families import glm5_next, qwen3_5, qwen4_exp
    from tensorfold.families.glm5_next.cuda import engine as glm_engine
    from tensorfold.families.qwen3_5.cuda import engine as q27_engine
    from tensorfold.families.qwen4_exp.cuda import engine as fn_engine

    made = []
    stub = lambda *a, **k: made.append(k) or SimpleNamespace(**k)      # noqa: E731
    monkeypatch.setattr(q27_engine, "Qwen27Engine", stub)
    monkeypatch.setattr(fn_engine, "FlashNextEngine", stub)
    monkeypatch.setattr(glm_engine, "GlmEngine", stub)

    # the 27B drafts with DFlash2: without it, only the serial reference
    with pytest.raises(ValueError, match="tensorfold pull z-lab/Qwen3.8-27B-DFlash2"):
        qwen3_5.cuda_engine(tmp_path, drafter="")
    assert qwen3_5.cuda_engine(tmp_path, drafter="", no_drafts=True).allow_copy is False
    monkeypatch.setattr(qwen3_5, "gb10", lambda: True)
    one = qwen3_5.cuda_engine(tmp_path, drafter=str(tmp_path))
    assert (one.max_rows, one.tree_rows) == (128, 16)          # one stream on one GB10: copies widen, trees at 16
    monkeypatch.setattr(qwen3_5, "gb10", lambda: False)
    other = qwen3_5.cuda_engine(tmp_path, drafter=str(tmp_path))
    assert (other.max_rows, other.tree_rows) == (12, None)     # other GPUs keep 0.5.0's rows until measured
    monkeypatch.setattr(qwen3_5, "gb10", lambda: True)
    many = qwen3_5.cuda_engine(tmp_path, drafter=str(tmp_path), parallel=4)
    assert (many.max_rows, many.tree_rows) == (12, None)       # concurrent streams keep their rows
    ranks = qwen3_5.cuda_engine(tmp_path, drafter=str(tmp_path), tp=2, master="192.0.2.10")
    assert (ranks.max_rows, ranks.tree_rows) == (12, None)     # two ranks too

    # Flash Next drafts with the checkpoint's MTP head: a checkpoint without it serves only the serial reference
    index = {"weight_map": {"model.layers.0.mlp.gate.weight": "model.safetensors"}}
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    with pytest.raises(ValueError, match="no MTP head"):
        qwen4_exp.cuda_engine(tmp_path)
    assert qwen4_exp.cuda_engine(tmp_path, no_drafts=True).depth == 0
    index["weight_map"]["mtp.fc.weight"] = "model.safetensors"
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))
    assert qwen4_exp.cuda_engine(tmp_path).depth == 6
    assert made[-1]["confidence"] == 0.7                                   # one stream or many
    assert qwen4_exp.cuda_engine(tmp_path, mtp_confidence=0.6).confidence == 0.6
    assert made[-1]["share"] == 0.0                                        # whole prompt passes unless asked
    assert qwen4_exp.cuda_engine(tmp_path, decode_share=0.25).share == 0.25

    # GLM: --mtp-drafts 0 with the DFlash2 drafter still drafts (DFlash2 alone); without it, the serial reference
    glm = dict(tp=2, master="192.0.2.10")
    assert glm5_next.cuda_engine(tmp_path, drafter=str(tmp_path), mtp_drafts=0, **glm).policy == "fc5:0.3"
    assert glm5_next.cuda_engine(tmp_path, mtp_drafts=0, **glm).policy == "0"
    assert glm5_next.cuda_engine(tmp_path, drafter=str(tmp_path), **glm).policy == "auto"
    assert glm5_next.cuda_engine(tmp_path, mtp_drafts=2, **glm).policy == "2"


@pytest.mark.parametrize("flag, streams", [(None, None), ("auto", None), ("1", None), ("4", 4)])
def test_cuda_parallel_is_one_request_at_a_time_unless_a_number_asks(tmp_path, monkeypatch, request, flag, streams):
    import tensorfold.cuda.server as server

    made = []
    engine = cuda_cli_engine(context_window=4096)
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: cuda_cli_app(effective_context_window=4096))
    install_cli_server(monkeypatch, request, server)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"] + (["--parallel", flag] if flag else [])
    assert cli._serve_cuda(cli.build_parser().parse_args(command), family, tmp_path, 4096) == 0
    assert made[0].get("parallel") == streams


@pytest.mark.parametrize("available,capability,name,expected", [
    (True, (12, 1), "NVIDIA GB10", True), (True, (12, 1), "", True), (True, (11, 0), "NVIDIA GB10", True),
    (True, (12, 0), "NVIDIA GeForce RTX 5090", False), (True, (9, 0), "NVIDIA H100 80GB HBM3", False),
    (False, (12, 1), "NVIDIA GB10", False)])
def test_gb10_reads_the_devices_capability_or_name(monkeypatch, available, capability, name, expected):
    """The lone stream's wide windows are for a GB10: compute capability 12.1 or the device's name."""

    import sys
    from types import ModuleType

    from tensorfold.families import qwen3_5

    torch = ModuleType("torch")
    torch.cuda = SimpleNamespace(is_available=lambda: available, get_device_capability=lambda i: capability,
                                 get_device_name=lambda i: name)
    monkeypatch.setitem(sys.modules, "torch", torch)
    assert qwen3_5.gb10() is expected


@pytest.mark.parametrize("declares, expected", [(True, 6), (False, None)])
def test_checkpoint_slots_reach_a_cuda_engine_only_when_its_family_declares_them(tmp_path, monkeypatch, request, declares,
                                                                                    expected):
    import tensorfold.cuda.server as server

    made = []
    engine = cuda_cli_engine(context_window=4096)
    family = _family(cuda_engine=lambda *a, **k: made.append(k) or engine,
                     **({"CUDA_CHECKPOINT_SLOTS": True} if declares else {}))
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: cuda_cli_app(effective_context_window=4096))
    install_cli_server(monkeypatch, request, server)
    command = ["serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--parallel", "2", "--checkpoint-slots", "6"]
    assert cli._serve_cuda(cli.build_parser().parse_args(command), family, tmp_path, 4096) == 0
    assert made[0].get("checkpoint_slots") == expected


@pytest.mark.parametrize("flags, message", [
    (["--checkpoint-slots", "6"], "one stream keeps 4"),
    (["--checkpoint-slots", "6", "--parallel", "1"], "one stream keeps 4"),
    (["--checkpoint-slots", "6", "--parallel", "01"], "one stream keeps 4"),
    (["--checkpoint-slots", "6", "--parallel", "+1"], "one stream keeps 4"),
    (["--checkpoint-slots", "6", "--parallel", "0"], "one stream keeps 4"),
    (["--checkpoint-slots", "6", "--parallel", "-1"], "one stream keeps 4"),
    (["--checkpoint-slots", "0", "--parallel", "2"], "1 or more, not 0"),
])
def test_27b_checkpoint_slots_are_refused_before_any_download_where_they_would_not_act(tmp_path, monkeypatch, flags,
                                                                                         message):
    from tensorfold import families, hub
    from tensorfold.families import qwen3_5

    found = SimpleNamespace(title=qwen3_5.TITLE, package=qwen3_5, model_type="qwen3_5")
    monkeypatch.setattr(families, "detect", lambda path: found)
    monkeypatch.setattr(cli, "_backend", lambda choice, fam: "cuda")
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weights were fetched before the refusal"))
    monkeypatch.setattr(families, "require_readable",
                        lambda *a: pytest.fail("the checkpoint was read before the refusal"))
    with pytest.raises(ValueError, match=message):
        cli.cmd_serve(cli.build_parser().parse_args(["serve", str(tmp_path), "--no-update-check"] + flags))


@pytest.mark.parametrize("backend, flags", [("cuda", ["--parallel", "2", "--checkpoint-slots", "6"]),
                                            ("mlx", ["--checkpoint-slots", "6"])])
def test_27b_checkpoint_slots_pass_with_parallel_streams_and_on_mlx(tmp_path, backend, flags):
    from tensorfold.families import qwen3_5

    args = cli.build_parser().parse_args(["serve", str(tmp_path)] + flags)
    family = SimpleNamespace(title=qwen3_5.TITLE, package=qwen3_5, model_type="qwen3_5")
    assert cli._check_serve_options(args, family, backend) is None


@pytest.mark.torch
@pytest.mark.parametrize("slots, keep", [(None, None), (6, 6)])
def test_27b_cuda_engine_takes_the_checkpoint_slots_as_its_kept_states(tmp_path, monkeypatch, slots, keep):
    from tensorfold.families import qwen3_5
    from tensorfold.families.qwen3_5.cuda import engine as qwen27

    made = []
    monkeypatch.setattr(qwen27, "Qwen27Engine", lambda *a, **k: made.append(k) or SimpleNamespace(**k))
    options = {"parallel": 2} | ({"checkpoint_slots": slots} if slots is not None else {})
    qwen3_5.cuda_engine(tmp_path, drafter=str(tmp_path), **options)
    assert made[-1]["keep"] == keep and made[-1]["streams"] == 2


@pytest.mark.parametrize("flags,backend,message", [
    (["--vision-offload"], "cuda", "--vision-offload needs --vision"),
    (["--vision", "--vision-offload"], "mlx", "--vision-offload is for the CUDA backend"),
])
def test_vision_offload_is_a_cuda_option_that_needs_vision(tmp_path, flags, backend, message):
    from tensorfold.families import qwen3_5

    args = cli.build_parser().parse_args(["serve", str(tmp_path)] + flags)
    family = SimpleNamespace(title=qwen3_5.TITLE, package=qwen3_5, model_type="qwen3_5")
    with pytest.raises(ValueError, match=message):
        cli._check_serve_options(args, family, backend)


@pytest.mark.parametrize("flags,backend,message", [
    (["--tokenizer", "t.json"], "mlx", "--tokenizer is read by the CUDA server"),
    (["--tokenizer", "t.json", "--vision"], "cuda", "cannot be combined with --vision"),
    (["--tokenizer", "t.json", "--tp", "2", "--master", "10.0.0.1"], "cuda", "serves one rank only"),
    (["--tokenizer", "t.json"], "cuda", None),
])
def test_tokenizer_override_is_a_cuda_text_option(tmp_path, flags, backend, message):
    """Image prompts are encoded by the vision processor's own checkpoint tokenizer, which --tokenizer does not reach;
    other ranks would compile grammars from a tokenizer nobody checks against rank 0's."""

    from tensorfold.families import qwen3_5

    args = cli.build_parser().parse_args(["serve", str(tmp_path)] + flags)
    family = SimpleNamespace(title=qwen3_5.TITLE, package=qwen3_5, model_type="qwen3_5")
    if message is None:
        assert cli._check_serve_options(args, family, backend) is None
        return
    with pytest.raises(ValueError, match=message):
        cli._check_serve_options(args, family, backend)


@pytest.mark.parametrize("rank", [1, 2, 3])
def test_four_rank_followers_retire_the_owned_engine_without_http(tmp_path, monkeypatch, rank):
    events = []
    engine = SimpleNamespace(follow=lambda: events.append("follow"), close=lambda: events.append("close"))
    family = _family(CUDA_TP4=True, cuda_engine=lambda *a, **k: engine)
    family.model_type = "test"
    monkeypatch.setattr(cli.stacks, "arm", lambda: None)
    args = cli.build_parser().parse_args([
        "serve", str(tmp_path), "--tp", "4", "--rank", str(rank), "--master", "192.0.2.11", "--no-drafts",
    ])
    assert cli._serve_cuda(args, family, tmp_path) == 0
    assert events == ["follow", "close"]


def test_invalid_numeric_options_fail_before_auth_environment_or_file_access(tmp_path, monkeypatch):
    from tensorfold.server import authentication

    monkeypatch.setenv("TENSORFOLD_API_KEY", "fixture-key")
    monkeypatch.setattr(authentication, "configure", lambda *a: pytest.fail("authentication acquired before refusal"))
    args = cli.build_parser().parse_args([
        "serve", str(tmp_path), "--decode-share", "nan", "--api-key-file", str(tmp_path / "missing-keys"),
    ])
    with pytest.raises(ValueError, match="--decode-share must be a finite number"):
        cli.cmd_serve(args)
    assert cli.os.environ["TENSORFOLD_API_KEY"] == "fixture-key"


@pytest.mark.parametrize("tokens", [None, 16384])
def test_auth_and_image_budget_reach_the_owned_cuda_listener(tmp_path, monkeypatch, request, tokens):
    from tensorfold.cuda import server
    from tensorfold.server.authentication import KeyStore

    seen = []
    engine = cuda_cli_engine(max_len=8192)
    family = _family(cuda_engine=lambda *a, **k: engine)
    family.model_type = "test"

    def app(*a, **kw):
        value = cuda_cli_app(effective_context_window=8192)
        seen.append((value, kw))
        return value

    def handler(app):
        assert app.auth is keys
        return object()

    monkeypatch.setattr(server, "App", app)
    monkeypatch.setattr(server, "make_handler", handler)
    install_cli_server(monkeypatch, request, server)
    flags = [] if tokens is None else ["--vision", "--vision-image-tokens", str(tokens)]
    args = cli.build_parser().parse_args([
        "serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--max-http-connections", "3",
        "--max-pending-requests", "5", *flags,
    ])
    keys = args.auth = KeyStore(["fixture-key"])
    assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
    assert seen[0][1]["max_pending_requests"] == 5
    if tokens is None:
        assert "vision_image_tokens" not in seen[0][1]
    else:
        assert seen[0][1]["vision_image_tokens"] == tokens


@pytest.mark.parametrize("template_option", ["file", "name"])
def test_explicit_model_bindings_and_generation_defaults_reach_cuda_owners(tmp_path, monkeypatch, request,
                                                                         template_option):
    import json

    from tensorfold.cuda import server

    (tmp_path / "generation_config.json").write_text(json.dumps({
        "max_new_tokens": 41, "do_sample": True, "temperature": .7, "top_k": 20, "top_p": .95, "min_p": .1,
    }))
    template = tmp_path / "custom.jinja"
    template.write_text("{{ messages | length }}")
    option, value = (("--chat-template", template) if template_option == "file"
                     else ("--chat-template-name", "tool_use"))
    seen_engine, seen_app = [], []

    def engine(*args, **kwargs):
        seen_engine.append(kwargs)
        return cuda_cli_engine(eos=tuple(kwargs["eos_token_ids"]), max_len=8192)

    def app(*args, **kwargs):
        seen_app.append(kwargs)
        return cuda_cli_app(effective_context_window=8192)

    family = _family(cuda_engine=engine)
    family.model_type = "test"
    monkeypatch.setattr(server, "App", app)
    install_cli_server(monkeypatch, request, server)
    args = cli.build_parser().parse_args([
        "serve", str(tmp_path), "--backend", "cuda", "--no-drafts", "--eos-token-id", "0", "--eos-token-id", "7",
        option, str(value), "--max-tokens", "23", "--temperature", "0", "--top-k", "0", "--top-p", "0", "--min-p", "0",
    ])
    assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
    assert len(seen_engine) == len(seen_app) == 1
    assert seen_engine[0]["eos_token_ids"] == [0, 7]
    key = "chat_template_file" if template_option == "file" else "chat_template_name"
    assert seen_engine[0][key] == seen_app[0][key] == str(value)
    assert "eos_token_ids" not in seen_app[0]
    assert seen_app[0]["max_tokens"] == 23
    assert seen_app[0]["sampling"] == {"temperature": 0.0, "top_k": 0, "top_p": 0.0, "min_p": 0.0}


@pytest.mark.parametrize("metadata, expected", [({}, 4096), ({"max_length": 71}, 4096), ({"max_new_tokens": 41}, 41)])
def test_cuda_app_inherits_only_the_reply_limit_metadata(tmp_path, monkeypatch, request, metadata, expected):
    import json

    from tensorfold.cuda import server

    (tmp_path / "generation_config.json").write_text(json.dumps(metadata))
    seen = []
    family = _family(cuda_engine=lambda *a, **k: cuda_cli_engine(max_len=8192))
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: seen.append(k) or cuda_cli_app(effective_context_window=8192))
    install_cli_server(monkeypatch, request, server)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
    assert seen[0]["max_tokens"] == expected


@pytest.mark.parametrize("metadata", [{"max_new_tokens": 0}, {"do_sample": True, "temperature": "invalid"}])
def test_invalid_metadata_defaults_are_refused_before_cuda_engine_acquisition(tmp_path, metadata):
    import json

    (tmp_path / "generation_config.json").write_text(json.dumps(metadata))
    family = _family(cuda_engine=lambda *a, **k: pytest.fail("engine acquired before invalid defaults were refused"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts"])
    with pytest.raises(ValueError):
        cli._serve_cuda(args, family, tmp_path, 8192)


@pytest.mark.parametrize("flags, message", [(["--eos-token-id", "-1"], "--eos-token-id")])
def test_invalid_model_binding_flags_are_refused_before_metadata_or_authentication(tmp_path, monkeypatch, flags,
                                                                                message):
    from tensorfold.server import authentication

    monkeypatch.setattr(authentication, "configure", lambda *a: pytest.fail("authentication acquired before refusal"))
    monkeypatch.setattr(cli, "_config_dir", lambda *a: pytest.fail("metadata read before refusal"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", *flags])
    with pytest.raises(ValueError, match=message):
        cli.cmd_serve(args)


@pytest.mark.parametrize("maximum", [0, -7])
def test_explicit_cli_reply_limit_keeps_existing_owner_clamp(tmp_path, monkeypatch, request, maximum):
    from tensorfold.cuda import server

    actual_app = server.App
    # A launch override still wins over malformed inherited reply metadata.
    (tmp_path / "generation_config.json").write_text('{"max_new_tokens":0}')
    seen = []
    family = _family(cuda_engine=lambda *a, **k: cuda_cli_engine(max_len=8192))
    family.model_type = "test"
    monkeypatch.setattr(server, "App", lambda *a, **k: seen.append(k) or cuda_cli_app(effective_context_window=8192))
    install_cli_server(monkeypatch, request, server)
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-drafts",
                                        "--max-tokens", str(maximum)])
    assert cli._serve_cuda(args, family, tmp_path, 8192) == 0
    assert seen[0]["max_tokens"] == maximum
    assert actual_app._requested_tokens(SimpleNamespace(max_tokens=maximum), {}) == 1


@pytest.mark.parametrize("flags, kind, message", [
    (["--tp", "2"], "test", "needs --master"),
    (["--rank", "1"], "test", "--rank 1 needs --tp 2"),
    (["--tp", "2", "--rank", "2", "--master", "192.0.2.11"], "test", "--rank 2 needs --tp 3"),
    (["--tp", "4", "--master", "192.0.2.11"], "test", "runs on one or two GPUs"),
    (["--tp", "2", "--master", "192.0.2.11", "--master-port", "-1"], "test", "1 to 65535"),
    (["--tp", "2", "--master", "192.0.2.11", "--master-port", "65536"], "test", "1 to 65535"),
    (["--tp", "2", "--master", "192.0.2.11", "--parallel", "2"], "qwen4_exp", "one request at a time"),
    (["--vision"], "qwen4_exp", "one GPU with --parallel 2 or more"),
    (["--vision", "--tp", "2", "--master", "192.0.2.11"], "qwen4_exp", "one GPU with --parallel 2 or more"),
])
def test_unsupported_topology_is_refused_before_weight_download(tmp_path, monkeypatch, flags, kind, message):
    from tensorfold import families, hub

    family = _family(cuda_engine=lambda *a, **k: pytest.fail("engine acquired before refusal"))
    family.model_type = kind
    monkeypatch.setattr(families, "detect", lambda *a: family)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weights resolved before refusal"))
    monkeypatch.setattr(families, "require_readable", lambda *a: pytest.fail("checkpoint read before refusal"))
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", "--no-update-check", *flags])
    with pytest.raises(ValueError, match=message):
        cli.cmd_serve(args)


@pytest.mark.parametrize("tp, rank, port", [(1, 0, 0), (1, 0, 29551), (2, 1, 1), (4, 0, 65535), (4, 3, 29551)])
def test_declared_topologies_and_port_endpoints_pass_early_validation(tmp_path, tp, rank, port):
    family = _family(cuda_engine=lambda *a, **k: None, CUDA_TP4=True)
    family.model_type = "test"
    args = cli.build_parser().parse_args([
        "serve", str(tmp_path), "--tp", str(tp), "--rank", str(rank), "--master", "192.0.2.11",
        "--master-port", str(port),
    ])
    assert cli._check_serve_options(args, family, "cuda") is None


@pytest.mark.parametrize("kind, tp, rank", [("qwen3_5", 2, 0), ("qwen3_5", 2, 1),
                                             ("glm5_next", 4, 0), ("glm5_next", 4, 3)])
def test_multi_rank_port_zero_is_refused_before_any_weight_resolution(tmp_path, monkeypatch, kind, tp, rank):
    import importlib

    from tensorfold import families, hub

    package = importlib.import_module(f"tensorfold.families.{kind}")
    family = SimpleNamespace(title=package.TITLE, model_type=kind, package=package)
    monkeypatch.setattr(families, "detect", lambda *a: family)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weights resolved for an undiscoverable port"))
    monkeypatch.setattr(package, "cuda_engine", lambda *a, **k: pytest.fail("engine acquired for port zero"))
    args = cli.build_parser().parse_args([
        "serve", str(tmp_path), "--backend", "cuda", "--no-update-check", "--tp", str(tp), "--rank", str(rank),
        "--master", "192.0.2.11", "--master-port", "0",
    ])
    with pytest.raises(ValueError, match="1 to 65535 for multiple ranks"):
        cli.cmd_serve(args)


def test_mlx_keeps_an_unused_port_zero_option(tmp_path):
    family = _family(load=lambda *a, **k: None)
    family.model_type = "test"
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "mlx", "--master-port", "0"])
    assert cli._check_serve_options(args, family, "mlx") is None


@pytest.mark.parametrize("depth", [0, 4])
@pytest.mark.parametrize("drafter", ["auto", "none"])
def test_explicit_qwen_moe_mtp_does_not_resolve_the_implicit_external_drafter(monkeypatch, depth, drafter):
    from tensorfold import hub

    family = _family(DRAFTER="z-lab/Qwen3.6-35B-A3B-DFlash", CUDA_DRAFTER="")
    family.model_type = "qwen3_5_moe"
    monkeypatch.setattr(hub, "cached", lambda *a, **k: pytest.fail("implicit DFlash resolved for selected MTP"))
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("external drafter resolved for selected MTP"))
    assert cli._drafter(family, drafter, "mlx", mtp_drafts=depth) == ""


@pytest.mark.parametrize("flags, refused", [
    (["--mtp-drafts", "4", "--drafter", "z-lab/custom"], True),
    (["--mtp-drafts", "0", "--drafter", "z-lab/custom"], True),
    (["--mtp-drafts", "4", "--drafter", "z-lab/custom", "--no-drafts"], False),
    (["--drafter", "z-lab/custom"], False),
])
def test_qwen_moe_mtp_rejects_only_an_active_conflicting_external_drafter(tmp_path, flags, refused):
    family = _family(load=lambda *a, **k: None)
    family.model_type = "qwen3_5_moe"
    args = cli.build_parser().parse_args(["serve", str(tmp_path), *flags])
    if refused:
        with pytest.raises(ValueError, match="cannot be combined"):
            cli._check_serve_options(args, family, "mlx")
    else:
        assert cli._check_serve_options(args, family, "mlx") is None


def test_qwen_moe_implicit_dflash_and_explicit_external_drafter_keep_their_paths(tmp_path, monkeypatch):
    from tensorfold import hub

    family = _family(DRAFTER="z-lab/Qwen3.6-35B-A3B-DFlash", CUDA_DRAFTER="")
    family.model_type = "qwen3_5_moe"
    cached, resolved = [], []
    monkeypatch.setattr(hub, "cached", lambda repo: cached.append(repo) or tmp_path)
    monkeypatch.setattr(hub, "_cached_weights_complete", lambda *a, **k: True)
    monkeypatch.setattr(hub, "resolve", lambda repo: resolved.append(repo) or tmp_path)
    assert cli._drafter(family, "auto", "mlx") == str(tmp_path)
    assert cached == ["z-lab/Qwen3.6-35B-A3B-DFlash"]
    assert cli._drafter(family, "z-lab/custom", "mlx") == str(tmp_path)
    assert resolved == ["z-lab/custom"]
    assert cli._drafter(family, "auto", "cuda") == ""


def test_qwen_moe_invalid_mtp_cap_refuses_before_resolving_checkpoint_weights(tmp_path, monkeypatch):
    from tensorfold import families, hub

    family = _family(load=lambda *a, **k: pytest.fail("model loaded before invalid cap refusal"))
    family.model_type = "qwen3_5_moe"
    monkeypatch.setattr(families, "detect", lambda *a: family)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weights resolved before invalid cap refusal"))
    args = cli.build_parser().parse_args([
        "serve", str(tmp_path), "--backend", "mlx", "--mtp-drafts", "-1", "--no-update-check",
    ])
    with pytest.raises(ValueError, match="nonnegative integer"):
        cli.cmd_serve(args)


def test_the_drafter_precision_flag_reaches_only_the_families_that_declare_it(tmp_path, monkeypatch, request):
    """``--drafter-bits`` is refused on a CUDA family that does not declare ``CUDA_DRAFTER_BITS`` (never silently
    ignored) and reaches the engine of one that does; MLX families read it in their own loaders."""

    import tensorfold.cuda.server as server
    from tensorfold.serve_options import check

    parser = cli.build_parser()
    plain, declared = _family(), _family(CUDA_DRAFTER_BITS=True)
    for family in (plain, declared):
        family.model_type = "test"

    def args(*flags):
        return parser.parse_args(["serve", str(tmp_path), "--backend", "cuda", *flags])

    with pytest.raises(ValueError, match="--drafter-bits: Test family on CUDA"):
        check(args("--drafter-bits", "8"), plain, "cuda")
    check(args("--drafter-bits", "8"), declared, "cuda")
    check(args("--drafter-bits", "8"), plain, "mlx")

    made = []
    monkeypatch.setattr(server, "App", lambda *a, **k: cuda_cli_app(effective_context_window=8185))
    install_cli_server(monkeypatch, request, server)
    declared.package.cuda_engine = lambda *a, **k: made.append(k) or cuda_cli_engine(max_len=8192)
    for flags, expected in ((["--drafter-bits", "8"], 8), ([], None)):
        assert cli._serve_cuda(args("--no-drafts", *flags), declared, tmp_path, 8192) == 0
        assert made[-1].get("drafter_bits") == expected
