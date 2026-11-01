"""CPU ownership/fault controls; actual CUDA graph numerics remain a GPU gate."""
from contextlib import nullcontext
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import Gemma4TextConfig  # noqa: E402 - optional runtime admitted above
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda import e2b_mlp_graphs  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy  # noqa: E402 - optional runtime admitted above


def native():
    cfg=Gemma4TextConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=1,num_kv_shared_layers=0,
                        use_double_wide_mlp=False,hidden_activation="gelu_pytorch_tanh")
    return Gemma4TextMLP(cfg,0).to(torch.bfloat16).eval().requires_grad_(False)


class Stream:
    def __init__(self,**_):
        self.waits=[]
        self.drains=0
        self.fail_sync=None

    def wait_stream(self,stream):
        self.waits.append(stream)

    def wait_event(self,event):
        self.waits.append(event)

    def synchronize(self):
        self.drains+=1
        if self.fail_sync is not None:
            raise self.fail_sync


class Event:
    def record(self,stream):
        self.stream=stream


def fake_cuda(monkeypatch,*,enter_error=None,exit_error=None):
    control=SimpleNamespace(graphs=[],contexts=[],current=Stream(),active=False,pools=[])

    class Graph:
        def __init__(self):
            self.callback=None
            self.resets=0
            self.fail_reset=None
            self.fail_replay=None
            control.graphs.append(self)

        def replay(self):
            if self.fail_replay is not None:
                raise self.fail_replay
            self.callback()

        def reset(self):
            self.resets+=1
            if self.fail_reset is not None:
                raise self.fail_reset
            self.callback=None

    class Capture:
        def __init__(self,graph,**kwargs):
            self.graph,self.kwargs=graph,kwargs
            self.enters=self.exits=0
            control.contexts.append(self)

        def __enter__(self):
            self.enters+=1
            control.active=True
            if enter_error is not None:
                raise enter_error

        def __exit__(self,*_):
            self.exits+=1
            control.active=False
            if exit_error is not None:
                raise exit_error

    def pool():
        value=object()
        control.pools.append(value)
        return value

    monkeypatch.setattr(torch.cuda,"Stream",Stream)
    monkeypatch.setattr(torch.cuda,"Event",Event)
    monkeypatch.setattr(torch.cuda,"current_stream",lambda *_:control.current)
    monkeypatch.setattr(torch.cuda,"stream",lambda _:nullcontext())
    monkeypatch.setattr(torch.cuda,"CUDAGraph",Graph)
    monkeypatch.setattr(torch.cuda,"graph",Capture)
    monkeypatch.setattr(torch.cuda,"graph_pool_handle",pool)
    monkeypatch.setattr(e2b_mlp_graphs,"_geometry",lambda modules:(torch.device("cpu"),modules[0].hidden_size))
    return control


def replay_callbacks(owner):
    # Substitute only CUDA execution transport: real native CPU MLP operations
    # feed the graph's stable output in fault/ownership tests.
    for (rows,index),entry in owner.entries.items():
        def run(module=owner.modules[index],value=owner.inputs[rows],output=entry.output):
            output.copy_(module(value))
        entry.graph.callback=run


@pytest.mark.parametrize("owned",[False,True])
def test_prepare_handoff_complete_copy_replay_output_ownership_and_close(monkeypatch,owned):
    control=fake_cuda(monkeypatch)
    model=native()
    owner=e2b_mlp_graphs.NativeMLPGraphs([model],rows=(1,2),owned_output=owned)
    owner.prepare()
    assert owner.capture_stream.waits==[control.current]
    assert all(context.enters==context.exits==1 for context in control.contexts)
    assert all(context.kwargs["capture_error_mode"]=="thread_local" for context in control.contexts)
    owner.bind_stream(control.current)
    assert control.current.waits==[owner.ready]
    replay_callbacks(owner)
    x=torch.randn(1,2,16,dtype=torch.bfloat16)
    first=owner.forward(0,x)
    assert torch.equal(first,model(x))
    before=first.clone()
    second=owner.forward(0,x+1)
    assert torch.equal(second,model(x+1))
    if owned:
        assert first.data_ptr()!=second.data_ptr() and torch.equal(first,before)
    else:
        assert first is second is owner.entries[2,0].output
    assert owner.logical_buffer_bytes()>0
    owner.close()
    owner.close()
    assert owner.closed and not owner.entries and not owner.inputs and not owner.modules
    assert all(graph.resets==1 for graph in control.graphs)


def test_shared_pool_per_width_enforces_complete_layer_order_before_gpu_mutation(monkeypatch):
    control=fake_cuda(monkeypatch)
    owner=e2b_mlp_graphs.NativeMLPGraphs([native(),native()],rows=(1,2),shared_pool=True)
    owner.prepare()
    owner.bind_stream(control.current)
    replay_callbacks(owner)
    assert len(control.pools)==2
    assert control.contexts[0].kwargs["pool"] is control.contexts[1].kwargs["pool"]
    assert control.contexts[2].kwargs["pool"] is control.contexts[3].kwargs["pool"]
    x=torch.randn(1,1,16,dtype=torch.bfloat16)
    with pytest.raises(RuntimeError,match="fixed-order"):
        owner.forward(1,x)
    owner.forward(0,x)
    with pytest.raises(RuntimeError,match="fixed-order"):
        owner.forward(0,x)
    owner.forward(1,x)
    owner.forward(0,x.expand(1,2,16))
    owner.forward(1,x.expand(1,2,16))
    owner.close()


def test_failed_enter_never_calls_exit_or_releases_pending_capture_owners(monkeypatch):
    failure=RuntimeError("capture enter failed after unknown phase")
    control=fake_cuda(monkeypatch,enter_error=failure)
    owner=e2b_mlp_graphs.NativeMLPGraphs([native()],rows=(1,))
    with pytest.raises(RuntimeError) as caught:
        owner.prepare()
    assert caught.value is failure and owner.terminal and owner.poisoned
    assert control.contexts[0].enters==1 and control.contexts[0].exits==0
    assert control.graphs[0].resets==0 and owner.modules and owner.inputs
    assert owner in failure._tensorfold_retained_owners
    with pytest.raises(RuntimeError,match="terminate"):
        owner.close()


@pytest.mark.parametrize("exit_fails",[False,True])
def test_body_failure_exits_exactly_once_preserves_primary_and_retains_if_unknown(monkeypatch,exit_fails):
    primary=RuntimeError("native MLP capture body failed")
    cleanup=RuntimeError("capture exit failed")
    control=fake_cuda(monkeypatch,exit_error=cleanup if exit_fails else None)
    model=native()

    def fail_during_capture(*_):
        if control.active:
            raise primary

    model.register_forward_pre_hook(fail_during_capture)
    owner=e2b_mlp_graphs.NativeMLPGraphs([model],rows=(1,))
    with pytest.raises(RuntimeError) as caught:
        owner.prepare()
    assert caught.value is primary
    assert control.contexts[0].enters==control.contexts[0].exits==1
    if exit_fails:
        assert owner.terminal and owner.modules and owner in primary._tensorfold_retained_owners
        assert control.graphs[0].resets==0
    else:
        assert owner.closed and not owner.modules and control.graphs[0].resets==1


def test_exit_failure_without_body_failure_is_terminal_and_not_retried(monkeypatch):
    failure=RuntimeError("capture exit failed")
    control=fake_cuda(monkeypatch,exit_error=failure)
    owner=e2b_mlp_graphs.NativeMLPGraphs([native()],rows=(1,))
    with pytest.raises(RuntimeError) as caught:
        owner.prepare()
    assert caught.value is failure and owner.terminal
    assert control.contexts[0].exits==1 and control.graphs[0].resets==0


@pytest.mark.parametrize("failure",["drain","reset"])
def test_cleanup_failure_retains_graph_inputs_and_weights_until_terminal_teardown(monkeypatch,failure):
    control=fake_cuda(monkeypatch)
    owner=e2b_mlp_graphs.NativeMLPGraphs([native()],rows=(1,2))
    owner.prepare()
    owner.bind_stream(control.current)
    injected=RuntimeError("native retirement failed")
    if failure=="drain":
        owner.capture_stream.fail_sync=injected
    else:
        control.graphs[0].fail_reset=injected
    with pytest.raises(RuntimeError) as caught:
        owner.close()
    assert caught.value is injected and owner.terminal and owner.poisoned
    assert owner.inputs and owner.modules and owner.entries
    assert control.graphs[1].resets==(0 if failure=="drain" else 1)
    with pytest.raises(RuntimeError,match="terminate"):
        owner.close()


def test_disabled_install_leaves_modules_unchanged_and_reference_bypasses_graph(monkeypatch):
    control=fake_cuda(monkeypatch)
    model=native()
    layers=[SimpleNamespace(mlp=model)]
    language=SimpleNamespace(layers=layers)
    policy=E2BRowPolicy()
    assert e2b_mlp_graphs.install_mlp_graphs(language,policy) is None and layers[0].mlp is model
    keys=tuple(model.state_dict())
    owner=e2b_mlp_graphs.install_mlp_graphs(language,policy,enabled=True,rows=(1,))
    assert tuple(layers[0].mlp.state_dict())==keys
    assert layers[0].mlp.gate_proj.weight is model.gate_proj.weight
    owner.bind_stream(control.current)
    replay_callbacks(owner)
    x=torch.randn(1,1,16,dtype=torch.bfloat16)
    policy.prompt=False
    assert torch.equal(layers[0].mlp(x),model(x))
    monkeypatch.setattr(owner,"forward",lambda *_:pytest.fail("prompt/reference entered MLP graph"))
    policy.prompt=True
    assert torch.equal(layers[0].mlp(x),model(x))
    policy.prompt=False
    with policy.native_reference():
        assert torch.equal(layers[0].mlp(x),model(x))
    owner.close()


def test_invalid_geometry_and_widths_fail_before_capture(monkeypatch):
    with pytest.raises(ValueError,match="CUDA"):
        e2b_mlp_graphs.NativeMLPGraphs([native()])
    fake_cuda(monkeypatch)
    for rows in ((),(0,),(9,),(1,1),(True,)):
        with pytest.raises(ValueError,match="widths"):
            e2b_mlp_graphs.NativeMLPGraphs([native()],rows=rows)
