"""Unconnected stateless native-MLP CUDA graph candidate.

Capture the original HF operators, including native GELU's BF16 output and the
separate BF16 product. No compiler fusion, weight copy, cache policy or provider
replacement is introduced. One caller-owned exclusive CUDA stream replays.
"""
from dataclasses import dataclass
import sys

import torch
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextMLP

from tensorfold.cleanup import raise_failures, rollback


@dataclass
class _Entry:
    graph: object
    output: object


def _geometry(modules):
    if not modules or any(type(module) is not Gemma4TextMLP or module.training for module in modules):
        raise ValueError("MLP graphs require original eval-mode Gemma4TextMLP modules")
    hidden = modules[0].hidden_size
    device = modules[0].gate_proj.weight.device
    if device.type != "cuda":
        raise ValueError("MLP graphs require the qualified CUDA inference device")
    for module in modules:
        if module.hidden_size != hidden:
            raise ValueError("MLP graph layers must share the input hidden width")
        if any(parameter.device != device or parameter.dtype != torch.bfloat16 or parameter.requires_grad
               for parameter in module.parameters()):
            raise ValueError("MLP graphs borrow immutable BF16 weights on one device")
    return device, hidden


class NativeMLPGraphs:
    """Bounded startup-prepared graphs; no per-request capture or fallback.

    Caller holds the qualified BLAS policy, serializes preparation/capture with
    other CUDA work, then binds one exclusive replay stream. Shared pools are
    opt-in per row width and require complete layer0..N-1 replay in capture order;
    a partial failed shared-pool pass is not silently resumed from layer0.

    Borrowed outputs live until the same entry replays (and shared-pool outputs
    until the next layer). Native decoder post-MLP norm consumes them immediately
    into fresh tensors. owned_output=True instead clones the complete result.

    Unknown capture/retirement state is terminal: close refuses to release graph
    inputs/weights. The primary error retains this owner; the startup caller must
    retain that error and terminate the process, not continue CUDA serving.
    """

    def __init__(self, modules, *, rows=tuple(range(1, 9)), shared_pool=False, owned_output=False):
        modules, rows = tuple(modules), tuple(rows)
        if (not rows or len(rows) != len(set(rows))
                or any(type(row) is not int or not 1 <= row <= 8 for row in rows)):
            raise ValueError("MLP graph rows must be distinct admitted widths1..8")
        if type(shared_pool) is not bool or type(owned_output) is not bool:
            raise ValueError("MLP graph pool/output switches must be boolean")
        self.device, self.hidden = _geometry(modules)
        self.modules, self.rows = modules, tuple(sorted(rows))
        self.shared_pool, self.owned_output = shared_pool, owned_output
        self.inputs, self.entries, self.pools = {}, {}, {}
        self.capture_stream = self.replay_stream = self.ready = None
        self.pending_graph = self.pending_context = None
        self.prepared = self.closed = self.poisoned = self.terminal = False
        self._row_in_progress, self._next_layer = None, 0

    def _healthy(self):
        if self.closed or self.poisoned:
            raise RuntimeError("MLP graph owner is closed or poisoned")

    def _capture(self, module, value, pool):
        graph = torch.cuda.CUDAGraph()
        self.pending_graph = graph
        context = torch.cuda.graph(graph, pool=pool, stream=self.capture_stream, capture_error_mode="thread_local")
        self.pending_context = context
        entered = False
        try:
            # Match Python context-manager semantics: never call __exit__ when
            # __enter__ failed. The pinned manager may fail with capture live.
            context.__enter__()
            entered = True
            try:
                output = module(value)
            except BaseException as primary:
                status = sys.exc_info()
                try:
                    context.__exit__(*status)
                except BaseException as cleanup:
                    self.terminal = True
                    raise_failures(primary, [cleanup])
                raise
            try:
                context.__exit__(None, None, None)
            except BaseException:
                self.terminal = True
                raise
        except BaseException:
            if not entered:
                self.terminal = True
            raise
        return _Entry(graph, output)

    @torch.inference_mode()
    def prepare(self):
        self._healthy()
        if self.prepared:
            raise RuntimeError("MLP graphs are prepared once before serving")
        try:
            self.capture_stream = torch.cuda.Stream(device=self.device)
            self.capture_stream.wait_stream(torch.cuda.current_stream(self.device))
            for rows in self.rows:
                if self.shared_pool:
                    self.pools[rows] = torch.cuda.graph_pool_handle()
                with torch.cuda.stream(self.capture_stream):
                    self.inputs[rows] = torch.zeros((1, rows, self.hidden), device=self.device, dtype=torch.bfloat16)
                for index, module in enumerate(self.modules):
                    with torch.cuda.stream(self.capture_stream):
                        for _ in range(3):
                            module(self.inputs[rows])
                    self.capture_stream.synchronize()
                    self.entries[rows, index] = self._capture(module, self.inputs[rows], self.pools.get(rows))
                    self.pending_graph = self.pending_context = None
            self.ready = torch.cuda.Event()
            self.ready.record(self.capture_stream)
            self.prepared = True
        except BaseException as primary:
            self.poisoned = True
            rollback(self, primary, self.close)

    def bind_stream(self, stream):
        self._healthy()
        if not self.prepared:
            raise RuntimeError("prepare MLP graphs before binding the replay stream")
        if self.replay_stream is not None and self.replay_stream != stream:
            raise ValueError("MLP graphs belong to one exclusive replay stream")
        if self.replay_stream is None:
            self.replay_stream = stream
            try:
                stream.wait_event(self.ready)
            except BaseException:
                self.poisoned = True
                raise

    @torch.inference_mode()
    def forward(self, index, value):
        self._healthy()
        if not self.prepared or self.replay_stream is None:
            raise RuntimeError("MLP graphs need startup preparation and an exclusive bound stream")
        if (type(index) is not int or not 0 <= index < len(self.modules)
                or not isinstance(value, torch.Tensor) or value.ndim != 3 or value.shape[0] != 1
                or value.shape[1] not in self.rows or value.shape[2] != self.hidden
                or value.dtype != torch.bfloat16 or value.device != self.device or value.requires_grad):
            raise ValueError("MLP graph input differs from its admitted layer/row/precision/device")
        rows = value.shape[1]
        if self.shared_pool:
            if index != self._next_layer or (self._row_in_progress is not None and self._row_in_progress != rows):
                raise RuntimeError("shared MLP graph pools require complete fixed-order layer replay")
        entry = self.entries[rows, index]
        try:
            self.inputs[rows].copy_(value)
            entry.graph.replay()
            output = entry.output.clone() if self.owned_output else entry.output
        except BaseException:
            self.poisoned = True
            raise
        if self.shared_pool:
            self._next_layer = (index+1) % len(self.modules)
            self._row_in_progress = rows if self._next_layer else None
        return output

    def logical_buffer_bytes(self):
        """Logical IO owners, not allocator/private-pool/workspace residency."""
        tensors = [*self.inputs.values(), *(entry.output for entry in self.entries.values())]
        return sum(value.numel()*value.element_size() for value in tensors)

    def close(self):
        if self.closed:
            return
        if self.terminal:
            raise RuntimeError("MLP capture/retirement state is unknown; retain owners and terminate the process")
        errors = []
        for stream in (self.capture_stream, self.replay_stream):
            if stream is not None:
                try:
                    stream.synchronize()
                except BaseException as error:
                    errors.append(error)
        if errors:
            self.poisoned = self.terminal = True
            raise_failures(None, errors)
        graphs = [entry.graph for entry in self.entries.values()]
        if self.pending_graph is not None:
            graphs.append(self.pending_graph)
        for graph in graphs:
            try:
                graph.reset()
            except BaseException as error:
                errors.append(error)
        if errors:
            self.poisoned = self.terminal = True
            raise_failures(None, errors)
        self.entries.clear()
        self.inputs.clear()
        self.pools.clear()
        self.pending_graph = self.pending_context = self.ready = None
        self.capture_stream = self.replay_stream = None
        self.modules = ()
        self.closed = True


class GraphedMLP(Gemma4TextMLP):
    def __init__(self, native, policy, owner, index):
        torch.nn.Module.__init__(self)
        for name, module in native.named_children():
            self.add_module(name, module)
        self.config, self.hidden_size, self.intermediate_size = native.config, native.hidden_size, native.intermediate_size
        self.policy, self.owner, self.index = policy, owner, index
        self.train(native.training)

    def forward(self, value):
        if self.policy.prompt:
            return super().forward(value)
        return self.owner.forward(self.index, value)


def install_mlp_graphs(language_model, policy, *, enabled=False, rows=tuple(range(1, 9)),
                       shared_pool=False, owned_output=False):
    """Disabled leaves native modules unchanged; enabled prepares before publication.

    Caller binds the returned owner to its existing exclusive engine stream
    before any decode. Prompt/native_reference remains the original HF MLP.
    """
    if type(enabled) is not bool:
        raise ValueError("MLP graph selection must be boolean")
    if not enabled:
        return None
    modules = tuple(layer.mlp for layer in language_model.layers)
    owner = NativeMLPGraphs(modules, rows=rows, shared_pool=shared_pool, owned_output=owned_output)
    owner.prepare()
    try:
        wrappers = tuple(GraphedMLP(module, policy, owner, index) for index, module in enumerate(modules))
        for layer, wrapper in zip(language_model.layers, wrappers):
            layer.mlp = wrapper
    except BaseException as primary:
        errors = []
        for layer, native in zip(language_model.layers, modules):
            try:
                layer.mlp = native
            except BaseException as error:
                errors.append(error)
        try:
            owner.close()
        except BaseException as error:
            errors.append(error)
        raise_failures(primary, errors)
    return owner
