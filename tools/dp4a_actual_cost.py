"""Finite actual-bank routed-only cost diagnostic; never Engine/P3/quality qualification."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch

from tensorfold.cuda.sm70.dp4a_experts import Prototype, Graph, extension, _signature
from tensorfold.cuda.sm70.execution_policy import _family_runtime
from tools.ncu_experts_actual import fixture
from tools.v100_backend_probe import finish_failures

ROWS = (1, 8, 32)
SAMPLES, REPLAYS, UNROLL, WARMUP = 3, 8, 4, 8


class RoutedBurst(Prototype):
    """Original native sequence, excluding ONLY shared assembly; final veto retained."""
    def routed_once(self):
        if self.exact_only:
            raise RuntimeError('actual bank is not eligible DP4A; fallback is not a timing result')
        module = extension()
        module.quantize(self.x, self.qx, 0, self.status)
        module.project(self.qx, self.picks, self.bank.gate.data, self.bank.gate.ggml_type, False, self.gate, self.status)
        module.project(self.qx, self.picks, self.bank.up.data, self.bank.up.ggml_type, False, self.up, self.status)
        module.activate(self.gate, self.up, self.hidden)
        module.quantize(self.hidden.view(-1, 80), self.qh, self.bank.down.column_offset, self.status)
        module.project(self.qh, self.picks, self.bank.down.data, 20, True, self.out, self.status)
        module.mask(self.out, self.status)
        return self.out

    def _run(self):
        for _ in range(UNROLL):
            self.routed_once()
        return self.out


class PackedBurst:
    """Adapt original Graph lifetime to the stateless packed reference, without DP scratch."""
    def __init__(self, reference, out):
        self.reference, self.out = reference, out
        self.stream, self.graph_owner = reference.stream, None
        self.failed, self.closed, self.retiring = None, False, False
        self.binding = _signature(out)
        self.anchors = (reference.x, reference.picks, reference.bank, out)

    def _validate(self):
        if self.closed or self.retiring or self.failed is not None:
            raise RuntimeError('packed diagnostic owner is not reusable') from self.failed
        self.reference._validate()
        if _signature(self.out) != self.binding:
            raise RuntimeError('packed diagnostic output storage changed')

    def run(self):
        self._validate()
        if torch.cuda.is_current_stream_capturing() and not (
                self.graph_owner is not None and self.graph_owner.state == 'capturing'):
            raise RuntimeError('packed diagnostic capture requires its Graph borrower')
        for _ in range(UNROLL):
            self.reference.bank.forward_blocks2w(self.reference.x, self.reference.picks, self.out)
        return self.out

    def capture(self):
        self._validate()
        if self.graph_owner is not None or self.stream.cuda_stream == 0:
            raise RuntimeError('packed diagnostic requires one graph on a nondefault stream')
        graph = Graph(self)
        self.graph_owner = graph
        graph.prepare()
        return graph

    def close(self):
        if self.closed:
            return
        if self.graph_owner is not None:
            raise RuntimeError('packed graph borrower must retire first')
        self.retiring = True
        self.stream.synchronize()
        self.closed = True
        self.reference = self.out = self.anchors = None


def bits(tensor):
    return tensor.detach().cpu().contiguous().view(torch.uint8)


def digest(tensor):
    return hashlib.sha256(bits(tensor).numpy().tobytes()).hexdigest()


def timed(operation, stream):
    """GPU timeline of8 fixed-data bursts; includes enqueue gaps, not a model benchmark."""
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record(stream)
    for _ in range(REPLAYS):
        operation()
    end.record(stream)
    end.synchronize()
    elapsed = float(start.elapsed_time(end))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError("CUDA event duration is not finite and positive")
    return elapsed


def actual_case(bank, rows):
    errors = [None] * 5  # All retirement outcomes precede accepted work.
    owners = [None, None]  # Packed owner first, then DP owner; retained on uncertain retirement.
    native_roots = [None, None, False]  # Retained by this frame if outcome compaction itself exhausts.
    owner = packed = dp_graph = packed_graph = None
    primary, result = None, None
    original = torch.cuda.current_stream()
    stream = torch.cuda.Stream()
    try:
        stream.wait_stream(original)  # Weight/input ownership handoff precedes the nondefault stream.
        torch.cuda.set_stream(stream)
        generator = torch.Generator(device='cpu').manual_seed(19201 + rows)
        x = (torch.randn((rows, 2560), generator=generator) * .1).half().cuda()
        picks = torch.stack([torch.randperm(512, generator=generator)[:10] for _ in range(rows)]).to(torch.int32).cuda()
        if picks.shape != (rows, 10) or int(picks.min()) < 0 or int(picks.max()) >= 512:
            raise ValueError('routed-only diagnostic cannot contain the shared/invalid slot')
        output = torch.empty((rows, 10, 2560), device='cuda', dtype=torch.float16)
        owner = object.__new__(RoutedBurst)
        owners[1] = owner
        owner.__init__(bank, x, picks, output)
        if owner.exact_only:
            raise RuntimeError('actual retained formats selected exact fallback; DP4A timing prohibited')
        packed = object.__new__(PackedBurst)
        owners[0] = packed
        packed.__init__(owner, torch.empty_like(output))
        owner.run()
        packed.run()
        stream.synchronize()
        status_before = owner.status.cpu().tolist()
        if status_before != [0] or not torch.isfinite(output).all() or not torch.isfinite(packed.out).all():
            raise RuntimeError('actual operator has a failed status/nonfinite result', status_before)
        expected_dp, expected_packed = bits(output), bits(packed.out)
        # Independent original quantizer on actual fixed inputs/hidden values.
        from dp4a_profile_oracle import quantize
        qx, sx = quantize(x.cpu().numpy())
        qh, sh = quantize(owner.hidden.cpu().numpy().reshape(rows * 10, 80), bank.down.column_offset)
        if sx or sh or not torch.equal(bits(owner.qx), torch.from_numpy(qx).view(torch.uint8)) or not torch.equal(bits(owner.qh), torch.from_numpy(qh).view(torch.uint8)):
            raise RuntimeError('actual activation packing differs from independent profile quantizer')
        # Original Graph owns accepted capture/end/retirement for both operations.
        dp_graph = owner.capture()
        packed_graph = packed.capture()
        raw = {mode: {name: [] for name in ('packed', 'dp4a')} for mode in ('event', 'graph')}
        for mode in ('event', 'graph'):
            operations = {'packed': packed.run, 'dp4a': owner.run} if mode == 'event' else {
                'packed': packed_graph.graph.replay, 'dp4a': dp_graph.graph.replay}
            for operation in operations.values():
                for _ in range(WARMUP):
                    operation()
            stream.synchronize()
            for sample in range(SAMPLES):
                # Fixed data/immutable bindings over each exclusive batch. Raw
                # graph replay excludes repeated Python guard walks equally.
                owner._validate()
                packed._validate()
                order = ('packed', 'dp4a') if sample % 2 == 0 else ('dp4a', 'packed')
                for name in order:
                    raw[mode][name].append(timed(operations[name], stream))
        stream.synchronize()
        status_after = owner.status.cpu().tolist()
        if status_after != [0] or not torch.equal(bits(output), expected_dp) or not torch.equal(bits(packed.out), expected_packed):
            raise RuntimeError('timing changed output bits or failed the monotonic status latch', status_after)
        actual, reference = output.float(), packed.out.float()
        delta = actual - reference
        norm = float(torch.linalg.vector_norm(reference))
        result = {'rows': rows, 'slots': 10, 'picks_dtype': str(picks.dtype), 'picks_sha256': digest(picks),
                  'input_sha256': digest(x), 'dp4a_output_sha256': digest(output), 'packed_output_sha256': digest(packed.out),
                  'status': owner.status.cpu().tolist(), 'scratch_bytes': sum(t.numel() * t.element_size() for t in owner.scratch),
                  'raw_batch_ms': raw, 'calls_per_batch': REPLAYS * UNROLL, 'samples': SAMPLES,
                  'error_vs_packed': {'max_abs': float(delta.abs().max()), 'rms': float(delta.square().mean().sqrt()),
                    'relative_l2': float(torch.linalg.vector_norm(delta)) / norm if norm else None,
                    'output_half_mismatches': int((output.view(torch.int16) != packed.out.view(torch.int16)).sum())},
                  'bits_repeatable_within_each_path': True, 'exact_only': owner.exact_only,
                  'scope': 'routed-only includesquantize/project/activate/down/finalmask; no sharedassembly; changednumericprofile/errornotqualityacceptance'}
    except BaseException as error:
        primary = error
        native_roots[0] = BaseException.__cause__.__get__(error)
        native_roots[1] = BaseException.__context__.__get__(error)
        native_roots[2] = BaseException.__suppress_context__.__get__(error)
    # Observe the published borrower even if prepare raised before capture()
    # returned. Graph.close retains uncertain owners; owner.close then refuses
    # dependent release while that borrower remains.
    for index in (0, 1):
        value = owners[index]
        graph = None if value is None else getattr(value, 'graph_owner', None)
        if graph is not None:
            try:
                graph.close()
            except BaseException as error:
                errors[index] = error
    for index in (0, 1):
        value = owners[index]
        if value is not None and getattr(value, 'stream', None) is not None:
            try:
                value.close()
            except BaseException as error:
                errors[index + 2] = error
    try:
        torch.cuda.set_stream(original)
    except BaseException as error:
        errors[4] = error
    # native_roots and owners remain traceback-owned even if the borrowed
    # outcome helper cannot compact its fixed slots and replaces the cause.
    finish_failures(primary, errors)
    return result


def resource_observation():
    from tensorfold.cuda.capacity import host_memory_snapshot
    return {'cuda_free_total_bytes': list(torch.cuda.mem_get_info()),
            'torch_allocated_bytes': torch.cuda.memory_allocated(),
            'torch_reserved_bytes': torch.cuda.memory_reserved(),
            'torch_peak_allocated_bytes_so_far': torch.cuda.max_memory_allocated(),
            'host_advisory': host_memory_snapshot()}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    _family_runtime(torch)  # Observe pinned flags; no precision setters.
    torch.set_num_threads(1)
    resources_before_loader = resource_observation()
    loader = fixture(args.source)
    loader.MODEL = args.model
    bank, spans = loader.make(0, 8)
    retained = []
    for span, tensor in zip(spans, (bank.gate, bank.up, bank.down), strict=True):
        retained.append({**span, 'ggml_type': tensor.ggml_type, 'column_offset': tensor.column_offset,
                         'retained_sha256': digest(tensor.data), 'shape': list(tensor.data.shape)})
    if ((bank.gate.ggml_type not in (18, 22)) or bank.up.ggml_type not in (18, 22)
            or bank.down.ggml_type != 20 or bank.down.column_offset != 16 or bank.count != 513):
        raise ValueError('actual retained bank is outside declared18/22+20/crop16 diagnostic',
                         (bank.gate.ggml_type, bank.up.ggml_type, bank.down.ggml_type, bank.down.column_offset, bank.count))
    resources_after_loader = resource_observation()
    bank_bytes = sum(t.numel() * t.element_size() for t in
                     (bank.gate.data, bank.up.data, bank.down.data, *(part.weight for part in bank.shared)))
    down_raw_bytes = 512 * 2560 * loader.checkpoint().tensors['blk.0.ffn_down_exps.weight'].row_bytes
    cases = [actual_case(bank, rows) for rows in ROWS]
    for tensor, record in zip((bank.gate, bank.up, bank.down), retained, strict=True):
        if digest(tensor.data) != record['retained_sha256']:
            raise RuntimeError('actual packed weights changed during diagnostic')
    _family_runtime(torch)
    module = extension()
    report = {'schema': 'tensorfold.dp4a-actual-routed-cost-v1', 'application_enrollment': False,
              'source_spans': retained, 'native_module': module.__file__,
              'native_module_sha256': hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest(),
              'resource_observations': {'before_loader': resources_before_loader,
                'after_loader': resources_after_loader, 'after_cases': resource_observation(),
                'retained_bank_tensor_bytes': bank_bytes, 'loader_down_raw_bytes': down_raw_bytes,
                'loader_down_concatenation_arrays_bytes': 2 * down_raw_bytes,
                'scope': 'tensor extents and observed Torch/CUDA/host-cgroup remainder; loader concatenation excludes runtime/cache overhead; no reservation, physical peak bound or Engine admission claim'},
              'rows': list(ROWS), 'samples': SAMPLES, 'replays': REPLAYS, 'unroll': UNROLL, 'warmup': WARMUP,
              'input_policy': 'seededactual512bank/rank3shardof8/fixedwarmdata/noL2flush/nohostclockchanges',
              'timing_scope': 'CUDAeventtimeline includesenqueuegaps; graphmodeprevalidatesimmutablebindings thenusesrawGraphreplay; noCPUstatusread/resetinsidebatches',
              'cases': cases, 'scope': 'componentdiagnosticonly; noState/NCCL/model/P3/outputquality/24admissionclaim'}
    args.out.mkdir(exist_ok=False)
    (args.out/'result.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'cases': len(cases), 'scope': report['scope']}), flush=True)


if __name__ == '__main__':
    main()
