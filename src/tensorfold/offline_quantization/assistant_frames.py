"""Unconnected, bounded target-conditioned assistant calibration primitives.

A loader/coordinator binds immutable provenance to target.calibration_source.
One exclusive actor lends one frame at a time; no tensor-frame DataLoader or
large persistent KV archive is provided. The consumer completes forward/loss/
backward/hooks INSIDE the frame lease. Retirement failure is terminal and keeps
borrowed owners alive. ModelOpt is deliberately not imported here.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import threading
import weakref
from types import MappingProxyType

from tensorfold.cleanup import raise_failures
from tensorfold.families.gemma4.assistant_metadata import AssistantConfig

CORE22 = tuple(sorted(name.removesuffix('.weight') for name, shape in AssistantConfig().shapes().items()
                      if len(shape) == 2 and not name.startswith('masked_embedding.')
                      and name != 'model.embed_tokens.weight'))
INITIAL20 = tuple(name for name in CORE22 if name not in ('pre_projection', 'post_projection'))
PHASES = ('calibration', 'sensitivity', 'qad', 'holdout')


def _digest(value, name):
    if type(value) is not str or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must be a lowercase SHA256 digest')


def _revision(value, name):
    if type(value) is not str or len(value) not in (40, 64) or any(c not in '0123456789abcdef' for c in value):
        raise ValueError(f'{name} must identify an immutable source revision')


def config_digest(raw):
    """Canonical RUNTIME raw-config digest, distinct from source-file byte hash."""
    return hashlib.sha256(json.dumps(raw, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class CalibrationSource:
    kind: str
    main_revision: str
    config_sha256: str
    runtime_sha256: str
    tokenizer_sha256: str
    template_sha256: str
    processor_sha256: str
    quant_manifest_sha256: str | None = None
    payloads_sha256: str | None = None

    def __post_init__(self):
        if self.kind not in ('bf16_reference', 'quantized_verifier', 'synthetic_fixture'):
            raise ValueError('calibration source kind must distinguish BF16, quantized target or synthetic fixture')
        _revision(self.main_revision, 'main_revision')
        for name in ('config_sha256', 'runtime_sha256', 'tokenizer_sha256', 'template_sha256', 'processor_sha256'):
            _digest(getattr(self, name), name)
        if self.kind != 'synthetic_fixture':
            _digest(self.payloads_sha256, 'consumed payloads_sha256')
        elif self.payloads_sha256 is not None:
            _digest(self.payloads_sha256, 'payloads_sha256')
        if self.kind == 'quantized_verifier':
            _digest(self.quant_manifest_sha256, 'quant_manifest_sha256')
        elif self.quant_manifest_sha256 is not None:
            raise ValueError('nonquantized calibration source must not claim a quantization manifest')


@dataclass(frozen=True)
class AssistantStartingSource:
    """Loader/coordinator-validated starting assistant, not trained output proof."""
    revision: str
    config_sha256: str
    payloads_sha256: str

    def __post_init__(self):
        _revision(self.revision, 'assistant starting revision')
        _digest(self.config_sha256, 'assistant starting config_sha256')
        _digest(self.payloads_sha256, 'assistant consumed payloads_sha256')


@dataclass(frozen=True)
class PhaseQuota:
    max_frames: int
    max_forward_steps: int
    gradients: bool = False

    def __post_init__(self):
        if any(type(v) is not int or not 0 < v <= 2**31 - 1 for v in (self.max_frames, self.max_forward_steps)):
            raise ValueError('calibration quotas must be positive signed32 counts')
        if type(self.gradients) is not bool:
            raise ValueError('calibration gradient policy must be boolean')


@dataclass(frozen=True)
class FrameSample:
    phase: str
    input_sha256: str
    seed: int
    trajectory: str

    def __post_init__(self):
        if self.phase not in PHASES or self.trajectory not in ('teacher_forced', 'assistant_rollout'):
            raise ValueError('calibration phase/trajectory must be explicit')
        _digest(self.input_sha256, 'input_sha256')
        if type(self.seed) is not int or not 0 <= self.seed <= 2**64 - 1:
            raise ValueError('calibration seed must be unsigned64')


class AssistantCalibrationActor:
    """Unconnected producer requiring an actual loader-bound source identity.

    exclusive() owns the model coordinator AND correct execution stream. retire()
    synchronizes that stream directly; do not use Engine.drain from its active
    request callback. accepted_tokens() belongs to that same coordinator and
    describes its complete committed transcript, including actual media IDs.
    TARGET source/weights/config are immutable throughout this actor session.
    The assistant remains the explicit phase consumer's calibration/training
    student; its starting revision is lineage, not a final exported-weight hash.
    selected_modules is an observation inventory, not quantizer configuration:
    the provider must enforce the saved recipe's BF16 exceptions separately.
    """
    def __init__(self, target, state, *, exclusive, retire, accepted_tokens, quotas,
                 assistant_revision, assistant_config_sha256, recipe_sha256,
                 calibration_data_sha256, heldout_data_sha256, max_borrowed_bytes,
                 selected_modules=INITIAL20, max_recurrent_steps=7):
        source = getattr(target, 'calibration_source', None)
        if not isinstance(source, CalibrationSource):
            raise ValueError('target loader has not bound calibration_source; real quantized calibration is unavailable')
        if state.owner is not target:
            raise ValueError('calibration state belongs to another target')
        _revision(assistant_revision, 'assistant_revision')
        for name, value in (('assistant_config_sha256', assistant_config_sha256), ('recipe_sha256', recipe_sha256),
                            ('calibration_data_sha256', calibration_data_sha256), ('heldout_data_sha256', heldout_data_sha256)):
            _digest(value, name)
        if calibration_data_sha256 == heldout_data_sha256:
            raise ValueError('calibration and holdout manifests must be separately identified')
        if (not quotas or any(name not in PHASES or not isinstance(q, PhaseQuota) for name, q in quotas.items())
                or ('holdout' in quotas and quotas['holdout'].gradients)):
            raise ValueError('bounded per-phase quotas are required; holdout cannot train')
        if not isinstance(selected_modules, (tuple, list)) or not 0 < len(selected_modules) <= len(CORE22):
            raise ValueError('selected assistant modules must be a bounded list/tuple')
        selected = tuple(selected_modules)
        if len(set(selected)) != len(selected) or any(name not in CORE22 for name in selected):
            raise ValueError('selected assistant modules must be an explicit subset of the historical22 core paths')
        if type(max_borrowed_bytes) is not int or max_borrowed_bytes <= 0:
            raise ValueError('explicit positive borrowed-frame byte budget required')
        if type(max_recurrent_steps) is not int or not 1 <= max_recurrent_steps <= 7:
            raise ValueError('assistant calibration recurrence must be bounded1..7')
        if not all(callable(fn) for fn in (exclusive, retire, accepted_tokens)):
            raise ValueError('exclusive actor, retirement and transcript callbacks are required')
        self.target, self.state, self.source = target, state, source
        self._exclusive, self._retire, self._accepted_tokens = exclusive, retire, accepted_tokens
        self.quotas, self.selected = MappingProxyType(dict(quotas)), selected
        self.max_borrowed_bytes, self.max_recurrent_steps = max_borrowed_bytes, max_recurrent_steps
        self._thread = self._active = None
        self._assistant_ref = self._assistant_binding = None
        self._failed = False
        self._seen_inputs = {'train': set(), 'holdout': set()}
        self._counts = {phase: {'frames': 0, 'forward_steps': 0, 'observations': {name: 0 for name in selected}}
                       for phase in quotas}
        self._records = []  # Bounded JSON metadata only, never hidden/KV tensors.
        self.provenance = MappingProxyType(dict(assistant_revision=assistant_revision, assistant_config_sha256=assistant_config_sha256,
                               recipe_sha256=recipe_sha256, calibration_data_sha256=calibration_data_sha256,
                               heldout_data_sha256=heldout_data_sha256))
        self._check_source()

    @classmethod
    def from_engine(cls, engine, **kwargs):
        """Reuse the existing exclusive lane; still requires loader source binding."""
        @contextmanager
        def exclusive():
            import torch
            engine.drain()
            with engine._lane(False) as stream, torch.cuda.stream(stream):
                yield
        return cls(engine.target, engine._state, exclusive=exclusive, retire=engine._stream.synchronize, **kwargs)

    def _check_source(self):
        if (getattr(self.target, 'calibration_source', None) is not self.source
                or config_digest(self.target.raw_config) != self.source.config_sha256):
            raise RuntimeError('calibration source/config identity changed')
        raw = self.target.raw_config
        declared = any(block.get('quantization_config') or block.get('quantization')
                       for block in (raw, raw.get('text_config') or {}))
        if self.source.kind != 'synthetic_fixture' and getattr(self.target, 'calibration_payloads_sha256', None) != self.source.payloads_sha256:
            raise ValueError('real calibration requires consumed target payload provenance from the loader')
        if self.source.kind == 'quantized_verifier':
            if (not declared or getattr(self.target, 'quantization_manifest_sha256', None)
                    != self.source.quant_manifest_sha256):
                raise ValueError('quantized calibration requires consumed-loader manifest and quantized runtime metadata')
        elif self.source.kind == 'bf16_reference' and declared:
            raise ValueError('quantized runtime cannot be labeled as the original BF16 reference')

    def _require_actor(self):
        if self._failed or self._thread != threading.get_ident():
            raise RuntimeError('calibration requires its healthy active exclusive actor/thread')
        self._check_source()

    @contextmanager
    def session(self):
        if self._thread is not None or self._failed:
            raise RuntimeError('calibration actor is already active or failed')
        scope = self._exclusive()
        try:
            scope.__enter__()
        except BaseException:
            self._failed = True
            # Failed acquisition is the context owner's cleanup responsibility;
            # no invented __exit__ or retry under unknown ownership.
            raise
        self._thread = threading.get_ident()
        primary, errors = None, []
        try:
            yield self
            if self._active is not None:
                raise RuntimeError('calibration frame outlived its actor scope')
        except BaseException as error:
            primary, self._failed = error, True
        if self._active is not None and not self._active.retirement_failed:
            errors.extend(self._release_frame(self._active))
        try:
            scope.__exit__(None if primary is None else type(primary), primary,
                           None if primary is None else primary.__traceback__)
        except BaseException as error:
            errors.append(error)
            self._failed = True
        self._thread = None
        raise_failures(primary, errors)

    @contextmanager
    def stream(self, requests):
        """One-pass synchronous frame stream over small (anchor, sample) descriptors.

        The descriptor producer may prepare the next committed target prefix
        ONLY between leases. No asynchronous DataLoader/prefetch is supported.
        Early consumer exit closes the current lease before another sample can
        start; incomplete planned work remains visible in phase counters.
        """
        self._require_actor()
        def iterate():
            for anchor, sample in requests:
                with self.frame(anchor, sample) as borrowed:
                    yield borrowed
        iterator = iterate()
        primary, errors = None, []
        try:
            yield iterator
        except BaseException as error:
            primary = error
        try:
            iterator.close()
        except BaseException as error:
            errors.append(error)
        raise_failures(primary, errors)

    @contextmanager
    def frame(self, anchor, sample):
        self._require_actor()
        if self._active is not None or not isinstance(sample, FrameSample) or sample.phase not in self.quotas:
            raise ValueError('one declared calibration frame may be active at a time')
        count = self._counts[sample.phase]
        if count['frames'] >= self.quotas[sample.phase].max_frames:
            raise RuntimeError('calibration phase frame quota exhausted')
        split = 'holdout' if sample.phase == 'holdout' else 'train'
        if sample.input_sha256 in self._seen_inputs['train' if split == 'holdout' else 'holdout']:
            raise ValueError('same input cannot cross calibration/train and holdout splits')
        held = self.target.assistant_state(self.state)
        borrowed = _Frame(self, held, anchor, sample)
        self._active = borrowed
        count['frames'] += 1
        self._seen_inputs[split].add(sample.input_sha256)
        primary, errors = None, []
        try:
            yield borrowed
        except BaseException as error:
            primary = error
            if not isinstance(error, GeneratorExit):
                self._failed = True
        if not borrowed._closed and not borrowed.retirement_failed:
            errors.extend(self._release_frame(borrowed))
        if isinstance(primary, GeneratorExit) and errors:
            # generator.close swallows GeneratorExit, so it MUST NOT swallow a
            # failed device retirement carried merely as its secondary cause.
            raise_failures(None, errors)
        raise_failures(primary, errors)

    def _release_frame(self, borrowed):
        errors = []
        try:
            self._retire()
        except BaseException as error:
            errors.append(error)
            self._failed = True
            self.state.poisoned = True
            # Keep ALL borrowed owners in this failed actor. The outer owner
            # terminates work/process; neither retry nor target reuse is safe.
            borrowed.retirement_failed = True
        if not errors:
            borrowed._clear()
            self._active = None
        return errors

    def receipt(self):
        if self._thread is not None and self._thread != threading.get_ident():
            raise RuntimeError('active calibration receipt belongs to its actor thread')
        # A metadata snapshot, not writable aliases to quota/coverage authority.
        result = {'schema': 'tensorfold.conditioned_assistant_frames.v1', 'source': asdict(self.source),
                  'provenance': dict(self.provenance), 'selected_modules': list(self.selected),
                  'historical_core_inventory': list(CORE22),
                  'bounds': {'max_borrowed_bytes': self.max_borrowed_bytes,
                             'max_recurrent_steps': self.max_recurrent_steps, 'max_active_frames': 1},
                  'pre_post_selected': any(name in self.selected for name in ('pre_projection', 'post_projection')),
                  'quotas': {p: asdict(q) for p, q in self.quotas.items()}, 'counts': self._counts,
                  'records': self._records, 'failed': self._failed,
                  'assistant_starting_source': None if self._assistant_binding is None else asdict(self._assistant_binding),
                  'assistant_source_status': 'synthetic-unverified' if self.source.kind == 'synthetic_fixture'
                      else ('loader-starting-binding-validated' if self._assistant_binding is not None else 'not-yet-validated'),
                  'evidence': 'synthetic-fixture-only' if self.source.kind == 'synthetic_fixture' else 'conditioned-forward-coverage-only',
                  'provider_calibration_status': 'not-invoked-or-certified-by-this-adapter',
                  'quantized_model_quality_or_capacity_verified': False}
        return json.loads(json.dumps(result, allow_nan=False))


class _Frame:
    def __init__(self, actor, held, anchor, sample):
        import torch
        self.actor, self.sample = actor, sample
        if (not isinstance(held, dict) or set(held) != {'position', 'last_hidden_state', 'attention_mask', 'shared_kv_states'}
                or not isinstance(held['last_hidden_state'], torch.Tensor)):
            raise ValueError('calibration target feature interface differs')
        self._held, self._hidden = held, held['last_hidden_state'].detach()
        self._latest_hidden = None
        self._closed = self.retirement_failed = False
        self.step, self.anchor = 0, anchor
        cfg = actor.target.config
        vocab, hidden, kv, local_dim, full_dim, window = (cfg.vocab, cfg.hidden, cfg.kv_heads,
                                                       cfg.head_dim, cfg.global_head_dim, cfg.window)
        position = held['position']
        if type(position) is not int or position != actor.state.position or not 0 < position < actor.state.limit - 1:
            raise ValueError('calibration needs a committed target prefix with anchor capacity')
        if type(anchor) is not int or not 0 <= anchor < vocab:
            raise ValueError('calibration anchor is outside the actual target vocabulary')
        expected = {'sliding_attention': (1, kv, min(position, window), local_dim),
                    'full_attention': (1, kv, position, full_dim)}
        if set(held['shared_kv_states']) != set(expected):
            raise ValueError('calibration donor layer types differ')
        values = [held['last_hidden_state']]
        if tuple(held['last_hidden_state'].shape) != (1, 1, hidden):
            raise ValueError('calibration accepted target hidden shape differs')
        for kind, shape in expected.items():
            pair = held['shared_kv_states'][kind]
            if len(pair) != 2 or any(not isinstance(t, torch.Tensor) or tuple(t.shape) != shape for t in pair):
                raise ValueError('calibration donor KV geometry differs')
            values.extend(pair)
        if any(t.dtype != torch.bfloat16 or t.device != actor.target.device for t in values):
            raise ValueError('calibration target features must use the actual BF16 device contract')
        mask = held['attention_mask']
        if mask is not None and (not isinstance(mask, torch.Tensor) or tuple(mask.shape) != (1, position) or mask.device != actor.target.device):
            raise ValueError('calibration target validity mask differs')
        if actor.source.kind != 'synthetic_fixture' and (hidden, kv, local_dim, full_dim, window, vocab) != (
                2560, 2, 256, 512, 512, 262144):
            raise ValueError('real calibration requires the actual native E4B source geometry')
        borrowed_bytes = sum(t.numel()*t.element_size() for t in values)
        borrowed_bytes += 0 if mask is None else mask.numel()*mask.element_size()
        if borrowed_bytes > actor.max_borrowed_bytes:
            raise ValueError('calibration borrowed frame byte budget exceeded')
        if actor.quotas[sample.phase].gradients and any(torch.is_inference(t) for t in values + ([] if mask is None else [mask])):
            raise ValueError('gradient calibration needs target state prepared under no_grad, not inference_mode; no full-KV clone is hidden')
        self.position, self.borrowed_bytes = position, borrowed_bytes
        self._max_steps = min(actor.max_recurrent_steps, actor.state.limit - position - 1)
        self._stamp = self._state_stamp(held)
        digest = hashlib.sha256()
        tokens = actor._accepted_tokens()
        if len(tokens) != position:
            raise ValueError('accepted transcript does not match committed target position')
        for token in tokens:
            if type(token) is not int or not 0 <= token < vocab:
                raise ValueError('accepted transcript contains an invalid target token')
            digest.update(token.to_bytes(4, 'little'))
        self.token_sha256 = digest.hexdigest()
        self._positions = torch.tensor([[position]], dtype=torch.long, device=actor.target.device)

    @staticmethod
    def _state_stamp(held):
        values = [held['last_hidden_state'], *(t for pair in held['shared_kv_states'].values() for t in pair)]
        mask = held['attention_mask']
        if mask is not None:
            values.append(mask)
        return held['position'], tuple((t.data_ptr(), tuple(t.shape), tuple(t.stride()), t.storage_offset()) for t in values)

    def _check(self):
        if self._closed or self.retirement_failed or self.actor is None:
            raise RuntimeError('calibration frame has expired or failed')
        self.actor._require_actor()
        if self.actor._active is not self:
            raise RuntimeError('calibration frame is not the current actor lease')
        held = self.actor.target.assistant_state(self.actor.state)
        if self._state_stamp(held) != self._stamp:
            raise RuntimeError('target mutated or restored while calibration borrowed its state')

    def advance_recurrent_token(self, token, *, origin):
        self._check()
        expected = 'target_teacher' if self.sample.trajectory == 'teacher_forced' else 'assistant_draw'
        if origin != expected or self._latest_hidden is None:
            raise ValueError('recurrent calibration needs an actual prior assistant output and explicit token origin')
        if type(token) is not int or not 0 <= token < self.actor.target.config.vocab:
            raise ValueError('recurrent calibration token is outside the vocabulary')
        if self.step + 1 >= self._max_steps:
            raise RuntimeError('calibration recurrent-step quota exhausted')
        self.anchor, self._hidden = token, self._latest_hidden
        self._latest_hidden = None
        self.step += 1

    def _clear(self):
        self._closed = True
        self._held = self._hidden = self._latest_hidden = self._positions = None
        self.actor = None


def modelopt_forward_step(model, frame):
    """ModelOpt-compatible forward_step(model, data) with an ACTIVE frame lease.

    The surrounding consumer must complete its loss/backward/hooks before frame
    exit. Do not put frames in an asynchronous/prefetching DataLoader. This
    function neither imports ModelOpt nor claims provider calibration completed.
    """
    import torch
    if not isinstance(frame, _Frame):
        raise ValueError('assistant forward requires an active conditioned frame')
    frame._check()
    actor, sample = frame.actor, frame.sample
    quota, counts = actor.quotas[sample.phase], actor._counts[sample.phase]
    if counts['forward_steps'] >= quota.max_forward_steps:
        raise RuntimeError('calibration phase forward-step quota exhausted')
    if quota.gradients and (not torch.is_grad_enabled() or torch.is_inference_mode_enabled()):
        raise ValueError('gradient calibration needs an autograd-enabled actor scope')
    if actor.source.kind != 'synthetic_fixture':
        from transformers import Gemma4AssistantForCausalLM
        if not isinstance(model, Gemma4AssistantForCausalLM):
            raise ValueError('real conditioned calibration requires the actual native assistant class')
        binding = getattr(model, 'calibration_assistant_source', None)
        if (not isinstance(binding, AssistantStartingSource)
                or binding.revision != actor.provenance['assistant_revision']
                or binding.config_sha256 != actor.provenance['assistant_config_sha256']):
            raise ValueError('assistant starting source must be independently validated and bound by its loader/coordinator')
        if actor._assistant_ref is not None and (actor._assistant_ref() is not model or actor._assistant_binding is not binding):
            raise RuntimeError('calibration assistant identity changed within its actor session')
        actor._assistant_ref, actor._assistant_binding = weakref.ref(model), binding
    modules = dict(model.named_modules())
    if any(name not in modules for name in actor.selected):
        raise ValueError('selected assistant calibration module is absent')
    with torch.no_grad():
        ids = torch.tensor([[frame.anchor]], device=actor.target.device, dtype=torch.long)
        embedding = actor.target.model.get_input_embeddings()(ids).detach()
    hidden = frame._hidden
    if tuple(embedding.shape) != tuple(hidden.shape) or embedding.dtype != torch.bfloat16:
        raise ValueError('public scaled target embedding does not match accepted/projected hidden')
    inputs = torch.cat((embedding, hidden), dim=-1)
    if not bool(torch.isfinite(inputs).all()):
        raise ValueError('calibration concatenated target inputs contain nonfinite values')
    observed, hooks, primary, errors, output = {}, [], None, [], None
    def observe(name):
        def hook(module, args):
            value = args[0]
            if not isinstance(value, torch.Tensor):
                raise ValueError('selected assistant projection received no tensor activation')
            record = observed.setdefault(name, {'calls': 0, 'shape': list(value.shape), 'dtype': str(value.dtype)})
            record['calls'] += 1
        return hook
    counts['forward_steps'] += 1
    try:
        for name in actor.selected:
            hooks.append(modules[name].register_forward_pre_hook(observe(name)))
        scope = nullcontext() if quota.gradients else torch.no_grad()
        with scope:
            output = model(inputs_embeds=inputs, position_ids=frame._positions,
                           shared_kv_states={kind: tuple(t.detach() for t in pair)
                                             for kind, pair in frame._held['shared_kv_states'].items()},
                           attention_mask=frame._held['attention_mask'], use_cache=False, return_dict=True)
        frame._check()
        if set(observed) != set(actor.selected) or any(v['calls'] != 1 for v in observed.values()):
            raise RuntimeError('selected assistant projections were not each observed exactly once')
        if (tuple(output.last_hidden_state.shape) != tuple(hidden.shape)
                or output.last_hidden_state.dtype != torch.bfloat16
                or tuple(output.logits.shape) != (1, 1, actor.target.config.vocab)
                or not bool(torch.isfinite(output.last_hidden_state).all())
                or not bool(torch.isfinite(output.logits).all())):
            raise ValueError('assistant calibration output geometry or finite domain differs')
        for name in observed:
            counts['observations'][name] += 1
        actor._records.append({'phase': sample.phase, 'input_sha256': sample.input_sha256,
                              'seed': sample.seed, 'trajectory': sample.trajectory, 'gradients': quota.gradients,
                              'assistant_model_class': type(model).__module__ + '.' + type(model).__qualname__,
                              'logits_dtype': str(output.logits.dtype),
                              'position': frame.position, 'recurrent_step': frame.step, 'anchor': frame.anchor,
                              'accepted_token_sha256': frame.token_sha256, 'borrowed_bytes': frame.borrowed_bytes,
                              'observations': observed})
        frame._latest_hidden = output.last_hidden_state
    except BaseException as error:
        primary, actor._failed = error, True
    for hook in hooks:
        try:
            hook.remove()
        except BaseException as error:
            errors.append(error)
    if errors:
        actor._failed = True
    raise_failures(primary, errors)
    return output
