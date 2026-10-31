"""Native Gemma4 assistant with deterministic proposals and exact delta-q law.

The target exclusively owns committed donor KV, masks and final hidden states.
An assistant call borrows these only until the next target mutation. Every
recurrent draft step uses the same anchor position and frozen donor KV, matching
Transformers5.17 SinglePositionMultiTokenCandidateGenerator.
"""
from __future__ import annotations

from pathlib import Path

import torch

from tensorfold.cuda.tensor_file import checkpoint_path
from tensorfold.engine.rejection import ChainPacket, ProbabilityRow
from ..assistant_metadata import checked_headers


class _State:
    """No duplicate hidden/KV authority: target snapshots own all committed state."""
    nbytes = 0

    def __init__(self, owner):
        self.owner = owner

    def reset(self):
        pass

    def snapshot(self):
        return None

    def restore(self, snapshot):
        if snapshot is not None:
            raise ValueError("Gemma assistant has no independent recurrent cache")


class _Chains:
    def __init__(self, tokens, history, sampled):
        self.tokens, self.history, self.sampled = tuple(tokens), history, sampled

    def _validate(self, row, count, history, sampled):
        if row != 0 or history is not self.history or self.sampled is not sampled:
            raise ValueError("Gemma assistant packet policy differs")
        if type(count) is not int or count < 0:
            raise ValueError("Gemma assistant proposal count must be nonnegative")
        return self.tokens[:count]

    def finish(self, row, max_nodes, history=None, *, stop=None):
        tokens = self._validate(row, max_nodes, history, False)
        # Native argmax proposal has unit probability, hence zero surprisal.
        return list(tokens), list(range(-1, len(tokens) - 1)), [0.0] * len(tokens)

    def sample(self, row, max_nodes, sampling, first_position, history=None):
        tokens = self._validate(row, max_nodes, history, True)
        if sampling is None or sampling.temperature <= 0:
            raise ValueError("Sampled target verification needs positive temperature")
        # Actual proposal draw is deterministic even for a stochastic target.
        # Reporting softmax logits here would misdescribe q and bias rejection.
        return ChainPacket(tokens, tuple(ProbabilityRow([t], [1.0]) for t in tokens), first_position)


class Gemma4AssistantHead:
    feature_source = "final_hidden"
    tap_layers = ()
    bits = 16

    def __init__(self, assistant_dir: str | Path, target, context_limit, device=None):
        from safetensors import safe_open
        from transformers import Gemma4AssistantForCausalLM

        self.cfg, _ = checked_headers(assistant_dir)
        self.cfg.state_bytes(context_limit)
        if target.config.hidden != self.cfg.backbone_hidden or target.config.vocab != self.cfg.vocab:
            raise ValueError("Gemma assistant requires the native E4B target geometry")
        self.target, self.context_limit = target, context_limit
        self.device = torch.device(device) if device is not None else target.device
        if self.device != target.device or target.embedding_weight.dtype != torch.bfloat16:
            raise ValueError("Gemma assistant and target must share a BF16 device")
        # Validate the only data-dependent indexing authority once before GPU use.
        with safe_open(str(checkpoint_path(assistant_dir, "model.safetensors")), framework="pt", device="cpu") as file:
            ordering = file.get_tensor("masked_embedding.token_ordering")
        if not torch.equal(ordering.sort().values, torch.arange(self.cfg.vocab, dtype=torch.int64)):
            raise ValueError("Gemma assistant token ordering must be a complete vocabulary permutation")
        model, loading = Gemma4AssistantForCausalLM.from_pretrained(
            assistant_dir, local_files_only=True, trust_remote_code=False, output_loading_info=True,
            dtype=torch.bfloat16, attn_implementation="sdpa")
        try:
            if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise ValueError(f"Gemma assistant checkpoint was not completely consumed: {loading}")
            self.model = model.eval().to(self.device)
        except BaseException:
            # Drop this constructor's sole alias; native loader/transfer frames
            # retain their own failure context under library ownership.
            del model
            raise
        self.block = self.rows = self.cfg.block
        self.parameter_bytes = self.cfg.parameter_bytes
        self.resident_bytes, self.loading_peak_bytes = self.cfg.memory_bytes()
        self._bound = None

    def bind_state(self, state):
        if self._bound is not None:
            raise RuntimeError("Gemma assistant target state is bound once")
        if state.owner is not self.target:
            raise ValueError("Gemma assistant target state belongs to another model")
        self._bound = state

    def make_state(self):
        if self._bound is None:
            raise RuntimeError("Bind the target state before creating assistant state")
        return _State(self)

    def append(self, state, hidden, *, prompt=False):
        if (state.owner is not self or hidden.ndim != 2 or hidden.shape[1] != self.cfg.backbone_hidden
                or hidden.dtype != torch.bfloat16 or hidden.device != self.device):
            raise ValueError("Gemma assistant accepted target hidden geometry differs")
        # The target already published the accepted last hidden during rollback.

    @torch.inference_mode()
    def propose(self, state, anchor, count, history=None, *, sampled=False, sampling=None):
        from tensorfold.cuda.sampling import sample_rows

        if state.owner is not self or self._bound is None:
            raise ValueError("Gemma assistant draft/target ownership differs")
        if type(anchor) is not int or not 0 <= anchor < self.cfg.vocab:
            raise ValueError("Gemma assistant anchor is outside the target vocabulary")
        if type(count) is not int or not 0 <= count < self.block:
            raise ValueError("Gemma assistant supports zero through seven proposed tokens")
        if sampled and (sampling is None or sampling.temperature <= 0):
            raise ValueError("Sampled target verification needs positive temperature")
        if not count:
            return None
        held = self.target.assistant_state(self._bound)
        position = held["position"]
        if type(position) is not int or not 0 < position < self.context_limit:
            raise ValueError("Gemma assistant anchor exceeds committed target context")
        # Deterministic recurrent prefixes do not depend on requested depth.
        # Omit descendants the admitted target cannot possibly verify.
        count = min(count, self.context_limit - position - 1)
        if count <= 0:
            return None
        hidden = held["last_hidden_state"]
        token = anchor
        positions = torch.tensor([[position]], device=self.device, dtype=torch.long)
        tokens = []
        for _ in range(count):
            ids = torch.tensor([[token]], device=self.device, dtype=torch.long)
            embedding = self.target.model.get_input_embeddings()(ids)
            out = self.model(inputs_embeds=torch.cat((embedding, hidden), dim=-1), position_ids=positions,
                             attention_mask=held["attention_mask"], shared_kv_states=held["shared_kv_states"],
                             use_cache=False)
            # Native ordered logits include the finite floor outside selected
            # centroids. Keep that arithmetic; only read the deterministic argmax.
            token = sample_rows(out.logits.reshape(1, -1), [position], None)[0]
            hidden = out.last_hidden_state
            tokens.append(token)
        return _Chains(tokens, history, sampled)
