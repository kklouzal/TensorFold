"""Opt-in target-owned committed native-assistant features.

HF's shared sliding donor output can contain512 useful rows while its persistent
cache retains511. Keep that final native window explicitly; rejected verifier
rows must never enter a later assistant call. The exclusive target lane owns
all mutations; borrowed views live only until its next mutation.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class AssistantSnapshot:
    last_hidden: object
    sliding_key: object
    sliding_value: object


class AssistantHistory:
    def __init__(self, config):
        self.window = config.sliding_window
        donors = config.layer_types[:config.num_hidden_layers-config.num_kv_shared_layers]
        self.full_layer = max(i for i, kind in enumerate(donors) if kind == "full_attention")
        self.last_hidden = self.sliding_key = self.sliding_value = None
        self.pending = None

    def reset(self):
        self.last_hidden = self.sliding_key = self.sliding_value = None
        self.pending = None

    def stage(self, start, hidden, shared):
        if self.pending is not None:
            raise RuntimeError("assistant target history needs commit before another forward")
        key, value = shared["sliding_attention"]
        self.pending = (start, hidden, key, value)

    def commit(self, position):
        if self.pending is None:
            return
        start, hidden, key, value = self.pending
        accepted = position-start
        if not 0 <= accepted <= hidden.shape[0]:
            raise ValueError("assistant feature commit is outside its tentative forward")
        if accepted:
            # Shared output ends at start+all_forward_rows; select only accepted
            # rows before taking the last native512-key window.
            stop = key.shape[-2] - hidden.shape[0] + accepted
            begin = max(0, stop-self.window)
            last_hidden = hidden[accepted-1:accepted].unsqueeze(0).clone()
            sliding_key = key[:, :, begin:stop].clone()
            sliding_value = value[:, :, begin:stop].clone()
            self.last_hidden, self.sliding_key, self.sliding_value = last_hidden, sliding_key, sliding_value
        self.pending = None

    def snapshot(self):
        if self.pending is not None:
            raise RuntimeError("cannot snapshot uncommitted assistant target features")
        return AssistantSnapshot(*(None if x is None else x.clone()
                                   for x in (self.last_hidden, self.sliding_key, self.sliding_value)))

    def restore(self, snapshot):
        if not isinstance(snapshot, AssistantSnapshot):
            raise ValueError("assistant target snapshot contract differs")
        last_hidden, key, value = (None if x is None else x.clone()
                                   for x in (snapshot.last_hidden, snapshot.sliding_key, snapshot.sliding_value))
        self.last_hidden, self.sliding_key, self.sliding_value = last_hidden, key, value
        self.pending = None

    def view(self, state):
        if state.poisoned or self.pending is not None or state.position < 1 or self.last_hidden is None:
            raise RuntimeError("assistant needs a committed healthy target prefix")
        full = state.cache.layers[self.full_layer]
        return {"position": state.position, "last_hidden_state": self.last_hidden,
                "attention_mask": state.attention_mask,
                "shared_kv_states": {"sliding_attention": (self.sliding_key, self.sliding_value),
                                     "full_attention": (full.keys[:, :, :state.position],
                                                        full.values[:, :, :state.position])}}
