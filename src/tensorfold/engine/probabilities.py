"""Request-owned target probabilities, keyed by absolute position across replay."""

import math


class Probabilities:
    def __init__(self, top: int, start: int, count: int):
        self.top, self.start, self.count = top, start, count
        self.rows: dict[int, dict] = {}

    def add(self, positions, tokens, values, top_ids, top_values):
        for pos, token, value, ids, scores in zip(positions, tokens, values, top_ids, top_values, strict=True):
            if not self.start <= pos < self.start + self.count:
                continue
            if not all(math.isfinite(x) for x in (value, *scores)):
                raise RuntimeError("target log probabilities are not finite")
            row = {"id": int(token), "logprob": value, "top": list(zip(ids, scores, strict=True))}
            old = self.rows.get(pos)
            if old is not None and old != row:
                raise RuntimeError("a replay changed the target log probabilities")
            self.rows[pos] = row

    def emitted(self, tokens):
        rows = [self.rows[self.start + i] for i in range(len(tokens))]
        if [row["id"] for row in rows] != list(tokens):
            raise RuntimeError("target probabilities do not match emitted tokens")
        return rows


class LabelProbabilities(Probabilities):
    """Collect label logits and full-vocabulary logsumexp only at the prompt's final position."""

    def __init__(self, labels, start: int):
        super().__init__(0, start, 1)
        self.labels = [int(token) for token in labels]
        self.label_logits: list[float] | None = None
        self.logsumexp: float | None = None

    def add_labels(self, position: int, logits: list[float], logsumexp: float) -> None:
        if position == self.start:
            self.label_logits, self.logsumexp = logits, logsumexp


class TeacherForcing(Probabilities):
    """Private scoring request: the reply is ``forced`` (never sampled) and every prompt row is scored too.

    ``rows`` holds the reply's target log probabilities by absolute position (``add``); ``prompt_rows`` the prompt
    tokens' own (``add_prompt``: position p scores prompt[p] from the row of p - 1); ``where`` records which row
    of which window scored each reply position (``"prompt-end"`` or ``(row, window rows)`` of a decode round).
    Only engines declaring ``supports_teacher_forcing`` honor ``forced``; others must refuse this sink.
    """

    def __init__(self, top: int, prompt, forced):
        self.prompt = [int(token) for token in prompt]
        self.forced = [int(token) for token in forced]
        if len(self.prompt) < 1 or not self.forced:
            raise ValueError("teacher forcing needs a prompt and a nonempty forced reply")
        super().__init__(top, len(self.prompt), len(self.forced))
        self.prompt_rows: dict[int, dict] = {}
        self.where: dict[int, object] = {}

    def add_prompt(self, positions, tokens, values, top_ids, top_values):
        for pos, token, value, ids, scores in zip(positions, tokens, values, top_ids, top_values, strict=True):
            if not 1 <= pos < self.start or self.prompt[pos] != token:
                raise RuntimeError("a scored prompt row differs from the request's prompt")
            if not all(math.isfinite(x) for x in (value, *scores)):
                raise RuntimeError("target log probabilities are not finite")
            row = {"id": int(token), "logprob": value, "top": list(zip(ids, scores, strict=True))}
            old = self.prompt_rows.get(pos)
            if old is not None and old != row:
                raise RuntimeError("a replay changed the prompt log probabilities")
            self.prompt_rows[pos] = row

    def origins(self, positions, origins):
        for pos, origin in zip(positions, origins, strict=True):
            self.where.setdefault(pos, origin)
