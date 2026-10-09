"""Mac ``/v1/decisions`` requests: one prompt-lane prefill per question, scored from its last row."""

from __future__ import annotations

import uuid
from typing import Any

from tensorfold.server.decisions import DecisionError, _render_text, _wording, build_response, prepare
from tensorfold.server.errors import RequestError
from tensorfold.server.scheduler import ChatJob


class DecisionRequests:
    """Submit each decision question as a prompt-lane job and return its label logits."""

    def _decision_keep(self, text: str, content: str, prompt_ids: list[int]) -> tuple[int, tuple[int, ...], int]:
        """History length, shared-prefix cuts, and where a different question stops sharing tokens."""

        from tensorfold.server.checkpoints import longest_common_prefix
        from tensorfold.server.text import render_prompt_ids

        # The input stays when the question changes. A chat keeps a system block with these same cuts.
        messages = [{"role": "user", "content": content}]
        probe = [{"role": "user", "content": f"{text}\n\n\u2063probe"}]
        try:
            with self.tokenizer_lock:
                history = render_prompt_ids(self.tokenizer, messages, enable_thinking=False,
                                            add_generation_prompt=False)
                other = render_prompt_ids(self.tokenizer, probe, enable_thinking=False)
        except Exception:  # noqa: BLE001 - a template quirk must not fail the request
            return 0, (), 0
        prefix = 0 < len(history) < len(prompt_ids) and prompt_ids[:len(history)] == history
        history_len = len(history) if prefix else 0
        shared = longest_common_prefix(prompt_ids, other)
        if not 512 <= shared < len(prompt_ids):
            return history_len, (), shared if 0 < shared < len(prompt_ids) else 0
        cuts = tuple(n for n in (shared - 2048, shared - 512, shared) if n >= 512)
        return history_len, cuts, shared

    def decisions(self, body: dict[str, Any]) -> dict[str, Any]:
        """Answer typed questions from the last row of a prompt-lane prefill. No text is generated."""

        try:
            with self.tokenizer_lock:
                prepared = prepare(self.tokenizer, body, context_len=self.context_window or None)
        except DecisionError as exc:
            raise RequestError(str(exc)) from exc
        text = _render_text(body.get("input"))
        jobs = []
        prefixes = []
        try:
            for item, question in zip(prepared, body["questions"]):
                _, _, _, content = _wording(text, question)
                history_len, shared, prefix_len = self._decision_keep(text, content, list(item.prompt_ids))
                job = ChatJob(
                    job_id=f"decision-{uuid.uuid4().hex[:8]}",
                    prompt_ids=list(item.prompt_ids),
                    max_tokens=1,
                    temperature=0.0,
                    drafts=False,
                    label_ids=tuple(item.label_ids),
                    history_len=history_len,
                    shared_prefix_lens=shared,
                )
                jobs.append((item, job))     # retain ownership even if submit fails after accepting it
                self.scheduler.submit(job)
                prefixes.append(prefix_len)
            scored = []
            for (item, job), prefix_len in zip(jobs, prefixes):
                if not job.done.wait(600.0):
                    raise TimeoutError("the engine did not score the prompt in time")
                if isinstance(job.error, ValueError):
                    raise RequestError(f"question {item.id!r}: {job.error}") from job.error
                if job.error is not None:
                    raise job.error
                logits, total = job.scored
                shown = ",".join(repr(value) for value in logits)
                print(f"[tensorfold] decision {job.job_id} cached={job.cached_tokens} "
                      f"prefix={prefix_len} logits={shown} logsumexp={total!r}", flush=True)
                scored.append(job.scored)
            return build_response(body, prepared, scored)
        except BaseException as error:
            # The scheduler retains active arrays until its next cancellation point;
            # queued jobs finish immediately. A failed request owns no further scoring.
            previous = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                        BaseException.__suppress_context__.__get__(error))
            failures = []
            for _, job in jobs:
                if not job.done.is_set():
                    try:
                        self.scheduler.cancel(job.cancellation)
                    except BaseException as cleanup:
                        failures.append((job, cleanup))
            try:
                other = []
                if failures:
                    for failure in (*previous[:2], *(cleanup for _, cleanup in failures)):
                        if failure is not None and failure is not error and all(failure is not item for item in other):
                            other.append(failure)
                    for job, cleanup in failures:
                        try:
                            name = type.__dict__["__name__"].__get__(type(cleanup), type)[:80]
                            BaseException.add_note(error, f"cancelling decision job {job.job_id} failed: {name}")
                        except BaseException as annotation:
                            if annotation is not error and all(annotation is not item for item in other):
                                other.append(annotation)
                            break
                if other:
                    namespace = BaseException.__dict__["__dict__"].__get__(error, type(error))
                    dict.__setitem__(namespace, "_tensorfold_decision_failures", other)
                    cause = other[0] if len(other) == 1 else BaseExceptionGroup("decision cancellation failures", other)
                    dict.pop(namespace, "_tensorfold_decision_failures", None)
                    raise error from cause
                BaseException.__cause__.__set__(error, previous[0])
                BaseException.__context__.__set__(error, previous[1])
                BaseException.__suppress_context__.__set__(error, previous[2])
                raise error
            except BaseException as transport:
                if transport is error:
                    raise
                # All cancellations ran; named roots/statuses stay reachable in
                # this frame even if cold diagnostic retention/grouping fails.
                raise error from transport
