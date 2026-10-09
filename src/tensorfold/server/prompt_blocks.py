"""Render chat prompts and warm saved system blocks."""

from __future__ import annotations

from concurrent.futures import Future
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from tensorfold.server.checkpoints import longest_common_prefix
from tensorfold.server.cancellation import Cancellation
from tensorfold.server.scheduler import ChatJob, Scheduler
from tensorfold.server.text import render_prompt_ids
from tensorfold.thread_work import ThreadWork

_REQUEST = threading.local()


class PromptBlocks:
    """ChatApp prompt rendering and saved system-block warmup."""

    def render(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        thinking: bool | None = None,
    ) -> tuple[list[int], int]:
        """Prompt ids plus the length of the rendered history that prefixes them."""

        thinking = self.enable_thinking if thinking is None else bool(thinking)
        effort = self.effort_for((getattr(_REQUEST, "sampling", None) or {}).get("reasoning_effort"))
        with self.tokenizer_lock:
            prompt = render_prompt_ids(
                self.tokenizer,
                messages,
                tools=tools,
                enable_thinking=thinking,
                reasoning_effort=effort,
                late_system=self.late_system,
            )
            history = render_prompt_ids(
                self.tokenizer,
                messages,
                tools=tools,
                enable_thinking=thinking,
                reasoning_effort=effort,
                add_generation_prompt=False,
                late_system=self.late_system,
            )
        history_len = len(history) if 0 < len(history) < len(prompt) and prompt[: len(history)] == history else 0
        if not history_len and len(prompt) > 1 and history == prompt:
            # Gemma 4 after a tool result has no generation suffix, so the checkpoint sits one token short of the end.
            history_len = len(prompt) - 1
        return prompt, history_len

    def system_prefix_len(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        prompt_ids: list[int],
        thinking: bool | None = None,
    ) -> int:
        """A reusable system prefix, found with a probe in place of the first user message; zero for short matches."""

        effort = self.effort_for((getattr(_REQUEST, "sampling", None) or {}).get("reasoning_effort"))
        first_user = next((i for i, m in enumerate(messages) if m.get("role") == "user"), None)
        if first_user is None:
            return 0
        probe = [*messages[:first_user], {"role": "user", "content": "⁣probe"}]
        try:
            with self.tokenizer_lock:
                other = render_prompt_ids(
                    self.tokenizer,
                    probe,
                    tools=tools,
                    enable_thinking=self.enable_thinking if thinking is None else bool(thinking),
                    reasoning_effort=effort,
                    late_system=self.late_system,
                )
        except Exception:  # noqa: BLE001 - a template quirk must not fail the request
            return 0
        shared = longest_common_prefix(prompt_ids, other)
        return shared if shared >= 512 else 0

    def _warm_known_blocks(self, snapshot_dir: Path, model_id: str) -> None:
        """Compute a saved system block as owned background jobs, a prompt chunk each.

        ``warmup_result`` retains the terminal outcome; /health reports a failed
        warmup's error type. App.close cancels its jobs and joins its worker.
        """

        from tensorfold.engine.prefill_plan import block_jobs
        from tensorfold.engine.prefix_snapshots import blocks_to_warm

        blocks = blocks_to_warm(snapshot_dir, model_id, registry=self.snapshot_registry)[:1]
        if not blocks:
            return
        pad = int(self.tokenizer.encode("\n", add_special_tokens=False)[-1])
        cancellation = self.warmup_cancellation = Cancellation()
        result = self.warmup_result = Future()

        def warm() -> None:
            try:
                warm_blocks()
            except BaseException as error:
                # Retain the worker's terminal outcome without its arrays or frames.
                result.set_exception(Scheduler._terminal_error(error))
            else:
                result.set_result(None)
            finally:
                self.warming = False

        def warm_blocks() -> None:
            for tokens in blocks:
                started = time.perf_counter()
                jobs = block_jobs(self.engine.prefill_plan, tokens, pad)
                for i, (prompt, at) in enumerate(jobs):
                    final = i == len(jobs) - 1
                    while True:
                        cancellation.check()
                        job = ChatJob(
                            job_id=f"warm-{uuid.uuid4().hex[:8]}",
                            prompt_ids=prompt,
                            max_tokens=1,
                            temperature=0.0,
                            history_len=at,
                            shared_prefix_lens=(at,) if final else (),
                            drafts=False,
                            background=True,
                            cancellation=cancellation,
                        )
                        self.scheduler.submit(job)
                        while job.chunks.get() is not None:
                            pass
                        if job.error is not None:
                            raise job.error
                        if not job.preempted:
                            break
                print(
                    f"[tensorfold] warmed system block tokens={jobs[-1][1] if jobs else 0} of {len(tokens)} in "
                    f"{time.perf_counter() - started:.1f}s",
                    flush=True,
                )

        print(
            f"[tensorfold] warming {len(blocks)} saved system block(s) for these kernels in the background: "
            "until it ends, a request first waits for one prompt chunk (GET /health reports warming)",
            flush=True,
        )
        self.warming = True
        self.warmup_work = ThreadWork(warm)
        self.warmup_thread = threading.Thread(target=self.warmup_work.run, name="warm-blocks", daemon=True)
        try:
            self.warmup_work.launch(self.warmup_thread)
        except BaseException as error:
            self.warming = False
            result.set_exception(Scheduler._terminal_error(error))
            raise
