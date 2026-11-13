"""Owned-CID retirement for the short V100 diagnostic host script.

Construct before Docker effects. Call ``retire`` in ``finally`` and publish only
afterwards; publication exceptions must propagate to the caller's nonzero exit.
This owns no model, HTTP, profile, or qualification policy.
"""
from __future__ import annotations

import json
import math
import re
import subprocess
import time


class OwnedContainer:
    def __init__(self, run=subprocess.run):
        self.run = run
        self.cid = None
        self.state = None
        self.errors = [None] * 6
        # Allocate the retirement command storage before container acquisition.
        self.inspect = ["docker", "inspect", "--format", "{{json .State}}", None]
        self.stop = ["docker", "kill", "--signal=SIGINT", None]
        self.wait = ["docker", "wait", None]
        self.kill = ["docker", "kill", "--signal=SIGKILL", None]

    def _run(self, argv, timeout):
        result = self.run(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)
        result.check_returncode()
        return result

    def create(self, argv, publish):
        """Bind the successful create response before fallible journal work."""
        result = self._run(argv, 60)
        cid = result.stdout.strip().decode("ascii")
        if re.fullmatch(r"[0-9a-f]{64}", cid) is None:
            raise ValueError("Docker create did not return an owned full CID")
        self.cid = cid
        self.inspect[-1] = self.stop[-1] = self.wait[-1] = self.kill[-1] = cid
        publish(argv, result)
        return cid

    def retire(self, *, wait_timeout_s=None):
        """Attempt every required retirement step without logging/formatting.

        Exact-CID SIGKILL is containment after failed normal retirement; its
        original error remains in the preallocated slots and prevents success.
        """
        if wait_timeout_s is not None and (isinstance(wait_timeout_s,bool) or not isinstance(wait_timeout_s,(int,float))
                or not math.isfinite(wait_timeout_s) or wait_timeout_s<0):
            raise ValueError('retirement wait must be finite and nonnegative')
        # No explicit budget preserves the existing90-second Docker wait.
        # A caller's remaining operation allowance also pays inspect/SIGINT
        # command latency; never restart it after those commands return.
        deadline=None if wait_timeout_s is None else time.monotonic()+wait_timeout_s
        if self.cid is None:
            return
        running = True
        try:
            self.state = json.loads(self._run(self.inspect, 30).stdout)
            running = self.state["Running"]
        except BaseException as error:
            self.errors[0] = error
        if running:
            try:
                self._run(self.stop, 30)
            except BaseException as error:
                self.errors[1] = error
            try:
                self._run(self.wait,90 if deadline is None else max(0,deadline-time.monotonic()))
            except BaseException as error:
                self.errors[2] = error
                try:
                    self._run(self.kill, 30)
                except BaseException as failure:
                    self.errors[3] = failure
                try:
                    self._run(self.wait, 30)
                except BaseException as failure:
                    self.errors[4] = failure
        try:
            self.state = json.loads(self._run(self.inspect, 30).stdout)
        except BaseException as error:
            self.errors[5] = error

    def exit_code(self, primary):
        """Host callers must use this outcome, never their own fall-through 0."""
        state = self.state
        return int(primary is not None or any(error is not None for error in self.errors)
                   or type(state) is not dict or state.get("Status") != "exited"
                   or state.get("Running") is not False or state.get("Pid") != 0
                   or type(state.get("ExitCode")) is not int or state["ExitCode"] != 0
                   or state.get("OOMKilled") is not False)


def retire_client(process, errors):
    """Retire the owned Docker exec CLI into three caller-preallocated slots.

    Call after container retirement. A failed kill never skips the final wait;
    retain the initial timeout and every containment failure without formatting.
    """
    if process is None:
        return
    try:
        process.wait(timeout=30)
    except BaseException as error:
        errors[0] = error
        try:
            process.kill()
        except BaseException as failure:
            errors[1] = failure
        try:
            process.wait(timeout=30)
        except BaseException as failure:
            errors[2] = failure
