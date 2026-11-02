"""One request at a time on an engine that decodes one, background requests (priority "background") last."""
# A background reply yields between rounds to an arriving foreground request, then replays from its prompt.

from __future__ import annotations

import threading
from typing import Callable

from tensorfold.server.cancellation import RequestCancelled


class Turns:
    def __init__(self) -> None:
        self.cv = threading.Condition()
        self.busy = False
        self.waiting = 0                             # foreground requests waiting for the engine
        self.parked = 0                              # every request in take(), background included

    def take(self, background: bool, cancelled: Callable[[], bool] | None = None,
             *, tick: Callable[[], bool] | None = None) -> None:
        """Wait for the engine; optional keepalive/cancellation IO runs outside its lock."""
        with self.cv:
            self.parked += 1
            self.waiting += not background
        try:
            while True:
                with self.cv:
                    if not self.busy and not (background and self.waiting):
                        self.busy = True
                        return
                    self.cv.wait(timeout=1.0)
                # A slow/disconnected writer must not hold the engine scheduler
                # lock or block its owner from giving the turn back.
                if ((cancelled is not None and cancelled())
                        or (tick is not None and tick())):
                    raise RequestCancelled("the client left before the request started")
        finally:
            with self.cv:
                self.parked -= 1
                self.waiting -= not background

    def give(self) -> None:
        with self.cv:
            self.busy = False
            self.cv.notify_all()

    def wanted(self) -> bool:
        """Whether a foreground request waits for the engine."""

        return self.waiting > 0


class Yield:
    """A background request's cut while a foreground one waits: after this round's tokens; it replays later."""

    replay = True                                # its reply is decoded again from the same prompt, never resumed

    def __init__(self, turns: Turns) -> None:
        self.turns = turns

    def cut(self, tokens) -> tuple[int, list[int]] | None:
        return (len(tokens), []) if tokens and self.turns.wanted() else None

    def observe(self, token: int) -> None:
        pass
