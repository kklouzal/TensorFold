"""This process's identity: the instance id and start time a client compares to notice a restart.

Both are fixed when the server module first loads (before the model), so they name the serving process, whose prefix
cache and counters a restart empties.
"""

from __future__ import annotations

import time
import uuid

INSTANCE_ID = uuid.uuid4().hex
STARTED_UNIX_S = time.time()            # wall clock, for display and Prometheus's process_start_time_seconds
STARTED_MONOTONIC = time.monotonic()    # uptime runs on the monotonic clock


__all__ = ["INSTANCE_ID", "STARTED_MONOTONIC", "STARTED_UNIX_S"]
