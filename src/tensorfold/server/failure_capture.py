"""Failure-only diagnostic capture: one failed request's body, its generated token ids and stats, as one JSON file.

Nothing is logged for a normal reply. Where the server detects a failure it names (a reply that ended at an end token
inside its think block), it prints one warning line; only when ``TENSORFOLD_FAILURE_CAPTURE_DIR`` is set does it also
write that request to the directory, its image and video parts redacted as the request log redacts them. The
directory is bounded: a capture that would make it hold more than ``MAX_FILES`` captures or ``MAX_BYTES`` bytes is
skipped (with a line saying so), never written over an older one; remove captures to make room. Files are written
whole (a temporary file renamed into place), readable by the server's user alone. A capture never changes the reply
and never raises: a failed write prints why.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from tensorfold import __version__

ENV = "TENSORFOLD_FAILURE_CAPTURE_DIR"
MAX_FILES = 32
MAX_BYTES = 256 * 1024**2
SUFFIX = ".capture.json"
_WRITING = threading.Lock()         # one capture at a time: its bound check and write are one step


def redact_images(value: Any) -> Any:
    """A request body to record: every image or video part's URL or data replaced, the rest kept."""

    if isinstance(value, list):
        return [redact_images(item) for item in value]
    if not isinstance(value, dict):
        return value
    if value.get("type") in ("image_url", "video_url"):
        return {**value, value["type"]: {"url": "<redacted>"}}
    return {key: redact_images(item) for key, item in value.items()}


def capture(kind: str, request_id: str, body: Any, record: dict[str, Any]) -> Path | None:
    """Write one failed request (``record``: its token ids, stats and what the failure was) to the capture directory;
    its path, or None when capture is off or skipped."""

    directory = os.environ.get(ENV)
    if not directory:
        return None
    document = {"kind": kind, "request_id": request_id, "time_unix_s": round(time.time(), 3),
                "server_version": __version__, **record, "body": redact_images(body)}
    try:
        data = json.dumps(document).encode()
        with _WRITING:
            folder = Path(directory)
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            held = [entry.stat().st_size for entry in os.scandir(folder)
                    if entry.is_file() and entry.name.endswith(SUFFIX)]
            if len(held) >= MAX_FILES or sum(held) + len(data) > MAX_BYTES:
                print(f"[tensorfold] failure capture skipped for {request_id}: {folder} holds {len(held)} captures "
                      f"({sum(held)} bytes; at most {MAX_FILES} and {MAX_BYTES} bytes); remove old ones", flush=True)
                return None
            # a client's request id can repeat: the name's own suffix keeps every capture
            stamp = time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
            target = folder / f"{stamp}-{request_id}-{uuid.uuid4().hex[:8]}{SUFFIX}"
            handle, temporary = tempfile.mkstemp(dir=folder, prefix=".capture-")      # mode 0600
            try:
                with os.fdopen(handle, "wb") as out:
                    out.write(data)
                os.replace(temporary, target)
            except BaseException:
                Path(temporary).unlink(missing_ok=True)
                raise
    except (OSError, TypeError, ValueError) as exc:
        print(f"[tensorfold] warning: failure capture for {request_id} to {directory} failed: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None
    print(f"[tensorfold] failure captured: {target}", flush=True)
    return target


__all__ = ["ENV", "MAX_BYTES", "MAX_FILES", "capture", "redact_images"]
