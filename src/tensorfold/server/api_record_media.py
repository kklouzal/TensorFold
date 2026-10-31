"""Immutable encoded media dependencies for durable logical API items.

Public item JSON is preserved. An execution-only snapshot resolves newly added
remote images once, before generation/publication. Existing item references use
their original snapshot; no URL is silently refreshed on a later turn.
"""
from __future__ import annotations

import base64
import sys
import time

from tensorfold.server.api_records import RecordCapacity
from tensorfold.server.errors import CapacityError, RequestError
from tensorfold.cleanup import raise_failures, rollback


def video_capable(app):
    frontend = getattr(app, "vision", None)
    return bool(getattr(frontend, "videos", False) and (
        getattr(frontend, "responses_video", False)
        or getattr(frontend, "is_media_runtime", False)
        and getattr(frontend, "media_envelope", None) is not None))


def _parts(item):
    value = item.get("content") if item["type"] == "message" else item.get("output")
    return value if isinstance(value, list) else ()


def capture_media(operation, app):
    store, principal = operation.store, operation.principal
    pending = []
    for item in operation.added:
        parts = _parts(item)
        if any(part.get("type") == "input_file" for part in parts) and not video_capable(app):
            raise RequestError("video input_file requires the admitted SM70 temporal media runtime")
        with store._lock:
            row = store._connection.execute("SELECT body,execution FROM items WHERE owner=? AND id=?", (principal, item["id"])).fetchone()
            if row is not None:
                if row["body"] != store._blob(item):
                    raise RequestError("an existing item ID cannot change its logical contents")
                if row["execution"] is not None:
                    operation.snapshots[item["id"]] = store._decode(row["execution"])
                continue
        remote = [index for index, part in enumerate(parts) if part.get("type") == "input_image"
                  and not part["image_url"].startswith("data:")]
        if remote and operation.byte_reservation:
            snapshot = store._decode(store._blob(item))
            operation.snapshots[item["id"]] = snapshot
            pending.extend((_parts(snapshot)[index], item["id"]) for index in remote)
    if not pending:
        return
    frontend = getattr(app, "vision", None)
    if frontend is None or not getattr(frontend, "allow_urls", False):
        raise RequestError("image URLs are off on this server; send a data URL or enable vision URLs")
    from tensorfold.vision.images import DEFAULT_LIMITS, ImageSource, _check_source
    from tensorfold.vision.images_http import ImageInputError
    sources = [ImageSource(part["image_url"], part.get("detail", "auto")) for part, _ in pending]
    bounded = getattr(frontend, "media_envelope", None)
    try:
        if bounded is not None:
            from tensorfold.vision.replay import MediaBusy, open_replay
            try:
                lease = frontend.store.stage(sources, allow_urls=True, cancelled=frontend.frontend.cancelled)
                try:
                    for ref, (part, identifier) in zip(lease.refs, pending, strict=True):
                        _admit_conversion(operation, ref.byte_length, identifier, part)
                        with open_replay(ref, bounded) as source:
                            _replace(operation, part, source.read(ref.byte_length), identifier)
                except BaseException as primary:
                    rollback(lease, primary, lease.close)
                lease.close()
            except MediaBusy as error:
                _check_media_health(frontend, store, error)
                raise CapacityError(str(error)) from error
            except BaseException as error:
                _check_media_health(frontend, store, error)
                raise
        else:
            from tensorfold.vision.images_http import fetch_image
            limits = getattr(app, "image_limits", DEFAULT_LIMITS)
            if len(sources) > limits.max_images:
                raise RequestError("durable image capture exceeds the image count allowance")
            deadline, total = time.monotonic() + limits.total_timeout_seconds, 0
            for source, (part, identifier) in zip(sources, pending, strict=True):
                _check_source(source, limits, True)
                remaining = min(limits.max_encoded_bytes, limits.max_total_encoded_bytes - total)
                remaining = min(remaining, _raw_allowance(operation, identifier, part))
                if remaining <= 0:
                    raise RequestError("durable image capture exceeds the total encoded byte allowance")
                _admit_conversion(operation, remaining, identifier, part)
                data, _ = fetch_image(source.url, max_bytes=remaining,
                    deadline=min(deadline, time.monotonic() + limits.timeout_seconds),
                    max_redirects=limits.max_redirects, max_url_chars=limits.max_url_chars)
                total += len(data)
                _replace(operation, part, data, identifier)
    except ImageInputError as error:
        raise RequestError(str(error)) from error


def _check_media_health(frontend, store, primary):
    try:
        frontend.check_media_health()
    except BaseException as health:
        with store._lock:
            if store._failure is None:
                store._failure = primary
        raise_failures(primary, [health])


def _allowances(operation, identifier, part):
    store = operation.store
    current = sum(len(store._blob(item)) for item in operation.added)
    current += sum(len(store._blob(item)) for item in operation.snapshots.values())
    # The replacement's ASCII JSON length is known before allocating base64.
    from tensorfold.server.api_records import _json
    adjustment = len(_json("data:application/octet-stream;base64,")) - len(_json(part["image_url"]))
    encoded = min(store.limits.max_record_bytes - len(store._blob(operation.snapshots[identifier])),
                  operation.byte_reservation - current) - adjustment
    materialized = store.check_materialized(operation.history, operation.execution_history, operation.added, operation.snapshots)
    memory = store.limits.max_materialized_bytes - materialized - 3 * sys.getsizeof("") - sys.getsizeof(b"")
    return encoded, memory


def _raw_allowance(operation, identifier, part):
    encoded, memory = _allowances(operation, identifier, part)
    groups, tail = divmod(max(0, memory), 15)  # raw3 + three base64 copies4 each
    memory_raw = 3 * groups + (2 if tail >= 14 else 1 if tail >= 13 else 0)
    return max(0, min(3 * (encoded // 4), memory_raw))


def _admit_conversion(operation, raw_size, identifier, part):
    encoded_allowance, memory = _allowances(operation, identifier, part)
    encoded = 4 * ((raw_size + 2) // 3)
    if encoded > encoded_allowance or raw_size + 3 * encoded > memory:
        raise RecordCapacity("immutable media/base64/JSON overlap exceeds the declared API record allowance")


def _replace(operation, part, data, identifier):
    _admit_conversion(operation, len(data), identifier, part)
    part["image_url"] = "data:application/octet-stream;base64," + base64.b64encode(data).decode("ascii")
    # This canonical serialized execution view is the separately charged
    # dependency; the public original URL/metadata is never overwritten.
    operation.store._blob(operation.snapshots[identifier])
    if sum(len(operation.store._blob(item)) for item in operation.added) + sum(len(operation.store._blob(item)) for item in operation.snapshots.values()) > operation.byte_reservation:
        raise RecordCapacity("immutable execution snapshot exceeds its reserved API storage")
