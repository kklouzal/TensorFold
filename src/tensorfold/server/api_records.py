"""Bounded durable logical API records, independent of retained computation.

One local Store owns its private SQLite file. Conversation mutation is an
operation-owned transaction, never a client revision protocol. Conversation
objects and their independently retained item relations have separate lifetimes.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, fields
import fcntl
import json
import math
import os
from pathlib import Path
import sqlite3
import sys
import threading
import time
import uuid

from tensorfold.cleanup import raise_failures, rollback
from tensorfold.file_io import FileStreams
from tensorfold.server.errors import RequestError


class RecordNotFound(RequestError):
    status = 404


class RecordConflict(RequestError):
    status = 409


class RecordCapacity(RequestError):
    status = 429
    code = "insufficient_quota"


_EXECUTION = ContextVar("tensorfold-logical-api-operation", default=None)


def current_operation():
    """Trusted same-dispatch operation; never inferred from client cache hints."""
    return _EXECUTION.get()


def _json(value):
    return json.dumps(value, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":"))


def _owner(principal):
    if type(principal) is not str or not principal:
        raise ValueError("logical records require an explicit trusted principal")
    return principal


@dataclass(frozen=True)
class ApiRecordLimits:
    max_responses: int
    max_conversations: int
    max_items: int
    max_total_bytes: int
    max_record_bytes: int
    max_input_items: int
    max_output_items: int
    max_history_items: int
    max_history_bytes: int
    max_active_operations: int
    max_principal_responses: int
    max_principal_conversations: int
    max_principal_items: int
    max_principal_bytes: int
    max_principal_operations: int
    max_materialized_bytes: int
    max_database_bytes: int
    max_operation_bytes: int
    response_retention_seconds: float | None

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name == "response_retention_seconds":
                try:
                    valid = value is None or (not isinstance(value, bool) and isinstance(value, (int, float))
                                               and math.isfinite(value) and value > 0)
                except OverflowError:
                    valid = False
                if not valid:
                    raise ValueError("response_retention_seconds must be positive finite seconds or explicit null")
            elif type(value) is not int or not 0 < value < 1 << 63:
                raise ValueError(field.name + " must be an explicit positive signed-int64 count")
        if self.max_operation_bytes > min(self.max_total_bytes, self.max_principal_bytes):
            raise ValueError("the global and principal byte quotas must admit one complete operation")
        if self.max_input_items > self.max_history_items:
            raise ValueError("the history item allowance must admit one input")

    @classmethod
    def from_record(cls, record):
        if type(record) is not dict:
            raise ValueError("API record limits must be a JSON object")
        record = dict(record)
        version = record.pop("version", None)
        if record.pop("schema", None) != "tensorfold.api-record-limits" or type(version) is not int or version != 1:
            raise ValueError("unsupported API record limits schema")
        if set(record) != {field.name for field in fields(cls)}:
            raise ValueError("API record limits must provide every declared field and no unknown fields")
        return cls(**record)


class ApiRecordStore:
    """Durable API JSON/items; no GPU slot, model state or checkpoint promise."""
    durable_conversations = True

    def __init__(self, path, *, limits: ApiRecordLimits, reference_unavailable=None, reference_committed=None):
        if type(limits) is not ApiRecordLimits:
            raise TypeError("API record storage requires explicit typed limits")
        self.limits, self.pid = limits, os.getpid()
        self._lock, self._operations = threading.RLock(), {}
        self._connection = None
        self._files = FileStreams(max_files=1)
        self._closed = self._retiring = False
        self._failure = None
        self._reference_unavailable = self._reference_committed = None
        self._listeners_bound = False
        if reference_unavailable is not None or reference_committed is not None:
            self.bind_reference_listeners(unavailable=reference_unavailable, committed=reference_committed)
        path = Path(path).absolute()
        parent = path.parent.resolve(strict=True)
        info = parent.stat()
        if parent != path.parent or info.st_uid != os.geteuid() or info.st_mode & 0o022:
            raise ValueError("API record directory must be canonical, service-owned and protected from other writers")
        self.path = path
        try:
            record, before = self._files.open_descriptor(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                                                          max_bytes=limits.max_database_bytes)
            if before.st_uid != os.geteuid() or before.st_mode & 0o077:
                raise ValueError("API record database must be private to the service (chmod600)")
            # A second process/store cannot reinterpret an in-flight mutation.
            # SQLite's own transactions still govern crash-atomic publication.
            fcntl.flock(record.owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._connection = sqlite3.connect(path, check_same_thread=False, isolation_level=None, timeout=0)
            after = path.stat()
            if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
                raise ValueError("API record database identity changed during acquisition")
            self._connection.row_factory = sqlite3.Row
            self._connection.execute("PRAGMA journal_mode=DELETE")
            self._connection.execute("PRAGMA synchronous=FULL")
            pagesize = self._connection.execute("PRAGMA page_size").fetchone()[0]
            pages = limits.max_database_bytes // pagesize
            if pages <= 0 or pages > 4_294_967_294:
                raise ValueError("max_database_bytes is outside the supported SQLite page-count domain")
            self._connection.execute(f"PRAGMA max_page_count={pages}")
            with self._transaction():
                self._connection.execute("CREATE TABLE IF NOT EXISTS api_schema(version INTEGER NOT NULL)")
                versions = self._connection.execute("SELECT version FROM api_schema LIMIT 2").fetchall()
                if not versions:
                    self._connection.execute("INSERT INTO api_schema VALUES(1)")
                elif len(versions) != 1 or versions[0][0] != 1:
                    raise ValueError("unsupported API record database schema")
                self._connection.execute("CREATE TABLE IF NOT EXISTS objects(owner TEXT,id TEXT,kind TEXT,body TEXT,bytes INTEGER,created REAL,expires REAL,generation INTEGER,PRIMARY KEY(owner,id))")
                self._connection.execute("CREATE TABLE IF NOT EXISTS items(owner TEXT,id TEXT,body TEXT,execution TEXT,bytes INTEGER,PRIMARY KEY(owner,id))")
                self._connection.execute("CREATE TABLE IF NOT EXISTS links(owner TEXT,kind TEXT,parent TEXT,ordinal INTEGER,item TEXT,PRIMARY KEY(owner,kind,parent,ordinal),UNIQUE(owner,kind,parent,item))")
                self._connection.execute("CREATE INDEX IF NOT EXISTS item_links ON links(owner,item)")
                self._connection.execute("CREATE INDEX IF NOT EXISTS owned_objects ON objects(owner,kind)")
            self._check_capacity()
        except BaseException as primary:
            rollback(self, primary, self.close)

    def _check(self):
        self.check_health()
        if not self._connection.in_transaction:
            expired = self._connection.execute("SELECT owner,id,generation FROM objects WHERE kind='response' AND expires<=?", (time.time(),)).fetchall()
            if expired:
                with self._transaction():
                    for owner, identifier, generation in expired:
                        self._connection.execute("DELETE FROM objects WHERE owner=? AND id=?", (owner, identifier))
                        self._connection.execute("DELETE FROM links WHERE owner=? AND parent=? AND kind IN ('input','output')", (owner, identifier))
                    self._collect_items()
                for owner, identifier, generation in expired:
                    self._unavailable(owner, "response:" + identifier, generation, "expired")

    def check_health(self):
        """Nonmutating containment predicate for API error projection."""
        if self.pid != os.getpid() or self._closed or self._retiring:
            raise RuntimeError("API record store is retired or belongs to another process")
        if self._failure is not None:
            raise RuntimeError("API record transaction requires containment") from self._failure

    def bind_reference_listeners(self, *, unavailable, committed):
        """Bind catalog ownership once, before handlers; callbacks retain only IDs.

        Each callback must be idempotent for principal/sequence/generation.
        Unavailability releases that association, never shared physical readers.
        """
        if not callable(unavailable) or not callable(committed):
            raise TypeError("both logical reference listeners must be callable")
        with self._lock:
            self.check_health()
            if self._listeners_bound or self._operations:
                raise RuntimeError("logical reference listeners must bind once before operations")
            self._reference_unavailable, self._reference_committed = unavailable, committed
            self._listeners_bound = True

    def _unavailable(self, principal, sequence, generation, reason):
        if self._reference_unavailable is not None:
            try:
                self._reference_unavailable(principal, sequence, generation=generation, reason=reason)
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
                raise

    def _committed(self, principal, sequence, generation):
        if self._reference_committed is not None:
            try:
                self._reference_committed(principal, sequence, generation=generation)
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
                raise

    @contextmanager
    def _transaction(self):
        if self._connection.in_transaction:
            raise RuntimeError("nested logical API transactions are unsupported")
        try:
            self._connection.execute("BEGIN IMMEDIATE")
            yield
            self._connection.execute("COMMIT")
        except BaseException as primary:
            try:
                if self._connection.in_transaction:
                    self._connection.execute("ROLLBACK")
            except BaseException as cleanup:
                if self._failure is None:
                    self._failure = primary
                raise_failures(primary, [cleanup])
            if isinstance(primary, sqlite3.Error) and primary.sqlite_errorcode == sqlite3.SQLITE_FULL:
                raise RecordCapacity("physical API database capacity is exhausted") from primary
            raise

    @contextmanager
    def _admission_boundary(self):
        """Sweep expiry before resolving dependencies, then pin in one snapshot.

        The caller owns the store lock. Nested admission helpers share this
        transaction, so get_item/_check cannot expire a just-resolved dependency
        before its operation token is published.
        """
        self._check()
        if self._connection.in_transaction:
            yield
        else:
            with self._transaction():
                yield

    def _blob(self, value):
        raw = _json(value)
        if len(raw) > self.limits.max_record_bytes:
            raise RecordCapacity("one logical API record exceeds its configured byte allowance")
        return raw

    def _admit_publication(self, body, items, snapshots=(), *, maximum=None):
        raw = self._blob(body) if body is not None else None
        size = len(raw) if raw is not None else 0
        size += sum(len(self._blob(item)) for item in items)
        size += sum(len(self._blob(item)) for item in snapshots)
        if size > (self.limits.max_operation_bytes if maximum is None else maximum):
            raise RecordCapacity("canonical records/snapshots exceed the complete operation byte allowance")
        return raw

    def _decode(self, raw):
        if len(raw) > self.limits.max_record_bytes:
            raise RecordCapacity("stored API record exceeds the active record allowance")
        try:
            value = json.loads(raw)
        except (ValueError, RecursionError) as error:
            if self._failure is None:
                self._failure = error
            raise RuntimeError("stored logical API record is invalid JSON") from error
        self.check_materialized(value)
        return value

    def check_materialized(self, *values):
        """Bound accepted Python record views; serialized parse/transient bounds remain separate.

        This accounts owned Python trees, including container overhead and
        shared identities exactly once. It is not a process/allocator RSS cap.
        """
        size, pending, seen = 0, list(values), set()
        while pending:
            part = pending.pop()
            if id(part) in seen:
                continue
            seen.add(id(part))
            size += sys.getsizeof(part)
            if size > self.limits.max_materialized_bytes:
                raise RecordCapacity("materialized API record exceeds its configured memory allowance")
            if isinstance(part, dict):
                pending.extend(part.keys())
                pending.extend(part.values())
            elif isinstance(part, (list, tuple)):
                pending.extend(part)
        return size

    def _row(self, principal, identifier, kind):
        row = self._connection.execute("SELECT * FROM objects WHERE owner=? AND id=? AND kind=?",
                                       (principal, identifier, kind)).fetchone()
        if row is None or row["expires"] is not None and row["expires"] <= time.time():
            raise RecordNotFound("the requested " + kind + " is unavailable")
        if type(row["generation"]) is not int or not 0 <= row["generation"] < 1 << 63:
            error = RuntimeError("stored logical generation is outside its signed-int64 identity domain")
            if self._failure is None:
                self._failure = error
            raise error
        return row

    @staticmethod
    def _next_generation(row):
        if row["generation"] == (1 << 63) - 1:
            raise RecordCapacity("logical generation identity is exhausted")
        return row["generation"] + 1

    def _append_offset(self, principal, conversation, count):
        if not count:
            return 0
        last = self._connection.execute("SELECT max(ordinal) FROM links WHERE owner=? AND kind='conversation' AND parent=?", (principal, conversation)).fetchone()[0]
        if last is None:
            return 0
        if type(last) is not int or not 0 <= last < 1 << 63:
            error = RuntimeError("stored item ordinal is outside its signed-int64 domain")
            if self._failure is None:
                self._failure = error
            raise error
        if last + count >= 1 << 63:
            raise RecordCapacity("conversation item ordinal is exhausted")
        return last + 1

    def _usage(self, principal=None):
        where, args = (" WHERE owner=?", (principal,)) if principal is not None else ("", ())
        counts = {row[0]: row[1] for row in self._connection.execute(
            "SELECT kind,count(*) FROM objects" + where + " GROUP BY kind", args)}
        item_count, item_bytes = self._connection.execute("SELECT count(*),coalesce(sum(bytes),0) FROM items" + where, args).fetchone()
        body_bytes = self._connection.execute("SELECT coalesce(sum(bytes),0) FROM objects" + where, args).fetchone()[0]
        return counts.get("response", 0), counts.get("conversation", 0), item_count, body_bytes + item_bytes

    def _check_capacity(self, principal=None, *, response=0, conversation=0, items=0, size=0, exclude=None):
        limits = self.limits
        for owner in (None, principal) if principal is not None else (None,):
            used = self._usage(owner)
            active = [op for op in self._operations.values() if op is not exclude and not op.published and not op.read_only
                      and (owner is None or op.principal == owner)]
            reserved = (sum(op.stored for op in active), sum(op.conversation_reservation for op in active), sum(op.item_reservation for op in active),
                        sum(op.byte_reservation for op in active))
            maximum = ((limits.max_responses, limits.max_conversations, limits.max_items, limits.max_total_bytes)
                       if owner is None else (limits.max_principal_responses, limits.max_principal_conversations,
                                             limits.max_principal_items, limits.max_principal_bytes))
            if any(actual + reservation + added > cap for actual, reservation, added, cap in
                   zip(used, reserved, (response, conversation, items, size), maximum)):
                raise RecordCapacity("logical API record capacity is exhausted")

    def _write_object(self, principal, identifier, kind, raw, *, created, generation=0, expires=None):
        self._connection.execute("INSERT INTO objects VALUES(?,?,?,?,?,?,?,?)",
            (principal, identifier, kind, raw, len(raw), created, expires, generation))

    def _write_items(self, principal, items, snapshots=None):
        for item in items:
            raw = self._blob(item)
            execution = (snapshots or {}).get(item["id"])
            execution = self._blob(execution) if execution is not None else None
            previous = self._connection.execute("SELECT body,execution FROM items WHERE owner=? AND id=?", (principal, item["id"])).fetchone()
            if previous is not None:
                if previous[0] != raw or execution is not None and previous[1] != execution:
                    raise RequestError("an existing item ID cannot change its logical contents")
            else:
                self._connection.execute("INSERT INTO items VALUES(?,?,?,?,?)", (principal, item["id"], raw, execution, len(raw) + len(execution or "")))

    def _link(self, principal, kind, parent, items, *, offset=0):
        if type(offset) is not int or offset < 0 or items and offset + len(items) > 1 << 63:
            raise RecordCapacity("item ordinal range exceeds signed-int64 storage")
        for ordinal, item in enumerate(items, offset):
            if self._connection.execute("SELECT 1 FROM links WHERE owner=? AND kind=? AND parent=? AND item=?",
                                        (principal, kind, parent, item["id"])).fetchone() is not None:
                raise RequestError("an item ID may occur only once in one item collection")
            self._connection.execute("INSERT INTO links VALUES(?,?,?,?,?)", (principal, kind, parent, ordinal, item["id"]))

    def _history(self, principal, kinds, identifier):
        selected = ",".join("?" for _ in kinds)
        query = " FROM links l JOIN items i ON i.owner=l.owner AND i.id=l.item WHERE l.owner=? AND l.kind IN (" + selected + ") AND l.parent=?"
        args = (principal, *kinds, identifier)
        count, size = self._connection.execute("SELECT count(*),coalesce(sum(i.bytes),0)" + query, args).fetchone()
        if count > self.limits.max_history_items or size > self.limits.max_history_bytes:
            raise RecordCapacity("logical API history exceeds its materialization allowance")
        rows = self._connection.execute("SELECT i.body,i.execution" + query + " ORDER BY CASE l.kind WHEN 'input' THEN 0 ELSE 1 END,l.ordinal", args)
        result, execution = [], []
        for row in rows:
            value = self._decode(row["body"])
            result.append(value)
            execution.append(self._decode(row["execution"]) if row["execution"] is not None else value)
        self.check_materialized(result, execution)
        return tuple(result), tuple(execution)

    def _unclaimed(self, principal, conversation, *, exclude=None):
        if any(op is not exclude and not op.read_only and op.principal == principal and op.conversation_id == conversation
               for op in self._operations.values()):
            raise RecordConflict("the conversation already has an active mutation")

    def create_conversation(self, principal, *, metadata, items, operation=None):
        principal = _owner(principal)
        body = {"id": "conv_" + uuid.uuid4().hex, "object": "conversation", "created_at": int(time.time()), "metadata": metadata}
        with self._lock:
            self._check()
            encoded = self._admit_publication(body, items, operation.snapshots.values() if operation is not None else (),
                                              maximum=operation.byte_reservation if operation is not None else None)
            with self._transaction():
                self._write_items(principal, items, operation.snapshots if operation is not None else None)
                self._check_capacity(principal, conversation=1, size=len(encoded), exclude=operation)
                self._write_object(principal, body["id"], "conversation", encoded, created=body["created_at"])
                self._link(principal, "conversation", body["id"], items)
            if operation is not None:
                operation.outcome = "completed"
        return body

    def get_conversation(self, principal, identifier):
        with self._lock:
            self._check()
            return self._decode(self._row(_owner(principal), identifier, "conversation")["body"])

    def update_conversation(self, principal, identifier, metadata):
        principal = _owner(principal)
        with self._lock:
            self._check()
            self._unclaimed(principal, identifier)
            with self._transaction():
                row = self._row(principal, identifier, "conversation")
                generation = self._next_generation(row)
                body = self._decode(row["body"])
                body["metadata"] = metadata
                raw = self._admit_publication(body, ())
                self._connection.execute("UPDATE objects SET body=?,bytes=?,generation=? WHERE owner=? AND id=?", (raw, len(raw), generation, principal, identifier))
                self._check_capacity(principal)
        return body

    def delete_conversation(self, principal, identifier):
        principal = _owner(principal)
        with self._lock:
            self._check()
            self._unclaimed(principal, identifier)
            with self._transaction():
                self._row(principal, identifier, "conversation")
                # Deliberately retain item rows AND links. This is the standard
                # object deletion, not item/checkpoint/media destruction.
                self._connection.execute("DELETE FROM objects WHERE owner=? AND id=? AND kind='conversation'", (principal, identifier))
        return {"id": identifier, "object": "conversation.deleted", "deleted": True}

    def add_items(self, principal, identifier, items, *, operation=None):
        principal = _owner(principal)
        with self._lock:
            self._check()
            self._unclaimed(principal, identifier, exclude=operation)
            self._admit_publication(None, items, operation.snapshots.values() if operation is not None else (),
                                    maximum=operation.byte_reservation if operation is not None else None)
            with self._transaction():
                generation = self._next_generation(self._row(principal, identifier, "conversation"))
                offset = self._append_offset(principal, identifier, len(items))
                self._write_items(principal, items, operation.snapshots if operation is not None else None)
                self._link(principal, "conversation", identifier, items, offset=offset)
                self._connection.execute("UPDATE objects SET generation=? WHERE owner=? AND id=?", (generation, principal, identifier))
                self._check_capacity(principal, exclude=operation)
            if operation is not None:
                operation.outcome = "completed"
        return items

    def list_items(self, principal, identifier, *, after=None, limit=20, order="desc"):
        principal = _owner(principal)
        with self._lock:
            self._check()
            self._row(principal, identifier, "conversation")
            return self._page(principal, "conversation", identifier, after=after, limit=limit, order=order)

    def _page(self, principal, kind, identifier, *, after, limit, order):
        if type(limit) is not int or not 1 <= limit <= 100 or order not in {"asc", "desc"}:
            raise RequestError("item pagination requires limit1..100 and order asc or desc")
        args = [principal, kind, identifier]
        boundary = ""
        if after is not None:
            row = self._connection.execute("SELECT ordinal FROM links WHERE owner=? AND kind=? AND parent=? AND item=? ORDER BY ordinal DESC LIMIT 1", (principal, kind, identifier, after)).fetchone()
            if row is None:
                raise RecordNotFound("the pagination item is unavailable")
            boundary = " AND l.ordinal " + (">" if order == "asc" else "<") + " ?"
            args.append(row[0])
        args.append(limit + 1)
        rows = self._connection.execute("SELECT i.body FROM links l JOIN items i ON i.owner=l.owner AND i.id=l.item WHERE l.owner=? AND l.kind=? AND l.parent=?" + boundary + " ORDER BY l.ordinal " + order + " LIMIT ?", args).fetchall()
        values = [self._decode(row[0]) for row in rows[:limit]]
        self.check_materialized(values)
        return {"object": "list", "data": values, "first_id": values[0]["id"] if values else None,
                "last_id": values[-1]["id"] if values else None, "has_more": len(rows) > limit}

    def get_item(self, principal, identifier):
        with self._lock:
            self._check()
            row = self._connection.execute("SELECT body FROM items WHERE owner=? AND id=?", (_owner(principal), identifier)).fetchone()
            if row is None:
                raise RecordNotFound("the requested item is unavailable")
            return self._decode(row[0])

    def conversation_item(self, principal, conversation, identifier):
        principal = _owner(principal)
        with self._lock:
            self._check()
            self._row(principal, conversation, "conversation")
            row = self._connection.execute("SELECT 1 FROM links WHERE owner=? AND kind='conversation' AND parent=? AND item=?", (principal, conversation, identifier)).fetchone()
            if row is None:
                raise RecordNotFound("the requested conversation item is unavailable")
            return self.get_item(principal, identifier)

    def delete_item(self, principal, conversation, identifier):
        principal = _owner(principal)
        with self._lock:
            self._check()
            self._unclaimed(principal, conversation)
            with self._transaction():
                self.conversation_item(principal, conversation, identifier)
                generation = self._next_generation(self._row(principal, conversation, "conversation"))
                self._connection.execute("DELETE FROM links WHERE owner=? AND kind='conversation' AND parent=? AND item=?", (principal, conversation, identifier))
                self._connection.execute("UPDATE objects SET generation=? WHERE owner=? AND id=?", (generation, principal, conversation))
                self._collect_items()
            return self.get_conversation(principal, conversation)

    def _collect_items(self, *, exclude=None):
        pinned = {(op.principal, item["id"]) for op in self._operations.values() if op is not exclude
                  for item in op.history + op.added}
        rows = self._connection.execute("SELECT owner,id FROM items i WHERE NOT EXISTS(SELECT 1 FROM links l WHERE l.owner=i.owner AND l.item=i.id)").fetchall()
        for owner, identifier in rows:
            if (owner, identifier) not in pinned:
                self._connection.execute("DELETE FROM items WHERE owner=? AND id=?", (owner, identifier))

    def get(self, identifier, *, principal):
        with self._lock:
            self._check()
            try:
                return self._decode(self._row(_owner(principal), identifier, "response")["body"])
            except RecordNotFound:
                return None

    def delete(self, identifier, *, principal):
        principal = _owner(principal)
        with self._lock:
            self._check()
            with self._transaction():
                row = self._row(principal, identifier, "response")
                self._connection.execute("DELETE FROM objects WHERE owner=? AND id=? AND kind='response'", (principal, identifier))
                self._connection.execute("DELETE FROM links WHERE owner=? AND parent=? AND kind IN ('input','output')", (principal, identifier))
                self._collect_items()
            self._unavailable(principal, "response:" + identifier, row["generation"], "deleted")
        return True

    def begin_response(self, principal, identifier, *, previous_response_id, conversation_id, added, stored, read_only=False, _operation=None):
        principal = _owner(principal)
        if previous_response_id is not None and conversation_id is not None:
            raise RequestError("conversation and previous_response_id cannot be used together")
        operation, retire = _operation, None
        try:
            with self._lock, self._admission_boundary():
                if operation is None:
                    operation = RecordOperation(self, principal, identifier, previous_response_id, conversation_id,
                                                tuple(self.normalize_items(principal, added)), stored)
                    retire = operation.close
                operation.read_only = read_only
                if len(self._operations) >= self.limits.max_active_operations or sum(op.principal == principal for op in self._operations.values()) >= self.limits.max_principal_operations:
                    raise RecordCapacity("logical API operation capacity is exhausted")
                if conversation_id is not None:
                    row = self._row(principal, conversation_id, "conversation")
                    if not read_only:
                        self._unclaimed(principal, conversation_id)
                    operation.base_generation = row["generation"]
                    operation.generation = row["generation"] if read_only else self._next_generation(row)
                    operation.history, operation.execution_history = self._history(principal, ("conversation",), conversation_id)
                elif previous_response_id is not None:
                    parent = self._row(principal, previous_response_id, "response")
                    operation.parent_generation = parent["generation"]
                    operation.history, operation.execution_history = self._history(principal, ("input", "output"), previous_response_id)
                history = operation.history + operation.added
                if len({item["id"] for item in history}) != len(history):
                    raise RequestError("an input item ID is already present in the resolved history")
                if len(history) > self.limits.max_history_items or sum(len(self._blob(item)) for item in history) > self.limits.max_history_bytes:
                    raise RecordCapacity("the complete logical input exceeds the configured history allowance")
                self.check_materialized(operation.history, operation.execution_history, operation.added)
                if not read_only and (stored or conversation_id is not None):
                    self._admit_publication(None, operation.added)
                    operation.item_reservation = len(added) + self.limits.max_output_items
                    operation.byte_reservation = self.limits.max_operation_bytes
                    self._check_capacity(principal, response=int(stored), items=operation.item_reservation,
                                         size=operation.byte_reservation)
                self._operations[operation.token] = operation
            return operation
        except BaseException as primary:
            if retire is not None:
                rollback(operation, primary, retire)
            raise

    def begin_mutation(self, principal, items, *, conversation=None, create=False, _operation=None):
        operation = _operation or RecordOperation(self, _owner(principal), None, None, conversation, tuple(items), False)
        retire = operation.close if _operation is None else None
        try:
            with self._lock, self._admission_boundary():
                self._admit_publication(None, items)
                if len(self._operations) >= self.limits.max_active_operations or sum(op.principal == principal for op in self._operations.values()) >= self.limits.max_principal_operations:
                    raise RecordCapacity("logical API operation capacity is exhausted")
                if conversation is not None:
                    self._row(principal, conversation, "conversation")
                    self._unclaimed(principal, conversation)
                operation.item_reservation = len(items)
                operation.byte_reservation = self.limits.max_operation_bytes
                operation.conversation_reservation = int(create)
                self._check_capacity(principal, conversation=int(create), items=len(items), size=operation.byte_reservation)
                self._operations[operation.token] = operation
            return operation
        except BaseException as primary:
            if retire is not None:
                rollback(operation, primary, retire)
            raise

    def response_operation(self, principal, identifier, *, previous_response_id, conversation_id, added, stored, read_only=False):
        return self._operation_scope(principal, identifier, previous_response_id=previous_response_id,
            conversation_id=conversation_id, items=added, stored=stored, read_only=read_only)

    def mutation(self, principal, items, *, conversation=None, create=False, maximum=None):
        return self._operation_scope(principal, None, conversation_id=conversation, items=items,
                                      stored=False, create=create, maximum=maximum)

    @contextmanager
    def _operation_scope(self, principal, identifier, *, conversation_id, items, stored, previous_response_id=None,
                         read_only=False, create=False, maximum=None):
        operation, retire, primary = None, None, None
        try:
            with self._lock, self._admission_boundary():
                operation = RecordOperation(self, _owner(principal), identifier, previous_response_id, conversation_id,
                                            tuple(self.normalize_items(principal, items, maximum=maximum)), stored)
                # Bind retirement before publishing any claim. Its exact owner
                # remains in this frame through constructor/publication faults.
                retire = operation.close
                if identifier is None:
                    self.begin_mutation(principal, operation.added, conversation=conversation_id, create=create, _operation=operation)
                else:
                    self.begin_response(principal, identifier, previous_response_id=previous_response_id,
                        conversation_id=conversation_id, added=operation.added, stored=stored, read_only=read_only,
                        _operation=operation)
            yield operation
        except BaseException as error:
            primary = error
            raise
        finally:
            if retire is not None:
                try:
                    retire()
                except BaseException as cleanup:
                    failure = cleanup if primary is None else primary
                    try:
                        with self._lock:
                            if dict.get(self._operations, operation.token) is operation and self._failure is None:
                                self._failure = failure
                    except BaseException as inspection:
                        if self._failure is None:
                            self._failure = failure
                        raise_failures(failure, [cleanup, inspection])
                    # A failed cleanup retains its exact operation in the
                    # authoritative map/traceback. Proven-gone claims stay gone.
                    raise_failures(failure, [cleanup])

    def normalize_items(self, principal, values, *, maximum=None):
        """Own supported standard input items; item references resolve only in this trust domain."""
        from tensorfold.server.responses_translate import messages

        principal = _owner(principal)
        maximum = self.limits.max_input_items if maximum is None else min(maximum, self.limits.max_input_items)
        if isinstance(values, str):
            values = [{"role": "user", "content": values}]
        if not isinstance(values, list) or len(values) > maximum:
            raise RequestError("items must be a list within the declared input item allowance")
        result = []
        for value in values:
            if not isinstance(value, dict):
                raise RequestError("each input item must be an object")
            if value.get("type") == "item_reference":
                if not isinstance(value.get("id"), str):
                    raise RequestError("item_reference requires an item id")
                item = self.get_item(principal, value["id"])
            else:
                # Serialize once before ownership/publication. The same bounded
                # codec is used for disk and request-owned immutable snapshots.
                item = self._decode(self._blob(value))
                messages([item], allow_video=True)  # wire validation; actual capability gates at execution/capture
                kind = item.setdefault("type", "message")
                prefix = {"message": "msg", "function_call": "fc", "function_call_output": "fco", "reasoning": "rs"}[kind]
                if "id" not in item:
                    item["id"] = prefix + "_" + uuid.uuid4().hex
                identifier = item["id"]
                if not isinstance(identifier, str) or not identifier:
                    raise RequestError("an item id must be a nonempty string")
                try:
                    identifier.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise RequestError("item identifiers must be valid UTF-8 strings") from error
                item.setdefault("status", "completed")
                if kind == "message" and isinstance(item["content"], str):
                    part = "output_text" if item["role"] == "assistant" else "input_text"
                    item["content"] = [{"type": part, "text": item["content"]}]
            result.append(item)
        if len({item["id"] for item in result}) != len(result):
            raise RequestError("an item may appear only once in one input")
        if sum(len(self._blob(item)) for item in result) > self.limits.max_history_bytes:
            raise RecordCapacity("logical input items exceed the configured history byte allowance")
        return result

    def response_items(self, principal, identifier, *, after=None, limit=20, order="desc"):
        with self._lock:
            self._check()
            self._row(_owner(principal), identifier, "response")
            return self._page(principal, "input", identifier, after=after, limit=limit, order=order)

    def close(self):
        if self.pid != os.getpid():
            raise RuntimeError("API record store belongs to another process")
        with self._lock:
            if self._closed:
                return
            if self._operations:
                raise RuntimeError("API record operations still own their logical mutations")
            self._retiring = True
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            errors = self._files.drain()
            if not self._files.retired and not errors:
                errors = [RuntimeError("API database file owner did not retire")]
            raise_failures(None, errors)
            self._closed = True


class RecordOperation:
    """A bounded logical snapshot/mutation lease, independent of model retention."""
    def __init__(self, store, principal, identifier, parent, conversation, added, stored):
        self.store, self.principal, self.response_id = store, principal, identifier
        self.parent_response_id, self.conversation_id = parent, conversation
        self.parent_generation = None
        self.added, self.stored, self.token = added, bool(stored), object()
        self.history, self.base_generation, self.generation = (), 0, 1
        self.execution_history = ()
        self.snapshots = {}
        self.conversation_reservation = 0
        self.item_reservation = self.byte_reservation = 0
        self.outcome, self.closed = None, False
        self.read_only = False

    @property
    def published(self):
        return self.outcome is not None

    @property
    def committed(self):
        return self.outcome == "completed"

    @property
    def logical_sequence_id(self):
        return ("conversation:" + self.conversation_id) if self.conversation_id is not None else ("response:" + self.response_id)

    @property
    def target_generation(self):
        return None if self.read_only else self.generation

    @property
    def logical_retained(self):
        """Logical record intent, never an implicit computed-backing promise."""
        return not self.read_only and (self.stored or self.conversation_id is not None)

    @contextmanager
    def execution_scope(self):
        token = _EXECUTION.set(self)
        try:
            yield self
        finally:
            _EXECUTION.reset(token)

    def output_budget(self, base):
        return _OutputBudget(self, base)

    def capture_media(self, app):
        from tensorfold.server.api_record_media import capture_media
        capture_media(self, app)
        self.store.check_materialized(self.history, self.execution_history, self.added, self.snapshots)

    def execution_input(self):
        return self.execution_history + tuple(self.snapshots.get(item["id"], item) for item in self.added)

    def commit(self, response):
        self._publish(response, failed=False)

    def fail(self, response):
        # A failure reporting a later transport/catalog problem must not
        # overwrite an already durable completed model response.
        if not self.published:
            self._publish(response, failed=True)

    def _publish(self, response, *, failed):
        store, principal = self.store, self.principal
        if self.read_only:
            raise RuntimeError("a read-only logical snapshot cannot publish a generated response")
        output = tuple(response["output"])
        if len(output) > store.limits.max_output_items:
            raise RecordCapacity("response output item count exceeds the declared record allowance")
        identifiers = [item["id"] for item in self.history + self.added + output]
        if len(set(identifiers)) != len(identifiers):
            raise RequestError("an output item ID must be distinct from its complete input history")
        if not self.stored and (failed or self.conversation_id is None):
            self.outcome = "failed" if failed else "completed"
            return
        items = self.added + output
        body = store._admit_publication(response if self.stored else None, items, self.snapshots.values(), maximum=self.byte_reservation)
        with store._lock:
            store._check()
            if self.closed or store._operations.get(self.token) is not self or self.published:
                raise RuntimeError("logical API operation is retired or already published")
            with store._transaction():
                if self.conversation_id is not None and not failed:
                    row = store._row(principal, self.conversation_id, "conversation")
                    if row["generation"] != self.base_generation:
                        raise RecordConflict("conversation changed before response publication")
                store._write_items(principal, items, self.snapshots)
                if self.stored:
                    retention = store.limits.response_retention_seconds
                    expires = None if retention is None else time.time() + retention
                    store._write_object(principal, self.response_id, "response", body, created=response["created_at"], expires=expires, generation=self.generation)
                    store._link(principal, "input", self.response_id, self.history + self.added)
                    store._link(principal, "output", self.response_id, output)
                if self.conversation_id is not None and not failed:
                    offset = store._append_offset(principal, self.conversation_id, len(items))
                    store._link(principal, "conversation", self.conversation_id, items, offset=offset)
                    store._connection.execute("UPDATE objects SET generation=? WHERE owner=? AND id=?", (self.generation, principal, self.conversation_id))
                store._check_capacity(principal, exclude=self)
            self.outcome = "failed" if failed else "completed"
            if failed:
                return
            if self.conversation_id is not None:
                store._committed(principal, self.logical_sequence_id, self.generation)
            if self.stored:
                store._committed(principal, "response:" + self.response_id, self.generation)

    def close(self):
        with self.store._lock:
            if self.closed:
                return
            try:
                if self.store._operations.get(self.token) is self:
                    aborting = (self.response_id is not None and (self.stored or self.conversation_id is not None)
                                and not self.read_only and not self.committed)
                    if aborting and self.conversation_id is not None:
                        # Spend the admitted target even when no item is
                        # appended. A later attempt must not reuse an aborted
                        # generation while delayed catalog work can still arrive.
                        with self.store._transaction():
                            row = self.store._row(self.principal, self.conversation_id, "conversation")
                            if row["generation"] == self.base_generation:
                                self.store._connection.execute("UPDATE objects SET generation=? WHERE owner=? AND id=?",
                                    (self.generation, self.principal, self.conversation_id))
                            elif row["generation"] != self.generation:
                                raise RuntimeError("aborted conversation generation no longer owns its target")
                    if aborting:
                        self.store._unavailable(self.principal, self.logical_sequence_id, self.generation, "operation_aborted")
                        if self.stored and self.conversation_id is not None:
                            self.store._unavailable(self.principal, "response:" + self.response_id, self.generation, "operation_aborted")
                    with self.store._transaction():
                        self.store._collect_items(exclude=self)
            except BaseException as error:
                if self.store._failure is None:
                    self.store._failure = error
                raise
            self.store._operations.pop(self.token, None)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, kind, error, traceback):
        if error is not None:
            rollback(self, error, self.close)
        self.close()


class _OutputBudget:
    """Incremental JSON-string accounting before retaining generated text.

    A retained response stores output in its response JSON and immutable items.
    Lengths are exact ASCII JSON bytes, with per-item status completion only
    shortening in_progress. Final usage/diagnostics are checked at publication.
    """
    def __init__(self, operation, base):
        self.operation, self.limits = operation, operation.store.limits
        self.base = len(_json(base))
        self.added = sum(len(_json(item)) for item in operation.added) + sum(len(_json(item)) for item in operation.snapshots.values())
        self.sizes = {}
        self.total = 0
        if self.base > self.limits.max_record_bytes or (operation.byte_reservation and
                (self.base if operation.stored else 0) + self.added > operation.byte_reservation):
            raise RecordCapacity("response envelope exceeds its declared logical output allowance")

    def _check(self, identifier, size, *, new=False):
        count = len(self.sizes) + int(new)
        total = self.total - self.sizes.get(identifier, 0) + size
        response = self.base + total + max(0, count - 1)
        stored = response if self.operation.stored else 0
        item_bytes = self.added + total if self.operation.byte_reservation else 0
        if count > self.limits.max_output_items or max(size, response) > self.limits.max_record_bytes or (self.operation.byte_reservation and stored + item_bytes > self.operation.byte_reservation):
            raise RecordCapacity("generated response exceeds its declared logical output allowance")
        self.sizes[identifier], self.total = size, total

    def item(self, item):
        self._check(item["id"], len(_json(item)), new=True)

    def text(self, item, text):
        self._check(item["id"], self.sizes[item["id"]] + len(_json(text)) - 2)
