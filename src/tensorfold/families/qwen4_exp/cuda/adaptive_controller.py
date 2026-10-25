"""Bounded SM70 execution plans over an unchanged resident-session promise.

The allocated envelope is the safe static plan. Until same-profile measurements
justify another plan, adaptation only schedules actual work inside that envelope;
it neither changes numerical formats nor infers that idle histories are free.
The TP leader publishes the complete plan, including its fair request order.
"""

from __future__ import annotations

import threading
import hashlib
import json
import re
from contextlib import contextmanager

from tensorfold.cleanup import raise_failures

IDENTITY_DIGESTS = frozenset({"target", "draft", "tokenizer", "codec", "numerics", "source",
                             "kernel_build", "runtime", "topology", "retention", "target_weight_encoding", "placement"})
IDENTITY_FIELDS = IDENTITY_DIGESTS | {"context_tokens", "resident_sessions", "history_layout", "method"}
WORK_FIELDS = frozenset({"active_requests", "decode_requests", "prefill_requests", "max_history_tokens",
                         "history_tokens", "retained_sessions", "pending_prompt_tokens", "pending_arrivals",
                         "indexer_work", "expert_group_max"})
EXECUTION_FIELDS = frozenset({"row_budget", "proposal_limit", "prefill_tokens"})
EVIDENCE_ROLES = frozenset({"correctness", "resources", "performance"})
MAX_PROFILES = 256
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _digest(value):
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _interval(value):
    if type(value) is int and value >= 0:
        return value, value
    if (isinstance(value, dict) and set(value) == {"min", "max"}
            and type(value["min"]) is int and type(value["max"]) is int
            and 0 <= value["min"] <= value["max"]):
        return value["min"], value["max"]
    raise ValueError("a measured workload axis needs a nonnegative point or explicit inclusive min/max bounds")


def _covers(region, work):
    for name, observed in work.items():
        declared = region[name]
        if name in ("indexer_work", "expert_group_max") and declared == {"coverage": "all-supported"}:
            continue
        if name == "pending_arrivals":
            if observed != declared:
                return False
        else:
            lower, upper = _interval(declared)
            if observed is None or not lower <= observed <= upper:
                return False
    return True


def _overlaps(left, right):
    for name in WORK_FIELDS:
        if name == "pending_arrivals":
            if left[name] != right[name]:
                return False
        elif (left[name] == {"coverage": "all-supported"}
              or right[name] == {"coverage": "all-supported"}):
            continue
        else:
            low, high = _interval(left[name]), _interval(right[name])
            if low[1] < high[0] or high[1] < low[0]:
                return False
    return True


class AdaptiveController:
    def __init__(self, *, slots: int, row_budget: int, proposal_limit: int, prefill_tokens: int):
        self.maximum = {"row_budget": row_budget, "proposal_limit": proposal_limit,
                        "prefill_tokens": prefill_tokens, "session_ceiling": slots}
        self.slots = slots
        if type(slots) is not int or slots < 1:
            raise ValueError("resident slots must be a positive integer")
        self._validate(self.maximum)
        self._lock = threading.Lock()
        self.configured = dict(self.maximum)
        self.effective = dict(self.maximum)
        self.pending = None
        self.epoch = 0
        self.last_served = None
        self.steps = 0
        self.rounds = 0
        self.round_observation = {}
        self.observation = {}
        self.total_committed_tokens = self.total_prefill_tokens = self.total_target_rows = 0
        from .execution_observations import ExecutionObservations

        self.work_observations = ExecutionObservations(proposal_limit)
        self.identity = None
        self.binding_invalidated_reason = None
        self.evidence = {}
        self.profiles = {}
        self.selected_profile = None
        self.profile_table_digest = None
        self.migration = None
        self.applied_migration = None
        self._profile_reservation = None
        self._deferred_migration = None

    def reset_observations(self):
        """After synthetic startup work retires, begin the serving measurement plane."""
        with self._lock:
            self.steps = self.rounds = 0
            self.observation, self.round_observation = {}, {}
            self.total_committed_tokens = self.total_prefill_tokens = self.total_target_rows = 0
            self.work_observations.reset()

    def _validate(self, values):
        for name, value in values.items():
            minimum = 0 if name == "proposal_limit" else 1
            if type(value) is not int or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
            if value > self.maximum[name]:
                raise ValueError(f"{name} exceeds the allocated maximum {self.maximum[name]}")
        if values["row_budget"] < values["proposal_limit"] + 1:
            raise ValueError("row_budget must hold an anchor and the selected proposals")

    def request(self, *, prefill_tokens=None, proposal_limit=None, row_budget=None, session_ceiling=None):
        """Queue a coalesced change; no CUDA work or live-session state is touched."""
        changes = {name: value for name, value in (("prefill_tokens", prefill_tokens),
                   ("proposal_limit", proposal_limit), ("row_budget", row_budget),
                   ("session_ceiling", session_ceiling)) if value is not None}
        if not changes:
            raise ValueError("at least one execution setting is required")
        with self._lock:
            values = {**self.configured, **changes}
            self._validate(values)
            self.configured = values
            self.pending = dict(values) if values != self.effective else None
            return self._record()

    def boundary(self):
        """Apply supported knobs at the single coordinator's batch boundary."""
        with self._lock:
            if self.pending is not None:
                self.effective = self.pending
                self.pending = None
            return dict(self.effective)

    def freeze_boundary(self, settings=None):
        """Freeze effective settings for every ordered TP operation, without a new plan epoch."""
        with self._lock:
            values = dict(self.effective) if settings is None else dict(settings)
            if set(values) != set(self.maximum):
                raise ValueError("boundary settings differ from the allocated execution contract")
            self._validate(values)
            return {"epoch": self.epoch, "settings": values}

    def apply_boundary(self, boundary):
        """Publish the coordinator's validated settings before any admission/model work."""
        if (not isinstance(boundary, dict) or set(boundary) != {"epoch", "settings"}
                or type(boundary["epoch"]) is not int or not isinstance(boundary["settings"], dict)
                or set(boundary["settings"]) != set(self.maximum)):
            raise ValueError("invalid tensor-parallel execution boundary")
        self._validate(boundary["settings"])
        with self._lock:
            if boundary["epoch"] != self.epoch:
                raise ValueError("execution boundary epoch differs from the ordered plan stream")
            # Request threads may already have queued another configuration.
            # This operation owns only its frozen effective values, never that
            # newer configured/pending request or the next plan's epoch.
            self.effective = dict(boundary["settings"])

    def freeze(self, sessions, *, workload=None):
        """Issue one globally ordered plan; peers must not choose from local load."""
        if workload is not None:
            self._work(workload)
        with self._lock:
            if self.pending is not None:
                self.effective = self.pending
                self.pending = None
            values = dict(self.configured)
            selected = None if workload is None else next(
                (profile for profile in self.profiles.values() if _covers(profile["workload"], workload)), None)
            if selected is not None and all(selected["execution"][name] <= values[name]
                                            for name in EXECUTION_FIELDS):
                values.update(selected["execution"])
                profile_id = selected["id"]
            else:
                profile_id = None
            migration = None if self.migration is None else {
                key: self.migration[key] for key in ("requested_degree", "request_epoch", "pending")}
            if migration is not None and self.migration["blocked"] is not None:
                migration["pending"] = False  # A blocked copy waits for an explicit retry, never every decode step.
        order = list(sessions)
        if len(order) != len(set(order)) or len(order) > self.slots:
            raise ValueError("an execution plan requires distinct admitted session IDs")
        if self.last_served is not None:
            after = next((index for index, sid in enumerate(order) if sid > self.last_served), len(order))
            order = order[after:] + order[:after]
        return {"epoch": self.epoch + 1, **values, "order": order,
                "profile_id": profile_id, "migration": migration}

    def admission_limit(self):
        with self._lock:
            return self.effective["session_ceiling"]

    def bind_identity(self, identity, *, evidence=None):
        """Bind startup-verified content/runtime facts, never an HTTP assertion.

        The startup manifest owner verifies actual artifact bytes and raw proof
        artifacts before calling this method. Public table registration can only
        refer to those already bound evidence digests.
        """
        if not isinstance(identity, dict) or set(identity) != IDENTITY_FIELDS:
            raise ValueError("a complete exact execution-profile identity is required")
        for name in IDENTITY_DIGESTS:
            if not _digest(identity[name]) and not (name == "draft" and identity[name] == "none"):
                raise ValueError(f"execution-profile identity {name} must be a SHA256 content digest")
        if (type(identity["context_tokens"]) is not int or identity["context_tokens"] < 1
                or type(identity["resident_sessions"]) is not int or identity["resident_sessions"] != self.slots
                or identity["history_layout"] not in ("unique", "replicated")
                or identity["method"] not in ("none", "mtp", "dspark", "dflash")):
            raise ValueError("execution-profile identity differs from the allocated context/method contract")
        evidence = {} if evidence is None else evidence
        if not isinstance(evidence, dict) or len(evidence) > 3 * MAX_PROFILES:
            raise ValueError("bound profile evidence exceeds its supported receipt catalogue")
        checked = {}
        for digest, receipt in evidence.items():
            if (not _digest(digest) or not isinstance(receipt, dict)
                    or receipt.get("schema") != "tensorfold.execution-profile-evidence"
                    or type(receipt.get("version")) is not int or receipt["version"] != 1
                    or receipt.get("kind") not in EVIDENCE_ROLES
                    or receipt.get("identity") != identity or receipt.get("qualification") != "live"
                    or receipt.get("verdict") != "passed" or not isinstance(receipt.get("proof"), dict)
                    or not receipt["proof"]):
                raise ValueError("bound profile evidence lacks exact live qualification and proof")
            proof = receipt["proof"]
            if (set(proof) != {"protocol", "record_sha256", "artifacts"}
                    or proof.get("protocol") != "tensorfold.execution-profile-" + receipt["kind"] + "-v1"
                    or not _digest(proof.get("record_sha256"))
                    or not isinstance(proof.get("artifacts"), list) or not 1 <= len(proof["artifacts"]) <= 64):
                raise ValueError("profile proof requires the owned protocol and bounded raw artifacts")
            for artifact in proof["artifacts"]:
                if (not isinstance(artifact, dict) or set(artifact) != {"path", "size", "sha256"}
                        or not isinstance(artifact["path"], str) or not 1 <= len(artifact["path"]) <= 4096
                        or type(artifact["size"]) is not int or artifact["size"] < 1
                        or not _digest(artifact["sha256"])):
                    raise ValueError("profile proof artifact identity is malformed")
            if proof["record_sha256"] not in {artifact["sha256"] for artifact in proof["artifacts"]}:
                raise ValueError("profile proof record must be one of its verified raw artifacts")
            checked[digest] = json.loads(_canonical(receipt))
        with self._lock:
            self.identity = dict(identity)
            self.evidence = checked
            self.profiles = {}
            self.selected_profile = None
            self.profile_table_digest = None
            self.binding_invalidated_reason = None

    def _invalidate_binding(self, reason):
        """Caller owns the controller lock; queued execution controls survive."""
        self.identity = None
        self.evidence = {}
        self.profiles = {}
        self.selected_profile = None
        self.profile_table_digest = None
        self.binding_invalidated_reason = reason

    def invalidate_binding(self, reason):
        """Retire stale proofs after observed facts change, preserving commitments."""
        if type(reason) is not str or not reason:
            raise ValueError("profile binding invalidation needs a nonempty diagnostic reason")
        with self._lock:
            self._invalidate_binding(reason[:512])
            return self._record()

    @contextmanager
    def profile_transaction(self):
        """Reserve the published placement through distributed commit acknowledgment.

        Only the serialized GPU worker enters this scope. The manifest owner
        keeps every rank in it through its final collective commit/abort vote;
        releasing it earlier would permit a leader-only generation change.
        HTTP requests coalesce a deferred target without waiting for file IO or
        another rank. They publish at release and invalidate old-placement
        evidence before the next inference plan can be frozen.
        """
        reservation = object()
        with self._lock:
            if self._profile_reservation is not None:
                raise RuntimeError("an execution-profile transaction is already active")
            self._profile_reservation = reservation
        primary = None
        try:
            yield
        except BaseException as error:
            primary = error
        errors = []
        try:
            with self._lock:
                if self._profile_reservation is not reservation:
                    raise RuntimeError("execution-profile reservation ownership changed")
                self._profile_reservation = None
                deferred = self._deferred_migration
                self._deferred_migration = None
                if deferred is not None:
                    self._request_migration(deferred)
        except BaseException as error:
            errors.append(error)
        raise_failures(primary, errors)

    def _work(self, workload):
        if not isinstance(workload, dict) or set(workload) != WORK_FIELDS:
            raise ValueError("execution profile needs the complete supported workload axes")
        for name, value in workload.items():
            if name == "pending_arrivals":
                valid = type(value) is bool
            else:
                valid = (value is None and name in ("indexer_work", "expert_group_max")) or (
                    type(value) is int and value >= 0)
            if not valid:
                raise ValueError(f"execution profile workload axis {name} is invalid")
        if (workload["active_requests"] > self.slots or workload["retained_sessions"] > self.slots
                or workload["decode_requests"] + workload["prefill_requests"] > workload["active_requests"]):
            raise ValueError("execution-profile workload exceeds resident or active session bounds")

    def _region(self, workload):
        if not isinstance(workload, dict) or set(workload) != WORK_FIELDS:
            raise ValueError("profile region needs every supported workload axis")
        for name, value in workload.items():
            if name == "pending_arrivals":
                if type(value) is not bool:
                    raise ValueError("pending-arrival coverage must be an exact boolean")
            elif name in ("indexer_work", "expert_group_max") and value == {"coverage": "all-supported"}:
                continue
            else:
                _interval(value)
        if any(_interval(workload[name])[1] > self.slots
               for name in ("active_requests", "decode_requests", "prefill_requests", "retained_sessions")):
            raise ValueError("measured workload region exceeds the allocated session capacity")
        return _canonical(workload)

    def replace_verified_profiles(self, identity, *, evidence, table, expected_migration_epoch):
        """Validate a whole replacement before one locked publication.

        The worker's manifest verifier supplies actual content and placement
        facts. A newer administrative migration invalidates that verification;
        its queued request must never be cleared by a slow manifest read.
        """
        staged = AdaptiveController(slots=self.slots, row_budget=self.maximum["row_budget"],
                                    proposal_limit=self.maximum["proposal_limit"],
                                    prefill_tokens=self.maximum["prefill_tokens"])
        staged.bind_identity(identity, evidence=evidence)
        staged.install_profiles(table)
        with self._lock:
            actual_epoch = None if self.migration is None else self.migration["request_epoch"]
            if (actual_epoch != expected_migration_epoch
                    or self.migration is not None and self.migration["pending"]):
                raise ValueError("placement changed or migration is pending during profile verification")
            self.identity, self.evidence, self.profiles = staged.identity, staged.evidence, staged.profiles
            self.profile_table_digest = staged.profile_table_digest
            self.selected_profile = None
            self.binding_invalidated_reason = None
            return self._record()

    def install_profiles(self, table):
        """Register bounded exact-workload winners backed by startup-bound proofs."""
        if (not isinstance(table, dict) or set(table) != {"schema", "version", "identity", "profiles"}
                or table["schema"] != "tensorfold.execution-profiles"
                or type(table["version"]) is not int or table["version"] != 1
                or not isinstance(table["profiles"], list) or not len(table["profiles"]) <= MAX_PROFILES):
            raise ValueError("invalid or oversized execution-profile table")
        with self._lock:
            if self.identity is None or table["identity"] != self.identity:
                raise ValueError("execution-profile identity is unbound or differs from actual startup facts")
            checked, names = {}, set()
            for profile in table["profiles"]:
                if (not isinstance(profile, dict) or set(profile) != {"id", "workload", "execution", "evidence"}
                        or not isinstance(profile["id"], str) or not 1 <= len(profile["id"]) <= 128
                        or profile["id"] in names or not isinstance(profile["execution"], dict)
                        or set(profile["execution"]) != EXECUTION_FIELDS
                        or not isinstance(profile["evidence"], dict) or set(profile["evidence"]) != EVIDENCE_ROLES):
                    raise ValueError("execution-profile record is incomplete or duplicated")
                work = self._region(profile["workload"])
                self._validate({**self.maximum, **profile["execution"]})
                if any(_overlaps(profile["workload"], existing["workload"]) for existing in checked.values()):
                    raise ValueError("measured profile regions overlap without an evidence-backed winner ordering")
                for role, digest in profile["evidence"].items():
                    if not _digest(digest):
                        raise ValueError("execution-profile evidence references must be SHA256 digests")
                    receipt = self.evidence.get(digest)
                    if (receipt is None or receipt["kind"] != role
                            or receipt.get("workload") != profile["workload"]
                            or receipt.get("execution") != profile["execution"]):
                        raise ValueError("execution profile lacks bound exact-scope " + role + " evidence")
                names.add(profile["id"])
                checked[work] = json.loads(_canonical(profile))
            self.profiles = checked
            self.selected_profile = None
            self.profile_table_digest = hashlib.sha256(_canonical(table).encode()).hexdigest()
            return self._record()

    def enable_migration(self, degree):
        if type(degree) is not int or degree not in (1, 2, 4):
            raise ValueError("initial attention degree must be 1, 2 or 4")
        with self._lock:
            self.migration = {"requested_degree": degree, "request_epoch": 0,
                              "pending": False, "blocked": None}
            self.applied_migration = {"requested_degree": degree, "request_epoch": 0, "pending": False}

    def request_migration(self, degree):
        if type(degree) is not int or degree not in (1, 2, 4):
            raise ValueError("attention degree must be 1, 2 or 4")
        with self._lock:
            if self.migration is None:
                raise ValueError("this decoder has no bounded unique-history migration path")
            if self._profile_reservation is not None:
                self._deferred_migration = degree
            else:
                self._request_migration(degree)
            return self._record()

    def _request_migration(self, degree):
        """Caller owns the controller lock; the latest deferred target wins."""
        if degree != self.migration["requested_degree"] or self.migration["blocked"] is not None:
            self.migration = {"requested_degree": degree,
                              "request_epoch": self.migration["request_epoch"] + 1,
                              "pending": True, "blocked": None}
            # A performance proof for an old placement does not qualify
            # a mixed owner table or a new placement. Rebind after it is
            # measured; serving keeps the admitted canonical static plan.
            self._invalidate_binding("attention ownership changed; rebind after stationary-placement verification")

    def migration_status(self, *, request_epoch, pending, blocked=None):
        with self._lock:
            if self.migration is not None and self.migration["request_epoch"] == request_epoch:
                self.migration = {**self.migration, "pending": pending, "blocked": blocked}

    def accept(self, plan):
        """Validate the leader's bounded plan before any state or collective work."""
        if set(plan) != {"epoch", "order", "profile_id", "migration", *self.maximum}:
            raise ValueError("execution plan fields do not match the supported contract")
        values = {name: plan[name] for name in self.maximum}
        self._validate(values)
        if type(plan["epoch"]) is not int or plan["epoch"] != self.epoch + 1:
            raise ValueError("execution plan epoch is stale or out of sequence")
        order = plan["order"]
        if (not isinstance(order, list) or len(order) > self.slots
                or any(type(sid) is not int or sid < 0 for sid in order)
                or len(order) != len(set(order))):
            raise ValueError("execution plan session order is invalid")
        profile = plan["profile_id"]
        if profile is not None and (not isinstance(profile, str) or not 1 <= len(profile) <= 128):
            raise ValueError("execution plan profile identity is invalid")
        migration = plan["migration"]
        if migration is not None:
            if (self.migration is None or not isinstance(migration, dict)
                    or set(migration) != {"requested_degree", "request_epoch", "pending"}
                    or type(migration["requested_degree"]) is not int or migration["requested_degree"] not in (1, 2, 4)
                    or type(migration["pending"]) is not bool
                    or type(migration["request_epoch"]) is not int
                    or migration["request_epoch"] < self.applied_migration["request_epoch"]):
                raise ValueError("execution plan migration generation is stale or invalid")
        with self._lock:
            self.effective = values
            self.epoch = plan["epoch"]
            self.selected_profile = profile
            if migration is not None:
                self.applied_migration = dict(migration)
            if migration is not None and migration["request_epoch"] > self.migration["request_epoch"]:
                self.migration = {**migration, "blocked": None}
                self._invalidate_binding("attention ownership changed; rebind after stationary-placement verification")

    def select(self, work, plan):
        """Fair complete-window microbatch; never wait to fill an otherwise ready step."""
        rows = dict(work)
        if len(rows) != len(work) or set(rows).difference(plan["order"]):
            raise ValueError("ready sessions differ from the globally planned session set")
        selected, available = [], plan["row_budget"]
        for sid in plan["order"]:
            if sid not in rows:
                continue
            count = rows[sid]
            if type(count) is not int or not 1 <= count <= plan["proposal_limit"] + 1:
                raise ValueError("a verification window exceeds its selected proposal contract")
            if count > available:
                # Stopping here keeps the skipped oldest request first on the
                # next step; smaller late requests cannot starve a wide window.
                break
            selected.append(sid)
            available -= count
        if selected:
            self.last_served = selected[-1]
        return selected

    def observe(self, *, active, scheduled, target_rows, prefill_tokens, committed_tokens,
                max_history_tokens, elapsed_s):
        with self._lock:
            self.steps += 1
            self.total_target_rows += target_rows
            self.total_prefill_tokens += prefill_tokens
            self.total_committed_tokens += committed_tokens
            self.observation = {"active_requests": active, "scheduled_decode_requests": scheduled,
                                "target_rows": target_rows, "prefill_tokens": prefill_tokens,
                                "committed_tokens": committed_tokens,
                                "max_history_tokens": max_history_tokens, "elapsed_s": elapsed_s}

    def round_started(self):
        with self._lock:
            return self.total_target_rows, self.total_prefill_tokens, self.total_committed_tokens

    def observe_round(self, *, before, elapsed_s, migration_s, outcome):
        with self._lock:
            self.rounds += 1
            self.round_observation = {"elapsed_s": elapsed_s, "migration_s": migration_s, "outcome": outcome,
                                      "target_rows": self.total_target_rows - before[0],
                                      "prefill_tokens": self.total_prefill_tokens - before[1],
                                      "committed_tokens": self.total_committed_tokens - before[2]}

    def _record(self):
        return {"configured": dict(self.configured),
                "pending": None if self.pending is None else dict(self.pending),
                "effective": dict(self.effective), "allocated_maximum": dict(self.maximum),
                "epoch": self.epoch, "steps": self.steps, "rounds": self.rounds,
                "selection": "qualified-workload-region" if self.selected_profile is not None else
                             "bounded-static-envelope-fair-rotation",
                "performance_qualification": "registered-workload-region" if self.selected_profile is not None else "pending",
                "profile_identity": None if self.identity is None else dict(self.identity),
                "binding_invalidated_reason": self.binding_invalidated_reason,
                "profile_transaction_active": self._profile_reservation is not None,
                "deferred_migration": self._deferred_migration,
                "profile_table_sha256": self.profile_table_digest, "registered_profiles": len(self.profiles),
                "selected_profile": self.selected_profile,
                "migration": None if self.migration is None else dict(self.migration),
                "applied_migration": None if self.applied_migration is None else dict(self.applied_migration),
                "last_step": dict(self.observation),
                "timing_scope": "host_wall_including_observed_waits",
                "device_step_elapsed_s": None,
                "last_round": dict(self.round_observation),
                "timing_contract": "last_round includes migration, prefill, decode and host control; last_step is a phase",
                "work_observations": self.work_observations.record(),
                "total_target_rows": self.total_target_rows,
                "total_prefill_tokens": self.total_prefill_tokens,
                "total_committed_tokens": self.total_committed_tokens}

    def record(self):
        with self._lock:
            return self._record()
