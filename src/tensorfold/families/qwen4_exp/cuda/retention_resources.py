"""Global host commitments and additional per-rank retained-state allocations.

These are source payload bounds, not an allocator/RSS or V100 certificate.
The header and loaded frontend stages share the same immutable policy record.
The existing capacity owner supplies physical/cgroup admission; it is never
replaced with a process RSS threshold or a new administrator budget.
"""

from dataclasses import dataclass, asdict
import hashlib
import json
import struct
import sys

from .retention_policy import RetentionPolicy

INITIAL_COHORT_VALUE_BYTES = 8 << 20


def _size(value, name, *, zero=False):
    if type(value) is not int or not (0 if zero else 1) <= value <= sys.maxsize:
        raise ValueError(f"retention {name} must be a {'nonnegative' if zero else 'positive'} host-sized integer")
    return value


def _total(values):
    result = sum(values)
    return _size(result, "resource sum", zero=True)


@dataclass(frozen=True)
class HeaderResources:
    global_host_backing_bytes: int
    global_host_progress_bytes: int
    per_rank_gpu_overlap_bytes: int
    world: int
    global_host_native_data_bytes: int = 0
    global_host_native_control_bytes: int = 0
    global_host_native_rpc_bytes: int = 0

    @property
    def global_host_bytes(self):
        return _total((self.global_host_backing_bytes, self.global_host_progress_bytes,
                       self.global_host_native_data_bytes, self.global_host_native_control_bytes,
                       self.global_host_native_rpc_bytes))

    def record(self):
        return {**asdict(self), "schema": "tensorfold.retention-header-resources-v1",
                "phase": "steady-serving; max with separate loading scratch",
                "evidence": "source-candidate; allocator/device qualification pending"}


@dataclass(frozen=True)
class HostBaseline:
    baseline_host_bytes: int
    materialized_host_bytes: int
    prepaid_native_fixed_payload_bytes: int = 0
    native_state_slots: int | None = None
    native_fd_limit: int | None = None
    prepaid_native_prefix_payload_bytes: int = 0
    native_host_bytes: int = 0

    def __post_init__(self):
        _size(self.baseline_host_bytes, "ordinary steady host obligation", zero=True)
        _size(self.materialized_host_bytes, "owned ordinary host materialization", zero=True)
        _size(self.prepaid_native_fixed_payload_bytes, "prepaid fixed protocol payload", zero=True)
        _size(self.prepaid_native_prefix_payload_bytes, "prepaid prefix protocol payload", zero=True)
        _size(self.native_host_bytes, "native configured host bytes", zero=True)
        if self.native_state_slots is not None:
            _size(self.native_state_slots, "native maximum State count")
        if self.native_fd_limit is not None:
            _size(self.native_fd_limit, "native hard descriptor limit")
        if self.materialized_host_bytes > self.baseline_host_bytes:
            raise ValueError("ordinary materialized credit exceeds its own declared baseline")
        if self.prepaid_native_fixed_payload_bytes + self.prepaid_native_prefix_payload_bytes > self.baseline_host_bytes:
            raise ValueError("prepaid protocol payload exceeds its declared ordinary baseline")


@dataclass(frozen=True)
class StoreFieldGuard:
    """Cohort-owned field capability; no model/Store reference or transport."""
    value_bytes: int | None

    def __post_init__(self):
        if self.value_bytes is not None:
            _size(self.value_bytes, "native value field bytes")

    def key(self, key):
        from tensorfold.cuda.store_fields import validate_store_key

        validate_store_key(key)

    def value(self, value):
        if self.value_bytes is None:
            return
        if type(value) is not str:
            raise ValueError("cohort publication must be an encoded JSON string")
        if len(value) > self.value_bytes:
            raise ValueError("cohort value exceeds the native field byte limit")
        size = len(value) if value.isascii() else len(value.encode("utf-8"))
        if size > self.value_bytes:
            raise ValueError("cohort value exceeds the native field byte limit")

    def received(self, value):
        # Received bytes are a new boundary even when every owned publisher
        # used canonical_runtime_json. Bound before Python decoding/expansion.
        from .execution_identity import PHASE_VALUE_BYTES

        if type(value) is not bytes or len(value) > min(PHASE_VALUE_BYTES, self.value_bytes or PHASE_VALUE_BYTES):
            raise ValueError("received cohort value exceeds its native/application byte limit")


def retention_header_resources(policy, world, *, slots=None):
    """Known reservations before model payload; global RAM is not rank-multiplied."""
    _size(world, "rank count")
    if policy is not None and type(policy) is not RetentionPolicy:
        raise ValueError("retention resources require the actual typed policy")
    ram = policy is not None and policy.tier == "ram"
    if policy is not None:
        policy.validate_world(world)
    # Actual single-rank Engine has no communicator/Store. RAM still owns
    # its local backing/progress buffers; native RPC/daemon roles do not exist.
    if world == 1:
        return HeaderResources(policy.ram_bytes if ram else 0, policy.progress_bytes if ram else 0,
                               policy.overlap_bytes if ram else 0, world)
    control = native_sync_control_payload(world=world, chunk_bytes=policy.chunk_bytes if ram else 0,
                                          ram=ram, slots=slots)
    if not ram:
        return HeaderResources(0, 0, 0, world, 0, control["global_native_control_value_bytes"],
                               control["global_native_rpc_value_bytes"])
    native = native_sync_data_payload(world=world, chunk_bytes=policy.chunk_bytes)
    return HeaderResources(policy.ram_bytes, policy.progress_bytes, policy.overlap_bytes, world,
                           native["global_native_data_payload_bytes"],
                           control["global_native_control_value_bytes"], control["global_native_rpc_value_bytes"])


def native_sync_data_payload(*, world, chunk_bytes):
    """Additional owned data payload for the selected synchronous Store.

    Existing progress pays pinned+ordinary2Wc. Conservatively p active native
    SET calls and q open chunk waves satisfy p<=q<=W. Native roles excluding
    SendBuffer are (W+p+q+1)c: SET string/vector, other GET vectors, stored map
    vectors and ONE serial daemon local. SelectedGCC13 SendBuffer's allocated
    peak is <=c+H+2878, including old/new storage during range insertion.
    Maximize p,q at W; do not silently assume one global publisher.

    This term intentionally excludes other key/object/control/metadata terms;
    those have separate ownership. It is not an allocator/RSS certificate.
    """
    from tensorfold.cuda.store_fields import MAX_KEY_BYTES

    _size(world, "native rank count")
    _size(chunk_bytes, "native chunk maximum")
    header = max(32, MAX_KEY_BYTES + 9, 2878)
    slack = world * (header + 2878 + 1)  # Native heap string terminator; selected runtime14 constructor.
    data = (4 * world + 1) * chunk_bytes
    return {"schema": "tensorfold.sync-store-data-payload-v1", "scope": "global-host",
            "world": world, "chunk_bytes_maximum": chunk_bytes, "open_wave_bound": world,
            "pinned_ordinary_bytes_already_in_progress": 2 * world * chunk_bytes,
            "native_chunk_payload_bytes": data, "sendbuffer_header_capacity_bytes": slack,
            "global_native_data_payload_bytes": _total((data, slack)),
            "source": "Torch04cf2557/libstdc++13; owned synchronous daemon provenance required",
            "remaining_terms": "key/control/metadata/allocator/kernel budgets are distinct"}


def native_sync_control_payload(*, world, chunk_bytes, ram, slots=None):
    """GLOBAL synchronous stored/active control values, excluding backing.

    Counts source-owned values, not allocator RSS. One daemon's current
    handler is paid once across data, worker/actor phases and TP RPC. Fixed
    key/container memory is a distinct contract. Existing encoded/parsed
    manifest backing M is never re-added here.
    """
    from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
    from tensorfold.cuda.store_fields import MAX_KEY_BYTES
    from tensorfold.engine.model_identity import canonical_runtime_json
    from .execution_identity import PHASE_VALUE_BYTES, phase_failure_value_bytes

    _size(world, "control rank count")
    _size(chunk_bytes, "control data chunk", zero=not ram)
    if type(ram) is not bool:
        raise ValueError("native control mode must be the actual RAM capability")
    if slots is not None:
        _size(slots, "control slot count")
    def encoded(value):
        # Only the ASCII literal/int sizing prototypes below reach this local.
        return len(canonical_runtime_json(value, max_bytes=PHASE_VALUE_BYTES))
    true = encoded({"ok": True, "value": True})
    published = encoded({"ok": True, "value": {"session_error": False}})
    envelope = encoded({"ok": True, "value": {"fragment": ""}})
    # These are the existing _phase diagnostic character limits. Worstcase
    # JSON escaping is six bytes/character, including Unicode/control input.
    refusal = phase_failure_value_bytes()
    header = encoded({"ok": True, "value": {"bytes": sys.maxsize, "pieces": sys.maxsize, "sha256": "f" * 64}})
    if slots is None:
        actor = PHASE_VALUE_BYTES
    else:
        pair = len(f"[{sys.maxsize},{sys.maxsize}]")
        actor = encoded({"ok": True, "value": {"complete": []}}) + slots * (pair + 1) - 1
        if actor > PHASE_VALUE_BYTES:
            raise ValueError("retained ready table exceeds the existing phase byte contract")
    width = min(chunk_bytes, (PHASE_VALUE_BYTES - envelope) // 2) if ram else 0
    fragment = 2 * width + envelope if ram else 0
    worker = max(fragment, header, refusal) if ram else 0
    if ram:
        daemon_worker = max(world * header + 2 * world * fragment,
            max(world * (header + true), 3 * world * true, world * refusal)
            + world * max(header, true, refusal))
    else:
        daemon_worker = 0
    daemon_actor = 2 * world * actor + world * (true + published)
    bootstrap = INITIAL_COHORT_VALUE_BYTES
    daemon_startup = world * bootstrap + world * (refusal + 2 * true)
    serving = daemon_worker + daemon_actor + 3 * world * (worker + actor)
    startup = daemon_startup + 3 * world * bootstrap
    peers = world - 1
    command = MAX_MESSAGE_BYTES if peers else 0
    ack = MAX_MESSAGE_BYTES if peers else 0  # Actual publisher/receiver enforce the same application cap.
    # Per-caller exclusivity: root publishes its vector batch OR collects
    # ACKs; each follower receives a command OR publishes its ACK. Map values
    # persist until the publishing side's ACK retirement.
    rpc = (max((2 * peers + 1) * command, peers * ack)
           + peers * max(command, 3 * ack) + peers * (command + ack)) if peers else 0
    key_header = max(32, MAX_KEY_BYTES + 9, 2878) + 2878
    # Data already pays W worker SendBuffer/NUL slack. Only actor slack is
    # additional here; no worker exists when RAM is disabled.
    slack = world * key_header + max(world, 2 * peers)
    # The base rendezvous UID stays stored for the communicator lifetime.
    # Every rank's configured-backend and loading-ready rows are also fixed
    # names, overwritten rather than appended on profile replacement.
    rendezvous_values = 128 + world * (len("nccl\0owned-unique-store-v1") + 1)
    fixed_values = (world * world + 37 * world + 33 if ram else 0) + 4 * world + rendezvous_values
    handler_delta = max(0, max(worker, actor, bootstrap, command, ack) - (chunk_bytes if ram else 0))
    control = _total((max(serving, startup), slack, fixed_values, handler_delta))
    return {"schema": "tensorfold.sync-store-control-value-v1", "scope": "global-host",
            "worker_frame_bytes": worker, "actor_frame_bytes": actor, "rpc_command_bytes": command,
            "rpc_ack_bytes": ack, "daemon_worker_value_bytes": daemon_worker,
            "daemon_actor_value_bytes": daemon_actor, "active_serving_value_bytes": 3 * world * (worker + actor),
            "constructor_value_bytes": startup, "shared_daemon_handler_delta_bytes": handler_delta,
            "persistent_rendezvous_value_bytes": rendezvous_values,
            "global_native_control_value_bytes": control, "global_native_rpc_value_bytes": _total((rpc,)),
            "excluded": "already reserved manifest backing; fixed key/container and Python RPC ownership are separate",
            "source": "Torch04cf2557 synchronous backend; native-table-assist source ownership proof"}


def native_sync_prefix_payload(*, world, slots):
    """Additional GLOBAL TP stop-boundary values; keys are paid separately.

    A round has at most one decode group plus one joined group per State.
    Count is a host-sized nonnegative integer and the pair's Boolean encodes
    no longer than ``false``. One source multi_set is active at a time;
    previous groups may remain stored until their enclosing operation ACK.
    The existing RPC term already pays the shared daemon handler up to96MiB.
    """
    from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES

    _size(world, "prefix rank count")
    _size(slots, "prefix maximum State count")
    peers = world - 1
    empty = len(json.dumps({"decisions": [], "failed": False}, separators=(",", ":")))
    pair = len(f"[{sys.maxsize},false]")
    frame = empty + slots * (pair + 1) - 1
    stored = (slots + 1) * peers * frame
    active = (3 * peers + 1) * frame if peers else 0
    handler = max(0, frame - MAX_MESSAGE_BYTES) if peers else 0
    return {"schema": "tensorfold.sync-store-prefix-value-v1", "scope": "global-host",
            "world": world, "maximum_state_slots": slots, "frame_bytes": frame,
            "stored_value_bytes": stored, "active_native_value_bytes": active,
            "shared_daemon_handler_delta_bytes": handler,
            "global_native_prefix_payload_bytes": _total((stored, active, handler)),
            "source": "TP._prefix_decisions/_round/_joined; synchronous daemon; separate fixed key budget"}


def native_sync_fixed_payload(*, world, ram, slots, fd_limit=None, host_bytes=0):
    """Known GLOBAL key/container payload for the sealed synchronous ABI.

    Torch04cf2557/GCC13/runtime14 uses 32-byte strings, 24-byte vectors, cached
    72-byte string-key nodes, and eight-byte bucket pointers. These are
    allocation requests/object layouts, not allocator usable sizes or RSS.
    ``slots`` is the communicator's maximum admitted State count: retained
    hash buckets survive erasure and a smaller current profile is no credit.
    The runtime ABI gate owns applicability of these measured type layouts.
    Native callers pass len(comm.store.host.encode("utf-8")); zero is only
    the independent Store-free reference input, not a native-host estimate.
    """
    from tensorfold.cuda.store_fields import MAX_KEY_BYTES

    _size(world, "fixed native rank count")
    _size(slots, "fixed native maximum State count")
    _size(host_bytes, "native configured hostname bytes", zero=True)
    if type(ram) is not bool:
        raise ValueError("native fixed mode must be the actual RAM capability")
    if fd_limit is None:
        import resource

        fd_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
    _size(fd_limit, "finite native hard descriptor limit")
    if fd_limit <= 3:
        raise ValueError("native hard descriptor limit cannot hold the listener/control pipe and a client")
    # Every retained accepted Socket still owns a descriptor. Three immutable
    # descriptors belong to this daemon's listener and two control-pipe ends.
    # The HARD limit remains valid if the soft limit is raised during service.
    accepted = fd_limit - 3
    socket_slots = 1 << (accepted - 1).bit_length()
    poll_slots = 1 << (accepted + 1).bit_length()
    peers = world - 1
    # Base W, worker W, wake control W, senders(W-1)+receiver1. The single
    # endpoint nonce client precedes RAM clones; without RAM it adds one.
    clients = 4 * world if ram else world + 1
    callers = (2 * world + 1) if ram else world
    # Each actor/worker WAIT carries <=W keys; the leader wake WAIT has one.
    # pybind owns one array and TCPStore::wait another prefixed array.
    batch_keys = 2 * world * world + 1 if ram else world * world
    # _round has one alternative decode boundary plus <=slots joined prompts.
    # _pass has only joins. All these keys retire before the enclosing ACK.
    prefix_keys = (slots + 1) * peers
    rpc_keys = 2 * peers  # Command and ACK, both publisher-retired in FIFO.
    bootstrap_keys = 2 * world + 2  # UID, backend W, loading W, endpoint nonce.
    actor_keys = 8 * world  # <=4W outcomes plus two retirement waves of2W.
    worker_keys = 8 * world if ram else 0
    data_keys = world * world + world if ram else 0  # ACKs and open chunks.
    wake_keys = 2 * world + 2 if ram else 0
    map_keys = _total((actor_keys, worker_keys, data_keys, wake_keys,
                       rpc_keys, prefix_keys, bootstrap_keys))
    # A Store may outlive a RAM profile. Its value/wait bucket highwater is
    # bounded using the same W/maximum-slots RAM region even when RAM is off.
    map_highwater = world * world + 23 * world + 2 + prefix_keys
    wait_highwater = 2 * world * world + 1
    # A protocol key is waited on by <=W peers. GCC13 vector<int> doubles;
    # erasure keeps capacity. Only one daemon handler reallocates at a time.
    socket_capacity = 1 << (world - 1).bit_length()
    key_chars = MAX_KEY_BYTES + 1  # Heap NUL; native '/' is already in limit.
    # Key arrays, daemon WAIT/temporary, scalar value/caster temporaries,
    # and the leader's batch values. Payload buffers are charged elsewhere.
    string_headers = 2 * batch_keys + world + 1 + 2 * callers + peers
    # Two key-array vectors/caller, two value/return vectors/caller, leader
    # batch elements, two daemon key/receive vectors, two daemon value/return
    # vectors and four outer batch/conversion/return vector objects.
    vector_headers = 4 * callers + peers + 8
    terms = {
        "stored_key_chars_and_nodes": map_keys * (key_chars + 72),
        "client_and_handler_key_chars": (2 * batch_keys + world) * key_chars + MAX_KEY_BYTES,
        "client_and_handler_string_objects": string_headers * 32,
        "client_and_handler_vector_objects": vector_headers * 24,
        # SendBuffer embeds its vector, watermark and client reference.
        "active_sendbuffer_objects": callers * 40,
        "native_ack_bytes_wrappers": peers * 8,
        "waiting_key_chars_and_nodes": batch_keys * (key_chars + 72),
        "waiting_socket_vector_storage": 4 * (batch_keys * socket_capacity + socket_capacity // 2),
        "current_tcpstore_and_client_objects": clients * (216 + 8),
        # Actual private SDK layout: daemon288 includes its four maps,
        # Socket/vector/base/pipe fields; the loop owns another vector24.
        "daemon_and_poll_vector_objects": 288 + 24 + 8 + 32,
        # Default load_factor1; no project reserve/rehash. Runtime14 grows
        # only with live n>=old buckets, to P(max(n+1,2*old)). Its selected
        # prime table has next<=2*previous+211 (maximum211 verified in the
        # sealed ELF). Including old+new rehash arrays therefore costs
        # <=8*(5N+211), independent of erase/reinsert history. This pays
        # source bucket pointers; it does not pay native allocator arenas.
        "value_and_wait_bucket_pointer_storage": 8 * (5 * map_highwater + 211)
            + 8 * (5 * wait_highwater + 211),
        # keysAwaited keeps zero entries after successful waits; disconnect
        # observation need not follow client destruction promptly. Also pay
        # miscellaneous accepted-FD entries and both retained bucket arrays
        # using the finite descriptor universe, not current Python clients.
        "accepted_fd_bookkeeping_nodes": accepted * (24 + 16),
        "accepted_fd_bucket_pointer_storage": 2 * 8 * (5 * accepted + 211),
        "accepted_socket_vector_storage": 8 * (socket_slots + socket_slots // 2),
        "poll_vector_storage": 8 * (poll_slots + poll_slots // 2),
        "socket_implementation_objects": 48 * (accepted + clients + 1),
        # socket.cpp formats host/numeric port into NI_MAXHOST1025 and
        # NI_MAXSERV32 arrays. IPv6 adds three delimiters; fmt::to_string
        # constructs an exact-sized std::string, including one heap NUL.
        "socket_endpoint_chars": (accepted + clients) * (1025 + 32 + 2),
        # getHost() returns addr_.host by const reference; clones copy that
        # same string. A Python constructor additionally holds its caster
        # string while the copied by-value argument moves into addr_.host.
        # At most W constructors overlap. The216-byte Store and existing
        # caller string-header allowance already pay the15-byte SSO case.
        "configured_host_string_storage": (clients + world) * (host_bytes + 1 if host_bytes > 15 else 0),
        # At most W client constructors and one daemon accept format at once.
        # Two nested fmt buffers (each536, inline500) grow by3/2; for a
        # <=1058-character result old+new heap storage is <=5*1058/2.
        "endpoint_format_temporaries": (world + 1) * (
            2 * 536 + 1025 + 32 + 2 * (1058 + 1 + 32)
            + 2 * ((5 * 1058 + 1) // 2)),
        # Actor/worker SendBuffer capacities are already in data/value terms;
        # only the independent leader wake receiver is additional here.
        "wake_receiver_sendbuffer_storage": (max(32, MAX_KEY_BYTES + 9, 2878) + 2878) if ram else 0,
    }
    return {"schema": "tensorfold.sync-store-fixed-payload-v1", "scope": "global-host",
            "world": world, "ram": ram, "maximum_state_slots": slots,
            "nofile_hard_limit": fd_limit, "native_host_bytes": host_bytes,
            "accepted_socket_bound": accepted,
            "current_client_owner_bound": clients, "active_caller_bound": callers,
            "client_key_slots_per_array": batch_keys, "stored_key_bound": map_keys,
            "prefix_boundary_key_bound": prefix_keys,
            "value_map_key_highwater_bound": map_highwater,
            "wait_map_key_highwater_bound": wait_highwater,
            "waiting_socket_vector_capacity": socket_capacity,
            "terms": terms, "global_native_fixed_payload_bytes": _total(terms.values()),
            "source": "Torch04cf2557/GCC13/runtime14; sealed SDK layout and native-table-assist",
            "preconditions": "shared8192-byte bound for every protocol key; immutable W/maximum-slots/hard-NOFILE/actual-host Store region",
            "remaining_native_constraints": (
                "allocator arenas/usable sizes and kernel socket buffers are not source payload",),
            "evidence": "known source-owned payload only; not complete native residency qualification"}


def require_native_payload_runtime():
    """Verify the actual selected ABI inputs behind the native size equations.

    Uses the existing bounded descriptor capture, not deserialized attestations
    or a new certification system. The exact runtime14 library proof and the
    Torch04cf2557/GCC13 template source are versioned qualification inputs.
    The owned socket correction preserves the prior headers, Store layouts
    and allocation equations; the rejected older fork is not an ABI fallback.
    """
    if sys.implementation.name != "cpython" or sys.version_info[:3] != (3, 12, 3):
        raise ValueError("native unique-profile payload bounds require the selected CPython3.12.3 layout")
    import platform
    import resource
    from pathlib import Path
    import torch
    from tensorfold.engine.model_identity import capture_model_identity
    from .execution_identity import _small

    revision = "04cf2557f315a350e7eea7ab064288def65db75a"
    library_hash = "1fd75fe70354a416d75aef22bcae68c47bd25d20e2d0568c30b1a9838cf62f11"
    library_size = 2592224
    build = torch.__config__.show()
    if (sys.platform != "linux" or platform.machine() != "x86_64" or struct.calcsize("P") != 8
            or torch.version.git_version != revision or not torch.compiled_with_cxx11_abi()
            or "GCC 13.3" not in build):
        raise ValueError("native retained payload bound requires the selected Torch04cf2557/GCC13 Linux64 ABI")
    _soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard == resource.RLIM_INFINITY or type(hard) is not int or not 0 < hard <= sys.maxsize:
        raise ValueError("native retained FD metadata requires a finite existing hard descriptor limit")
    mapped = set()
    for line in _small("/proc/self/maps", 4 << 20).decode().splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and "/libstdc++.so" in fields[5]:
            if fields[5].endswith(" (deleted)"):
                raise ValueError("native retained payload bound cannot use a deleted C++ runtime")
            mapped.add(Path(fields[5]).resolve(strict=True))
    if len(mapped) != 1:
        raise ValueError("native retained payload bound requires one actual loaded C++ runtime")
    captured = capture_model_identity({"native-budget/libstdcxx": mapped.pop()},
        runtime_identity={"torch_git": revision, "compiler_sha256": hashlib.sha256(build.encode()).hexdigest()},
        authorize=lambda _name, path: path.resolve(strict=True), max_files=1,
        max_file_bytes=library_size, max_total_bytes=library_size)
    if captured.files[0].size != library_size or captured.files[0].sha256 != library_hash:
        raise ValueError("native retained payload bound has no proof for the actual C++ runtime content")
    return {"torch_git": revision, "cxx11_abi": True, "compiler": "GCC13.3",
            "python_implementation": sys.implementation.name, "python_version": list(sys.version_info[:3]),
            "libstdcxx_sha256": library_hash, "content_identity": captured.model_prefix,
            "hard_nofile": hard,
            "evidence": "source ABI only; allocator/kernel/device/P3 qualification pending"}


def retained_recovery_extra(existing_bytes, *, slots, keep, points):
    """Charge private idle versions beyond the existing global KEEP allowance.

    Existing family geometry charges (KEEP + slots*POINTS) versions. An idle
    protected generation may retain KEEP versions independently of the global
    optional prefix list. Reuse the authoritative category's exact width,
    including method-specific snapshots when called for that category.
    """
    for name, value in (("existing recovery bytes", existing_bytes), ("prefix count", keep), ("cut count", points)):
        _size(value, name, zero=True)
    _size(slots, "GPU slot count")
    versions = keep + slots * points
    if versions == 0:
        if existing_bytes:
            raise ValueError("recovery bytes have no admitted version count")
        return 0
    width, remainder = divmod(existing_bytes, versions)
    if remainder:
        raise ValueError("recovery category does not match the actual family version geometry")
    return _total((max(0, slots * keep - keep) * width,))


def retention_device_resources(breakdown, *, policy, slots, keep, points, mtp, hidden_width,
                               media_inputs_enabled=False):
    """Chosen rank geometry -> missing header delta and runtime pool limits.

    Both stages use the SAME actual core/DF snapshot categories. Media HC and
    coordinates reuse the maintained boundary contract, including its explicit
    credit for HC already present in ordinary protected snapshots.
    """
    if type(breakdown) is not dict or type(mtp) is not bool or type(media_inputs_enabled) is not bool:
        raise ValueError("retention resources require the actual chosen geometry and method capabilities")
    if policy is not None and type(policy) is not RetentionPolicy:
        raise ValueError("retained device pools require the actual typed policy or None")
    _size(slots, "GPU slot count")
    _size(keep, "prefix count", zero=True)
    _size(points, "cut count", zero=True)
    _size(hidden_width, "completed HC width", zero=not mtp)
    versions = keep + slots * points
    protected = keep if policy is None else slots * keep
    width = 0
    for name in ("retained_recovery", "dflash_recovery_snapshots"):
        value = _size(breakdown.get(name, 0), name, zero=True)
        if versions == 0:
            if value:
                raise ValueError("snapshot bytes have no source-header versions")
        else:
            unit, remainder = divmod(value, versions)
            if remainder:
                raise ValueError("snapshot category differs from actual version geometry")
            width += unit
    pending = live_extra = coordinate_extra = 0
    if mtp:
        from .media_mtp import boundary_resources

        actual = boundary_resources(hidden_width=hidden_width, slots=slots, keep=keep, points=points)
        pending = actual["live_hc_bytes"]
        if media_inputs_enabled:
            if breakdown.get("media_mtp_boundary", 0) != actual["additional_device_bytes"]:
                raise ValueError("retained media resources lack the actual existing boundary header credit")
            pending += actual["live_coordinate_bytes"]
            coordinate_width = actual["live_coordinate_bytes"] // slots
            width += coordinate_width
            coordinate_extra = max(0, protected - keep) * coordinate_width
        else:
            live_extra = pending
    return {"schema": "tensorfold.retention-device-resources-v1", "scope": "per-rank",
            "header_extra_device_bytes": _total((live_extra, coordinate_extra)),
            "text_live_hc_header_extra_bytes": live_extra,
            "retained_media_coordinate_extra_bytes": coordinate_extra,
            "protected_versions": protected,
            "limits": {"recovery_pool_bytes": _total((protected * width,)),
                       "pending_hc_bytes": pending, "snapshot_bytes": width},
            "evidence": "source-header allocation payload; allocator/device qualification pending"}


@dataclass(frozen=True)
class PresentationBound:
    tokenizer_identity: str
    vocab_size: int
    piece_utf8_bytes: int
    decoded_utf8_bytes_per_token: int
    decoded_utf8_fixed_bytes: int
    working_utf8_bytes_per_token: int
    working_utf8_fixed_bytes: int
    decoder: str
    lexical_id_wire_ratio: tuple | None = None
    decoded_id_wire_ratio: tuple | None = None
    escaped_id_wire_ratio: tuple | None = None


def tokenizer_presentation_bound(tokenizer, *, precise_bytelevel=True):
    """Bind the ACTUAL loaded decoder and every lexical/added token.

    tokenizers 0.23.3 ByteLevel maps characters to bytes (or retains a token's
    UTF8 bytes) then uses UTF8 lossy conversion. Each input byte can produce at
    most one three-byte replacement character. Encode-only normalization and
    postprocessing cannot expand decode. Builtin decoder composition follows
    the exact 0.23.3 decode_chain sources; it does not implement decoding again.
    Plain ByteLevel concatenates each piece's byte vector before lossy UTF8.
    Joining can complete a boundary scalar, replacing at least two separately
    emitted replacement characters; it cannot increase their UTF8 byte sum.
    Its enumerated per-piece maximum therefore bounds every concatenation.
    Other compositions keep affine bounds, including empty Fuse/Replace input.
    """
    import tokenizers
    from tensorfold.token_metadata import tokenizer_identity, tokenizer_vocab_size

    if type(precise_bytelevel) is not bool:
        raise ValueError("tokenizer payload projection requires an explicit precision mode")
    if type(tokenizer) is not tokenizers.Tokenizer or tokenizers.__version__ != "0.23.3":
        raise ValueError("tiered resource promises require the supported tokenizers 0.23.3 frontend")
    decoder = tokenizer.decoder
    state = None if decoder is None else json.loads(decoder.__getstate__())
    vocab = _size(tokenizer_vocab_size(tokenizer), "actual tokenizer vocabulary")
    maximum = decoded_maximum = 0
    lexical_ratio = decoded_ratio = escaped_ratio = (0, 1)
    plain_bytelevel = precise_bytelevel and state is not None and state["type"] == "ByteLevel"
    def larger_ratio(previous, numerator, denominator):
        return (numerator, denominator) if numerator * previous[1] > previous[0] * denominator else previous
    for index in range(vocab):
        piece = tokenizer.id_to_token(index)
        if piece is not None:  # Prepared UNUSED matrix rows have no text.
            maximum = max(maximum, len(piece.encode("utf-8")))
            if plain_bytelevel:
                decoded = decoder.decode([piece])
                decoded_bytes = len(decoded.encode("utf-8"))
                decoded_maximum = max(decoded_maximum, decoded_bytes)
                denominator = len(str(index)) + 1  # decimal ID and its array separator
                lexical_ratio = larger_ratio(lexical_ratio, len(piece.encode("utf-8")), denominator)
                decoded_ratio = larger_ratio(decoded_ratio, decoded_bytes, denominator)
                escaped_ratio = larger_ratio(escaped_ratio, len(json.encoder.encode_basestring_ascii(decoded)) - 2, denominator)
    if plain_bytelevel:
        return PresentationBound(tokenizer_identity(tokenizer), vocab, maximum, decoded_maximum, 0,
                                 max(maximum, decoded_maximum), 0, "tokenizers-0.23.3/bytelevel-concat-v2",
                                 lexical_ratio, decoded_ratio, escaped_ratio)
    a, b, count_a, count_b = maximum, 0, 1, 0
    peak_a, peak_b = a, b
    pending = [state]
    while pending:
        operation = pending.pop()
        kind = None if operation is None else operation["type"]
        if kind == "Sequence":
            pending.extend(reversed(operation["decoders"]))
            continue
        if kind is None:  # Tokenizer's no-decoder fallback joins pieces with spaces.
            a, b = a + count_a, b + count_b
        elif kind in ("ByteLevel", "ByteFallback"):
            a, b = 3 * a, 3 * b
            if kind == "ByteLevel":
                count_a, count_b = 0, 1
        elif kind == "Fuse":
            count_a, count_b = 0, 1
        elif kind == "WordPiece":
            a, b = a + count_a, b + count_b
        elif kind == "BPEDecoder":
            if not operation["suffix"]:
                a, b = 2 * a + count_a, 2 * b + count_b
        elif kind == "CTC":
            if operation["cleanup"] and not operation["word_delimiter_token"]:
                a, b = 2 * a + count_a, 2 * b + count_b
        elif kind == "Replace":
            size = len(operation["content"].encode("utf-8"))
            # Nonoverlapping matches, including zero-width boundaries, are
            # bounded by bytes+piece count. Content is inserted literally.
            a, b = (1 + size) * a + size * count_a, (1 + size) * b + size * count_b
        elif kind not in ("Metaspace", "Strip"):
            raise ValueError("resource projection does not recognize this effective builtin decoder")
        _size(a, "decoder affine slope", zero=True)
        _size(b, "decoder affine intercept", zero=True)
        peak_a, peak_b = max(peak_a, a), max(peak_b, b)
    return PresentationBound(tokenizer_identity(tokenizer), vocab, maximum, a, b, peak_a, peak_b,
                             "tokenizers-0.23.3/builtin-affine-v1")


def _python_layout():
    # List over-allocation below is the supported CPython 3.12 list_resize
    # contract, not a universal Python/allocator assumption. These limits apply
    # only to the optional new tier promise; ordinary backends remain unchanged.
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 12):
        raise ValueError("tiered Python payload projection requires the selected CPython 3.12 runtime")
    return {"pointer": struct.calcsize("P"), "list": sys.getsizeof([]), "tuple": sys.getsizeof(()),
            "dict": sys.getsizeof({}), "dict_entry": sys.getsizeof({0: 0}),
            "bytes": sys.getsizeof(b""), "unicode": sys.getsizeof("\U0010ffff"),
            "bytearray": sys.getsizeof(bytearray()), "integer": sys.getsizeof(sys.maxsize)}


@dataclass(frozen=True)
class _JsonPayloadPolicy:
    identity: str
    files: tuple
    recursion_limit: int
    frame_work: int

    def record(self):
        return {"schema": "tensorfold.cpython-json-payload", "version": 1,
                "runtime_identity": self.identity, "files": self.files,
                "fresh_tree_bytes_per_encoded_byte": 45, "c_container_depth": 10000,
                "python_recursion_limit": self.recursion_limit, "frame_work_bytes": self.frame_work,
                "scope": "selected object/vector payload; allocator/C-stack/native workspace qualification separate"}


def _json_payload_policy():
    """Bind the actual Ubuntu CPython build before using its tighter layout.

    The C parser's depth is a compiled runtime property, distinct from Python's
    mutable recursion setting. Source/header pairing and all byte identities
    are recorded in the selected-runtime derivation, not inferred from 3.12.3.
    """
    import _json
    import copy
    import platform
    import sysconfig
    from pathlib import Path
    from tensorfold.engine.model_identity import capture_model_identity
    from .execution_identity import _small

    if (sys.implementation.name != "cpython" or sys.version_info[:3] != (3, 12, 3)
            or sys.platform != "linux" or platform.machine() != "x86_64" or struct.calcsize("P") != 8
            or sys.get_int_max_str_digits() != 4300
            or sysconfig.get_config_var("Py_DEBUG") not in (None, 0)
            or sysconfig.get_config_var("Py_TRACE_REFS") not in (None, 0)):
        raise ValueError("JSON payload projection requires the selected CPython3.12.3 Linux64 build")
    if (sys.getsizeof("") != 41 or sys.getsizeof([]) != 56 or sys.getsizeof({}) != 64
            or sys.getsizeof({"": None}) != 184 or sys.getsizeof((1 << 64) - 1) != 36):
        raise ValueError("JSON payload projection requires its actual selected object layout")
    expected = {
        "python": (Path(sys.executable).resolve(), 8020928, "e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f"),
        "json-native": (Path(_json.__file__).resolve(), 48952, "ca8e35c58f7bc8e27258c4af43651c941413bc6318903621253485b0395758a3"),
        "libpython": (Path("/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0"), 9061000,
                      "1021f236227334ded6d1bbbc4bb6f5d933f793d8397c969896cfbdf03ce96223"),
        "pystate": (Path("/usr/include/python3.12/cpython/pystate.h"), 17228,
                    "11494795dd1d6945fbf8036af5ccf247463e94d405a760924a1c78f78ebfc4ec"),
        "ceval": (Path("/usr/include/python3.12/internal/pycore_ceval.h"), 5265,
                  "1af22b50c9ececfc4c3a5f37ecb70cc1c0eefad5a2656e6e22fc088cae54e226"),
        "frame": (Path("/usr/include/python3.12/internal/pycore_frame.h"), 9256,
                  "97c7e2722af4c30c240b8b3c867a54b2fe49a4207d2e566c6a19a9d190fd0f8a"),
        "copy": (Path(copy.__file__).resolve(), 8412,
                 "cbd25547933176fcf6bb05c2adc9f4796d15ac20b9b82dcf890daea7203daeab"),
        "json-api": (Path(json.__file__).resolve(), 14020,
                     "d5d41e2c29049515d295d81a6d40b4890fbec8d8482cfb401630f8ef2f77e4d5"),
        "json-decoder": (Path(json.decoder.__file__).resolve(), 12473,
                         "9f02654649816145bc76f8c210a5fe3ba1de142d4d97a1c93105732e747c285b"),
    }
    mapped = {"python": set(), "libpython": set(), "json-native": set()}
    for line in _small("/proc/self/maps", 4 << 20).decode().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6:
            continue
        for role in mapped:
            candidate = fields[5].removesuffix(" (deleted)")
            if Path(candidate).name == expected[role][0].name:
                if fields[5].endswith(" (deleted)"):
                    raise ValueError("JSON payload projection cannot use a retired runtime artifact")
                mapped[role].add(Path(fields[5]).resolve(strict=True))
    required = ("python", "json-native")
    if (any(mapped[role] != {expected[role][0].resolve(strict=True)} for role in required)
            or mapped["libpython"] and mapped["libpython"] != {expected["libpython"][0].resolve(strict=True)}):
        raise ValueError("JSON payload projection requires its actual loaded parser and Python runtime")
    captured = capture_model_identity({name: row[0] for name, row in expected.items()},
        runtime_identity={"policy": "cpython-json-payload-v1"},
        authorize=lambda _name, path: path.resolve(strict=True), max_files=len(expected),
        max_file_bytes=max(row[1] for row in expected.values()),
        max_total_bytes=sum(row[1] for row in expected.values()))
    if any((item.size, item.sha256) != expected[item.logical_name][1:] for item in captured.files):
        raise ValueError("JSON payload projection has no proof for the actual runtime/header content")
    captured.verify_unchanged()
    functions = (copy.deepcopy, copy._deepcopy_list, copy._deepcopy_dict, copy._keep_alive,
                 json.loads, json.JSONDecoder.decode, json.JSONDecoder.raw_decode,
                 json.JSONDecoder.__init__, json.JSONDecodeError.__init__)
    slots = max(fn.__code__.co_nlocals + len(fn.__code__.co_cellvars) + len(fn.__code__.co_freevars)
                + fn.__code__.co_stacksize for fn in functions)
    # Materialized heap frame, active backing, traceback, iterator/items/pair,
    # bound append and transient id. Include Python's error-reporting headroom.
    per_frame = (136 + 8 * slots) + (72 + 8 * slots) + 56 + 72 + 40 + 56 + 72 + 36
    recursion = _size(sys.getrecursionlimit(), "actual Python recursion limit")
    return _JsonPayloadPolicy(captured.model_prefix,
        tuple((item.logical_name, item.size, item.sha256) for item in captured.files),
        recursion, _size((recursion + 50) * per_frame + 36, "JSON/copy frame payload"))


def _fresh_dict_payload(count, entry_bytes):
    """Fresh monotone selected-CPython table, not the separate churn bound."""
    if not count:
        return 64, 0
    capacity = 8 if count <= 5 else 1 << (((3 * count + 1) // 2) - 1).bit_length()
    def table(rows):
        width = 1 if rows < 256 else 2 if rows < 65536 else 4 if rows < 1 << 32 else 8
        return 32 + rows * width + (2 * rows // 3) * entry_bytes
    return 64 + table(capacity), table(capacity // 2) if capacity > 8 else 0


def _append_list_payload(count):
    capacity = 0 if not count else count + (count >> 3) + 6
    return 56 + 8 * capacity


def _json_tree_payload(encoded_bytes, policy):
    if policy is None:
        layout = _python_layout()
        return encoded_bytes * max(layout["dict_entry"], layout["unicode"] + 4, layout["integer"])
    return _size(45 * encoded_bytes, "fresh JSON tree payload", zero=True)


def _json_copy_working(encoded_bytes, policy):
    containers = encoded_bytes // 2
    memo, old = _fresh_dict_payload(containers + 1, 24)
    _keys, old_keys = _fresh_dict_payload(encoded_bytes // 3, 16)
    list_overlap = _append_list_payload(containers) - 56
    return _total((memo, 36 * (containers + 1), _append_list_payload(containers),
                   max(old, old_keys, list_overlap), policy.frame_work))


def _json_parse_payload(encoded_bytes, policy):
    """Default selected scanner, including malformed-prefix/error peaks."""
    _size(encoded_bytes, "JSON input bytes", zero=True)
    depth, keys = min(encoded_bytes, 10000), (encoded_bytes + 1) // 4
    memo, old = _fresh_dict_payload(keys, 16)
    raw = sys.getsizeof(b"") + encoded_bytes
    digits = len(str(encoded_bytes + 1))
    error_chars = max(72 + 3 * digits, 136 + digits, 83, 111 + digits,
                      108 + 2 * digits, 49, 45, 28)
    def ascii_string(chars):
        return 41 + chars
    error_writer = 2 * (80 + 4 * max(error_chars + error_chars // 4 + 1, 237))
    error_strings = (ascii_string(49) + ascii_string(28) + ascii_string(9) + ascii_string(56)
        + ascii_string(error_chars) + 3 * ascii_string(digits) + 72 + 64 + error_writer)
    fixed = 512 + 88 + 64 + 184 + 320 + 144 + 1448 + error_strings + policy.frame_work
    terms = {"fresh_tree": _json_tree_payload(encoded_bytes, policy), "open_containers": 45 * depth,
        "key_memo": memo, "one_container_resize": max(old, _append_list_payload(encoded_bytes // 2) - 56),
        "raw_input": raw, "unicode_error_input_copy": raw, "decoded_document": 2 * (80 + 4 * encoded_bytes),
        "unicode_writer": 2 * (80 + 4 * (encoded_bytes + encoded_bytes // 4 + 1)),
        "number_source": 49 + encoded_bytes, "fixed_headers_error_strings_frames": fixed}
    return {"schema": "tensorfold.selected-json-parse-payload-v1", "terms": terms,
            "total_bytes": _total(terms.values()), "c_depth_bound": depth, "key_count_bound": keys}


def _responses_cache_wrappers(limit, connections):
    # OrderedDict's keys table and fast-node mirror may both be old/new at
    # resize. Nodes themselves are separately linked; this is the churn cap.
    capacity = 1 << (max(8, 3 * (limit + 1)) - 1).bit_length()
    width = 1 if capacity < 256 else 2 if capacity < 65536 else 4 if capacity < 1 << 32 else 8
    table = 32 + capacity * width + (2 * capacity // 3) * 24
    ordered = 128 + 2 * (table + 8 * capacity) + 32 * (limit + 1)
    entries = limit + 1 + 2 * connections
    return {"ordered_dict_keys_mirrors_nodes": ordered, "entry_count": entries,
            "entry_payload_bytes": entries * 536, "shared_attribute_keys": 768,
            "total_bytes": _total((ordered, entries * 536, 768))}


def continuation_payload_bytes(*, context, vocab_size, presentation, max_body_bytes, copy_enabled,
                               presentation_fixed=0, json_policy=None):
    """Named future owned Python/NumPy payloads for ONE logical CPU owner.

    InputClaim independently pays the original full prompt and encoded replay.
    This pays copies/continuation/output through the full supported context,
    including thinking/gating subruns. Shared model/tokenizer tables are already
    loaded and excluded. Runtime allocator arenas/native libraries remain an
    explicit qualification obligation, not hidden inside these byte counts.
    """
    for name, value in (("context", context), ("vocabulary", vocab_size), ("body limit", max_body_bytes)):
        _size(value, name)
    if type(copy_enabled) is not bool:
        raise ValueError("copy draft resources require the actual prepared decoder boolean")
    _size(presentation, "presentation bytes per token", zero=True)
    _size(presentation_fixed, "presentation fixed bytes", zero=True)
    layout = _python_layout()
    integer = max(layout["integer"], sys.getsizeof(vocab_size - 1), sys.getsizeof(context))

    def sequence(n):
        return layout["list"] + (n + (n >> 3) + 6) * layout["pointer"] + n * integer

    # Actual Stream lists, CUDA App out, and StreamDecoder.ids. Temporary token
    # slices in decode/stop/copy-proposal work coexist with those owners.
    common_lists = ("stream.out", "stream.context", "stream.committed", "stream.drafts", "stream.stops", "stream.owed")
    leader_lists = ("app.out", "decoder.ids", "decoder.before_ids", "decoder.after_ids", "stop.tail_ids")
    copy_lists = ("copy.query", "copy.starts", "copy.best", "copy.context_slice") if copy_enabled else ()
    token_lists = common_lists + leader_lists + copy_lists
    lists = len(token_lists) * sequence(context)
    # One mailbox node per token is the legal adverse chunking. Each owns a
    # one-item list, the (list,count) tuple and a deque pointer pair.
    from collections import deque

    mailbox = context * (sequence(1) + layout["tuple"] + 2 * layout["pointer"]
                         + integer + sys.getsizeof(deque((None,))))
    # CPU CopyIndex: at most context n-grams, each owns an eight-ID tuple and a
    # start-list; singleton dictionary storage dominates its CPython table per
    # entry. Bulk hashes/starts + sort/hash preparation are separate NumPy u64
    # arrays (six simultaneous arrays in the source update/probe expression).
    copies = 0
    if copy_enabled:
        from tensorfold.families.qwen3_5.cuda.draft_tree import COPY_MIN_MATCH

        import numpy as np

        array = context * 8 + sys.getsizeof(np.empty(0, dtype=np.uint64))
        copies = context * (layout["dict_entry"] + layout["tuple"]
                            + COPY_MIN_MATCH * (layout["pointer"] + integer) + sequence(1)) + 6 * array
    # Incremental decode keeps before/after/accumulated text; stop/markup/reply
    # splitting can retain old and new presentation strings at the same call.
    # Count actual source string roles rather than assuming one byte/character.
    chars = context * presentation + presentation_fixed
    string = layout["unicode"] + 4 * chars + chars  # PEP393 payload plus possible UTF8 cache.
    text_roles = ("decoder.before", "decoder.after", "decoder.text", "decoder.append_slice", "decoder.new_text",
                  "final.before", "final.decoded", "final.slice", "final.joined", "app.raw", "app.reasoning",
                  "app.answer", "app.answer_raw", "app.delta_reasoning", "app.delta_content", "app.result_text",
                  "stop.decoded", "stop.visible", "markup.input", "markup.normalized", "markup.tail",
                  "markup.fragments", "markup.joined", "tool.argument_source")
    text = len(text_roles) * string
    # Parsed permissive tool outputs and JSON/SSE serialization may coexist
    # with the decoded reply. Use the same conservative node projection as
    # bounded input JSON, plus six-byte JSON escaping of each output codepoint.
    # Project-created tool/output wrappers are not a fresh parsed tree whose
    # own encoded length equals the decoded model text. Keep their independent
    # generic source-role bound until that schema has a stronger proof.
    reply_graph = _json_tree_payload(chars, None)
    reply_wire = 6 * chars + layout["bytes"]
    # Original stop/tool/request fields can remain reachable through an exact
    # suspended/completed callback owner after the HTTP parser returns.
    request_graph = _json_tree_payload(max_body_bytes, json_policy)
    return {"continuation_lists": lists, "mailbox": mailbox, "copy_index": copies,
            "presentation_strings": text, "reply_graph_and_wire": reply_graph + reply_wire,
            "request_graph": request_graph, "input_prompt": 0,
            "token_list_roles": token_lists, "text_roles": text_roles,
            "list_partition": {"context_tokens": context,
                "common_stream_bytes": len(common_lists) * sequence(context),
                "leader_frontend_bytes": len(leader_lists) * sequence(context),
                "copy_bytes": len(copy_lists) * sequence(context),
                "per_list_pointer_bytes": sequence(context) - context * integer,
                "prompt_integer_pool_bytes": context * integer},
            "project_payload_bytes": _total((lists, mailbox, copies, text, reply_graph, reply_wire, request_graph))}


def tp_rpc_python_payload(continuation_terms, *, world, context, max_pending_requests, max_http_body_bytes,
                          json_policy=None, ordinary_fifo_children=False):
    """Named Python owners for the serialized TP actor, separate from native IO.

    The selected protocol bounds commands and ACKs by MAX_MESSAGE_BYTES and
    hashes status records incrementally. Completed follower bodies and sent
    leader text have already retired their redundant locals. Native copies,
    backing manifests, HTTP trees and the leader InputClaim are not charged
    here. Runtime allocator/RSS remains a separate qualification obligation.
    """
    import io
    import types

    from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES

    for name, value in (("rank count", world), ("context", context),
                        ("pending requests", max_pending_requests), ("HTTP body bytes", max_http_body_bytes)):
        _size(value, name)
    layout = _python_layout()
    partition = continuation_terms["list_partition"]
    if partition["context_tokens"] != context:
        raise ValueError("RPC projection context differs from its continuation payload")
    for name in ("common_stream_bytes", "leader_frontend_bytes", "copy_bytes",
                 "per_list_pointer_bytes", "prompt_integer_pool_bytes"):
        _size(partition[name], name, zero=name == "copy_bytes")
    per_follower = {
        "common_stream_lists": partition["common_stream_bytes"],
        "copy_lists": partition["copy_bytes"],
        "copy_index": _size(continuation_terms["copy_index"], "copy index bytes", zero=True),
        "prompt_pointer_list": partition["per_list_pointer_bytes"],
        "prompt_integer_pool": partition["prompt_integer_pool_bytes"],
    }
    follower_scope = _total(per_follower.values())
    if world == 1:
        return {"global_rpc_python_working_bytes": 0, "per_follower_scope_bytes": follower_scope,
                "per_follower_scope": per_follower, "named_rpc_terms": {}, "frame_bytes": MAX_MESSAGE_BYTES,
                "per_http_rpc_error_tail_bytes": 0}

    peers, pending, frame = world - 1, max_pending_requests, MAX_MESSAGE_BYTES
    node = max(layout["dict_entry"], layout["unicode"] + 4, layout["integer"])
    if continuation_terms["request_graph"] != _json_tree_payload(max_http_body_bytes, json_policy):
        raise ValueError("RPC projection needs its actual continuation HTTP/body layout")
    graph = _json_tree_payload(frame, json_policy)
    text = layout["unicode"] + frame  # Both bounded publishers produce ASCII.
    raw = layout["bytes"] + frame
    # _execute publishes only an error class name, a 64-hex state digest,
    # and an optional bool/two-bool completion record. The class name can
    # fill the wire bound; five times its ASCII-encoded extent covers even
    # a wide PEP393 string and UTF8 cache. Received malformed ACKs still use
    # the generic parsed-graph bound below, before their equality check.
    ack_prototype = {"error": "", "state": "f" * 64,
                     "complete": {"drained": False, "changed": False}}
    ack_pending, ack_seen, ack_headers = [ack_prototype], set(), 0
    while ack_pending:
        value = ack_pending.pop()
        if id(value) in ack_seen:
            continue
        ack_seen.add(id(value))
        ack_headers += sys.getsizeof(value)
        if type(value) is dict:
            ack_pending.extend(value.keys())
            ack_pending.extend(value.values())
    ack_graph = _size(5 * frame + ack_headers, "owned RPC acknowledgment graph")

    def pointers(count):
        return layout["list"] + (count + (count >> 3) + 6) * layout["pointer"]

    # CPython3.12.3's combined dictionary allocation, including deleted-slot
    # churn and retained highwater. This is calculate_log2_keysize(used*3),
    # not a live-entry/singleton estimate (the small-size OR matters).
    keys_header = struct.calcsize("nBBBxInn")
    def dictionary_table(count, entry_pointers):
        log = (((3 * count) | 8) - 1 | 7).bit_length()
        capacity = 1 << log
        index_width = 1 if log < 8 else 2 if log < 16 else 4 if log < 32 else 8
        return (keys_header + capacity * index_width
                + ((2 * capacity) // 3) * entry_pointers * layout["pointer"]), capacity

    from tensorfold.cuda.streams import Stream
    from .tp_scheduler import MAX_PROTOCOL_SEQUENCE

    map_table, map_capacity = dictionary_table(pending, 3)  # hash,key,value general entries.
    # Wire is assigned only once before media binding/admission. These are the
    # non-dataclass fields assigned by Scheduler/Waiting/TP preparation; later
    # arbitrary foreign attribute mutation is not a serving input contract.
    wire_preparation_fields = frozenset(("_fifo_started", "_media_outcome_retired", "admission_eligibility",
        "admission_status", "admission_ticket", "blocked_input", "delivery_owner", "output_capacity",
        "principal", "production_budget", "retirement_host", "_media_rank_owned", "_retention_staged",
        "_tp_token_limit", "retention_intent", "_media_wire"))
    stream_fields = set(Stream.__dataclass_fields__) | wire_preparation_fields
    if len(Stream.__dataclass_fields__) <= 30:
        raise ValueError("Stream metadata no longer has the selected combined dictionary layout")
    stream_table, stream_capacity = dictionary_table(len(stream_fields), 2)  # Unicode attribute keys.
    stream_header = sys.getsizeof(object.__new__(Stream))
    wire_integer = sys.getsizeof(MAX_PROTOCOL_SEQUENCE)
    identity_integer = sys.getsizeof((1 << (8 * layout["pointer"])) - 1)
    leader_maps = {
        "request_and_identity_map_objects": 2 * layout["dict"],
        "request_and_identity_map_tables": 2 * map_table,
        "optional_stream_metadata_backing": pending * (stream_header + layout["dict"] + stream_table),
        # Resizing is one non-callback C dictionary operation under the GIL;
        # its old table coexists with the new one, never every map's at once.
        "one_dictionary_resize_overlap": max(map_table, stream_table),
        "wire_integer_pool": pending * wire_integer,
        "identity_integer_pool": pending * identity_integer,
        "next_wire_counter": wire_integer,
        "one_identity_lookup_temporary": identity_integer,
    }

    # All three stdlib encoder generator functions have input-independent frame
    # layouts. Observe their actual selected CPython frame headers, not an
    # invented per-recursion multiplier. Its existing recursion limit bounds
    # simultaneous generators/marker entries; this process setting is recorded.
    encoder = json.JSONEncoder().iterencode({"key": [0]})
    frame_header = 0
    try:
        for _chunk in encoder:
            active = encoder
            while isinstance(active, types.GeneratorType):
                frame_header = max(frame_header, sys.getsizeof(active) + sys.getsizeof(active.gi_frame))
                active = active.gi_yieldfrom
    finally:
        encoder.close()
    recursion = sys.getrecursionlimit()
    iterator_header = max(sys.getsizeof(iter([])), sys.getsizeof(iter(())), sys.getsizeof(iter({}.items())))
    marker_header = layout["dict_entry"] + layout["integer"]  # id(container) is a new shared marker key.
    encoder_metadata = recursion * (frame_header + iterator_header + sys.getsizeof({}.items()) + marker_header)
    # CPython3.12 UnicodeWriter grows by25% on Unix/50% on Windows. ASCII-only
    # append never realizes StringIO's UCS4 buffer. Keep old/new writer storage
    # through reallocation plus the current escaped JSON chunk; getvalue's
    # returned string/recreated writer fits the same lifetime maximum.
    capacity = frame + frame // (2 if sys.platform == "win32" else 4) + 1
    writer = layout["unicode"] + capacity
    with io.StringIO() as output:
        # Empty CPython3.12 StringIO also allocates its initial three-UCS4
        # fallback cells, even though append-only encoding stays accumulating.
        stringio_header = sys.getsizeof(output) + 3 * struct.calcsize("I")
    encoding = encoder_metadata + stringio_header + 2 * writer + text

    # _state_digest borrows draft/owed/cur payloads; only these pointer
    # containers and len(out) scalar are new. Incremental hashing retains at
    # most one JSON chunk and its encoded bytes, not a joined N-request string.
    digest_row = (layout["tuple"] + 12 * layout["pointer"]
        + pointers(6) + pointers(8) + layout["tuple"] + 3 * layout["pointer"] + layout["integer"])
    digest_metadata = pending * digest_row + 2 * pointers(pending) + pointers(4) + pointers(4) + 2 * pointers(2)
    digest_working = digest_metadata + encoder_metadata + text + raw

    # One source pack/control graph: prompt/owed/carry aliases are not new
    # payloads. R*node bounds only the newly created RPC metadata subset.
    # Variable source subtrees are declared JSON/token lists (source tuples
    # are smaller than the equivalent append lists). Fixed Stats/controller
    # merges need their own table allowance rather than the parsed-tree lemma.
    # Counts include old/working/result records and merge input footprints.
    source_records = {"stats_own": 10, "stats_both": 7, "stats_return_inputs": 19,
        "controller_maximum": 4, "controller_configured": 4, "controller_effective": 4,
        "freeze_values_and_update": 8, "freeze_family": 2, "freeze_migration": 3,
        "freeze_record": 10, "freeze_workload": 10, "boundary_values": 4,
        "boundary_family_and_replacement": 4, "boundary_record": 5}
    source_record_tables = {}
    for name, fields in source_records.items():
        # A merge may reserve for both current and incoming fixed-key maps.
        # The churn table rule and two tables also fund its old/new cut.
        table, _capacity = dictionary_table(2 * fields, 2)
        source_record_tables[name] = layout["dict"] + 2 * table
    source_fixed = _total(source_record_tables.values())
    source_graph = frame * node if json_policy is None else graph + source_fixed
    parsed = None if json_policy is None else _json_parse_payload(frame, json_policy)
    parse_extra = 0 if parsed is None else parsed["total_bytes"] - graph - raw - text
    receive = graph + raw + text + parse_extra
    dispatch = graph + max(encoding + 5 * layout["dict_entry"], digest_working)
    # During ACK publication the current command graph and new status graph
    # remain live. Status encoding and digest construction happen serially.
    peer_ack = graph + ack_graph + max(encoding, digest_working)
    peer_working = max(receive, dispatch, peer_ack)
    ack_batch = pointers(peers) + peers * raw
    # The last decoded remote status remains across the next json.loads RHS.
    ack_parse = text + 2 * graph + parse_extra
    stages = {
        "source_encoding": source_graph + encoding,
        "publication_and_peer_work": source_graph + text + peers * peer_working,
        "dispatch_and_peer_work": source_graph + max(encoding, digest_working) + peers * peer_working,
        "ack_batch_and_peer_return": (source_graph + ack_graph + ack_batch + ack_parse
                                      + peers * (graph + ack_graph + text)),
    }
    # Stream/carry/media/retention children survive their parent body through
    # ordinary references. These new follower RPC metadata subsets are paid
    # once per logical caller; prompt integers/owed lists use their roles above.
    child_schema = None
    child_payload = graph
    if ordinary_fifo_children:
        # Enforced FIFO carry=None removes the opaque dictionary root.
        # Prompt/owed lists and integers have separate follower roles above;
        # follow() rebuilds grammar Spec.text instead of retaining its int list.
        # Strings in Spec/principal/typed Intent consume disjoint wire extents.
        # Three LogicalReferences and the bounded typed records have <=30
        # attributes each; charge their full selected attribute backing.
        typed_record = stream_header + 336
        integer = max(layout["integer"], sys.getsizeof(10 ** 4300 - 1))
        child_schema = {"schema": "tensorfold.fifo-text-child-payload-v1", "carry": None,
            "variable_string_bytes": 5 * frame,
            "stream_object_and_attributes": stream_header + layout["dict"] + stream_table,
            "sampling_spec_constraint_intent_references": 8 * typed_record,
            "aliases_and_scalar_headers": pointers(3) + 16 * integer + 12 * layout["unicode"],
            "largest_retained_integer_bytes": integer}
        child_payload = _total(value for name, value in child_schema.items()
                               if type(value) is int and name != "largest_retained_integer_bytes")
    persistent_children = peers * pending * child_payload
    # N request/error journals plus one managed actor error root per rank.
    # Raised receive/publication paths preserve byte/text/graph diagnostics.
    error_owners = (pending + 1) * (source_graph + peers * graph + world * (raw + text))
    leader_prompt_refs = pending * partition["per_list_pointer_bytes"]
    fixed = {"follower_owned_rpc_children": persistent_children,
             "preserved_rpc_error_owners": error_owners,
             "leader_prepared_prompt_pointer_lists": leader_prompt_refs,
             "leader_request_map_metadata": _total(leader_maps.values())}
    total = _total((*fixed.values(), max(stages.values())))
    return {"global_rpc_python_working_bytes": total, "per_follower_scope_bytes": follower_scope,
            "per_follower_scope": per_follower, "named_rpc_terms": fixed,
            "phase_working_bytes": stages, "frame_bytes": frame,
            "json_graph_bytes": graph, "ascii_text_bytes": text, "python_bytes_bytes": raw,
            "source_json_graph_bytes": source_graph, "parse_working": parsed,
            "source_fixed_record_tables": source_record_tables,
            "persistent_child_schema": child_schema,
            "source_ack_graph_bytes": ack_graph, "source_ack_schema": "error-state-completion-v1",
            "per_http_rpc_error_tail_bytes": max(source_graph, graph) + raw + text,
            "encoder_working_bytes": encoding, "digest_working_bytes": digest_working,
            "runtime_recursion_limit": recursion, "encoder_frame_header_bytes": frame_header,
            "leader_request_map_metadata": leader_maps,
            "leader_request_map_limits": {"request_highwater": pending, "map_capacity": map_capacity,
                "map_table_bytes": map_table, "stream_dictionary_table_bytes": stream_table,
                "stream_preparation_fields": len(stream_fields), "stream_dictionary_capacity": stream_capacity,
                "wire_integer_bytes": wire_integer, "identity_integer_bytes": identity_integer},
            "evidence": "selected-CPython payload/source owners; allocator/RSS/native qualification separate"}


def retained_constraint_reason(catalog, *, logical_retained):
    if type(logical_retained) is not bool:
        raise ValueError("retention eligibility must come from the trusted logical operation")
    policy = catalog.policy
    if policy is not None and (policy.tier == "ram" or logical_retained):
        return "structured output with retained/tiered execution has no admitted native matcher CPU-memory envelope"
    return None


def _native_host_commitment(catalog, baseline):
    """One authoritative selected-native protocol obligation for every caller."""
    native_slots = baseline.native_state_slots or len(catalog.states)
    if native_slots < len(catalog.states):
        raise ValueError("native protocol State highwater is below the actual admitted State count")
    if catalog.world > 1 and baseline.native_fd_limit is not None:
        import resource

        hard = resource.getrlimit(resource.RLIMIT_NOFILE)[1]
        if hard == resource.RLIM_INFINITY or hard > baseline.native_fd_limit:
            raise ValueError("native descriptor capability exceeds the captured host envelope")
    native_fixed = native_prefix = None
    unpaid_fixed = unpaid_prefix = 0
    if catalog.world > 1:
        native_fixed = native_sync_fixed_payload(world=catalog.world,
            ram=catalog.policy is not None and catalog.policy.tier == "ram", slots=native_slots,
            fd_limit=baseline.native_fd_limit, host_bytes=baseline.native_host_bytes)
        native_prefix = native_sync_prefix_payload(world=catalog.world, slots=native_slots)
        unpaid_fixed = max(0, native_fixed["global_native_fixed_payload_bytes"] - baseline.prepaid_native_fixed_payload_bytes)
        unpaid_prefix = max(0, native_prefix["global_native_prefix_payload_bytes"] - baseline.prepaid_native_prefix_payload_bytes)
    return native_fixed, native_prefix, unpaid_fixed, unpaid_prefix


def _model_host_projection(catalog, *, tokenizer, max_pending_requests, max_waiting_input_bytes,
                           body_bytes, output_queue_tokens, fifo_child_schema=None, precise_payloads=True):
    """Shared real model/caller projection; endpoint roles are added by adapters."""
    catalog.check_health()
    baseline = catalog.host_resources
    if type(baseline) is not HostBaseline:
        raise ValueError("FIFO host admission requires the actual ordinary steady header baseline")
    for name, value in (("pending request limit", max_pending_requests),
                        ("waiting input bytes", max_waiting_input_bytes), ("source payload bytes", body_bytes)):
        _size(value, name)
    if output_queue_tokens is not None:
        _size(output_queue_tokens, "output queue tokens")
    decoder = catalog._cohort.multi
    if decoder is None:
        raise ValueError("endpoint host resources require the actual bound decoder")
    copy_enabled = decoder.copy
    bound = tokenizer_presentation_bound(tokenizer, precise_bytelevel=precise_payloads)
    # Tight native/JSON phase accounting is qualified for the actual plain
    # ByteLevel frontend. Other supported decoder compositions keep their
    # existing conservative projection rather than borrowing this proof.
    json_policy = (_json_payload_policy()
                   if bound.decoder == "tokenizers-0.23.3/bytelevel-concat-v2" else None)
    context = catalog.states[0].logical_context
    if any(state.logical_context != context for state in catalog.states):
        raise ValueError("endpoint host resources require one admitted logical context")
    terms = continuation_payload_bytes(context=context, vocab_size=bound.vocab_size,
        presentation=bound.decoded_utf8_bytes_per_token, presentation_fixed=bound.decoded_utf8_fixed_bytes,
        max_body_bytes=body_bytes, copy_enabled=copy_enabled, json_policy=json_policy)
    if output_queue_tokens is not None:
        terms["mailbox"] = terms["mailbox"] * min(context, output_queue_tokens) // context
        terms["project_payload_bytes"] = _total(value for name, value in terms.items()
                                                if name not in ("project_payload_bytes", "input_prompt", "token_list_roles", "text_roles", "list_partition"))
    header = retention_header_resources(catalog.policy, catalog.world, slots=len(catalog.states))
    sessions = 0 if catalog.policy is None else catalog.policy.max_sessions
    ordinary_children = (json_policy is not None and catalog.policy is None
        and fifo_child_schema is not None and getattr(decoder, "media_runtime", None) is None)
    if fifo_child_schema is not None and (fifo_child_schema != {"schema": "tensorfold.tp-fifo-retained-children",
            "version": 1, "context": context, "vocab": bound.vocab_size}):
        raise ValueError("FIFO child schema differs from the actual model/tokenizer geometry")
    rpc_python = tp_rpc_python_payload(terms, world=catalog.world, context=context,
        max_pending_requests=max_pending_requests, max_http_body_bytes=body_bytes, json_policy=json_policy,
        ordinary_fifo_children=ordinary_children)
    # Replicated Stream state exists on every rank; detok/HTTP are leader-only.
    per_owner = terms["project_payload_bytes"] + (catalog.world - 1) * rpc_python["per_follower_scope_bytes"]
    generation_owners = max_pending_requests
    input_floor = None
    if catalog.policy is None and json_policy is not None:
        input_floor = _size(decoder.input_reservation_bytes(), "mandatory logical input claim")
        generation_owners = min(generation_owners, max_waiting_input_bytes // input_floor)
    model_cpu = (generation_owners + sessions) * per_owner
    return {
        "baseline": baseline,
        "decoder": decoder,
        "copy_enabled": copy_enabled,
        "bound": bound,
        "json_policy": json_policy,
        "context": context,
        "terms": terms,
        "header": header,
        "sessions": sessions,
        "rpc_python": rpc_python,
        "generation_owners": generation_owners,
        "input_floor": input_floor,
        "model_cpu": model_cpu,
    }


@dataclass(frozen=True)
class FiniteBackendCaller:
    """Prepared finite inputs and a synchronous bounded token/result consumer.

    Input preparation and immutable artifact reads complete before admission.
    This caller has no HTTP, Store, media or matcher route. It retains at most
    the declared results; its callback validates IDs before appending them.
    The native model/context/cache contract is unchanged.
    """
    max_source_bytes: int
    max_input_tokens: int
    max_output_tokens: int
    retained_results: int
    max_receipt_bytes: int

    def __post_init__(self):
        for name, value in asdict(self).items():
            _size(value, "finite caller " + name)


def _scheduler_child_schema(scheduler):
    from .tp_scheduler import TensorParallelDecoder

    adapter = scheduler.decoder
    return adapter.fifo_retained_child_schema() if type(adapter) is TensorParallelDecoder else None


def _admit_host_record(catalog, record, materialized):
    from tensorfold.cuda.capacity import admit_host_commitment

    if catalog.has_retained:
        raise ValueError("host resources must finalize before request admission")
    identity = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    old = getattr(catalog, "host_resource_envelope", None)
    if old is not None and old["identity"] != identity:
        raise ValueError("the active retention endpoint resource envelope is immutable")
    authority = admit_host_commitment(record["global_host_commitment_bytes"], materialized_owned_bytes=materialized)
    result = {**record, "identity": identity, "authority": authority}
    catalog.host_resource_envelope = result
    return result


def finalize_backend_resources(catalog, *, tokenizer, scheduler, request_limit, caller):
    """Admit the real model and a finite local caller through the common owner.

    HTTP/metadata/Store terms are absent because this is a distinct enforced
    caller interface, not because HTTP limits were set to zero. No submission
    may precede the ordinary physical/cgroup commitment admission below.
    """
    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.server.request_limits import RequestLimit
    from .tokenizer_payload import bytelevel_decode_payload

    if type(caller) is not FiniteBackendCaller or type(request_limit) is not RequestLimit or type(scheduler) is not Scheduler:
        raise ValueError("finite backend admission requires its typed caller, Scheduler and RequestLimit owners")
    if request_limit.maximum != 1:
        raise ValueError("finite backend caller requires exactly one logical request owner")
    if (scheduler.admission_policy != "fifo" or scheduler._submitted or request_limit.used
            or scheduler._request_limit not in (None, request_limit)):
        raise ValueError("finite backend resources must precede submission under the same real FIFO owner")
    if catalog.policy is not None or getattr(catalog._cohort.multi, "media_runtime", None) is not None:
        raise ValueError("finite backend caller supports ordinary text state without retained/media ownership")
    pending = _size(request_limit.maximum, "finite logical request limit")
    waiting = _size(scheduler.max_waiting_input_bytes, "actual finite FIFO input credit")
    core = _model_host_projection(catalog, tokenizer=tokenizer, max_pending_requests=pending,
        max_waiting_input_bytes=waiting, body_bytes=caller.max_source_bytes,
        output_queue_tokens=scheduler.output_queue_tokens, fifo_child_schema=_scheduler_child_schema(scheduler))
    context, bound, policy = core["context"], core["bound"], core["json_policy"]
    if policy is None or caller.max_input_tokens + caller.max_output_tokens > context:
        raise ValueError("finite caller input/output bounds must fit the actual native logical context")
    decoder = catalog._cohort.multi
    owned = scheduler.decoder.decoder if hasattr(scheduler.decoder, "decoder") else scheduler.decoder
    if owned.states is not decoder.states or owned.w is not decoder.w:
        raise ValueError("finite caller Scheduler differs from the model catalog owner")
    native_fixed, native_prefix, unpaid_fixed, unpaid_prefix = _native_host_commitment(catalog, core["baseline"])
    native_decode = bytelevel_decode_payload(bound, context=caller.max_output_tokens)
    layout = _python_layout()
    integer = max(layout["integer"], sys.getsizeof(bound.vocab_size - 1))
    caller_terms = {
        "prepared_source_strings_and_buffers": 2 * (80 + 5 * caller.max_source_bytes),
        "original_input_ids": _append_list_payload(caller.max_input_tokens) + caller.max_input_tokens * integer,
        "retained_result_ids": caller.retained_results * (
            _append_list_payload(caller.max_output_tokens) + caller.max_output_tokens * integer),
        "retained_decoded_text": caller.retained_results * (80 + 5 * native_decode["decoded_utf8_bytes"]),
        "one_native_decode_and_conversion": max(native_decode["native_heap_requested_bytes"],
                                                  native_decode["result_to_python_payload_bytes"]),
        "bounded_report_graph": caller.max_receipt_bytes * layout["dict_entry"],
        "bounded_report_encoding_and_wire": 5 * caller.max_receipt_bytes + policy.frame_work,
    }
    baseline, header, rpc = core["baseline"], core["header"], core["rpc_python"]
    total = _total((baseline.baseline_host_bytes, header.global_host_bytes, waiting,
        core["model_cpu"], rpc["global_rpc_python_working_bytes"],
        unpaid_fixed, unpaid_prefix, *caller_terms.values()))
    materialized = baseline.materialized_host_bytes
    if catalog._buffer is not None and catalog._buffer.is_pinned():
        materialized += catalog.world * catalog._buffer.nbytes
    record = {"schema": "tensorfold.finite-backend-host-envelope-v1", "caller": asdict(caller),
        "caller_role": "prepared-finite-inputs/synchronous-token-callback/local-bounded-reports",
        "context_tokens": context, "world": catalog.world, "state_slots": len(catalog.states),
        "max_pending_requests": pending, "global_waiting_input_bytes": waiting,
        "tokenizer": asdict(bound), "json_payload_policy": policy.record(), "header": header.record(),
        "continuation": core["terms"], "generation_owner_bound": {"version": 1,
            "owners": core["generation_owners"], "mandatory_input_claim_bytes": core["input_floor"]},
        "global_model_cpu_projection_bytes": core["model_cpu"], "tp_rpc_python": rpc,
        "global_rpc_python_working_bytes": rpc["global_rpc_python_working_bytes"],
        "native_fixed": native_fixed, "native_prefix": native_prefix,
        "global_unpaid_native_fixed_payload_bytes": unpaid_fixed,
        "global_unpaid_native_prefix_payload_bytes": unpaid_prefix,
        "ordinary_host_baseline": asdict(baseline), "caller_payload": caller_terms,
        "native_decode": native_decode, "global_host_commitment_bytes": total,
        "materialized_owned_bytes": materialized,
        "evidence": "source payload commitment; allocator/C-stack/driver/device/P3 qualification separate"}
    return _admit_host_record(catalog, record, materialized)


def finalize_retention_resources(catalog, *, tokenizer, max_pending_requests,
                                max_waiting_input_bytes, max_http_connections, output_queue_tokens=None,
                                max_http_body_bytes=None, logical_api_limits=None, fifo_child_schema=None):
    """Freeze actual endpoint obligations before handlers/accepted generation.

    Physical admission is a current source-candidate check, not an exclusive OS
    grant: concurrent unrelated consumers may still exhaust the host. Existing
    OS reserve/cgroup authorities apply without rank-multiplying global caps.
    """
    from tensorfold.server.request_body import LIMIT

    _size(max_http_connections, "HTTP connection limit")
    body_bytes = LIMIT if max_http_body_bytes is None else max(LIMIT, _size(max_http_body_bytes, "HTTP route body bound"))
    # The complete tighter HTTP/preparation/cache phase proof is a separate
    # open region. Preserve its conservative frontend projection while the
    # finite local caller uses the independently closed model-only boundary.
    core = _model_host_projection(catalog, tokenizer=tokenizer, max_pending_requests=max_pending_requests,
        max_waiting_input_bytes=max_waiting_input_bytes, body_bytes=body_bytes,
        output_queue_tokens=output_queue_tokens, fifo_child_schema=fifo_child_schema, precise_payloads=False)
    baseline = core["baseline"]
    copy_enabled = core["copy_enabled"]
    bound = core["bound"]
    json_policy = core["json_policy"]
    context = core["context"]
    terms = core["terms"]
    header = core["header"]
    sessions = core["sessions"]
    rpc_python = core["rpc_python"]
    generation_owners = core["generation_owners"]
    input_floor = core["input_floor"]
    model_cpu = core["model_cpu"]
    # App.run's logical RequestLimit ends before the HTTP handler serializes
    # and writes its final response. Slow completed sockets and new model jobs
    # may therefore coexist; the real HTTP connection cap owns that tail.
    completed_http = max_http_connections * (terms["presentation_strings"] + terms["reply_graph_and_wire"]
        + terms["continuation_lists"] // len(terms["token_list_roles"]))
    http_rpc_errors = max_http_connections * rpc_python["per_http_rpc_error_tail_bytes"]
    cpu = model_cpu + completed_http + http_rpc_errors
    layout = _python_layout()
    # A JSON graph has no more nodes than encoded bytes. The singleton table
    # bound covers a dictionary/list/scalar node, including its pointer cell.
    # This deliberately conservative wire bound is independent of token shape.
    node = max(layout["dict_entry"], layout["unicode"] + 4, layout["integer"])
    incoming_json = None if json_policy is None else _json_parse_payload(body_bytes, json_policy)
    transport = max_http_connections * (body_bytes * node if incoming_json is None else incoming_json["total_bytes"])
    # The backing cap already pays encoded manifest bytes. Pay additional
    # parsed graph/object headers for all admitted protected generations.
    backed = (0 if catalog.policy is None or catalog.policy.tier != "ram" else min(
        sessions + max_pending_requests + catalog.prefix_limit,
        catalog.policy.ram_bytes // catalog.minimum_future_bytes))
    # Published and replacement parsed generations may coexist; encoding,
    # fragments and the joined input are separate source-owned wire buffers.
    manifests = backed * catalog._manifest_bytes * (2 * node + 3)
    if logical_api_limits is None:
        import inspect
        from tensorfold.server.responses import Store

        # The existing lazy default cache still owns deep-copied standard
        # response/message graphs. Read its actual constructor contract, not a
        # second hard-coded 256MiB default or a throwaway Store instance.
        signature = inspect.signature(Store)
        cached = _size(signature.parameters["max_bytes"].default, "default Responses cache bytes")
        cache_limit = _size(signature.parameters["limit"].default, "default Responses cache count")
        # Current cached trees, old pinned generations and their clone/memo
        # work are distinct. Conversation also owns flattened pointer lists.
        cache_current = _json_tree_payload(cached, json_policy)
        cache_copy = (cached * node if json_policy is None else
            90 * cached + _json_copy_working(cached, json_policy)
            + _append_list_payload(cached // 2) + _append_list_payload(cache_limit))
        cache_wrappers = (None if json_policy is None else
                          _responses_cache_wrappers(cache_limit, max_http_connections))
        logical_api = cache_current + max_http_connections * cache_copy
        if cache_wrappers is not None:
            logical_api += cache_wrappers["total_bytes"]
        logical_receipt = {"mode": "existing bounded in-memory Responses cache", "encoded_cache_bytes": cached,
            "cached_tree_bytes": cache_current, "per_http_old_source_clone_memo_bytes": cache_copy,
            "cache_entry_limit": cache_limit, "cache_wrappers": cache_wrappers,
            "projection_status": "existing conservative HTTP baseline; tighter complete HTTP region remains open"}
    else:
        from tensorfold.server.api_records import ApiRecordLimits

        if type(logical_api_limits) is not ApiRecordLimits:
            raise ValueError("logical API host resources require the actual parsed ApiRecordLimits")
        # GET/list materializations and active response operations have
        # separately bounded ownership. Parse temporaries coexist with their
        # checked trees; encoded history is not the incoming body cap.
        logical_api = ((logical_api_limits.max_active_operations + max_http_connections)
                       * logical_api_limits.max_materialized_bytes
                       + max_http_connections * (logical_api_limits.max_history_bytes
                           + logical_api_limits.max_operation_bytes) * node)
        logical_receipt = {"mode": "configured standard SQL records", "limits": asdict(logical_api_limits),
                           "native_sqlite_allocator": "qualification pending; not an RSS certificate"}
    fixed = header.global_host_bytes
    native_fixed, native_prefix, unpaid_fixed, unpaid_prefix = _native_host_commitment(catalog, baseline)
    total = _total((baseline.baseline_host_bytes, fixed, max_waiting_input_bytes, cpu, transport, manifests,
                    logical_api, rpc_python["global_rpc_python_working_bytes"], unpaid_fixed, unpaid_prefix))
    materialized = baseline.materialized_host_bytes
    if catalog._buffer is not None and catalog._buffer.is_pinned():
        materialized += catalog.world * catalog._buffer.nbytes
    # Finalize at startup, before any owned payload/accepted request exists.
    # Local RAM object counts are not a global observation and never provide a
    # rank-multiplied credit. Only equal fixed buffers have that proof here.
    if catalog.has_retained:
        raise ValueError("endpoint retention resources must finalize before request admission")
    record = {"schema": "tensorfold.retention-host-envelope-v1", "context_tokens": context,
        "world": catalog.world, "max_pending_requests": max_pending_requests,
        "max_http_connections": max_http_connections, "output_queue_tokens": output_queue_tokens,
        "max_http_body_bytes": body_bytes, "copy_enabled": copy_enabled,
        "json_payload_policy": (json_policy.record() if json_policy is not None else
            {"mode": "existing conservative builtin-decoder projection", "tight_native_scope": False}),
        "incoming_json": incoming_json,
        "generation_owner_bound": {"version": 1, "owners": generation_owners,
            "mandatory_input_claim_bytes": input_floor,
            "scope": "existing global FIFO input credit through complete logical caller lifetime"},
        "tokenizer": asdict(bound), "header": header.record(), "continuation": terms,
        "global_future_cpu_payload_bytes": cpu, "global_waiting_input_bytes": max_waiting_input_bytes,
        "global_model_cpu_projection_bytes": model_cpu, "global_completed_http_projection_bytes": completed_http,
        "global_http_rpc_error_tail_bytes": http_rpc_errors,
        "global_transport_projection_bytes": transport, "global_manifest_projection_bytes": manifests,
        "tp_rpc_python": rpc_python, "native_fixed": native_fixed, "native_prefix": native_prefix,
        "global_rpc_python_working_bytes": rpc_python["global_rpc_python_working_bytes"],
        "global_unpaid_native_fixed_payload_bytes": unpaid_fixed,
        "global_unpaid_native_prefix_payload_bytes": unpaid_prefix,
        "ordinary_host_baseline": asdict(baseline), "global_logical_api_projection_bytes": logical_api,
        "logical_api": logical_receipt,
        "max_backed_generations": backed,
        "global_host_commitment_bytes": total, "materialized_owned_bytes": materialized,
        "allocator_evidence": "CPython object payload projection; allocator arenas/native RSS unqualified",
        "native_constraints": ("ordinary unretained matcher residency/allocator qualification pending" if catalog.policy is None
                               else "retained/tiered combination refused before compilation"),
        "evidence": catalog.evidence}
    return _admit_host_record(catalog, record, materialized)
