"""Flash Next expert-only follower and per-slot NCCL exchange.

Rank zero alone owns the model's sequence state, routers and shared experts.
Workers return disjoint unweighted slots. The normal model consumer combines
those slots in its original order; NCCL never performs floating-point arithmetic.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import uuid
from dataclasses import replace
from pathlib import Path

from tensorfold.cleanup import raise_failures, rollback
from tensorfold.cuda.expert_parallel import MAX_ROWS, MAX_SLOTS, AffineBank, ownership

TIMEOUT = 600.0


def decode_frame(raw, *, sequence, banks):
    if not isinstance(raw, bytes) or len(raw) > 512:
        raise ValueError("expert control frame exceeds its 512-byte envelope")
    frame = json.loads(raw)
    if (not isinstance(frame, list) or len(frame) != 4
            or any(type(v) is not int for v in frame)):
        raise ValueError("expert control frame requires four native integer fields")
    seq, bank, rows, prefill = frame
    if seq != sequence or not 0 <= seq < 2**63 or prefill not in (0, 1):
        raise ValueError("expert control sequence or arithmetic mode is invalid")
    if bank == -1 and rows == 0 and prefill == 0:
        return None
    if not 0 <= bank < banks or not 1 <= rows <= MAX_ROWS:
        raise ValueError("expert control bank/rows exceeds admitted geometry")
    return bank, rows, bool(prefill)


class DistributedExperts:
    kernel = "expert_parallel"
    capturable = False

    def __init__(self, session, key, local):
        self.session, self.key, self.local = session, key, local
        self.width, self.dims = local.width, local.dims
        self.count = sum(session.owner.counts) + 1

    def device_tensors(self):
        return self.local.device_tensors()


class Session:
    """One serialized expert wire with one bounded work item outstanding per rank.

    Store headers wake idle followers without enqueuing an unbounded CUDA wait.
    Ready acknowledgements precede collectives. Native events plus async error
    checks bound host waits; cleanup aborts first on failure and preserves owners.
    """
    def __init__(self, model_dir, *, ranks, shards, rank, master, port, budget, mtp, pin_mode="off"):
        # Prepare all cleanup recording before acquiring native/model owners.
        # Slots: notification, communicator, reader, cache, source, payloads.
        self._close_outcomes = [None] * 6
        import torch

        from tensorfold.cuda.capacity import config
        from tensorfold.cuda.comm import BACKEND_ENV, NCCL
        from tensorfold.cuda.local_store import open_store

        from ..ram_experts import layout
        from .weight_types import Config

        backend = os.environ.get(BACKEND_ENV, "").strip().lower() or "nccl"
        if backend != "nccl":
            raise ValueError(f"expert parallelism requires {BACKEND_ENV}=nccl; unsupported override {backend!r}")
        self.comm = self.reader = self.cache = None
        self.source_files = []
        self.source_identity = []
        self.closed = self.retired = False
        self.sequence = 0
        self.lock = threading.RLock()
        self.abort_requested = threading.Event()
        self.running = threading.Event()
        self.following = threading.Event()
        self.banks = {}
        self.nv_workspace = {}
        self.packed_cache = None
        self.model_dir, self.budget, self.mtp = Path(model_dir), budget, mtp
        self.pin_mode = pin_mode
        self.rank, self.world, self.timeout = rank, ranks, TIMEOUT
        self.cfg = Config.read(self.model_dir)
        text = config(model_dir)
        self.owner = ownership(ranks, shards, rank, self.cfg.experts)
        if self.cfg.top_k + 1 > MAX_SLOTS:
            raise ValueError("expert parallelism exceeds the native 1024-expert plan geometry")
        if self.cfg.quant not in ("mlx", "modelopt"):
            raise ValueError("expert parallelism supports affine4 and NVFP4 Flash Next checkpoints")
        if budget is None:
            raise ValueError("expert parallelism requires --vram-experts GIB|auto on each rank")
        torch.cuda.set_device(0)
        self.torch = torch
        self.slots, self.dims = self.cfg.top_k + 1, self.cfg.hidden
        base = layout(model_dir, budget, mtp=mtp)
        total = self.cfg.experts + (base.format == "affine4")
        local = len(self.owner.ids) + (rank == 0 and base.format == "affine4")
        if base.host_bytes % total:
            raise ValueError("expert authority geometry does not divide into complete per-expert cells")
        self.layout = replace(base, host_bytes=(base.host_bytes // total) * local)
        # The same-host group must fit collectively before followers allocate
        # their disjoint authorities. Per-rank admission alone races this bound.
        self.group_host_bytes = base.host_bytes + ranks * (base.staging_bytes + base.loading_bytes
            + base.metadata_host_bytes + base.control_host_bytes + (4 << 20))
        if pin_mode not in ("off", "auto", "on"):
            raise ValueError("expert pin mode must be off, auto or on")
        from tensorfold.cuda.expert_pin import backing_bytes

        from ..expert_authority import allocations
        self.allocation_bytes = allocations(text=text, layout=base,
                                           owned=len(self.owner.ids), shared=rank == 0)
        self.layout = replace(self.layout, host_bytes=sum(self.allocation_bytes),
                              authority_allocations=self.allocation_bytes)
        self.pin_future_bytes = 0
        if pin_mode != "off":
            backing = sum(backing_bytes(size) for r, count in enumerate(self.owner.counts)
                          for size in allocations(text, base, owned=count, shared=r == 0))
            self.group_host_bytes += max(0, backing - base.host_bytes)
        self.workspace_bytes = MAX_ROWS * (self.dims * 2 + self.slots * 4 + self.slots * self.dims * 12
                                           + self.slots * self.cfg.moe_width * 2) + (4 << 20)
        try:
            self.store = open_store(master, port, ranks, rank, timeout_s=self.timeout)
            identity = self._identity(text)
            self.store.set(f"ep/identity/{rank}", json.dumps(identity))
            peers = [self._read_json(f"ep/identity/{r}", 1 << 20) for r in range(ranks)]
            if (any(p[1:] != peers[0][1:] for p in peers)
                    or len({p[0] for p in peers}) != ranks):
                raise ValueError("expert ranks require distinct GPUs and matching checkpoint, geometry and expert pin policy")
            self.comm = NCCL()
            self.comm.open(rank, ranks, master, port, store=self.store, key="tf_ep_nccl_uid")
        except BaseException as error:  # noqa: BLE001 - rollback re-raises cancellation and failure
            rollback(self, error, lambda: self.close(abort=True))

    def _identity(self, text):
        prop = self.torch.cuda.get_device_properties(0)
        device_uuid = str(prop.uuid)
        with Path("/proc/sys/kernel/random/boot_id").open("rb") as host:
            host_epoch = str(uuid.UUID(host.read(64).decode("ascii").strip()))
        digest = hashlib.sha256()
        for name in ("config.json", "model.safetensors.index.json"):
            from tensorfold.cuda.tensor_file import checkpoint_path
            path = checkpoint_path(self.model_dir, name)
            if not path.exists() and name.endswith(".index.json"):
                continue
            if path.stat().st_size > 96 << 20:
                raise ValueError("expert checkpoint metadata exceeds the 96MiB boundary")
            with path.open("rb") as metadata:
                raw = metadata.read((96 << 20) + 1)
                if len(raw) > 96 << 20:
                    raise ValueError("expert checkpoint metadata exceeds the 96MiB boundary")
                digest.update(raw)
        from tensorfold.cuda.tensor_file import checkpoint_path

        from .reader import checkpoint_inventory
        inventory = checkpoint_inventory(self.model_dir)
        files = []
        for shard in sorted(set(inventory.values())):
            from tensorfold.families.qwen4_exp.table_file import TableFile
            file = TableFile(checkpoint_path(self.model_dir, shard))
            self.source_files.append(file)
            file.open()
            st = os.fstat(file.fileno())
            files.append([shard, st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns])
            self.source_identity.append(files[-1])
        if len(json.dumps(files).encode()) > 1 << 19:
            raise ValueError("expert checkpoint source identity exceeds 512KiB")
        return [device_uuid, host_epoch, digest.hexdigest(), list(self.owner.counts), self.slots, self.dims,
                self.cfg.moe_width, bool(self.mtp), self.cfg.quant, self.pin_mode, files]

    def _read_json(self, key, maximum):
        raw = self.store.get(key)
        if not isinstance(raw, bytes) or len(raw) > maximum:
            raise ValueError("expert rendezvous metadata exceeds its envelope")
        return json.loads(raw)

    def _receive(self, key, *, idle=False):
        store = self.store  # retain the listener/borrow through this accepted control operation
        deadline = None if idle else time.monotonic() + self.timeout
        while not store.check([key]):
            if self.abort_requested.is_set():
                raise RuntimeError("expert group retirement interrupted control wait")
            for rank in range(self.world):
                if store.check([f"ep/failed/{rank}"]):
                    raise RuntimeError(f"expert rank {rank} failed; the group must retire")
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("expert control acknowledgement exceeded its deadline")
            time.sleep(.01)
        return store.get(key)

    def admit_followers(self):
        """Leader's full host/backbone admission completed; followers may allocate their banks."""
        if self.rank != 0:
            raise RuntimeError("only the expert leader grants group admission")
        self.store.set("ep/admitted", "ready")

    def _ensure_reader(self):
        if self.reader is None:
            from .reader import _Reader
            self.reader = _Reader(self.model_dir, "cpu")

    def _allocate_buffers(self):
        if hasattr(self, "x"):
            return
        torch = self.torch
        self.x = torch.empty((MAX_ROWS, self.dims), dtype=torch.bfloat16, device="cuda")
        self.pick = torch.empty((MAX_ROWS, self.slots), dtype=torch.int32, device="cuda")
        self.output = torch.empty((MAX_ROWS, self.slots, self.dims), dtype=torch.float32, device="cuda")
        self.peer = torch.empty_like(self.output)
        self.low = torch.empty_like(self.output, dtype=torch.bfloat16)

    def _check_source(self):
        from tensorfold.cuda.tensor_file import checkpoint_path
        for identity in self.source_identity:
            name, *expected = identity
            st = os.stat(checkpoint_path(self.model_dir, name))
            if [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns] != expected:
                raise ValueError("expert checkpoint source changed during loading")

    def load_bank(self, reader, name, prefix, cfg, kind):
        """Called before the ordinary loader reads any unowned expert payload."""
        from tensorfold.cuda.expert_cache import CachedExperts, HostExpertCache

        self._ensure_reader()
        self._check_source()
        if self.cache is None:
            from tensorfold.cuda.expert_pin import ExpertPinPolicy
            self._allocate_buffers()
            policy = ExpertPinPolicy(sum(self.allocation_bytes), mode=self.pin_mode,
                                     allocation_bytes=self.allocation_bytes,
                                     future_bytes=self.layout.loading_bytes + self.layout.staging_bytes
                                     + self.layout.metadata_host_bytes + self.layout.control_host_bytes
                                     + self.pin_future_bytes)
            self.cache = HostExpertCache(self.layout.gpu_bytes, self.layout.entry_bytes, "cuda", pin_policy=policy)
            if kind == "mlx":
                from tensorfold.cuda.packed_experts import PackedExpertCache
                self.packed_cache = PackedExpertCache.from_environment()
        key = prefix + name
        if key in self.banks:
            raise ValueError("expert layer registered twice")
        if kind == "mlx":
            from tensorfold.cuda.host_experts import load_host_experts
            host = load_host_experts(self.reader, key, owned_ids=self.owner.ids, include_shared=self.rank == 0,
                                     authority=self.cache, packed_cache=self.packed_cache)
            local = AffineBank(CachedExperts(host, self.cache, len(self.banks)))
        else:
            from .nvfp4_moe import load_moe_host
            local = load_moe_host(self.reader, key, cfg, "cuda", cache=self.cache,
                                  owned_ids=self.owner.ids, include_shared=self.rank == 0,
                                  workspace=self.nv_workspace)
        self.banks[key] = local
        return DistributedExperts(self, key, local)

    def finish_loading(self):
        self._check_source()
        if self.reader is not None:
            self.reader.close()
            self.reader.release()
            self.reader = None
        keys = sorted(self.banks)
        self.keys = keys
        manifest = [[k, self.banks[k].width, self.banks[k].dims] for k in keys]
        body = json.dumps(manifest)
        if len(body.encode()) > 65536:
            raise ValueError("expert bank manifest exceeds 64KiB")
        self.store.set(f"ep/banks/{self.rank}", body)
        if any(self._read_json(f"ep/banks/{r}", 65536) != manifest for r in range(self.world)):
            raise ValueError("expert ranks disagree on registered layer geometry")

    def _wait(self):
        event = self.torch.cuda.Event()
        event.record(self.torch.cuda.current_stream())
        deadline = time.monotonic() + self.timeout
        while not event.query():
            if self.abort_requested.is_set():
                raise RuntimeError("expert group retirement interrupted native wait")
            self.comm.check()
            if time.monotonic() >= deadline:
                raise TimeoutError("expert rank operation exceeded its completion deadline")
            time.sleep(.001)
        self.comm.check()

    def _publish(self, bank, rows, prefill):
        if self.sequence >= 2**63:
            raise OverflowError("expert work sequence exhausted")
        frame = json.dumps([self.sequence, bank, rows, int(prefill)])
        for rank in range(1, self.world):
            self.store.set(f"ep/work/{rank}", frame)
        ready = [f"ep/ready/{rank}" for rank in range(1, self.world)]
        for key in ready:
            if self._receive(key) != str(self.sequence).encode():
                raise RuntimeError("expert follower acknowledged another work sequence")
            self.store.delete_key(key)

    def run(self, distributed, x, picks, out, *, prefill):
        if self.rank != 0 or self.closed:
            raise RuntimeError("expert leader admission is closed")
        with self.lock:
            if self.closed or self.abort_requested.is_set():
                raise RuntimeError("expert leader admission is closed")
            self.running.set()
            try:
                bank = self.keys.index(distributed.key)
                for start in range(0, len(x), MAX_ROWS):
                    rows = min(MAX_ROWS, len(x) - start)
                    self.x[:rows].copy_(x[start:start + rows])
                    self.pick[:rows].copy_(picks[start:start + rows])
                    self._publish(bank, rows, prefill)
                    self.comm.broadcast(self.x[:rows])
                    self.comm.broadcast(self.pick[:rows])
                    target = self.low[:rows] if prefill else self.output[:rows]
                    borrowed = distributed.local.run_picks(self.x[:rows], self.owner.local(self.pick[:rows]), target,
                                                           prefill=prefill)
                    self.output[:rows].copy_(target)
                    for rank in range(1, self.world):
                        self.comm.recv(self.peer[:rows], rank)
                        begin, end = sum(self.owner.counts[:rank]), sum(self.owner.counts[:rank + 1])
                        mask = ((self.pick[:rows] >= begin) & (self.pick[:rows] < end)).unsqueeze(-1)
                        self.output[:rows].copy_(self.torch.where(mask, self.peer[:rows], self.output[:rows]))
                    out[start:start + rows].copy_(self.output[:rows])
                    self._wait()
                    del borrowed
                    self.sequence += 1
            except BaseException as error:  # noqa: BLE001 - rollback re-raises cancellation and failure
                rollback(self, error, lambda: self.close(abort=True))
            finally:
                self.running.clear()

    def follow(self):
        if self.rank == 0:
            raise RuntimeError("expert leader cannot enter the follower loop")
        with self.lock:
            if self.closed:
                raise RuntimeError("expert follower admission is closed")
            self.following.set()
            try:
                while True:
                    raw = self._receive(f"ep/work/{self.rank}", idle=True)
                    self.store.delete_key(f"ep/work/{self.rank}")
                    work = decode_frame(raw, sequence=self.sequence, banks=len(self.keys))
                    self.store.set(f"ep/ready/{self.rank}", str(self.sequence))
                    if work is None:
                        break
                    bank, rows, prefill = work
                    self.comm.broadcast(self.x[:rows])
                    self.comm.broadcast(self.pick[:rows])
                    target = self.low[:rows] if prefill else self.output[:rows]
                    borrowed = self.banks[self.keys[bank]].run_picks(self.x[:rows], self.owner.local(self.pick[:rows]), target,
                                                                    prefill=prefill)
                    self.output[:rows].copy_(target)
                    self.comm.send(self.output[:rows], 0)
                    self._wait()
                    del borrowed
                    self.sequence += 1
            except BaseException as error:  # noqa: BLE001 - rollback re-raises cancellation and failure
                rollback(self, error, lambda: self.close(abort=True))
            finally:
                self.following.clear()

    def close(self, *, abort=False):
        if type(abort) is not bool:
            raise ValueError("typed expert-group abort policy required")
        if abort or self.running.is_set() or self.following.is_set():
            self.abort_requested.set()  # wake an owner before waiting for its wire lease
        with self.lock:
            if self.retired:
                return
            failed = abort or self.closed or self.abort_requested.is_set()
            self.closed = True
            outcomes = self._close_outcomes
            outcomes[0] = outcomes[1] = outcomes[2] = outcomes[3] = outcomes[4] = outcomes[5] = None
            if failed and getattr(self, "store", None) is not None:
                try:
                    self.store.set(f"ep/failed/{self.rank}", "failed")
                except BaseException as error:  # noqa: BLE001 - cleanup failures are aggregated
                    outcomes[0] = error
            elif not failed and self.rank == 0 and getattr(self, "keys", None) is not None:
                try:
                    self._publish(-1, 0, False)
                except BaseException as error:  # noqa: BLE001 - abort native work after a failed stop frame
                    outcomes[0] = error
                    failed = True
            if self.comm is not None:
                try:
                    self.comm.close(abort=failed)
                except BaseException as error:  # noqa: BLE001 - retain unresolved native owner
                    outcomes[1] = error
                else:
                    self.comm = None
            if self.reader is not None:
                try:
                    self.reader.close()
                except BaseException as error:  # noqa: BLE001 - retain unresolved reader
                    outcomes[2] = error
                else:
                    self.reader = None
            # Payloads cannot retire until native work and accepted readers have
            # reached terminal completion, including on communicator loss.
            if self.comm is None and self.reader is None:
                if self.cache is not None:
                    try:
                        self.cache.close()
                    except BaseException as error:  # noqa: BLE001 - retain unresolved leases
                        outcomes[3] = error
                    else:
                        self.cache = None
                while self.source_files:
                    file = self.source_files[-1]
                    try:
                        file.close()
                    except BaseException as error:  # noqa: BLE001 - report and retain failed file owner
                        outcomes[4] = error
                        break
                    self.source_files.pop()
            if self.comm is self.reader is self.cache is None and not self.source_files:
                # Wire/native consumers are terminal before their banks, shared
                # NV scratch and transport tensors lose the session's ownership.
                # An unresolved close retains every one of these references.
                try:
                    self.banks.clear()
                    self.nv_workspace.clear()
                    self.packed_cache = None
                    for name in ("x", "pick", "output", "peer", "low"):
                        self.__dict__.pop(name, None)
                    self.store = None
                    self.retired = True  # publish only after every owned reference has retired
                except BaseException as error:  # noqa: BLE001 - retain prior failures and permit partial-release retry
                    outcomes[5] = error
            # Fast locals retain each original status even if preparing the
            # diagnostic list fails and a later close reuses the fixed journal.
            notification_error = outcomes[0]
            communicator_error = outcomes[1]
            reader_error = outcomes[2]
            cache_error = outcomes[3]
            source_error = outcomes[4]
            payload_error = outcomes[5]
            if notification_error is not None:
                primary = notification_error
            elif communicator_error is not None:
                primary = communicator_error
            elif reader_error is not None:
                primary = reader_error
            elif cache_error is not None:
                primary = cache_error
            elif source_error is not None:
                primary = source_error
            else:
                primary = payload_error
            try:
                errors = [error for error in outcomes if error is not None]
                raise_failures(primary, errors)
            except BaseException as failure:
                if primary is None or failure is primary:
                    raise
                # Transport exhaustion preserves all original statuses/owners
                # in this frame rather than replacing the primary failure.
                raise primary from failure


class Follower:
    """CLI owner whose sole runtime is expert work, without any backbone or sequence cache."""
    def __init__(self, session):
        self.session, self.w = session, None
        try:
            if session._receive("ep/admitted") != b"ready":
                raise RuntimeError("expert leader refused group admission")
            self.session._ensure_reader()
            from tensorfold.cuda.capacity import available_bytes, cuda_limit_bytes, host_cgroup_bytes, host_stream_bytes
            host_need = (session.layout.host_bytes + session.layout.staging_bytes
                         + session.layout.loading_bytes + session.layout.metadata_host_bytes)
            rooms = [room for room in (host_stream_bytes(), host_cgroup_bytes()) if room is not None]
            if rooms and host_need > min(rooms):
                raise MemoryError("expert follower host authority/staging exceeds admitted host capacity")
            room = available_bytes(session.torch)
            limit = cuda_limit_bytes()
            if limit is not None:
                room = min(room, max(0, limit - int(session.torch.cuda.memory_allocated())))
            if (session.layout.gpu_bytes + session.workspace_bytes
                    + getattr(session.layout, "loading_device_bytes", 0) > room):
                raise MemoryError("expert follower device pool/loading/workspace exceeds admitted device capacity")
            names = session.reader.where
            prefix = "language_model." if "language_model.model.embed_tokens.weight" in names else ""
            base = "model.language_model." if "model.language_model.embed_tokens.weight" in names else "model."
            banks = [f"{prefix}{base}layers.{i}.mlp" for i in range(session.cfg.layers)]
            if session.mtp and f"{prefix}mtp.fc_embedding.weight" in names:
                banks.append(f"{prefix}mtp.layers.0.mlp")
            for key in banks:
                session.load_bank(None, key, "", session.cfg, session.cfg.quant)
            session.finish_loading()
            if session.budget == "auto":
                self._configure_auto_experts()
        except BaseException as error:  # noqa: BLE001 - rollback re-raises cancellation and failure
            rollback(self, error, lambda: self.close(abort=True))

    def _configure_auto_experts(self):
        """Spend remaining absolute/device room while retaining bounded future expert workspace."""
        from tensorfold.cuda.capacity import cuda_limit_bytes

        from ..ram_experts import AUTO_HEADROOM_BYTES, auto_pool_bytes

        session, torch = self.session, self.session.torch
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free, total = map(int, torch.cuda.mem_get_info())
        allocated = int(torch.cuda.memory_allocated())
        limit = cuda_limit_bytes()
        room = free if limit is None else max(0, min(free, limit - allocated))
        decision = auto_pool_bytes(room, session.cache.entry_bytes, session.cache.score_history_entries,
                                   workspace_bytes=session.workspace_bytes)
        session.cache.configure_before_use(decision["gpu_bytes"])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        final_free, final_total = map(int, torch.cuda.mem_get_info())
        final_allocated = int(torch.cuda.memory_allocated())
        final_spare = int(torch.cuda.memory_reserved()) - final_allocated
        future = AUTO_HEADROOM_BYTES + session.workspace_bytes
        if (final_total != total or final_free < AUTO_HEADROOM_BYTES
                or final_free + final_spare < future
                or (limit is not None and final_allocated + future > limit)):
            raise MemoryError("automatic follower expert pool cannot retain its declared workspace")
        session.auto_pool_decision = {**decision, "explicit_device_limit_bytes": limit,
                                      "fixed_allocated_bytes": allocated,
                                      "final_allocated_bytes": final_allocated}

    def follow(self):
        self.session.follow()

    def close(self, *, abort=False):
        self.session.close(abort=abort)
