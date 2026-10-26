"""Model-owned startup, bounded input arenas and leader-only CPU media resources.

The explicit profile is source admission, never a hardware qualification. The
existing startup Store owns content/software agreement before CUDA acquisition;
the resident activation then owns one encoder and one producer per TP rank.
"""
from __future__ import annotations

import math
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch

from tensorfold.cleanup import rollback

from .media_producer import MediaInputProducer, producer_resources


def media_unit_limit(envelope):
    """Whole attention units, distinct from the encoded replay-file quota."""
    videos = min(envelope.max_videos, envelope.max_replay_items)
    images = min(envelope.max_images, envelope.max_replay_items - videos)
    return min(envelope.max_expanded_rows, images + videos * ((envelope.max_selected_frames + 1) // 2))


def media_phase(engine, label, operation, *, may_refuse=False):
    """Use the existing fatal-aware CPU cohort before dependent device work."""
    from .execution_identity import _ManifestRefusal, _fatal, _phase, _retire_store

    channel = "tensorfold/flashnext/media-startup/" + label

    def retire():
        try:
            _retire_store(engine, channel, stages=("prepare",))
        except BaseException as error:
            _fatal(engine, error)
            raise

    def perform():
        try:
            result = _phase(engine, channel, "prepare", operation, may_refuse=may_refuse)
        except _ManifestRefusal as error:
            rollback(engine, error, retire)
        retire()
        return result

    construction = getattr(engine.comm, "_profile_construction", None)
    if construction is not None and construction.active:
        return construction.protocol("media-" + label, perform, refusal=_ManifestRefusal)
    return perform()


def media_resources(profile, *, hidden, world):
    """Exact numeric payloads and declared work; allocator/RSS remain separate."""
    from tensorfold.cuda.direct_read import ALIGN
    from .gguf_weights import STAGING_BYTES

    execution, cpu = profile.execution, profile.cpu
    result = producer_resources(rows=execution.input_rows, patches=execution.encoder_patches,
                                cache_rows=execution.projected_cache_rows, hidden=hidden,
                                world=world, encoder_rank=execution.encoder_rank)
    # The pinned independent axial rotary has two FP32 half-head vectors.
    frequencies = 2 * (profile.layout.hidden // profile.layout.heads // 4) * 4
    resident = profile.mmproj_plan.resident_bytes + frequencies
    static = list(result["static_device_bytes_by_rank"])
    static[execution.encoder_rank] += resident
    transient = list(result["temporary_device_bytes_by_rank"])
    transient[execution.encoder_rank] += execution.vision_workspace_bytes
    # Final CPU parameters plus temporal stack parts; reads may retain aligned
    # raw storage, FP32 decode, FP16 cast and finite masks concurrently.
    temporal = max((weight.resident_bytes for weight in profile.mmproj_plan.weights.values()
                    if weight.temporal_stack), default=0)
    import_host = profile.mmproj_plan.resident_bytes + temporal + 8 * STAGING_BYTES + 3 * ALIGN + frequencies
    host_payload = (cpu.max_staging_bytes + cpu.max_replay_bytes
                    + result["cpu_pixel_cast_overlap_bytes"]
                    + world * result["cpu_coordinate_arrays_bytes"])
    if any(value >= 1 << 63 for value in (*static, *transient, import_host, host_payload)):
        raise ValueError("combined media resources exceed signed-int64 accounting")
    return {**result, "vision_weight_bytes": profile.mmproj_plan.resident_bytes,
            "vision_frequency_bytes": frequencies, "static_with_vision_bytes_by_rank": tuple(static),
            "transient_with_vision_bytes_by_rank": tuple(transient),
            "media_transient_by_rank": tuple(transient),
            "vision_loading_host_payload_bound": import_host,
            "shared_host_payload_bound": host_payload,
            "worker_address_scope": "per-child virtual RLIMIT_AS; not physical RSS",
            "host_scope": "bounded numeric/file payload; parent Python/allocator RSS qualification pending"}


def prepare_media_source(engine, profile):
    """Hash the complete input closure once, before loading target/vision data."""
    from tensorfold.cuda.sm70.media_software_binding import issue_media_binding
    from tensorfold.engine.model_identity import capture_shared_model_identity
    from .execution_identity import _ManifestRefusal, _fatal, _phase, _retire_store, artifact_files, digest

    cohort = SimpleNamespace(rank=engine.rank, tp=engine.tp, multi=None,
                             comm=SimpleNamespace(store=engine._startup_store))
    channel = "tensorfold/flashnext/media-source"
    stages = ("preflight", "inventory", "agreement", "content", "verification", "issuance", "adoption")
    local = {}

    def exchange(stage, operation):
        return _phase(cohort, channel, stage, operation)

    def preflight():
        files, _ = artifact_files(engine)
        # Keep this closure explicit even for callers whose general inventory
        # predates media. The same paths remain pinned throughout serving.
        files.update({"media/profile": profile.manifest_path, "media/vision": profile.vision_path})
        local["files"] = files
        sizes = [path.stat().st_size for path in files.values()]
        local["maximum"], local["total"] = max(sizes), sum(sizes)
        return digest({"profile": profile.identity, "paths": {key: str(value) for key, value in files.items()}})

    try:
        try:
            selected = exchange("preflight", preflight)
            if any(value != selected[0] for value in selected):
                raise _ManifestRefusal("media ranks selected different source/profile inputs")
            pairing = getattr(engine, "_dflash_pairing", None)
            prior = None if pairing is None else pairing.authority
            runtime_identity = ({"schema": "tensorfold.sm70-media-source", "version": 1}
                                if prior is None else json.loads(prior.runtime_json))
            authority = capture_shared_model_identity(local["files"],
                runtime_identity=runtime_identity,
                authorize=lambda _name, path: path.resolve(strict=True),
                max_file_bytes=local["maximum"], max_total_bytes=local["total"],
                rank=engine.rank, world=engine.tp, exchange=exchange,
                reuse=prior, reuse_is_immutable=prior is not None)

            def adopt():
                profile.verify_source(authority)
                local["software"] = issue_media_binding()
                return local["software"].record()["identity_sha256"]

            software = exchange("adoption", adopt)
            if any(value != software[0] for value in software):
                raise _ManifestRefusal("media ranks adopted different protected software")
        except _ManifestRefusal as error:
            rollback(cohort, error, lambda: _retire_store(cohort, channel, stages=stages))
        _retire_store(cohort, channel, stages=stages)
        return authority, local["software"]
    except BaseException as error:
        if not isinstance(error, _ManifestRefusal) or getattr(cohort, "_execution_manifest_uncertain", False):
            engine._abort_comm = True
            _fatal(cohort, error)
        raise


class MediaRuntime:
    """Actual partial owner used by Engine.vision on the explicit SM70 path."""
    is_media_runtime = True

    @classmethod
    def create(cls, engine, profile, *, authority, software_binding, allow_urls, register):
        from tensorfold.cuda.direct_read import Reader
        from .gguf_weights import STAGING_BYTES
        from .mmproj_vision import SM70VisionEncoder

        if not callable(register) or type(allow_urls) is not bool:
            raise ValueError("media runtime requires its model owner journal and typed URL policy")
        owner = cls.__new__(cls)
        owner.engine, owner.profile, owner.authority = engine, profile, authority
        owner.software_binding, owner.allow_urls = software_binding, allow_urls
        owner.pid, owner.closed = os.getpid(), False
        owner.encoder = owner.producer = owner.reader = owner.directory = None
        owner.directory_failure = None
        owner.store = owner.executor = owner.frontend = owner.planner = owner.contract = owner.tokenizer = owner.work = owner.requests = None
        owner.identity = owner._warm_output = owner._warm_pixels = None
        owner.actual_encoder_bytes = 0
        owner.videos, owner.media_envelope = bool(profile.cpu.max_videos), profile.cpu
        owner.resources = media_resources(profile, hidden=engine.w.cfg.hidden, world=engine.tp)
        owner.resources["media_transient_bytes"] = owner.resources["media_transient_by_rank"][engine.rank]
        owner.warm_receipt = None
        register(owner)
        def load():
            profile.verify_source(authority)
            if engine.rank == profile.execution.encoder_rank:
                owner.reader = Reader()
                owner.encoder = SM70VisionEncoder.create(profile.mmproj_plan, owner.reader,
                    staging_bytes=STAGING_BYTES, device=engine.w.device,
                    max_unit_patches=profile.execution.encoder_patches, software_binding=software_binding,
                    register=lambda encoder: setattr(owner, "encoder", encoder))
                owner.actual_encoder_bytes = sum(value.nbytes for value in owner.encoder.model.parameters()) + sum(
                    value.nbytes for value in owner.encoder.model.buffers())
                expected = owner.resources["vision_weight_bytes"] + owner.resources["vision_frequency_bytes"]
                if owner.actual_encoder_bytes != expected:
                    raise MemoryError("loaded vision parameter/buffer payload differs from its admitted geometry")
                owner.reader.close()
                owner.reader = None
            authority.verify_unchanged()
            return {"rank": engine.rank, "encoder": owner.encoder is not None,
                    "encoder_bytes": owner.actual_encoder_bytes}

        placed = media_phase(engine, "encoder-load", load)
        resident = owner.resources["vision_weight_bytes"] + owner.resources["vision_frequency_bytes"]
        if any(item != {"rank": rank, "encoder": rank == profile.execution.encoder_rank,
                        "encoder_bytes": resident if rank == profile.execution.encoder_rank else 0}
               for rank, item in enumerate(placed)):
            raise RuntimeError("media encoder placement differs from its admitted unique owner")
        # Materialize the independent vision path before issuing the runtime
        # namespace. Later producer transport warmup adds no encoder/provider.
        receipts = media_phase(engine, "encoder-warm", owner._warm_encoder)
        owner.warm_receipt = {"encoder_ranks": receipts, "transport": "pending"}
        return owner

    def _warm_encoder(self):
        from tensorfold.cuda.sm70.media_software_binding import reissue_media_binding

        self.engine._media_numeric_scope.verify()
        receipt = {"rank": self.engine.rank, "encoder": False}
        if self.encoder is not None:
            patches = self.profile.execution.encoder_patches
            groups = patches // 4
            height = math.isqrt(groups)
            while groups % height:
                height -= 1
            grid = torch.tensor([[1, 2 * height, 2 * (groups // height)]], dtype=torch.int64, device="cpu")
            device = self.engine.w.device
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_allocated(device)
            free_before = int(torch.cuda.mem_get_info(device)[0])
            torch.cuda.reset_peak_memory_stats(device)
            self._warm_pixels = torch.zeros((patches, 1536), dtype=torch.float16, device=device)
            self._warm_output = self.encoder.encode_device_unit(self._warm_pixels, grid)
            self.encoder._retire_inputs()
            peak = max(0, torch.cuda.max_memory_allocated(device) - before - self._warm_pixels.nbytes)
            if peak > self.profile.execution.vision_workspace_bytes:
                raise MemoryError("largest-unit vision warmup exceeds its declared workspace bound")
            self._warm_output = self._warm_pixels = None
            torch.cuda.synchronize(device)
            receipt = {"rank": self.engine.rank, "encoder": True,
                "scope": "largest-unit synthetic startup shape; no model/quality/performance qualification",
                "patches": patches, "grid": grid.tolist(), "peak_allocator_extra_bytes": peak,
                "physical_free_delta_bytes": free_before - int(torch.cuda.mem_get_info(device)[0]),
                "external_workspace_scope": "before/after free only, including allocator caching; transient external peaks require V100 qualification"}
            self.encoder.adopt_after_warm()
            self.software_binding = self.encoder.software_binding
        else:
            self.software_binding = reissue_media_binding(self.software_binding)
        receipt["software"] = self.software_binding.record()
        return receipt

    def bind(self, weights):
        """Publish producer only after actual coordinate/context installation."""
        from tensorfold.token_metadata import load_tokenizer, tokenizer_identity, tokenizer_override
        from tensorfold.vision.media_frontend import MediaExecutor, ReplayMediaFrontend
        from tensorfold.vision.replay import ReplayStore
        from tensorfold.vision.media_work import MediaWork
        from ..gguf import GGUFCheckpoint, resolve_gguf
        from ..media_plan import MediaPlanContract, QwenMediaPlanner
        from .execution_identity import digest
        from .media_requests import MediaRequests

        if self.closed or self.identity is not None or weights is not self.engine.w:
            raise ValueError("media runtime cannot bind another or already active model")
        local = {}

        def metadata():
            self.authority.verify_unchanged()
            self.identity = weights.meta["coordinate_profile_identity"]
            files = {item.logical_name: item for item in self.authority.files}
            tokenizer_file = files["metadata/tokenizer.json"]
            override, override_file = tokenizer_override(), files.get("metadata/tokenizer_override.json")
            if ((override is None) != (override_file is None)
                    or override is not None and override.sha256 != override_file.sha256):
                raise ValueError("media tokenizer override differs from its captured source authority")
            template = self.engine.chat_template
            if template is None:
                raise ValueError("media startup requires the model's selected chat template")
            template_identity = digest({"routes": [template.provenance(tools=route)["sha256"] for route in (False, True)]})
            checkpoint = GGUFCheckpoint.open(resolve_gguf(self.engine.model_dir))
            image_id = checkpoint.metadata.get("qwen4exp.ple.image_token_id")
            if type(image_id) is not int or not 0 <= image_id < weights.cfg.vocab:
                raise ValueError("media startup requires the actual GGUF image/PLE token role")
            if self.engine.rank == 0:
                self.tokenizer = load_tokenizer(self.engine.model_dir)
                selected_tokenizer = tokenizer_identity(self.tokenizer)
            else:
                selected_tokenizer = None
            self.authority.verify_unchanged()
            local.update(template_identity=template_identity, image_id=image_id)
            return {"profile": self.identity, "tokenizer_source": tokenizer_file.sha256,
                    "tokenizer": selected_tokenizer,
                    "template": template_identity, "image_id": image_id}

        sources = media_phase(self.engine, "planner-source", metadata, may_refuse=True)
        source = {key: value for key, value in sources[0].items() if key != "tokenizer"}
        if (sources[0]["tokenizer"] is None
                or any({key: value for key, value in item.items() if key != "tokenizer"} != source
                       or item["tokenizer"] is not None for item in sources[1:])):
            raise RuntimeError("media ranks selected different source/tokenizer/template roles")
        selected_tokenizer = sources[0]["tokenizer"]
        template_identity, image_id = local["template_identity"], local["image_id"]
        max_segments = media_unit_limit(self.profile.cpu)

        def plan():
            if self.engine.rank != 0:
                return None
            self.planner = QwenMediaPlanner(self.tokenizer, profile_identity=self.identity,
                tokenizer_identity=selected_tokenizer, template_identity=template_identity,
                vision_identity=self.profile.vision_sha256, processor_id=self.profile.processor_id,
                expected_image_id=image_id, max_rows=self.engine.context_window,
                max_segments=max_segments,
                max_visual_rows=self.profile.execution.encoder_patches // 4,
                max_total_visual_rows=self.profile.cpu.max_expanded_rows,
                max_rendered_bytes=self.profile.execution.max_rendered_bytes,
                max_expanded_bytes=self.profile.execution.max_expanded_bytes)
            self.authority.verify_unchanged()
            return self.planner.contract.record()

        records = media_phase(self.engine, "planner", plan, may_refuse=True)

        def adopt():
            if any(record is not None for record in records[1:]):
                raise ValueError("only the media leader may issue the tokenizer contract")
            self.contract = MediaPlanContract.from_record(records[0])
            if (self.contract.identities[0] != self.identity
                    or self.contract.identities[1] != selected_tokenizer
                    or self.contract.identities[2] != template_identity
                    or self.contract.identities[3] != self.profile.vision_sha256
                    or self.contract.roles[0] != image_id
                    or self.contract.processor_id != self.profile.processor_id
                    or self.contract.vocab_size != weights.cfg.vocab
                    or self.contract.max_rows != self.engine.context_window
                    or self.contract.max_visual_rows != self.profile.execution.encoder_patches // 4
                    or self.contract.max_total_visual_rows != self.profile.cpu.max_expanded_rows
                    or self.contract.max_segments != max_segments):
                raise ValueError("media contract differs from its actual resident profile/model")
            if self.engine.rank == 0:
                self._create_replay_directory()
                self.store = ReplayStore(self.directory, self.profile.cpu)
                self.executor = MediaExecutor(self.store)
                self.work = MediaWork.create(self.executor, register=lambda owner: setattr(self, "work", owner))
                self.requests = MediaRequests(self)
                self.frontend = ReplayMediaFrontend(self.store, self.executor, self.planner,
                    cancelled=lambda: self.engine._closing, allow_urls=self.allow_urls)
            return self.contract.record()

        adopted = media_phase(self.engine, "contract", adopt)
        if any(record != adopted[0] for record in adopted):
            raise RuntimeError("media ranks adopted different tokenizer/position contracts")
        execution = self.profile.execution
        def allocate():
            self.producer = MediaInputProducer.create(weights, encoder=self.encoder,
                encoder_rank=execution.encoder_rank, rows=execution.input_rows, patches=execution.encoder_patches,
                cache_rows=execution.projected_cache_rows, register=lambda value: setattr(self, "producer", value))
            return {"rank": self.engine.rank}
        media_phase(self.engine, "producer", allocate)

    def _create_replay_directory(self):
        # Journal the acquired string before Path allocation or canonicalization
        # can fail. close accepts either representation and spends it once.
        self.directory = tempfile.mkdtemp(prefix="tensorfold-media-")
        self.directory = Path(self.directory).resolve()

    def frontend_for(self, cancelled):
        from tensorfold.vision.media_frontend import ReplayMediaFrontend

        self.check_media_health()
        if self.frontend is None:
            raise ValueError("only the leader owns CPU media preparation")
        return ReplayMediaFrontend(self.store, self.executor, self.planner,
                                   cancelled=cancelled, allow_urls=self.allow_urls)

    def check_media_health(self):
        if os.getpid() != self.pid or self.closed or self.producer is None:
            raise RuntimeError("media runtime is retired, unbound or belongs to another process")
        if self.work is not None:
            self.work.check_health()
        if self.executor is not None:
            self.executor.check_health()

    def prepare_sources(self, *args, **kwargs):
        self.check_media_health()
        if self.frontend is None:
            raise ValueError("only the leader owns media sources")
        return self.frontend.prepare_sources(*args, **kwargs)

    def continued(self, prepared, token_ids):
        """Append canonical text while lending the same logical replay owner.

        The logical App request retains final close authority across backend
        subruns. This view acquires no replay borrow, worker, file or pixels.
        """
        from tensorfold.vision.media_frontend import PreparedReplayMedia
        from tensorfold.vision.replay import ReplayLease

        self.check_media_health()
        if (self.executor is None or type(prepared) is not PreparedReplayMedia
                or type(prepared.replay) is not ReplayLease or prepared.executor is not self.executor
                or prepared.replay._store is not self.store
                or prepared.plan.profile_identity != self.identity
                or self.contract.identities[0] != self.identity
                or type(token_ids) not in (list, tuple)
                or not len(prepared.token_ids) < len(token_ids) <= self.contract.max_rows
                or tuple(token_ids[:len(prepared.token_ids)]) != prepared.token_ids):
            raise ValueError("media continuation requires this live runtime and an unchanged canonical prefix extension")
        with self.store._lock:
            self.store._check_owner()
            if prepared.replay._closed or prepared.replay._retiring:
                raise ValueError("media continuation cannot reuse a retired replay owner")
            plan = self.contract.from_expanded(token_ids, prepared.probes)
            if (plan.units != prepared.plan.units or plan.probe_identities != prepared.plan.probe_identities
                    or plan.media_identity != prepared.plan.media_identity):
                raise ValueError("media continuation changed immutable visual units, order or source identity")
            return PreparedReplayMedia(plan, prepared.probes, prepared.replay, prepared.executor)

    def warm(self):
        """Warm actual placed P2P/broadcast buffers after largest-unit execution."""
        self.check_media_health()
        if self.warm_receipt["transport"] != "pending":
            return
        self.engine._media_numeric_scope.verify()

        def prepare():
            if self.producer.pixels is not None:
                self.producer.pixels.fill_(1 if self.engine.rank == 0 else 0)
            self.producer.values.fill_(self.engine.rank + 1)
            return {"rank": self.engine.rank}
        media_phase(self.engine, "transport-inputs", prepare)
        encoder_rank = self.profile.execution.encoder_rank
        if encoder_rank != 0:
            if self.engine.rank == 0:
                self.engine.comm.send(self.producer.pixels, encoder_rank)
            elif self.engine.rank == encoder_rank:
                self.engine.comm.recv(self.producer.pixels, 0)
        self.engine.comm.broadcast(self.producer.values, root=encoder_rank)

        def retire():
            torch.cuda.synchronize(self.engine.w.device)
            values = self.producer.values.view(-1)
            if values[0].item() != encoder_rank + 1 or values[-1].item() != encoder_rank + 1:
                raise ValueError("media startup broadcast endpoints differ from their placed source")
            if self.engine.rank == encoder_rank:
                pixels = self.producer.pixels.view(-1)
                if pixels[0].item() != 1 or pixels[-1].item() != 1:
                    raise ValueError("media startup patch-transfer endpoints differ from their leader source")
            return {"rank": self.engine.rank, "scope": "actual fixed producer P2P and broadcast startup execution"}
        self.warm_receipt["transport"] = media_phase(self.engine, "transport-retire", retire)
        self.engine.capacity_plan["media_profile"]["startup"] = self.warm_receipt

    def close(self):
        if self.closed:
            return
        if os.getpid() != self.pid:
            raise RuntimeError("media runtime belongs to another process")
        if self.work is not None:
            self.work.close()
            self.work = None
        if self.requests is not None:
            self.requests.check_drained()
        if self.producer is not None:
            self.producer.close()
            self.producer = None
        if self.encoder is not None:
            self.encoder.close()
            self.encoder = None
        self._warm_output = self._warm_pixels = None
        if self.reader is not None:
            self.reader.close()
            self.reader = None
        if self.executor is not None:
            self.executor.check_health()
            # Request/work owners must retire all leases and child workers
            # before model teardown can remove their shared backing directory.
            with self.store._lock:
                if (self.store._leases or self.store._staging or any(self.store._active_workers)
                        or any(self.store._retained_workers)):
                    raise RuntimeError("media runtime retains live replay/staging/worker owners")
        if self.directory is not None:
            directory = self.directory
            # Spend path authority before entering removal. An ambiguous error
            # is retained and surfaced; no destructor/retry can hit a reused path.
            self.directory = None
            try:
                os.rmdir(directory)
            except BaseException as error:
                self.directory_failure = error
                raise
        if self.directory_failure is not None:
            raise self.directory_failure
        self.frontend = self.executor = self.store = self.planner = self.contract = self.tokenizer = self.requests = None
        self.closed = True
