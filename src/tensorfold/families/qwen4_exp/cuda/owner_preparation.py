"""Opt-in, pre-admission ownership of native attention workspaces and graphs.

Preparation is capability work, not release qualification. Tensor storage is
already included in the subsequent physical-free baseline and is adopted by
the real serving buffers, never allocated a second time by their geometry.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
import re
from tensorfold.cleanup import finish, raise_failures
from .owner_execution import (
    FULL_COLLECTIVE, HEAD_COLLECTIVE, KERNEL_FAMILY, RN_KERNEL_FAMILY, ATEN_KERNEL_FAMILY, FLASH_KERNEL_FAMILY,
    NATIVE_FAMILIES, UNCAPTURED, graph_family_for, row_buckets,
)

_ROLES = frozenset({"decode", "mtp", "prefill"})
_CHANNEL = "tensorfold/owner-preparation/v1"


ARITHMETIC_POLICY = 'reference-aten128-three-islands-v1'


@dataclass(frozen=True)
class ArithmeticRequest:
    graph_budget_bytes: int
    policy: str = ARITHMETIC_POLICY

    def __post_init__(self):
        if (self.policy != ARITHMETIC_POLICY or type(self.graph_budget_bytes) is not int
                or not 0 < self.graph_budget_bytes < 2**63):
            raise ValueError('reference arithmetic requires its exact policy and explicit positive graph budget')

    @classmethod
    def parse(cls, value):
        if value is None:
            return None
        if type(value) is not dict or set(value) != {'policy', 'graph_budget_bytes'}:
            raise ValueError('reference arithmetic preparation requires policy and graph_budget_bytes')
        return cls(value['graph_budget_bytes'], value['policy'])


@dataclass(frozen=True)
class FlashRequest:
    panel_rows: int
    tensor_budget_bytes: int
    native_manifest: str
    native_manifest_sha256: str

    def __post_init__(self):
        if (type(self.panel_rows) is not int or not 1 <= self.panel_rows <= 256
                or type(self.tensor_budget_bytes) is not int or not 0 < self.tensor_budget_bytes < 2**63
                or type(self.native_manifest) is not str or not Path(self.native_manifest).is_absolute()
                or type(self.native_manifest_sha256) is not str
                or re.fullmatch('[0-9a-f]{64}', self.native_manifest_sha256) is None):
            raise ValueError('Flash owner needs bounded panels/credit and an absolute sealed manifest/SHA')

    @classmethod
    def parse(cls, value):
        fields = {'panel_rows', 'tensor_budget_bytes', 'native_manifest', 'native_manifest_sha256'}
        if type(value) is not dict or set(value) != fields:
            raise ValueError('Flash owner needs its exact flash_tiles contract')
        return cls(**value)

    def as_dict(self):
        return {'panel_rows': self.panel_rows, 'tensor_budget_bytes': self.tensor_budget_bytes,
                'native_manifest': self.native_manifest, 'native_manifest_sha256': self.native_manifest_sha256}


def _dense_backend(value):
    """Cold explicit dispatch; HC has its own finite manifest/role contract."""
    if type(value) is dict and value.get('backend') == 'bitblas-hc':
        from . import bitblas_hc as backend
    else:
        from . import dense_projections as backend
    return backend, backend.Selection.parse(value)


def parse_preparation(value):
    if value is None:
        return None
    required = {"kernel_family", "graph_family", "graph_budget_bytes", "trial"}
    if (not isinstance(value, dict) or not required <= value.keys()
            or set(value).difference(required | {"row_buckets", "collective_family", "reference_arithmetic", "rn_heads_per_block",
                                               "aten_tiles", "flash_tiles", "dense_projections", "native_gdn_capture", "exact_graph_policy", "native_hc_writeback_capture", "native_hc_writeback_target_eager", "gdn_value8_capture", "native_family", "owned_q8_window_prepare", "owned_q8_window_dispatch", "prepared_linears", "descriptor_upload"})
            or value.get("collective_family", FULL_COLLECTIVE) not in (FULL_COLLECTIVE, HEAD_COLLECTIVE)
            or type(value["kernel_family"]) is not str or value["kernel_family"] not in NATIVE_FAMILIES
            or value["graph_family"] != UNCAPTURED and value["graph_family"] != graph_family_for(value["kernel_family"])
            or type(value["trial"]) is not bool or type(value["graph_budget_bytes"]) is not int
            or value["graph_budget_bytes"] < 0
            or (value["graph_family"] != UNCAPTURED) != (value["graph_budget_bytes"] > 0)):
        raise ValueError("native preparation needs its exact family, explicit trial flag and bounded graph allowance")
    rn = value["kernel_family"] == RN_KERNEL_FAMILY
    if (rn and (type(value.get("rn_heads_per_block")) is not int or value["rn_heads_per_block"] not in (1, 4, 12))
            or not rn and "rn_heads_per_block" in value):
        raise ValueError("RN numerical preparation requires an explicit1/4/12-head block choice only for that family")
    buckets = value.get("row_buckets", {})
    if not isinstance(buckets, dict) or set(buckets).difference(_ROLES):
        raise ValueError("native preparation row buckets must name supported execution roles")
    for rows in buckets.values():
        if not isinstance(rows, list) or not rows:
            raise ValueError("a native execution role requires a nonempty bounded bucket list")
        row_buckets(rows[-1], rows)
    request = ArithmeticRequest.parse(value.get("reference_arithmetic"))
    aten = value['kernel_family'] == ATEN_KERNEL_FAMILY
    tiles = value.get('aten_tiles')
    if aten:
        if (type(tiles) is not dict or set(tiles) != {'panel_rows', 'tensor_budget_bytes'}
                or type(tiles['panel_rows']) is not int or not 1 <= tiles['panel_rows'] <= 256
                or type(tiles['tensor_budget_bytes']) is not int or not 0 < tiles['tensor_budget_bytes'] < 2**63
                or not value['trial'] or value['graph_family'] != UNCAPTURED or request is not None):
            raise ValueError('ATen owner trial needs explicit panels/tensor credit and uncaptured owner execution')
    elif tiles is not None:
        raise ValueError('ATen tile credits require their explicit attention family')
    flash = value['kernel_family'] == FLASH_KERNEL_FAMILY
    flash_tiles = None
    if flash:
        flash_tiles = FlashRequest.parse(value.get('flash_tiles'))
        if not value['trial'] or value['graph_family'] != UNCAPTURED or request is not None:
            raise ValueError('Flash owner requires an explicit uncaptured unqualified trial')
    elif value.get('flash_tiles') is not None:
        raise ValueError('Flash tile credits require their explicit attention family')
    dense = None
    if value.get('dense_projections') is not None:
        _, dense = _dense_backend(value['dense_projections'])
    if dense is not None and not value['trial']:
        raise ValueError('dense projection selection is an explicit unqualified trial')
    from .exact_graph_policy import parse_policy
    exact_graph_policy = parse_policy(value.get('exact_graph_policy'))
    if exact_graph_policy is not None and not value['trial']:
        raise ValueError('exact graph subset selection is an explicit unqualified trial')
    hc_capture = value.get('native_hc_writeback_capture', False)
    if type(hc_capture) is not bool or hc_capture and not value['trial']:
        raise ValueError('native HC writeback capture requires an explicit boolean unqualified trial')
    hc_eager = value.get('native_hc_writeback_target_eager', False)
    if type(hc_eager) is not bool or hc_eager and not value['trial']:
        raise ValueError('native target-eager HC writeback requires an explicit boolean unqualified trial')
    prepared_linears = value.get('prepared_linears')
    if prepared_linears is not None:
        from .padded_linears import OWNED_POLICY
        if (prepared_linears != OWNED_POLICY or not value['trial'] or not rn
                or value['graph_family'] != UNCAPTURED or request is not None or dense is not None):
            raise ValueError('prepared M32 requires its explicit RN trial without competing dense/reference arithmetic')
    native_family = value.get('native_family')
    if native_family is not None:
        from tensorfold.cuda.sm70.owned_family import POLICY
        if (native_family != POLICY or not value['trial'] or not rn
                or value['graph_family'] != UNCAPTURED or request is not None):
            raise ValueError('native family requires its explicit RN owner trial without reference arithmetic')
    q8_window = value.get('owned_q8_window_prepare', False)
    if (type(q8_window) is not bool or q8_window and (not value['trial']
            or value['kernel_family'] not in (RN_KERNEL_FAMILY, ATEN_KERNEL_FAMILY, FLASH_KERNEL_FAMILY)
            or value['graph_family'] != UNCAPTURED)):
        raise ValueError('owned Q8 window preparation requires an explicit uncaptured numerical trial')
    q8_dispatch = value.get('owned_q8_window_dispatch', 'leaf-loop-v1')
    if ('owned_q8_window_dispatch' in value and not q8_window
            or q8_dispatch not in ('leaf-loop-v1', 'grid-v1')):
        raise ValueError('owned Q8 window dispatch requires its selected window trial and versioned policy')
    descriptor_upload = value.get('descriptor_upload')
    if descriptor_upload is not None:
        from .descriptor_upload import POLICY
        if descriptor_upload != POLICY or not value['trial']:
            raise ValueError('descriptor upload ring requires its explicit versioned private trial')
    capture = value.get('native_gdn_capture', False)
    if type(capture) is not bool or capture and not value['trial']:
        raise ValueError('native GDN capture requires an explicit boolean unqualified trial')
    value8 = value.get('gdn_value8_capture', False)
    if type(value8) is not bool or value8 and (not value['trial'] or not capture):
        raise ValueError('Value8 capture requires an explicit trial with native GDN capture')
    return {"kernel_family": value["kernel_family"], "graph_family": value["graph_family"],
            "graph_budget_bytes": value["graph_budget_bytes"], "trial": value["trial"],
            "collective_family": value.get("collective_family", FULL_COLLECTIVE),
            "row_buckets": deepcopy(buckets),
            **({"rn_heads_per_block": value["rn_heads_per_block"]} if rn else {}),
            **({'aten_tiles': dict(tiles)} if aten else {}),
            **({'flash_tiles': flash_tiles.as_dict()} if flash else {}),
            **({'dense_projections': dense.as_dict()} if dense is not None else {}),
            **({'native_gdn_capture': True} if capture else {}),
            **({'native_family': native_family} if native_family is not None else {}),
            **({'owned_q8_window_prepare': True} if q8_window else {}),
            **({'owned_q8_window_dispatch': 'grid-v1'} if q8_dispatch == 'grid-v1' else {}),
            **({'prepared_linears': prepared_linears} if prepared_linears is not None else {}),
            **({'descriptor_upload': descriptor_upload} if descriptor_upload is not None else {}),
            **({'gdn_value8_capture': True} if value8 else {}),
            **({'native_hc_writeback_capture': True} if hc_capture else {}),
            **({'native_hc_writeback_target_eager': True} if hc_eager else {}),
            **({'exact_graph_policy': exact_graph_policy} if exact_graph_policy is not None else {}),
            **({"reference_arithmetic": {"policy": request.policy, "graph_budget_bytes": request.graph_budget_bytes}}
               if request is not None else {})}


class OwnerPreparation:
    def __init__(self, specification, *, role_rows, manifest_sha256, reference_tiles=False):
        self.specification = parse_preparation(specification)
        if self.specification is None or not isinstance(role_rows, dict) or not {"decode", "prefill"} <= role_rows.keys():
            raise ValueError("native preparation requires the actual decode and prefill envelopes")
        if set(role_rows).difference(_ROLES) or set(self.specification["row_buckets"]).difference(role_rows):
            raise ValueError("native preparation names a role absent from the selected method")
        if type(reference_tiles) is not bool or reference_tiles and (
                self.specification["trial"] or self.specification["kernel_family"] != KERNEL_FAMILY
                or self.specification["graph_family"] != UNCAPTURED
                or self.specification["collective_family"] != FULL_COLLECTIVE):
            raise ValueError("reference tile preparation requires the uncaptured full-TP8 reference policy")
        self.reference_arithmetic_request = ArithmeticRequest.parse(self.specification.get("reference_arithmetic"))
        if self.reference_arithmetic_request is not None and not reference_tiles:
            raise ValueError("reference arithmetic requires the existing trusted private reference tile capability")
        self.reference_arithmetic, self.reference_arithmetic_bytes = {}, 0
        self.reference_tiles_requested = reference_tiles
        self.reference_preparation = None
        self.reference_arithmetic_source = None
        self.rn_numerical_identity = None
        self.role_rows = {role: role_rows[role] for role in ("decode", "mtp", "prefill") if role in role_rows}
        self.buckets = {role: row_buckets(rows, self.specification["row_buckets"].get(role))
                        for role, rows in self.role_rows.items()}
        self.manifest_sha256 = manifest_sha256
        self.workspaces, self.graphs = {}, {}
        self.head_comm = None
        self.comm = self._open_head = None
        self.source_foundation = None
        self.native_family = None
        self.prepared_m32 = None
        self.phases = []
        self.before = self.after = None
        self.graph_physical_bytes = 0
        self.prepared = self.closed = False
        self.allocations = {}
        self.private_spare_bytes = None
        self.outside_active_bytes = None
        self.aten_requirements, self.aten_operator_identity = {}, None
        self.aten_temporary_extra_bytes = 0
        self.flash_request = self.flash_validated = self.flash_artifact = None
        self.flash_requirements = {}
        if self.specification['kernel_family'] == ATEN_KERNEL_FAMILY:
            from tensorfold.cuda.sm70.attention_backends import owner_requirements
            self.aten_requirements = {role: owner_requirements(rows, self.specification['aten_tiles']['panel_rows'])
                                      for role, rows in self.role_rows.items()}
            needed = (sum(row['persistent_bytes'] for row in self.aten_requirements.values())
                      + max(row['operator_temporary_bytes'] for row in self.aten_requirements.values()))
            if needed > self.specification['aten_tiles']['tensor_budget_bytes']:
                raise MemoryError('ATen owner roles exceed their aggregate explicit tensor credit')
        if self.specification['kernel_family'] == FLASH_KERNEL_FAMILY:
            from tensorfold.cuda.sm70.attention_backends import flash_owner_requirements
            from tensorfold.cuda.sm70.flash_v100_artifact import validate_manifest
            self.flash_request = FlashRequest.parse(self.specification['flash_tiles'])
            self.flash_requirements = {role: flash_owner_requirements(rows, self.flash_request.panel_rows)
                                       for role, rows in self.role_rows.items()}
            if sum(row['persistent_bytes'] for row in self.flash_requirements.values()) > self.flash_request.tensor_budget_bytes:
                raise MemoryError('Flash owner roles exceed their aggregate complete tensor credit')
            # Metadata/source/ELF validation is Torch-absent and precedes CUDA
            # allocation; no later pointer or plan lookup can grow this owner.
            self.flash_validated = validate_manifest(self.flash_request.native_manifest,
                                                      self.flash_request.native_manifest_sha256)
        self.dense_owner = None
        self.dense_forecast = None
        self.dense_binding_started = False
        self.dense_binding_memory = None

    @staticmethod
    def _memory(torch, device):
        return {"physical_free_bytes": int(torch.cuda.mem_get_info(device)[0]),
                "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "torch_reserved_bytes": int(torch.cuda.memory_reserved(device))}

    def _phase(self, engine, name, operation, *, may_refuse=True):
        from .execution_identity import _phase

        self.phases.append(name)
        return _phase(engine, _CHANNEL, name, operation, may_refuse=may_refuse)

    def prepare(self, engine, text, pair, *, page_tokens, device):
        import torch
        from tensorfold.engine.model_identity import canonical_runtime_json
        from .execution_identity import _fatal, _retire_store
        from .history_owner import OwnerBuffers
        from .owner_graphs import OwnerAttentionGraphs

        if self.prepared or self.closed:
            raise RuntimeError("native owner preparation is a one-time startup operation")
        self.comm = engine.comm
        construction = getattr(engine.comm, "_profile_construction", None)
        contract = {"specification": self.specification, "manifest_sha256": self.manifest_sha256,
                    "roles": self.role_rows, "page_tokens": page_tokens, "pair": pair.identity,
                    "heads": text["num_attention_heads"], "kv_heads": text["num_key_value_heads"],
                    "head_dim": text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"],
                    "index_budget": text.get("indexer_budget", 2048),
                    "index_ratio": text.get("indexer_compress_ratio", 4)}
        import json

        contract = json.loads(canonical_runtime_json(contract, max_bytes=4 << 20))
        if self.specification['kernel_family'] in (ATEN_KERNEL_FAMILY, FLASH_KERNEL_FAMILY):
            from tensorfold.cuda.sm70.attention_backends import OWNER_MAX_KEYS
            from .history_placement import attention_temporary_bytes
            if ((pair.key.bits, pair.key.codec, pair.value.bits, pair.value.codec) != (8, 0, 8, 0)
                    or contract['index_budget'] + contract['index_ratio'] - 1 != OWNER_MAX_KEYS):
                raise ValueError('ATen/Flash owner family requires its original Q8/2051-key selected profile')
            if self.specification['kernel_family'] == ATEN_KERNEL_FAMILY:
                baseline = attention_temporary_bytes(contract['heads'] // engine.tp, contract['head_dim'])
                self.aten_temporary_extra_bytes = max(0, max(
                    row['operator_temporary_bytes'] for row in self.aten_requirements.values()) - baseline)
        dense_selection = dense_backend = None
        if self.specification.get('dense_projections') is not None:
            dense_backend, dense_selection = _dense_backend(self.specification['dense_projections'])
            self.dense_forecast = dense_backend.forecast(dense_selection, text, self.role_rows, world=engine.tp)
        if self.specification.get('prepared_linears') is not None:
            from .padded_linears import PreparedM32
            self.prepared_m32 = PreparedM32(text, self.role_rows, device)
        primary = None
        try:
            def prepared_contract():
                from tensorfold.cuda.sm70.foundation_headers import selected_identity

                if self.numerical_selection is not None and (
                        getattr(engine, 'position_profile', None) is None
                        or getattr(getattr(engine, 'method_geometry', None), 'method', None) not in ('none', 'mtp')):
                    raise ValueError('new owner numerics require the bound NONE/MTP coordinate region')
                if self.specification.get('native_gdn_capture', False) and not getattr(
                        engine, '_native_gdn_capture_enabled', False):
                    raise ValueError('native GDN capture requires the fixed private exact graph region')
                if self.specification.get('exact_graph_policy') is not None and not getattr(
                        engine, '_exact_graph_subset_enabled', False):
                    raise ValueError('exact graph subset requires the fixed private exact graph region')
                if self.specification.get('native_hc_writeback_capture', False) and not getattr(
                        engine, '_native_hc_writeback_capture_enabled', False):
                    raise ValueError('native HC writeback capture requires the fixed private exact graph region')
                if self.specification.get('native_hc_writeback_target_eager', False) and not getattr(
                        engine, '_native_hc_writeback_target_eager_enabled', False):
                    raise ValueError('native target-eager HC writeback requires the fixed private graph-enabled region')
                if dense_selection is not None and getattr(engine, '_local_hc_readout_requested', False):
                    raise ValueError('dense projection trials do not cover local HCReadoutGraphs')
                if self.specification.get('gdn_value8_capture', False) and not getattr(
                        engine, '_gdn_value8_capture_enabled', False):
                    raise ValueError('Value8 capture requires the fixed private native GDN graph region')
                if (self.specification.get('native_family') is not None
                        or self.specification.get('prepared_linears') is not None):
                    if (not getattr(engine, '_owned_expert_requested', False)
                            or getattr(engine, '_owned_expert_admin_requested', True)
                            or getattr(engine, 'sm70_profile', None) != 'reference'
                            or getattr(engine, 'expert_partition', None) is None
                            or engine.expert_partition.etp != 4 or engine.tp != 8
                            or int(text['hidden_size']) != 2560 or int(text.get('hc_count', 4)) != 4
                            or self.reference_tiles_requested
                            or getattr(engine, '_local_hc_readout_requested', False)):
                        raise ValueError('native family/prepared M32 requires the private TP8/ETP4 HC4 reference-owner region')
                self.source_foundation = selected_identity()
                if self.reference_tiles_requested:
                    from tensorfold.cuda.sm70.reference_owner_tiles import prepare
                    self.reference_preparation = prepare()
                arithmetic_source = {}
                if self.specification["kernel_family"] == RN_KERNEL_FAMILY:
                    from tensorfold.cuda.sm70.paged_attention import rn_preparation_identity
                    self.rn_numerical_identity = deepcopy(rn_preparation_identity())
                    arithmetic_source["rn_numerical_identity"] = self.rn_numerical_identity
                if self.specification['kernel_family'] == ATEN_KERNEL_FAMILY:
                    from tensorfold.cuda.sm70.attention_backends import operator_identity
                    self.aten_operator_identity = operator_identity()
                    arithmetic_source['aten_operator_identity'] = self.aten_operator_identity
                if self.flash_validated is not None:
                    arithmetic_source['flash_native_identity'] = self.flash_validated.record()
                if self.reference_arithmetic_request is not None:
                    from . import reference_arithmetic
                    import hashlib
                    from pathlib import Path
                    self.reference_arithmetic_source = hashlib.sha256(Path(reference_arithmetic.__file__).read_bytes()).hexdigest()
                    arithmetic_source["reference_arithmetic_source_sha256"] = self.reference_arithmetic_source
                return {**contract, **arithmetic_source, "source_foundation": self.source_foundation,
                        **({"reference_tile_preparation": self.reference_preparation}
                           if self.reference_tiles_requested else {})}

            peers = self._phase(engine, "contract", prepared_contract)
            if any(peer != peers[0] for peer in peers):
                raise ValueError("native owner preparation differs across TP8 ranks")
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            self.before = self._memory(torch, device)
            if self.specification.get('native_family') is not None:
                def family_prepare():
                    from tensorfold.cuda.sm70.owned_family import prepare
                    self.native_family = prepare(device)
                    return self.native_family.record()
                family_records = self._phase(engine, 'native-family-prepare', family_prepare, may_refuse=False)
                if any(record != family_records[0] for record in family_records):
                    raise ValueError('native family preparation differs across TP8 ranks')
            if self.flash_validated is not None:
                def flash_load():
                    from tensorfold.cuda.sm70.flash_v100_artifact import load_native
                    self.flash_artifact = load_native(self.flash_validated)
                    return self.flash_artifact.record()
                # Loading/role warmup is inside the measured commitment window,
                # before any State is allocated. Code maps may remain resident.
                self._phase(engine, 'flash-native-load', flash_load, may_refuse=False)
            if dense_selection is not None:
                def dense_prepare():
                    owner = dense_backend.Owner.prepare(dense_selection, text, self.role_rows, device, world=engine.tp,
                        register=lambda owner: setattr(self, 'dense_owner', owner))
                    return owner.record()
                self._phase(engine, 'dense-projections-prepare', dense_prepare, may_refuse=False)
            if self.specification["collective_family"] == HEAD_COLLECTIVE:
                def head_owner():
                    from tensorfold.cuda.comm import NCCL

                    self.head_comm = NCCL()  # Journal before fallible native initialization.
                    self._open_head = partial(self.head_comm.open, engine.rank % 4, 4, "", 0, store=engine.comm.store,
                                              key=_CHANNEL + "/head/" + str(engine.rank // 4))
                    if construction is not None:
                        construction.bind_protocol(self, "_open_head", "owner/head-init", refusal=())
                    return {"members": list(range(engine.rank // 4 * 4, engine.rank // 4 * 4 + 4))}

                # CPU owner/arguments/hooks finish on all8 before any raw init
                # GO. A post-operation phase vote alone cannot establish this.
                self._phase(engine, "head-owner", head_owner, may_refuse=False)
                def head_group():
                    self._open_head()
                    if construction is not None:
                        construction.bind_submit(self.head_comm, "all_gather", "owner/head-gather")
                    return {"members": list(range(engine.rank // 4 * 4, engine.rank // 4 * 4 + 4))}

                self._phase(engine, "head-group", head_group, may_refuse=False)
                # Every subgroup member has consumed its UID before this phase
                # completes. Fixed two keys cannot grow with sessions/buckets.
                def retire_uids():
                    if engine.rank == 0:
                        for group in range(2):
                            engine.comm.store.delete_key(_CHANNEL + "/head/" + str(group))
                    return {}

                self._phase(engine, "head-uid-retirement", retire_uids, may_refuse=False)
            for role, rows in self.role_rows.items():
                def allocate(role=role, rows=rows):
                    owner = OwnerBuffers.from_text(text, rows=rows, prefill=role == "prefill",
                        idw=contract["index_budget"] + contract["index_ratio"], world=engine.tp,
                        rank=engine.rank, device=device, comm=engine.comm,
                        register=lambda owner: self.workspaces.__setitem__(role, owner),
                        **({"reference_preparation": self.reference_preparation} if self.reference_tiles_requested else {}))
                    owner.prepare_native(pair=pair, page_tokens=page_tokens, buckets=self.buckets[role],
                        **({'descriptor_upload_policy': self.specification['descriptor_upload']}
                           if self.specification.get('descriptor_upload') is not None else {}))
                    owner.configure_native_family(self.specification["kernel_family"], self.specification.get("rn_heads_per_block"),
                        **({'owned_q8_window_prepare': True, 'owned_q8_window_role': role}
                           if self.specification.get('owned_q8_window_prepare', False) and role != 'prefill' else {}),
                        **({'owned_q8_window_dispatch': 'grid-v1'}
                           if self.specification.get('owned_q8_window_dispatch') == 'grid-v1' and role != 'prefill' else {}),
                        **({'aten_panel_rows': self.specification['aten_tiles']['panel_rows']}
                           if self.specification['kernel_family'] == ATEN_KERNEL_FAMILY else {}),
                        **({'flash_artifact': self.flash_artifact,
                            'flash_panel_rows': self.flash_request.panel_rows,
                            'flash_tensor_budget_bytes': self.flash_requirements[role]['persistent_bytes']}
                           if self.flash_artifact is not None else {}))
                    owner.configure_native_transport(self.specification["collective_family"],
                                                     self.head_comm or engine.comm)
                    return owner.allocation_record()

                records = self._phase(engine, "allocate-" + role, allocate, may_refuse=False)
                self.allocations[role] = records
                owner = self.workspaces[role]
                def reference_wire(owner=owner):
                    owner.warm_reference_transport()
                    return {"full8_reference_rows": owner.rows}

                self._phase(engine, "reference-wire-" + role, reference_wire, may_refuse=False)
                if self.reference_arithmetic_request is not None:
                    def arithmetic_owner(role=role, owner=owner):
                        from .reference_arithmetic import ReferenceArithmetic
                        remaining = self.reference_arithmetic_request.graph_budget_bytes - self.reference_arithmetic_bytes
                        if remaining <= 0:
                            raise ValueError("reference arithmetic roles exhausted their explicit aggregate budget")
                        graph = object.__new__(ReferenceArithmetic)
                        self.reference_arithmetic[role] = graph  # Journal before any capture/allocation.
                        graph.__init__(owner.reference_tiles, budget_bytes=remaining)
                        owner.reference_tiles.arithmetic = graph
                        graph.prepare(quiescent=lambda: engine.multi is None and not engine._calls)
                        self.reference_arithmetic_bytes += graph.charged_bytes
                        if self.reference_arithmetic_bytes > self.reference_arithmetic_request.graph_budget_bytes:
                            raise ValueError("reference arithmetic aggregate preparation exceeds its explicit budget")
                        return graph.record()
                    self._phase(engine, "reference-arithmetic-" + role, arithmetic_owner, may_refuse=False)
                if self.specification["graph_family"] != UNCAPTURED:
                    remaining = self.specification["graph_budget_bytes"] - self.graph_physical_bytes
                    before_graph = self._memory(torch, device)

                    def graph_owner(role=role, owner=owner, remaining=remaining):
                        if remaining <= 0:
                            raise ValueError("native owner graph preparation exhausted its aggregate allowance")
                        graphs = object.__new__(OwnerAttentionGraphs)
                        self.graphs[role] = graphs
                        graphs.__init__(owner, rows=owner.rows, budget_bytes=remaining, comm=engine.comm,
                                        buckets=self.buckets[role])
                        return {"rows": list(graphs.buckets)}

                    self._phase(engine, "graph-owner-" + role, graph_owner, may_refuse=False)
                    graphs = self.graphs[role]
                    graphs.prepare(quiescent=lambda: engine.multi is None and not engine._calls)
                    torch.cuda.synchronize(device)
                    charged = max(0, before_graph["physical_free_bytes"] - self._memory(torch, device)["physical_free_bytes"])
                    self.graph_physical_bytes += charged

                    def check_budget():
                        if self.graph_physical_bytes > self.specification["graph_budget_bytes"]:
                            raise ValueError("native graph roles exceed their aggregate physical preparation allowance")
                        return {"physical_preparation_bytes": charged, **graphs.record()}

                    self._phase(engine, "graph-budget-" + role, check_budget)
                else:
                    # Eager execution uses this same finite padded catalogue;
                    # eligible library/kernel shape residency precedes admission.
                    def warm(owner=owner, role=role):
                        for bucket in self.buckets[role]:
                            owner.clear_native_inputs(bucket)
                            owner.warm_native_transport(bucket)
                            owner.wait_native_inputs()
                            owner.native_run(bucket)
                            torch.cuda.synchronize(device)
                        return {"rows": list(self.buckets[role])}

                    self._phase(engine, "warm-" + role, warm, may_refuse=False)
            torch.cuda.empty_cache()
            self.after = self._memory(torch, device)
            def commitment():
                from tensorfold.cuda.capacity import cuda_limit_bytes

                limit, committed = cuda_limit_bytes(), self.committed_bytes()
                if limit is not None and committed is None:
                    raise ValueError("absolute CUDA cap requires settled private-pool and preparation residency accounting")
                if limit is not None and committed > limit:
                    raise ValueError("native owner preallocation exceeds the absolute CUDA cap")
                return {"proved_nonreusable_bytes": committed, "absolute_cap_bytes": limit}

            self._phase(engine, "commitment", commitment)
            private = [graph.private_spare_bytes for graph in
                       (*self.graphs.values(), *self.reference_arithmetic.values())]
            self.private_spare_bytes = None if any(value is None for value in private) else sum(private)
            committed = self.committed_bytes()
            if committed is not None:
                self.outside_active_bytes = committed - (
                    self.after["torch_allocated_bytes"] - self.before["torch_allocated_bytes"])
            self.prepared = True
        except BaseException as error:
            primary = error
            if not isinstance(error, ValueError) or getattr(engine, "_execution_manifest_uncertain", False):
                _fatal(engine, error)
        errors = []
        if not getattr(engine, "_execution_manifest_uncertain", False):
            try:
                _retire_store(engine, _CHANNEL, stages=tuple(self.phases))
            except BaseException as error:
                _fatal(engine, error)
                errors.append(error)
        raise_failures(primary, errors)

    def adopt(self, weights, decoder):
        if not self.prepared or self.closed:
            raise RuntimeError("native resources were not prepared before model admission")
        if weights.meta.get("owner_workspaces") is not self.workspaces:
            raise RuntimeError("model weights did not publish the exact prepared owner workspace catalogue")
        from tensorfold.cuda.sm70.foundation_headers import selected_identity

        if selected_identity() != self.source_foundation:
            raise ValueError("selected source foundation differs from native preparation")
        for graphs in (*self.graphs.values(), *self.reference_arithmetic.values()):
            graphs.validate_adoption()
        for owner in self.workspaces.values():
            if (owner.native_kernel_family != self.specification["kernel_family"]
                    or owner.native_heads_per_block != self.specification.get("rn_heads_per_block")):
                raise RuntimeError("serving owner numerical policy differs from its prepared source identity")
            owner.bind_native_states(decoder.states)
        for role, buffers in (("decode", decoder.buf), ("mtp", decoder.mbuf), ("prefill", decoder.pbuf)):
            if buffers is None:
                continue
            if buffers.history_owner is not self.workspaces[role]:
                raise RuntimeError("serving buffers did not adopt their preallocated owner workspace")
            if getattr(buffers, 'native_family', None) is not self.native_family:
                raise RuntimeError('serving buffers did not adopt their prepared native family capability')
            if self.prepared_m32 is not None and role in self.prepared_m32.role_rows:
                if buffers.padded_linears is not self.prepared_m32.buffers.get(role):
                    raise RuntimeError('serving role did not adopt its paid prepared M32 buffers')
            buffers.owner_graphs = self.graphs.get(role)
        # Every owner frame is bounded by its adopted workspace, even when a
        # larger model batch is split into several such attention operations.
        graph_rows = decoder.buf.rows if self.graphs else 0
        decoder.controller.configure_execution_families(native=True, graph_rows=graph_rows,
            native_family=self.specification["kernel_family"],
            trial={"kernel_family": self.specification["kernel_family"], "graph_family": self.specification["graph_family"]}
                  if self.specification["trial"] else None)
        decoder.apply_execution_family()
        if self.dense_owner is not None:
            if not self.dense_owner.bound_weights:
                raise RuntimeError('dense projection shapes must bind before State construction')
            self.dense_owner.adopt(weights, decoder)

    @property
    def numerical_selection(self):
        selected = {}
        if self.specification.get('prepared_linears') is not None:
            from .padded_linears import owned_numerical_identity
            selected['prepared_linears'] = owned_numerical_identity()
        if self.specification.get('native_family') is not None:
            from tensorfold.cuda.sm70.owned_family import numerical_identity
            selected['native_family'] = numerical_identity()
        if self.specification.get('owned_q8_window_prepare', False):
            selected['owned_q8_window_preparation'] = {
                'policy': ('owned-q8-window-grid-preparation-v1'
                           if self.specification.get('owned_q8_window_dispatch') == 'grid-v1'
                           else 'owned-q8-window-preparation-v1'),
                'roles': [role for role in ('decode', 'mtp') if role in self.role_rows],
                'codec': 'original-floor-q8',
                'native_rows_per_dispatch': ('window' if self.specification.get('owned_q8_window_dispatch') == 'grid-v1' else 1),
                'native_dispatches_per_window': (1 if self.specification.get('owned_q8_window_dispatch') == 'grid-v1' else 'rows'),
                'maximum_window_rows': 256,
                'compact_indexer': 'reference-window-transaction', 'qualified': False}
        if self.specification['kernel_family'] == ATEN_KERNEL_FAMILY:
            from tensorfold.cuda.sm70.attention_backends import OWNER_MAX_KEYS, POLICY
            selected['attention'] = {'family': ATEN_KERNEL_FAMILY, 'operator': POLICY,
                'max_keys': OWNER_MAX_KEYS, 'panel_rows': self.specification['aten_tiles']['panel_rows'],
                'key_capacity_policy': 'fixed2051-mask-v1', 'local_output': 'normalized-half',
                'global_merge': 'ordered-float-lse-weights-v1'}
        if self.flash_validated is not None:
            selected['attention'] = {'family': FLASH_KERNEL_FAMILY,
                'native_manifest_sha256': self.flash_validated.manifest_sha256,
                'native_profile_sha256': self.flash_validated.profile_sha256,
                'panel_rows': self.flash_request.panel_rows, 'max_keys': 2051,
                'key_capacity_policy': 'stable-compact2051-fixed-panel-v1',
                'query_layout': 'fixed12-head-as-row', 'local_output': 'normalized-half',
                'global_merge': 'ordered-float-lse-weights-v1'}
        if self.specification.get('dense_projections') is not None:
            _, selected_dense = _dense_backend(self.specification['dense_projections'])
            selected['dense_projections'] = selected_dense.numerical_identity
        return selected or None

    @property
    def additional_temporary_bytes(self):
        dense = self.dense_forecast
        if dense is None:
            return self.aten_temporary_extra_bytes
        return (self.aten_temporary_extra_bytes + dense['future_temporary_bytes']
                + (dense['cold_temporary_bytes'] if self.dense_binding_memory is None else 0))

    def bind_dense_weights(self, engine, weights):
        """Isolate post-weight/pre-State code residency from the early baseline."""
        if self.dense_owner is None and self.prepared_m32 is None:
            return
        if not self.prepared or self.closed or self.dense_binding_started:
            raise RuntimeError('dense weights require one live prepared pre-State binding')
        self.dense_binding_started = True
        stage = 'dense-projections-weight-binding'
        def bind():
            owner = self.dense_owner if self.dense_owner is not None else self.prepared_m32
            record = owner.bind_weights(weights)
            memory = record['binding_memory']
            for name in ('committed_delta_bytes', 'outside_active_delta_bytes'):
                if type(memory.get(name)) is not int or memory[name] < 0:
                    raise ValueError('dense binding residency is unproven; State admission must stop')
            self.dense_binding_memory = deepcopy(memory)
            if self.outside_active_bytes is None:
                raise ValueError('dense binding requires the original absolute-cap preparation proof')
            self.outside_active_bytes += memory['outside_active_delta_bytes']
            return record
        from .execution_identity import _fatal, _retire_store
        self._phase(engine, stage, bind, may_refuse=False)
        try:
            _retire_store(engine, _CHANNEL, stages=(stage,))
        except BaseException as primary:
            try:
                _fatal(engine, primary)
            except BaseException as containment:
                raise_failures(primary, [containment])
            raise

    @property
    def requires_worker_handoff(self):
        return self.dense_owner is not None or self.flash_artifact is not None

    def handoff_to_current_thread(self):
        if self.dense_owner is not None:
            self.dense_owner.handoff_to_current_thread()
        if self.flash_artifact is not None:
            for owner in self.workspaces.values():
                owner.handoff_native_thread()

    def committed_bytes(self):
        """Settled nonreusable preallocation, excluding ordinary reusable cache."""
        if self.after is None:
            return None
        captures = (*self.graphs.values(), *self.reference_arithmetic.values())
        if any(graph.private_spare_bytes is None for graph in captures):
            return None
        active = self.after["torch_allocated_bytes"] - self.before["torch_allocated_bytes"]
        reserved = self.after["torch_reserved_bytes"] - self.before["torch_reserved_bytes"]
        physical = self.before["physical_free_bytes"] - self.after["physical_free_bytes"]
        outside = physical - reserved
        if active < 0 or outside < 0:
            return None  # Competing/freeing activity cannot establish a positive owner charge.
        if self.dense_binding_started and self.dense_binding_memory is None:
            return None
        extra = 0 if self.dense_binding_memory is None else self.dense_binding_memory['committed_delta_bytes']
        return active + outside + sum(graph.private_spare_bytes for graph in captures) + extra

    def live_bytes(self):
        """Credit ordinary spare only after excluding proved private-pool ownership."""
        import torch
        from tensorfold.cuda.capacity import available_bytes, cuda_limit_bytes
        from tensorfold.cuda.kv_arena import mapped_bytes

        active = int(torch.cuda.memory_allocated())
        spare = (max(0, int(torch.cuda.memory_reserved()) - active - self.private_spare_bytes)
                 if self.private_spare_bytes is not None else 0)
        free = int(available_bytes(torch)) + spare
        limit = cuda_limit_bytes()
        if limit is None:
            return free
        if self.outside_active_bytes is None:
            return 0
        return max(0, min(free, limit - active - mapped_bytes() - self.outside_active_bytes))

    def graph_live_bytes(self, *, additional_outside_active_bytes):
        """Physical-only HC caller view, retaining this owner's absolute-cap proof.

        The additional settled charge includes private inactive graph storage
        and driver residency outside Torch's active counter. Physical free
        already excludes both; subtract it only in the absolute-cap domain.
        """
        if type(additional_outside_active_bytes) is not int or additional_outside_active_bytes < 0:
            raise ValueError("additional graph residency must be nonnegative integer bytes")
        import torch
        from tensorfold.cuda.capacity import available_bytes, cuda_limit_bytes
        from tensorfold.cuda.kv_arena import mapped_bytes

        free = int(available_bytes(torch))  # No ordinary or private spare credit.
        limit = cuda_limit_bytes()
        if limit is None:
            return free
        if self.outside_active_bytes is None:
            return 0
        active = int(torch.cuda.memory_allocated())
        return max(0, min(free, limit - active - mapped_bytes() - self.outside_active_bytes
                          - additional_outside_active_bytes))

    def identity(self):
        # The explicit trial switch selects a candidate for measurement; it
        # changes neither representation nor resources. Excluding it permits
        # the resulting verified execution proof to be installed for release.
        return {"specification": {name: deepcopy(value) for name, value in self.specification.items() if name != "trial"},
                "source_foundation_headers": deepcopy(self.source_foundation),
                **({"rn_numerical_identity": deepcopy(self.rn_numerical_identity)}
                   if self.specification["kernel_family"] == RN_KERNEL_FAMILY else {}),
                "roles": dict(self.role_rows),
                **({'native_family': self.native_family.record()} if self.native_family is not None else {}),
                **({'prepared_linears': self.prepared_m32.record()} if self.prepared_m32 is not None else {}),
                **({"reference_arithmetic_source_sha256": self.reference_arithmetic_source}
                   if self.reference_arithmetic_request is not None else {}),
                "row_buckets": {role: list(rows) for role, rows in self.buckets.items()},
                "allocations": deepcopy(self.allocations),
                **({'aten_operator_identity': deepcopy(self.aten_operator_identity),
                    'aten_tensor_requirements': deepcopy(self.aten_requirements)}
                   if self.aten_requirements else {}),
                **({'flash_native_identity': self.flash_validated.record(),
                    'flash_tensor_requirements': deepcopy(self.flash_requirements),
                    'flash_native_owner': None if self.flash_artifact is None else self.flash_artifact.record()}
                   if self.flash_validated is not None else {}),
                **({'dense_projections': self.dense_owner.record()} if self.dense_owner is not None else {}),
                "allocator_accounting": "exclude-proved-private-spare; physical-only-when-unavailable",
                **({"reference_tile_preparation": dict(self.reference_preparation)}
                   if self.reference_preparation is not None else {})}

    def record(self):
        before, after = self.before, self.after
        return {**self.identity(), "prepared": self.prepared, "closed": self.closed,
                "before": None if before is None else dict(before), "after": None if after is None else dict(after),
                "observed_physical_bytes": None if after is None else before["physical_free_bytes"] - after["physical_free_bytes"],
                "observed_torch_allocated_delta": None if after is None else after["torch_allocated_bytes"] - before["torch_allocated_bytes"],
                "observed_torch_reserved_delta": None if after is None else after["torch_reserved_bytes"] - before["torch_reserved_bytes"],
                "graph_physical_preparation_bytes": self.graph_physical_bytes,
                "proved_nonreusable_commitment_bytes": self.committed_bytes(),
                "private_pool_spare_bytes": self.private_spare_bytes,
                "outside_active_torch_commitment_bytes": self.outside_active_bytes,
                'additional_operator_temporary_bytes': self.additional_temporary_bytes,
                **({'dense_binding_memory': deepcopy(self.dense_binding_memory)}
                   if self.dense_binding_started else {}),
                "graphs": {role: graph.record() for role, graph in self.graphs.items()},
                "reference_arithmetic": {role: graph.record() for role, graph in self.reference_arithmetic.items()},
                "qualification_trial": self.specification["trial"],
                "included_in_physical_free_baseline": True, "future_owner_persistent_bytes": 0,
                "qualification": "UNQUALIFIED: source-built foundation and real SM70 model/quality/performance gates required"}

    def close(self):
        if self.closed:
            return
        construction = None if self.comm is None else getattr(self.comm, "_profile_construction", None)
        if construction is not None:
            construction.retire_bindings(self)
        finish((graph.close for graph in (*self.graphs.values(), *self.reference_arithmetic.values())))
        finish((owner.close for owner in self.workspaces.values()))
        if self.dense_owner is not None:
            self.dense_owner.close()
        if self.prepared_m32 is not None:
            self.prepared_m32.close()
        if self.flash_artifact is not None:
            self.flash_artifact.close()
            self.flash_artifact = None
        if self.head_comm is not None:
            if construction is not None:
                construction.retire_bindings(self.head_comm)
            self.head_comm.close()
            self.head_comm = None
        self.comm = self._open_head = None
        self.graphs, self.workspaces, self.reference_arithmetic = {}, {}, {}
        self.closed = True
