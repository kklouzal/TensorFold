"""Typed trial protocol controls; fixtures are not native execution evidence."""
import copy
import itertools
from types import SimpleNamespace

import pytest

from tests.test_v100_prepared_teacher_contract import data
from tools import v100_numerical_teacher as trial
from tools import v100_owned_model_contract as model


def inputs(flags=(False, False, False)):
    operation, case, profile, plan, client = data()
    authority = {'schema': trial.numa.NUMA_SCHEMA, 'placement_source': {'bytes': 1, 'sha256': '1'*64}}
    for key, (name, _) in trial.FILES.items():
        authority[key] = {'path': '/operator/' + name, 'bytes': 1, 'sha256': '2'*64}
    selected = dict(zip(trial.AXES, flags))
    preparation = {**operation['prepared_execution'], **{k: v for k, v in trial.AXES.items() if selected[k]}}
    operation.update(numa=authority, numerical_trial=selected, prepared_execution=preparation)
    client.update(schema=trial.CLIENT_SCHEMA, numa=copy.deepcopy(authority), numerical_trial=copy.deepcopy(selected),
                  prepared_execution=copy.deepcopy(preparation))
    profile['preparation'] = copy.deepcopy(preparation)
    return operation, case, profile, plan, client


def test_eight_explicit_trials_preserve_strict_expert21_contract():
    for flags in itertools.product((False, True), repeat=3):
        operation, _, _, _, client = inputs(flags)
        trial.validate_operation(operation)
        trial.validate_teacher_client(client)
        for selector in trial.selectors():
            operation['selector'] = selector
            trial.validate_operation(operation)
        with pytest.raises(ValueError):
            trial.numa.validate_operation(operation)
        assert trial.preparation_projection(operation['prepared_execution'], operation['numerical_trial']) == data()[0]['prepared_execution']


@pytest.mark.parametrize('bad', [None, 0, 1, 'true', [], {}])
def test_axes_require_real_booleans(bad):
    operation, *_ = inputs()
    operation['numerical_trial']['native_family'] = bad
    with pytest.raises(ValueError, match='boolean'):
        trial.validate_operation(operation)


def test_extra_fields_and_unrelated_families_do_not_enter_new_scope():
    operation, *_ = inputs((True, True, True))
    for field, value in [('graph_family', 'captured'), ('rn_heads_per_block', 8), ('dense_projections', {})]:
        changed = copy.deepcopy(operation)
        changed['prepared_execution'][field] = value
        with pytest.raises(ValueError):
            trial.validate_operation(changed)
    operation['prepared_execution']['native_family'] = 'other'
    with pytest.raises(ValueError):
        trial.validate_operation(operation)


def test_candidate_cannot_generate_its_own_reference(monkeypatch):
    monkeypatch.setattr(trial.numa, 'load_inputs', lambda _: {'plan': {'ordered_gpu_uuids': ['gpu']*8, 'ranks': [{}]*8}})
    for flags in itertools.product((False, True), repeat=3):
        operation, case, profile, plan, client = inputs(flags)
        if any(flags):
            with pytest.raises(ValueError, match='control captures'):
                trial.validate_teacher_operation_plan(operation, plan, client, case=case, profile=profile)
        else:
            assert trial.validate_teacher_operation_plan(operation, plan, client, case=case, profile=profile)['mode'] == 'capture'


def test_typed_trial_does_not_allow_profile_or_client_substitution(monkeypatch):
    operation, case, profile, plan, client = inputs()
    monkeypatch.setattr(trial.numa, 'load_inputs', lambda _: {'plan': {'ordered_gpu_uuids': ['gpu']*8, 'ranks': [{}]*8}})
    profile['preparation']['native_family'] = trial.AXES['native_family']
    with pytest.raises(ValueError, match='manifest'):
        trial.validate_teacher_operation_plan(operation, plan, client, case=case, profile=profile)
    client['numerical_trial']['native_family'] = True
    with pytest.raises(ValueError):
        trial.validate_teacher_client(client)


def test_new_schema_dispatch_is_distinct_from_old_oracles():
    assert model.teacher_contract({'schema': trial.SCHEMA}) is trial
    assert model.teacher_contract({'mode': trial.MODE}) is trial
    assert model.teacher_contract({'schema': trial.numa.SCHEMA}) is trial.numa
    assert model.prepared_teacher_contract() is trial.base


def test_engine_requires_actual_retained_capabilities(monkeypatch):
    from tools import v100_owned_qualification as owner
    operation, *_ = inputs((True, True, True))
    monkeypatch.setattr(owner, '_native_family_receipt', lambda e: {'actual': 'verified by owner'})
    monkeypatch.setattr(owner, '_prepared_m32_receipt', lambda e: {'actual': 'verified by owner'})
    monkeypatch.setattr(owner, '_owned_q8_preparation_receipt', lambda e: {'window_roles': {'decode': {}}})
    trial.validate_engine(SimpleNamespace(), operation)
    monkeypatch.setattr(owner, '_native_family_receipt', lambda e: None)
    with pytest.raises(ValueError, match='retained'):
        trial.validate_engine(SimpleNamespace(), operation)


def test_retention_identity_refactor_preserves_original_fingerprint_input():
    from tensorfold.cuda.markers import MIN_GAP
    from tensorfold.families.qwen4_exp.cuda.execution_identity import retention_identity_payload
    from tensorfold.families.qwen4_exp.cuda.prompt_plan import CUTS, POINTS
    engine = SimpleNamespace(w=SimpleNamespace(meta={'history_layout': 'unique', 'attention_degree': 4}),
        indexer_history='gpu', multi=SimpleNamespace(full_windows=True, keep=24), points=None,
        capacity_plan={'execution_envelope': {'scope': 'fixture'}}, _owner_preparation=SimpleNamespace(identity=lambda: {'actual': 'fixture'}))
    expected = {'history_layout': 'unique', 'attention_degree': 4, 'indexer_history': 'gpu',
        'full_windows': True, 'keep': 24, 'recovery_points_enabled': False, 'recovery_points_per_piece': POINTS,
        'interior_points': CUTS, 'minimum_gap': MIN_GAP, 'execution_envelope': {'scope': 'fixture'},
        'owner_preparation': {'actual': 'fixture'}}
    assert retention_identity_payload(engine) == expected


def test_full_vocabulary_metrics_pair_explicit_families_without_hiding_differences(tmp_path):
    import hashlib

    import numpy as np

    from tools import v100_rotorquant_quality as quality
    def side(name, preparation, bump):
        root = tmp_path/name
        root.mkdir()
        values = np.zeros(248320, dtype='<f4')
        values[7] = bump
        path = root/'frame.npy'
        np.save(path, values, allow_pickle=False)
        raw = path.read_bytes()
        vector = {'file': 'frame.npy', 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                  'shape': [248320], 'dtype': 'float32', 'full_vocab': True}
        frame = {'step': 0, 'position': 3, 'teacher_id': 7, 'full_vocab': 248320, 'role': 'decode',
                 'execution_family': preparation, 'selected_original': 7, 'vector': vector}
        return {'teacher_root': root, 'authority': {'vocab': 248320, 'frames': [frame]}}
    old = inputs()[0]['prepared_execution']
    new = inputs((True, False, False))[0]['prepared_execution']
    left, right = side('reference', old, 1.), side('candidate', new, 2.)
    with pytest.raises(ValueError, match='pairing'):
        quality._candidate_teacher_metrics(left, right, quality._teacher_read_control(30))
    rows, metrics = quality._candidate_teacher_metrics(left, right, quality._teacher_read_control(30),
                                                      execution_families=(old, new))
    assert rows[0]['different_uint32_values'] == 1
    assert rows[0]['candidate_nll'] < rows[0]['base_nll']
    assert metrics['ppl_ratio'] < 1 and metrics['exact_frames'] == 0
    with pytest.raises(ValueError, match='pairing'):
        quality._candidate_teacher_metrics(left, right, quality._teacher_read_control(30), execution_families=(old, old))


def test_numerical_registration_uses_actual_owner_absence_not_empty_alias():
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    for flags in itertools.product((False, True), repeat=3):
        operation, *_ = inputs(flags)
        owner = object.__new__(OwnerPreparation)
        owner.specification = operation['prepared_execution']
        owner.flash_validated = None
        owner.role_rows = {'decode': 24, 'prefill': 256}
        selected = owner.numerical_selection
        assert selected == trial.expected_numerics(operation['numerical_trial'])
        numbers = {'owned_expert_native': 'fixture'}
        if selected is not None:
            numbers['owner_preparation_numerics'] = selected
        trial.validate_numerics(numbers, operation['numerical_trial'])
        if selected is None:
            for bad in (None, {}):
                with pytest.raises(ValueError, match='absent/present'):
                    trial.validate_numerics({**numbers, 'owner_preparation_numerics': bad}, operation['numerical_trial'])


def test_all_off_startup_accepts_actual_absent_numerical_field_and_retention_digest():
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    from tensorfold.families.qwen4_exp.cuda.owner_preparation import OwnerPreparation
    from tests.test_v100_prepared_teacher_comparison import plan as expert_plan
    operation, *_ = inputs()
    owner = object.__new__(OwnerPreparation)
    owner.specification, owner.flash_validated = operation['prepared_execution'], None
    assert owner.numerical_selection is None
    retention = {'owner_preparation': {'specification': {
        k: v for k, v in operation['prepared_execution'].items() if k != 'trial'}}}
    plan = expert_plan(operation['selector'])
    profile = {'rank': 0, 'world': 8, 'target_banks': 48, 'mtp_banks': 0, 'capability': [7, 0],
        'plan_identity': plan, 'teacher_semantics': {'prepared_execution': operation['prepared_execution'],
            'prefill_rows': 256, 'target_rows': 24, 'context': 262144},
        'teacher_execution_registration': {'numerics': {}, 'binding': {'identity': {'retention': digest(retention)}}},
        'numerical_trial_retention': retention}
    row = {'rank': 0, 'world': 8, 'plan_identity': plan, 'teacher_profile': profile}
    trial.validate_startup(row, operation)
    profile['teacher_execution_registration']['numerics']['owner_preparation_numerics'] = {}
    with pytest.raises(ValueError, match='absent/present'):
        trial.validate_startup(row, operation)
