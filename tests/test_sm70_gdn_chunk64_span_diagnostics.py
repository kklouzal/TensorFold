"""CPU procedure controls only; synthetic records do not prove native results."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.state import State
from tools import v100_gdn_chunk64_trial as trial


def report():
    metrics = {'finite':True,'reference_nonfinite_elements':0}
    layer = {'actual_cut_bank_copy_bits':True,'last_span_initial_bank_unchanged_vs_cut_copy_bits':True,
        'cut_conv_tail_bits':True,'final_conv_tail_bits':True,'immutable_projection_weights_parameters_bits':True,
        'inactive_capacity_bits':True,'frontend':{name:dict(metrics) for name in ('q','k','v','raw_gate','beta')},
        **{key:dict(metrics) for key in ('output_vs_serial','sums_vs_serial','raw_vs_serial','final_state_vs_serial',
             'cut_state_vs_serial','epilogue_actual_raw_output','epilogue_actual_raw_sums')}}
    records = [{'rows':rows,'cut':cut,'capacity':trial.SPAN_STATE_CAPACITY,
        'bank_parity':True,'both_layers_submitted_before_explicit_retirement':True,
        'adapter_event_synchronize_calls_unchanged_before_retirement':True,'active_reuse_refused':True,
        'alias_refused':True,'bad_cut_refused':True,'preflight_state_bits_unchanged':True,
        'positive_retirement':True,'storage_metadata_unchanged':True,'state_reader_journal_empty':True,
        'layers':[{**copy.deepcopy(layer),'layer':index,
                   'decay_representation':'GGUF-A_neg' if index == 0 else 'HF-A_log'} for index in range(trial.SPAN_STATE_LAYERS)]}
        for rows,cut in trial.SPAN_STATE_CASES]
    return {'schema':trial.SPAN_STATE_SCHEMA,'phase':'spans-state','main_numerical_cases':'unrun',
        'quality_acceptance':False,'performance_qualified':False,'case_count':len(records),
        'capacity':trial.SPAN_STATE_CAPACITY,'layers':trial.SPAN_STATE_LAYERS,'results':records,
        'host_failure_control':{'genuine_completion_before_host_exception':True,
            'single_span_bank_publication':True,'single_span_initial_bank_readonly_bits':True,
            'primary_preserved':True,'state_poisoned':True,'owners_retained':True}}


def test_real_state_constructor_allocates_only_finite_diagnostic_geometry():
    state = trial.diagnostic_state(torch.device('cpu'))
    assert type(state) is State and state.conv.shape == (2,3,1280)
    assert state.rec.shape == (2,2,6,128,128) and state.rec.dtype == torch.float32
    assert state.execution_dtype == torch.float16 and state.kc == [] and state.layers == 0
    assert state.mtp_len == 0 and len(state.scratch) == 2
    assert state.capacity == trial.SPAN_STATE_CAPACITY


@pytest.mark.parametrize('dtype',[torch.float16,torch.float32])
def test_copy_bit_oracle_distinguishes_signed_zero(dtype):
    a = torch.tensor([0.,-0.],dtype=dtype)
    b = torch.tensor([-0.,0.],dtype=dtype)
    assert torch.equal(a,b)
    assert not trial.same_storage_bits(a,b)
    assert trial.same_storage_bits(a,a.clone())


def test_event_observer_counts_only_delegated_api_and_host_failure_follows_completion():
    calls = []
    event = trial.ObservedCUDAEvent(SimpleNamespace(record=lambda stream:calls.append(('record',stream)),
                                                    synchronize=lambda:calls.append('actual-completion')))
    event.record('stream-token')
    assert event.record_calls == 1 and event.synchronize_calls == 0
    primary = RuntimeError('declared host failure')
    event.host_failure = primary
    with pytest.raises(RuntimeError) as caught:
        event.synchronize()
    assert caught.value is primary and calls == [('record','stream-token'),'actual-completion']
    assert event.synchronize_calls == event.completed_synchronizations == 1


@pytest.mark.parametrize('fault',['main','quality','census','copy','finite','layers','completion','bank-publication','initial-borrow'])
def test_finite_procedure_audit_cannot_admit_missing_cases_or_quality_claims(fault):
    value = report()
    trial.validate_spans_state_result(value)
    if fault == 'main':value['main_numerical_cases'] = 'completed'
    elif fault == 'quality':value['quality_acceptance'] = True
    elif fault == 'census':value['results'].pop()
    elif fault == 'copy':value['results'][0]['layers'][0]['actual_cut_bank_copy_bits'] = False
    elif fault == 'finite':value['results'][0]['layers'][0]['output_vs_serial']['finite'] = False
    elif fault == 'layers':value['results'][0]['layers'][1]['layer'] = 0
    elif fault == 'bank-publication':value['host_failure_control']['single_span_bank_publication'] = False
    elif fault == 'initial-borrow':value['host_failure_control']['single_span_initial_bank_readonly_bits'] = False
    else:value['host_failure_control']['genuine_completion_before_host_exception'] = False
    with pytest.raises(SystemExit):
        trial.validate_spans_state_result(value)


def test_spans_phase_routes_its_own_audit_and_leaves_legacy18_unrun(monkeypatch,tmp_path):
    value = report()
    monkeypatch.setattr(trial,'run',lambda args,progress: value)
    output = tmp_path/'report.json'
    trial.main(['--cuda-home','/not-used','--nvcc-sha256','0'*64,'--phase','spans-state',
                '--output',str(output),'--stream-trace-output',str(tmp_path/'reserved-unused-trace.json')])
    actual = json.loads(output.read_text())
    assert actual['main_numerical_cases'] == 'unrun' and actual['schema'] == trial.SPAN_STATE_SCHEMA
    assert actual['quality_acceptance'] is actual['performance_qualified'] is False
    assert not (tmp_path/'reserved-unused-trace.json').exists()


def test_nonfinite_reference_diagnostics_remain_json_safe_without_hiding_counts():
    result = trial.boundary_errors(torch.tensor([1.]),torch.tensor([float('nan')]))
    assert result['reference_nonfinite_elements'] == 1 and result['reference_max_abs'] is None
    json.dumps(result,allow_nan=False)


@pytest.mark.parametrize('unexpected',[False,True])
def test_controlled_host_failure_never_masks_an_unexpected_native_error(unexpected):
    primary = RuntimeError('declared host observer failure')
    actual = RuntimeError('authentic unexpected native retirement failure') if unexpected else primary
    def wait():
        raise actual
    frame = SimpleNamespace(wait_native_reuse=wait)
    if unexpected:
        with pytest.raises(RuntimeError) as caught:
            trial.retire_controlled_host_failure(frame,primary)
        assert caught.value is actual
    else:
        assert trial.retire_controlled_host_failure(frame,primary)
