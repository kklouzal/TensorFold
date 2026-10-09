"""Bounded startup-only selection between exact NumPy victim-ranking cores.

Private int64 scratch; deterministic representative LFU/recency/tie patterns;
alternated raw measurements. No registered authority, frequency, recency,
resident or HOT timing mutation. Numeric order is identical for both algorithms.
Caller declares startup workspace/work budgets and retains this receipt under
its model/runtime ownership. The model-owned policy retains the selected cutoff and its raw provenance.
"""
import platform
import statistics
import time

import numpy as np

BLOCKED=(1<<63)-1
COUNTS=(2,4,8,12,16,20,24,28,32,36,40,48,64)
ROUNDS=9
PATTERNS=4


def scan(scores,count):
    chosen=[]
    for _ in range(count):
        slot=int(scores.argmin())
        if scores[slot]==BLOCKED:
            raise ValueError('unprotected rank capacity exhausted')
        chosen.append(slot)
        scores[slot]=BLOCKED
    return chosen


def partition(scores,count):
    threshold=np.partition(scores,count-1)[count-1]
    if threshold==BLOCKED:
        raise ValueError('unprotected rank capacity exhausted')
    below=np.flatnonzero(scores<threshold)
    ties=np.flatnonzero(scores==threshold)[:count-len(below)]
    chosen=np.concatenate((below,ties))
    order=np.argsort(scores[chosen],kind='stable')
    return [int(slot) for slot in chosen[order]]


def calibrate(capacity,*,workspace_budget_bytes,rank_work_budget_elements):
    """Return one evidence-backed cutoff and complete bounded raw provenance."""
    for name,value in (('capacity',capacity),('workspace budget',workspace_budget_bytes),('rank work budget',rank_work_budget_elements)):
        if type(value) is not int or value<1:
            raise ValueError(name+' must be a positive integer')
    if capacity>2**31-1:
        raise ValueError('rank capacity exceeds original signed-int32 policy bound')
    counts=tuple(count for count in COUNTS if count<=capacity)
    workspace_bound=64*capacity+65536
    work_per_round=capacity*(sum(counts)+10*len(counts))*PATTERNS
    rounds=min(ROUNDS,rank_work_budget_elements//max(1,work_per_round))
    rounds-=1-rounds%2
    if workspace_bound>workspace_budget_bytes or rounds<3:
        # Optional optimization discovery cannot invalidate a pool whose
        # original policy/storage allocations are otherwise supported. Preserve
        # its original victim strategy; do not guess an unmeasured cutoff.
        return {'schema':1,'status':'uncalibrated-original-victims','capacity':capacity,'scan_limit':None,
                'count_variants':counts,'patterns':PATTERNS,'rounds':0,'rank_calls_bound':0,
                'rank_work_elements_bound':0,'private_workspace_bytes_bound':0,
                'candidate_workspace_required_bytes':workspace_bound,'candidate_work_per_round':work_per_round,
                'workspace_budget_bytes':workspace_budget_bytes,'rank_work_budget_elements':rank_work_budget_elements,
                'reason':'optional representative rank search does not fit its private startup budget',
                'numpy':np.__version__,'machine':platform.machine(),'Python':platform.python_version(),
                'authority_residency_frequency_mutation':False,'HOT_timing':False}
    work_bound=work_per_round*rounds
    indices=np.arange(capacity,dtype=np.int64)
    stride=65*capacity+2
    patterns=[('all_equal',np.zeros(capacity,dtype=np.int64)),
              ('LFU_tied',(indices%17)*stride+indices//9),
              ('LFU_permuted',(indices*17%256)*stride+indices*37%max(capacity,2))]
    protected=patterns[2][1].copy()
    protected[-min(9,capacity):]=BLOCKED
    patterns.append(('protected_LFU',protected))
    scratch=np.empty(capacity,dtype=np.int64)
    rows=[]
    started=time.perf_counter_ns()
    for label,values in patterns:
        eligible=int((values!=BLOCKED).sum())
        for count in counts:
            if count>eligible:
                continue
            timings={'scan':[],'partition':[]}
            for repeat in range(rounds):
                order=('scan','partition') if repeat%2==0 else ('partition','scan')
                for kind in order:
                    np.copyto(scratch,values)             # Input restoration is outside the timed rank core.
                    begin=time.perf_counter_ns()
                    scan(scratch,count) if kind=='scan' else partition(scratch,count)
                    timings[kind].append(time.perf_counter_ns()-begin)
            rows.append({'pattern':label,'count':count,'eligible':eligible,'samples_ns':timings,
                         'median_ns':{kind:statistics.median(sample) for kind,sample in timings.items()}})
    # Select one monotone cutoff by complete measured objective, not a guessed
    # capacity interpolation. Zero means partition for every count>=2.
    options=(0,*counts)
    costs={cutoff:sum(row['median_ns']['scan' if row['count']<=cutoff else 'partition'] for row in rows) for cutoff in options}
    cutoff=min(options,key=lambda value:(costs[value],value))
    return {'schema':1,'status':'calibrated','capacity':capacity,'scan_limit':cutoff,'count_variants':counts,'patterns':len(patterns),
            'rounds':rounds,'rank_calls_bound':2*PATTERNS*rounds*len(counts),'rank_work_elements_bound':work_bound,
            'private_workspace_bytes_bound':workspace_bound,'workspace_budget_bytes':workspace_budget_bytes,
            'rank_work_budget_elements':rank_work_budget_elements,
            'work_budget_definition':'bounded rank-input element units: countargmin sweeps plus10full-vector equivalent input passes perpartition; notCPUinstruction/walltime guarantee',
            'raw_rows':rows,'cutoff_objective_ns':costs,
            'elapsed_ns':time.perf_counter_ns()-started,'numpy':np.__version__,'machine':platform.machine(),
            'Python':platform.python_version(),'objective':'minimum sum of pattern/count median rankcore latency; lower cutoff on evidence tie',
            'authority_residency_frequency_mutation':False,'HOT_timing':False}
