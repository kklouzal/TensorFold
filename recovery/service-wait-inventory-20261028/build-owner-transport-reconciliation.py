import ast, collections, hashlib, json, pathlib, subprocess
R=pathlib.Path('/Warehouse/TensorFold-portable-audit-20261026')
P=R/'gpu-wait-inventory-20261028'
W=pathlib.Path('/Warehouse/TensorFold-wt/v100-joint-hc-moe-20261028')
BASE='28ac5b2da8660542c156adf60b8a3e5072b89244'
HOT='f0300ac8166349882a3de5f82687e0bf836eeff5'
OLD='7de2259b1a402542b91fde9e906a9ebb07f4c949'
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((P/n).read_text())
def git(c,p):return subprocess.check_output(['git','-C',str(W),'show',c+':'+p])
register={x['id']:x for x in read('drilldown-register.json')['entries']}
owner=read('owner-graphs-inventory.json');transport=read('transport-inventory.json')
byid={x['id']:x for x in owner['sites']}
E={
 'owner_assessment':'gpu-wait-inventory-20261028/owner-drilldown.json',
 'graph_assessment':'gpu-wait-inventory-20261028/graph-dispatch-drilldown.json',
 'hc_assessment':'gpu-wait-inventory-20261028/hc-readout-drilldown.json',
 'service':'gpu-wait-inventory-20261028/service-drilldown.json',
 'startup':'gpu-wait-inventory-20261028/startup-io-drilldown.json',
 'decode':'gpu-wait-inventory-20261028/decoding-entry-register.json',
 't60_source':'gpu-wait-inventory-20261028/transport-t60-drilldown.json',
 't60_actual':'input-mailbox-selfwait-20261028/actual-first-all6/comparison-vs-86cb.json',
 't60_review':'input-mailbox-selfwait-20261028/combined-review-final.json',
 'fifo_actual':'fifo-snapshot-combined-20261028/selection.json',
 'ring_actual':'owner-descriptor-ring-20261027/actual-first-all6/root-ring4-vs-best-v2.json',
 'ring_review':'owner-descriptor-ring-20261027/combined-review-final.json',
 'frame_coverage':'gpu-wait-inventory-20261028/owner-drilldown/frame-1d83-counter-coverage.json',
 'frame_source':'frame-descriptors-integration-20261028/ledger.json',
 'n1_actual':'joint-single-profile-20261028/actual-findings.json',
 'candidate_actual':'joint-candidate-finish-integration-20261028/actual-first-all6-fbaa/independent-audit.json',
 'hc_actual':'joint-hc-moe-integration-20261028/actual-first-all6-4d2b/independent-audit.json',
 'hc_source':'joint-hc-moe-integration-20261028/structural-source-proof.json',
 'hc_review':'joint-hc-moe-integration-20261028/combined-review-final.json',
 'pool_actual':'joint-hc-moe-pool-expansion-20261028/actual-one-pool-5cd45/pytest-result.json',
 'pool_source':'joint-hc-moe-pool-expansion-20261028/source-proof.json',
 'pool_scope':'joint-hc-moe-pool-expansion-20261028/checkpoint-scope.json',
 'packed_source':'coalesced-candidate-readback-20261028/structural-proof-resumed.json',
 'packed_review':'coalesced-candidate-readback-20261028/combined-review-resumed-final.json'
}
for n in E.values():assert (R/n).is_file(),n
G={}
def group(k,activation,outcome,assessment,remaining,evidence):
 G[k]=dict(current_activation=activation,source_assessed_outcome=outcome,assessment=assessment,remaining_proof_or_profile=remaining,evidence_keys=evidence)
group('descriptor_host','Current native target/MTP attention cuts, including eager owner inserts between N1 bound pieces. CPU mirror fence and GPU reader/upload dependencies are separate.','Required ownership waits retained; new bounded host-construction alternative genuinely pending.',
 'stage_native_descriptors repeatedly rebuilds State storage signatures and 16-word descriptor rows, allocates a temporary CPU torch.tensor per live row, then copies the existing pinned mirror H2D. Native_done protects device table reuse; descriptor_upload CPU synchronization protects only reuse of the single pinned mirror. Reducing Python/CPU temporary construction is independent of changing upload placement or replacing that mirror.',
 'Prove exact descriptor bits, int64 rejection, target/MTP pointer roles, inactive-row zeroing, generations, lane offsets and error ordering. A static base cache needs full storage/shape/dtype/layout/placement invalidation and strong ownership through native readers. Any once-frame signature validation needs explicit no-rebind/resize/migration/reset lease. Profile host construction separately from deferred DMA wait; then same-instrumentation all-six comparison.',
 ['owner_assessment','n1_actual','ring_actual','frame_coverage','frame_source'])
group('batch_prelude','Current native dense owner batch prelude when selected; one multi-State staging packet per submit.','Required single pinned-slab reuse frontier retained.',
 'BatchPrelude.submit owns bounded [rows,4] pinned inputs. Its upload completion guards host overwrite, while subsequent GPU work retains device-side dependencies. It is not interchangeable with the history descriptor mirror or a full round fence.',
 'A separate multi-slot or prebound host-fill alternative needs exact batch ordering, inactive lanes, generations, capacity/funding, positive retire/poison and integrated all-six evidence. No dedicated current per-site timing establishes a bottleneck.', ['owner_assessment','startup'])
group('state_generation','Conditional resize/new-sequence/reset transitions; generation publication can occur between requests, not every eager kernel.','Required raw-reader retirement before canonical State mutation.',
 'Resize/recovery and generation publication must outlive every borrowed pointer and prior native reader. Generation words are authority, not a cache that may be reordered around outstanding readers.',
 'Profile transition frequency/tails before considering an event-scoped retirement replacement. Prove all streams/raw consumers, aliases, cancellation and failure paths before narrowing global/stream completion.', ['owner_assessment','startup','service'])
group('gpu_dependency','Current conditional cross-stream/cache staging or two-State transfer.','GPU stream dependency, not CPU readback; retain unless issue-order proof covers every producer/consumer.',
 'wait_event orders consuming stream after the producer and permits the CPU to enqueue onward. Same-stream issue order can satisfy a proved edge; different-stream staging, migration and reuse cannot assume it.',
 'Need actual stream identities and all supported transfer/owner modes plus lifetime and failure proof. A source wait is not a measured CPU stall or independently removable time.', ['owner_assessment'])
group('mailbox','Current input preparation phases before dependent collective/forward work.','Source-assessed protocol; ADD-return self-wait elision tested but NOT SELECTED from first all-six screen.',
 'Ready confirms all payloads published; consumed and ack prevent deleting or reusing keys while another rank still borrows them. Native Store ADD response ordering can certify the final publisher self condition only; the historical candidate preserved published latches/deletion/poison. That candidate is absent from current selected base.',
 'No blind fence removal. A new phase-work elimination proposal needs exact per-frame ordinal/contract/error/slot-reuse proofs, both Store backends and current integrated all-six evidence. Existing no-gain screen does not prove global impossibility or repeatable inferiority.', ['t60_source','t60_actual','t60_review','n1_actual'])
group('bound_capture','Selected N1 BoundGraphs cold preparation/key agreement; replay maps remain dynamic at owner cuts.','Cold capture/identity/retirement fences required; replay topology assessed, no implemented 15-to-2 change.',
 'Current target verify graph has 15 pieces and MTP has 2 pieces. Canonical owner work and map/readiness boundaries remain between pieces; 15 and 2 describe different roles. Captured compute does not itself remove State authority, descriptor inputs, collectives or status completion.',
 'A larger boundary needs dynamic State binding and prior/following collective/owner lifetime proof, exact numerical oracle, cold memory funding and integrated all-N evidence. Cold capture vote/D2H and final retirement are not steady replay costs.', ['graph_assessment','n1_actual','hc_actual','pool_scope'])
group('bound_maps','Current unique N1 static map GPU event dependency; nonunique reference branch separately classified.','Map publication/reuse is required; same-stream or stable-binding simplification needs a specific contract.',
 'Static unique maps preserve per-State canonical map meaning while an event orders consumers after staging. Native attention remains eager between BoundGraph pieces. This is not host status readback.',
 'Prove key completeness, map stability across generation/resize/reuse, and producer/consumer streams. Do not replace event ordering with object identity alone.', ['graph_assessment','n1_actual'])
group('bound_reference_map','Conditional nonunique/reference map branch; inactive for the selected unique N1 static mapping.','Required CPU completion on its supported reference branch; not counted as current unique-path hot wait.',
 'Host event synchronization makes the reference mapping publication/reuse safe. The selected unique path uses a stream wait instead.',
 'Any change requires a workload actually reaching this branch and preservation of nonunique map/error/lifetime behavior; no selected-path gain inferred.', ['graph_assessment','owner_assessment'])
group('multi_capture','Current multi-row graph preparation and retirement. 28ac has target rows30/31/32; f030 extends to 16 hot rows using one owned pool.','Current graph work already partly covered by MoE+HC/candidate integration; f030 result is Root-owned and not inferred here.',
 'HC/MoE catalogue captures original pure math after prior NCCL and before following partial/NCCL. Candidate catalogue separately captures the original packet producer. State/attention/PLE and status readback stay outside. Cold global fences, capture stream handoffs, keyed storage and positive resets preserve lifetime. Current HC hits in 28ac first screen occur only at N16/N24.',
 'Pool expansion requires fixed complete48-layer order, nonconcurrent execution, actual pool identity, stable bindings and outputs consumed before reuse; explicit budget deduplication and full24 funding remain required. Test-only192-key/384-replay pool oracle is not Engine/accounting/P3 qualification. R1..3 full catalogue enrollment remains blocked by N1 bound-plan participation/cursor contract.', ['hc_assessment','hc_source','hc_review','hc_actual','candidate_actual','pool_actual','pool_source','pool_scope'])
group('optional_owner_graph','Optional owner-attention graph family; selected current manifest remains UNCAPTURED for that family.','Assessed conditional capture path; not a current native-owner graph replay optimization.',
 'Optional capture contains existing owner kernels and two collectives. Its all-rank agreement, side/current stream joins, cold global synchronization and close fence protect identical participation and graph storage. Current eager native-owner insertions remain.',
 'Eligibility, all-rank shape/collective order, dynamic descriptors and capture ownership must be proven on the intended selected profile before an actual oracle and integrated comparison. Do not count optional captures as current hits.', ['owner_assessment','graph_assessment','n1_actual'])
group('native_cold','Cold authoritative native role/slab/kernel/wire preparation before admission.','Required prepared residency and funding boundary, not per-token global fence.',
 'Native preparation warms bounded launch envelopes, role buffers and NCCL wire before materialization/admission publication. Final fences certify cold work and release ordering.',
 'Startup overlap or more precise events need rank-consistent readiness, allocation/error containment, unchanged funded envelopes/full24 floor and startup-tail evidence. No steady throughput saving from source-only cold fences.', ['owner_assessment','startup'])
group('input_mirror','Current target or MTP input preparation; target IDs/PLE or MTP mirror reuse.','Required pinned host DMA mirror retirement; source work-elimination candidate still proof/profile blocked.',
 'b.staged synchronization permits rewriting pinned host inputs after prior asynchronous copies; it is narrower than a model-completion fence. Removing it without another owned mirror can race DMA.',
 'Profile host preparation and copy completion per role; prebinding/reusing host representations can preserve the same fence. A ring needs bounded prepaid slots, actual per-slot event retirement and cancel/failure proof; avoid extending tensor/view lifetime accidentally.', ['owner_assessment','hc_assessment','n1_actual','startup'])
group('status','Current KV numeric/status completion and containment.','Required status-first observation; packed readback/candidate graphs are separate assessed changes.',
 'KVWords.settle and all-rank status agreement prevent state/token publication following invalid numerical work. Packed candidate readback keeps one completion frontier; graphing the candidate producer does not authorize sampling before status validation.',
 'Map each status domain to the exact candidate packet owner. Preserve first-error precedence, all-rank failure, positive DMA completion and RNG/tie/EOS order. Split readback alternative remains inconclusive; retained packed incumbent is not proof every stream synchronization can be coalesced.', ['decode','packed_source','packed_review','candidate_actual','n1_actual'])
group('exception_retire','Conditional retained-request failure/cancellation/retirement, not normal steady round.','Required synchronization before State retirement/reuse and primary-safe cleanup.',
 'Error paths must prove no GPU reader still references retained State before release or generation reuse. Retire callbacks and owner counts must complete even when another failure is active.',
 'Only narrow after source-proven last-consumer event coverage across all streams and terminal poison behavior. Exercise cancel/error/cleanup failures; normal throughput profiles cannot justify deleting exceptional retirement.', ['service','owner_assessment'])
group('construction','Managed startup/profile-construction transactions only; ordinary hot replay is direct.','Required coherent all-rank publication and read-ack/key reclamation.',
 'Construction barriers ensure every rank either commits the same collective/prepared resource or reports failure. Finish outcomes and read acknowledgements keep Store keys live until every reader is done; local device fence certifies cold work.',
 'Any reduced cold barrier needs exact managed-vs-hot caller proof, rank-order/failure/timeout and key lifetime proof plus startup evidence. Poll sleep is conditional waiting for GO, not an unconditional decode delay.', ['startup','graph_assessment','owner_assessment'])
group('prefix_migration','Conditional prefix restore/copy or balanced ownership migration; fixed resident benchmark normally avoids history growth/migration.','Required payload/page/generation transfer retirement.',
 'Source and destination owners retain KV/scales/pages/coordinates until stream/device and all-rank completion. Generation publication follows successful migration, and partial failure must not expose reused tables.',
 'Measure actual transition frequency, payload and cross-rank critical path. Event narrowing/overlap needs all borrowed pointer readers and producer/consumer dependencies, atomic generation publication, rollback/poison and retained-session I/O proof.', ['owner_assessment','service','startup'])
group('store_startup','Current process/Store/NCCL startup before decode.','Required endpoint identity, cohort/backend agreement and bootstrap.',
 'Owned legacy TCPStore constructor/connect/listener and reply-fenced nonce handshake establish exact daemon provenance and common backend/world. wait_for_workers=False removes a constructor cohort barrier only. Required waits cannot be confused with per-token work.',
 'Startup overlap must retain listener ownership, nonce reply ordering, all-rank backend/UID agreement, timeout/cleanup and failure diagnosis. No steady throughput gain without a hot caller.', ['startup','owner_assessment'])
group('nccl_stream','Current TP communication at model cuts; host wrapper may block on its close lock/native issue, while collective progress is a GPU/vendor dependency.','Required collective order, rank-ordered math and buffer lifetime retained.',
 'Plain NCCL current-stream enqueue is selected. FP32 partials are consumed in deterministic rank order; all-reduce or head-group transport substitutions are not numerically/operationally equivalent by assertion. Wrapper has no owned peer-spin/fence algorithm. HC/MoE graphs deliberately stop before next NCCL; candidate graphs stop before candidate gather.',
 'Need same-REGION all-rank enqueue/start/end/consumer attribution to distinguish skew, backlog and communication. Any collective reduction needs participant/order, head-group, dtype/math, in-place slice and failure/retirement proof. Vendor polling is contract/profile evidence, not an exhaustive NCCL audit or removable wall time.', ['owner_assessment','n1_actual','hc_actual','candidate_actual'])
group('reference_wire','Retained reference owner attention/warmup transport, conditional to reference selection; current native wire family separately accounted.','Required deterministic normalization/PV wire and supported reference behavior.',
 'Reference normalization partial maxima/denominators and PV merges require head-group participation and rank order. Warm_reference_transport also issues these wires during cold native preparation.',
 'Separate cold warmup from any actual reference fallback hits before optimizing. Preserve exact normalization and FP32 accumulation, transfer counts and participation; no current native-hot time inferred.', ['owner_assessment','startup'])
group('admission','Current cold all-rank capacity/profile/materialization seal; growth votes are conditional when resident capacities change.','Required common policy and complete funding before divergent allocation or publication.',
 'GPU votes/D2H establish common row/session/history/precision/materialization policy, minimum available room and complete full24 commitments. Host/GPU capacity are different authorities. Current fixed full-resident decode need not repeat a growth vote per frame.',
 'Consolidation needs unchanged decision/error ordering, all locally mutable facts revalidated at commitment, resource lifetime and full24 floor. Need actual startup/growth call counts before a hot-path claim. A seal or host scalar metadata is not an ongoing tensor readback.', ['startup','pool_scope','hc_actual'])
group('tp_control','Current concurrent TP controller protocol around frames and request/control boundaries.','Source-assessed required command/order/outcome protocol; bounded host serialization alternative pending specific evidence.',
 'Rank0 publishes one command/ordinal, peers execute identical choices, digest outcomes and ACK before slot/key reclamation. _state_digest uses CPU metadata/output lists, not direct CUDA readback. Follower Store waits are progress waits; GIL/JSON/hash work can delay publication even after device work completes.',
 'A prepared immutable command header or batched notifications may reduce repeated host construction, but cannot merge across distinct error/state authority or reclaim before ACK. Need current per-command counts/exclusive host costs, byte-identical schema/digest checks, failure/slow-rank/reuse proof and integrated evidence.', ['service','t60_source','n1_actual'])
group('stop_decision','Current conditional token/stop decision handoff before State publication.','Required rank-consistent decision; no stop/RNG authority migration authorized.',
 'Peers consume rank0 decision before committing token state. This is separate from candidate generation and status completion; omission/reordering may diverge future collectives.',
 'Candidate graph/packed readback evidence covers producers/transport only. A reduced handoff needs exact stop/EOS/grammar/RNG/commit ordering, stale-slot and exception handling, then full output/counter validation.', ['decode','candidate_actual','service'])
group('readiness','Current startup readiness barrier or conditional communicator barrier.','Required phase completion and diagnostics; device-wide barrier is stronger than an enqueue.',
 'Ready polling with timeout is conditional waiting, not a fixed60s delay. Comm.barrier allocates tiny scratch and completes device work after gather; startup construction uses explicit phase readiness before warm/allocation.',
 'Trace actual callers/frequency before replacing with a narrower event/all-rank marker. Retain all preceding stream work, late error detection, timeouts and terminal cleanup. No active decode invocation assumed for startup APIs.', ['startup','owner_assessment'])
group('comm_check','Conditional asynchronous communicator health check.','CPU native status query; successful check is NOT GPU completion.',
 'ncclCommGetAsyncError observes deferred native error under close ownership. It neither fences consumers nor certifies send/receive completion; current CDLL calls release GIL while Python validation/lock/formatting does not.',
 'Only profile if polling is hot. Preserve actual deferred-error/fatal containment and close exclusion; do not use success to authorize storage reuse.', ['owner_assessment','service'])
group('comm_close','Shutdown/terminal failure lifecycle, not normal decode.','Required no-new-admission plus device/native/Store retirement.',
 'Close lock excludes concurrent submissions; GPU resources and collective buffers must outlive work until documented terminal retirement. Native destroy/abort and owned daemon outcome are separately observed.',
 'Any asynchronous drain proposal needs defined caller completion, deadlines, positive fence or terminal poison, all child outcomes and primary-safe cleanup. Removing final wait creates a lifetime change.', ['service','startup'])
group('serial_api','Serial Engine API paths; current finite all-six uses concurrent scheduler/TP controller.','Supported conditional API, inactive as the current concurrent hot loop.',
 'Serial Store coordination still has rank-order/error semantics. App serial Turns are bypassed for engine.concurrent=True; MLX scheduler is not the active CUDA path.',
 'Only assess gain on a caller actually taking serial path; retain serial correctness/diagnostics without counting as concurrent round overhead.', ['service','startup'])
group('graph_policy','Current graph execution policy/configuration and vendor-managed NCCL dependencies.','Configuration/contract boundary; not an independently measured owned wait.',
 'NCCL graph policy, selected profile and capture scope determine legality. Current pure HC/MoE/candidate regions exclude surrounding collectives; N1 preexisting captures retain their own contracts.',
 'No policy toggle without pinned runtime support, identical rank behavior, exact component oracle, stable storage/funding and integrated evidence. Do not infer vendor graph pools or synchronization from option names.', ['graph_assessment','hc_actual','pool_source'])
group('inactive_direct_upload','Generic checkpoint CUDA ReadAhead/upload staging; inactive selected target and GGUF MTP CPU materializer.','Conditional supported path retained; no current GPU mirror wait attributed.',
 'Generic direct_read GPU slab draining is distinct from the selected CPU archive materializer and PLE host read pools. MTP may use direct OS reads without selecting those GPU upload slabs.',
 'Any optimization requires proof that the caller selects this path and buffer/FD/future error retirement. No selected finite decode saving from inactive source.', ['startup'])
group('scheduler','Current CPU actor/control/output credit paths; HTTP delivery is conditional, finite benchmark callback runs on caller.','Source-assessed required credits/backpressure; FIFO snapshot candidate first-screen unselected.',
 'Submission caller blocks on result queue; actor controls GPU scheduling. Output callbacks/network wfile writes occur on caller threads, but all slow clients can exhaust bounded output credits and pause decode. Conditions/notification predicates and queue bounds protect progress and shutdown. FIFO snapshot elision was tested without changing those contracts and is absent from current selected source.',
 'Batched notifications or reduced host metadata work need a specific current profile and preserved fairness, control responsiveness, per-request order, credit accounting, cancellation/publisher drain and shutdown races. HTTP-only overhead must not be credited to finite no-HTTP runs.', ['service','fifo_actual','n1_actual'])
group('lifecycle','Current Engine lifecycle conditions and drain/close boundaries; waits often occur on startup/shutdown.','Required active-call/resource lifetime exclusion.',
 'Lifecycle Condition prevents release while accepted callers, workers or callbacks retain engine state. Cancellation or timeout alone does not retire those consumers.',
 'Need concrete contended hot-call evidence before a lock redesign; preserve closed/admission transitions, reentrancy, drain and primary/error outcome semantics.', ['service','startup'])
group('diagnostic_process','Finite probe/supervisor/cohort/launcher process and failure instrumentation; not shipping mathematical work.','Diagnostic/control overhead assessed; baseline and candidate instrumentation must remain equivalent.',
 'Worker process polling, OS pipes, burst barriers, readiness/Event sampling and Future drains enforce offered/completed population, timing window, failure coverage and no cross-cohort leaked owners. They can perturb host launch but are not optional success-only observations.',
 'Controlled overhead study needs identical baseline/candidate tool change preserving100Hz/all-N fairness/lifecycle/identity/error coverage and budget. No observer edit or shipping decode gain claimed; do not drop drains to improve reported throughput.', ['service','n1_actual'])
group('numa_inactive','Conditional RAM-history/NUMA path; selected text profile has GPU history.','Inactive placement/admission branch; no current actor/NCCL proxy affinity claim.',
 'Thread affinity and NUMA placement affect only the actual calling thread/archive ownership. Current GPU history bypasses HistoryPlan/prepare_history; inherited OS/container affinity remains uncontrolled separately.',
 'A placement proposal requires actual thread topology, host/cgroup/residency and memory commitments plus an affected workload. Do not infer placement from device ID or unused helper.', ['startup'])
group('startup_io','Current PLE/table prefetch/background startup futures and CPU row retrieval; generic upload fanout separately inactive.','Required I/O/future ownership and observed errors; bounded startup/active CPU work already assessed.',
 'Engine drains accepted startup reads; host_table._prefetch owns CPU I/O pools. Active PLE row retrieval can gate input preparation, while archive/materialization occurs cold. Three-file complete content hashing remains authoritative and verified.',
 'Separate actual row I/O/fault/warm residency from cold jobs; preserve future ownership, file/content identity, partial-I/O/error/cancel and host/cgroup commitments. No stat-only integrity cache or mlock policy edit is authorized.', ['startup','service','n1_actual'])
group('startup_hash','Current three-worker complete content hashing and publication; inventory originally described one worker.','Already retained verified startup change; stale original one-worker description corrected here.',
 'All required content results and accepted worker outcomes are drained before publication; retained current startup workers=3. Hashing is a cold authority boundary, not per-token work.',
 'No re-audit/reimplementation or stat-only cache. Any further overlap must preserve complete3-file hashing, source identity, bounded workers/host commitment and cleanup; no decoded throughput claim.', ['startup','hc_actual'])
group('inactive_p2p','Retained grouped exchange/send/recv APIs for expert-rank/media conditional callers, excluded current text/MTP selected profile.','Explicit noncurrent transport; not a current hot wait.',
 'Native group submission still borrows buffers and requires valid peers/counts/streams and eventual retirement, but no selected finite text caller is established for this API.',
 'Reopen only for an eligible caller; require collective/P2P progress, supported backend, cancellation/close and no scratch reuse before completion. No current optimization or timing claim.', ['owner_assessment','startup'])
# Explicit one-to-one group assignment; no residual default.
OM={
 'prefix_migration':['0102'], 'descriptor_host':['0135','0137','0142'],
 'native_cold':['0148','0769','0778','0780','0954'], 'exception_retire':['0150','0180','0899','0913'],
 'batch_prelude':['0273'], 'state_generation':['0379','0381'], 'gpu_dependency':['0529','0566','0577'],
 'mailbox':['0584','0585','0586'], 'bound_capture':['0637','0638','0639','0647','0665'],
 'bound_maps':['0655'],'bound_reference_map':['0656'],
 'multi_capture':['0707','0712','0715','0716','0725'],
 'optional_owner_graph':['0736','0747','0751','0754','0755','0756','0757'],
 'input_mirror':['0791','0856'],'status':['0842'], 'construction':['0935','0937','0940']}
TM={
 'store_startup':[1,2,3,4,5,19], 'nccl_stream':[6,7,8,9,10,11,16,17,52],
 'reference_wire':[12,13,14], 'native_cold':[15], 'status':[18],
 'admission':[20,21,22,23,57], 'tp_control':[24,25,27], 'stop_decision':[26],
 'readiness':[28,29,32], 'construction':[30,31], 'prefix_migration':[33,34],
 'comm_check':[35], 'comm_close':[36], 'serial_api':[37], 'graph_policy':[38],
 'descriptor_host':[39], 'exception_retire':[40], 'inactive_direct_upload':[41,42],
 'scheduler':[43,44,45,46], 'lifecycle':[47], 'diagnostic_process':[48,49,50,51],
 'numa_inactive':[53,54], 'startup_io':[55], 'startup_hash':[56], 'inactive_p2p':[58,59], 'mailbox':[60]}
om={f'owner-{n}':k for k,ns in OM.items() for n in ns}
tm={f'T{n:02d}':k for k,ns in TM.items() for n in ns}
assert set(om)==set(owner['triaged_wait_sites']) and len(om)==47
assert set(tm)=={x['id'] for x in transport['sites']} and len(tm)==60
source_manifest={}; source_cache={}
def functions(text):
 tree=ast.parse(text);out=[]
 def visit(node,prefix=''):
  for n in ast.iter_child_nodes(node):
   if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
    q=prefix+'.'+n.name if prefix else n.name
    if not isinstance(n,ast.ClassDef):out.append((q,n.lineno,n.end_lineno))
    visit(n,q)
   else:visit(n,prefix)
 visit(tree);return out

def bind(ref):
 p=ref['path']
 if p not in source_cache:
  b=git(BASE,p);h=git(HOT,p)
  try:old=git(OLD,p)
  except subprocess.CalledProcessError:old=b
  source_cache[p]=(b.decode(),old.decode())
  source_manifest[p]={'current28ac_sha256':sha(b),'current_f030_sha256':sha(h),'changed_28ac_to_f030':b!=h,'line_count28ac':len(b.splitlines())}
 text,old=source_cache[p];num=ref.get('line',1)
 of=[f for f in functions(old) if f[1]<=num<=f[2]] if p.endswith('.py') else []
 f=min(of,key=lambda x:x[2]-x[1]) if of else None
 matches=[x for x in functions(text) if f and x[0]==f[0]] if p.endswith('.py') else []
 loc={'path':p,'inventory_line':num,'current28ac_file_sha256':source_manifest[p]['current28ac_sha256']}
 if matches:
  q,a,z=matches[0];loc.update(current_function=q,current_function_start_line=a,current_function_end_line=z)
  oldline=old.splitlines()[num-1].strip() if num<=len(old.splitlines()) else ''
  exact=[i for i,l in enumerate(text.splitlines(),1) if a<=i<=z and oldline and l.strip()==oldline]
  loc['current_matching_statement_lines']=exact
 else:loc.update(current_function=None,current_function_start_line=1,current_function_end_line=len(text.splitlines()))
 return loc
entries=[]
for part,items,mapping in [('owner_graphs',[byid[n] for n in owner['triaged_wait_sites']],om),('transport',transport['sites'],tm)]:
 for x in items:
  ident=f"{part}:{x['id']}"; g=mapping[x['id']]; rg=register[ident]
  refs=[{'path':x['path'],'line':x['line']}] if part=='owner_graphs' else x['sources']
  entry={'id':ident,'assessment_group':g,'current_source_bindings':[bind(r) for r in refs],
   'original_call_chain':x.get('call_chain',x.get('callchain')),
   'original_frequency':x.get('frequency'),
   'original_mechanism':x.get('kind',x.get('wait_kind')),
   'required_site_specific_purpose':x.get('purpose',x.get('purpose_and_causality')),
   'site_specific_contract':x.get('correctness_requirement',x.get('needed_proof')),
   'inventory_candidate_or_needed_proof':x.get('candidate_and_needed_proof',{'candidate':x.get('candidate'),'needed_proof':x.get('needed_proof')}),
   'master_register_state_preserved':{k:rg.get(k) for k in ['state','disposition','candidate','proof_needed','evidence']},
   'assessment':G[g],
   'timing_claim':'No isolated current per-entry timing/gain is inferred. Diagnostic family timings, where linked, are overlapping/clipped host wall and version-specific.',
   'master_register_written':False}
  entries.append(entry)
# Per-entry distinctions within shared groups.
notes={
 'owner_graphs:owner-0135':'Device native_done dependency protects reuse of the canonical device descriptor/native workspace; NOT the CPU descriptor_upload mirror fence.',
 'owner_graphs:owner-0137':'CPU descriptor_upload event completion protects host DMA mirror reuse only; candidate host construction preserves this exact wait.',
 'owner_graphs:owner-0142':'Consuming GPU stream waits for descriptor upload publication; CPU construction changes do not remove it.',
 'owner_graphs:owner-0584':'Ready latch: payload visibility and full-rank preparation outcome before forward.',
 'owner_graphs:owner-0585':'Consumed latch: every reader has finished borrowing payloads before leader deletion.',
 'owner_graphs:owner-0586':'Ack latch: followers stop borrowing consumed marker before leader deletes/reuses it.',
 'owner_graphs:owner-0842':'KVWords.settle is broader than the local candidate producer graph; do not identify every status wait with packed decode transport.',
 'transport:T16':'Generic candidates_gather after local packet production; current candidate catalogue graphs stop before this collective.',
 'transport:T17':'Private N1 TargetGreedy packet path; N1 finish synchronization predominantly represents deferred GPU completion in the linked fbaa profile, not pure head CPU compute.',
 'transport:T21':'Engine gather sites cover several startup/admission branches. Do not assign all original locations a single steady frequency.',
 'transport:T22':'This is the 12-int64 model/method/KV/graph/precision/policy seal, not a claim that current finite benchmark invokes a label-scoring hot loop.',
 'transport:T23':'SharedMemoryGate minimum-room initial vote and conditional cache-growth vote. Full resident24 makes per-step growth participation unestablished.',
 'transport:T27':'Host request/State metadata digest; no direct GPU scalar readback in _state_digest. Prepared header reuse cannot cache changing output/generation/error digest.',
 'transport:T32':'Comm.barrier explicitly completes all prior device work; not semantically interchangeable with ordinary all_gather enqueue.',
 'transport:T35':'Successful ncclCommGetAsyncError means no observed asynchronous error; it is not a completion token.',
 'transport:T39':'Crosslinks owner-0135/0137/0142 and host construction candidate. Preserve device event and CPU mirror event as different contracts.',
 'transport:T41':'Inactive generic CUDA checkpoint ReadAhead; selected CPU PLE ReadAhead is active and separately covered by T55.',
 'transport:T43':'CUDA scheduler actor; no MLX scheduler relevance.',
 'transport:T45':'Finite callbacks execute on submitting caller threads; HTTP wfile write/flush is conditional. Shared bounded output credits can still indirectly stop actor decode when every client is slow.',
 'transport:T48':'Finite backend benchmark, no HTTP listener; service8000 not deployed by these results.',
 'transport:T52':'Native ctypes.CDLL calls release GIL. Python checks, argument creation and close-lock acquisition still occupy host execution; native duration does not imply GIL held.',
 'transport:T55':'Mixed activation entry: selected startup background PLE reads and active CPU table row read pool; generic GPU upload pool inactive.',
 'transport:T56':'Original inventory single-worker language is historical. Current28ac/f030 retains verified3 startup hash workers; no hashing re-audit.',
 'transport:T57':'Mixed cold materialization and terminal retire boundaries; no single per-round frequency applies.',
 'transport:T60':'Input prepare has nested per-stage phases; fbaa N1 diagnostic shows1092 prepare calls and728 exchange/retire calls for364 clipped bound ranges, not one exchange per round or an N24 multiplier. Historical ADD self-wait elision not selected.'}
for e in entries:
 if e['id'] in notes:e['entry_specific_reconciliation']=notes[e['id']]
# Pin additional directly relevant current sources for candidate / integration boundaries.
for p in ['src/tensorfold/families/qwen4_exp/cuda/history_owner.py','src/tensorfold/families/qwen4_exp/cuda/moe_region_catalogue.py','src/tensorfold/families/qwen4_exp/cuda/hc_moe_region.py','src/tensorfold/families/qwen4_exp/cuda/candidate_region_graphs.py']:
 try:bind({'path':p,'line':1})
 except subprocess.CalledProcessError:pass
manifest={k:{'path':str(R/v),'sha256':sha((R/v).read_bytes())} for k,v in E.items()}
required={f'owner_graphs:{n}' for n in owner['triaged_wait_sites']}|{f"transport:{x['id']}" for x in transport['sites']}
assert {e['id'] for e in entries}==required and len(entries)==len(required)==107
mastersha=sha((P/'drilldown-register.json').read_bytes())
result={
 'schema':'tensorfold.owner-transport-drilldown-reconciliation-v1',
 'status':'107 entries reconciled to source assessments/evidence or explicit remaining proof; not global optimization closure',
 'scope':{'owner_graphs_primary_waits':47,'transport_sites':60,'total':107,'source_only':True,'read_hop_limit':3,'no_gpu_build_remote_or_source_edits':True,'master_register_written':False},
 'instructions':{'direct_user_Lite_supersedes_repo_old_two_reviewer_rule':True,'sole_task_D13':'/root/cleanup_review','artifact_only_scope_no_new_coding_review_or_gpu_gate':True},
 'sources':{'current28ac':BASE,'candidate_f030':HOT,'source_worktree':str(W),'inventory_original_commit':OLD,'source_manifest':source_manifest,'inventory_hashes':{n:sha((P/n).read_bytes()) for n in ['owner-graphs-inventory.json','transport-inventory.json']},'master_register_sha256_before_after_unchanged':mastersha},
 'evidence_manifest':manifest,
 'interpretation':[
  'Raw master pending states are preserved as historical register data; later actual assessments are linked independently. A source-assessed required wait is not optimization selection or a measured impossibility proof.',
  '15 target-verify pieces and2 MTP pieces are different graph role topologies, not a measured15-to2 replacement.',
  'CPU blocking, GPU stream dependency, CPU metadata/serialization, cold construction, conditional lifecycle and vendor progress remain distinct.',
  'The fbaa N1 diagnostic is not current28ac/f030 timing and cannot be extrapolated toN24. Descriptor1882 EventSync1.5102s, mailbox728 exchange+retire1.00621s and other nested durations are not additive removable seconds.',
  'Current28ac/4d2b actual all-six HC graphs execute only N16/N24. Sharedpool192-key oracle is test-only; f030 hot16 current all-six result is Root-owned and not presumed here.',
  'No HTTP deployment, release, broad quality/stability, full-window payload or global optimality claim. Historical quality failures and all-N first-screen changes remain in original artifacts.'
 ],
 'existing_candidate_dispositions':{
  'T60_ADD_return_self_wait_elision':'First all-six70.6N24 vs86cb73.3; NOT SELECTED. Native protocol/CPU154/D13 evidence remains valid; no repeatable inferiority claim.',
  'FIFO_snapshot_elision':'First all-six100.8N24 vs146f/7c0a103.8; NOT SELECTED. Other tiers and quality/prefix/resource audit preserved in selection.json.',
  'descriptor_ring4':'First all-six62.9N24 vs named v2best65.3, N1 equal andN2/4/8/16 lower; NOT SELECTED. Does not prove all2/8slot alternatives impossible.',
  'frame_frontload_c2de_1d83':'Historical unselected negative screens; prepared full-frame DMA/math frontload is not the proposed host-row-construction-only change. Existing actual coverage and Root-reported comparison retained by owner assessment.',
  'packed_candidate_readback':'Packed incumbent retained; split alternative inconclusive. Status-first, original sampling math and positive completion remain required.',
  'candidate_HC_MoE_graphs':'Current retained28ac integrated actual-resource/output/counter evidence; one-screen differences alone are not repeatable superiority or additive attribution.',
  'onepool_hot16':'Test-only onepool actual oracle passed192keys384replays. f030 extends hot rows; current actual all-six run belongs Root. R1..3 full32 participation remains a separate blocker.'
 },
 'highest_remaining_nonoverlapping_candidate':{
  'name':'Native descriptor host-row construction elimination with unchanged DMA and per-layer scheduling',
  'source':'src/tensorfold/families/qwen4_exp/cuda/history_owner.py',
  'locations':{'storage_collection':[681,693],'cold_catalogue_and_retained_owners':[695,735],'hot_validation_and_spans':[743,778],'mirror_frontier_and_row_materialization':[779,799]},
  'crosslinks':['owner_graphs:owner-0135','owner_graphs:owner-0137','owner_graphs:owner-0142','transport:T39'],
  'actual_work':'Per segment/layer _state_storage + _storage_signature walks every retained target/MTP tensor. Per live row, torch.tensor(16words,int64) allocates CPU storage then copy_ fills the already-owned pinned mirror. This host work precedes dependent GPU submission.',
  'minimal_alternative':'Prepare/retain a CPU view or immutable per-State/layer/role row base and write exact dynamic fields into the existing pinned descriptor mirror, eliminating per-row temporary Tensor construction. Keep current mirror synchronization, GPU native_done/upload events, per-layer H2D placement and all compute/NCCL order unchanged.',
  'dynamic_fields':'slot_generation, history_generation, live lane pointer/length, row offset and placement owner; target versusMTP position pointer selection must remain correct. Position pointer value is aGPU authority, never replace it with a cached CPU position. Base scalar/layout and every Tensor binding need lifetime/invalidation proof.',
  'stronger_variant_blocker':'Hoisting full storage signatures once per serialized State/frame requires an explicit exclusion lease for resize/rebind/migration/reset and captured/raw readers; do not remove validation on assumed immutability.',
  'admissibility':'Need exact16word oracle across roles/rows/generations/layout/errors, inactive zero tails, int64 overflow behavior, mirror lifetime through DMA, failure prefix and primary-safe retirement. No new host pin policy or extra ring resources presumed.',
  'profile_and_P3_gate':'First attribute CPU construction/allocation separately from deferred event completion on current source; then same-tool allN end-to-end24>16>8>4>2>1, capacity/prefix/quality/resources and uncertainty. No gain amount or removable seconds is inferred.',
  'nonoverlap':'Does not frontload frame DMA/math, replace wait withring, modify hot16/DP4A/math/graph pool, alter State authority, change transport or weaken status/collectives.',
  'disposition':'genuine pending admissibility/profile; source candidate only, no implementation authorized in this assessment'
 },
 'remaining_distinct_gaps':[
  'Current descriptor host-construction attribution and cache/invalidation admissibility are pending; old frame/ring measurements do not answer this narrower proposal.',
  'N1 target15/MTP2 piece coarsening requires owner/State/collective cut proof; it is not already implemented by multirow HC/MoE or candidate catalogue.',
  'Status, candidate packet and input mirror fences require separate ownership domains; none is universally replaced by packed readback or graph capture.',
  'TP command/digest preparation and batch notifications remain source candidates requiring current exclusive-cost and adversarial protocol proof; T60/FIFO first screens already have disposition.',
  'NCCL rank skew/progress/bandwidth attribution needs approved representative all-rank capture. No owned spin loop was found and no upstream exhaustive transport audit is claimed.',
  'Conditional migration, retained errors, startup/close, inactive P2P and reference branches are source-assessed with required waits retained; performance value is unmeasured unless explicitly linked.',
  'f030 pool/hotrow actual selection and activeDP4A work remain separately owned; no overlapping implementation proposed.'
 ],
 'entries':entries,
 'validation':{'unique_ids':107,'expected_ids_exact':True,'all_entries_have_current_source_bindings':all(e['current_source_bindings'] for e in entries),'all_group_evidence_paths_exist':True,'group_counts':dict(collections.Counter(e['assessment_group'] for e in entries))}
}
assert sha((P/'drilldown-register.json').read_bytes())==mastersha
out=P/'owner-transport-drilldown-reconciliation.json'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'path':str(out),'sha256':sha(out.read_bytes()),'entries':len(entries),'groups':len(G),'source_files':len(source_manifest),'master_sha256_unchanged':mastersha},indent=2))
