"""Serialized EXL3 loader/engine parity with compact host expert authorities.

Synthetic original trellises exercise real file reads and unmodified native
kernels, not a mocked model or trained-quality claim. Root controls GPU runs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gc
import importlib.util
from pathlib import Path
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen4_exp.cuda import exl3, forward as forward_module, weights  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine  # noqa: E402
from tensorfold.families.qwen4_exp.kv_formats import FORMATS  # noqa: E402
from tensorfold.families.qwen4_exp import ram_experts  # noqa: E402
from tensorfold.families.qwen4_exp.rope import RopeParameters  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402

RQC = next(item.name for item in FORMATS if item.codec >= 3)
# Pairwise compositions cover every cell count, cache format and RoPE policy.
REGIONS = ((1,"bf16",False),(1,"int8",True),(2,"bf16",True),
           (2,RQC,False),(3,"int8",False),(3,RQC,True))
PROMPT = [5,17,99,7,64,3,11,12,13,42,9,31,2,17,6,4,23]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = Path(__file__).parents[1] / "flashnext_exl3_fixture.py"
    spec = importlib.util.spec_from_file_location("flashnext_exl3_test_fixture",path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path_factory.mktemp("flashnext-exl3-complete")
    receipt = module.write_checkpoint(directory)
    return directory,receipt


def budget(directory,cells):
    entry = ram_experts.layout(directory,.001,mtp=True).entry_bytes
    return (cells*entry+13)/2**30


def policy(directory,yarn):
    import json
    return RopeParameters.from_config(json.loads((directory / "config.json").read_text()),
                                      yarn_factor=2 if yarn else None)


def same_state(a,b):
    assert a.pos == b.pos and a.mtp_len == b.mtp_len
    # Restore canonicalizes the physical recurrence bank to0. Its authoritative
    # per-layer value, not incidental parity or old scratch, is the contract.
    for index,(left,right) in enumerate(zip(a.cur,b.cur)):
        assert torch.equal(a.rec[left,index],b.rec[right,index])
    for x,y in ((a.conv,b.conv),(a.ple_tail,b.ple_tail)):
        assert torch.equal(x,y)
    for index,(left,right) in enumerate(zip(a.kc+[a.mtp_kc],b.kc+[b.mtp_kc])):
        assert left.format == right.format
        count = a.pos if index < len(a.kc) else a.mtp_len
        for name in ("k","v","ks","vs"):
            source,target = getattr(left,name),getattr(right,name)
            assert torch.equal(source[:count],target[:count]),name
    for left,right in zip(a.ikc,b.ikc):
        assert torch.equal(left[:a.pos],right[:b.pos])
    for left,right in zip(a.pooled,b.pooled):
        assert torch.equal(left[:a.pos//a.ratio],right[:b.pos//b.ratio])


def owned_device_storage_bytes(owner):
    """Independent object-graph inventory, not production device_tensors hooks."""
    pending,seen,storages = [owner],set(),{}
    while pending:
        value=pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        if isinstance(value,torch.Tensor):
            if value.device.type == "cuda":
                storage=value.untyped_storage()
                storages[storage.data_ptr()]=storage.nbytes()
        elif isinstance(value,dict):
            pending.extend(value.values())
        elif isinstance(value,(tuple,list)):
            pending.extend(value)
        elif type(value).__module__.startswith("tensorfold") and hasattr(value,"__dict__"):
            pending.extend(vars(value).values())
    return sum(storages.values())


@pytest.mark.parametrize("cells,dtype,yarn",REGIONS)
def test_file_loader_prefill_serial_mtp_prefix_growth_and_reset_match_resident(checkpoint,cells,dtype,yarn,monkeypatch):
    directory,_ = checkpoint
    rope = policy(directory,yarn)
    native = weights.load(directory,mtp=True,draft_vocab=None,rope=rope)
    # Offload startup must never call the resident expert-table uploader.
    def forbidden_resident_upload(*args,**kwargs):
        raise AssertionError("host loader attempted resident expert trellis upload")
    monkeypatch.setattr(exl3,"expert_table",forbidden_resident_upload)
    cached = weights.load(directory,mtp=True,draft_vocab=None,rope=rope,vram_experts=budget(directory,cells))
    cache = cached.meta["expert_cache"]
    try:
        assert cache.capacity == cells and cache._sealed and not cache._policy.resident
        assert cached.mtp is not None and cached.x3.moe.host_waves.device_bytes > 0
        assert len(cache._layers) == 3
        routing = {id(native):[],id(cached):[]}
        original = forward_module._exl3_moe
        def traced(m,w,b,rows):
            output = original(m,w,b,rows)
            routing[id(w)].append((rows,b.prefill,b.moe.pick[:rows].cpu().clone(),b.moe.wts[:rows].cpu().clone()))
            return output
        monkeypatch.setattr(forward_module,"_exl3_moe",traced)
        def paired(left,right):
            routing[id(native)].clear()
            routing[id(cached)].clear()
            first,second = left(),right()
            one,two = routing[id(native)],routing[id(cached)]
            assert len(one) == len(two)
            for x,y in zip(one,two):
                assert x[:2] == y[:2]
                assert torch.equal(x[2],y[2]) and torch.equal(x[3],y[3])
            return first,second
        a,b = [Engine(w,capacity=512,max_rows=8,prefill_rows=17,kv_dtype=dtype) for w in (native,cached)]
        first,second = paired(lambda:prefill(a,PROMPT,None),lambda:prefill(b,PROMPT,None))
        assert first == second
        same_state(a.st,b.st)
        first,second = paired(lambda:a.forward([11,23,31]).clone(),lambda:b.forward([11,23,31]).clone())
        assert torch.equal(first,second)
        same_state(a.st,b.st)
        for sampling in (None,Sampling(seed=31,top_k=20,top_p=.95)):
            expected,serial = paired(
                lambda:serial_decode(a,prefill(a,PROMPT,sampling,mtp=False),8,sampling).tokens,
                lambda:serial_decode(b,prefill(b,PROMPT,sampling,mtp=False),8,sampling).tokens)
            draft = mtp_decode(b,prefill(b,PROMPT,sampling),8,sampling,depth=2).tokens
            assert expected == serial == draft
        prefix = PROMPT + [7,8,9,10,11,12,13,14]
        paired(lambda:prefill(a,prefix,None,keep_at=16),lambda:prefill(b,prefix,None,keep_at=16))
        snapshots = (a.kept,b.kept)
        extended = prefix + [42,43,44,45]
        first,second = paired(lambda:prefill(a,extended,None,resume=snapshots[0]),
                              lambda:prefill(b,extended,None,resume=snapshots[1]))
        assert first == second
        same_state(a.st,b.st)
        for engine in (a,b):
            fork = engine.st.clone()
            fork.reset(engine.w)
            fork.copy_prefix(engine.st,engine.st.pos,engine.st.mtp_len)
            fork.restore(engine.st.snapshot())
            same_state(engine.st,fork)
            reference=forward_module.forward(engine.w,engine.st,engine.buf,[19]).clone()
            restored=forward_module.forward(engine.w,fork,engine.buf,[19]).clone()
            assert torch.equal(reference,restored)
        # Cache growth changes every pointer while retaining committed bytes.
        for engine in (a,b):
            engine.st.limit = 1024
            engine.st.resize(1024)
        same_state(a.st,b.st)
        first,second = paired(lambda:a.forward([77,88]).clone(),lambda:b.forward([77,88]).clone())
        assert torch.equal(first,second)
        for engine in (a,b):
            engine.reset()
        first,second = paired(lambda:prefill(a,PROMPT,None),lambda:prefill(b,PROMPT,None))
        assert first == second
        same_state(a.st,b.st)
        assert cache.misses > cells and cache.copied_bytes > 0
    finally:
        cache.close()


@pytest.mark.parametrize("cells",[1,2,3])
def test_cold_loader_authority_original_bytes_types_device_accounting_and_peak(checkpoint,cells,monkeypatch,record_property):
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack
    directory,receipt = checkpoint
    plan = ram_experts.layout(directory,budget(directory,cells),mtp=True)
    original_read = Pack.read
    inspected_pack = Pack(directory)
    spans = {}
    for name in inspected_pack.where:
        if ram_experts.exl3_expert_tensor(name) and name.endswith(".trellis"):
            file,begin,end,_,_ = inspected_pack.entry(name)
            spans.setdefault(file,[]).append((begin,end,name))
    # Retain every range owner: borrowed dtype/offset/strided aliases share the
    # same storage even when their data_ptr differs. Protect only original
    # trellis byte ranges, allowing unrelated scale metadata in a joint read.
    expert_sources = {}
    def inspected_read(self,file,begin,end):
        value = original_read(self,file,begin,end)
        protected = [(value.data_ptr()+max(begin,lo)-begin,value.data_ptr()+min(end,hi)-begin,name)
                     for lo,hi,name in spans.get(file,()) if begin < hi and lo < end]
        if protected:
            assert value.device.type == "cpu" and value.dtype == torch.uint8 and value.is_contiguous()
            key = value.untyped_storage().data_ptr()
            owner,regions = expert_sources.setdefault(key,(value,[]))
            del owner
            regions.extend(protected)
        return value
    original_to = torch.Tensor.to
    def inspected_to(value,*args,**kwargs):
        if value.device.type == "cpu" and value.numel() and value.untyped_storage().data_ptr() in expert_sources:
            destination = kwargs.get("device",args[0] if args else None)
            if isinstance(destination,torch.Tensor):
                destination = destination.device
            elif type(destination) is int:
                destination = torch.device("cuda",destination)
            if isinstance(destination,(str,torch.device)) and torch.device(destination).type == "cuda":
                low = value.data_ptr()
                high = low+value.element_size()*(1+sum((n-1)*stride for n,stride in zip(value.shape,value.stride())))
                if any(low < hi and lo < high for lo,hi,_ in expert_sources[value.untyped_storage().data_ptr()][1]):
                    raise AssertionError("cold host loader uploaded an original expert trellis alias")
        return original_to(value,*args,**kwargs)
    monkeypatch.setattr(Pack,"read",inspected_read)
    monkeypatch.setattr(torch.Tensor,"to",inspected_to)
    # EXL3 Scratch/users form cycles. Reclaim prior completed test models before
    # measuring this loader, rather than letting GC subtract them mid-delta.
    gc.collect()
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    w = weights.load(directory,mtp=True,draft_vocab=None,vram_experts=budget(directory,cells))
    cache = w.meta["expert_cache"]
    try:
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()-before
        record_property("fixture_sha256",receipt["safetensors_sha256"])
        record_property("loader_allocated_bytes",allocated)
        record_property("loader_peak_extra_bytes",torch.cuda.max_memory_allocated()-before)
        actual_owned=owned_device_storage_bytes(w)
        record_property("loader_unique_owned_device_bytes",actual_owned)
        assert actual_owned == w.nbytes()+w.inv_freq.untyped_storage().nbytes()
        assert allocated >= actual_owned
        assert cache.gpu_bytes == plan.gpu_bytes and cache.host_payload_bytes == plan.host_bytes
        assert cache.metadata_host_bytes == plan.metadata_host_bytes
        assert cache.control_device_bytes == plan.control_device_bytes
        assert cache.control_host_bytes == plan.control_host_bytes
        assert cache.pinned_bytes == plan.staging_bytes
        assert len(expert_sources) > 0 and not cache._policy.resident
        # Assert the observer catches a borrowed offset/dtype alias itself;
        # rejection happens before any CUDA transfer or allocation.
        source,regions = next(iter(expert_sources.values()))
        assert regions[0][1]-regions[0][0] >= 4
        offset = regions[0][0]-source.data_ptr()+2
        alias = source[offset:offset+2].view(torch.int16).reshape(1,1)
        assert alias.data_ptr() != source.data_ptr() and alias.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        with pytest.raises(AssertionError,match="trellis alias"):
            alias.to("cuda")
        pack = Pack(directory)
        bases = ["model.language_model.layers.0.mlp","model.language_model.layers.1.mlp","mtp.layers.0.mlp"]
        for layer,(name,owner) in enumerate(zip(bases,[*w.layers,w.mtp.layer])):
            authority = cache._layers[layer].authority
            assert authority.data.device.type == "cpu" and not authority.data.is_pinned()
            assert authority.data.dtype == torch.uint8
            assert all(t.device.type == "cpu" for t in authority.source)
            for expert in range(4):
                prefix = name + (f".experts.{expert}" if expert<3 else ".shared_expert")
                for projection,part in enumerate(("gate_proj","up_proj","down_proj")):
                    original = pack.get(prefix + "." + part + ".trellis")
                    assert torch.equal(authority.projection(expert,projection).view(torch.uint8),original.view(torch.uint8))
            logical = owner.moe.experts._tables
            assert logical.suh_g.dtype == torch.float16 and logical.svh_d.dtype == torch.float16
            assert logical.gate_k2.dtype == torch.int32 and logical.gate_ptr.dtype == torch.int64
        cached_bytes = w.nbytes()
        cache.close()
        assert cached_bytes-w.nbytes() == plan.gpu_bytes+plan.control_device_bytes
    finally:
        cache.close()


@pytest.mark.parametrize("streams,cells,dtype,yarn",[(2,1,"int8",False),(4,3,RQC,True)])
def test_public_engine_admission_scheduler_prefix_and_pool_match_resident(checkpoint,streams,cells,dtype,yarn,record_property):
    directory,_ = checkpoint
    common = dict(depth=2,confidence=0,draft_vocab=None,max_len=512,context_explicit=True,
                  prefetch=False,graphs=False,streams=streams,kv_dtype=dtype,yarn_factor=2 if yarn else None)
    native = FlashNextEngine(directory,**common)
    cached = None
    try:
        cached = FlashNextEngine(directory,**common,vram_experts=budget(directory,cells))
        admitted = cached.capacity_plan["vram_experts"]
        cache = cached.w.meta["expert_cache"]
        assert admitted["format"] == "exl3" and admitted["slots"] == cells
        assert admitted["host_bytes"] == cache.host_payload_bytes and admitted["gpu_bytes"] == cache.gpu_bytes
        assert admitted["control_device_bytes"] == cache.control_device_bytes
        record_property("capacity_plan",str(cached.capacity_plan))
        prompts = [[(i*17+index*31)%254 for i in range(280+index)] for index in range(streams)]
        def run(engine,prompt,draft=True):
            out=[]
            stats=engine.generate(prompt,5,None,lambda tokens:out.extend(tokens),draft=draft,stop_eos=False)
            return out,stats
        reference = [run(native,prompt)[0] for prompt in prompts]
        barrier = threading.Barrier(streams)
        def joined(prompt):
            barrier.wait(timeout=30)
            return run(cached,prompt)
        with ThreadPoolExecutor(max_workers=streams) as pool:
            futures=[pool.submit(joined,prompt) for prompt in prompts]
            actual=[future.result(timeout=120) for future in futures]
        assert [tokens for tokens,_ in actual] == reference
        repeated,stats = run(cached,prompts[0])
        assert repeated == reference[0] and stats["cached"] > 0
        serial,_ = run(cached,prompts[0],draft=False)
        assert serial == repeated
        assert cache.misses > cells
        # A capacity1 forward visiting3+ distinct IDs can legitimately miss
        # every time. Prove warm hits with an independently controlled reuse.
        experts=cached.w.layers[0].moe.experts
        size=int(cache._layers[0].authority.trellis_bytes[0])
        with experts.lease([0]):
            slot=cache._policy.resident[(0,0)]
            first=cache._cells[slot,:size].cpu().clone()
        hits,misses,copied=cache.hits,cache.misses,cache.copied_bytes
        with experts.lease([0]):
            slot=cache._policy.resident[(0,0)]
            assert torch.equal(cache._cells[slot,:size].cpu(),first)
        assert cache.hits == hits+1 and cache.misses == misses and cache.copied_bytes == copied
        cached.close()
        assert cached.scheduler is None and cache._pool is None
    finally:
        if cached is not None:
            cached.close()
        native.close()


def test_failed_real_file_read_closes_cache_and_preserves_primary_exception(checkpoint,monkeypatch):
    from tensorfold.cuda.exl3 import host_experts
    Exl3HostExpertCache = host_experts.Exl3HostExpertCache
    from tensorfold.families.qwen4_exp.cuda.exl3_pack import Pack
    directory,_ = checkpoint
    original_read,original_close = Pack.read,Exl3HostExpertCache.close
    file,begin,end,_,_ = Pack(directory).entry("model.language_model.layers.1.mlp.experts.0.gate_proj.trellis")
    closed,readers,failed_threads = [],[],[]
    original_ahead = host_experts.ReadAhead
    path = Path(__file__).parents[1]/"exl3_read_recorder.py"
    spec = importlib.util.spec_from_file_location("exl3_read_observer",path)
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    TrackedReadAhead = observer.read_ahead_recorder(original_ahead,readers)
    failure=OSError("injected original expert file read failure")
    def failed_read(self,source,low,high):
        if source == file and low < end and begin < high:
            failed_threads.append(threading.current_thread().name)
            raise failure
        return original_read(self,source,low,high)
    def tracked_close(self):
        original_close(self)
        closed.append(self)
        raise RuntimeError("injected cleanup diagnostic")
    monkeypatch.setattr(Pack,"read",failed_read)
    monkeypatch.setattr(host_experts,"ReadAhead",TrackedReadAhead)
    monkeypatch.setattr(Exl3HostExpertCache,"close",tracked_close)
    with pytest.raises(OSError,match="original expert file read failure") as caught:
        weights.load(directory,mtp=True,draft_vocab=None,vram_experts=budget(directory,1))
    assert caught.value is failure
    assert closed and all(cache._pool is None for cache in closed)
    assert any("cleanup also failed" in note for note in getattr(failure,"__notes__",[]))
    assert failed_threads and all(name.startswith("read-ahead") for name in failed_threads)
    assert len(readers) >= 2 and all(reader.observed_closed and reader.pool is None for reader in readers)
    assert all(future.done() for reader in readers for future in reader.observed_futures)
    assert all(not thread.is_alive() for reader in readers for thread in reader.observed_threads)
