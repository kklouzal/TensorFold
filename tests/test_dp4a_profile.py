"""Independent specified Q8/int64 controls; no current-profile tolerance changes."""
from pathlib import Path
from types import SimpleNamespace as NS
import json

import numpy as np
import pytest

from dp4a_profile_oracle import quantize,group_oracle,warp_sum,MAX_NUMERATOR


DATA=Path(__file__).parent/'fixtures/dp4a_ggml_hot.npz'


def test_pinned_original_scalar_fixture_sources_and_shapes():
    with np.load(DATA)as data:
        provenance=json.loads(data['provenance'].tobytes())
        assert provenance['revision']=='b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9'
        assert provenance['source_sha256']['ggml-quants.c']=='5574a2dccf7c07e75b143733e04a5412d3d8c819e7945f5217a7b83a2b2ff8ab'
        assert provenance['source_sha256']['ggml-common.h']=='0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c'
        assert data['raw_18'].shape==(64,98) and data['raw_22'].shape==(64,82) and data['raw_20'].shape==(32,18)


def test_quantizer_ties_away_zeros_and_exact_36byte_contract():
    x=np.zeros((1,2560),np.float16)
    x[0,:8]=[127,.5,-.5,1.5,-1.5,0.,-0.,np.nextafter(np.float16(0),np.float16(1))]
    q,status=quantize(x)
    assert q.shape==(1,80,36) and status==0
    assert np.frombuffer(q[0,0,:4].tobytes(),'<f4')[0]==1
    np.testing.assert_array_equal(q[0,0,4:11].view(np.int8),[127,1,-1,2,-2,0,0])
    assert not np.any(q[0,1:])


@pytest.mark.parametrize('value',[np.float16(65504),np.nextafter(np.float16(0),np.float16(1))])
def test_finite_half_extremes_keep_nonzero_fp32_scale_and_codes(value):
    x=np.full((1,80),value,np.float16)
    q,status=quantize(x,16)
    assert status==0
    scales=q[:,:,:4].copy().view('<f4').reshape(1,3)
    assert np.all(scales>0) and np.isfinite(scales).all()
    assert np.all(np.abs(q[:,:,4:].view(np.int8))<=127)


@pytest.mark.parametrize('bad',[np.nan,np.inf,-np.inf])
def test_nonfinite_is_refused_before_any_integer_cast(bad):
    x=np.zeros((1,80),np.float16)
    x[0,0]=bad
    q,status=quantize(x,16)
    assert status==4 and not q[0,0].any()


@pytest.mark.parametrize('offset',[0,16])
def test_crop_alignment_retains_exact_three_32value_blocks(offset):
    x=np.arange(80,dtype=np.float16).reshape(1,80)
    q,status=quantize(x,offset)
    assert status==0 and q.shape==(1,3,36)
    if offset:
        assert not q[0,0,4:20].any()
    else:
        assert not q[0,2,20:].any()


@pytest.mark.parametrize('kind',[18,22,20])
def test_original_scalar_coefficients_derive_independent_int64_oracle(kind):
    with np.load(DATA)as data:
        raw=np.tile(data['raw_'+str(kind)][0],10 if kind!=20 else 3)
        decoded=np.tile(data['bits_'+str(kind)][0].view(np.float32),10 if kind!=20 else 3)
    x=np.linspace(-.1,.1,2560 if kind!=20 else 80,dtype=np.float16).reshape(1,-1)
    q,status=quantize(x,0 if kind!=20 else 16)
    integer,terms=group_oracle(raw,decoded,q[0],kind)
    assert status==0 and np.isfinite(terms).all() and np.isfinite(warp_sum(terms))
    assert max(abs(int(value))for value in integer)<=MAX_NUMERATOR[kind]<(1<<24)


def test_shared_mtp_and_other_formats_are_explicit_exact_fallbacks():
    from tensorfold.cuda.sm70.dp4a_experts import eligible
    bank=NS(count=513,width=80,dims=2560,gate=NS(ggml_type=18,column_offset=0,n=40960,k=2560),
            up=NS(ggml_type=22,column_offset=0,n=40960,k=2560),down=NS(ggml_type=20,column_offset=16,n=1310720,k=80),limit=0.,swiglu=True,shared=[None]*3)
    assert eligible(bank,'target') and not eligible(bank,'mtp')
    bank.gate.ggml_type=21
    assert not eligible(bank,'target')
