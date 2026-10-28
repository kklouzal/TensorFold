"""Specified-Q8/int64 oracle from independent scalar-GGML decoded coefficients.

No CUDA unpacker or codebook is imported. Ratios below recover exact integer
units from the original GGML scalar coefficients and their binary16 base scale.
"""
import math
import numpy as np

MAX_NUMERATOR={18:7811008,22:5417312,20:516128}


def quantize(x,prefix=0):
    if x.dtype!=np.float16 or x.ndim!=2 or x.shape[1]not in (80,2560):
        raise ValueError('oracle requires finite-Half profile geometry')
    groups=(x.shape[1]+prefix+31)//32
    result=np.zeros((x.shape[0],groups,36),np.uint8)
    status=0
    padded=np.zeros((x.shape[0],groups*32),np.float32)
    padded[:,prefix:prefix+x.shape[1]]=x.astype(np.float32)
    for row in range(x.shape[0]):
        for group in range(groups):
            values=padded[row,group*32:(group+1)*32]
            if not np.isfinite(values).all():
                status|=1 if x.shape[1]==2560 else 4
                continue
            maximum=np.max(np.abs(values))
            scale=np.float32(maximum/np.float32(127.0)) if maximum else np.float32(0)
            result[row,group,:4]=np.frombuffer(scale.tobytes(),np.uint8)
            for lane,value in enumerate(values):
                ratio=np.float32(value/scale) if maximum else np.float32(0)
                magnitude=math.floor(abs(float(ratio))+.5)
                code=-magnitude if ratio<0 else magnitude
                assert -127<=code<=127
                result[row,group,4+lane]=code&255
    return result,status


def group_oracle(raw,decoded,q,kind):
    """One packed row. Exact int64 numerator plus specified FP32 scale term."""
    scale_units={18:4,22:8,20:1}[kind]
    groups=q.shape[0]
    integers=np.zeros(groups,np.int64)
    terms=np.zeros(groups,np.float32)
    block_bytes={18:98,22:82,20:18}[kind]
    for group in range(groups):
        block=group if kind==20 else group//8
        weight_scale=np.frombuffer(raw[block*block_bytes:block*block_bytes+2].tobytes(),'<f2')[0].astype(np.float32)
        values=decoded[group*32:(group+1)*32]
        if not np.isfinite(weight_scale):
            raise ValueError('nonfinite scalar weight scale')
        codes=q[group,4:].view(np.int8)
        if weight_scale:
            units=[int(round(float(value)/float(weight_scale)*scale_units))for value in values]
            assert all(float(value)==float(weight_scale)*unit/scale_units for value,unit in zip(values,units,strict=True))
            integer=sum(unit*int(code)for unit,code in zip(units,codes,strict=True))
        else:
            integer=0
        assert abs(integer)<=MAX_NUMERATOR[kind] and -(1<<31)<=integer<(1<<31)
        integers[group]=integer
        activation_scale=np.frombuffer(q[group,:4].tobytes(),'<f4')[0]
        factor=np.float32(weight_scale*activation_scale)
        scaled=np.float32(np.float32(integer)/np.float32(scale_units))
        terms[group]=np.float32(factor*scaled)
    return integers,terms


def warp_sum(terms):
    lanes=np.zeros(32,np.float32)
    for lane in range(32):
        for index in range(lane,len(terms),32):
            lanes[lane]=np.float32(lanes[lane]+terms[index])
    for offset in (16,8,4,2,1):
        before=lanes.copy()
        for lane in range(32):
            lanes[lane]=np.float32(before[lane]+before[lane+offset if lane+offset<32 else lane])
    return lanes[0]
