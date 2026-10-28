"""Independent standalone DP4A format oracle; unmodified pinned GGML scalars.

Original GGML sources are readonly inputs. No owned CUDA decode/codebook derives
expected coefficients. This tool generates reference data, not runtime weights.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from tools.generate_gguf_mixed_fixture import SOURCES, REVISION, function

FORMATS = {18:("iq3_xxs",98),22:("iq2_s",82),20:("iq4_nl",18)}


def generate(quants, common, output):
    for path,name in [(quants,"ggml-quants.c"),(common,"ggml-common.h")]:
        if hashlib.sha256(path.read_bytes()).hexdigest()!=SOURCES[name]:
            raise ValueError("pinned scalar source mismatch: "+name)
    shim='''#include <assert.h>
#include <stdint.h>
#include <string.h>
#define GGML_COMMON_DECL_C
#define GGML_COMMON_IMPL_C
#include "ggml-common.h"
#define GGML_RESTRICT restrict
static float fp16_to_fp32(uint16_t bits) {
    _Float16 h; memcpy(&h,&bits,sizeof h); return (float)h;
}
#define GGML_FP16_TO_FP32(x) fp16_to_fp32(x)
'''
    source=quants.read_text()
    shim+='\n\n'.join(function(source,name)for name,_size in FORMATS.values())
    result={}
    with tempfile.TemporaryDirectory(prefix="tf-dp4a-scalar-")as folder:
        folder=Path(folder)
        shutil.copyfile(common,folder/'ggml-common.h')
        (folder/'oracle.c').write_text(shim)
        flags=['-std=gnu11','-O2','-ffp-contract=off','-fno-strict-aliasing','-shared','-fPIC']
        subprocess.run(['cc',*flags,str(folder/'oracle.c'),'-o',str(folder/'oracle.so')],check=True)
        library=ctypes.CDLL(str(folder/'oracle.so'))
        rng=np.random.default_rng(31415)
        for kind,(name,size)in FORMATS.items():
            count=64 if kind!=20 else 32
            raw=rng.integers(0,256,(count,size),dtype=np.uint8)
            raw[:,:2]=np.frombuffer(np.float16(.001).tobytes(),np.uint8)
            # Enumerate every codebook entry independently of owned tables.
            if kind==18:
                raw[:,2:66]=np.arange(count*64,dtype=np.uint64).astype(np.uint8).reshape(count,64)
            if kind==22:
                ids=(np.arange(count*32)%1024).reshape(count,32)
                raw[:,2:34]=(ids&255).astype(np.uint8)
                raw[:,66:74]=np.sum(((ids.reshape(count,8,4)>>8)&3)<<np.array([0,2,4,6]),axis=2).astype(np.uint8)
            if kind==20:
                low=np.arange(count*16,dtype=np.uint64).reshape(count,16)%16
                raw[:,2:]=((low&15)|((15-low)<<4)).astype(np.uint8)
            values=np.empty((count,256 if kind!=20 else 32),np.float32)
            fn=getattr(library,'dequantize_row_'+name)
            fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int64]
            fn(raw.ctypes.data,values.ctypes.data,values.size)
            result['raw_'+str(kind)]=raw
            result['bits_'+str(kind)]=values.view(np.uint32)
        result['provenance']=np.frombuffer(json.dumps({'revision':REVISION,'source_sha256':SOURCES,
            'scalar_functions':[name for name,_ in FORMATS.values()], 'flags':flags,
            'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}).encode(),np.uint8)
        np.savez_compressed(output,**result)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--quants',type=Path,required=True)
    p.add_argument('--common',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    generate(a.quants,a.common,a.output)
