"""Stdlib-only sealed BitBLAS CUDA library boundary; no compiler/runtime import.

Libraries and operand owners remain alive through positive stream completion and
all graph retirement. Loading is cold-only; launch checks the actual SDK runtime
and uses the caller's current CUDA stream, with no implicit allocation or sync.
"""
import ctypes
import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path


def file_identity(path):
    path = Path(path)
    if not stat.S_ISREG(path.stat().st_mode):
        raise ValueError('sealed artifacts must be regular files')
    with path.open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('sealed artifacts must be regular files')
        digest = hashlib.sha256()
        remaining = before.st_size
        while remaining:
            data = stream.read(min(remaining, 1 << 20))
            if not data:
                raise ValueError('artifact truncated during hashing')
            digest.update(data)
            remaining -= len(data)
        if stream.read(1):
            raise ValueError('artifact grew during hashing')
        after = os.fstat(stream.fileno())
    def stamp(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if stamp(before) != stamp(after) or stamp(before) != stamp(path.stat()):
        raise ValueError('artifact identity changed during hashing')
    return {'bytes': before.st_size, 'sha256': digest.hexdigest()}

def runtime_provider():
    path = Path('/usr/local/cuda/lib64/libcudart.so').resolve(strict=True)
    return {'path': str(path), **file_identity(path)}

@dataclass(frozen=True)
class PreparedLibrary:
    native: object
    cudart: object
    dimensions: tuple[int, int, int]
    runtime_provider: dict
    alignment_bytes: tuple[int, int, int]
    native_provider: dict | None = None


def _cuda_status(cudart):
    status = cudart.cudaGetLastError()
    if status:
        message = cudart.cudaGetErrorString(status)
        raise RuntimeError('prebuilt CUDA launch failed: ' + (message.decode() if message else str(status)))


def checked_runtime(library, expected):
    path = Path(expected['path']).resolve(strict=True)
    if str(path) != runtime_provider()['path']:
        raise ValueError('runtime provider differs from the protected CUDA SDK path')
    if file_identity(path) != {key: expected[key] for key in ('bytes', 'sha256')}:
        raise ValueError('loaded CUDA runtime provider content differs')
    functions = (library.cudaRuntimeGetVersion, library.cudaGetLastError, library.cudaGetErrorString)
    with open('/proc/self/maps', encoding='utf-8') as stream:
        mappings = [line.rstrip('\n').split(None, 5) for line in stream]
    for function in functions:
        address = ctypes.cast(function, ctypes.c_void_p).value
        found = False
        for fields in mappings:
            begin, end = (int(value, 16) for value in fields[0].split('-'))
            if begin <= address < end:
                if (len(fields) != 6 or 'x' not in fields[1] or fields[5].endswith(' (deleted)')
                        or Path(fields[5]).resolve(strict=True) != path):
                    raise ValueError('CUDA error function does not belong to the declared live SDK runtime')
                found = True
                break
        if not found:
            raise ValueError('CUDA runtime function mapping is unavailable')
    library.cudaRuntimeGetVersion.argtypes, library.cudaRuntimeGetVersion.restype = [ctypes.POINTER(ctypes.c_int)], ctypes.c_int
    version = ctypes.c_int()
    if library.cudaRuntimeGetVersion(ctypes.byref(version)) != 0 or version.value != 12090:
        raise ValueError('prebuilt pilot requires the protected CUDA12.9 runtime')
    library.cudaGetLastError.argtypes, library.cudaGetLastError.restype = [], ctypes.c_int
    library.cudaGetErrorString.argtypes, library.cudaGetErrorString.restype = [ctypes.c_int], ctypes.c_char_p
    return {'path': str(path), **file_identity(path), 'binding': 'actual-error-function-executable-VMA'}



def bind_native_functions(library, path):
    path = Path(path).resolve(strict=True)
    with open('/proc/self/maps', encoding='utf-8') as stream:
        mappings = [line.rstrip('\n').split(None, 5) for line in stream]
    for function in (library.init, library.call):
        address = ctypes.cast(function, ctypes.c_void_p).value
        found = False
        for fields in mappings:
            begin, end = (int(value, 16) for value in fields[0].split('-'))
            if begin <= address < end:
                if (len(fields) != 6 or 'x' not in fields[1] or fields[5].endswith(' (deleted)')
                        or Path(fields[5]).resolve(strict=True) != path):
                    raise ValueError('generated native entry point differs from its sealed artifact')
                found = True
                break
        if not found:
            raise ValueError('generated native entry point mapping is unavailable')
    return {'path': str(path), **file_identity(path), 'binding': 'actual-init-and-call-executable-VMAs'}

def load_verified_library(directory, record, dimensions, alignment_bytes):
    """Load producer-verified source/config artifacts under a caller-bound manifest.

    The caller validates its finite algorithm/geometry contract before this
    common boundary. Every declared file and the actual executable functions
    must still match the producer's sealed bytes. No source is compiled here.
    """
    directory = Path(directory).resolve(strict=True)
    if (record.get('runtime_policy') != 'shared-cuda12.9-checked'
            or 'libcudart.so.12' not in record.get('needed_libraries', [])):
        raise ValueError('static unchecked reference library cannot run through the checked ABI')
    for name,key in [('generated.cu','source'),('library.so','library'),('bounds.json','bounds')]:
        path=directory/name
        if path.is_symlink() or file_identity(path) != record[key]:
            raise ValueError('sealed library artifact content differs')
    library=ctypes.CDLL(str(directory/'library.so'))
    library.init.argtypes,library.init.restype=[],None
    library.call.argtypes,library.call.restype=[ctypes.c_void_p]*4,None
    native_provider=bind_native_functions(library,directory/'library.so')
    if {key:native_provider[key] for key in ('bytes','sha256')} != record['library']:
        raise ValueError('mapped generated native artifact content changed')
    provider=checked_runtime(library,record['runtime_provider'])
    library.init()
    _cuda_status(library)
    return PreparedLibrary(library,library,tuple(dimensions),provider,tuple(alignment_bytes),native_provider)


HC_SCHEMA='tensorfold.bitblas-hc-libraries-v1'
BITBLAS_COMMIT='9c9620678d2c292843a8d91506b573ff3d12eebc'
BITBLAS_WHEEL='5beaf38899671dc35da63ff31b8d7bea4d98a2c929a102fe40d2f3004cec391b'
PADDING_POLICY='default-volta-simt-n8-k128-v2'
HC_KERNELS={'hc-down':(32,320,10240,8),'hc-inject':(32,8,10240,16),'hc-up':(32,10240,320,8)}


def _same(left,right):
    return json.dumps(left,sort_keys=True,separators=(',',':')) == json.dumps(right,sort_keys=True,separators=(',',':'))


def hc_manifest(path, expected_sha256):
    """Validate the bounded producer manifest before loading any executable code."""
    path=Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError('HC manifest requires its exact persistent source identity')
    info=path.stat()
    if info.st_size > (1<<20):
        raise ValueError('HC library manifest exceeds its finite schema bound')
    if not stat.S_ISREG(info.st_mode):
        raise ValueError('HC manifest must be a regular file')
    with path.open('rb') as stream:
        before=os.fstat(stream.fileno())
        if before.st_size > (1<<20) or not stat.S_ISREG(before.st_mode):
            raise ValueError('HC library manifest exceeds its finite schema bound')
        data=stream.read(before.st_size+1)
        after=os.fstat(stream.fileno())
    def stamp(v):
        return (v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
    if (len(data)!=before.st_size or stamp(before)!=stamp(after) or stamp(before)!=stamp(path.stat())
            or hashlib.sha256(data).hexdigest()!=expected_sha256):
        raise ValueError('HC manifest requires its exact persistent source identity')
    value=json.loads(data)
    if (type(value) is not dict or value.get('schema') != HC_SCHEMA
            or value.get('policy') != 'fixed-m32-hc-only-f32-accumulation-v1'
            or value.get('project_python_commit') != BITBLAS_COMMIT
            or value.get('whole_wheel_sha256') != BITBLAS_WHEEL
            or set(value.get('kernels',{})) != set(HC_KERNELS)):
        raise ValueError('HC sealed library schema or kernel set differs')
    for role,(m,n,k,align) in HC_KERNELS.items():
        item=value['kernels'][role]
        if (path.parent/role).is_symlink():
            raise ValueError('HC artifact directory cannot be a symlink')
        if (item.get('directory') != role or item.get('dimensions') != [m,n,k]
                or item.get('alignment_bytes') != [align,align,4]
                or item.get('input_dtype') != 'float16' or item.get('output_dtype') != 'float32'
                or item.get('accumulation_dtype') != 'float32'):
            raise ValueError('HC sealed library geometry/precision differs')
        # The producer authenticated the finite emitter's source/index/ABI
        # contract. This expected manifest hash is the admission authority;
        # source/ELF/bounds checks below preserve it across loading.
        record=item['record']
        logical={'id':role+'-m32','role':role,'M':m,'N':4 if role=='hc-inject' else n,'K':k,
                 'A_dtype':'float16','W_dtype':'float16','out_dtype':'float32','accum_dtype':'float32',
                 'layout':'nt','propagate_a':False,'propagate_b':False,'with_bias':False}
        physical={**logical,'N':n}
        config={'logical':logical,'physical':physical,'padding_policy':PADDING_POLICY}
        config_sha=hashlib.sha256(json.dumps(config,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if (not _same(record.get('case'),logical) or not _same(record.get('physical_case'),physical)
                or record.get('config_sha256') != config_sha or record.get('padding_policy') != PADDING_POLICY
                or record.get('runtime_policy') != 'shared-cuda12.9-checked'
                or 'libcudart.so.12' not in record.get('needed_libraries',[])):
            raise ValueError('HC producer finite config/runtime differs')
        for filename,key in [('generated.cu','source'),('library.so','library'),('bounds.json','bounds')]:
            source=path.parent/role/filename
            if source.is_symlink() or file_identity(source) != record[key]:
                raise ValueError('HC producer artifact bytes changed')
    return value

def call_library(library, a, w, output, torch):
    """Borrow prepared code and operands through positive stream/graph retirement.

    The caller retains every object, including this library, through completion
    and any capture/replay lifetime. The helper neither allocates nor retires it.
    """
    if type(library) is not PreparedLibrary:
        raise TypeError('prepared static library binding required')
    rows, columns, reduction = library.dimensions
    if (type(a) is not torch.Tensor or type(w) is not torch.Tensor or type(output) is not torch.Tensor
            or a.dtype != torch.float16 or w.dtype != torch.float16 or output.dtype != torch.float32
            or a.shape != (rows, reduction) or w.shape != (columns, reduction)
            or output.shape != (rows, columns) or a.device.type != 'cuda' or a.device != w.device
            or a.device != output.device or any(not value.is_contiguous() or value.requires_grad or value.is_neg() or value.is_conj()
                                              for value in (a, w, output))):
        raise ValueError('prebuilt call requires exact contiguous Half/Half/Float CUDA matrices')
    if any(value.data_ptr() % alignment for value, alignment in zip((a, w, output), library.alignment_bytes)):
        raise ValueError('prebuilt operand base violates the reviewed emitter alignment')
    if output.is_inference() and not torch.is_inference_mode_enabled():
        raise RuntimeError('prebuilt inference output mutated outside inference mode')
    begin, end = output.data_ptr(), output.data_ptr() + output.numel() * output.element_size()
    if any(begin < value.data_ptr() + value.numel() * value.element_size() and value.data_ptr() < end for value in (a, w)):
        raise ValueError('prebuilt output aliases a read operand')
    with torch.cuda.device(a.device):
        stream = torch.cuda.current_stream(device=a.device).cuda_stream
        library.native.call(a.data_ptr(), w.data_ptr(), output.data_ptr(), stream)
        _cuda_status(library.cudart)
    torch.autograd.graph.increment_version(output)
    return output

