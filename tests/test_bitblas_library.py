"""Source/manifest boundary checks need neither Torch nor a device."""
import hashlib
import json

import pytest

from tensorfold.cuda.sm70 import bitblas_library as library
from tools import v100_bitblas_prepare as producer


def manifest(tmp_path):
    kernels={}
    for role,(m,n,k,align) in library.HC_KERNELS.items():
        case=next(c for c in producer.cases() if c['role']==role and c['M']==32)
        directory=tmp_path/role;directory.mkdir()
        record={'case':case,'physical_case':producer.physical_case(case),'config_sha256':producer.config_identity(case),
                'padding_policy':producer.PADDING_POLICY,'runtime_policy':'shared-cuda12.9-checked',
                'needed_libraries':['libcudart.so.12']}
        for name,key in [('generated.cu','source'),('library.so','library'),('bounds.json','bounds')]:
            (directory/name).write_bytes(b'boundary fixture, never executable')
            record[key]=library.file_identity(directory/name)
        kernels[role]={'directory':role,'dimensions':[m,n,k],'alignment_bytes':[align,align,4],
                       'input_dtype':'float16','output_dtype':'float32','accumulation_dtype':'float32','record':record}
    value={'schema':library.HC_SCHEMA,'policy':'fixed-m32-hc-only-f32-accumulation-v1',
           'project_python_commit':library.BITBLAS_COMMIT,'whole_wheel_sha256':library.BITBLAS_WHEEL,'kernels':kernels}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(value))
    return path,value


def seal(path,value):
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_hc_manifest_binds_finite_config_and_actual_source_elf_bytes(tmp_path):
    path,value=manifest(tmp_path)
    sha=seal(path,value)
    assert library.hc_manifest(path,sha)==value
    (tmp_path/'hc-up/library.so').write_bytes(b'changed')
    with pytest.raises(ValueError,match='bytes changed'):
        library.hc_manifest(path,sha)


@pytest.mark.parametrize('field,value',[('dimensions',[32,4,10240]),('alignment_bytes',[8,8,4]),
                                        ('input_dtype','float32'),('directory','../elsewhere')])
def test_unreviewed_inject_geometry_is_rejected_before_loading(tmp_path,field,value):
    path,record=manifest(tmp_path)
    record['kernels']['hc-inject'][field]=value
    with pytest.raises(ValueError,match='geometry/precision'):
        library.hc_manifest(path,seal(path,record))


def test_static_runtime_or_stale_config_cannot_hide_behind_correct_outer_dimensions(tmp_path):
    path,record=manifest(tmp_path)
    record['kernels']['hc-down']['record']['runtime_policy']='static-reference'
    with pytest.raises(ValueError,match='finite config/runtime'):
        library.hc_manifest(path,seal(path,record))


def test_changed_or_oversized_manifest_fails_before_reading_or_native_loading(tmp_path,monkeypatch):
    path,_value=manifest(tmp_path)
    with pytest.raises(ValueError,match='source identity'):
        library.hc_manifest(path,'0'*64)
    path.write_bytes(b' ' *((1<<20)+1))
    monkeypatch.setattr(library,'file_identity',lambda *a:pytest.fail('hashed oversized manifest'))
    with pytest.raises(ValueError,match='finite schema bound'):
        library.hc_manifest(path,'0'*64)
