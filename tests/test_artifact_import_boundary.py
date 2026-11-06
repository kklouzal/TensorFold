"""Exercise neutral on-disk reads with family and accelerator imports forbidden."""
import json
from pathlib import Path
import subprocess
import sys

import tensorfold

from test_flashnext_gguf import fixture


def test_neutral_inventory_sidecar_and_unknown_family_need_no_backend(tmp_path):
    path = fixture(tmp_path / "other.gguf", [("w", (1,), 0, bytes(4))],
                   metadata={"general.architecture": "other-family"})
    sidecar = tmp_path / "prepared"
    sidecar.mkdir()
    (sidecar / "config.json").write_text(json.dumps({"gguf_entrypoint": str(path)}))
    package_file = Path(tensorfold.__file__).resolve()
    package_parent = package_file.parent.parent
    code = r'''
import importlib.abc
import sys
sys.path.insert(0, sys.argv[1])
blocked = ('torch', 'mlx', 'triton', 'tensorfold.cuda', 'tensorfold.families')
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if any(fullname == name or fullname.startswith(name + '.') for name in blocked):
            raise AssertionError('neutral artifact read imported ' + fullname)
sys.meta_path.insert(0, Block())
import tensorfold
from pathlib import Path
assert Path(tensorfold.__file__).resolve() == Path(sys.argv[3]).resolve()
from tensorfold.gguf_file import GGUFCheckpoint, resolve_gguf
from tensorfold.checkpoint_metadata import read_config
from tensorfold.tensor_file import read_metadata_json
entry = resolve_gguf(sys.argv[2])
assert read_metadata_json(sys.argv[2] + '/config.json')['gguf_entrypoint'] == str(entry)
checkpoint = GGUFCheckpoint.open(entry)
assert checkpoint.metadata['general.architecture'] == 'other-family'
assert checkpoint.tensors['w'].nbytes == 4
try:
    read_config(entry)
except ValueError as error:
    assert 'unsupported GGUF model architecture' in str(error)
else:
    raise AssertionError('generic inventory was advertised as supported inference')
assert not any(name == prefix or name.startswith(prefix + '.') for name in sys.modules for prefix in blocked)
'''
    result = subprocess.run([sys.executable, "-c", code, str(package_parent), str(sidecar), str(package_file)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
