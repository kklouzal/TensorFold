"""No-GPU build-backend API admission; does not build a wheel."""
import importlib.metadata as md
import json

import setuptools.build_meta
import setuptools_scm

if not callable(setuptools.build_meta.build_wheel) or not callable(setuptools_scm.get_version):
    raise RuntimeError("required public Python build APIs are unavailable")
print(json.dumps({"scope": "CPU build backend imports only; no source build",
                  "setuptools": md.version("setuptools"), "wheel": md.version("wheel"),
                  "setuptools-scm": md.version("setuptools-scm")}))
