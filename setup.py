"""Build the journal-before-open POSIX owner against CPython's 3.11 Stable ABI."""
import hashlib
from pathlib import Path

from setuptools import Extension, setup

source = Path(__file__).resolve().parent / "src/tensorfold/_fd_owner.c"
source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

setup(
    ext_modules=[Extension(
        "tensorfold._fd_owner",
        sources=["src/tensorfold/_fd_owner.c"],
        define_macros=[("Py_LIMITED_API", "0x030B0000"),
                       ("TENSORFOLD_FD_OWNER_SOURCE_SHA256", '"' + source_sha256 + '"')],
        py_limited_api=True,
        extra_compile_args=["-std=c11", "-Wall", "-Wextra", "-Werror"],
    )],
    options={"bdist_wheel": {"py_limited_api": "cp311"}},
)
