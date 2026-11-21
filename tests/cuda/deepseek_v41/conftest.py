"""DeepSeek-V4.1 vendored-kernel tests.

Ported upstream tests keep their own ``CUDA required`` markers. The ``*_interpret`` tests run on CPU under
``TRITON_INTERPRET=1`` (set before the first Triton import, i.e. in the process environment).
"""

from v41_interpreter import install

install()
