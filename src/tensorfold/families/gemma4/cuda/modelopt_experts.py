"""Direct-scale ModelOpt experts with the existing Gemma BF16 GeGLU/ordered combine."""

from .nvfp4_experts import Experts


class ModelOptExperts(Experts):
    """Only a loader-validated DIRECT bank may enter this arithmetic adapter.

    Inherited forward preserves GeGLU activation selection and ascending-expert
    BF16 weighted accumulation; CT's default wrapper still rejects DIRECT banks.
    """

    direct_input = True
