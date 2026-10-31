"""Clones share one attention list: a grow maps more of the same arenas for every clone (no row moves); a resume
stays exact."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kv_arena import owned  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import clone_state, draft_decode, prefill  # noqa: E402

from qwen_engine_checks import same_state  # noqa: E402
from test_qwen27_prefill import _model, _prompt  # noqa: E402

ATT = 1                                                  # the tiny model's attention layer


@pytest.mark.parametrize("grow_by", ["prompt", "decode"])
def test_a_grow_reaches_every_clone_in_place(grow_by):
    w = _model()
    st, first = prefill(w, _prompt(1000, seed=9), None)
    kept = clone_state(st)
    rows, ptr = st.kv[ATT][0].shape[0], st.kv[ATT][0].data_ptr()      # the first granule's rows
    assert rows >= 1000 and kept.kv is st.kv and owned(st.kv[ATT][0], st.kv_owner) is not None
    prompt = _prompt(rows + 300, seed=9)
    assert prompt[:1000] == _prompt(1000, seed=9)
    if grow_by == "prompt":
        prefill(w, prompt[:rows + 100], None, state=st)
    else:
        draft_decode(w, st, prompt[:1000], first, rows - 1000 + 100, None, None, stop_eos=False)
    assert kept.kv[ATT][0].shape[0] > rows and kept.kv[ATT][0].data_ptr() == ptr      # one arena for every clone
    resumed, first_resumed = prefill(w, prompt[:1200], None, state=kept)
    fresh, first_fresh = prefill(w, prompt[:1200], None)
    same_state(fresh, resumed)
    assert first_fresh == first_resumed
