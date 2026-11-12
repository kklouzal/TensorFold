"""Private native batch capture eligibility; no GPU execution evidence."""
import pytest

from tensorfold.families.qwen4_exp.cuda.engine import _native_gdn_capture_request


@pytest.mark.parametrize('method', ['none', 'mtp'])
def test_fixed_owned_region_can_prepare_native_gdn_capture(method):
    assert _native_gdn_capture_request(requested=True, batch_region=True, graphs=True, exact_owner=True,
                                      method=method, cached_experts=None, vision=False)


@pytest.mark.parametrize('field,value', [
    ('requested', False), ('batch_region', False), ('graphs', False), ('exact_owner', False),
    ('method', 'dspark'), ('method', 'dflash2'), ('cached_experts', 1), ('vision', True),
])
def test_other_regions_keep_their_existing_gdn_policy(field, value):
    request = dict(requested=True, batch_region=True, graphs=True, exact_owner=True, method='mtp',
                   cached_experts=None, vision=False)
    request[field] = value
    assert not _native_gdn_capture_request(**request)
