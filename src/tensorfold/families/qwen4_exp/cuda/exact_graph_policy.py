"""Cold selection of paid exact captures; every other shape remains eager.

The evidence digest identifies the operator's version-bound all-rank cost
packet. It is provenance, not an assertion that a selected capture fits: live
per-key and settled admission remain authoritative.
"""

import re


POLICY = 'paid-subset-v1'


def parse_policy(value):
    if value is None:
        return None
    if (type(value) is not dict or set(value) != {'policy', 'keys', 'evidence_sha256'}
            or type(value['policy']) is not str or value['policy'] != POLICY
            or type(value['evidence_sha256']) is not str
            or re.fullmatch('[0-9a-f]{64}', value['evidence_sha256']) is None
            or type(value['keys']) not in (list, tuple) or not value['keys']):
        raise ValueError('exact graph subset requires its version, nonempty keys and cost-evidence SHA256')
    keys = []
    for key in value['keys']:
        if (type(key) not in (list, tuple) or len(key) != 3
                or type(key[0]) is not str or key[0] not in ('verify', 'mtp')
                or any(type(item) is not int or item <= 0 for item in key[1:])
                or key[1] < key[2] or key[1] % key[2]):
            raise ValueError('exact graph subset keys require role, total rows and equal-width participants')
        keys.append(tuple(key))
    if len(set(keys)) != len(keys):
        raise ValueError('exact graph subset keys must be unique')
    return {'policy': POLICY, 'keys': [list(key) for key in sorted(keys)], 'evidence_sha256': value['evidence_sha256']}
