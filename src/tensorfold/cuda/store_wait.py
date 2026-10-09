"""Classify only TCPStore's acknowledged wait timeout, never a store failure.

TCPStore.wait uses DistStoreError for structural responses too. The current
provider's timeout payload is checked against the exact duration and prefixed
keys; native exception arguments avoid invoking foreign formatting hooks.
"""
from __future__ import annotations


def timed_out(error, keys, timeout):
    from torch.distributed import DistStoreError

    if type(error) is not DistStoreError:
        return False
    arguments = BaseException.args.__get__(error)
    if len(arguments) != 1 or type(arguments[0]) is not str:
        return False
    milliseconds = (timeout.days * 86400 + timeout.seconds) * 1000 + timeout.microseconds // 1000
    expected = f"wait timeout after {milliseconds}ms, keys: " + ", ".join("/" + key for key in keys)
    return arguments[0] == expected
