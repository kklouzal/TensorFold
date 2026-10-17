#!/usr/bin/env python3
"""Owned CC/CXX aliases; install beside the identical foundation_tools module."""

import json
import os
import signal
import sys
from pathlib import Path

from foundation_tools import CONTRACT_ENV, compiler_command


def main() -> None:
    roles = {"tensorfold-cc": "cc", "tensorfold-cxx": "cxx"}
    role = roles.get(Path(sys.argv[0]).name)
    if role is None:
        raise ValueError("invoke the owned tensorfold-cc or tensorfold-cxx alias")
    path = os.environ.get(CONTRACT_ENV)
    if path is None or not Path(path).is_absolute():
        raise ValueError("foundation compiler requires its absolute sealed manifest path")
    with Path(path).open() as stream:
        contract = json.load(stream)["toolchain"]
    if contract["contract_path"] != path:
        raise ValueError("foundation compiler manifest path differs from its contract")
    command = compiler_command(contract, role=role, arguments=sys.argv[1:])
    # Triton/Torch use subprocess's default restore_signals=True. Python startup
    # ignores these signals again; restore that native-child contract before exec.
    for name in ("SIGPIPE", "SIGXFZ", "SIGXFSZ"):
        value = getattr(signal, name, None)
        if value is not None:
            signal.signal(value, signal.SIG_DFL)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
