"""Prepare an explicit, unqualified SM70 NVFP4 source/numerical manifest."""

import argparse
import json
import os
from pathlib import Path
import tempfile

from tensorfold.families.qwen4_exp.cuda.sm70_weight_manifest import prepare
from tensorfold.cuda.sm70.nvfp4_layout import EXPERT_DEVICE, EXPERT_REFERENCE, LINEAR_REFERENCE, LINEAR_WMMA
from tensorfold.cuda.tensor_file import read_metadata_json
from tensorfold.cleanup import finish, raise_failures


def _same(path, body):
    existing = read_metadata_json(path)
    return json.dumps(existing, sort_keys=True, separators=(",", ":")) == json.dumps(body, sort_keys=True, separators=(",", ":"))


def publish(path, body):
    """Publish/reuse one immutable identity; report every failed retirement."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not _same(path, body):
            raise FileExistsError("the numerical manifest path already names a different immutable identity")
    else:
        temporary, primary, errors = None, None, []
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                stream.write(json.dumps(body, sort_keys=True, indent=2) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if not _same(path, body):
                    raise FileExistsError("a concurrent publisher selected another immutable numerical identity") from None
        except BaseException as error:
            primary = error
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except BaseException as error:
                    errors.append(error)
        raise_failures(primary, errors)
    directory = os.open(path.parent, os.O_RDONLY)
    finish((lambda: os.fsync(directory), lambda: os.close(directory)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--norm-offset", required=True, type=int, choices=(0, 1),
                        help="actual source norm storage:0 for gamma,1 for a centered trained parameter")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-kernel", required=True, choices=("reference", "wmma"),
                        help="explicit unqualified implementation; no automatic performance selection")
    parser.add_argument("--expert-kernel", required=True, choices=("reference", "device"),
                        help="CPU oracle grouping or maintained device CUB grouping/WMMA candidate")
    args = parser.parse_args()
    body = prepare(args.model, repository=args.repository, revision=args.revision, norm_offset=args.norm_offset,
                   linear_kernel_family=LINEAR_REFERENCE if args.linear_kernel == "reference" else LINEAR_WMMA,
                   expert_kernel_family=EXPERT_REFERENCE if args.expert_kernel == "reference" else EXPERT_DEVICE)
    publish(args.output, body)
    print(f"Prepared {args.output}; numerical/model/V100 qualification remains pending")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
