"""Generate/check the private native row ABI from its owned Python schema."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorfold.cuda.sm70.paged_attention_descriptor import render_header  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="fail if the maintained generated header is stale")
    args = parser.parse_args()
    target = ROOT / "src/tensorfold/cuda/sm70/paged_attention_descriptor.cuh"
    expected = render_header()
    if args.check:
        if not target.is_file() or target.read_text() != expected:
            parser.exit(1, "native paged descriptor header is stale; run this generator without --check\n")
    else:
        target.write_text(expected)


if __name__ == "__main__":
    main()
