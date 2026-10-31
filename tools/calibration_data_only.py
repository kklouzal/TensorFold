"""Resolve/download a frozen Arrow wheel graph in an ephemeral CPU data container.

No host/SDK/producer installation. The separate tokenizer command must run in a
fresh container without this temporary Arrow prefix. Raw corpus files are data;
no benchmark Python source or test cases are executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

from tensorfold.offline_quantization.text_corpus import acquire_wikitext, _write_new


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    args = parser.parse_args()
    root = args.raw.resolve(strict=True)
    report = root / "pyarrow-resolver.json"
    wheels = root / "pyarrow-wheels"
    prefix = Path("/tmp/tensorfold-corpus-arrow")
    if report.exists() or wheels.exists() or prefix.exists():
        raise FileExistsError("data-only resolver output/prefix already exists")
    env = {**os.environ, "PIP_CACHE_DIR": "/tmp/tensorfold-corpus-pip-cache",
           "PIP_DISABLE_PIP_VERSION_CHECK": "1"}
    subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed",
                    "--only-binary=:all:", "--report", str(report), "pyarrow==26.0.0"], env=env, check=True)
    resolved = json.loads(report.read_text())
    downloads, installations, expected = [], [], {}
    for item in resolved["install"]:
        name, version = item["metadata"]["name"], item["metadata"]["version"]
        source = item["download_info"]
        digest = source["archive_info"]["hashes"]["sha256"]
        downloads.append(f"{name} @ {source['url']} --hash=sha256:{digest}\n")
        installations.append(f"{name}=={version} --hash=sha256:{digest}\n")
        expected[name.lower().replace("_", "-")] = {"version": version, "sha256": digest, "url": source["url"]}
    _write_new(root / "pyarrow-download.lock", "".join(downloads).encode())
    _write_new(root / "pyarrow-install.lock", "".join(installations).encode())
    wheels.mkdir()
    subprocess.run([sys.executable, "-m", "pip", "download", "--only-binary=:all:", "--require-hashes",
                    "--dest", str(wheels), "-r", str(root / "pyarrow-download.lock")], env=env, check=True)
    actual = {hashlib.sha256(path.read_bytes()).hexdigest(): path.name for path in wheels.iterdir()}
    if set(actual) != {item["sha256"] for item in expected.values()}:
        raise ValueError("downloaded wheel bytes differ from complete native resolver graph")
    subprocess.run([sys.executable, "-m", "pip", "install", "--ignore-installed", "--only-binary=:all:",
                    "--no-index", "--find-links", str(wheels), "--require-hashes", "--target", str(prefix),
                    "-r", str(root / "pyarrow-install.lock")], env=env, check=True)
    acquire_wikitext(root)
    # The child is CPU/data-only; no protected producer prefix is imported or mutated.
    reader_env = {**env, "PYTHONPATH": str(prefix) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    subprocess.run([sys.executable, "-m", "tensorfold.offline_quantization.text_corpus", "normalize-wikitext",
                    "--parquet", str(root / "wikitext-train.parquet"), "--output", str(root)],
                   env=reader_env, check=True)
    record = {"schema": 1, "scope": "ephemeral CPU Arrow data-only; no SDK/producer/host install",
              "python": sys.version, "machine": platform.machine(), "pip": md.version("pip"),
              "resolved": expected, "wheels": actual, "temporary_prefix": str(prefix),
              "installation_target_contract": "pip --target /tmp/tensorfold-corpus-arrow; no SDK/producer install requested", "tokenizer_stage": "MUST use separate container without Arrow prefix"}
    _write_new(root / "data-only-environment.json", (json.dumps(record, sort_keys=True) + "\n").encode())
    print(json.dumps(record, sort_keys=True))


if __name__ == "__main__":
    main()
