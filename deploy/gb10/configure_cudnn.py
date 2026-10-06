"""Expose every cuDNN SONAME spelling from the pinned wheel in a project directory.

cuDNN tries full-version names before major-version names. The inference base
ships full-version aliases for its reduced library set, while wheel files have
major names. Merely prepending the wheel directory mixes those implementations.
These build-time aliases select one complete provider without changing either
installed dependency tree.
"""

import argparse
from importlib.metadata import distribution
from pathlib import Path


def configure(output: Path):
    package = distribution("nvidia-cudnn-cu13")
    parts = package.version.split(".")
    if len(parts) != 4 or not all(part.isdecimal() for part in parts):
        raise ValueError("expected a pinned four-component cuDNN wheel version")
    library_dir = Path(package.locate_file("nvidia/cudnn/lib")).resolve(strict=True)
    if output.resolve().is_relative_to(library_dir):
        raise ValueError("cuDNN aliases must be outside the installed dependency directory")
    libraries = sorted(library_dir.glob(f"libcudnn*.so.{parts[0]}"))
    if not libraries or not any("engines_precompiled" in path.name for path in libraries):
        raise ValueError("cuDNN wheel is missing its complete precompiled engine library")
    output.mkdir(parents=True, exist_ok=True)
    for library in libraries:
        target = library.resolve(strict=True)
        stem = library.name.rsplit(".so.", 1)[0]
        for length in (1, 2, 3):
            alias = output / (stem + ".so." + ".".join(parts[:length]))
            if alias.is_symlink() and alias.resolve(strict=True) == target:
                continue
            if alias.exists() or alias.is_symlink():
                raise ValueError(f"refusing to replace an unexpected cuDNN alias: {alias}")
            alias.symlink_to(target)
    print(f"Selected {len(libraries)} cuDNN {package.version} wheel libraries in {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    configure(parser.parse_args().output)
