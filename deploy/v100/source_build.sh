#!/usr/bin/env bash
# Run only inside Dockerfile.source-builder with the state mounted at /work.
set -euo pipefail

tf_state=${TF_SOURCE_STATE:-/work}
tf_project=${TF_PROJECT_ROOT:-/src}
tf_prefix=${TF_SOURCE_PREFIX:-/opt/tensorfold-native}
tf_jobs=${MAX_JOBS:-4}
if ! [[ "$tf_jobs" =~ ^[1-4]$ ]]; then
    echo "source foundation MAX_JOBS must be 1..4" >&2
    exit 2
fi
export MAX_JOBS="$tf_jobs" CMAKE_BUILD_PARALLEL_LEVEL="$tf_jobs" OMP_NUM_THREADS="$tf_jobs"
export PATH="$tf_state/venv/bin:$PATH"
export UV_CACHE_DIR="$tf_state/lint-cache/uv" UV_PYTHON_INSTALL_DIR="$tf_state/lint-python"
mkdir -p "$tf_state/build" "$tf_state/logs" "$tf_state/receipts" "$tf_state/wheels"

case ${1:-} in
lint-prepare)
    if [[ ! -x "$tf_state/lint-venv/bin/python" ]]; then
        python3.12 -m venv "$tf_state/lint-venv"
    fi
    /opt/source-venv/bin/python -m pip --python "$tf_state/lint-venv" install --no-index \
        --find-links "$tf_state/release/lint-wheels" --require-hashes \
        -r "$tf_project/deploy/v100/requirements-source-lint-offline-amd64.lock"
    "$tf_state/lint-venv/bin/uv" python install 3.10.20
    ;;
acquire)
    python "$tf_project/deploy/v100/source_foundation.py" acquire-sources \
        --pins "$tf_project/deploy/v100/source-pins.json" --directory "$tf_state/sources" \
        --lint-tool "$tf_state/lint-venv/bin/spin"
    python "$tf_project/deploy/v100/source_foundation.py" capture-sources \
        --pins "$tf_project/deploy/v100/source-pins.json" --directory "$tf_state/sources" --receipts "$tf_state/receipts"
    ;;
prepare)
    if [[ ! -x "$tf_state/venv/bin/python" ]]; then
        python3.12 -m venv "$tf_state/venv"
    fi
    /opt/source-venv/bin/python -m pip --python "$tf_state/venv" install --no-index \
        --find-links "$tf_state/release/python-wheels" --require-hashes \
        -r "$tf_project/deploy/v100/requirements-source-build-offline-amd64.lock"
    python "$tf_project/deploy/v100/source_foundation.py" verify-source-heads \
        --pins "$tf_project/deploy/v100/source-pins.json" --directory "$tf_state/sources"
    python "$tf_project/deploy/v100/source_foundation.py" verify-native \
        --pins "$tf_project/deploy/v100/source-pins.json" --directory "$tf_state/release/build-debs"
    python "$tf_project/deploy/v100/source_foundation.py" capture-sources \
        --pins "$tf_project/deploy/v100/source-pins.json" --directory "$tf_state/sources" --receipts "$tf_state/receipts"
    cmake -S "$tf_state/sources/cccl" -B "$tf_state/build/cccl" -DCCCL_SKIP_BUILD_CHECKS=ON \
        -DCMAKE_INSTALL_PREFIX="$tf_prefix/cccl-3.5.0"
    cmake --install "$tf_state/build/cccl"
    mkdir -p "$tf_prefix/cutlass-4.8.1/include"
    cp -a "$tf_state/sources/cutlass/include/." "$tf_prefix/cutlass-4.8.1/include/"
    mkdir -p "$tf_prefix/licenses/cccl/cub" "$tf_prefix/licenses/cccl/thrust" \
        "$tf_prefix/licenses/cccl/libcudacxx" "$tf_prefix/licenses/nccl" "$tf_prefix/licenses/cutlass"
    cp "$tf_state/sources/cccl/LICENSE" "$tf_prefix/licenses/cccl/"
    cp "$tf_state/sources/cccl/cub/LICENSE.TXT" "$tf_prefix/licenses/cccl/cub/"
    cp "$tf_state/sources/cccl/thrust/LICENSE" "$tf_prefix/licenses/cccl/thrust/"
    cp "$tf_state/sources/cccl/libcudacxx/LICENSE.TXT" "$tf_prefix/licenses/cccl/libcudacxx/"
    cp "$tf_state/sources/nccl/LICENSE.txt" "$tf_prefix/licenses/nccl/"
    cp "$tf_state/sources/cutlass/LICENSE.txt" "$tf_prefix/licenses/cutlass/"
    ;;
nccl)
    python "$tf_project/deploy/v100/source_foundation.py" capture-sources \
        --pins "$tf_project/deploy/v100/source-pins.json" --directory "$tf_state/sources" --receipts "$tf_state/receipts"
    make -C "$tf_state/sources/nccl" -j"$tf_jobs" src.build \
        BUILDDIR="$tf_state/build/nccl" CUDA_HOME=/usr/local/cuda \
        NVCC_GENCODE='-gencode=arch=compute_70,code=sm_70'
    /usr/local/cuda/bin/cuobjdump --list-elf "$tf_state/build/nccl/lib/libnccl.so.2.32.3" \
        > "$tf_state/receipts/nccl-cubins.txt"
    ;;
probes)
    tf_nvcc=/usr/local/cuda/bin/nvcc
    tf_probe_source="$tf_project/deploy/v100/probes"
    "$tf_nvcc" -std=c++17 -O3 --fmad=false -arch=sm_70 \
        -I"$tf_prefix/cccl-3.5.0/include" "$tf_probe_source/cccl_primitives.cu" \
        -o "$tf_state/build/cccl-primitives" > "$tf_state/logs/cccl-primitives-build.log" 2>&1
    "$tf_nvcc" -std=c++17 -O3 --fmad=false --expt-relaxed-constexpr -arch=sm_70 \
        -I"$tf_prefix/cccl-3.5.0/include" -I"$tf_prefix/cutlass-4.8.1/include" \
        -I"$tf_prefix/nccl-2.32.3/include" "$tf_probe_source/libraries.cu" \
        -L"$tf_prefix/nccl-2.32.3/lib" -Xlinker -rpath -Xlinker "$tf_prefix/nccl-2.32.3/lib" \
        -lcublas -lcublasLt -lcudnn -lnccl -o "$tf_state/build/libraries-probe" \
        > "$tf_state/logs/libraries-probe-build.log" 2>&1
    "$tf_nvcc" -std=c++17 -O3 --fmad=false -arch=sm_70 "$tf_probe_source/reconvergence_6111901.cu" \
        -o "$tf_state/build/reconvergence-6111901" > "$tf_state/logs/reconvergence-build.log" 2>&1
    for tf_probe in cccl-primitives libraries-probe reconvergence-6111901; do
        "$tf_state/build/$tf_probe" --metadata-only > "$tf_state/receipts/$tf_probe-metadata.json"
        /usr/local/cuda/bin/cuobjdump --list-elf "$tf_state/build/$tf_probe" \
            > "$tf_state/receipts/$tf_probe-cubins.txt"
        /usr/local/cuda/bin/cuobjdump --dump-sass "$tf_state/build/$tf_probe" \
            > "$tf_state/receipts/$tf_probe-sass.txt"
        sha256sum "$tf_state/build/$tf_probe" > "$tf_state/receipts/$tf_probe-binary.sha256"
    done
    ;;
export)
    python "$tf_project/deploy/v100/source_foundation.py" capture-build \
        --pins "$tf_project/deploy/v100/source-pins.json" --state "$tf_state" --project "$tf_project" \
        --output "$tf_state/receipts/source-build.json"
    mkdir -p "$tf_state/release/build-inputs" "$tf_state/release/wheels" \
        "$tf_state/release/native" "$tf_state/release/probes" "$tf_state/release/python-wheels" "$tf_state/release/build-debs"
    cp -a "$tf_prefix/." "$tf_state/release/native/"
    cp "$tf_state/wheels/"torch-*.whl "$tf_state/release/wheels/"
    cp "$tf_state/receipts/source-build.json" "$tf_state/receipts/"*-source-inventory.json \
        "$tf_state/receipts/"*-metadata.json "$tf_state/receipts/"*-cubins.txt "$tf_state/receipts/"*-binary.sha256 \
        "$tf_state/receipts/builder-dpkg.txt" "$tf_state/release/build-inputs/"
    cp "$tf_state/sources/pytorch/build/CMakeCache.txt" "$tf_state/release/build-inputs/"
    for tf_probe in cccl-primitives libraries-probe reconvergence-6111901; do
        cp "$tf_state/build/$tf_probe" "$tf_state/release/probes/"
    done
    python -m pip download --no-deps --no-index --find-links "$tf_state/release/python-wheels" --require-hashes \
        -r "$tf_project/deploy/v100/requirements-source-runtime-offline-amd64.lock" \
        --dest "$tf_state/release/python-wheels"
    ;;
framework)
    python "$tf_project/deploy/v100/source_foundation.py" capture-sources \
        --pins "$tf_project/deploy/v100/source-pins.json" --directory "$tf_state/sources" --receipts "$tf_state/receipts"
    test -f "$tf_prefix/cccl-3.5.0/include/cub/version.cuh"
    test -f "$tf_state/build/nccl/lib/libnccl.so.2.32.3"
    mkdir -p "$tf_prefix/nccl-2.32.3/include" "$tf_prefix/nccl-2.32.3/lib" "$tf_state/wheels"
    cp -a "$tf_state/build/nccl/include/." "$tf_prefix/nccl-2.32.3/include/"
    cp -a "$tf_state/build/nccl/lib/"libnccl.so* "$tf_prefix/nccl-2.32.3/lib/"
    export USE_CUDA=1 USE_CUDNN=1 USE_DISTRIBUTED=1 USE_NCCL=1 USE_SYSTEM_NCCL=1 USE_STATIC_NCCL=0
    export USE_FLASH_ATTENTION=0 USE_XPU=0 USE_NVSHMEM=0 USE_CUDSS=0 USE_CUSPARSELT=0 USE_MSLK=0 BUILD_TEST=0
    export CUDA_HOME=/usr/local/cuda CUDNN_ROOT=/usr TORCH_CUDA_ARCH_LIST=7.0
    export NCCL_INCLUDE_DIR="$tf_prefix/nccl-2.32.3/include" NCCL_LIB_DIR="$tf_prefix/nccl-2.32.3/lib"
    export NCCL_VERSION=2.32.3
    export PYTORCH_BUILD_VERSION=2.14.1+tfcu129sm70 PYTORCH_BUILD_NUMBER=1 PYTORCH_EXTRA_INSTALL_REQUIREMENTS=""
    # Source checkouts belong to the host UID. Trust only these pinned build
    # inputs in this disposable container, never every repository on the host.
    git config --global --fixed-value --get-all safe.directory "$tf_state/sources/pytorch" >/dev/null || \
        git config --global --add safe.directory "$tf_state/sources/pytorch"
    git -C "$tf_state/sources/pytorch" submodule foreach --recursive \
        'git config --global --fixed-value --get-all safe.directory "$PWD" >/dev/null || git config --global --add safe.directory "$PWD"'
    export SOURCE_DATE_EPOCH
    SOURCE_DATE_EPOCH=$(git -C "$tf_state/sources/pytorch" show -s --format=%ct HEAD)
    python - "$tf_state/receipts/framework-build-environment.json" <<'PY'
import json
import os
import sys
from pathlib import Path

names = (
    "CC", "CXX", "CUDAHOSTCXX", "NVCC_CCBIN", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS",
    "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "CUDA_HOME", "CUDNN_ROOT", "TORCH_CUDA_ARCH_LIST",
    "NCCL_INCLUDE_DIR", "NCCL_LIB_DIR", "NCCL_VERSION", "MAX_JOBS", "CMAKE_BUILD_PARALLEL_LEVEL",
    "OMP_NUM_THREADS", "SOURCE_DATE_EPOCH", "PYTORCH_BUILD_VERSION", "PYTORCH_BUILD_NUMBER",
    "PYTORCH_EXTRA_INSTALL_REQUIREMENTS", "USE_CUDA", "USE_CUDNN", "USE_DISTRIBUTED", "USE_NCCL",
    "USE_SYSTEM_NCCL", "USE_STATIC_NCCL", "USE_FLASH_ATTENTION", "USE_XPU", "USE_NVSHMEM", "USE_CUDSS",
    "USE_CUSPARSELT", "USE_MSLK", "BUILD_TEST",
)
Path(sys.argv[1]).write_text(json.dumps({name: os.environ.get(name) for name in names}, sort_keys=True) + "\n")
PY
    cd "$tf_state/sources/pytorch"
    python -m pip --log "$tf_state/logs/pytorch-pip-details.log" wheel -v --no-deps --no-build-isolation --wheel-dir "$tf_state/wheels" \
        --config-settings="cmake.args=-UNCCL_VERSION_DEFINED" \
        --config-settings="cmake.define.CUB_INCLUDE_DIR=$tf_prefix/cccl-3.5.0/include" \
        --config-settings="cmake.define.CMAKE_CUDA_FLAGS=-I$tf_prefix/cccl-3.5.0/include" \
        --config-settings="cmake.define.CMAKE_CUDA_ARCHITECTURES=70" \
        --config-settings="cmake.define.CMAKE_CXX_FLAGS=-I$tf_prefix/cccl-3.5.0/include" \
        --config-settings="cmake.define.NCCL_ROOT=$tf_prefix/nccl-2.32.3" \
        --config-settings="cmake.define.NCCL_VERSION=2.32.3" \
        --config-settings="cmake.define.NCCL_INCLUDE_DIRS=$tf_prefix/nccl-2.32.3/include" \
        --config-settings="cmake.define.NCCL_LIBRARIES=$tf_prefix/nccl-2.32.3/lib/libnccl.so" \
        --config-settings="cmake.define.CMAKE_REQUIRED_INCLUDES=/usr/local/cuda/include" \
        --config-settings="cmake.define.CMAKE_INSTALL_RPATH=$tf_prefix/nccl-2.32.3/lib;/usr/local/cuda/lib64" \
        --config-settings="cmake.define.BLAS=OpenBLAS" .
    ;;
*)
    echo "source_build.sh requires lint-prepare, acquire, prepare, nccl, probes, framework or export" >&2
    exit 2
    ;;
esac
