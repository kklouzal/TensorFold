#!/usr/bin/env bash
# CPU processor/decoder source builds in the selected Torch image, sources at /media.
set -euo pipefail

tf_media=${TF_MEDIA_STATE:-/media}
tf_project=${TF_PROJECT_ROOT:-/src}
tf_media_prefix=${TF_MEDIA_PREFIX:-/opt/tensorfold-media}
tf_jobs=${MAX_JOBS:-4}
if ! [[ "$tf_jobs" =~ ^[1-4]$ ]]; then
    echo "media source MAX_JOBS must be 1..4" >&2
    exit 2
fi
export MAX_JOBS="$tf_jobs" OMP_NUM_THREADS="$tf_jobs"
mkdir -p "$tf_media/build" "$tf_media/logs" "$tf_media/receipts" "$tf_media/wheels"

tf_stage=${1:-}
if [[ "$tf_stage" =~ ^(torchvision|transformers|ffmpeg|pyav|dav1d)$ ]]; then
    python "$tf_project/deploy/v100/media_foundation.py" verify-source \
        --pins "$tf_project/deploy/v100/media-pins.json" --downloads "$tf_media/downloads" \
        --sources "$tf_media/sources" --source-name "$tf_stage" \
        --receipt "$tf_media/receipts/source-$tf_stage.json"
fi

case "$tf_stage" in
acquire)
    python "$tf_project/deploy/v100/media_foundation.py" fetch \
        --pins "$tf_project/deploy/v100/media-pins.json" --downloads "$tf_media/downloads"
    python "$tf_project/deploy/v100/media_foundation.py" extract \
        --pins "$tf_project/deploy/v100/media-pins.json" --downloads "$tf_media/downloads" --sources "$tf_media/sources"
    python "$tf_project/deploy/v100/media_foundation.py" fetch-native \
        --pins "$tf_project/deploy/v100/media-native-apt.json" --downloads "$tf_media/apt-debs"
    ;;
prepare)
    (cd "$tf_media/apt-debs" && sha256sum --strict --check "$tf_project/deploy/v100/media-native-apt.sha256")
    dpkg -i "$tf_media/apt-debs/"*.deb
    python -m pip download --no-deps --require-hashes \
        -r "$tf_project/deploy/v100/requirements-media-build-amd64.lock" --dest "$tf_media/support-wheels"
    python -m pip download --no-deps --require-hashes \
        -r "$tf_project/deploy/v100/requirements-media-runtime-amd64.lock" --dest "$tf_media/support-wheels"
    python -m pip install --no-index --find-links "$tf_media/support-wheels" --require-hashes \
        -r "$tf_project/deploy/v100/requirements-media-build-offline-amd64.lock" \
        -r "$tf_project/deploy/v100/requirements-media-runtime-offline-amd64.lock"
    python -m pip check
    ;;
torchvision)
    # The pinned official processor consumes CPU FP32 torchvision transforms.
    # No CUDA op, NVJPEG or alternate backend is selected by this profile.
    export CUDA_VISIBLE_DEVICES=-1 FORCE_CUDA=0 TORCHVISION_USE_NVJPEG=0
    export TORCHVISION_USE_PNG=1 TORCHVISION_USE_JPEG=1 TORCHVISION_USE_WEBP=1
    export BUILD_VERSION=0.29.1+tfcpu
    python -m pip wheel --no-build-isolation --no-deps "$tf_media/sources/torchvision" \
        --wheel-dir "$tf_media/wheels"
    ;;
transformers)
    python -m pip wheel --no-build-isolation --no-deps "$tf_media/sources/transformers" \
        --wheel-dir "$tf_media/wheels"
    ;;
ffmpeg)
    mkdir -p "$tf_media/build/ffmpeg"
    cd "$tf_media/build/ffmpeg"
    export PKG_CONFIG_PATH="$tf_media_prefix/dav1d-1.5.4/lib/pkgconfig"
    "$tf_media/sources/ffmpeg/configure" --prefix="$tf_media_prefix/ffmpeg-9.0.2" \
        --disable-doc --disable-programs --disable-static --enable-shared --enable-pic \
        --disable-autodetect --disable-hwaccels --disable-encoders --disable-muxers --disable-devices \
        --disable-cuda --disable-cuvid --disable-nvenc --disable-libnpp --disable-vulkan \
        --disable-gpl --disable-nonfree --enable-zlib --enable-runtime-cpudetect --enable-libdav1d \
        --extra-ldflags="-Wl,-rpath,$tf_media_prefix/dav1d-1.5.4/lib"
    make -j"$tf_jobs"
    make install
    mkdir -p "$tf_media_prefix/licenses/ffmpeg"
    cp "$tf_media/sources/ffmpeg/"COPYING* "$tf_media/sources/ffmpeg/LICENSE.md" \
        "$tf_media_prefix/licenses/ffmpeg/"
    ;;
dav1d)
    meson setup "$tf_media/build/dav1d" "$tf_media/sources/dav1d" \
        --prefix="$tf_media_prefix/dav1d-1.5.4" --libdir=lib --buildtype=release \
        --default-library=shared --wrap-mode=nodownload \
        -Denable_tools=false -Denable_examples=false -Denable_tests=false -Denable_docs=false \
        '-Dc_args=-march=x86-64 -mtune=generic'
    ninja -C "$tf_media/build/dav1d" -j"$tf_jobs"
    meson install -C "$tf_media/build/dav1d"
    mkdir -p "$tf_media_prefix/licenses/dav1d"
    cp "$tf_media/sources/dav1d/COPYING" "$tf_media/sources/dav1d/doc/PATENTS" \
        "$tf_media_prefix/licenses/dav1d/"
    ;;
pyav)
    export PKG_CONFIG_PATH="$tf_media_prefix/ffmpeg-9.0.2/lib/pkgconfig"
    export LD_LIBRARY_PATH="$tf_media_prefix/ffmpeg-9.0.2/lib:${LD_LIBRARY_PATH:-}"
    export CFLAGS='-O3 -DNDEBUG'
    export LDFLAGS="-Wl,-rpath,$tf_media_prefix/ffmpeg-9.0.2/lib"
    export PIP_NO_INDEX=1 PIP_FIND_LINKS="$tf_media/support-wheels"
    export PIP_CONSTRAINT="$tf_project/deploy/v100/media-build-constraints.txt"
    cd "$tf_media/sources/pyav"
    # Follow its source AGENTS build command; the supported PYAV_PIP override
    # keeps Make's dependency preparation inside the exact offline lock.
    make -j"$tf_jobs" PYAV_PYTHON=python PYAV_PIP='python -m pip' \
        CFLAGS="\"$CFLAGS\"" LDFLAGS="\"$LDFLAGS\""
    python -m pip wheel --no-build-isolation --no-deps . --wheel-dir "$tf_media/wheels"
    ;;
export)
    python "$tf_project/deploy/v100/media_foundation.py" capture-build \
        --pins "$tf_project/deploy/v100/media-pins.json" --state "$tf_media" --project "$tf_project"
    ;;
*)
    echo "media_build.sh requires acquire, prepare, torchvision, transformers, dav1d, ffmpeg, pyav or export" >&2
    exit 2
    ;;
esac

if [[ "$tf_stage" =~ ^(torchvision|transformers|ffmpeg|pyav|dav1d)$ ]]; then
    python "$tf_project/deploy/v100/media_foundation.py" record-source \
        --pins "$tf_project/deploy/v100/media-pins.json" --downloads "$tf_media/downloads" \
        --sources "$tf_media/sources" --source-name "$tf_stage" \
        --receipt "$tf_media/receipts/source-$tf_stage.json"
fi
