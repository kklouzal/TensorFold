# syntax=docker/dockerfile:1.7
# Native Linux ARM64/AMD64. Build contexts contain actual per-target generated
# artifacts; missing native artifacts fail rather than selecting historical pins.
ARG BASE_IMAGE=nvcr.io/nvidia/cuda:13.4.3-cudnn-devel-ubuntu24.04@sha256:f2d23b75391766a02175908d3fa1c2afee9849aa13d09326d71649de2b33cfe9
FROM ${BASE_IMAGE} AS foundation
ARG BASE_IMAGE
ARG DEBIAN_FRONTEND=noninteractive
# SHA256 of the selected pins' interpreter archive (Ubuntu 24.04 has no CPython 3.15).
# It is checked before extraction; select_pins binds it to the pins.
ARG PYTHON_SHA256
COPY --from=foundation_artifacts / /opt/tensorfold-foundation-artifacts/
COPY deploy/gb10/foundation-sdk-pattern.txt deploy/gb10/foundation_sdk.sh /opt/tensorfold-pins/
RUN sh /opt/tensorfold-pins/foundation_sdk.sh capture /opt/tensorfold-pins /opt/tensorfold-pins/foundation-sdk-pattern.txt \
    && apt-get update \
    && apt-get full-upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends ca-certificates ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && sh /opt/tensorfold-pins/foundation_sdk.sh check /opt/tensorfold-pins /opt/tensorfold-pins/foundation-sdk-pattern.txt \
    && set -- /opt/tensorfold-foundation-artifacts/cpython-*-unknown-linux-gnu-install_only_stripped.tar.gz \
    && test "$#" -eq 1 && test -n "${PYTHON_SHA256}" \
    && printf '%s  %s\n' "${PYTHON_SHA256}" "$1" | sha256sum --check --strict \
    && mkdir /opt/python3.15 \
    && tar -xzf "$1" -C /opt/python3.15 --strip-components=1 --no-same-owner \
    && chmod -R go-w /opt/python3.15 \
    && /opt/python3.15/bin/python3.15 -m venv /opt/tensorfold-foundation \
    && /opt/python3.15/bin/python3.15 -m venv --without-pip /opt/tensorfold-venv
ENV PATH=/opt/tensorfold-venv/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONNOUSERSITE=1 \
    PYTHONDONTWRITEBYTECODE=1

# Discarded stage: owned wheels for selected versions without a usable target wheel,
# built from their exact sdists with only the generated build lock's wheels.
FROM foundation AS owned-builds
ARG TARGETARCH
ARG BASE_IMAGE
ARG PYTHON_SHA256
ARG MAX_JOBS=8
COPY --from=foundation_pins / /opt/tensorfold-pin-inputs/
COPY deploy/gb10/platform_pins.py deploy/gb10/select_pins.py deploy/gb10/foundation_policy.py \
     deploy/gb10/owned_builds.py /opt/tensorfold-owned-build/
RUN /opt/python3.15/bin/python3.15 /opt/tensorfold-owned-build/select_pins.py --architecture ${TARGETARCH} \
      --directory /opt/tensorfold-pin-inputs --base-image ${BASE_IMAGE} --interpreter-sha256 ${PYTHON_SHA256} \
    && /opt/python3.15/bin/python3.15 -m venv /opt/tensorfold-build \
    && /opt/tensorfold-build/bin/python -m pip install --no-cache-dir --require-hashes --no-deps \
         -r /opt/tensorfold-pin-inputs/requirements-build.lock \
    && /opt/tensorfold-build/bin/python -m pip check \
    && CMAKE_BUILD_PARALLEL_LEVEL=${MAX_JOBS} /opt/tensorfold-build/bin/python /opt/tensorfold-owned-build/owned_builds.py \
         --manifest /opt/tensorfold-pin-inputs/foundation-manifest.json --output /opt/tensorfold-owned-wheels

FROM foundation AS dependencies
ARG TARGETARCH
ARG BASE_IMAGE
ARG PYTHON_SHA256
COPY --from=foundation_pins / /opt/tensorfold-pin-inputs/
COPY --from=owned-builds /opt/tensorfold-owned-wheels/ /opt/tensorfold-owned-wheels/
COPY deploy/gb10/platform_pins.py deploy/gb10/select_pins.py deploy/gb10/foundation_policy.py \
     deploy/gb10/foundation_guard.py deploy/gb10/foundation_bootstrap.py deploy/gb10/foundation_runtime.py /opt/tensorfold-pins/
COPY src/tensorfold/cuda/foundation_tools.py /opt/tensorfold-pins/foundation_tools.py
COPY src/tensorfold/cuda/foundation_tools.py /opt/tensorfold-foundation-tools/foundation_tools.py
COPY --chmod=755 deploy/gb10/foundation_compiler.py /opt/tensorfold-foundation-tools/tensorfold-cc
COPY --chmod=755 deploy/gb10/foundation_compiler.py /opt/tensorfold-foundation-tools/tensorfold-cxx
RUN /opt/python3.15/bin/python3.15 /opt/tensorfold-pins/select_pins.py --architecture ${TARGETARCH} --directory /opt/tensorfold-pin-inputs \
      --base-image ${BASE_IMAGE} --interpreter-sha256 ${PYTHON_SHA256} \
    && chmod -R a+rX /opt/tensorfold-pins /opt/tensorfold-pin-inputs /opt/tensorfold-foundation-artifacts \
    && cp /opt/tensorfold-pin-inputs/nightly-pins.json /opt/tensorfold-pin-inputs/requirements-foundation.lock \
          /opt/tensorfold-pin-inputs/requirements-build.lock \
          /opt/tensorfold-pin-inputs/requirements-runtime.lock /opt/tensorfold-pin-inputs/requirements-test.lock \
          /opt/tensorfold-pin-inputs/foundation-constraints.txt /opt/tensorfold-pin-inputs/foundation-manifest.json /opt/tensorfold-pins/ \
    && for artifact in nightly-pins.json requirements-foundation.lock requirements-build.lock requirements-runtime.lock requirements-test.lock foundation-constraints.txt foundation-manifest.json; do \
         chmod 0644 /opt/tensorfold-pin-inputs/$artifact /opt/tensorfold-pins/$artifact; done \
    && /opt/python3.15/bin/python3.15 /opt/tensorfold-pins/foundation_guard.py native-addons
RUN --mount=type=cache,id=tensorfold-foundation-pip-${TARGETARCH},target=/root/.cache/pip \
    /opt/tensorfold-foundation/bin/python -m pip install --require-hashes --no-deps -r /opt/tensorfold-pins/requirements-foundation.lock \
    && /opt/tensorfold-foundation/bin/python -m pip check \
    && chmod -R a+rX,a-w /opt/tensorfold-foundation \
    && /opt/python3.15/bin/python3.15 /opt/tensorfold-pins/foundation_guard.py seal \
    && python3 /opt/tensorfold-pins/foundation_guard.py install --role runtime --lock /opt/tensorfold-pins/requirements-runtime.lock \
         --owned-wheels /opt/tensorfold-owned-wheels \
    && python3 -m pip check
ENV TF_FOUNDATION_STATE=/opt/tensorfold-pins/foundation-state.json \
    PIP_CONSTRAINT=/opt/tensorfold-pins/foundation-constraints.txt \
    CUDA_HOME=/usr/local/cuda

FROM dependencies AS application
ARG TARGETARCH
ARG MAX_JOBS=8
ENV MAX_JOBS=${MAX_JOBS}
ARG SOURCE_REVISION=uncommitted
LABEL org.opencontainers.image.revision=${SOURCE_REVISION} \
      org.opencontainers.image.source="https://github.com/kklouzal/TensorFold" \
      deployment.upstream="https://github.com/ashhart/TensorFold" \
      deployment.source_tree="/opt/TensorFold" \
      deployment.integration="flash-next-harness" \
      deployment.platform="linux/${TARGETARCH}" \
      deployment.foundation="/opt/tensorfold-pins/foundation-state.json"
ENV TENSORFOLD_SOURCE_REVISION=${SOURCE_REVISION} \
    TENSORFOLD_EXPERT_ENTRYPOINT=/opt/harness/tensorfold_harness.py
COPY pyproject.toml setup.py README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/TensorFold/
COPY LICENSES /opt/TensorFold/LICENSES
COPY src /opt/TensorFold/src
RUN python3 /opt/tensorfold-pins/foundation_guard.py audit \
    && python3 /opt/tensorfold-pins/foundation_guard.py run -- python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /opt/fork-wheel /opt/TensorFold \
    && python3 /opt/tensorfold-pins/foundation_guard.py application --wheel /opt/fork-wheel/*.whl \
    && python3 -m pip check \
    && python3 /opt/tensorfold-pins/foundation_guard.py run -- python3 -c "from pathlib import Path; import tensorfold, xgrammar; import tensorfold.families.qwen4_exp.native_ssd as reader; assert Path(reader.__file__).with_name('ssd_read.cpp').is_file()"
COPY deploy/gb10/tensorfold_harness.py /opt/harness/tensorfold_harness.py
COPY deploy/gb10/cuda_entrypoint.py deploy/gb10/cuda_driver_probe.py /opt/harness/
COPY deploy/gb10/foundation_guard.py deploy/gb10/foundation_policy.py deploy/gb10/foundation_runtime.py /opt/harness/
COPY src/tensorfold/cuda/foundation_tools.py /opt/harness/foundation_tools.py
ENTRYPOINT ["python3", "/opt/harness/cuda_entrypoint.py", "python3", "/opt/harness/tensorfold_harness.py"]
CMD ["serve", "--help"]

# Verification adds app/test dependencies and Git while preserving the sealed foundation.
FROM application AS verification
ARG TARGETARCH
RUN python3 /opt/tensorfold-pins/foundation_guard.py audit \
    && apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/* \
    && python3 /opt/tensorfold-pins/foundation_guard.py audit
RUN --mount=type=cache,id=tensorfold-foundation-pip-${TARGETARCH},target=/root/.cache/pip \
    python3 /opt/tensorfold-pins/foundation_guard.py install --role verification --lock /opt/tensorfold-pins/requirements-test.lock \
    && python3 -m pip check
COPY tests /opt/TensorFold/tests
COPY deploy /opt/TensorFold/deploy
COPY tools /opt/TensorFold/tools
COPY CHANGELOG.md /opt/TensorFold/CHANGELOG.md
COPY docs/api.md /opt/TensorFold/docs/api.md
# Private captures can have restrictive modes; independent test UIDs need access.
RUN chmod -R a+rX /opt/TensorFold
WORKDIR /opt/TensorFold
ENTRYPOINT ["python3", "/opt/harness/cuda_entrypoint.py", "python3", "-m", "pytest"]

FROM application AS runtime
