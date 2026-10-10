# syntax=docker/dockerfile:1.7
# Linux ARM64 and AMD64. This immutable OCI index selects each audited native
# NGC CUDA 13.4.1 base; the isolated venv uses target-specific genuine nightlies.
ARG BASE_IMAGE=nvcr.io/nvidia/cuda-dl-base:26.09-cuda13.4-inference-devel-ubuntu24.04@sha256:f86c199a302a3a7405d5f055adf1311cd660e85d8a66cf452be6eb45d5768639
FROM ${BASE_IMAGE} AS foundation
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev python3.12-venv ca-certificates ninja-build \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/tensorfold-venv
ENV PATH=/opt/tensorfold-venv/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM foundation AS dependencies
ARG TARGETARCH
COPY deploy/gb10/requirements-runtime.lock deploy/gb10/requirements-runtime-amd64.lock \
     deploy/gb10/requirements-test.lock deploy/gb10/requirements-test-amd64.lock \
     deploy/gb10/nightly-pins.json deploy/gb10/nightly-pins-amd64.json \
     deploy/gb10/platform_pins.py deploy/gb10/select_pins.py /opt/tensorfold-pins/
RUN python3 /opt/tensorfold-pins/select_pins.py --architecture ${TARGETARCH} --directory /opt/tensorfold-pins
RUN --mount=type=cache,id=tensorfold-cu134-nightly-pip-${TARGETARCH},target=/root/.cache/pip \
    python3 -m pip install --require-hashes -r /opt/tensorfold-pins/requirements-runtime.lock \
    && python3 -m pip check

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
      deployment.cuda="13.4.1" \
      deployment.torch="2.16.0.dev20261006+cu134" \
      deployment.triton="3.9.0+gitaad2a60d"
COPY pyproject.toml setup.py README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/TensorFold/
COPY LICENSES /opt/TensorFold/LICENSES
COPY src /opt/TensorFold/src
RUN python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /opt/fork-wheel /opt/TensorFold \
    && python3 -m pip install --no-deps /opt/fork-wheel/*.whl \
    && python3 -m pip check \
    && python3 -c "from pathlib import Path; import tensorfold, xgrammar; import tensorfold.families.qwen4_exp.native_ssd as reader; assert Path(reader.__file__).with_name('ssd_read.cpp').is_file()"
COPY deploy/gb10/tensorfold_harness.py /opt/harness/tensorfold_harness.py
COPY deploy/gb10/cuda_entrypoint.py /opt/harness/cuda_entrypoint.py
COPY deploy/gb10/configure_cudnn.py /opt/harness/configure_cudnn.py
RUN python3 /opt/harness/configure_cudnn.py --output /opt/tensorfold-cudnn
# Torch, CUDA and Triton upgrades get their own ABI-specific persistent cache.
ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/opt/tensorfold-cudnn:${LD_LIBRARY_PATH} \
    CPATH=/opt/tensorfold-venv/lib/python3.12/site-packages/nvidia/cu13/include:/opt/tensorfold-venv/lib/python3.12/site-packages/nvidia/cudnn/include \
    TORCH_EXTENSIONS_DIR=/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/${TARGETARCH}/torch_extensions \
    TRITON_CACHE_DIR=/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/${TARGETARCH}/triton \
    CUDA_CACHE_PATH=/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/${TARGETARCH}/cuda \
    TORCHINDUCTOR_CACHE_DIR=/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/${TARGETARCH}/inductor \
    XDG_CACHE_HOME=/cache/cu1341-torch216-dev20261006-triton39-aad2a60d/${TARGETARCH}/xdg
RUN python3 -c "import os; from pathlib import Path; paths=[Path(p) for p in os.environ['CPATH'].split(':')]; assert all(any((p / h).is_file() for p in paths) for h in ('cusparse.h', 'cublas_v2.h', 'cusolverDn.h', 'cudnn.h'))"
ENTRYPOINT ["python3", "/opt/harness/cuda_entrypoint.py", "python3", "/opt/harness/tensorfold_harness.py"]
CMD ["serve", "--help"]

# Verification adds pinned test tools only to this separate target.
FROM application AS verification
ARG TARGETARCH
RUN --mount=type=cache,id=tensorfold-cu134-nightly-pip-${TARGETARCH},target=/root/.cache/pip \
    python3 -m pip install --require-hashes -r /opt/tensorfold-pins/requirements-test.lock
COPY tests /opt/TensorFold/tests
COPY deploy /opt/TensorFold/deploy
COPY tools /opt/TensorFold/tools
COPY CHANGELOG.md /opt/TensorFold/CHANGELOG.md
# Private source captures can have 0700/0600 modes; test workers run as a different UID.
RUN chmod -R a+rX /opt/TensorFold
WORKDIR /opt/TensorFold
ENTRYPOINT ["python3", "-m", "pytest"]

# A plain docker build produces the serving image; tests require --target verification.
FROM application AS runtime
