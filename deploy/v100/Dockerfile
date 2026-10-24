# Pinned Volta development and verification foundation; see README for serving qualification.
FROM nvcr.io/nvidia/cuda:12.9.2-devel-ubuntu24.04@sha256:420850a3fd665171b3f1fd08946c51d50468d732a46d6c42345ea04444755048 AS resolver
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.12-venv python3.12-dev ca-certificates git && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir pip==25.3 packaging==25.0
ENV PATH="/opt/venv/bin:${PATH}" CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=4 OMP_NUM_THREADS=4 \
    TORCH_EXTENSIONS_DIR=/cache/sm70-fp16-cu1292-torch2141-cu126-triton38
WORKDIR /work
ENTRYPOINT ["/opt/venv/bin/python"]

FROM resolver AS verification
COPY deploy/v100/requirements-foundation-amd64.lock deploy/v100/requirements-runtime-amd64.lock \
     deploy/v100/requirements-test-amd64.lock /locks/
RUN --mount=type=cache,target=/root/.cache/pip python -m pip install --require-hashes \
    -r /locks/requirements-foundation-amd64.lock -r /locks/requirements-runtime-amd64.lock \
    -r /locks/requirements-test-amd64.lock
COPY . /src
RUN python -m pip wheel --no-build-isolation --no-deps /src --wheel-dir /wheels \
    && python -m pip install --no-deps /wheels/tensorfold-*.whl
COPY tests /verification/tests
COPY tools/v100_probe.py /verification/v100_probe.py
RUN printf '[pytest]\npythonpath = tests\n' > /verification/pytest.ini
WORKDIR /verification
