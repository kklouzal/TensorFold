# The digest pins the deployed Torch/CUDA/vision stack. The fork replaces the
# TensorFold package with a wheel built from this repository's owned source.
ARG BASE_IMAGE=ghcr.io/miaai-lab/qwen3.8-flash-next-single-dgx-spark-tensorfold@sha256:667693ccd7eea95d365ee5a17eed97067e13deb131a6e61fbbbbe6f69d0ebc69
FROM ${BASE_IMAGE} AS runtime
ARG SOURCE_REVISION=uncommitted
LABEL org.opencontainers.image.revision=${SOURCE_REVISION} \
      org.opencontainers.image.source="https://github.com/kklouzal/TensorFold" \
      deployment.upstream="https://github.com/ashhart/TensorFold" \
      deployment.source_tree="/opt/TensorFold" \
      deployment.integration="gb10-flash-next-harness"
COPY pyproject.toml README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/TensorFold/
COPY LICENSES /opt/TensorFold/LICENSES
COPY src /opt/TensorFold/src
RUN python3 -m pip wheel --no-deps --no-build-isolation --wheel-dir /opt/fork-wheel /opt/TensorFold \
    && python3 -m pip install --no-deps --force-reinstall /opt/fork-wheel/*.whl \
    && python3 -m pip install --no-deps --no-cache-dir xgrammar==0.2.8 \
    && python3 -c "from pathlib import Path; import tensorfold, xgrammar; import tensorfold.families.qwen4_exp.native_ssd as reader; assert Path(reader.__file__).with_name('ssd_read.cpp').is_file()"
COPY deploy/gb10/tensorfold_harness.py /opt/harness/tensorfold_harness.py
ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh", "python3", "/opt/harness/tensorfold_harness.py"]
CMD ["serve", "--help"]

# CPU verification has its own image target; it adds no dependencies to runtime.
FROM runtime AS verification
RUN python3 -m pip install --no-deps --no-cache-dir pytest==8.4.2 pluggy==1.6.0 iniconfig==2.1.0
COPY tests /opt/TensorFold/tests
COPY deploy /opt/TensorFold/deploy
WORKDIR /opt/TensorFold
ENTRYPOINT ["python3", "-m", "pytest"]
