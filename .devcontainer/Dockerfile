FROM ghcr.io/trxcllnt/devcontainer-images/cmake-ninja-sccache-llvm-cuda-mambaforge:latest AS base

COPY bin/setup-mrc-env.sh /opt/mrc-devcontainer/bin/setup-mrc-env

ENV MAMBA_NO_BANNER=1
ENV PATH="${PATH}:/opt/mrc-devcontainer/bin"
