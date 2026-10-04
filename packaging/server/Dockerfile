# syntax=docker/dockerfile:1
# Voicekit compute server: `voicekit serve` with NVIDIA CUDA, for x86-64 or
# arm64 (DGX Spark / GB10). Everything is built inside; nothing on the host
# but Docker and the NVIDIA container toolkit.
#
#   docker build -f packaging/server/Dockerfile -t voicekit-server .      (from the repository root)
#   docker run -d --name voicekit-server --hostname "$(hostname)" --gpus all -p 8199:8199 \
#     -v voicekit-data:/data --restart unless-stopped voicekit-server
#
# --hostname: clients show the machine's name, not the container id.
#
# Options: append `--token …`, `--name …` etc., or set NEKOTONE_SERVER_*.
# The voice-clone model (~1.3 GB) downloads into the volume on first start.

ARG CUDA_IMAGE=nvidia/cuda:13.0.1-cudnn-runtime-ubuntu24.04
ARG ORT_VERSION=1.30.0

FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config libasound2-dev libssl-dev \
    && rm -rf /var/lib/apt/lists/*
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH
WORKDIR /src
COPY Cargo.toml Cargo.lock rust-toolchain.toml ./
COPY crates crates
RUN --mount=type=cache,target=/root/.cargo/registry --mount=type=cache,target=/src/target \
    cargo build --release -p nekotone-voice-cli --features cuda && cp target/release/voicekit /voicekit

# ONNX Runtime GPU for this architecture, from its Python wheel (only the libraries)
FROM ubuntu:24.04 AS ort
ARG ORT_VERSION
RUN apt-get update && apt-get install -y --no-install-recommends python3-pip unzip && rm -rf /var/lib/apt/lists/*
RUN pip3 download --no-deps -d /w onnxruntime-gpu==${ORT_VERSION} \
    && mkdir /ort && unzip -j /w/*.whl 'onnxruntime/capi/*.so*' -d /ort \
    && ln -s libonnxruntime.so.${ORT_VERSION} /ort/libonnxruntime.so

FROM ${CUDA_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends libasound2t64 libssl3t64 ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=ort /ort /opt/onnxruntime
COPY --from=build /voicekit /usr/local/bin/voicekit
# ORT_DYLIB_PATH must point into the folder that holds the CUDA provider
# library, or ONNX Runtime silently runs everything on the CPU
ENV ORT_DYLIB_PATH=/opt/onnxruntime/libonnxruntime.so \
    HOME=/data XDG_DATA_HOME=/data NEKOTONE_SERVER_PRINTS=/data/server-prints
VOLUME /data
EXPOSE 8199
ENTRYPOINT ["voicekit", "serve"]
