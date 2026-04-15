ARG BASE_IMAGE=nvidia/cuda:12.1.0-devel-ubuntu22.04
ARG RUNTIME_IMAGE=nvidia/cuda:12.1.0-runtime-ubuntu22.04

# ── builder: compile kenlm + install all Python deps into a venv ─────────────
FROM ${BASE_IMAGE} AS builder
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y upgrade --only-upgrade systemd openssl cryptsetup && \
    apt-get install -y \
      build-essential cmake curl git \
      gcc g++ \
      libprotobuf-dev protobuf-compiler \
      python3.11 python3.11-dev python3.11-venv \
      libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip

ARG REQUIREMENTS=requirements.txt
COPY ${REQUIREMENTS} requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir ".[torch,st,diffusers]"

# ── runtime: lean CUDA runtime image, no compilers ───────────────────────────
FROM ${RUNTIME_IMAGE} AS runtime
SHELL ["/bin/bash", "-c"]

LABEL maintainer="Hugging Face"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get -y upgrade --only-upgrade systemd openssl cryptsetup && \
    apt-get install -y \
      python3.11 \
      git git-lfs \
      bzip2 tar curl \
      libsndfile1 \
      ffmpeg \
      libprotobuf23 \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /var/lib/dpkg && touch /var/lib/dpkg/status

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python

# Copy only the venv produced by the builder — no compilers or dev headers
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application
COPY src/huggingface_inference_toolkit huggingface_inference_toolkit
COPY src/huggingface_inference_toolkit/webservice_starlette.py webservice_starlette.py

# Copy entrypoint and change permissions
COPY --chmod=0755 scripts/entrypoint.sh entrypoint.sh

ENTRYPOINT ["bash", "-c", "./entrypoint.sh"]
