FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    bash \
    git \
    wget \
    curl \
    ca-certificates \
    build-essential \
    cmake \
    pkg-config \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-6 \
    libxi6 \
    libxrandr2 \
    libxxf86vm1 \
    libxcursor1 \
    libxinerama1 \
    libegl1 \
    libopengl0 \
    && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/miniconda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p $CONDA_DIR \
    && rm /tmp/miniconda.sh \
    && conda config --set auto_activate_base false \
    && conda config --set solver libmamba \
    && conda clean -afy

ENV PYTHONPATH=/workspace:/workspace/code

WORKDIR /workspace

CMD ["/bin/bash"]