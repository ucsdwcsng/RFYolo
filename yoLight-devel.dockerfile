# FROM jupyter/scipy-notebook:python-3.9.12
FROM nvcr.io/nvidia/pytorch:22.10-py3

ARG USER=torch
ARG UID=${UID}
ARG USERDIR=/home/${USER}

RUN adduser --gecos "" -u ${UID} ${USER}
RUN passwd -d ${USER}

RUN apt -y update && \
    DEBIAN_FRONTEND=noninteractive apt -y install \
    build-essential \
    software-properties-common \
    gnupg2 \
    libglu1-mesa-dev \
    libgl1-mesa-dev \
    libosmesa6-dev \
    xvfb \
    ffmpeg \
    curl \
    patchelf \
    libglfw3 \  
    libglfw3-dev \
    cmake \
    sudo 

RUN DEBIAN_FRONTEND=noninteractive apt -y install \
    zip \
    libgl1-mesa-glx

RUN pip3 install --upgrade \
    setuptools \
    pip

RUN pip3 install \
    matplotlib>=3.2.2 \
    numpy>=1.18.5 \
    # opencv-python \
    # opencv-contrib-python \
    Pillow>=7.1.2 \
    PyYAML>=5.3.1 \
    requests>=2.23.0 \
    scipy>=1.4.1 \
    tqdm>=4.41.0 \
    protobuf\<4.21.3 \
    pandas>=1.1.4 \
    seaborn>=0.11.0 \
    tensorboard>=2.4.1 \
    psutil \
    thop 

RUN pip3 install \
    magicattr \
    pytorch-ignite

RUN pip3 install \
    imageio

RUN wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list   && \
    wget -qO - https://dvc.org/deb/iterative.asc | gpg --dearmor > packages.iterative.gpg   && \
    install -o root -g root -m 644 packages.iterative.gpg /etc/apt/trusted.gpg.d/   && \
    rm -f packages.iterative.gpg    && \
    apt update              && \
    apt install dvc


RUN mkdir -p ${USERDIR}/.vscode-server && chown ${USER}:${USER} ${USERDIR}/.vscode-server
VOLUME ${USERDIR}/.vscode-server

EXPOSE 8888

USER ${USER}
