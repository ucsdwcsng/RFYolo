# FROM jupyter/scipy-notebook:python-3.9.12
FROM nvcr.io/nvidia/pytorch:21.08-py3

ARG USER=torch
ARG USERDIR=/home/${USER}

RUN adduser --gecos "" ${USER}
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

RUN pip3 install --upgrade pip

RUN pip3 install \
    matplotlib>=3.2.2 \
    numpy>=1.18.5 \
    opencv-python>=4.1.1 \
    Pillow>=7.1.2 \
    PyYAML>=5.3.1 \
    requests>=2.23.0 \
    scipy>=1.4.1 \
    torch>=1.7.0,!=1.12.0 \
    torchvision>=0.8.1,!=0.13.0 \
    tqdm>=4.41.0 \
    protobuf<4.21.3 \
    pandas>=1.1.4 \
    seaborn>=0.11.0 \
    tensorboard>=2.4.1 \
    psutil \
    thop 

EXPOSE 8888

USER ${USER}
