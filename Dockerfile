FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]
ENV LC_CTYPE en_US.UTF-8
ENV LANG en_US.UTF-8

RUN apt-get update && apt-get -y install sudo software-properties-common
RUN sudo add-apt-repository multiverse && \
    sudo apt-get install -y virtualenv python3-pip git-lfs

# create virtualenv
# RUN add-apt-repository ppa:deadsnakes/ppa
# RUN apt-get update
# RUN apt-get -y install python3.6 python3-pip
RUN rm -rf $HOME/n3denv && \
    virtualenv $HOME/n3denv --python=python3.6 && \
    source $HOME/n3denv/bin/activate && \
    pip install --upgrade pip && \
    sudo apt-get install -y build-essential python3-dev cmake && \
    pip install numpy cython && \
    pip install pyrvo2-danieldugas matplotlib ipython pyyaml snakeviz stable-baselines3  pyglet \
      typer strictfire \
      mlflow \
      grpcio==1.43.0 \
      navrep navdreams \
      mlagents==0.19.0 \
      torch==1.9.0 \
      tensorflow==1.13.2 Keras==2.3.1 \
      jedi==0.17 gym==0.18.0 && \
    pip install mlagents==0.27.0 mlagents-envs==0.27.0 --no-deps # separate because has a too strict pytorch requirement

# Pre-install the simulator binaries
RUN git lfs clone http://github.com/ethz-asl/navrep3d_lfs.git $HOME/navdreams_binaries
RUN cd $HOME/navdreams_binaries && \
  git reset --hard && \
  git lfs pull

# install pydreamer and navdreams in development mode
RUN mkdir -p $HOME/Code
RUN cd $HOME/Code/ && \
  git clone https://github.com/danieldugas/pydreamer.git --branch n3d
RUN cd $HOME/Code/ && \
  git clone https://github.com/danieldugas/NavDreams.git

# mitigates encoding errors in pip install
RUN sudo apt-get -y install language-pack-en-base
RUN sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8

RUN echo "source ~/n3denv/bin/activate" >> ~/.bashrc
RUN cd $HOME/Code/pydreamer && \
 source $HOME/n3denv/bin/activate && \
 pip install -e .
RUN cd $HOME/Code/NavDreams && \
 source $HOME/n3denv/bin/activate && \
 pip install -e . --no-deps

# Dependencies for glvnd and X11.
RUN apt-get update \
  && apt-get install -y -qq --no-install-recommends \
    libxext6 \
    libx11-6 \
    libglvnd0 \
    libgl1 \
    libglx0 \
    libegl1 \
    freeglut3-dev \
  && rm -rf /var/lib/apt/lists/*

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# needed by pyglet
RUN apt-get install -y  libfreetype6-dev libfontconfig1-dev
