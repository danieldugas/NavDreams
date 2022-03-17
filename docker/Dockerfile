
# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel

RUN apt-get update

# things usually on someone's computer, but not in a minimal ubuntu image
RUN apt-get install -y sudo
RUN sudo apt-get install -y build-essential python3-dev cmake gcc g++ # needed to build navrep dependencies
RUN sudo apt-get install -y git git-lfs # needed to download sim binaries
RUN sudo apt-get install -y python3-pip virtualenv
# GUI libraries needed by opencv, but during install a dependency asks for the timezone
ENV CONTAINER_TIMEZONE=America/New_York
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone
RUN DEBIAN_FRONTEND="noninteractive" sudo apt-get install -y python3-opencv

ENV VIRTUAL_ENV=navdreamsenv
RUN virtualenv $VIRTUAL_ENV --python=python3.6
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# pip
RUN pip3 install --upgrade pip # fixes ubuntu 20 pip pep517 error


RUN pip3 install numpy cython
RUN pip3 install \
  mlagents==0.27.0 \
  matplotlib ipython pyyaml snakeviz stable-baselines3  pyglet navrep typer strictfire \
  jedi==0.17 gym==0.18.0 \
  navdreams
 # jedi because newer versions break ipython (py3.6) gym error in (py3.6) https://stackoverflow.com/questions/69520829/openai-gym-attributeerror-module-contextlib-has-no-attribute-nullcontext
RUN git clone git@github.com:danieldugas/NavDreams.git
