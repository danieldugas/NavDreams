# syntax=docker/dockerfile:1
FROM ubuntu:18.04

RUN apt-get update

# things usually on someone's computer, but not in a minimal ubuntu image
RUN apt-get install -y sudo
RUN sudo apt-get install -y build-essential python3-dev cmake gcc g++ # needed to build navrep dependencies
RUN sudo apt-get install -y git git-lfs # needed to download sim binaries
RUN sudo apt-get install -y python3-pip virtualenv

# pip
RUN pip3 install --upgrade pip # fixes ubuntu 20 pip pep517 error

# RUN sudo apt-get install -y python3-opencv
RUN pip3 install numpy cython
RUN pip3 install torch==1.9.0
RUN pip3 install mlagents
RUN pip3 install matplotlib ipython pyyaml snakeviz stable-baselines3  pyglet navrep typer strictfire \
  jedi==0.17 gym==0.18.0 # jedi because newer versions break ipython (py3.6) gym error in (py3.6) https://stackoverflow.com/questions/69520829/openai-gym-attributeerror-module-contextlib-has-no-attribute-nullcontext
RUN pip3 install navdreams
