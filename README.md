# NavDreams: Social-Navigation World Models

Camera-based robot navigation simulator, world-models, and policies for the paper
**NavDreams: Towards Camera-Only RL Navigation Among Humans**

![dream](media/dreaming.gif)
![title](media/title.gif)

In NavDreams, we use the world-model/dreamer concept to learn a model which predicts the future,
then use this world model to do camera-based robot navigation.

![title](media/real_tests.gif)

More information available in [our paper](http://arxiv.org/abs/2203.12299)

## Simulator

Our simulator currently contains 7 scenarios. Some of these are based on 2D environments from [previous work](https://www.github.com/danieldugas/navrep).

![scenarios](media/scenarios.png)

<!-- To find out how to modify the simulator for your own needs, follow [this link](https://www.github.com/danieldugas/WaveEnv) -->

## Models and Tools

Model implementations, trained checkpoints, training tools, and plotting tools are made available in this repo. See the *How-to* section for examples of use. 

![models](media/models.gif)

# How-to

- Installation
- Running the simulator
- Training the world-model
- Training the controller
- Testing the controller

## Using Docker

First, make sure [docker is installed](https://docs.docker.com/engine/install/ubuntu/),
and, if your computer has a GPU, that [nvidia-docker is installed](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)


A pre-built container image is available [here](https://drive.google.com/file/d/1O6YxcyMxkSIkpwjaAvrSlYB0xTzxbMVp/view?usp=share_link).
To load it into your docker images:
```
docker load --input navdreams-docker-image-v2.tar
```

Then, you can run the container with:
```
xhost + # allows docker to create UI windows in ubuntu
docker run --rm --gpus all --env="DISPLAY" --net=host --name=n3d_test -it n3d
(inside running container) # python -m navdreams.navrep3danyenv --scenario replica
```

Not working? Check the [troubleshooting guide](https://github.com/danieldugas/NavDreams#troubleshooting-guide).


## Manual Install

If you already have the typical python/DL dependencies, installing should be as simple as
```
pip install navdreams
```
For a more details, refer to this [install script](install_dependencies.sh)

## Running the simulated environments

```
python -m navdreams.navrep3danyenv --scenario replica
```

See [simulator troubleshooting](wiki/troubleshoot_sim.md) if you encounter issues.

## Training the World-Model

Generate the dataset

```
python make_vae_dataset.py --scope SCR
```

Train the world-model

```
python train_gpt.py --dataset SCR
```

## Training the Controller

```
python train_gym_discrete_navrep3dtrainencodedenv.py --variant SCR --scenario city
```

and for the end-to-end baseline

```
python train_gym_discrete_e2enavrep3d_PPO.py
```

## Testing the Controller

Test the controller you trained, (or a [downloaded model](https://drive.google.com/drive/folders/17_o7jPLKKlRbgySIOxn6-Z1kUHcOgld5?usp=sharing)).
```
 python test_any.py --model-path \
   "~/navdreams_data/results/models/gym/navrep3dcityencodedenv_2022_02_18__18_26_31_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
   --n-episodes 100 --build-name "./city.x86_64" --difficulty-mode "hardest" --render True
```

## More

[Example Workflow: Evaluating World-model Dreams](wiki/worldmodel_error.md)

---

## Troubleshooting Guide

GPU / Docker check (the output should be text information about your GPU):
```
docker run -it --rm --gpus all pytorch/pytorch:1.9.0-cuda10.2-cudnn7-devel nvidia-smi
```

Check whether the graphical environment is working. A window with a rotating horse should appear.
```
sudo docker run --rm --gpus all --env="DISPLAY" --net=host -it n3d bash -c "glmark2"
```

### Running on a laptop

On my T480, the above line
```
docker run --rm --gpus all --env="DISPLAY" --net=host --name=n3d_test -it n3d
```
leads to an error due to missing gpu. But when removing the --gpus all flag, things seem to work, except that OpenGL complains about missing drivers, and the simulator sometimes crashes.
```
libGL error: failed to load driver: i915
libGL error: failed to open /dev/dri/card0: No such file or directory
```
in this case, replacing `--gpus all` with `--device /dev/dri/` solves these errors, and the simulator can then run smoothly on the laptop with integrated graphics.
```
xhost +
docker run --rm --device /dev/dri/ --env="DISPLAY" --net=host --name=n3d_test -it n3d
```

### No Screen? No problem.
Are you trying to run this on a computer without a screen (like a cloud server)?  
Note that due to using Unity as the simulation backend (MLAgents), a visual environment is required.  
It's still possible to do so on cloud servers like AWS (we tested this workflow and it works), but requires a good understanding of the graphics pipeline.  
Here's an example on how to start the XServer on a ubuntu server with a Tesla T4 GPU:
```
# create virtual display with nvidia drivers
# nvidia-xconfig --query-gpu-info
# With headless GPUs (e.g. Tesla T4), which don't have display outputs, remove the --use-display-device=none option
# sudo nvidia-xconfig --busid=PCI:0:30:0 --use-display-device=none --virtual=1280x1024
sudo nvidia-xconfig --busid=PCI:0:30:0 --virtual=1280x1024
sudo Xorg :0 & # unlike startx, this only starts the x server, no DEs
nvidia-smi # check that Xorg is running on the GPU
echo "export DISPLAY=:0" >> ~/.bashrc # applications need to know which display to use
```

[Read here](https://dugas.ch/lord_of_the_files/run_your_unity_ml_executable_in_the_cloud.html) for more details.

If you want to see the virtual display, you can use a vnc viewer, like for example:
```
sudo apt install -y x11vnc xfce4-session
x11vnc -display :0 -usepw -rfbport 5901
DISPLAY=:0 startxfce4
```

