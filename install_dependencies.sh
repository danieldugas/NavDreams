set -x
set -e

# system dependencies
sudo add-apt-repository multiverse
sudo apt install -y virtualenv python3-pip git-lfs

# create virtualenv
rm -rf ~/n3denv
virtualenv ~/n3denv --python=python3.6
source ~/n3denv/bin/activate

# pip install python dependencies
pip install --upgrade pip # fixes ubuntu 20 pip pep517 error
sudo apt install -y build-essential python3-dev cmake # needed to compile some of the pip packages
pip install numpy cython
pip install mlagents
pip install pyrvo2-danieldugas matplotlib ipython pyyaml snakeviz stable-baselines3  pyglet \
  typer strictfire \
  navrep navdreams \
  jedi==0.17 gym==0.18.0 # jedi because newer versions break ipython (py3.6) gym error in (py3.6) https://stackoverflow.com/questions/69520829/openai-gym-attributeerror-module-contextlib-has-no-attribute-nullcontext

# install pydreamer and navdreams in development mode
mkdir -p ~/Code
cd ~/Code/
git clone git@github.com:danieldugas/pydreamer.git --branch n3d
cd ~/Code/pydreamer
pip install -e .
cd ~/Code/
git clone git@github.com:danieldugas/NavDreams.git
cd ~/Code/NavDreams
pip install -e .

# Pre-install the simulator binaries
git lfs clone git@github.com:ethz-asl/navrep3d_lfs.git ~/navdreams_binaries

# ROS (useful if extracting rosbag)
# pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag tf tf2_ros
# pip install --extra-index-url https://rospypi.github.io/simple/ sensor_msgs std_srvs geometry_msgs nav_msgs std_msgs visualization_msgs cv_bridge
# on RTX 3070 (sm_86)
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
