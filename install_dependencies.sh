# virtualenv ~/cbcenv
# source ~/cbcenv/bin/activate
#
# pip install matplotlib numpy cython ipython pyyaml snakeviz
# cd lib_dwa
# pip install .
# cd ..
# git clone git@github.com:danieldugas/Python-RVO2.git
# cd Python-RVO2
# pip install .
#
set -x
set -e
rm -rf ~/cbc3env
virtualenv ~/cbc3env --python=python3.6
source ~/cbc3env/bin/activate
pip install --upgrade pip # fixes ubuntu 20 pip pep517 error

sudo apt install -y build-essential python3-dev cmake # needed to compile some of the pip packages

pip install numpy cython
pip install matplotlib ipython pyyaml snakeviz stable-baselines3 pyglet navrep fire
cd lib_dwa
pip install .
cd ..
git clone git@github.com:danieldugas/Python-RVO2.git
cd Python-RVO2
pip install .

# on RTX 3070 (sm_86)
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
