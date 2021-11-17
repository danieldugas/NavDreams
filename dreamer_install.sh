set -x
set -e
rm -rf ~/dreamervenv
virtualenv ~/dreamervenv --python=python3.6
source ~/dreamervenv/bin/activate
pip install --upgrade pip # fixes ubuntu 20 pip pep517 error

sudo apt install -y build-essential python3-dev cmake # needed to compile some of the pip packages

pip install numpy cython
pip install matplotlib ipython pyyaml snakeviz stable-baselines3  pyglet navrep strictfire \
  jedi==0.17 gym==0.18.0 # jedi because newer versions break ipython (py3.6) gym error in (py3.6) https://stackoverflow.com/questions/69520829/openai-gym-attributeerror-module-contextlib-has-no-attribute-nullcontext
pip install keras==2.6.* # later version is not compatible with python 3.6
pip install dreamerv2
pip install tensorflow==2.6.0 tensorflow_probability ruamel.yaml 'gym[atari]' dm_control
# on RTX 3070 (sm_86)
# pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
