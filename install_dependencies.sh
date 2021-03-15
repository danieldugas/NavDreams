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
virtualenv ~/cbc3env --python=python3.6
source ~/cbc3env/bin/activate

pip install matplotlib numpy cython ipython pyyaml snakeviz stable-baselines3 pyglet navrep
cd lib_dwa
pip install .
cd ..
git clone git@github.com:danieldugas/Python-RVO2.git
cd Python-RVO2
pip install .
