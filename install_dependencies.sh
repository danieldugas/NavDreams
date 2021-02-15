virtualenv ~/cbcenv
source ~/cbcenv/bin/activate

pip install matplotlib numpy cython ipython pyyaml snakeviz
cd lib_dwa
pip install .
cd ..
git clone git@github.com:danieldugas/Python-RVO2.git
cd Python-RVO2
pip install .
