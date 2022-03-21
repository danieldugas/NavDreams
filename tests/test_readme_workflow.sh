# install dependencies
cd /tmp
wget https://raw.githubusercontent.com/danieldugas/NavDreams/master/install_dependencies.sh
chmod +x install_dependencies.sh
./install_dependencies.sh

# fetch a minimal version of the navdreams_data folder, for testing only
mkdir -p ~/navdreams_data
cd ~/navdreams_data
wget -r -np -nH --cut-dirs 3 -R "index.html*" http://robotics.ethz.ch/~asl-datasets/2022_NavDreams/minimal_navdreams_data/

# Readme Example Commands
source ~/n3denv/bin/activate
cd ~/Code/NavDreams
python -m navdreams.navrep3danyenv --scenario replica
python make_vae_dataset.py --scope SCR --n-sequences 2
python train_gpt.py --dataset SCR --max-steps 2 --dry-run
python train_gym_discrete_navrep3dtrainencodedenv.py --variant SCR --scenario city --n 10 --dry-run
python test_any.py --model-path \
   "~/navdreams_data/results/models/gym/navrep3dcityencodedenv_2022_02_18__18_26_31_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" \
   --n-episodes 100 --build-name "./city.x86_64" --difficulty-mode "hardest" --render True
