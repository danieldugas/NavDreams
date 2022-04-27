import os
import numpy as np
import random
from matplotlib import pyplot as plt
from navrep.tools.wdataset import WorldModelDataset
from strictfire import StrictFire
from tqdm import tqdm
import copy
import pickle
from navrep.models.gpt import GPT, GPTConfig, load_checkpoint
from navdreams.rssm import RSSMWMConf, RSSMWorldModel
from navdreams.rssm_a0 import RSSMA0WMConf, RSSMA0WorldModel
from navdreams.tssm import TSSMWMConf, TSSMWorldModel
from navdreams.transformerL import TransformerLWMConf, TransformerLWorldModel
from navdreams.worldmodel import DummyWorldModel, GreyDummyWorldModel
from navdreams.auto_debug import enable_auto_debug
from paper_sequences import load_paper_sequences
from plot_gym_training_progress import make_legend_pickable

class TransformerWorldModel(GPT):
    def fill_dream_sequence(self, real_sequence, context_length):
        """ Fills dream sequence based on context from real_sequence
            real_sequence is a list of dicts, one for each step in the sequence.
            each dict has
            "obs": numpy image (W, H, CH) [0, 1]
            "state": numpy (2,) [-inf, inf]
            "action": numpy (3,) [-inf, inf]

            context_length (int): number of steps of the real sequence to keep in the dream sequence

            output:
            dream_sequence: same length as the real_sequence, but observations and states are predicted
                    open-loop by the worldmodel, while actions are taken from the real sequence
            """
        T = self.get_block_size()
        sequence_length = len(real_sequence)
        if sequence_length > T:
            print("Warning: sequence_length > block_size ({} > {} in {})!".format(
                sequence_length, T, type(self).__name__))
        dream_sequence = copy.deepcopy(real_sequence[:context_length])
        dream_sequence[-1]['action'] = None
        real_actions = [d['action'] for d in real_sequence]
        next_actions = real_actions[context_length-1:sequence_length-1]
        for action in next_actions:
            dream_sequence[-1]['action'] = action * 1.
            img_npred, goal_pred = self.get_next(dream_sequence[-T:])
            # update sequence
            dream_sequence.append(dict(obs=img_npred, state=goal_pred, action=None))
        dream_sequence[-1]['action'] = next_actions[-1] * 1.
        return dream_sequence




def main(dataset="SCR",
         gpu=False,
         dream_length=48,
         context_length=16,
         n_examples=2,
         error=False,
         dataset_info=False,
         offset=0,
         skip=3,
         samples=1000,
         gifs=False,
         paper_sequences=False,
         ):
    sequence_length = dream_length + context_length

    worldmodel_types = ["DummyWorldModel"]  # comparison image
    discrete_actions = False

    if dataset == "SCR":
        dataset_dir = [os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/navrep3dalt"),
                       os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/navrep3dcity"),
                       os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/navrep3doffice"),
                       os.path.expanduser("~/navdreams_data/wm_test_data/datasets/V/navrep3dasl"),
                       os.path.expanduser("~/navdreams_data/wm_experiments/datasets/V/rosbag")]


    def load_worldmodels(worldmodel_types):
        print("Loading worldmodels...")
        worldmodels = []
        for worldmodel_type in worldmodel_types:
            if worldmodel_type == "transformer":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/transformer_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                BLOCK_SIZE = 32
                _H = 64
                _C = 3
                mconf = GPTConfig(BLOCK_SIZE, _H)
                mconf.image_channels = _C
                if discrete_actions:
                    mconf.n_action = 4
                model = TransformerWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "RSSM_A1":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/RSSM_A1_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                mconf = RSSMWMConf()
                mconf.image_channels = 3
                model = RSSMWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "RSSM_A0" or worldmodel_type == "RSSM_A0_explicit":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/RSSM_A0_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                mconf = RSSMA0WMConf()
                mconf.image_channels = 3
                model = RSSMA0WorldModel(mconf, gpu=gpu)
                if worldmodel_type == "RSSM_A0_explicit":
                    class RSSMA0WorldModelExplicit(RSSMA0WorldModel):
                        noop = 0
                    model = RSSMA0WorldModelExplicit(mconf, gpu=gpu)
                    model.fill_dream_sequence = model.fill_dream_sequence_through_images
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "TSSM_V2":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/TSSM_V2_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                mconf = TSSMWMConf()
                mconf.image_channels = 3
                model = TSSMWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "TransformerL_V0":
                wm_model_path = "~/navdreams_data/wm_experiments/models/W/TransformerL_V0_{}".format(dataset)
                wm_model_path = os.path.expanduser(wm_model_path)
                mconf = TransformerLWMConf()
                mconf.image_channels = 3
                if discrete_actions:
                    mconf.n_action = 4
                model = TransformerLWorldModel(mconf, gpu=gpu)
                load_checkpoint(model, wm_model_path, gpu=gpu)
                worldmodel = model
            elif worldmodel_type == "DummyWorldModel":
                worldmodel = DummyWorldModel(gpu=gpu)
            elif worldmodel_type == "GreyDummyWorldModel":
                worldmodel = GreyDummyWorldModel(gpu=gpu)
            else:
                raise NotImplementedError
            worldmodels.append(worldmodel)
        print("Done.")
        return worldmodels

    worldmodels = load_worldmodels(worldmodel_types)

        # load example sequences
    seq_loader = WorldModelDataset(dataset_dir, sequence_length, lidar_mode="images",
                                   channel_first=False, as_torch_tensors=False, file_limit=10)

    print("{} sequences available".format(len(seq_loader)))

    seq_len = len(seq_loader)
    model = worldmodels[0]
    for n in tqdm(range(seq_len)):
        (x, a, y, x_rs, y_rs, dones) = seq_loader[n]
        example_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i], done=dones[i])
                            for i in range(sequence_length)]
        #dones = [step["done"] for step in real_sequence]

        dream =  model.fill_dream_sequence(example_sequence, context_length)

        real_data = np.zeros([sequence_length,64,64,3])
        dream_data = np.zeros([sequence_length, 64, 64, 3])
        for i in range(sequence_length):
            real_data[i] = example_sequence[i]['obs']
            dream_data[i] = dream[i]['obs']
        save_data = {}
        save_data['real'] = real_data
        save_data['fake'] = dream_data

        p = os.path.join('./gan_data', '{}.pkl'.format(n))
        with open(p, 'wb') as f:
            pickle.dump(save_data, f)



if __name__ == "__main__":
    enable_auto_debug()
    StrictFire(main)
