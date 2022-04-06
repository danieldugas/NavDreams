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

def single_sequence_n_step_error(real_sequence, dream_sequence, dones, context_length):
    sequence_length = len(real_sequence)
    dream_length = sequence_length - context_length
    dream_obs = np.array([d["obs"] for d in dream_sequence[context_length:]]) # (D, W, H, C)
    dream_vecobs = np.array([d["state"] for d in dream_sequence[context_length:]])
    real_obs = np.array([d["obs"] for d in real_sequence[context_length:]])
    real_vecobs = np.array([d["state"] for d in real_sequence[context_length:]])
    obs_error = np.mean( # mean over all pixels and channels
        np.reshape(np.square(dream_obs - real_obs), (dream_length, -1)),
        axis=-1) # now (D,)
    vecobs_error = np.mean(
        np.reshape(np.square(dream_vecobs - real_vecobs), (dream_length, -1)),
        axis=-1) # now (D, )
    # if a reset is in the sequence, ignore predictions for subsequent frames
    ignore_error = np.cumsum(dones[context_length:]) > 0 # (D,)
    obs_error[ignore_error] = np.nan
    vecobs_error[ignore_error] = np.nan
    return obs_error, vecobs_error

def worldmodel_n_step_error(worldmodel, test_dataset_folder,
                            sequence_length=32, context_length=16, samples=0):
    # parameters
    shuffle = True
    dream_length = sequence_length - context_length
    # load dataset
    seq_loader = WorldModelDataset(test_dataset_folder, sequence_length, lidar_mode="images",
                                   channel_first=False, as_torch_tensors=False, file_limit=10)
    N = min(samples, len(seq_loader))
    if samples == 0:
        N = len(seq_loader)
    obs_error = np.ones((N, dream_length)) * np.nan
    vecobs_error = np.ones((N, dream_length)) * np.nan
    indices = list(range(len(seq_loader)))
    if shuffle:
        random.Random(4).shuffle(indices)
    pbar = tqdm(indices[:N])
#     for i, (x, a, y, x_rs, y_rs, dones) in enumerate(seq_loader):
    for i, idx in enumerate(pbar):
        if idx >= len(seq_loader): # this shouldn't be necessary, but it is (len is not honored by for)
            continue
        x, a, y, x_rs, y_rs, dones = seq_loader[idx]
        real_sequence = [dict(obs=x[j], state=x_rs[j], action=a[j]) for j in range(sequence_length)]
        dream_sequence = worldmodel.fill_dream_sequence(real_sequence, context_length)
        obs_error[i], vecobs_error[i] = single_sequence_n_step_error(
            real_sequence, dream_sequence, dones, context_length)
        if i % 10 == 0:
            pbar.set_description(
                f"1-step error {np.nanmean(obs_error, axis=0)[0]:.5f} \
                  16-step error {np.nanmean(obs_error, axis=0)[15]:.5f}"
            )
    archive_path = "/tmp/{}_n_step_errors.npz".format(type(worldmodel).__name__)
    np.savez_compressed(archive_path, obs_error=obs_error, vecobs_error=vecobs_error)
    print(f"Saved n-step errors to {archive_path}")
    mean_obs_error = np.nanmean(obs_error, axis=0)
    mean_vecobs_error = np.nanmean(vecobs_error, axis=0)
    return mean_obs_error, mean_vecobs_error

def hide_axes_but_keep_ylabel(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if False:
        ax.set_axis_off()

def sequence_to_gif(dream_sequence, worldmodel_name, real_sequence=None, sequence_idx=0):
    from moviepy.editor import ImageSequenceClip
    dreamframes = [(d["obs"] * 255).astype(np.uint8) for d in dream_sequence]
    frames = dreamframes
    if real_sequence is not None:
        realframes = [(d["obs"] * 255).astype(np.uint8) for d in real_sequence]
        frames = [np.concatenate([r, d], axis=0) for r, d in zip(realframes, dreamframes)]
    clip = ImageSequenceClip(list(frames), fps=20)
    clip.write_gif("/tmp/dream_of_length{}_index{}_{}.gif".format(
        len(frames), sequence_idx, worldmodel_name), fps=20)

def sequences_to_comparison_gif(dream_sequences, worldmodel_names, real_sequence, sequence_idx=0):
    from moviepy.editor import ImageSequenceClip
    all_dreamframes = [[(d["obs"] * 255).astype(np.uint8) for d in dream_sequence]
                       for dream_sequence in dream_sequences]
    realframes = [(d["obs"] * 255).astype(np.uint8) for d in real_sequence]
    frames = [np.concatenate(imglist, axis=0) for imglist in zip(realframes, *all_dreamframes)]
    clip = ImageSequenceClip(list(frames), fps=20)
    clip.write_gif("/tmp/comparison_dream_of_length{}_index{}_{}.gif".format(
        len(frames), sequence_idx, '_'.join([n[:6] for n in worldmodel_names])), fps=20)

def black_sequence_after_done(dream_sequence):
    done = False
    for dic in dream_sequence:
        img = dic["obs"]
        if np.mean(img) < 0.1:
            done = True
        if done:
            dic["obs"] = np.zeros_like(img)

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

    worldmodel_types = ["TransformerL_V0"] # comparison image
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
    examples = list(range(len(seq_loader)))
    example_sequences = {examples[i]: None for i in range(len(examples))}
    for idx in example_sequences:
        if idx >= len(seq_loader):
            raise IndexError("{} is out of range".format(idx))
        (x, a, y, x_rs, y_rs, dones) = seq_loader[idx]
        example_sequence = [dict(obs=x[i], state=x_rs[i], action=a[i], done=dones[i])
                            for i in range(sequence_length)]
        example_sequences[idx] = example_sequence

    # fill dream sequences from world model
    #example_filled_sequences = []
    model = worldmodels[0]
    for n, idx in enumerate(tqdm(example_sequences)):
        if example_sequences[idx] is None:
            continue
        real_sequence = example_sequences[idx]
        #dones = [step["done"] for step in real_sequence]

        dream =  model.fill_dream_sequence(real_sequence, context_length)

        real_data = np.zeros([sequence_length,64,64,3])
        dream_data = np.zeros([sequence_length, 64, 64, 3])
        for i in range(sequence_length):
            real_data[i] = real_sequence[i]['obs']
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
