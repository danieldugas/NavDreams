import os
import numpy as np
import gym
from strictfire import StrictFire
from navrep.scripts.make_vae_dataset import generate_vae_dataset, SemiRandomMomentumPolicy, HumanControlPolicy

from navrep3d.navrep3danyenv import NavRep3DAnyEnv
from navrep3d.navrep3dtrainenv import (convert_continuous_to_discrete_action,
                                       convert_discrete_to_continuous_action)

class OneHotActionPolicyWrapper(object):
    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        self.policy.reset()

    def predict(self, obs, env):
        action = self.policy.predict(obs, env)
        discrete_action = convert_continuous_to_discrete_action(action)
        onehot_action = np.array([0, 0, 0, 0], dtype=np.uint8)
        onehot_action[discrete_action] = 1
        return onehot_action

class OneHotActionEnvWrapper(gym.core.ActionWrapper):
    """ for wrapping the env to allow it to take one hot actions """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = None # we don't want this to be used for training! only for dataset generation
        self.zero_action = np.array([0, 0, 0, 1])

    def action(self, action):
        if action is None:
            action = self.zero_action
        actionidx = np.argmax(action)
        cont_actions = convert_discrete_to_continuous_action(actionidx)
        return cont_actions

class QuantizedActionPolicyWrapper(object):
    """ returns a continuous action, but which is guaranteed to belong to the subset of discrete actions
    """
    def __init__(self, policy):
        self.policy = policy

    def reset(self):
        self.policy.reset()

    def predict(self, obs, env):
        action = self.policy.predict(obs, env)
        discrete_action = convert_continuous_to_discrete_action(action)
        quantized_action = convert_discrete_to_continuous_action(discrete_action)
        return quantized_action

def main(n_sequences=100, env="S", render=False, dry_run=False, subproc_id=0, n_subprocs=1,
         discrete_actions=False, quantized_actions=False):
    difficulty_mode = "random"
    if env == "S":
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dtrain")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dtrain"
        build_name = "./build.x86_64"
    elif env == "Salt":
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dalt")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dalt"
        build_name = "./alternate.x86_64"
    elif env == "CC": # City
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dcity"
        build_name = "./city.x86_64"
    elif env == "CO": # Office
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3doffice")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3doffice"
        build_name = "./office.x86_64"
    elif env == "R": # R
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3daslv2")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3daslv2"
        build_name = "staticasl"
    elif env == "OG":
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dgallery")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dgallery"
        build_name = "gallery"
    elif env == "OC":
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcathedral")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dcathedral"
        build_name = "cathedral"
    elif env == "K": # R
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dkozehdr")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dkozehdr"
        build_name = "kozehd"
        difficulty_mode = "bimodal"
    elif env == "rosbag": # only for testing, used in regen
        archive_dir = "/tmp/navrep3d/datasets/V/navrep3drosbag"
        build_name = "rosbag"
        if discrete_actions or quantized_actions:
            raise ValueError("discrete/quantized actions not supported for rosbag")
    else:
        raise NotImplementedError
    env = NavRep3DAnyEnv(verbose=0, collect_statistics=False,
                         build_name=build_name, port=25005+subproc_id,
                         tolerate_corruption=False, difficulty_mode=difficulty_mode)
    policy = SemiRandomMomentumPolicy() if True else HumanControlPolicy()
    if discrete_actions:
        archive_dir = archive_dir.replace("/V/navrep3d", "/V/discrete_navrep3d")
        env = OneHotActionEnvWrapper(env)
        policy = OneHotActionPolicyWrapper(policy)
    if quantized_actions:
        archive_dir = archive_dir.replace("/V/navrep3d", "/V/quantized_navrep3d")
        policy = QuantizedActionPolicyWrapper(policy)
    generate_vae_dataset(
        env, n_sequences=n_sequences,
        subset_index=subproc_id, n_subsets=n_subprocs,
        policy=policy,
        render=render, archive_dir=archive_dir)


if __name__ == "__main__":
    StrictFire(main)
