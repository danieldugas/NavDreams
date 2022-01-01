import os
import numpy as np
from strictfire import StrictFire
from navrep.scripts.make_vae_dataset import generate_vae_dataset, SemiRandomMomentumPolicy, HumanControlPolicy

from navrep3d.navrep3danyenv import NavRep3DAnyEnv
from navrep3d.navrep3dtrainenv import convert_continuous_to_discrete_action

class QuantizedActionPolicyWrapper(object):
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

def main(n_sequences=100, env="S", render=False, dry_run=False, subproc_id=0, n_subprocs=1,
         discrete_actions=False):
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
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dasl")
        if dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dasl"
        build_name = "staticasl"
    elif env == "rosbag": # only for testing, used in regen
        archive_dir = "/tmp/navrep3d/datasets/V/navrep3drosbag"
        build_name = "rosbag"
        if discrete_actions:
            raise ValueError("discrete actions not supported for rosbag")
    else:
        raise NotImplementedError
    if discrete_actions:
        archive_dir = archive_dir.replace("/V/navrep3d", "/V/discrete_navrep3d")
    env = NavRep3DAnyEnv(verbose=0, collect_statistics=False,
                         build_name=build_name, port=25005+subproc_id,
                         tolerate_corruption=False, difficulty_mode="random")
    policy = SemiRandomMomentumPolicy() if True else HumanControlPolicy()
    if discrete_actions:
        policy = QuantizedActionPolicyWrapper(policy)
    generate_vae_dataset(
        env, n_sequences=n_sequences,
        subset_index=subproc_id, n_subsets=n_subprocs,
        policy=policy,
        render=render, archive_dir=archive_dir)


if __name__ == "__main__":
    StrictFire(main)
