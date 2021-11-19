import os
from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO

from navrep3d.archiveenv import ArchiveEnv
from navrep3d.navrep3dtrainenv import DiscreteActionWrapper
from navrep3d.navrep3dtrainencodedenv import EncoderObsWrapper

from plot_gym_training_progress import get_variant

if __name__ == "__main__":
    args, _ = parse_common_args()
    shuffle = True

    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daltencodedenv_2021_10_08__16_34_19_DISCRETE_PPO_GPT_V_ONLY_V64M64_Salt_ckpt.zip" # noqa
    directory = [os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")]

    model = PPO.load(MODELPATH)
    variant = get_variant(os.path.basename(MODELPATH))
    if "DISCRETE" in MODELPATH:
        env = ArchiveEnv(directory, shuffle_episodes=shuffle)
        env = DiscreteActionWrapper(env)
        env = EncoderObsWrapper(env)
    else:
        raise NotImplementedError

    # TODO replace with test routine
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            env.reset()
