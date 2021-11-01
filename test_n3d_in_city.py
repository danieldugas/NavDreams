import os
from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO

from navrep3d.navrep3dtrainencodedenv import SubprocVecNavRep3DEncodedEnvDiscrete

from plot_gym_training_progress import get_variant

if __name__ == "__main__":
    args, _ = parse_common_args()

    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daltencodedenv_2021_10_18__08_14_32_DISCRETE_PPO_GPT_V_ONLY_V64M64_S_ckpt.zip" # noqa

    model = PPO.load(MODELPATH)
    variant = get_variant(os.path.basename(MODELPATH))
    if "DISCRETE" in MODELPATH:
        env = SubprocVecNavRep3DEncodedEnvDiscrete("GPT", "V_ONLY", variant, 4,
                                                   build_name="./city.x86_64",
                                                   debug_export_every_n_episodes=1 if args.render else 0)
    else:
        raise NotImplementedError

    # TODO replace with test routine
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones[0]:
            env.reset()
