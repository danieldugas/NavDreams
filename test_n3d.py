import os
from strictfire import StrictFire
from stable_baselines3 import PPO

from navrep3d.navrep3dtrainencodedenv import EncoderObsWrapper
from navrep3d.navrep3danyenv import NavRep3DAnyEnvDiscrete
from plot_gym_training_progress import get_variant

def main(build_name="staticasl", render=True, difficulty_mode="easy"):
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daslencodedenv_2021_12_11__00_23_55_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daslencodedenv_2021_12_08__10_18_09_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dkozehdoldencodedenv_2022_01_13__16_13_45_DISCRETE_PPO_GPT_V_ONLY_V64M64_SCR_bestckpt.zip" # noqa

    backend = "GPT"
    encoding = "V_ONLY"
    model = PPO.load(MODELPATH)
    variant = get_variant(os.path.basename(MODELPATH))
    env = NavRep3DAnyEnvDiscrete(build_name=build_name,
                                 debug_export_every_n_episodes=0,
                                 difficulty_mode=difficulty_mode)
    env = EncoderObsWrapper(env, backend=backend, encoding=encoding, variant=variant)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        if done:
            env.reset()
            if reward > 50.:
                print("Success!")
            else:
                print("Failure.")
        if render:
            if build_name == "rosbag":
                env.render(save_to_file=True, action_override=action)
            else:
                env.render(save_to_file=True)


if __name__ == "__main__":
    StrictFire(main)
