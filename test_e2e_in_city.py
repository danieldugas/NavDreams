from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO

from navrep3d.custom_policy import NavRep3DTrainEnvDiscreteFlattened

if __name__ == "__main__":
    args, _ = parse_common_args()

    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dtrainenv_2021_08_13__01_11_33_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip"

    model = PPO.load(MODELPATH)
    env = NavRep3DTrainEnvDiscreteFlattened(verbose=0, build_name="./city.x86_64")

    # TODO replace with test routine
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
        if args.render:
            env.render(save_to_file=True)
