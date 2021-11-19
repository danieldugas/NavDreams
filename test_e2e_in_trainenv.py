from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO

from navrep3d.custom_policy import NavRep3DTrainEnvDiscreteFlattened, NavRep3DTrainEnvFlattened

if __name__ == "__main__":
    args, _ = parse_common_args()

    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3daltenv_2021_11_10__12_38_52_DISCRETE_PPO_E2E_VCARCH_C64_bestckpt.zip"

    model = PPO.load(MODELPATH)
    if "DISCRETE" in MODELPATH:
        env = NavRep3DTrainEnvDiscreteFlattened(verbose=0, debug_export_every_n_episodes=1)
    else:
        env = NavRep3DTrainEnvFlattened(verbose=0, debug_export_every_n_episodes=1)

    # TODO replace with test routine
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
        if args.render:
            env.render(save_to_file=True)
