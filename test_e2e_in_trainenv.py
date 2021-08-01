from navrep.tools.commonargs import parse_common_args
from stable_baselines3 import PPO

from navrep3d.custom_policy import NavRep3DTupleCNN, NavRep3DTrainEnvFlattened

if __name__ == "__main__":
    args, _ = parse_common_args()

    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dtrainenv_2021_05_13__12_32_37_PPO_E2E_VCARCH_C64_ckpt.zip"

    model = PPO.load(MODELPATH)
    env = NavRep3DTrainEnvFlattened(verbose=0)

    # TODO replace with test routine
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
        if args.render:
            env.render(save_to_file=True)
