from stable_baselines3 import SAC

from navrep3dtrainenv import NavRep3DTrainEnv

if __name__ == "__main__":
    MILLION = 1000000
    TRAIN_STEPS = 60 * MILLION
    MODELPATH = "/home/daniel/navrep3d/models/gym/navrep3dtrainenv_2021_04_06__20_01_12_SAC_E2E_VCARCH_C64_ckpt.zip"
    env = NavRep3DTrainEnv(verbose=1)

    model = SAC.load(MODELPATH)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
        env.render()
