from stable_baselines3 import SAC

from navrep3denv import NavRep3DEnv

if __name__ == "__main__":
    MILLION = 1000000
    TRAIN_STEPS = 60 * MILLION
    MODELPATH = "SAC_navrep3d"
    env = NavRep3DEnv(verbose=1)

    model = SAC.load(MODELPATH)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
        env.render()
