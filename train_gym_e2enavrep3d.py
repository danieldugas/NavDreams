# from stable_baselines import TD3
# from stable_baselines.td3.policies import CnnPolicy
# from stable_baselines.ddpg.noise import NormalActionNoise

from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3 import SAC
from sb3_callbacks import NavRep3DLogCallback

from navrep3denv import NavRep3DEnv

if __name__ == "__main__":
    MILLION = 1000000
    TRAIN_STEPS = 60 * MILLION
    MODELPATH = "SAC_navrep3d"
    env = NavRep3DEnv(verbose=0)

    cb = NavRep3DLogCallback(logpath="SAC_navrep3d_log", savepath=MODELPATH, verbose=1)

    model = SAC(CnnPolicy, env, verbose=1, buffer_size=10000)
    model.learn(total_timesteps=TRAIN_STEPS, callback=cb)

    print("Saving model to {}".format(MODELPATH))
    model.save(MODELPATH)

    del model # remove to demonstrate saving and loading

    model = SAC.load(MODELPATH)

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones:
            env.reset()
        env.render()
