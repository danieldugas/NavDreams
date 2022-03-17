import os
from navrep.tools.envplayer import EnvPlayer
from strictfire import StrictFire

from navdreams.mlagents_gym_wrapper import (NavRep3DStaticASLEnv, NavRep3DStaticASLEnvDiscrete,
                                            MLAGENTS_BUILD_NAMES)
from navdreams.archiveenv import ArchiveEnv
from navdreams.navrep3dtrainenv import NavRep3DTrainEnv, NavRep3DTrainEnvDiscrete
from navdreams.custom_policy import FlattenN3DObsWrapper

# internally, scenarios were given different original names, that those used in the paper (for clarity)
scenario_to_build_name = {
    "simple": "./alternate.x86_64",
    "city": "./city.x86_64",
    "office": "./office.x86_64",
    "modern": "staticasl",
    "cathedral": "cathedral",
    "gallery": "gallery",
    "replica": "kozehd",
}

def NavRep3DAnyEnv(**kwargs):
    """ wrapper to hide the difference between the two kinds of navrep3d environments
    (crowdbotchallenge vs mlagents)
    allows creating either a navrep3dtrainenv (train, alt, city, office) or navprep3dstaticasl env
    depending on build name """
    build_name = kwargs.get('build_name', "./build.x86_64")
    if build_name in MLAGENTS_BUILD_NAMES:
        return NavRep3DStaticASLEnv(**kwargs)
    elif build_name == "rosbag":
        directory = os.path.expanduser("~/navrep3d_W/datasets/V/rosbag")
        return ArchiveEnv(directory, shuffle_episodes=True)
    else:
        return NavRep3DTrainEnv(**kwargs)

def NavRep3DAnyEnvDiscrete(**kwargs):
    """ same as NavRep3DAnyEnv but with DiscreteActionWrapper """
    build_name = kwargs.get('build_name', "./build.x86_64")
    if build_name in MLAGENTS_BUILD_NAMES:
        return NavRep3DStaticASLEnvDiscrete(**kwargs)
    elif build_name == "rosbag":
        raise NotImplementedError("No known uses for achive env with discrete action")
    else:
        return NavRep3DTrainEnvDiscrete(**kwargs)

def NavRep3DAnyEnvDiscreteFlattened(**kwargs):
    """ same as NavRep3DAnyEnv but with DiscreteActionWrapper
    and FlattenN3DObsWrapper (used by custom E2E policy) """
    return FlattenN3DObsWrapper(NavRep3DAnyEnvDiscrete(**kwargs))

def test_envs():
    for build_name in [
            "./build.x86_64",
            "./alternate.x86_64",
            "./city.x86_64",
            "./office.x86_64",
            "staticasl",
            "cathedral",
            "gallery",
            "kozehd",
            "rosbag",
    ]:
        for envtype in [NavRep3DAnyEnv, NavRep3DAnyEnvDiscrete]:
            if build_name == "rosbag" and envtype == NavRep3DAnyEnvDiscrete:
                continue
            env = envtype(build_name=build_name)
            print(build_name)
            print(env.reset())
            print(env.step(env.action_space.sample()))
            env.render()
            env.close()


def main(scenario="replica", difficulty_mode="progressive"):
    build_name = scenario_to_build_name[scenario]
    env = NavRep3DAnyEnv(build_name=build_name, difficulty_mode=difficulty_mode)
    player = EnvPlayer(env)
    player.run()


if __name__ == "__main__":
    StrictFire(main)
