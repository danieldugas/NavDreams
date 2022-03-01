from navrep.tools.envplayer import EnvPlayer
from strictfire import StrictFire

from navrep3d.navrep3danyenv import NavRep3DAnyEnv

def main(build_name=None, cont=0):
    build_names = [
        "./alternate.x86_64",
        "./city.x86_64",
        "./office.x86_64",
        "staticasl",
        "cathedral",
        "gallery",
        "kozehd",
    ]
    difficulty_modes = {
        "./alternate.x86_64": "hardest",
        "./city.x86_64":      "hardest",
        "./office.x86_64":    "random",
        "staticasl":          "medium",
        "cathedral":          "medium",
        "gallery":            "easy",
        "kozehd":             "easier",
    }
    if build_name is not None:
        build_names = [build_name]
    total_steps = cont
    for build_name in build_names:
        render_mode = "image_only"
        step_by_step = False
        print(build_name)
        env = NavRep3DAnyEnv(build_name=build_name, difficulty_mode=difficulty_modes[build_name])
        env.reset()
        env.step(env.action_space.sample())
        env.reset()
        env.total_steps = total_steps
        _ = EnvPlayer(env, render_mode, step_by_step, save_to_file=True)
        total_steps = env.total_steps


if __name__ == "__main__":
    StrictFire(main)
