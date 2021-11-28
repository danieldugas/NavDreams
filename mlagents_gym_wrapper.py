import gym
import time
from pandas import DataFrame
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
# from gym_unity.envs import UnityToGymWrapper

from navrep3d.navrep3dtrainenv import DiscreteActionWrapper

def obs_spec_to_obs_space(obs_spec):
    """
    Converts a MLAgents observation spec to an OpenAI Gym observation space.
    """
    return gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_spec.shape, dtype=np.float32)

class MLAgentsGymEnvWrapper(gym.Env):
    def __init__(self, unity_env):
        self.unity_env = unity_env
        self.unity_env.reset()
        if len(unity_env.behavior_specs) == 0:
            raise ValueError("There are no behaviors in this environment.")
        if len(unity_env.behavior_specs) > 1:
            raise ValueError("Only single behaviors are supported. Use the MLAgentsGymVecEnvWrapper instead")
        # fill gym spaces
        for behavior_name in unity_env.behavior_specs:
            obs_specs, act_specs = unity_env.behavior_specs[behavior_name]
            # observation
            if isinstance(obs_specs, list):
                obs_space = gym.spaces.Dict({
                    obs_spec.name: obs_spec_to_obs_space(obs_spec) for obs_spec in obs_specs
                })
            else:
                raise NotImplementedError
            # actions
            if len(act_specs.discrete_branches) != 0:
                # self.action_space = gym.spaces.Discrete(3)
                raise NotImplementedError("Only continuous actions are supported.")
            action_space = gym.spaces.Box(low=-np.inf, high=np.inf,
                                          shape=(act_specs.continuous_size,), dtype=np.float32)
            self.behavior_name = behavior_name
            self.action_space = action_space
            self.observation_space = obs_space

    def reset(self):
        self.unity_env.reset()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        decision_step = decision_steps[0]
        obs = decision_step.obs
        obs_dict = {key: ob for key, ob in zip(self.observation_space, obs)}
        return obs_dict

    def step(self, action):
        n_agents = 1
        action_tuple = ActionTuple(continuous=np.reshape(action, (n_agents,) + action.shape))
        self.unity_env.set_actions(self.behavior_name, action_tuple)
        self.unity_env.step()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        if len(decision_steps) == 1:
            # the agent requests an action at the next step
            done = False
            decision_step = decision_steps[0]
            obs = decision_step.obs
            reward = decision_step.reward
            obs_dict = {key: ob for key, ob in zip(self.observation_space, obs)}
            return obs_dict, reward, done, {}
        if len(terminal_steps) == 1:
            # episode has ended, next step should be reset
            done = True
            terminal_step = terminal_steps[0]
            obs = terminal_step.obs
            reward = terminal_step.reward
            obs_dict = {key: ob for key, ob in zip(self.observation_space, obs)}
            return obs_dict, reward, done, {}
        raise ValueError("Expected either a decision ({}) or a terminal step ({}).".format(
            len(decision_steps), len(terminal_steps)))

class NavRep3DStatisticsEnvWrapper(gym.core.Wrapper):
    """
    Wrapper which adds a NavRep3D compatible statistics collection (used to log training performance)
    """
    def __init__(self, env):
        super().__init__(env)
        self.unwrapped.episode_statistics = DataFrame(
            columns=[
                "total_steps",
                "scenario",
                "damage",
                "steps",
                "goal_reached",
                "reward",
                "num_agents",
                "num_walls",
                "wall_time",
            ])
        self.total_steps = 0
        self.scenario_name = "navrep3dasl"
        self.steps_since_reset = 0

    def reset(self):
        self.episode_reward = 0
        self.steps_since_reset = 0
        obs = super().reset()
        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        self.steps_since_reset += 1
        self.total_steps += 1
        self.episode_reward += reward
        goal_is_reached = reward > 50.0
        current_scenario = 0
        raise NotImplementedError("Set up difficulty side channel")
        self.unwrapped.episode_statistics.loc[len(self.episode_statistics)] = [
            self.total_steps,
            self.scenario_name,
            np.nan,
            self.steps_since_reset,
            goal_is_reached,
            self.episode_reward,
            current_scenario,
            100,
            time.time(),
        ]
        return obs, reward, done, info

class NavRep3DRendererEnvWrapper(gym.core.Wrapper):
    """
    Wrapper which adds a NavRep3D compatible render function
    """

    def __init__(self, env):
        super().__init__(env)
        # variables for rendering)
        self.viewer = None
        self.last_action = np.array([0, 0, 0])
        self.last_image = None
        self.last_crowd = None
        self.last_walls = None
        self.last_odom = None
        self.last_lidar = None
        self.goal_xy = None
        self.reset_in_progress = False
        self.verbose = False
        self.total_steps = 0

    def reset(self):
        self.episode_reward = 0
        self.total_steps += 1
        obs = super().reset()
        self.last_image = (obs['CameraSensor'] * 255).astype(np.uint8)
        self.goal_xy = obs['VectorSensor_size5'][:2]
        self.last_odom = np.array([0, 0, 0, 0, 0, 0, 0])
        self.unwrapped.last_action = np.zeros((3,)) # hack which allows encodedenv wrapper to get last action
        return obs

    def step(self, action):
        # the env player passes a len 3 action, but our environment expects action_space
        action_corrected = np.zeros(self.action_space.shape)
        action_corrected[:3] = action
        obs, reward, done, info = super().step(action_corrected)
        self.total_steps += 1
        self.episode_reward += reward
        self.last_image = (obs['CameraSensor'] * 255).astype(np.uint8)
        self.goal_xy = obs['VectorSensor_size5'][:2]
        self.last_odom = np.array([0, 0, 0, 0, 0, 0, 0])
        self.last_action = action
        self.unwrapped.last_action = action # hack which allows encodedenv wrapper to get last action
        return obs, reward, done, info

    def render(self, mode='human', close=False, save_to_file=False):
        GOAL_RADIUS = 0.5
        ROBOT_RADIUS = 0.3
        AGENT_RADIUS = 0.33
        from timeit import default_timer as timer
        tic = timer()
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return

        import pyglet
        from pyglet.gl import GLubyte
        arrimg = self.last_image
        if arrimg is None:
            return
        width = arrimg.shape[1]
        height = arrimg.shape[0]
        pixels = arrimg[::-1,:,:].flatten()
        rawData = (GLubyte * len(pixels))(*pixels)
        imageData = pyglet.image.ImageData(width, height, 'RGB', rawData)

        if self.verbose > 1:
            toc = timer()
            print("Render (fetch): {} Hz".format(1. / (toc - tic)))
            tic = timer()

        if mode == 'matplotlib':
            from matplotlib import pyplot as plt
            self.viewer = plt.figure()
            plt.imshow(arrimg)
            plt.ion()
            plt.show()
            plt.pause(0.1)
        elif mode in ['human', 'image_only']:
            image_only = mode == 'image_only'
            # Window and viewport size
            _256 = 256
            WINDOW_W = _256
            WINDOW_H = _256
            VP_W = WINDOW_W
            VP_H = WINDOW_H
            from gym.envs.classic_control import rendering
            import pyglet
            from pyglet import gl
            # Create self.viewer
            if self.viewer is None:
                self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
                self.score_label = pyglet.text.Label(
                    '0000', font_size=12,
                    x=20, y=WINDOW_H*2.5/40.00, anchor_x='left', anchor_y='center',
                    color=(255,255,255,255))
                self.currently_rendering_iteration = 0
            # Render in pyglet
            self.currently_rendering_iteration += 1
            win = self.viewer.window
            win.switch_to()
            win.dispatch_events()
            win.clear()
            gl.glViewport(0, 0, VP_W, VP_H)
            image_in_vp = rendering.Transform()
            image_in_vp.set_scale(VP_W / width, VP_H / height)
            # Render top-down
            if not image_only:
                image_in_vp.set_translation(2, _256 - 2 - width)
                image_in_vp.set_scale(1, 1)
                topdown_in_vp = rendering.Transform()
                topdown_in_vp.set_scale(10, 10)
                topdown_in_vp.set_translation(_256 // 2, _256 // 2)
                topdown_in_vp.set_rotation(np.pi/2.)
                # colors
                bgcolor = np.array([0.4, 0.8, 0.4])
                obstcolor = np.array([0.3, 0.3, 0.3])
                goalcolor = np.array([1., 1., 0.3])
                nosecolor = np.array([0.3, 0.3, 0.3])
                agentcolor = np.array([0., 1., 1.])
                robotcolor = np.array([1., 1., 1.])
                lidarcolor = np.array([1., 0., 0.])
                # Green background
                gl.glBegin(gl.GL_QUADS)
                gl.glColor4f(bgcolor[0], bgcolor[1], bgcolor[2], 1.0)
                gl.glVertex3f(0, VP_H, 0)
                gl.glVertex3f(VP_W, VP_H, 0)
                gl.glVertex3f(VP_W, 0, 0)
                gl.glVertex3f(0, 0, 0)
                gl.glEnd()
                topdown_in_vp.enable()
                # LIDAR
                if self.last_lidar is not None and self.last_odom is not None:
                    px, py, angle, _, _, _, _ = self.last_odom
                    # LIDAR rays
                    scan = self.last_lidar
                    lidar_angles = self.lidar_angles
                    x_ray_ends = px + scan * np.cos(lidar_angles)
                    y_ray_ends = py + scan * np.sin(lidar_angles)
                    is_in_fov = np.cos(lidar_angles - angle) >= 0.78
                    for ray_idx in range(len(scan)):
                        end_x = x_ray_ends[ray_idx]
                        end_y = y_ray_ends[ray_idx]
                        gl.glBegin(gl.GL_LINE_LOOP)
                        if is_in_fov[ray_idx]:
                            gl.glColor4f(1., 1., 0., 0.1)
                        else:
                            gl.glColor4f(lidarcolor[0], lidarcolor[1], lidarcolor[2], 0.1)
                        gl.glVertex3f(px, py, 0)
                        gl.glVertex3f(end_x, end_y, 0)
                        gl.glEnd()
                # Map closed obstacles ---
                if self.last_walls is not None:
                    for wall in self.last_walls:
                        gl.glBegin(gl.GL_LINE_LOOP)
                        gl.glColor4f(obstcolor[0], obstcolor[1], obstcolor[2], 1)
                        for vert in wall:
                            gl.glVertex3f(vert[0], vert[1], 0)
                        gl.glEnd()
                # Agent body
                def make_circle(c, r, res=10):
                    thetas = np.linspace(0, 2*np.pi, res+1)[:-1]
                    verts = np.zeros((res, 2))
                    verts[:,0] = c[0] + r * np.cos(thetas)
                    verts[:,1] = c[1] + r * np.sin(thetas)
                    return verts
                def gl_render_agent(px, py, angle, r, color):
                    # Agent as Circle
                    poly = make_circle((px, py), r)
                    gl.glBegin(gl.GL_POLYGON)
                    gl.glColor4f(color[0], color[1], color[2], 1)
                    for vert in poly:
                        gl.glVertex3f(vert[0], vert[1], 0)
                    gl.glEnd()
                    # Direction triangle
                    xnose = px + r * np.cos(angle)
                    ynose = py + r * np.sin(angle)
                    xright = px + 0.3 * r * -np.sin(angle)
                    yright = py + 0.3 * r * np.cos(angle)
                    xleft = px - 0.3 * r * -np.sin(angle)
                    yleft = py - 0.3 * r * np.cos(angle)
                    gl.glBegin(gl.GL_TRIANGLES)
                    gl.glColor4f(nosecolor[0], nosecolor[1], nosecolor[2], 1)
                    gl.glVertex3f(xnose, ynose, 0)
                    gl.glVertex3f(xright, yright, 0)
                    gl.glVertex3f(xleft, yleft, 0)
                    gl.glEnd()
                if self.last_odom is not None:
                    gl_render_agent(self.last_odom[0], self.last_odom[1], self.last_odom[2],
                                    ROBOT_RADIUS, robotcolor)
                if self.last_crowd is not None:
                    for n, agent in enumerate(self.last_crowd):
                        gl_render_agent(agent[1], agent[2], agent[3], AGENT_RADIUS, agentcolor)
                # Goal markers
                xgoal, ygoal = self.goal_xy
                r = GOAL_RADIUS
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                triangle = make_circle((xgoal, ygoal), r, res=3)
                for vert in triangle:
                    gl.glVertex3f(vert[0], vert[1], 0)
                gl.glEnd()
                if self.last_odom is not None:
                    gl.glBegin(gl.GL_LINES)
                    gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 0.5)
                    gl.glVertex3f(self.last_odom[0], self.last_odom[1], 0)
                    gl.glVertex3f(xgoal, ygoal, 0)
                    gl.glEnd()
                topdown_in_vp.disable()
            # Render image
            image_in_vp.enable()
            # black background
            gl.glBegin(gl.GL_QUADS)
            gl.glColor4f(1, 1, 1, 1.0)
            gl.glVertex3f(0, height, 0)
            gl.glVertex3f(width, height, 0)
            gl.glVertex3f(width, 0, 0)
            gl.glVertex3f(0, 0, 0)
            gl.glEnd()
            # image
            imageData.blit(0,0)
            image_in_vp.disable()
            # Text
            self.score_label.text = "{} S {} R {:.1f} A {:.1f} {:.1f} {:.1f}".format(
                '*' if self.reset_in_progress else '',
                self.infer_current_scenario(),
                self.episode_reward,
                self.last_action[0],
                self.last_action[1],
                self.last_action[2],
            )
            self.score_label.draw()
            win.flip()
            if save_to_file:
                pyglet.image.get_buffer_manager().get_color_buffer().save(
                    "/tmp/navrep3dtrainenv{:05}.png".format(self.total_steps))

        if self.verbose > 1:
            toc = timer()
            print("Render (display): {} Hz".format(1. / (toc - tic)))

    def infer_current_scenario(self):
        return "?"

    def _get_dt(self):
        return 0.2

    def _get_viewer(self):
        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        super().close()

class DictToTupleObsWrapper(gym.core.ObservationWrapper):
    """
    Wrapper for compatibility
    """

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Tuple(
            [env.observation_space[key] for key in env.observation_space])

    def observation(self, obs):
        obs_tuple = tuple([obs[key] for key in obs])
        return obs_tuple

def NavRep3DStaticASLEnvDiscrete(verbose=0, collect_statistics=True,
                                 debug_export_every_n_episodes=0, port=25001,
                                 unity_player_dir="LFS/executables", build_name="staticasl",
                                 start_with_random_rot=True, tolerate_corruption=True):
    """ Shorthand to create env made by stacking wrappers which is equivalent to NavRep3DTrainEnvDiscrete """
    if unity_player_dir != "LFS/executables":
        raise ValueError
    if build_name != "staticasl":
        raise ValueError
    if debug_export_every_n_episodes != 0:
        raise ValueError
    if not start_with_random_rot:
        raise ValueError
    unity_env = UnityEnvironment(file_name="LFS/executables/staticasl", seed=1, side_channels=[])
    env = MLAgentsGymEnvWrapper(unity_env)
    env = NavRep3DRendererEnvWrapper(env)
    env = NavRep3DStatisticsEnvWrapper(env)
    env = DictToTupleObsWrapper(env)
    env = DiscreteActionWrapper(env)
    WIP
    return env

def main(step_by_step=False, render_mode='human'):
    np.set_printoptions(precision=1, suppress=True)
    # This is a non-blocking call that only loads the environment.
    unity_env = UnityEnvironment(file_name="LFS/executables/staticasl", seed=1, side_channels=[])
    env = MLAgentsGymEnvWrapper(unity_env)
#     env = UnityToGymWrapper(unity_env, uint8_visual=True)
    env = NavRep3DRendererEnvWrapper(env)

    from navrep.tools.envplayer import EnvPlayer
    player = EnvPlayer(env, render_mode, step_by_step)
    player.run()


if __name__ == "__main__":
    from strictfire import StrictFire
    StrictFire(main)
