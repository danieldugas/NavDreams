import os
import gym
import time
from pandas import DataFrame
import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.base_env import ActionTuple
# from gym_unity.envs import UnityToGymWrapper

from navdreams.crowd_sim_info import Timeout, ReachGoal, Collision, CollisionOtherAgent
from navdreams.navrep3dtrainenv import DiscreteActionWrapper, mark_port_use, download_binaries_if_not_found

HOMEDIR = os.path.expanduser("~")
# DEFAULT_UNITY_EXE = os.path.join(HOMEDIR, "Code/cbsim/navrep3d/LFS/mlagents_executables")
UNITY_EXE_REPOSITORY = "https://github.com/ethz-asl/navrep3d_lfs"
UNITY_EXE_DIR = os.path.join(HOMEDIR, "navdreams_binaries")
DEFAULT_UNITY_EXE = os.path.join(UNITY_EXE_DIR, "mlagents_executables")

MLAGENTS_BUILD_NAMES = ["staticasl", "cathedral", "gallery", "kozehd"]

class MLAgentsGymEnvWrapper(gym.Env):
    """
    A generic wrapper, takes a unity_env and turns it into a gym env
    """
    def __init__(self, unity_env, port_lock_handle):
        self.port_lock_handle = port_lock_handle
        self.visual_to_uint8 = True
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
                    obs_spec.name: self.obs_spec_to_obs_space(obs_spec, obs_spec.name)
                    for obs_spec in obs_specs
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

    def obs_spec_to_obs_space(self, obs_spec, name):
        """
        Converts a MLAgents observation spec to an OpenAI Gym observation space.
        """
        if self.visual_to_uint8:
            if name == "CameraSensor":
                return gym.spaces.Box(low=0, high=255, shape=obs_spec.shape, dtype=np.uint8)
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=obs_spec.shape, dtype=np.float32)

    def reset(self):
        self.unity_env.reset()
        decision_steps, terminal_steps = self.unity_env.get_steps(self.behavior_name)
        decision_step = decision_steps[0]
        obs = decision_step.obs
        obs_dict = {key: ob for key, ob in zip(self.observation_space, obs)}
        if self.visual_to_uint8:
            obs_dict['CameraSensor'] = (obs_dict['CameraSensor'] * 255).astype(np.uint8)
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
            if self.visual_to_uint8:
                obs_dict['CameraSensor'] = (obs_dict['CameraSensor'] * 255).astype(np.uint8)
            return obs_dict, reward, done, {}
        if len(terminal_steps) == 1:
            # episode has ended, next step should be reset
            done = True
            terminal_step = terminal_steps[0]
            obs = terminal_step.obs
            reward = terminal_step.reward
            obs_dict = {key: ob for key, ob in zip(self.observation_space, obs)}
            if self.visual_to_uint8:
                obs_dict['CameraSensor'] = (obs_dict['CameraSensor'] * 255).astype(np.uint8)
            return obs_dict, reward, done, {}
        raise ValueError("Expected either a decision ({}) or a terminal step ({}).".format(
            len(decision_steps), len(terminal_steps)))

    def close(self):
        print("Closing unity environment...")
        self.unity_env.close()
        self.port_lock_handle.free()

class StaticASLToNavRep3DEnvWrapper(gym.Env):
    """
    Specific wrapper to transform staticASL scene inputs and outputs for navrep3d compatibility
    Removes extra actions (joint control, etc)
    Removes extra observations and stores them as episode info
    stores variables used by navrep3d utilities (pyglet rendering, env player, training callbacks)
    This should be the lowest level env, accessed as 'unwrapped' by all wrappers above.
    """
    def __init__(self, staticasl_env, build_name,
                 verbose=0, collect_statistics=True, debug_export_every_n_episodes=0,
                 difficulty_mode="progressive"):
        super().__init__()
        self.staticasl_env = staticasl_env
        # navrep3dtrainenv spaces
        MAX_VEL = 1. # m/s
        _H = 64
        _W = 64
        self.action_space = gym.spaces.Box(low=-MAX_VEL, high=MAX_VEL, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=255, shape=(_H, _W, 3), dtype=np.uint8),
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        ))
        # variables for rendering, callbacks, and player
        self.viewer = None
        self.last_action = np.array([0, 0, 0])
        self.last_image = None
        self.last_crowd = None
        self.last_walls = None
        self.last_odom = None
        self.last_lidar = None
        self.goal_xy = None
        self.reset_in_progress = False
        self.verbose = verbose
        self.total_steps = 0
        self.episode_statistics = DataFrame(
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
        self.scenario_name = "navrep3d" + build_name.replace("static", "")
        self.steps_since_reset = 0
        self.debug_export_every_n_episodes = debug_export_every_n_episodes
        self.total_episodes = 0
        self.difficulty_mode = difficulty_mode
        self.difficulty_to_set = None

    def set_difficulty(self, difficulty):
        """ difficulty [0, 1] """
        self.difficulty_to_set = difficulty

    def reset(self):
        self.episode_reward = 0
        self.steps_since_reset = 0
        self.total_steps += 1
        self.total_episodes += 1
        obs = self.staticasl_env.reset()
        self.last_image = obs['CameraSensor']
        self.goal_xy = obs['VectorSensor_size6'][:2]
        self.current_scenario = obs['VectorSensor_size6'][5]
        self.last_odom = np.array([0, 0, 0, 0, 0, 0, 0])
        self.last_action = np.zeros((3,)) # hack which allows encodedenv wrapper to get last action
        obs_tuple = (
            obs['CameraSensor'],
            obs['VectorSensor_size6'][:5]
        )
        max_dif = 1.0
        min_dif = 0.001 # 0 is interpreted as "no change"
        if self.difficulty_mode == "progressive":
            pass
        elif self.difficulty_mode == "random":
            self.set_difficulty(np.random.uniform())
        elif self.difficulty_mode == "bimodal":
            target_difficulty = min_dif
            if self.scenario_name == "navrep3dkozehd":
                max_dif = 0.4
            if np.random.uniform() > 0.5:
                target_difficulty = np.random.uniform(low=min_dif, high=max_dif)
            self.set_difficulty(target_difficulty)
        elif self.difficulty_mode == "easiest":
            self.set_difficulty(min_dif)
        elif self.difficulty_mode == "easier":
            self.set_difficulty(0.06 * max_dif)
        elif self.difficulty_mode == "easy":
            self.set_difficulty(0.2 * max_dif)
        elif self.difficulty_mode == "medium":
            self.set_difficulty(0.5 * max_dif)
        elif self.difficulty_mode == "hardest":
            self.set_difficulty(max_dif)
        else:
            raise NotImplementedError
        return obs_tuple

    def step(self, action):
        # the env player passes a len 3 action, but our environment expects action_space
        action_corrected = np.zeros(self.staticasl_env.action_space.shape)
        action_corrected[:3] = action
        if self.difficulty_to_set is not None:
            action_corrected[4] = self.difficulty_to_set
            self.difficulty_to_set = None
        obs, reward, done, info = self.staticasl_env.step(action_corrected)
        self.total_steps += 1
        self.steps_since_reset += 1
        self.episode_reward += reward
        self.last_image = obs['CameraSensor']
        self.goal_xy = obs['VectorSensor_size6'][:2]
        self.last_odom = np.array([0, 0, 0, 0, 0, 0, 0])
        self.last_action = action
        self.last_action = action # hack which allows encodedenv wrapper to get last action
        self.current_scenario = obs['VectorSensor_size6'][5]
        timeout = self.steps_since_reset > int(180. / 0.2) # TODO: do inside unity instead!
        if timeout:
            done = True
        if done:
            info["episode_scenario"] = self.current_scenario
            goal_is_reached = reward > 50.0
            self.episode_statistics.loc[len(self.episode_statistics)] = [
                self.total_steps,
                self.scenario_name,
                np.nan,
                self.steps_since_reset,
                goal_is_reached,
                self.episode_reward,
                self.current_scenario,
                self.current_scenario, # hack: num_walls is used to plot difficulty but for this env is fixed
                time.time(),
            ]
            info["event"] = None
            if np.allclose(reward, -0.01): # glitched (probably collision)
                info["event"] = Collision()
            if np.allclose(reward, -0.02): # toppled
                info["event"] = Collision()
            if np.allclose(reward, -0.03): # collision with object
                info["event"] = Collision()
            if np.allclose(reward, -0.04): # collision with person
                info["event"] = CollisionOtherAgent()
            if goal_is_reached:
                info["event"] = ReachGoal()
            if timeout:
                info["event"] = Timeout()
        # export episode frames for debugging
        if self.debug_export_every_n_episodes > 0:
            print("{} {}".format(self.total_steps, self.total_episodes), end="\r", flush=True)
            if self.total_episodes % self.debug_export_every_n_episodes == 0:
                self.render(save_to_file=True)
        obs_tuple = (
            obs['CameraSensor'],
            obs['VectorSensor_size6'][:5]
        )
        return obs_tuple, reward, done, info

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
            def make_circle(c, r, res=10):
                thetas = np.linspace(0, 2*np.pi, res+1)[:-1]
                verts = np.zeros((res, 2))
                verts[:,0] = c[0] + r * np.cos(thetas)
                verts[:,1] = c[1] + r * np.sin(thetas)
                return verts
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
            # Action
            if mode == "image_only":
                frnt, left, rght = (False, False, False)
                if self.last_action is not None:
                    if self.last_action[0] > 0.05:
                        frnt = True
                    if self.last_action[2] > 0.05:
                        left = True
                    if self.last_action[2] < -0.05:
                        rght = True
                offsize = 10
                offcolor = (0.6, 0.6, 0.8, 0.5)
                oncolor = (0.8, 0.2, 1., 1)
                bbcolor = (1., 1., 1., 1)
                # front
                center = (VP_W // 2, 20)
                color = offcolor
                size = offsize
                if frnt:
                    color = bbcolor
                    size = offsize * 1.2
                    gl.glBegin(gl.GL_TRIANGLES)
                    gl.glColor4f(color[0], color[1], color[2], color[3])
                    gl.glVertex3f(center[0], center[1]+size, 0)
                    gl.glVertex3f(center[0]-size, center[1]-size, 0)
                    gl.glVertex3f(center[0]+size, center[1]-size, 0)
                    gl.glEnd()
                    if False:
                        poly = make_circle((center[0], center[1]-0.3 * size), size+2)
                        gl.glBegin(gl.GL_POLYGON)
                        gl.glColor4f(color[0], color[1], color[2], color[3])
                        for vert in poly:
                            gl.glVertex3f(vert[0], vert[1], 0)
                        gl.glEnd()
                    size = offsize
                    color = oncolor
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(color[0], color[1], color[2], color[3])
                gl.glVertex3f(center[0], center[1]+size, 0)
                gl.glVertex3f(center[0]-size, center[1]-size, 0)
                gl.glVertex3f(center[0]+size, center[1]-size, 0)
                gl.glEnd()
                # left
                center = (VP_W // 2 - 40, 20)
                color = offcolor
                size = offsize
                if left:
                    color = bbcolor
                    size = offsize * 1.2
                    gl.glBegin(gl.GL_TRIANGLES)
                    gl.glColor4f(color[0], color[1], color[2], color[3])
                    gl.glVertex3f(center[0]-size, center[1], 0)
                    gl.glVertex3f(center[0]+size, center[1]-size, 0)
                    gl.glVertex3f(center[0]+size, center[1]+size, 0)
                    gl.glEnd()
                    color = oncolor
                    size = offsize
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(color[0], color[1], color[2], color[3])
                gl.glVertex3f(center[0]-size, center[1], 0)
                gl.glVertex3f(center[0]+size, center[1]-size, 0)
                gl.glVertex3f(center[0]+size, center[1]+size, 0)
                gl.glEnd()
                # right
                center = (VP_W // 2 + 40, 20)
                color = offcolor
                size = offsize
                if rght:
                    color = bbcolor
                    size = offsize * 1.2
                    gl.glBegin(gl.GL_TRIANGLES)
                    gl.glColor4f(color[0], color[1], color[2], color[3])
                    gl.glVertex3f(center[0]+size, center[1], 0)
                    gl.glVertex3f(center[0]-size, center[1]+size, 0)
                    gl.glVertex3f(center[0]-size, center[1]-size, 0)
                    gl.glEnd()
                    color = oncolor
                    size = offsize
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(color[0], color[1], color[2], color[3])
                gl.glVertex3f(center[0]+size, center[1], 0)
                gl.glVertex3f(center[0]-size, center[1]+size, 0)
                gl.glVertex3f(center[0]-size, center[1]-size, 0)
                gl.glEnd()
                self.score_label.text = ""
            else:
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
        return self.current_scenario

    def _get_dt(self):
        return 0.2

    def _get_viewer(self):
        return self.viewer

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        self.staticasl_env.close()

def NavRep3DStaticASLEnv(**kwargs): # using kwargs to respect NavRep3DTrainEnv signature
    """ Shorthand to create env made by stacking wrappers which is equivalent to NavRep3DTrainEnv,
    """
    build_name = kwargs.pop('build_name', "staticasl")
    unity_player_dir = kwargs.pop('unity_player_dir', DEFAULT_UNITY_EXE)
    start_with_random_rot = kwargs.pop('start_with_random_rot', True)
    port = kwargs.pop('port', 25001)
    collect_statistics = kwargs.pop('collect_statistics', True)
    debug_export_every_n_episodes = kwargs.pop('debug_export_every_n_episodes', 0)
    # these args are unityenv specific
    time_scale = kwargs.pop('time_scale', 20.0) # 20 is the value used when I run the default mlagents-learn
    seed = kwargs.pop('seed', 1)
    verbose = kwargs.pop('verbose', 0)
    difficulty_mode = kwargs.pop('difficulty_mode', "progressive")
    kwargs.pop('tolerate_corruption', 0)
    if kwargs:
        raise ValueError("Unexpected kwargs: {}".format(kwargs))
    if build_name not in MLAGENTS_BUILD_NAMES:
        raise ValueError
    if unity_player_dir is None:
        file_name = None
    else:
        download_binaries_if_not_found(unity_player_dir)
        file_name = os.path.join(unity_player_dir, build_name)
    if not start_with_random_rot:
        raise ValueError
    port_lock_handle = mark_port_use(port, True, auto_switch=True, process_info="staticasl")
    worker_id = port_lock_handle.port - 25001
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(file_name=file_name, seed=seed, worker_id=worker_id, side_channels=[channel])
    port_lock_handle.write(f"actual port {unity_env._port}\n")
    dt = 0.2 # we want unity to render one frame per RL timestep (otherwise animations get messed up)
    channel.set_configuration_parameters(time_scale=time_scale,
                                         capture_frame_rate=int(time_scale/dt))
    env = MLAgentsGymEnvWrapper(unity_env, port_lock_handle)
    env = StaticASLToNavRep3DEnvWrapper(env, build_name,
                                        verbose=verbose, collect_statistics=collect_statistics,
                                        debug_export_every_n_episodes=debug_export_every_n_episodes,
                                        difficulty_mode=difficulty_mode)
    return env

def NavRep3DStaticASLEnvDiscrete(**kwargs):
    """ Shorthand to create env made by stacking wrappers which is equivalent to NavRep3DTrainEnvDiscrete,
    used in the subprocvecenv initializer to run alongside navrep3dtrainenvs """
    env = NavRep3DStaticASLEnv(**kwargs)
    env = DiscreteActionWrapper(env)
    return env

def main(step_by_step=False, render_mode='human', difficulty_mode="progressive", build_name="staticasl"):
    from navrep.tools.envplayer import EnvPlayer
    np.set_printoptions(precision=1, suppress=True)
    env = NavRep3DStaticASLEnv(
        verbose=0, collect_statistics=True, build_name=build_name,
        debug_export_every_n_episodes=0, port=25004, difficulty_mode=difficulty_mode)
    player = EnvPlayer(env, render_mode, step_by_step)
    player.run()
    env.close()


if __name__ == "__main__":
    from strictfire import StrictFire
    StrictFire(main)
