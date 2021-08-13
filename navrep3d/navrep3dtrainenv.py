#!/usr/bin/env python3
import os
from sys import exit
import traceback
from tqdm import tqdm
import time
import numpy as np
from timeit import default_timer as timer
from navrep.tools.envplayer import EnvPlayer
from pose2d import apply_tf_to_vel, inverse_pose2d, apply_tf_to_pose
from pandas import DataFrame
import gym
import base64
from PIL import Image
import io
from CMap2D import CMap2D
import subprocess
from fire import Fire

import navrep3d.helpers as helpers
import navrep3d.socket_handler as socket_handler

_H = 64 # 36 # 120
_W = 64 # 36 # 160

MAX_VEL = 1. # m/s
FLOWN_OFF_VEL = 5. # m/s
OBSERVATION = "TUPLE" # BAKED, TUPLE, IMGONLY, RSONLY
rotation_deadzone = None # 0.1

# for now, goal is fixed
GOAL_XY = np.array([-7, 0])
GOAL_RADIUS = 0.5

ROBOT_RADIUS = 0.3
AGENT_RADIUS = 0.33

REBOOT_EVERY_N_EPISODES = 100

HOMEDIR = os.path.expanduser("~")
DEFAULT_UNITY_EXE = os.path.join(HOMEDIR, "Code/cbsim/navrep3d/LFS/executables")
# TODO: RELEASE - make a tool which downloads LFS files

class NavRep3DTrainEnv(gym.Env):
    def __init__(self, verbose=0, collect_statistics=True,
                 debug_export_every_n_episodes=0, port=25001,
                 unity_player_dir=DEFAULT_UNITY_EXE, build_name="./build.x86_64"):
        # gym env definition
        super(NavRep3DTrainEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-MAX_VEL, high=MAX_VEL, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Tuple((
            gym.spaces.Box(low=0, high=255, shape=(_H, _W, 3), dtype=np.uint8),
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        ))
        self.zero_action = np.array([0,0,0])
        # this
        HOST = '127.0.0.1'
        self.socket_host = HOST
        self.build_name = build_name
        self.socket_port = port
        self.collect_statistics = collect_statistics
        self.debug_export_every_n_episodes = debug_export_every_n_episodes
        self.verbose = verbose
        self.time_step = 0.2
        self.sleep_time = -1
        self.max_time = 180.0
        self.unity_player_dir = unity_player_dir
        self.output_lidar = False
        self.render_legs_in_lidar = True
        # variables
        self.difficulty_increase = 0
        self.last_odom = None
        self.last_crowd = None
        self.last_walls = None
        self.last_action = None
        self.reset_in_progress = False # necessary to differentiate reset pre-steps from normal steps
        self.unity_process = None
        # other tools
        self.viewer = None
        self.episode_statistics = None
        if self.collect_statistics:
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
        self.total_steps = 0
        self.total_episodes = 0
        self.steps_since_reset = None
        self.episode_reward = None
        self.converter_cmap2d = None

        self._reboot_unity()

        def handler(signal_received, frame):
            # Handle any cleanup here
            print('SIGINT or CTRL-C detected. Exiting gracefully')
            socket_handler.stop(self.s)
            exit(0)
#         signal(SIGINT, handler)

    def _reboot_unity(self):
        # close unity player and socket
        if self.unity_process is not None:
            socket_handler.stop(self.s)
            self.unity_process.wait()
        # start unity player and connect
        if self.unity_player_dir is not None:
            self.unity_process = subprocess.Popen([self.build_name, "-port", str(self.socket_port)],
                                                  cwd=self.unity_player_dir,
                                                  )
            time.sleep(5.0)
        self.s = socket_handler.init_socket(self.socket_host, self.socket_port)

    def _get_dt(self):
        return self.time_step

    def _get_viewer(self):
        return self.viewer

    def reset(self):
        if REBOOT_EVERY_N_EPISODES > 0 and self.total_episodes > 0:
            if self.total_episodes % REBOOT_EVERY_N_EPISODES == 0:
                self._reboot_unity()
        # change scenario if necessary
        if self.verbose > 0:
            print("Scenario # {} complete.".format(self.infer_current_scenario()))
        if self.difficulty_increase == 1:
            if self.infer_current_scenario() < 9:
                socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.next()))
            else:
                socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.reset()))
            self.difficulty_increase = 0
        elif self.difficulty_increase == -1:
            if self.infer_current_scenario() > 0:
                socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.previous()))
            else:
                socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.reset()))
            self.difficulty_increase = 0
        else:
            # same scenario
            socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.reset()))

        self.reset_in_progress = True
        i = 0
        while True:
            i += 1
            # reset sim
            self.pub = {'clock': 0, 'vel_cmd': (0, 0, 0), 'sim_control': 'i'}

            # send a few packet to be sure it is launched
            for _ in range(5):
                to_send = helpers.publish_all(self.pub)
                _ = socket_handler.send_and_receive(self.s, to_send)
                self.pub = helpers.do_step(self.time_step, self.pub)
                time.sleep(self.time_step)

            # wait for sim to load new scenario
            time.sleep(1)

            # reset variables
            self.last_odom = None
            self.last_walls = None
            self.last_map = None
            self.last_crowd = None
            self.last_sdf = None
            self.steps_since_reset = 0
            self.episode_reward = 0.
            self.distances_travelled_in_base_frame = {}
            self.flat_contours = None
            self.lidar_scan = None
            self.lidar_angles = None

            # double check that the new scenario is loaded correctly (doesn't return done or weirdness)
            self.last_image = None
            done = False
            if self.verbose > 0:
                print("Reset pre-load step")
            obs, _, done, _ = self.step(self.zero_action)
            if done:
                if self.verbose > 0:
                    print("sending reset signal")
                # we want to make sure the next step won't result in "done" being True
                socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.reset()))
                time.sleep(1)
                self.pub = {'clock': 0, 'vel_cmd': (0, 0, 0), 'sim_control': 'i'}
                self.episode_reward = 0.
                continue
            if self.last_image is None:
                continue
            break
        self.reset_in_progress = False

        if self.verbose > 0:
            print("Started scenario # " + str(self.infer_current_scenario()))

        # reset variables again (weird things may have happened in the meantime, screwing up logging)
        self.last_odom = None
        self.last_walls = None
        self.last_map = None
        self.last_crowd = None
        self.last_sdf = None
        self.steps_since_reset = 0
        self.episode_reward = 0.
        self.distances_travelled_in_base_frame = {}
        self.flat_contours = None
        self.lidar_scan = None
        self.lidar_angles = None

        return obs

    def step(self, actions):
        self.last_action = actions
        actions = np.array(actions)
        if rotation_deadzone is not None:
            actions[2] = 0. if abs(actions[2]) < rotation_deadzone else (
                (actions[2] - np.sign(actions[2]) * rotation_deadzone) / (1. - rotation_deadzone))
        if self.verbose > 1:
            print("Step: ...")
        self.total_steps += 1
        self.steps_since_reset += 1
        tic = timer()

        time_in = time.time()
        # making the raw string to send from the dict
        to_send = helpers.publish_all(self.pub)
        # sending and receiving raw data
        raw = socket_handler.send_and_receive(self.s, to_send)
        # getting dict from raw data
        dico = helpers.raw_data_to_dict(raw)

        try:
            self.last_walls = helpers.get_walls(dico)
        except Exception as e: # noqa
            print(dico["walls"])
            traceback.print_exc()
            self.last_walls = None

        crowd = None
        crowd_vel = None
        try:
            crowd = helpers.get_crowd(dico)
            crowd_vel = self._infer_crowd_vel(crowd, self.last_crowd)
            self.last_crowd = crowd
        except Exception as e: # noqa
            traceback.print_exc()
            self.last_crowd = None

        try:
            odom = helpers.get_odom(dico)
            x, y, th, vx, vy, vth, z = odom
        except IndexError:
            traceback.print_exc()
            print("Warning: odom message is corrupted")

#             print(dico)
        arrimg = None
        if dico["camera"] != '':
#                 jpgbytes = base64.decodestring(dico["camera"])
            jpgbytes = base64.b64decode(dico["camera"])
            img = Image.open(io.BytesIO(jpgbytes))
            arrimg = np.asarray(img)
        if arrimg is None:
            print("Warning: image message is corrupted")
            arrimg = np.zeros((_H, _W, 3), dtype=np.uint8)

        if self.output_lidar:
            self.raytrace_lidar(odom[:3], self.last_walls, crowd, crowd_vel)

        # do cool stuff here
#                 to_save = {k: dico[k] for k in ('clock', 'crowd', 'odom', 'report')}
#                 list_save.append(to_save)

        # @Fabien: how do I get the true goal?
        # avoid crashing if the odom message is corrupted
        difficulty_increase = None
        goal_is_reached = False
        fallen_through_ground = False
        flown_off = False
        colliding_object = False
        colliding_crowd = False
        robotstate_obs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        progress = 0

        # goal
        goal_dist = np.linalg.norm(GOAL_XY - odom[:2])
        goal_is_reached = (goal_dist < (GOAL_RADIUS + ROBOT_RADIUS))
        # progress
        if self.last_odom is not None:
            last_goal_dist = np.linalg.norm(GOAL_XY - self.last_odom[:2])
            progress = last_goal_dist - goal_dist
            if abs(progress) > 10:
                flown_off = True
        self.last_odom = odom
        # checks
        if np.linalg.norm(odom[3:5]) >= FLOWN_OFF_VEL:
            flown_off = True
        if odom[-1] < 0:
            fallen_through_ground = True
        # collision
        if self.last_walls is not None:
            colliding_object = self.check_wall_collisions(odom, self.last_walls)
        # agent distances
        if crowd is not None:
            if len(crowd) > 0:
                mindist = np.min(np.linalg.norm(crowd[:,1:3] - odom[:2][None, :], axis=-1))
                if mindist < (AGENT_RADIUS + ROBOT_RADIUS):
                    colliding_crowd = True
        # robotstate obs
        # shape (n_agents, 5 [grx, gry, vx, vy, vtheta]) - all in base frame
        baselink_in_world = odom[:3]
        world_in_baselink = inverse_pose2d(baselink_in_world)
        robotvel_in_world = odom[3:6]  # TODO: actual robot rot vel?
        robotvel_in_baselink = apply_tf_to_vel(robotvel_in_world, world_in_baselink)
        goal_in_world = np.array([GOAL_XY[0], GOAL_XY[1], 0])
        goal_in_baselink = apply_tf_to_pose(goal_in_world, world_in_baselink)
        robotstate_obs = np.hstack([goal_in_baselink[:2], robotvel_in_baselink])
        # bake robotstate into image state
        if False:
            arrimg = np.copy(arrimg)
            arrimg[:5,0,0] = robotstate_obs
            arrimg[:5,0,1] = robotstate_obs
            arrimg[:5,0,2] = robotstate_obs

        # theta>0 in cmd_vel turns right in the simulator, usually it's the opposite.
        self.pub['vel_cmd'] = (actions[0], actions[1], np.rad2deg(actions[2]))

        # reward
        reward = 0

        done = False
        if not flown_off and not fallen_through_ground:
            reward = progress * 0.1

        # turn punishment on first episode
        if self.infer_current_scenario() == 0:
            reward += -0.1 * abs(actions[2])

        # checking ending conditions
        if "clock" in dico:
            if float(dico["clock"]) > self.max_time:
                if self.verbose > 0:
                    print("Time limit reached")
                done = True

        if colliding_object:
            if self.verbose > 0:
                print("Colliding static obstacle")
            done = True
            reward = -25
            difficulty_increase = -1

        if colliding_crowd:
            if self.verbose > 0:
                print("Colliding agent")
            done = True
            reward = -25
            difficulty_increase = -1

        if fallen_through_ground:
            if self.verbose > 0:
                print("Fallen through ground")
            done = True
            reward = -25
            difficulty_increase = -1

        if flown_off:
            if self.verbose > 0:
                print("Flown off! (progress: {})".format(progress))
            done = True
            reward = -25
            difficulty_increase = -1

        if goal_is_reached:
            if self.verbose > 0:
                print("Goal reached")
            done = True
            reward = 100
            difficulty_increase = 1

        # log reward
        self.episode_reward += reward

        # doing a step
        self.pub = helpers.do_step(self.time_step, self.pub)

        time_out = time.time()

        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        elif time_out < time_in + self.time_step:
            pass
#             time.sleep(time_in + self.time_step - time_out)

#                 recorder.save_dico('/tmp/recorder/tests_'+str(i), list_save)

        if self.verbose > 1:
            toc = timer()
            print("Step: {} Hz".format(1. / (toc - tic)))
            print("Clock: {}".format(dico["clock"]))

        # log data
        if done and not self.reset_in_progress:
            if self.episode_reward >= 200 or self.episode_reward <= -200:
                raise ValueError("odom: {}, last_odom:{}, progress: {}".format(
                    odom, self.last_odom, progress))
            self.total_episodes += 1
            if self.collect_statistics:
                self.episode_statistics.loc[len(self.episode_statistics)] = [
                    self.total_steps,
                    'navrep3dtrain',
                    np.nan,
                    self.steps_since_reset,
                    goal_is_reached,
                    self.episode_reward,
                    np.clip(self.infer_current_scenario(), 0, 5),
                    np.clip(self.infer_current_scenario()*2, 0, 20),
                    time.time(),
                ]
            if difficulty_increase is not None:
                self.difficulty_increase = difficulty_increase

        # export episode frames for debugging
        if self.debug_export_every_n_episodes > 0:
            print("{} {}".format(self.total_steps, self.total_episodes), end="\r", flush=True)
            if self.total_episodes % self.debug_export_every_n_episodes == 0:
                self.render(save_to_file=True)

        self.last_image = arrimg
        obs = (arrimg, robotstate_obs)
        return obs, reward, done, {}

    def close(self):
        socket_handler.stop(self.s)
        time.sleep(1)

    def render(self, mode='human', close=False, save_to_file=False):
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
                topdown_in_vp.set_rotation(-np.pi/2.)
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
                if self.lidar_scan is not None and self.last_odom is not None:
                    px, py, angle, _, _, _, _ = self.last_odom
                    # LIDAR rays
                    scan = self.lidar_scan
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
                xgoal, ygoal = GOAL_XY
                r = GOAL_RADIUS
                gl.glBegin(gl.GL_TRIANGLES)
                gl.glColor4f(goalcolor[0], goalcolor[1], goalcolor[2], 1)
                triangle = make_circle((xgoal, ygoal), r, res=3)
                for vert in triangle:
                    gl.glVertex3f(vert[0], vert[1], 0)
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

    def _infer_crowd_vel(self, crowd, last_crowd):
        """ infer vx, vy, vtheta for crowd in sim frame """
        if crowd is None:
            return None
        crowd_vel = np.zeros_like(crowd)
        # for each human, get vel in base frame
        crowd_vel[:, 0] = crowd[:, 0] # ids
        # dig up rotational velocity from past states log
        if last_crowd is None:
            crowd_vel[:, 1] = 0 # vx
            crowd_vel[:, 2] = 0 # vy
            crowd_vel[:, 3] = 0 # vth
        else:
            # paranoid check that ids are matching
            for (id_, x, y, th), (last_id, last_x, last_y, last_th) in zip(crowd, last_crowd):
                if id_ != last_id:
                    print("Warning: changing crowd ids are not supported.")
            crowd_vel[:, 1] = (crowd[:, 1] - last_crowd[:, 1]) / self._get_dt() # vx
            crowd_vel[:, 2] = (crowd[:, 2] - last_crowd[:, 2]) / self._get_dt() # vy
            crowd_vel[:, 3] = (crowd[:, 3] - last_crowd[:, 3]) / self._get_dt() # vth
        return crowd_vel

    def _update_dist_travelled(self, crowd, crowd_vel):
        """ update dist travel var used for animating legs """
        if crowd is None:
            return
        # for each human, get vel in base frame
        for (id_, x, y, th), (_, vx, vy, vrot) in zip(crowd, crowd_vel):
            # transform world vel to base vel
            baselink_in_world = np.array([x, y, th])
            world_in_baselink = inverse_pose2d(baselink_in_world)
            vel_in_world_frame = np.array([vx, vy, vrot])
            vel_in_baselink_frame = apply_tf_to_vel(vel_in_world_frame, world_in_baselink)
            if id_ in self.distances_travelled_in_base_frame:
                self.distances_travelled_in_base_frame[id_] += vel_in_baselink_frame * self._get_dt()
            else:
                self.distances_travelled_in_base_frame[id_] = vel_in_baselink_frame * self._get_dt()

    def raytrace_lidar(self, robot_xytheta, contours, crowd, crowd_vel):
        from CMap2D import flatten_contours, render_contours_in_lidar, CMap2D, CSimAgent
        # inputs
        x, y, th = robot_xytheta
        # constants
        n_angles = 1080
        MAX_RANGE = 25.
        kLidarAngleIncrement = 0.00581718236208
        kLidarMergedMinAngle = 0
        kLidarMergedMaxAngle = 6.27543783188 + kLidarAngleIncrement
        # preprocessing if necessary
        self._update_dist_travelled(crowd, crowd_vel)
        if self.flat_contours is None and contours is not None:
            self.flat_contours = flatten_contours([list(polygon) for polygon in contours])
        if self.converter_cmap2d is None:
            self.converter_cmap2d = CMap2D()
            self.converter_cmap2d.set_resolution(1.)
        # self inputs
        flat_contours = self.flat_contours
        distances_travelled_in_base_frame = self.distances_travelled_in_base_frame
        converter_cmap2d = self.converter_cmap2d
        # raytrace
        lidar_pos = np.array([x, y, th], dtype=np.float32)
        ranges = np.ones((n_angles,), dtype=np.float32) * MAX_RANGE
        angles = np.linspace(kLidarMergedMinAngle,
                             kLidarMergedMaxAngle-kLidarAngleIncrement,
                             n_angles) + lidar_pos[2]
        if contours is not None:
            render_contours_in_lidar(ranges, angles, flat_contours, lidar_pos[:2])
        # agents
        if crowd is not None:
            other_agents = []
            for (a_id, a_x, a_y, a_th), (_, a_vx, a_vy, a_vrot) in zip(crowd, crowd_vel):
                pos = np.array([a_x, a_y, a_th], dtype=np.float32)
                if a_id in distances_travelled_in_base_frame:
                    dist = distances_travelled_in_base_frame[a_id].astype(np.float32)
                else:
                    raise ValueError("id not found in distances travelled")
                vel = np.array([a_vx, a_vy], dtype=np.float32)
                if self.render_legs_in_lidar:
                    agent = CSimAgent(pos, dist, vel)
                else:
                    agent = CSimAgent(pos, dist, vel, type_="trunk", radius=AGENT_RADIUS)
                other_agents.append(agent)
            # apply through converter map (res 1., origin 0,0 -> i,j == x,y)
            converter_cmap2d.render_agents_in_lidar(ranges, angles, other_agents, lidar_pos[:2])
        # output
        self.lidar_scan = ranges
        self.lidar_angles = angles

    def check_wall_collisions(self, odom, walls):
        if len(walls) == 0:
            return False
        if self.last_sdf is None:
            map_ = CMap2D()
            map_.from_closed_obst_vertices(walls, resolution=0.1)
            self.last_map = map_
            self.last_sdf = map_.as_sdf()
        # TODO: use SDF? probably faster
        ij = self.last_map.xy_to_ij(odom[:2][None, :], clip_if_outside=True)[0]
#         from matplotlib import pyplot as plt
#         from CMap2D import gridshow
#         plt.ion()
#         gridshow(self.last_sdf)
#         plt.scatter(ij[0], ij[1])
#         plt.pause(0.1)
        return self.last_sdf[ij[0], ij[1]] < ROBOT_RADIUS

    def infer_current_scenario(self):
        if self.last_walls is None:
            return -1
        return int((len(self.last_walls) - 4) / 2)

def check_running_unity_backends():
    from builtins import input
    if os.system('pgrep build.x86') == 0:
        print("Processes detected which might be unity players:")
        os.system('ps aux | grep build.x86') # display port numbers
        input("Are you sure you want to continue? (Ctrl-c: stop, Enter: continue)")

def debug_env_max_speed(env, render=False):
    env.reset()
    n_episodes = 0
    for i in tqdm(range(100000)):
        _,_,done,_ = env.step(np.random.uniform(size=(3,)))
        if i % 10 == 0 and render:
            env.render()
        if done:
            env.reset()
            n_episodes += 1
    env.close()

class NavRep3DTrainEnvDiscrete(NavRep3DTrainEnv):
    def __init__(self, **kwargs):
        super(NavRep3DTrainEnvDiscrete, self).__init__(**kwargs)
        self.action_space = gym.spaces.Discrete(3)
        self.zero_action = 0

    def step(self, actions):
        """ actions
        0: forward
        1: left
        2: right
        """
        if actions is None:
            actions = 1
        if actions == 0:
            cont_actions = np.array([1, 0, 0])
        elif actions == 1:
            cont_actions = np.array([0, 0, 0.5])
        elif actions == 2:
            cont_actions = np.array([0, 0,-0.5])
        return super(NavRep3DTrainEnvDiscrete, self).step(cont_actions)

def check_stablebaselines_compat(env):
    from stable_baselines.common.env_checker import check_env
    check_env(env)

# separate main function to define the script-relevant arguments used by Fire
def main(
    # NavRep3DTrainEnv args
    verbose=1, collect_statistics=True, debug_export_every_n_episodes=0, port=25001,
    unity_player_dir=DEFAULT_UNITY_EXE, build_name="./build.x86_64",
    # Player args
    render_mode='human', step_by_step=False,
    # Task args
    check_compat=False, profile=False,
):
    np.set_printoptions(precision=1, suppress=True)
    env = NavRep3DTrainEnv(verbose, collect_statistics, debug_export_every_n_episodes, port,
                           unity_player_dir, build_name)
    if check_compat:
        check_stablebaselines_compat(env)
    elif profile:
        debug_env_max_speed(env)
    else:
        player = EnvPlayer(env, render_mode, step_by_step)
        player.run()


if __name__ == "__main__":
    Fire(main)
