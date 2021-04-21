#!/usr/bin/env python3
from sys import exit
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

import helpers
import socket_handler

_H = 36 # 120
_W = 36 # 160

MAX_VEL = 1. # m/s
FLOWN_OFF_VEL = 5. # m/s
OBSERVATION = "BAKED" # BAKED, TUPLE, IMGONLY, RSONLY

class NavRep3DEnv(gym.Env):
    def __init__(self, verbose=0, collect_statistics=True):
        # gym env definition
        super(NavRep3DEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-MAX_VEL, high=MAX_VEL, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(_H, _W, 3), dtype=np.uint8)
#         self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5,), dtype=np.float32)
        # this
        HOST = '127.0.0.1'
        PORT = 25001
        self.collect_statistics = collect_statistics
        self.verbose = verbose
        self.time_step = 0.2
        self.sleep_time = -1
        self.max_time = 180.0
        self.s = socket_handler.init_socket(HOST, PORT)
        # variables
        self.current_scenario = 0
        self.difficulty_increase = 0
        self.last_odom = None
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
        self.steps_since_reset = None
        self.episode_reward = None

        def handler(signal_received, frame):
            # Handle any cleanup here
            print('SIGINT or CTRL-C detected. Exiting gracefully')
            socket_handler.stop(self.s)
            exit(0)
#         signal(SIGINT, handler)

    def _get_dt(self):
        return self.time_step

    def _get_viewer(self):
        return self.viewer

    def reset(self):
        self.pub = {'clock': 0, 'vel_cmd': (0, 0, 0), 'sim_control': 'i'}

        # send a few packet to be sure it is launched
        for _ in range(5):
            to_send = helpers.publish_all(self.pub)
            _ = socket_handler.send_and_receive(self.s, to_send)
            self.pub = helpers.do_step(self.time_step, self.pub)
            time.sleep(self.time_step)

        if self.current_scenario is not None:
            if self.difficulty_increase == 1:
                if self.current_scenario >= 9:
                    socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.reset()))
                else:
                    socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.next()))
                    self.current_scenario += 1
                self.difficulty_increase = 0
            elif self.difficulty_increase == -1:
                if self.current_scenario == 0:
                    socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.reset()))
                else:
                    socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.previous()))
                    self.current_scenario -= 1
                self.difficulty_increase = 0
            else:
                # same scenario
                socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.reset()))
        else:
            self.current_scenario = 0

        # wait for sim to load new scenario
        time.sleep(1)

        # reset variables
        self.last_odom = None
        self.steps_since_reset = 0
        self.episode_reward = 0.

        # make sure scenario is loaded
        self.last_image = None
        done = False
        while self.last_image is None or done:
            if self.verbose > 0:
                print("Reset pre-load step")
            obs, _, done, _ = self.step([0, 0, 0])

        if self.verbose > 0:
            print("Scenario # " + str(self.current_scenario))

        # reset variables again (weird things may have happened in the meantime, screwing up logging)
        self.last_odom = None
        self.steps_since_reset = 0
        self.episode_reward = 0.

        return obs

    def step(self, actions):
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

#             print(dico)
        arrimg = None
        if dico["camera"] != 'JPG':
#                 jpgbytes = base64.decodestring(dico["camera"])
            jpgbytes = base64.b64decode(dico["camera"])
            img = Image.open(io.BytesIO(jpgbytes))
            arrimg = np.asarray(img)
        if arrimg is None:
            arrimg = np.zeros((_H, _W, 3), dtype=np.uint8)

        # do cool stuff here
#                 to_save = {k: dico[k] for k in ('clock', 'crowd', 'odom', 'report')}
#                 list_save.append(to_save)

        # @Fabien: how do I get the true goal?
        # avoid crashing if the odom message is corrupted
        goal_is_reached = False
        fallen_through_ground = False
        flown_off = False
        robotstate_obs = np.array([0, 0, 0, 0, 0], dtype=np.float32)
        progress = 0
        GOAL_XY = np.array([-7, 0])
        GOAL_RADIUS = 1.
        try:
            odom = helpers.get_odom(dico)
            # goal
            goal_dist = np.linalg.norm(GOAL_XY - odom[:2])
            goal_is_reached = (goal_dist < GOAL_RADIUS)
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
            if arrimg is not None:
                arrimg = np.copy(arrimg)
                arrimg[:5,0,0] = robotstate_obs
                arrimg[:5,0,1] = robotstate_obs
                arrimg[:5,0,2] = robotstate_obs
        except IndexError:
            print("Warning: odom message is corrupted")
        # @Fabien: how do I get crowd velocities?
#         crowd = helpers.get_crowd(dico)

        # theta>0 in cmd_vel turns right in the simulator, usually it's the opposite.
        self.pub['vel_cmd'] = (actions[0], actions[1], np.rad2deg(actions[2]))

        # reward
        reward = 0

        done = False
        if not flown_off and not fallen_through_ground:
            reward = progress * 0.1

        # turn punishment on first episode
        if self.current_scenario == 0:
            reward += -0.1 * abs(actions[2])

        # checking ending conditions
        if "clock" in dico:
            if float(dico["clock"]) > self.max_time:
                if self.verbose > 0:
                    print("Time limit reached")
                done = True

        if fallen_through_ground:
            if self.verbose > 0:
                print("Fallen through ground")
            done = True
            self.difficulty_increase = -1

        if flown_off:
            if self.verbose > 0:
                print("Flown off! (progress: {})".format(progress))
            done = True
            self.difficulty_increase = -1

        if goal_is_reached:
            if self.verbose > 0:
                print("Goal reached")
            done = True
            reward = 100
            self.difficulty_increase = 1

        # log reward
        self.episode_reward += reward

        if self.episode_reward >= 200 or self.episode_reward <= -200:
            raise ValueError("odom: {}, last_odom:{}, progress: {}".format(odom, self.last_odom, progress))

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
        if done and self.steps_since_reset > 2:
            if self.collect_statistics:
                self.episode_statistics.loc[len(self.episode_statistics)] = [
                    self.total_steps,
                    'navrep3dtrain',
                    np.nan,
                    self.steps_since_reset,
                    goal_is_reached,
                    self.episode_reward,
                    np.clip(self.current_scenario, 0, 5),
                    np.clip(self.current_scenario*2, 0, 20),
                    time.time(),
                ]

        self.last_image = arrimg
        obs = arrimg
        if False: # DEBUG
            print(obs)
            # detect fd up situation after reset
            if np.allclose(actions, np.array([0,0,0])) and np.any(np.abs(obs) > 100.):
                print("WTF")
        return obs, reward, done, {}

    def close(self):
        socket_handler.stop(self.s)
        time.sleep(1)

    def render(self, mode='human', close=False,
               save_to_file=False):
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
        elif mode == 'human':
            # Window and viewport size
            WINDOW_W = width
            WINDOW_H = height
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
            imageData.blit(0,0)
            # Text
            self.score_label.text = "R {:.1f}".format(self.episode_reward)
            self.score_label.draw()
            win.flip()
            if save_to_file:
                pyglet.image.get_buffer_manager().get_color_buffer().save(
                    "/tmp/navrep3denv{:05}.png".format(self.total_steps))

        if self.verbose > 1:
            toc = timer()
            print("Render (display): {} Hz".format(1. / (toc - tic)))

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

def check_stablebaselines_compat(env):
    from stable_baselines.common.env_checker import check_env
    check_env(env)


if __name__ == "__main__":
    np.set_printoptions(precision=1, suppress=True)
    env = NavRep3DEnv(verbose=1)
#     check_stablebaselines_compat(env)
#     debug_env_max_speed(env)
    player = EnvPlayer(env)
    player.run()
