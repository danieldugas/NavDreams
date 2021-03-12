#!/usr/bin/env python3
from sys import exit
from tqdm import tqdm
import time
import numpy as np
from timeit import default_timer as timer
from navrep.tools.envplayer import EnvPlayer
import gym
import base64
from PIL import Image
import io

import helpers
import socket_handler

class NavRep3DEnv(gym.Env):
    def __init__(self, silent=False):
        # gym env definition
        super(NavRep3DEnv, self).__init__()
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(120, 160, 3), dtype=np.uint8)
        # this
        HOST = '127.0.0.1'
        PORT = 25001
        self.silent = silent
        self.time_step = 0.2
        self.sleep_time = -1
        self.max_time = 180.0
        self.s = socket_handler.init_socket(HOST, PORT)
        self.viewer = None
        self.current_scenario = 0
        self.increase_difficulty = False
        self.total_steps = 0
        self.last_odom = None
        def handler(signal_received, frame):
            # Handle any cleanup here
            print('SIGINT or CTRL-C detected. Exiting gracefully')
            socket_handler.stop(self.s)
            exit(0)
#         signal(SIGINT, handler)
        if not self.silent:
            print("Running simulation")

    def _get_dt(self):
        return self.time_step

    def _get_viewer(self):
        return self.viewer

    def reset(self):
        if not self.silent:
            print("Scenario # " + str(self.current_scenario))
        self.pub = {'clock': 0, 'vel_cmd': (0, 0, 0), 'sim_control': 'i'}

        # send a few packet to be sure it is launched
        for _ in range(5):
            to_send = helpers.publish_all(self.pub)
            _ = socket_handler.send_and_receive(self.s, to_send)
            self.pub = helpers.do_step(self.time_step, self.pub)
            time.sleep(self.time_step)

        if self.current_scenario is not None:
            if self.increase_difficulty:
                # next scenario!
                socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.next()))
                self.current_scenario += 1
                self.increase_difficulty = False
            else:
                # same scenario
                socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.reset()))
        else:
            self.current_scenario = 0

        # wait for sim to load new scenario
        time.sleep(1)

        # make sure scenario is loaded
        self.last_image = None
        while self.last_image is None:
            if not self.silent:
                print("Reset pre-load step")
            obs, _, _, _ = self.step([0, 0, 0])

        return obs

    def step(self, actions):
        if not self.silent:
            print("Step: ...")
        self.total_steps += 1
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
            arrimg = np.zeros(self.observation_space.shape, dtype=np.uint8)
        self.last_image = arrimg

        # do cool stuff here
#                 to_save = {k: dico[k] for k in ('clock', 'crowd', 'odom', 'report')}
#                 list_save.append(to_save)

        # @Fabien: how do I get the true goal?
        # avoid crashing if the odom message is corrupted
        goal_is_reached = False
        progress = 0
        try:
            odom = helpers.get_odom(dico)
            # goal
            goal_is_reached = odom[0] <= 0
            # progress
            if self.last_odom is not None:
                progress = self.last_odom[0] - odom[0]
            self.last_odom = odom
        except IndexError:
            print("Warning: odom message is corrupted")
        # @Fabien: how do I get crowd velocities?
#         crowd = helpers.get_crowd(dico)

        # theta>0 in cmd_vel turns right in the simulator, usually it's the opposite.
        self.pub['vel_cmd'] = (actions[0], actions[1], np.rad2deg(actions[2]))

        done = False
        reward = progress * 0.01
        # checking ending conditions
        if "clock" in dico:
            if float(dico["clock"]) > self.max_time:
                if not self.silent:
                    print("Time limit reached")
                done = True

        if goal_is_reached:
            if not self.silent:
                print("Goal reached")
            done = True
            reward = 100
            self.increase_difficulty = True

        # Debug: This skips the first test, remove
        if self.current_scenario == 0:
            if not self.silent:
                print("Skipping first scenario")
            done = True
            self.increase_difficulty = True

        # doing a step
        self.pub = helpers.do_step(self.time_step, self.pub)

        time_out = time.time()

        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        elif time_out < time_in + self.time_step:
            pass
#             time.sleep(time_in + self.time_step - time_out)

#                 recorder.save_dico('/tmp/recorder/tests_'+str(i), list_save)

        if not self.silent:
            toc = timer()
            print("Step: {} Hz".format(1. / (toc - tic)))
            print("Clock: {}".format(dico["clock"]))

        return arrimg, reward, done, {}

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

        if not self.silent:
            toc = timer()
            print("Render (fetch): {} Hz".format(1. / (toc - tic)))
            tic = timer()

        if mode == 'matplotlib':
            from matplotlib import pyplot as plt
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
            self.score_label.text = ""
            self.score_label.draw()
            win.flip()
            if save_to_file:
                pyglet.image.get_buffer_manager().get_color_buffer().save(
                    "/tmp/navrep3denv{:05}.png".format(self.total_steps))

        if not self.silent:
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
    env = NavRep3DEnv(silent=True)
#     check_stablebaselines_compat(env)
#     debug_env_max_speed(env)
    player = EnvPlayer(env)
    player.run()
