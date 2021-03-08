#!/usr/bin/env python3
from signal import signal, SIGINT
from sys import exit
import time
import numpy as np
from timeit import default_timer as timer
from navrep.tools.envplayer import EnvPlayer

import helpers
import socket_handler
from rvo import RVONavigationPlanner

class NavRep3DEnv(object):
    def __init__(self):
        HOST = '127.0.0.1'
        PORT = 25001
        self.time_step = 0.1
        self.sleep_time = -1
        self.s = socket_handler.init_socket(HOST, PORT)
        self.last_cmd_vel = (0, 0)
        self.viewer = None
        self.current_scenario = None
        self.total_steps = 0
        def handler(signal_received, frame):
            # Handle any cleanup here
            print('SIGINT or CTRL-C detected. Exiting gracefully')
            socket_handler.stop(self.s)
            exit(0)
        signal(SIGINT, handler)
        print("Running simulation")

    def _get_dt(self):
        return self.time_step

    def _get_viewer(self):
        return self.viewer

    def reset(self):
        print("Scenario # " + str(self.current_scenario))
        self.pub = {'clock': 0, 'vel_cmd': (0, 0), 'sim_control': 'i'}

        # send a few packet to be sure it is launched
        for _ in range(5):
            to_send = helpers.publish_all(self.pub)
            _ = socket_handler.send_and_receive(self.s, to_send)
            self.pub = helpers.do_step(self.time_step, self.pub)
            time.sleep(self.time_step)

        if self.current_scenario is not None:
            # next scenario!
            socket_handler.send_and_receive(self.s, helpers.publish_all(helpers.next()))

            # wait for sim to load new scenario
            time.sleep(1)
            self.current_scenario += 1
        else:
            self.current_scenario = 0

        obs, _, _, _ = self.step([0, 0, 0])
        return obs

    def step(self, actions):
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
        import base64
        from PIL import Image
        import io
        if dico["camera"] != 'JPG':
#                 jpgbytes = base64.decodestring(dico["camera"])
            jpgbytes = base64.b64decode(dico["camera"])
            img = Image.open(io.BytesIO(jpgbytes))
            arrimg = np.asarray(img)
        self.last_image = arrimg

        # do cool stuff here
#                 to_save = {k: dico[k] for k in ('clock', 'crowd', 'odom', 'report')}
#                 list_save.append(to_save)

        # @Fabien: how do I get the true goal?
        odom = helpers.get_odom(dico)
        goal_is_reached = False
        # avoid crashing if the odom message is corrupted
        try:
            goal_is_reached = odom[0] <= 0
        except IndexError:
            print("Warning: odom message is corrupted")
        # @Fabien: how do I get crowd velocities?
#         crowd = helpers.get_crowd(dico)
        # @Fabien: odom velocities are 0, should be higher
#         x, y, th, _, _, _ = odom
#         speed, rot = self.last_cmd_vel
#         odom[3] = speed * np.cos(th)
#         odom[4] = speed * np.sin(th)
#         odom[5] = rot

        speed = actions[0]
        rot = actions[2]
        self.last_cmd_vel = (speed, rot)

        # theta>0 in cmd_vel turns right in the simulator, usually it's the opposite.
        self.pub['vel_cmd'] = (speed, np.rad2deg(rot))

        done = False
        reward = 0
        # checking ending conditions
        if helpers.check_ending_conditions(180.0, -20, dico):
            done = True

        if goal_is_reached:
            done = True
            reward = 100

        # Debug: This skips the first test, remove
        if self.current_scenario == 0:
            done = True

        # doing a step
        self.pub = helpers.do_step(self.time_step, self.pub)

        time_out = time.time()

        if self.sleep_time > 0:
            time.sleep(self.sleep_time)
        elif time_out < time_in + self.time_step:
            pass
#             time.sleep(time_in + self.time_step - time_out)

#                 recorder.save_dico('/tmp/recorder/tests_'+str(i), list_save)

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

        toc = timer()
        print("Render (display): {} Hz".format(1. / (toc - tic)))

def debug_env_max_speed(env):
    env.reset()
    for i in range(10000):
        try:
            _,_,done,_ = env.step(np.array([1, 0, 0]))
        except IndexError:
            pass
        if i % 10 == 0:
            env.render()
        if done:
            env.reset()
    env.close()


if __name__ == "__main__":
    env = NavRep3DEnv()
#     debug_env_max_speed(env)
    player = EnvPlayer(env)
    player.run()
