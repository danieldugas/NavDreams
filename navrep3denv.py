#!/usr/bin/env python3
from signal import signal, SIGINT
from sys import exit
import time
import numpy as np
from timeit import default_timer as timer
from navrep.tools.envplayer import EnvPlayer

import helpers
import socket_handler
import robotnavigation
import recorder
import metrics
from rvo import RVONavigationPlanner
from dwa import DynamicWindowApproachNavigationPlanner

class NavRep3DEnv(object):
    def __init__(self):
        HOST='127.0.0.1'
        PORT=25001
        self.time_step=0.1
        self.sleep_time=-1
        self.s = socket_handler.init_socket(HOST, PORT)
        self.viewer = None
        self.current_scenario = None
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
                raw = socket_handler.send_and_receive(self.s, to_send)
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
            planner = RVONavigationPlanner()
            # @Fabien: how do I get static obstacles as polygons?
            planner.set_static_obstacles([])

            last_cmd_vel = (0, 0)
            while True:
                time_in = time.time()
                # making the raw string to send from the dict
                to_send = helpers.publish_all(self.pub)
                # sending and receiving raw data
                raw = socket_handler.send_and_receive(self.s, to_send)
                # getting dict from raw data
                dico = helpers.raw_data_to_dict(raw)

    #             print(dico)
                import base64
                if dico["camera"] != 'JPG':
    #                 jpgbytes = base64.decodestring(dico["camera"])
                    jpgbytes = base64.b64decode(dico["camera"])

                from PIL import Image
                import io
                img = Image.open(io.BytesIO(jpgbytes))
                arrimg = np.asarray(img)
                self.last_image = arrimg

                # do cool stuff here
#                 to_save = {k: dico[k] for k in ('clock', 'crowd', 'odom', 'report')}
#                 list_save.append(to_save)

                # @Fabien: how do I get the true goal?
                goal = np.array(helpers.get_odom(dico)[:2])
                goal[0] = 0.
                # @Fabien: how do I get crowd velocities?
                crowd = helpers.get_crowd(dico)
                # @Fabien: odom velocities are 0, should be higher
                odom = helpers.get_odom(dico)
                x, y, th, _, _, _ = odom
                speed, rot = last_cmd_vel
                odom[3] = speed * np.cos(th)
                odom[4] = speed * np.sin(th)
                odom[5] = rot

                tic = timer()
                cmd_vel = planner.compute_cmd_vel(
                    crowd,
                    odom,
                    goal,
                    show_plot=False,
                )
                toc = timer()
                print("{}Hz".format(1./(toc-tic)))
                last_cmd_vel = cmd_vel

                speed, rot = cmd_vel
                speed = actions[0]
                rot = actions[2]

                # theta>0 in cmd_vel turns right in the simulator, usually it's the opposite.
                self.pub['vel_cmd'] = (speed, np.rad2deg(rot))

                done = False
                reward = 0
                # checking ending conditions
                if helpers.check_ending_conditions(180.0, -20, dico):
                    done = True

                if helpers.get_odom(dico)[0] <= goal[0]:
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
                    time.sleep(time_in + self.time_step - time_out)

#                 recorder.save_dico('/tmp/recorder/tests_'+str(i), list_save)

                return arrimg, reward, done, {}

    def close(self):
        socket_handler.stop(self.s)
        time.sleep(1)

    def render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
            return
        arrimg = self.last_image
        width = arrimg.shape[1]
        height = arrimg.shape[0]

#         from matplotlib import pyplot as plt
#         plt.imshow(arrimg)
#         plt.ion()
#         plt.show()
#         plt.pause(0.1)

        import pyglet
        from pyglet.gl import GLubyte
        pixels = arrimg[::-1,:,:].flatten()
        rawData = (GLubyte * len(pixels))(*pixels)
        imageData = pyglet.image.ImageData(width, height, 'RGB', rawData)

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


if __name__ == "__main__":
    env = NavRep3DEnv()
    player = EnvPlayer(env)
    player.run()


