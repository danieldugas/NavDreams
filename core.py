#!/usr/bin/env python3
from signal import signal, SIGINT
from sys import exit
import time
import numpy as np
from timeit import default_timer as timer

import helpers
import socket_handler
import robotnavigation
import recorder
import metrics
from rvo import RVONavigationPlanner
from dwa import DynamicWindowApproachNavigationPlanner


# @Daniel: main function, if you have trouble with real time simulation, you can set time_step and sleep_time to bigger values
def run(HOST='127.0.0.1', PORT=25001, time_step=0.1, sleep_time=-1, planner_type="DWA"):
    s = socket_handler.init_socket(HOST, PORT)

    def handler(signal_received, frame):
        # Handle any cleanup here
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        socket_handler.stop(s)
        exit(0)
    signal(SIGINT, handler)

    print("Running simulation")

    # 21 scenarios
    for i in range(21):
        list_save = []
        print("Scenario # " + str(i))
        pub = {'clock': 0, 'vel_cmd': (0, 0), 'sim_control': 'i'}

        # send a few packet to be sure it is launched
        for _ in range(5):
            to_send = helpers.publish_all(pub)
            raw = socket_handler.send_and_receive(s, to_send)
            pub = helpers.do_step(time_step, pub)
            time.sleep(time_step)

        if planner_type == "RVO":
            planner = RVONavigationPlanner()
        elif planner_type == "DWA":
            planner = DynamicWindowApproachNavigationPlanner()
        else:
            raise NotImplementedError
        # @Fabien: how do I get static obstacles as polygons?
        planner.set_static_obstacles([])

        last_cmd_vel = (0, 0)
        while True:
            time_in = time.time()
            # making the raw string to send from the dict
            to_send = helpers.publish_all(pub)
            # sending and receiving raw data
            raw = socket_handler.send_and_receive(s, to_send)
            # getting dict from raw data
            dico = helpers.raw_data_to_dict(raw)

#             print(dico)
            import base64
            if dico["camera"] != 'JPG':
                jpgbytes = base64.decodestring(dico["camera"])

            from PIL import Image
            import io
            img = Image.open(io.BytesIO(jpgbytes))

            from matplotlib import pyplot as plt
            plt.imshow(np.asarray(img))
            plt.ion()
            plt.show()
            plt.pause(0.1)


            # do cool stuff here
            to_save = {k: dico[k] for k in ('clock', 'crowd', 'odom', 'report')}
            list_save.append(to_save)

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

            # theta>0 in cmd_vel turns right in the simulator, usually it's the opposite.
            pub['vel_cmd'] = (speed, -np.rad2deg(rot))

            # checking ending conditions
            if helpers.check_ending_conditions(180.0, -20, dico):
                break

            if helpers.get_odom(dico)[0] <= goal[0]:
                break

            # Debug: This skips the first test, remove
            if i == 0:
                break

            # doing a step
            pub = helpers.do_step(time_step, pub)

            time_out = time.time()

            if sleep_time > 0:
                time.sleep(sleep_time)
            elif time_out < time_in + time_step:
                time.sleep(time_in + time_step - time_out)

        # next scenario!
        socket_handler.send_and_receive(s, helpers.publish_all(helpers.next()))

        # wait for sim to load new scenario
        time.sleep(1)

#         recorder.save_dico('/tmp/recorder/tests_'+str(i), list_save)

    socket_handler.stop(s)
    time.sleep(1)


def load(file='recorder/SocialForces/go_straight_reactive_1.json'):
    loaded_data = recorder.load_file_as_list_of_dico(file)

    # print(len(loaded_data))
    # print([ map(float, loaded_data[i]['odom'].split(" ")[1:4]) for i,_ in enumerate(loaded_data) ] )
    data = []
    for i in range(len(loaded_data)):
        _, _, crowd = loaded_data[i]['crowd'].split(' ', 2)
        crowd = crowd.replace('(', '')
        crowd = crowd.replace(')', '')
        crowd = np.array(map(float, crowd.split(' '))).reshape((-1, 3))
        data.append(crowd)
    data = np.array(data)

    # reworked_data[0] is the list of x y corrdinates of first agent.
    reworked_data = np.array(
        [[data[i][j][1:] for i in range(len(data))] for j in range(len(data[0]))])

    def eucl_dist(a, b):
        return np.linalg.norm(np.array(b)-np.array(a))

    test = np.array([[eucl_dist(reworked_data[i][j], reworked_data[i][j+1])
                      for j in range(len(reworked_data[i])-1)] for i in range(len(reworked_data))])
    test2 = np.diff(test)
    test2 = np.where(np.abs(test2) < 2, test2, 0)

    is_near = np.array([[eucl_dist([0, 0], reworked_data[i][j]) < 5 for j in range(
        len(reworked_data[i])-1)] for i in range(len(reworked_data))])
    neighbor_vel = np.array([])

    for top_index,top_val in enumerate(test2):
        for index,val in enumerate(top_val):
            if is_near[top_index][index]:
                neighbor_vel = np.append(neighbor_vel, val )

    print(len(test2[0]))


def demo_run(HOST='127.0.0.1', PORT=25001, time_step=0.05):
    s = socket_handler.init_socket(HOST, PORT)

    def handler(signal_received, frame):
        # Handle any cleanup here
        print('SIGINT or CTRL-C detected. Exiting gracefully')
        socket_handler.stop(s)
        exit(0)
    signal(SIGINT, handler)

    print("Running simulation")

    pub = {'clock': 0, 'vel_cmd_pepper': (0, 0), 'vel_cmd_qolo': (0, 0), 'vel_cmd_turtlebot': (
        0, 0), 'vel_cmd_cuybot': (0, 0), 'vel_cmd_wheelchair': (0, 0), 'sim_control': 'i'}
    while True:
        time_in = time.time()
        # making the raw string to send from the dict
        to_send = helpers.publish_all(pub)

        # sending and receiving raw data
        raw = socket_handler.send_and_receive(s, to_send)

        # getting dict from raw data
        dico = helpers.raw_data_to_dict(raw)

        # do cool stuff here
        # recorder.save_dico('recorder/test_'+str(i),dico)

        pub['vel_cmd_pepper'] = robotnavigation.compute_rand_vel_cmd(dico)
        pub['vel_cmd_qolo'] = robotnavigation.compute_rand_vel_cmd(dico)
        pub['vel_cmd_turtlebot'] = robotnavigation.compute_rand_vel_cmd(dico)
        pub['vel_cmd_cuybot'] = robotnavigation.compute_rand_vel_cmd(dico)
        pub['vel_cmd_wheelchair'] = robotnavigation.compute_rand_vel_cmd(dico)

        # doing a step
        pub = helpers.do_step(time_step, pub)

        time_out = time.time()

        if time_out < time_in + time_step:
            time.sleep(time_in + time_step - time_out)

    socket_handler.stop(s)


def compute_metrics():
    directories = ['recorder/ORCA_anticipation/', 'recorder/ORCA_reflex/', 'recorder/SocialForces/']
    reactive = 'go_straight_reactive_'
    not_reactive = 'go_straight_not_reactive_'
    extension = '.json'

    #plot
    labels=['Talone/Tcrowded' , 'Lalone/Lcrowded', 'Jalone/Jcrowded', 'Vel_not_reactive_crowd/Vel_reactive_crowd', 'Vel_reactive_neighbors/Vel_reactive_crowd', 'Proximity']
    markers = [0, 0.2, 0.4, 0.6, 0.8, 1]
    str_markers = map(str, markers)

    ORCA_anticipation_results = np.array([])
    robot_alone_json = directories[0]+reactive+'0'+extension
    path_efficiency = np.array([])

    for scenario in range(1,21):
        path_efficiency = np.append(path_efficiency, metrics.compute_path_efficiency(robot_alone_json, directories[0]+reactive+str(scenario)+extension))
        # path_efficiency = np.append(path_efficiency, metrics.compute_path_efficiency(robot_alone_json, directories[0]+not_reactive+str(scenario)+extension))

    print(path_efficiency)
    # metrics.make_radar_chart("test", [0.1,0.5,0.2,0.3,0.8,0.9], labels, markers, str_markers) # example

    # def compute_path_efficiency(alone_json, crowded_json): return (Talone/Tcrowded , Lalone/Lcrowded, Jalone/Jcrowded)
    # def compute_effect_on_crowd(not_reactive_crowd_json, reactive_crowd_json): return (Vel_not_reactive_crowd/Vel_reactive_crowd, Vel_reactive_neighbors/Vel_reactive_crowd)
    # def compute_crowd_robot_interaction(scenario_json): return np.sum(Proxity)
    return

if __name__ == "__main__":
    run()
