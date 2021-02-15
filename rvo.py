import numpy as np
from matplotlib import pyplot as plt

import rvo2

def angle_difference_rad(target_angle, angle):
    """     / angle
           /
          / d
         /)___________ target
    """
    delta_angle = angle - target_angle
    delta_angle = np.arctan2(np.sin(delta_angle), np.cos(delta_angle))  # now in [-pi, pi]
    return delta_angle

class NavigationPlanner(object):
    robot_radius = 0.3  # [m]
    safety_distance = 0.1  # [m]
    robot_max_speed = 1.  # [m/s]
    robot_max_w = 1. # [rad/s]
    robot_max_accel = 0.5  # [m/s^2]
    robot_max_w_dot = 10.  # [rad/s^2]

    def __init__(self):
        raise NotImplementedError

    def set_static_obstacles(self, static_obstacles):
        self.static_obstacles = static_obstacles

    def compute_cmd_vel(self, crowd, robot_pose, goal, show_plot=True, debug=False):
        raise NotImplementedError

class RVONavigationPlanner(NavigationPlanner):
    def __init__(self):
        # variables
        self.sim = None
        # RVO parameters
        self.neighbor_dist = 10.
        self.max_neighbors = 10
        self.time_horizon = 5.
        self.time_horizon_obst = 5.

    def set_static_obstacles(self, static_obstacles):
        self.static_obstacles = static_obstacles

    def compute_cmd_vel(self, crowd, robot_pose, goal, show_plot=True, debug=False):
        x, y, th, vx, vy, w = robot_pose
        plt.ion()
        plt.figure(1)
        # these params could be defined in init, or received from simulator
        human_radius = 0.3
        dt = 0.5  # time step [s]

        # create sim with static obstacles if they don't exist
        if self.sim is None:
            self.sim = rvo2.PyRVOSimulator(dt,
                                           self.neighbor_dist, self.max_neighbors,
                                           self.time_horizon, self.time_horizon_obst,
                                           human_radius, 0.)
            for obstacle in self.static_obstacles:
                self.sim.addObstacle(obstacle)
            self.sim.processObstacles()

        self.sim.clearAgents()

        # add robot
        self.sim.addAgent((x, y),
                          self.neighbor_dist, self.max_neighbors,
                          self.time_horizon, self.time_horizon_obst,
                          self.robot_radius + self.safety_distance,
                          self.robot_max_speed, (vx, vy))
        pref_vel = goal[:2] - np.array([x, y])
        pref_vel = pref_vel / np.linalg.norm(pref_vel) * self.robot_max_speed
        self.sim.setAgentPrefVelocity(0, tuple(pref_vel))

        # add crowd
        for i, person in enumerate(crowd):
            self.sim.addAgent(tuple(person[1:3]),
                              self.neighbor_dist, self.max_neighbors,
                              self.time_horizon, self.time_horizon_obst,
                              human_radius + 0.01 + self.safety_distance,
                              0.5, (0,0))
            # TODO: use current agent vel
            pref_vel = (0, 0)
            self.sim.setAgentPrefVelocity(i + 1, pref_vel)

        self.sim.doStep()
        vx, vy = self.sim.getAgentVelocity(0)
        speed = np.linalg.norm([vx, vy])
        robot_angle = th
        heading_x = np.cos(robot_angle)
        heading_y = np.sin(robot_angle)
        desired_angle = np.arctan2(vy, vx)
        # rotate to reduce angle difference to desired angle
        rot = -angle_difference_rad(desired_angle, robot_angle)

        if show_plot:
            plt.cla()
            plt.plot([x, x+heading_x], [y, y+heading_y])
            plt.plot([x, x+vx], [y, y+vy])
            plt.gca().add_artist(plt.Circle((x, y), self.robot_radius, color='b'))
            for cid, cx, cy in crowd:
                plt.gca().add_artist(plt.Circle((cx, cy), human_radius, color='r'))
            plt.xlim([0, 22])
            plt.ylim([-5, 5])
            plt.pause(0.1)
        print("SOLUTION ------")
        print(speed, rot)

        return (speed, rot)
