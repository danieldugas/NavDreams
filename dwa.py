import matplotlib.pyplot as plt
import numpy as np

from rvo import angle_difference_rad, NavigationPlanner
from dynamic_window import single_trajectory_and_costs

class DynamicWindowApproachNavigationPlanner(NavigationPlanner):
    def __init__(self):
        # dynamic window parameters
        self.speed_resolution = 0.01  # [m/s]
        self.rot_resolution = 0.02  # [rad/s]
        self.prediction_horizon = 3.0  # [s]
        self.robot_is_stuck_velocity = 0.001
        # cost parameters
        self.goal_cost_weight = 0.15
        self.speed_cost_weight = 1.
        self.obstacle_cost_weight = 1.0
        # distance from robot to obstacle surface at which to ignore the obstacle
        # should be bigger than the robot radius
        self.obstacle_ignore_distance = 0.6

    def compute_cmd_vel(self, crowd, robot_pose, goal, show_plot=True, debug=False):
        x, y, th, vx, vy, w = robot_pose
        v = np.sqrt(vx*vx + vy*vy)

        # these params could be defined in init, or received from simulator
        human_radius = 0.3
        dt = 0.1  # time step [s]

        # Initialize the dynamic window
        # Velocity limits
        DWv_speed = [-self.robot_max_speed, self.robot_max_speed]
        DWv_rot = [-self.robot_max_w, self.robot_max_w]

        # Acceleration limits
        DWa_speed = [v - self.robot_max_accel * dt, v + self.robot_max_accel * dt]
        DWa_rot = [w - self.robot_max_w_dot * dt, w + self.robot_max_w_dot * dt]

        #  [v_min, v_max, yaw_rate_min, yaw_rate_max]
        W_speed = [max(min(DWv_speed), min(DWa_speed)), min(max(DWv_speed), max(DWa_speed))]
        W_rot = [max(min(DWv_rot), min(DWa_rot)), min(max(DWv_rot), max(DWa_rot))]

        # initialize variables
        best_cost = np.inf
        worst_cost = -np.inf
        best_cmd_vel = [0.0, 0.0]
        best_trajectory = None

        if show_plot:
            plt.ion()
            plt.figure(1)
            plt.cla()

        # Sample window to find lowest cost
        # evaluate all trajectory with sampled input in dynamic window
        for speed in np.arange(min(W_speed), max(W_speed), self.speed_resolution):
            for rot in np.arange(min(W_rot), max(W_rot), self.rot_resolution):
                cmd_vel = (speed, rot)

                # Trajectory
                trajectory, cost = single_trajectory_and_costs(
                        robot_pose, goal, cmd_vel,
                        self.prediction_horizon, dt,
                        crowd, self.static_obstacles, human_radius, self.obstacle_ignore_distance,
                        self.robot_radius, self.robot_max_speed,
                        self.goal_cost_weight, self.speed_cost_weight, self.obstacle_cost_weight,
                )

                # Update best cost
                if cost <= best_cost:
                    best_cost = cost
                    best_cmd_vel = [speed, rot]
                    best_trajectory = trajectory

                if cost >= worst_cost:
                    worst_cost = cost

        if debug:
            for speed in np.arange(min(W_speed), max(W_speed), self.speed_resolution):
                for rot in np.arange(min(W_rot), max(W_rot), self.rot_resolution):
                    cmd_vel = (speed, rot)

                    # Trajectory
                    trajectory, cost = single_trajectory_and_costs(
                            robot_pose, goal, cmd_vel,
                            self.prediction_horizon, dt,
                            crowd, self.static_obstacles, human_radius, self.obstacle_ignore_distance,
                            self.robot_radius, self.robot_max_speed,
                            self.goal_cost_weight, self.speed_cost_weight, self.obstacle_cost_weight,
                    )

                    if debug:
                        plt.plot(trajectory[:, 0], trajectory[:, 1],
                                 c=plt.cm.viridis((cost - worst_cost) / (0.00001 + best_cost - worst_cost)))
                        plt.title("{} - g {}, s {}, o {}".format(cost, goal_cost, speed_cost, ob_cost))
                        plt.axis('equal')
                        plt.pause(0.001)

        if best_trajectory is None:
            best_trajectory = np.array([robot_pose])
            print("No solution found!")

        # spin if stuck
        if abs(best_cmd_vel[0]) < self.robot_is_stuck_velocity \
                and abs(best_cmd_vel[1]) < self.robot_is_stuck_velocity:
            best_cmd_vel[1] = max(W_rot)

        if show_plot:
            plt.ion()
            plt.figure(1)
            plt.cla()
            plt.gca().add_artist(plt.Circle((robot_pose[0], robot_pose[1]), self.robot_radius, color='b'))
            for id_, x, y in crowd:
                plt.gca().add_artist(plt.Circle((x, y), human_radius, color='r'))
            plt.plot(best_trajectory[:, 0], best_trajectory[:, 1], "-g")
            plt.plot(goal[0], goal[1], "xb")
            plt.axis("equal")
            plt.xlim([0, 22])
            plt.ylim([-5, 5])
            plt.pause(0.1)

        print("SOLUTION")
        print(best_cost)
        print(best_cmd_vel)

        return tuple(best_cmd_vel)

    def step_robot_dynamics(self, pose, cmd_vel, dt):
        x, y, th, vx, vy, w = pose
        speed, rot = cmd_vel
        # first apply small rotation
        next_th = th + rot * dt
        # update velocities (no momentum)
        next_vx = speed * np.cos(next_th)
        next_vy = speed * np.sin(next_th)
        next_w = rot
        # then move along new angle
        next_x = x + next_vx * dt
        next_y = y + next_vy * dt

        next_pose = np.array([next_x, next_y, next_th, next_vx, next_vy, next_w])
        return next_pose

    def predict_trajectory(self, pose, cmd_vel, dt):
        # initialize
        timesteps = np.arange(0, self.prediction_horizon, dt)
        trajectory = np.zeros((len(timesteps)+1, pose.shape[0]))
        trajectory[0] = pose * 1.
        # fill trajectory poses
        for i, time in enumerate(timesteps):
            trajectory[i+1] = self.step_robot_dynamics(trajectory[i], cmd_vel, dt)

        return trajectory

    def obstacles_cost(self, trajectory, crowd, human_radius):
        inflated_robot_radius = self.robot_radius + self.safety_distance

        static_cost = 0.
        # static obstacles
        for obs in self.static_obstacles:
            # TODO
            distances = np.array([np.inf])
            if np.any(distances <= inflated_robot_radius):
                static_cost = np.inf
            else:
                static_cost = 1. / distances

        dynamic_cost = 0.
        # circular obstacles
        if len(crowd) > 0:
            xy_obstacles = crowd[:, 1:3]
            obstacle_radii = human_radius # [obstacles] or [1] if same radius
            # dx, dy distance between every obstacle and every pose in the trajectory
            deltas = trajectory[:, None, :2] - xy_obstacles[None, :, :2] # [timesteps, obstacles, xy]
            # distance from obstacle surfaces
            distances = np.linalg.norm(deltas, axis=-1) - obstacle_radii
            min_dist = np.min(distances)
            if min_dist <= inflated_robot_radius:
                dynamic_cost = np.inf
            elif min_dist > self.obstacle_ignore_distance:
                dynamic_cost = 0
            else:
                dynamic_cost = 1. / np.min(distances)

        return max(static_cost, dynamic_cost)

    def goal_cost(self, trajectory, goal):
        # looking only at the last step in the trajectory
        x, y, th, vx, vy, w = trajectory[-1]
        gx, gy = goal
        # cost increases if the final robot heading points away from the goal
        goal_heading = np.arctan2(gy - y, gx - x)  # heading of robot-goal vector
        robot_heading = th
        heading_difference_angle = angle_difference_rad(goal_heading, robot_heading)
        cost = np.abs(heading_difference_angle)

        # no cost if goal is reached
        if np.sqrt((gy - y)**2 + (gx - x)**2) <= self.robot_radius:
            cost = 0

        return cost
