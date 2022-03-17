import os
import numpy as np
import torch
from navrep.models.gpt import load_checkpoint
from CMap2D import CMap2D, gridshow
from sensor_msgs.msg import LaserScan

from navdreams.navrep3dtrainenv import NavRep3DTrainEnv
from multitask_train import TaskLearner

_100 = 100.

class ClassicalPlanner(object):
    def __init__(self, gpu=True):
        # load model
        task_channels = 1 # depth
        from_image = True
        label_is_onehot = False
        self.depth_model = TaskLearner(task_channels, from_image, label_is_onehot, gpu=gpu)
        depth_model_path = os.path.expanduser(
            "~/navdreams_data/results/models/multitask/baseline_depth_2021_11_08__22_21_30")
        load_checkpoint(self.depth_model, depth_model_path, gpu=gpu)
        self.segmentation_model = TaskLearner(6, from_image, True, gpu=gpu)
        segmentation_model_path = os.path.expanduser(
            "~/navdreams_data/results/models/multitask/baseline_segmenter_2021_11_09__04_13_38")
        load_checkpoint(self.segmentation_model, segmentation_model_path, gpu=gpu)

    def _obs_to_depth(self, image):
        x = np.moveaxis(image / 255., -1, 0)[None, :, :, :]
        x_t = torch.tensor(x * 1., dtype=torch.float)
        depth_t, _ = self.depth_model(x_t)
        depth = depth_t.detach().cpu().numpy()
        depth = np.moveaxis(depth[0], 0, -1)
        labels_t, _ = self.segmentation_model(x_t)
        labels = labels_t.detach().cpu().numpy()
        labels = np.moveaxis(labels[0], 0, -1)
        is_goal = np.argmax(labels, axis=-1) == 4
        depth[is_goal] = 100
        return depth

    def reset(self):
        pass

    def predict(self, obs, env):
        # simple heuristic - accelerate and face goal
        robotstate = obs[1]
        goal_xy = robotstate[:2]
        goal_dist = np.linalg.norm(goal_xy)
        best_xy = goal_xy / goal_dist if goal_dist != 0 else np.zeros_like(goal_xy)
        angle_to_goal = np.arctan2(goal_xy[1], goal_xy[0])
        best_rot = angle_to_goal * 0.3
        # turn to face goal
        if abs(angle_to_goal) > (np.pi / 6.):
            best_xy = best_xy * 0
            best_acc = np.array([best_xy[0], best_xy[1], best_rot])
            action = best_acc
            return action
        best_acc = np.array([best_xy[0], best_xy[1], best_rot])
        depth = self._obs_to_depth(obs[0])
        ranges = depth[32, ::-1, 0]
        scan = LaserScan()
        FOV = np.deg2rad(72.)
        scan.angle_min = -FOV / 2.
        scan.angle_max = FOV / 2.
        scan.angle_increment = FOV / len(ranges)
        scan.ranges = list(ranges)
        limits = np.array([[-5, 20],
                           [-10, 10]],
                          dtype=np.float32)
        local_map = CMap2D()
        local_map.from_scan(scan, limits=limits, resolution=0.2, legacy=False)
        goal_ij = local_map.xy_to_ij(goal_xy[None, :], clip_if_outside=True)[0]
        robot_ij = local_map.xy_to_ij(np.array([[0, 0]]), clip_if_outside=True)[0]
        fm = local_map.fastmarch(goal_ij)
        action = best_acc
        if True:
            d = depth * 1.
            d[32, :, :] = 0.2
            print(goal_xy)
            from matplotlib import pyplot as plt
            plt.ion()
            plt.figure("plan")
            plt.clf()
            fig, (ax1, ax2) = plt.subplots(2, 1, num="plan")
#             gridshow(fm)
            plt.sca(ax1)
            gridshow(local_map.occupancy())
            ax2.imshow(d)
            plt.scatter(goal_ij[0], goal_ij[1])
            plt.scatter(robot_ij[0], robot_ij[1])
            plt.pause(0.1)
        return action


np.set_printoptions(precision=1, suppress=True)
p = ClassicalPlanner(gpu=False)
env = NavRep3DTrainEnv()
obs = env.reset()
for i in range(1000):
    action = p.predict(obs, env)
    obs, _, done, _ = env.step(action)
    env.render()
    if done:
        obs = env.reset()
env.close()
