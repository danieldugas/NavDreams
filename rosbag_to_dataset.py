import numpy as np
import os
import traceback
import rosbag
from cv_bridge import CvBridge
import cv2
from pose2d import Pose2D, apply_tf, inverse_pose2d
import tf_bag
import rospy
from matplotlib import pyplot as plt
from tqdm import tqdm

from pyniel.python_tools.path_tools import make_dir_if_not_exists

bridge = CvBridge()

# bag_path = "~/irl_tests/hg_icra_round2.bag"
bag_path = "~/LIANsden/proto_round_rosbags/daniel_manip_spray.bag"

archive_dir = "~/navrep3d_W/datasets/V/rosbag"

DT = 0.2
FIXED_FRAME = "gmap"
ROBOT_FRAME = "base_footprint"
GOAL_REACHED_DIST = 0.5

resize_dim = (64, 64)
_W, _H = resize_dim
_CH = 3

odom_topic = '/pepper_robot/odom'
cmd_vel_topic = '/cmd_vel'
image_topic = '/camera/color/image_raw'
cmd_vel_enabled_topic = '/oculus/cmd_vel_enabled'
topics = [cmd_vel_enabled_topic, odom_topic, cmd_vel_topic, image_topic]
goal_topic = "/move_base_simple/goal"

bag_path = os.path.expanduser(bag_path)
os.system('rosbag info {} | grep -e {} -e {} -e {}'.format(
    bag_path, odom_topic, cmd_vel_topic, image_topic))

# sync concept: pick closest at each dt
# | | | | | | | | odom
#  |     |     |  image
#    | |    |  |  cmd_vel
# |   |   |   |   dt

print("Loading bag...")
bag = rosbag.Bag(bag_path)
print("Initializing Tf Transformer...")
bag_transformer = tf_bag.BagTfTransformer(bag)

start_time = bag.get_start_time()
end_time = bag.get_end_time()
times = np.arange(start_time, end_time, DT)
steps = len(times)

# image
# robotstates [gx, gy, vx, vy, vtheta]
# action = [vx, vy, vtheta]
images = np.ones((steps, _W, _H, _CH)) * np.nan
vels = np.ones((steps, 3)) * np.nan
nextvels = np.zeros_like(vels)

images_eps_min = np.ones((steps,)) * DT / 2.
odom_eps_min = np.ones((steps,)) * DT / 2.

for topic, msg, t in tqdm(bag.read_messages(topics=topics)):

    ts = t.to_sec()
    floatstep = (ts - start_time) / DT  # how many steps have passed since bag start
    closest_step = int(np.clip(np.round(floatstep), 0, steps-1))
    eps = np.abs(ts - times[closest_step])

    if topic == odom_topic:
        current_vel = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.angular.z]
        if eps < odom_eps_min[closest_step]:
            odom_eps_min[closest_step] = eps
            vels[closest_step] = current_vel
    if topic == image_topic:
        if eps < images_eps_min[closest_step]:
            images_eps_min[closest_step] = eps
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            cv_resized = cv2.resize(cv_image, resize_dim)
            images[closest_step] = cv_resized
nextvels[:-1] = vels[1:] # assume vel is action at previous timestep
missing_images = np.any(np.isnan(images.reshape((len(images), -1))), axis=-1)
missing_vels = np.any(np.isnan(vels), axis=-1)
missing_nextvels = np.any(np.isnan(nextvels), axis=-1)

# apply received goal messages to all future steps
goals_in_fix = np.ones((steps, 2)) * np.nan
goal_changed = np.zeros((steps,))
for topic, msg, t in bag.read_messages(topics=[goal_topic]):
    goal_in_msg = np.array([msg.pose.position.x, msg.pose.position.y])
    p2_msg_in_fix = Pose2D(bag_transformer.lookupTransform(
        FIXED_FRAME, msg.header.frame_id, msg.header.stamp))
    goal_in_fix = apply_tf(goal_in_msg[None, :], p2_msg_in_fix)[0]

    # calculate timestep from which to start applying goal
    ts = t.to_sec()
    floatstep = (ts - start_time) / DT  # how many steps have passed since bag start
    closest_step = int(np.clip(np.round(floatstep), 0, steps))

    goals_in_fix[closest_step:] = goal_in_fix
    goal_changed[closest_step] = 1

# add true robot position information to all steps
robot_in_fix = np.ones((steps, 3)) * np.nan
close_to_goal = np.zeros((steps,))
missing_pos = np.zeros((steps,))
for step, time in enumerate(times):
    try:
        p2_rob_in_fix = Pose2D(bag_transformer.lookupTransform(
            FIXED_FRAME, ROBOT_FRAME, rospy.Time(time)))
    except: # noqa
        traceback.print_exc()
        missing_pos[step] = 1
        continue
    robot_in_fix[step] = p2_rob_in_fix
    if not np.any(np.isnan(goals_in_fix[step])):
        close_to_goal[step] = np.linalg.norm(robot_in_fix[step, :2] - goals_in_fix[step]) < GOAL_REACHED_DIST

# cut into sequence of length > 10
# show each sequence in plot
MIN_SEQ_LENGTH = 24
sequence_ids = np.ones((steps,)) * -1
sequence_steps = np.ones((steps,)) * -1
sequence_ends = np.zeros((steps,))
current_sequence_id = 0
current_sequence_length = 0
for step in range(steps):
    if close_to_goal[step] or goal_changed[step] or \
            missing_images[step] or missing_vels[step] or missing_nextvels[step] or missing_pos[step]:
        if current_sequence_length >= MIN_SEQ_LENGTH: # terminate sequence if valid
            current_sequence_length = 0
            current_sequence_id += 1
            sequence_ends[step] = 1
        else:
            sequence_ids[step] = -1
    else:
        sequence_ids[step] = current_sequence_id
        sequence_steps[step] = current_sequence_length
        current_sequence_length += 1
n_sequences = current_sequence_id

# fill goals
for seq_id in range(n_sequences):
    seq_mask = sequence_ids == seq_id
    seq_robot_in_fix = robot_in_fix[seq_mask]
    seq_goals_in_fix = goals_in_fix[seq_mask]
    if np.any(np.isnan(seq_goals_in_fix)):
        print("Sequence {} (len {}): goal missing! Use last point as goal?".format(
            seq_id, len(seq_robot_in_fix)))
        plt.figure("missing goal")
        plt.title("Goal missing")
        plt.plot(seq_robot_in_fix[:, 0], seq_robot_in_fix[:, 1])
        plt.show()
        plt.close('all')
        if True:
            print("using last point as goal.")
            goals_in_fix[seq_mask] = seq_robot_in_fix[-1, :2]


# infer goal_in_robot
goals_in_robot = np.zeros_like(goals_in_fix)
for step in range(steps):
    if np.any(np.isnan(goals_in_fix[step])):
        continue
    p2_fix_in_robot = inverse_pose2d(robot_in_fix[step])
    goal_in_robot = apply_tf(goals_in_fix[step][None, :], p2_fix_in_robot)[0]
    goals_in_robot[step] = goal_in_robot

# plot sequences
plt.figure("sequences")
plt.title("Sequences")
legends = []
for seq_id in range(n_sequences):
    seq_mask = sequence_ids == seq_id
    seq_robot_in_fix = robot_in_fix[seq_mask]
    seq_goals_in_fix = goals_in_fix[seq_mask]
    seq_nextvels = nextvels[seq_mask]
    legends.append(str(seq_id))
    plt.plot(seq_robot_in_fix[:, 0], seq_robot_in_fix[:, 1])
plt.legend(legends)
plt.show()

# transform to robotstates
scans = np.ones((steps, _W, _H, _CH)) * np.nan
robotstates = np.ones((steps, 5)) * np.nan
actions = np.ones((steps, 3)) * np.nan
dones = np.zeros((steps,))

scans = images
robotstates[:, 0] = goals_in_robot[:, 0]
robotstates[:, 1] = goals_in_robot[:, 1]
robotstates[:, 2] = vels[:, 0]
robotstates[:, 3] = vels[:, 1]
robotstates[:, 4] = vels[:, 2]
actions = nextvels
dones = sequence_ends

# remove any data outside of valid sequence
valid_mask = sequence_ids != -1
scans = scans[valid_mask]
robotstates = robotstates[valid_mask]
actions = actions[valid_mask]
dones = dones[valid_mask]
rewards = np.zeros_like(dones)
print("Found {} sequences, {} total steps".format(n_sequences, len(robotstates)))

if np.any(np.isnan(scans)):
    raise ValueError
if np.any(np.isnan(robotstates)):
    raise ValueError
if np.any(np.isnan(actions)):
    raise ValueError
if np.any(np.isnan(rewards)):
    raise ValueError

# write
if n_sequences > 0:
    archive_dir = os.path.expanduser(archive_dir)
    make_dir_if_not_exists(archive_dir)
    archive_path = os.path.join(archive_dir, "{:03}_scans_robotstates_actions_rewards_dones.npz".format(0))
    np.savez_compressed(archive_path,
                        scans=scans, robotstates=robotstates, actions=actions, rewards=rewards, dones=dones)
    print("Saved to {}".format(archive_path))
