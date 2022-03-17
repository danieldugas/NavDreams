from __future__ import print_function
from builtins import input
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

from navdreams.auto_debug import enable_auto_debug

enable_auto_debug()

bridge = CvBridge()

# not usable
# bag_path = "~/irl_tests/hg_icra_round2.bag"
# bag_path = "~/rosbags/merged_demo2.bag"
# bag_path = "~/LIANsden/proto_round_rosbags/daniel_manip_spray.bag"
# bag_path = "~/rosbags/HG_rosbags/hg_map.bag"

# usable
# bag_path = "~/rosbags/CLA_rosbags/2019-06-14-10-04-06.bag"
# bag_path = "~/rosbags/CLA_rosbags/2019-06-14-10-13-03.bag"
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/Stefan_Kiss_HG_Dataset/onboard/2019-04-05-13-03-25.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/Stefan_Kiss_HG_Dataset/onboard/2019-04-05-11-56-07.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/Stefan_Kiss_HG_Dataset/onboard/2019-04-05-11-58-01.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/Stefan_Kiss_HG_Dataset/onboard/2019-04-05-12-01-42.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/Stefan_Kiss_HG_Dataset/onboard/2019-04-05-12-04-58.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/Stefan_Kiss_HG_Dataset/onboard/2019-04-05-13-01-00.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/Stefan_Kiss_HG_Dataset/onboard/2019-04-05-13-08-23.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/Stefan_Kiss_HG_Dataset/onboard/2019-04-05-13-12-11.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/ASL Crowdbot/Rosbags/ASL open lab day/corridor_koze_kids.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/ASL Crowdbot/Rosbags/ASL open lab day/2019-12-13-20-11-46.bag" # noqa
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/rosbags/meet_your_lab1.bag"
# bag_path = "~/Insync/daniel@dugas.ch/Google Drive - Shared drives/Pepper/rosbags/meet_your_lab2.bag"
# bag_path = "/media/lake/koze_n3d_tests/day1/2022-01-19-14-12-40.bag"
# bag_path = "/media/lake/koze_n3d_tests/day1/2022-01-19-18-39-38.bag"
bag_path = "/media/lake/koze_n3d_tests/day1/2022-01-19-18-50-01.bag"
bag_path = "~/Downloads/2022-02-09-16-09-51_30min_K2.bag"

archive_dir = "~/navdreams_data/wm_experiments/datasets/V/rosbag"

DT = 0.2
# FIXED_FRAME = "odom" # StefanKiss, merged_demo, meet_your_lab
# FIXED_FRAME = "map" # crowdbot CLA
FIXED_FRAME = "reference_map" # open-lab # koze tests
ROBOT_FRAME = "base_footprint"
GOAL_REACHED_DIST = 0.5
MANUALLY_ADDED_GOALS = []
# MANUALLY_ADDED_GOALS = [[-23.2, -24.1]] # meet_your_lab1
# MANUALLY_ADDED_GOALS = [[4.23, -28.5]] # meet_your_lab2


resize_dim = (64, 64)
_W, _H = resize_dim
_CH = 3

odom_topic = '/pepper_robot/odom'
cmd_vel_topic = '/cmd_vel'
image_topic = '/camera/color/image_raw'
cmd_vel_enabled_topic = '/oculus/cmd_vel_enabled'
topics = [cmd_vel_enabled_topic, odom_topic, cmd_vel_topic, image_topic]
# goal_topic = "/move_base_simple/goal"
goal_topic = "/global_planner/goal" # koze tests

print()
print("Required topics:")
print("      " + odom_topic)
print("      " + cmd_vel_topic)
print("      " + image_topic)
print("Optional topics:")
print("      " + goal_topic)
print("      " + cmd_vel_enabled_topic)
print()


bag_path = os.path.expanduser(bag_path)
os.system('rosbag info {} | grep -e {} -e {} -e {} -e {} -e {}'.format(
    bag_path.replace(" ", "\ "), odom_topic, cmd_vel_topic, image_topic, goal_topic, cmd_vel_enabled_topic))

input("Are the required topics present?")

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
images = images.astype(np.uint8) # save some space!

# apply received goal messages to all future steps
goals_in_fix = np.ones((steps, 2)) * np.nan
goal_changed = np.zeros((steps,))
for topic, msg, t in bag.read_messages(topics=[goal_topic]):
    goal_in_msg = np.array([msg.pose.position.x, msg.pose.position.y])
    if FIXED_FRAME != msg.header.frame_id:
        rospy.logwarn_once("""Warning! Goal frame used in rosbag ({}) != Fixed frame used in this script ({})
        Optimally, the fixed frame would be refmap (static). Worse, are gmap (SLAM), or even odom.
        Please ensure that the rosbag has the desired frame, and set FIXED_FRAME accordingly.
              """.format(msg.header.frame_id, FIXED_FRAME))
    try:
        p2_msg_in_fix = Pose2D(bag_transformer.lookupTransform(
            FIXED_FRAME, msg.header.frame_id, msg.header.stamp))
    except: # noqa
        traceback.print_exc()
        raise ValueError("Could not find goal position in fixed frame! Is fixed frame wrong? \
                         hint: usually goals / global plans are set in fixed frame")
    goal_in_fix = apply_tf(goal_in_msg[None, :], p2_msg_in_fix)[0]

    # calculate timestep from which to start applying goal
    ts = t.to_sec()
    floatstep = (ts - start_time) / DT  # how many steps have passed since bag start
    closest_step = int(np.clip(np.round(floatstep), 0, steps))

    goals_in_fix[closest_step:] = goal_in_fix
    goal_changed[closest_step] = 1

# add true robot position information to all steps
robot_in_fix = np.ones((steps, 3)) * np.nan
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
if np.any(missing_pos):
    plt.figure()
    plt.plot(missing_pos)
    plt.show()

# find when robot is close to goal
def find_close_to_goal(steps, times, goals_in_fix, robot_in_fix, GOAL_REACHED_DIST):
    print("Finding close-to-goal steps")
    close_to_goal = np.zeros((steps,))
    for step, time in enumerate(times):
        if not np.any(np.isnan(goals_in_fix[step])):
            close_to_goal[step] = np.linalg.norm(
                robot_in_fix[step, :2] - goals_in_fix[step]
            ) < GOAL_REACHED_DIST
        for manual_goal_in_fix in MANUALLY_ADDED_GOALS:
            close_to_manual_goal = np.linalg.norm(
                robot_in_fix[step, :2] - manual_goal_in_fix
            ) < GOAL_REACHED_DIST
            close_to_goal[step] = close_to_goal[step] or close_to_manual_goal
    return close_to_goal


close_to_goal = find_close_to_goal(steps, times, goals_in_fix, robot_in_fix, GOAL_REACHED_DIST)

def cut_sequences(steps,
                  close_to_goal, goal_changed,
                  missing_images, missing_vels, missing_nextvels, missing_pos):
    print("Cutting sequences")
    # cut into sequence of length > 10
    # show each sequence in plot
    MIN_SEQ_LENGTH = 24
    sequence_ids = np.ones((steps,)) * -1
    sequence_steps = np.ones((steps,)) * -1
    sequence_ends = np.zeros((steps,))
    current_sequence_id = 0
    current_sequence_length = 0
    for step in range(steps):
        last_step = step == steps-1
        if close_to_goal[step] or goal_changed[step] or last_step or \
                missing_images[step] or missing_vels[step] or missing_nextvels[step] or missing_pos[step]:
            if current_sequence_length >= MIN_SEQ_LENGTH: # terminate sequence if valid
                current_sequence_length = 0
                current_sequence_id += 1
                sequence_ends[step-1] = 1
            else:
                sequence_ids[step] = -1
        else:
            sequence_ids[step] = current_sequence_id
            sequence_steps[step] = current_sequence_length
            current_sequence_length += 1
    n_sequences = current_sequence_id
    return sequence_ids, sequence_steps, sequence_ends, n_sequences


sequence_ids, sequence_steps, sequence_ends, n_sequences = cut_sequences(
    steps, close_to_goal, goal_changed, missing_images, missing_vels, missing_nextvels, missing_pos)

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

# re-calculate close-to-goal, re-cut sequences
close_to_goal = find_close_to_goal(steps, times, goals_in_fix, robot_in_fix, GOAL_REACHED_DIST)
sequence_ids, sequence_steps, sequence_ends, n_sequences = cut_sequences(
    steps, close_to_goal, goal_changed, missing_images, missing_vels, missing_nextvels, missing_pos)

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

# plot all sequence-ending conditions
plt.figure()
legends = []
for i, varname in enumerate([
        "missing_images", "missing_vels", "missing_pos",
        "missing_nextvels", "close_to_goal", "goal_changed"]):
    series = locals().get(varname)
    legends.append(varname)
    # slightly separate our plotted lines for clarity
    plt.plot(series + i * 0.02)
plt.legend(legends)
plt.show()

# show evenly spaced images
N = 10
thumbnails = scans[::len(scans) // N]
fig, axes = plt.subplots(1, N)
for i, ax in enumerate(axes):
    ax.imshow(thumbnails[i].astype(np.uint8))
plt.show()

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
    bag_name = os.path.splitext(os.path.basename(bag_path))[0].replace('_', '')
    archive_dir = os.path.expanduser(archive_dir)
    make_dir_if_not_exists(archive_dir)
    archive_path = os.path.join(archive_dir,
                                "{}_scans_robotstates_actions_rewards_dones.npz".format(bag_name))
    np.savez_compressed(archive_path,
                        scans=scans, robotstates=robotstates, actions=actions, rewards=rewards, dones=dones)
    print("Saved to {}".format(archive_path))
