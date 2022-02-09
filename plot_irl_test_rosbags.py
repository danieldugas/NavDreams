from __future__ import print_function
import numpy as np
import os
import rosbag
from tqdm import tqdm
from pose2d import Pose2D, apply_tf
from matplotlib import pyplot as plt
from CMap2D import CMap2D
from strictfire import StrictFire

from pyniel.pyplot_tools.interactive import make_legend_pickable

DOWNSAMPLE = None  # 17 (1080 -> 64 rays)
OLD_FORMAT = False
ANIMATE = False

FIXED_FRAME = "gmap"
ROBOT_FRAME = "base_footprint"

GOAL_REACHED_DIST = 0.5

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    # from stackoverflow how-to-extract-a-subset-of-a-colormap-as-a-new-colormap-in-matplotlib
    from matplotlib import colors
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def blue(u=np.random.rand()):
    cmap = truncate_colormap(plt.cm.bwr, 0., .4)
    u = np.clip(u, 0., 1.)
    return cmap(u)

def orange(u=np.random.rand()):
    cmap = truncate_colormap(plt.cm.Wistia, .3, .7)
    u = np.clip(u, 0., 1.)
    return cmap(u)


def main(clean=False):
#     bag_path = os.path.expanduser("~/irl_tests/manip_corner_julian_jenjen.bag")
#     bag_path = os.path.expanduser("/media/lake/koze_n3d_tests/day1/2022-01-19-18-50-01.bag")
    bag_path = os.path.expanduser("/media/daniel/Samsung T5/2022-02-09-16-09-51_30min_K2.bag")

    cmdvel_topics = ["/cmd_vel"]
    joy_topics = ["/joy"]
    odom_topics = ["/pepper_robot/odom"]
    reward_topics = ["/goal_reached"]
    failure_topics = ["/goal_failed"]
    stopped_topics = ["/is_stopped"]
    resumed_topics = ["/is_resumed"]
    goal_topics = ["/global_planner/goal"]
    map_topics = ["/gmap"]
    topics = (cmdvel_topics + joy_topics + odom_topics + reward_topics + failure_topics
              + goal_topics + map_topics + stopped_topics + resumed_topics)

    bag_name = os.path.basename(bag_path)
    print("Loading {}...".format(bag_name))
    bag = rosbag.Bag(bag_path)
    try:
        import tf_bag
        bag_transformer = tf_bag.BagTfTransformer(bag)
    except ImportError:
        print("WARNING: Failed to import tf_bag. No goal information will be saved.")
        bag_transformer = None
    if bag.get_message_count(topic_filters=goal_topics) == 0:
        print("WARNING: No goal messages ({}) in rosbag. No goal information will be saved.".format(
            goal_topics))
        bag_transformer = None

    trajectories = []
    goals = []
    goals_failed = []
    goals_reached = []
    goals_close = []
    are_stopped = []
    # continual variables
    is_stopped = False
    goal = None
    end_episode = False
    # single trajectory data
    trajectory = []
    ep_goal = None
    ep_stopped = False
    reached_idx = 0
    failed_idx = 0
    goal_close_idx = 0
    for topic, msg, t in tqdm(bag.read_messages(topics=topics),
                              total=bag.get_message_count(topic_filters=topics)):

        if topic in stopped_topics:
            is_stopped = True
            end_episode = True

        if topic in resumed_topics:
            is_stopped = False
            end_episode = True

        # process messages
        if topic in odom_topics:
            # position
            try:
                p2_rob_in_fix = Pose2D(bag_transformer.lookupTransform(
                    FIXED_FRAME, ROBOT_FRAME, msg.header.stamp))
            except:  # noqa
                continue

            if goal is not None:
                trajectory.append(p2_rob_in_fix)

                if np.linalg.norm(p2_rob_in_fix[:2] - goal) < GOAL_REACHED_DIST:
                    goal_close_idx = len(trajectory)-1

        if topic in reward_topics:
            # goal is reached
            reached_idx = len(trajectory)-1
            end_episode = True

        if topic in failure_topics:
            failed_idx = len(trajectory)-1
            end_episode = True

        if topic in goal_topics:
            goal_in_msg = np.array([msg.pose.position.x, msg.pose.position.y])
            p2_msg_in_fix = Pose2D(bag_transformer.lookupTransform(
                FIXED_FRAME, msg.header.frame_id, msg.header.stamp))
            goal_in_fix = apply_tf(goal_in_msg[None, :], p2_msg_in_fix)[0]

            # end episode if goal change
            if goal is None:
                end_episode = True
            else:
                if np.linalg.norm(goal - goal_in_fix) > GOAL_REACHED_DIST:
                    end_episode = True

            goal = goal_in_fix

        if end_episode:
            # store episode
            if len(trajectory) > 0:
                trajectories.append(np.array(trajectory))
                are_stopped.append(ep_stopped)
                goals.append(ep_goal)
                goals_failed.append(failed_idx)
                goals_reached.append(reached_idx)
                goals_close.append(goal_close_idx)
            # reset trajectory
            trajectory = []
            ep_goal = goal
            ep_stopped = is_stopped
            reached_idx = 0
            failed_idx = 0
            goal_close_idx = 0
            # reset
            end_episode = False

        if topic in map_topics:
            mapmsg = msg

    fig, ax = plt.subplots(1, 1)

    if mapmsg is not None:
        map2d = CMap2D()
        map2d.from_msg(mapmsg)
        assert mapmsg.header.frame_id == FIXED_FRAME
        contours = map2d.as_closed_obst_vertices()
        for c in contours:
            cplus = np.concatenate((c, c[:1, :]), axis=0)
            ax.plot(cplus[:,0], cplus[:,1], color='k')
        plt.axis('equal')

    i_frame = 0
    legends = []
    linegroups = []
    for n, (t, g, s, f, gc, st) in enumerate(zip(
            trajectories, goals, goals_reached, goals_failed, goals_close, are_stopped)):
        line_color = blue(len(t)/1000.) if s != 0 or gc != 0 else orange(len(t)/1000.)
        line_style = None
        if st:
            line_color = "grey"
        if g is None:
            line_style = "--"
            line_color = "grey"
        zorder = 2 if s else 1
        if ANIMATE:
            yanim = np.ones_like(t[:,1]) * np.nan
            line, = ax.plot(t[:,0], yanim, color=line_color, zorder=zorder)
            ax.add_artist(plt.Circle((g[0], g[1]), 0.3, color="red", zorder=2))
            plt.pause(0.01)
            N = 10
            for i in range(0, len(yanim), N):
                yanim[i:i+N] = t[i:i+N,1]
                line.set_ydata(yanim)
                plt.pause(0.01)
                plt.savefig("/tmp/plot_irl_test_rosbags_{:05}.png".format(i_frame))
                i_frame += 1
        else:
            if line_color != "grey":
                line, = ax.plot(t[:,0], t[:,1], color=line_color, zorder=zorder,
                                linestyle=line_style, alpha=0.8)
                if f != 0:
                    ax.scatter(t[f, 0], t[f, 1], color="red", marker="x", zorder=3)
                if g is not None:
                    cr = plt.Circle((g[0], g[1]), 0.3, color="mediumorchid", zorder=2, fill=False)
                    ax.add_artist(cr)
                linegroups.append([line, cr])
                legends.append(str(n))
            if not clean:
                if line_color == "grey":
                    line, = ax.plot(t[:,0], t[:,1], color=line_color, zorder=zorder, linestyle=line_style)
                else:
                    if g is not None:
                        l, = ax.plot([t[0, 0], g[0]], [t[0, 1], g[1]], color='k', zorder=zorder,
                                     linestyle="--")
                        linegroups[-1].append(l)
                if gc != 0:
                    ax.scatter(t[gc, 0], t[gc, 1], color="green", marker=">")

    L = fig.legend([lines[0] for lines in linegroups], legends)
    make_legend_pickable(L, linegroups)

    goals_failed = []
    ax.set_title(bag_name)
    ax.axis("equal")
    ax.set_adjustable('box')

    plt.show()


if __name__ == "__main__":
    StrictFire(main)
