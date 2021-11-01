import numpy as np
import os
from strictfire import StrictFire
from pyniel.python_tools.path_tools import make_dir_if_not_exists
from navrep.scripts.make_vae_dataset import SemiRandomMomentumPolicy

from navrep3d.navrep3dtrainenv import NavRep3DTrainEnv

def generate_segmentation_dataset(env, n_sequences,
                                  episode_length=1000,
                                  subset_index=0, n_subsets=1,
                                  render=True,
                                  policy=SemiRandomMomentumPolicy(),
                                  archive_dir=os.path.expanduser("~/navrep/datasets/V/ian")
                                  ):
    """
    if n_subsets is None, the whole set of sequences is generated (n_sequences)
    if n_subsets is a number > 1, this function only generates a portion of the sequences
    """
    indices = np.arange(n_sequences)
    if n_subsets > 1:  # when multiprocessing
        indices = np.array_split(indices, n_subsets)[subset_index]
    for n in indices:
        images = []
        labels = []
        robotstates = []
        actions = []
        rewards = []
        dones = []
        policy.reset()
        obs = env.reset()
        for i in range(episode_length):
            # step
            action = policy.predict(obs, env)
            obs, rew, done, info = env.step(action)
            images.append(obs[0])
            robotstates.append(obs[1])
            actions.append(action)
            rewards.append(rew)
            dones.append(done)
            labels.append(info["segmentation_image"])
            if render:
                env.render()
            if done:
                policy.reset()
                obs = env.reset()
            print("{} - {} {}".format(n, i, "done" if done else "     "), end="\r")
        dones[-1] = True

        images = np.array(images)
        labels = np.array(labels)
        robotstates = np.array(robotstates)
        actions = np.array(actions)
        rewards = np.array(rewards)
        dones = np.array(dones)
        data = dict(images=images, labels=labels,
                    robotstates=robotstates, actions=actions, rewards=rewards, dones=dones)
        if archive_dir is not None:
            make_dir_if_not_exists(archive_dir)
            archive_path = os.path.join(
                archive_dir, "{:03}_images_labels.npz".format(n)
            )
            np.savez_compressed(archive_path, **data)
            print(archive_path, "written.")
    env.close()
    return data

def visual_archive_check(archive_dir):
    from matplotlib import pyplot as plt
    archive_path = os.path.join(archive_dir, "000_images_labels.npz")
    data = np.load(archive_path)
    images = data["images"]
    labels = data["labels"]
    actions = data["actions"]
    dones = data["dones"]
    robotstates = data["robotstates"]
    plt.figure("check")
    for i, (im, lb, a, d, rs) in enumerate(zip(images, labels, actions, dones, robotstates)):
        plt.clf()
        fig, (ax1, ax2) = plt.subplots(1, 2, num="check")
        ax1.imshow(im)
        ax1.set_title("image {}".format(i))
        ax2.imshow(lb)
        ax2.set_title("labels")
        fig.suptitle(archive_path + "\n" + "{} {} {}".format(a, d, rs))
        plt.pause(0.1)

def main(n_sequences=100, env="S", render=False, dry_run=False,
         subproc_id=0, n_subprocs=1,
         check_archive=False):
    np.set_printoptions(precision=2, suppress=True)
    archive_dir = os.path.expanduser("~/navrep3d_W/datasets/multitask/navrep3dalt_segmentation")
    if dry_run:
        archive_dir = "/tmp/navrep3d/datasets/multitask/navrep3dalt_segmentation"
    if check_archive:
        visual_archive_check(archive_dir)
        return
    build_name = "./alternate_segmentation.x86_64"
    env = NavRep3DTrainEnv(verbose=0, collect_statistics=False,
                           build_name=build_name, port=25005+subproc_id)
    policy = SemiRandomMomentumPolicy()
    generate_segmentation_dataset(
        env, n_sequences=n_sequences,
        subset_index=subproc_id, n_subsets=n_subprocs,
        policy=policy,
        render=render, archive_dir=archive_dir)


if __name__ == "__main__":
    StrictFire(main)