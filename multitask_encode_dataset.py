import numpy as np
import os
from tqdm import tqdm
from strictfire import StrictFire

from navdreams.encodedenv3d import EnvEncoder

from multitask_generate_labels import basic_archive_check

_RS = 5
_H = 64
encoder_types = ["E2E", "N3D", "sequenceN3D"]

def check_encodings(archive_dir):
    basic_archive_check(archive_dir, filename_mask=".npz")
    for encoder_type in encoder_types:
        filenames = []
        for dirpath, dirnames, dirfilename in os.walk(archive_dir):
            for filename in [
                f
                for f in dirfilename
                if f.endswith("_{}encodings_labels.npz".format(encoder_type))
            ]:
                filenames.append(os.path.join(dirpath, filename))
        filenames = sorted(filenames)
        all_encodings = []
        print()
        print(encoder_type)
        print("{} files found.".format(len(filenames)))
        for archive_file in filenames:
            archive_path = os.path.join(archive_dir, archive_file)
            data = np.load(archive_path)
            encodings = data["encodings"]
            all_encodings.append(encodings)
        encodings = np.concatenate(all_encodings, axis=0)
        print("min {} max {}".format(np.min(encodings), np.max(encodings)))
        print("avg {} std {}".format(np.mean(encodings), np.std(encodings)))
        if np.any(np.isnan(encodings)):
            raise ValueError


def main(dry_run=False, check_archive=False, gpu=True):
    np.set_printoptions(precision=2, suppress=True)
    archive_dir = os.path.expanduser("~/navrep3d_W/datasets/multitask/navrep3dalt_segmentation")
    if check_archive:
        check_encodings(archive_dir)
        return

    filenames = []
    for dirpath, dirnames, dirfilename in os.walk(archive_dir):
        for filename in [
            f
            for f in dirfilename
            if f.endswith("images_labels.npz")
        ]:
            filenames.append(os.path.join(dirpath, filename))
    filenames = sorted(filenames)

    for encoder_type in encoder_types:
        if encoder_type == "E2E":
            # img, rs -> 64
            modelpath = os.path.expanduser("~/navrep3d/models/gym/navrep3daltenv_2021_11_01__08_52_03_DISCRETE_PPO_E2E_VCARCH_C64_ckpt.zip") # noqa
            encoder = EnvEncoder("E2E", "V_ONLY", gpt_model_path=modelpath, gpu=gpu)
        elif encoder_type == "N3D":
            # img, rs -> 64
            # (can use z)
            modelpath = os.path.expanduser("~/navrep3d_W/models/W/transformer_Salt")
            encoder = EnvEncoder("GPT", "V_ONLY", gpt_model_path=modelpath, gpu=gpu)
        elif encoder_type == "sequenceN3D":
            # {(img, a, rs, d), (img, a, rs, d), ...} -> (64, 64, 64, 64)
            # (uses get_h)
            modelpath = os.path.expanduser("~/navrep3d_W/models/W/transformer_Salt")
            encoder = EnvEncoder("GPT", "M_ONLY", gpt_model_path=modelpath, gpu=gpu)
        else:
            raise NotImplementedError

        for archive_file in filenames:
            archive_path = os.path.join(archive_dir, archive_file)
            data = np.load(archive_path)
            print("{} loaded.".format(archive_path))
            images = data["images"]
            labels = data["labels"]
            depths = data["depths"]
            actions = data["actions"]
            rewards = data["rewards"]
            dones = data["dones"]
            robotstates = data["robotstates"]

            encodings = []
            for image, label, depth, action, done, robotstate in tqdm(zip(
                    images, labels, depths, actions, dones, robotstates)):
                obs = (image, robotstate)
                h = encoder._encode_obs(obs, action)
                h_no_rs = h[:-_RS]
                assert h_no_rs.shape == (_H,)
                if done:
                    encoder.reset()
                encodings.append(h_no_rs)
            encoder.reset()
            encodings = np.array(encodings)
            data = dict(encodings=encodings, labels=labels, depths=depths,
                        robotstates=robotstates, actions=actions, rewards=rewards, dones=dones,
                        modelpath=np.array(modelpath))
            write_path = archive_path.replace("_images_", "_{}encodings_".format(encoder_type))
            if dry_run:
                write_path = os.path.join("/tmp", os.path.basename(write_path))
            np.savez_compressed(write_path, **data)
            print(write_path, "written.")


if __name__ == "__main__":
    StrictFire(main)
