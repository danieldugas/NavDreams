import os
from navrep.tools.commonargs import parse_multiproc_args
from navrep.scripts.make_vae_dataset import generate_vae_dataset, SemiRandomMomentumPolicy, HumanControlPolicy

from navrep3d.navrep3dtrainenv import NavRep3DTrainEnv

if __name__ == "__main__":
    args, _ = parse_multiproc_args()
    n_sequences = 100
    if args.n is not None:
        n_sequences = args.n

    TYPE = "S"

    if TYPE == "S":
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dtrain")
        if args.dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dtrain"
        build_name = None
    elif TYPE == "C": # C
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dcity")
        if args.dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dcity"
        build_name = "./city.x86_64"
    elif TYPE == "R": # R
        archive_dir = os.path.expanduser("~/navrep3d_W/datasets/V/navrep3dreal")
        if args.dry_run:
            archive_dir = "/tmp/navrep3d/datasets/V/navrep3dreal"
        raise NotImplementedError
    env = NavRep3DTrainEnv(verbose=0, collect_statistics=False,
                           build_name=build_name, port=25005+args.subproc_id)
    policy = SemiRandomMomentumPolicy() if True else HumanControlPolicy()
    generate_vae_dataset(
        env, n_sequences=n_sequences,
        subset_index=args.subproc_id, n_subsets=args.n_subprocs,
        policy=policy,
        render=args.render, archive_dir=archive_dir)
