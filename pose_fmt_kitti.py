import numpy as np
import time
from itertools import chain
from collections import defaultdict
import glob
import torch

from network import utility
from utils.coding import encode, decode, loadcode

import open3d as o3d
import time
from utils.motion_util import Isometry
from tqdm import tqdm

import pdb


if __name__ == '__main__':
    import cv2

    import os
    import sys
    #import argparse
    from utils import exp_util
    from dataset.kitti_dataset import KITTIDataset_onread

    parser = exp_util.ArgumentParserX()
    args = parser.parse_args()


    model, args_model = utility.load_model(args.training_hypers, args.using_epoch)
    args.model = args_model
    args.mapping = exp_util.dict_to_args(args.mapping)

    if torch.cuda.device_count() > 1:
        main_device, aux_device = torch.device("cuda", index=0), torch.device("cuda", index=1)
    elif torch.cuda.device_count() == 1:
        main_device, aux_device = torch.device("cuda", index=0), None
    else:
        assert False, "You must have one GPU."

    dataset = KITTIDataset_onread(args.basedir, args.sequence)


    # now start iterating
    pose_dir = './treasure/pyicp_slam_record/kitti00'
    os.makedirs('./output',exist_ok=True)
    from os.path import join
    import glob

    #pose_files = np.loadtxt(open(pose_dir, "rb"), delimiter=",", skiprows=1)
    pi_lms = dict()
    pose_stack = dict()
    interm_stack = dict()
    last_loop = 0
    for frame_i in tqdm(range(0, len(dataset))):
        # 1. get poses
        unoptimized_ps_file = glob.glob(join(pose_dir,'pose%d_unopt*'%(frame_i)))
        optimized_ps_file = glob.glob(join(pose_dir,'pose%d_opt*'%(frame_i)))

        assert(len(unoptimized_ps_file) == 1)
        assert(len(optimized_ps_file) <= 1)

        optimized_poses = None if len(optimized_ps_file) == 0 else np.loadtxt(open(optimized_ps_file[0], "rb"), delimiter=",", skiprows=0)

        if frame_i % 10 != 0:
            continue

        unoptimized_poses = np.loadtxt(open(unoptimized_ps_file[0], "rb"), delimiter=",", skiprows=0)
       
        if frame_i == 0:
            pose = unoptimized_poses.reshape((4,4))
        else:
            unoptimized_poses[:,-1] = 1
            pose = unoptimized_poses[frame_i,:].reshape((4,4))
        pose_stack[frame_i] = pose
        # transform frame
        pi, pin = dataset[frame_i]
        pi = (pose[:3,:3]@pi.T + pose[:3,(3,)]).T
        pin = (pose[:3,:3]@pin.T).T 

        # 
        pi_lm = encode(pi, pin, model, main_device, aux_device, args)
        store = [pi_lm.extract_interm(), pose, pose]
        interm_stack[frame_i] = store
        pi_lms[frame_i] = pi_lm
        if optimized_poses is not None and (frame_i - last_loop > 1000 or frame_i > 4520):
            optimized_poses[:,-1] = 1
            last_loop = frame_i
            optimized_poses = optimized_poses.reshape((-1,4,4))
            unoptimized_poses = unoptimized_poses.reshape((-1,4,4))
            print('Loooooooop')
            for opt_i in tqdm(list(interm_stack.keys())[1:-1]):
                if ((optimized_poses[opt_i,:3,:]-interm_stack[opt_i][2][:3,:])**2).sum() > 1e-3:
                    #interm_old = pi_lms[opt_i].extract_interm()
                    interm_old, init_pose, last_old_pose = interm_stack[opt_i]
                    pi_lms[0].remove_local(interm_old)

                    interm_new = pi_lms[opt_i].transform(init_pose, optimized_poses[opt_i])

                    pi_lms[0].fuse_local(interm_new)
                    pose_stack[opt_i] = optimized_poses[opt_i]
                    interm_stack[opt_i] = [interm_new, init_pose, optimized_poses[opt_i]]

        if frame_i > 0:
            interm_i = pi_lms[frame_i].extract_interm()
            pi_lms[0].fuse_local(interm_i)


        if frame_i in [0, 500, 1800, 4540]:
            pi_lms[0].save(os.path.join('./output/recons_%d.pt'%frame_i))
            map_mesh = pi_lms[0].extract_mesh(args.resolution, int(4e6), max_std=1e5,
                    extract_async=args.run_async, interpolate=True, no_cache = True, large_scale=False)
            o3d.io.write_triangle_mesh('./output/recons_%d.ply'%frame_i, map_mesh)




