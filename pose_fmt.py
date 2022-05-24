import os
import numpy as np

from itertools import chain
from collections import defaultdict
from pyquaternion import Quaternion
import glob
import time

import ifr.ifr_main as ifr
from ifr.ifr_main import vis_param

import pdb


# namespace
args = vis_param.args
sequence = vis_param.sequence
# scene name
scene = args.scene
# where the pose stream is
elasticfusion_pose_folder = args.pose_folder
# where we store the incremental result for demonstration
os.makedirs(args.outdir,exist_ok=True)


contains = glob.glob(elasticfusion_pose_folder+'poses_*.txt')
idxs = [int(ct.split('.')[-2].split('_')[-1]) for ct in contains]
contains_dict = {}
for i,idx in enumerate(idxs):
    contains_dict[idx] = contains[i]

    
def read_elasticfusion_file(pose_id, kf_idx):
    with open(contains_dict[pose_id]) as f:
        lines = f.readlines()
    poses = []
    for line_id, line in enumerate(lines):
        if line_id in kf_idx:
            vs = [float(v) for v in line.strip().split(' ')]
            v_t = vs[1:4]
            #v_q = vs[4:] # xyzw
            v_q = Quaternion(vs[-1],*vs[4:-1])
            pose = v_q.transformation_matrix
            pose[:3,3] = np.array(v_t)
            poses.append(pose)
    return poses





if __name__ == '__main__':
    import os
    import sys
    from dataset_ptam import TUMRGBDDataset, ICLNUIMDataset, ReplicaRGBDDataset

    dataset = args.dataset_type

    if 'tum' in dataset.lower():
        dataset = TUMRGBDDataset(sequence.path)
    elif 'replica' in dataset.lower():
        dataset = ReplicaRGBDDataset(sequence.path)
    else:
        assert "Not supported data type"

    '''
        load gt traj to check correctness
    '''
    GT = args.use_gt
    if GT:
        gt_traj = np.genfromtxt(str(sequence.path)+'/livingRoom'+scene+'.gt.freiburg')
        gt_poses = []


    durations = []
    data_i = 0
    kf_idx = []
    def run_algo(vis):
        global data_i
        i = data_i#data_next()
        data_i += 1


        if i % 20 == 0:#
            is_keyframe = True
            kf_idx.append(i)
        else:
            is_keyframe = False
        if dataset.timestamps is None:
            timestamp = i / 20.
        else:
            timestamp = dataset.timestamps[i]

        time_start = time.time()  

        # 0. check if current keyframe
        if is_keyframe:
            gt_pose = gt_traj[i,:] if GT else None
            # 1. prepare current frame to get torch frame_data
            frame_data = (dataset.rgb[i],dataset.depth[i])

            # 2. get all the poses of keyframe
            new_poses = []
            if not GT:
                poses = read_elasticfusion_file(i, kf_idx)
                new_poses= poses
            else:
                gt_poses.append(gt_pose)
                new_poses = gt_poses 

            # 3.2 if some pose changed, update map
            ifr.refresh(frame_data, new_poses, frame_id = i, vis=vis, ptam_p = not GT, scene_name = 'lrkt'+scene)
        else:
            return


        duration = time.time() - time_start
        durations.append(duration)
        print('duration', duration)
        print()
        print()
        
    if ifr.engine:
        ifr.engine.register_animation_callback(callback_func = run_algo) 
        vis_ph = ifr.vis_util.wireframe_bbox([-4., -4., -4.], [4., 4., 4.])
        ifr.engine.add_geometry(vis_ph)
        ifr.engine.remove_geometry(vis_ph, reset_bounding_box=False)
        ifr.engine.run()
        ifr.engine.destroy_window()
    else:
        try:
            while True:
                run_algo(None)
        except Exception as e:
            print(e)



    print('num frames', len(durations))
    print('num keyframes', len(kf_idx))
    print('average time', np.mean(durations))

