import os
import numpy as np

import time
from itertools import chain
from collections import defaultdict
from scipy.spatial.transform import Rotation as R

import ifr.ifr_exp_transform as ifr
from ifr.ifr_exp_transform import vis_param

import glob
import pdb



from pyquaternion import Quaternion
 
# namespace
args = vis_param.args
sequence = vis_param.sequence
# scene name
scene = args.scene
# where we store the incremental result for demonstration
args.outdir_transform = args.outdir[:-1]+'_transform'
os.makedirs(args.outdir_transform,exist_ok=True)



if __name__ == '__main__':
    import os
    import sys

    from dataset_ptam import TUMRGBDDataset, ICLNUIMDataset
    

    args.dataset = args.dataset_type

    if 'tum' in args.dataset.lower():
        dataset = TUMRGBDDataset(sequence.path)
    else:
        assert "Not supported data type"

    '''
        load gt traj to check correctness
    '''
    GT = True
    if GT:
        gt_traj = np.genfromtxt(str(sequence.path)+'/livingRoom'+scene+'.gt.freiburg')
        gt_poses = []



    durations = []
    data_i = 0
    #for i in range(len(dataset))[:]:
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
            #gt_pose = g2o.Isometry3d(g2o.Quaternion(gt_traj[i,-1],gt_traj[i,4],gt_traj[i,5],gt_traj[i,6]), gt_traj[i,1:4])
            gt_pose = gt_traj[i,:] if GT else None
            # 1. prepare current frame to get torch frame_data
            frame_data = (dataset.rgb[i],dataset.depth[i])

            # 2. get all the poses of keyframe
            new_poses = []
            if not GT:
                assert(False)
                poses = read_elasticfusion_file(i, kf_idx)
                new_poses= poses
            else:
                gt_poses.append(gt_pose)
                new_poses = gt_poses 

            # 3.2 if some pose changed, update map
            ifr.refresh(frame_data, new_poses, frame_id = i, vis=vis, ptam_p = not GT)
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


