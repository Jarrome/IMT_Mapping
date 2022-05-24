import importlib
import open3d as o3d
import argparse
import logging
import time
import torch
import numpy as np

from threading import Thread

from utils import exp_util, vis_util
from utils.motion_util import Isometry, Quaternion
from network import utility
from system import map
import system.tracker
from dataset.production import FrameData, FrameIntrinsic 
from dataset.production.icl_nuim import ICLNUIMSequence
import pdb

vis_param = argparse.Namespace()
#vis_param.n_left_steps = 0
vis_param.args = None
vis_param.mesh_updated = True
vis_param.local_maps = dict()

parser = exp_util.ArgumentParserX()#base_config_path = '/home/yijun/Desktop/di_ifr_fusion/di-fusion/configs/ifr-fusion-lr-kt.yaml', add_hyper_arg = False)
args = parser.parse_args()
logging.basicConfig(level=logging.INFO)
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
vis_param.sequence = ICLNUIMSequence(**args.sequence_kwargs)

engine = None

debug = True






def update_geometry(geom, name, vis):
    if not isinstance(geom, list):
        geom = [geom]

    if name in vis_param.__dict__.keys():
        for t in vis_param.__dict__[name]:
            vis.remove_geometry(t, reset_bounding_box=False)
    for t in geom:
        vis.add_geometry(t, reset_bounding_box=False)
    vis_param.__dict__[name] = geom


world_pose = None
def ptamposes2difusionposes(ps, ptam_p = True):
    global world_pose
    cano_quat = Isometry(q=Quaternion(axis=[0.0, 0.0, 1.0], degrees=180.0))
    new_ps = []
    traj = []
    for pose in ps:
        if ptam_p:
            new_pose = Isometry.from_matrix(pose)
            cur_q = pose[:3,:3]#Quaternion(imaginary=cur_p[4:7], real=cur_p[-1]).rotation_matrix
            cur_t = pose[:3,3]
        else: # directly from traj
            cur_q = Quaternion(imaginary=pose[4:7], real=pose[-1]).rotation_matrix
            cur_t = pose[1:4]
        cur_q[1] = -cur_q[1]
        cur_q[:, 1] = -cur_q[:, 1]
        cur_t[1] = -cur_t[1]
        cur_iso = Isometry(q=Quaternion(matrix=cur_q), t=cur_t)
        camera_ext = cano_quat.dot(cur_iso)

        traj.append(camera_ext)
    if world_pose is None:
        change_iso = vis_param.sequence.first_iso.dot(traj[0].inv())
        world_pose = traj[0].inv()
    else:
        change_iso = vis_param.sequence.first_iso.dot(world_pose)

    new_ps = [change_iso.dot(t) for t in traj]




    return new_ps




def compare_poses(poses1, poses2):
    assert(len(poses1)==len(poses2))

    update_mask = np.zeros(len(poses1),dtype=bool)

    def compare_pose(ps1,ps2):
        return ps1.dot(ps2.inv()).matrix


    for i in range(len(poses1)):
        diffe = compare_pose(poses1[i],poses2[i])
        if (diffe-np.eye(4)).sum() > 1e-5:
            update_mask[i] = True
    if abs(update_mask.sum()) != 0:
        return update_mask
    else: 
        return None




def ptamframe2difusionframe(rgb_data,depth_data):
    frame_data = FrameData()
    depth_data = torch.from_numpy(depth_data.astype(np.float32)).cuda() / calib[4]
    rgb_data = torch.from_numpy(rgb_data).cuda().float() / 255.
    frame_data.calib = FrameIntrinsic(calib[0], calib[1], calib[2], calib[3], calib[4])
    frame_data.depth = depth_data
    frame_data.rgb = rgb_data

    return frame_data




def refresh(frame_data,ptam_poses,frame_id, vis, ptam_p=True, scene_name='tmp'):
    #if vis_param.n_left_steps == 0:
    #    return False


    #frame_data, ptam_poses, frame_id):
    # ptam frame to difusion frame
    frame_data = ptamframe2difusionframe(frame_data[0],frame_data[1])

    # ptam pose to difusion pose
    new_poses = ptamposes2difusionposes(ptam_poses, ptam_p=ptam_p)

    # start work!
    if vis:
        # This spares slots for meshing thread to emit commands.
        time.sleep(0.02)

    if not vis_param.mesh_updated and vis_param.args.run_async:
        map_mesh = vis_param.map.extract_mesh(vis_param.args.resolution, 0, extract_async=True)
        if map_mesh is not None:
            vis_param.mesh_updated = True
            update_geometry(map_mesh, "mesh_geometry", vis)


    #vis_param.n_left_steps -= 1

    logging.info(f"Frame ID = {frame_id}")
    #frame_data = next(vis_param.sequence)
    #if (vis_param.sequence.frame_id - 1) % vis_param.args.integrate_interval == 0: # Is keyframe


    # Prune invalid depths
    frame_data.depth[torch.logical_or(frame_data.depth < vis_param.args.depth_cut_min,
                                      frame_data.depth > vis_param.args.depth_cut_max)] = np.nan

    # 1. set_pose to tracker.
    frame_pose = vis_param.tracker.track_camera(frame_data.rgb, frame_data.depth, frame_data.calib,
                                            set_pose = new_poses[-1]) #vis_param.sequence.first_iso if len(vis_param.tracker.all_pd_pose) == 0 else new_poses[-1].dot(vis_param.sequence.first_iso))
    tracker_pc, tracker_normal = vis_param.tracker.last_processed_pc

    if vis:
        pc_geometry = vis_util.pointcloud(frame_pose @ tracker_pc.cpu().numpy())
        update_geometry(pc_geometry, "pc_geometry", vis)
        update_geometry(vis_util.frame(), "frame", vis)
        update_geometry(vis_util.trajectory([t.t for t in vis_param.tracker.all_pd_pose]), "traj_geometry", vis)
        update_geometry(vis_util.camera(frame_pose, scale=0.15, color_id=3), "camera_geometry", vis)


    # keyframe selection from ptam, the inputed is forsure keyframe
    #if (vis_param.sequence.frame_id - 1) % vis_param.args.integrate_interval == 0: # Is keyframe
    if True:
        # 2 establish local frames 
        opt_depth = frame_pose @ tracker_pc
        opt_normal = frame_pose.rotation @ tracker_normal
        # 2.1 initial a local frame
        local_map = map.DenseIndexedMap(model, args.mapping, args.model.code_length, main_device,
                                        args.run_async, aux_device) # first pose is important
        local_map.integrate_keyframe(opt_depth, opt_normal, async_optimize=vis_param.args.run_async,
                                         do_optimize=False)        
        # 2.2 create intermedia that local_map is fixed, during loop, merely intermedia will changed . 
        local_intermedia = local_map.extract_interm() # latent_vecs, vpos, voxel_obs_count
        vis_param.local_maps[frame_id]=[local_map, frame_pose, frame_pose, local_intermedia] # map, initial_pose, looped_pose, intermedia


        # 2.3 update frame i to global
        if not debug:
            vis_param.map.integrate_keyframe(opt_depth, opt_normal, async_optimize=vis_param.args.run_async,
                                             do_optimize=False)

        # 3. detecte loop, return a mask with changed frames
        loop_signal = compare_poses(vis_param.tracker.all_pd_pose, new_poses)


        no_cache = False # This is for mesh

        if loop_signal is None:
            loop_signal = np.zeros(len(new_poses),dtype=bool)
        loop_signal[-1] = True
        '''
        loop_signal[:] = True
        rot = np.array([[  0.3611576,  0.8962684, -0.2574260],
  [-0.2574260,  0.3611576,  0.8962684],
  [ 0.8962684, -0.2574260,  0.3611576 ]])


        tmp_iso = Isometry(q=Quaternion(matrix=rot), t=np.array([0.1,0,.1]))
        tmp_iso_ = Isometry(q=Quaternion(matrix=rot), t=np.array([0.1,0,.1]))
        '''



        if True:
            #loop_signal = np.ones(loop_signal.shape)
            frame_poses = []
            js = np.where(loop_signal)[0]
            local_map_find_table = np.fromiter(vis_param.local_maps.keys(),dtype=int)
            for j in js:#np.fromiter(vis_param.local_maps.keys(),dtype=int)[js]:
                j_frame_id = local_map_find_table[j]
                if j == js[-1]: # new frame
                    interm = vis_param.local_maps[j_frame_id][3]
                    vis_param.map.fuse_local(interm)
                    continue


                # 2.3 global fm remove local fm intermedia
                st = time.time()
                vis_param.map.remove_local(vis_param.local_maps[j_frame_id][3])
                print('REMOVE TAKE:      ', time.time()-st)
                print('after remove',vis_param.map.latent_vecs.shape)

                # 2.5 transform each fm with changed pose, merely rot feat
                '''
                for i in range(js.shape[0]):
                    tmp_iso = tmp_iso.dot(tmp_iso_)
                print(tmp_iso)
                '''
                interm_ = vis_param.local_maps[j_frame_id][0].transform(vis_param.local_maps[j_frame_id][1],new_poses[j].matrix)
                #interm_ = vis_param.local_maps[j_frame_id][0].transform(vis_param.local_maps[j_frame_id][1],tmp_iso.dot(new_poses[j]).matrix)

                vis_param.local_maps[j_frame_id][2] = new_poses[j]#loop_det.graph.kfs[j].pose
                vis_param.local_maps[j_frame_id][3] = interm_
                print(interm_[1].max(1))
                # 2.6 update new local fm into global, interpolate
                st = time.time()
                vis_param.map.fuse_local(interm_)
                print('FUSE TAKE:      ', time.time()-st)
                print('after fuse',vis_param.map.latent_vecs.shape)
                # 2.7 update poses in tracker [make all_pd_pose the same as new_poses]
                vis_param.tracker.all_pd_pose[j] = new_poses[j]


                #vis_param.tracker.all_pd_pose[j] = di_Isometry.from_matrix(mat=loop_det.graph.kfs[j].pose.matrix())

            no_cache = True

        if vis:
            if loop_signal is not None:
                vis.clear_geometries()
            fast_preview_vis = vis_param.map.get_fast_preview_visuals()
            update_geometry(fast_preview_vis[0], "block_geometry", vis)
            update_geometry((vis_util.wireframe_bbox(vis_param.map.bound_min.cpu().numpy(),
                                                     vis_param.map.bound_max.cpu().numpy(), color_id=4)), "bound_box", vis)
            map_mesh = vis_param.map.extract_mesh(vis_param.args.resolution, int(float(vis_param.args.max_n_triangles)), max_std=vis_param.args.max_std,
                                                  extract_async=vis_param.args.run_async, interpolate=True, no_cache = no_cache)
            vis_param.mesh_updated = map_mesh is not None
            if map_mesh is not None:
                o3d.io.write_triangle_mesh(args.outdir+'/recons_%d.ply'%(frame_id), map_mesh)
                poses = [new_pose.matrix for new_pose in new_poses]
                with open(args.outdir+'/poses_%d.npy'%(frame_id), 'wb') as f:
                    np.save(f, np.stack(poses))


                # Note: This may be slow:
                # map_mesh.merge_close_vertices(0.01)
                # map_mesh.compute_vertex_normals()
                update_geometry(map_mesh, "mesh_geometry", vis)
        print('pass')













#vis_param.sequence = ICLNUIMSequence(**args.sequence_kwargs)

# Load in network.  (args.model is the network specification)
model, args_model = utility.load_model(args.training_hypers, args.using_epoch)
args.model = args_model
args.mapping = exp_util.dict_to_args(args.mapping)
args.tracking = exp_util.dict_to_args(args.tracking)

# Mapping
if torch.cuda.device_count() > 1:
    main_device, aux_device = torch.device("cuda", index=0), torch.device("cuda", index=1)
elif torch.cuda.device_count() == 1:
    main_device, aux_device = torch.device("cuda", index=0), None
else:
    assert False, "You must have one GPU."


vis_param.map = map.DenseIndexedMap(model, args.mapping, args.model.code_length, main_device,
                                    args.run_async, aux_device)
vis_param.tracker = system.tracker.SDFTracker(vis_param.map, args.tracking)

vis_param.args = args

#calib = [481.2, 480.0, 319.50, 239.50, 5000.0] # for ICL-NUIM
#calib = [525.0, 525.0, 319.5, 239.5, 5000]
calib = args.calib

if args.vis:
    # Run the engine. Internal clock driven by Open3D visualizer.
    engine = o3d.visualization.Visualizer()
    engine.create_window(window_name="Implicit SLAM", width=1280, height=720, visible=True)
    engine.get_render_option().mesh_show_back_face = True





