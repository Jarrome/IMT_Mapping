import sys
import numpy as np
import cv2
import os
import time

from collections import defaultdict, namedtuple

from threading import Thread, Lock
from multiprocessing import Process, Queue
import glob
from tqdm import tqdm

import open3d as o3d



sys.path.append('/home/yijun/Desktop/overlap/OverlapTransformer/')
from tools.utils.utils import range_projection



import pdb


class ImageReader(object):
    def __init__(self, ids, timestamps=None, cam=None):
        self.ids = ids
        self.timestamps = timestamps
        self.cam = cam
        self.cache = dict()
        self.idx = 0

        self.ahead = 10      # 10 images ahead of current index
        self.waiting = 1.5   # waiting time

        self.preload_thread = Thread(target=self.preload)
        self.thread_started = False

    def read(self, path):
        img = cv2.imread(path, -1)
        if self.cam is None:
            return img
        else:
            return self.cam.rectify(img)
        
    def preload(self):
        idx = self.idx
        t = float('inf')
        while True:
            if time.time() - t > self.waiting:
                return
            if self.idx == idx:
                time.sleep(1e-2)
                continue
            
            for i in range(self.idx, self.idx + self.ahead):
                if i not in self.cache and i < len(self.ids):
                    self.cache[i] = self.read(self.ids[i])
            if self.idx + self.ahead > len(self.ids):
                return
            idx = self.idx
            t = time.time()
    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        self.idx = idx
        # if not self.thread_started:
        #     self.thread_started = True
        #     self.preload_thread.start()

        if idx in self.cache:
            img = self.cache[idx]
            del self.cache[idx]
        else:   
            img = self.read(self.ids[idx])
                    
        return img

    def __iter__(self):
        for i, timestamp in enumerate(self.timestamps):
            yield timestamp, self[i]

    @property
    def dtype(self):
        return self[0].dtype
    @property
    def shape(self):
        return self[0].shape




class ICLNUIMDataset(object):
    '''
    path example: 'path/to/your/ICL-NUIM R-GBD Dataset/living_room_traj0_frei_png'
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        481.20, 480.0, 319.5, 239.5, 5000)
    def __init__(self, path):
        path = os.path.expanduser(path)
        self.rgb = ImageReader(self.listdir(os.path.join(path, 'rgb')))
        self.depth = ImageReader(self.listdir(os.path.join(path, 'depth')))
        self.timestamps = None

    def sort(self, xs):
        return sorted(xs, key=lambda x:int(x[:-4]))

    def listdir(self, dir):
        files = [_ for _ in os.listdir(dir) if _.endswith('.png')]
        return [os.path.join(dir, _) for _ in self.sort(files)]

    def __len__(self):
        return len(self.rgb)




def make_pair(matrix, threshold=1):
    assert (matrix >= 0).all()
    pairs = []
    base = defaultdict(int)
    while True:
        i = matrix[:, 0].argmin()
        min0 = matrix[i, 0]
        j = matrix[0, :].argmin()
        min1 = matrix[0, j]

        if min0 < min1:
            i, j = i, 0
        else:
            i, j = 0, j
        if min(min1, min0) < threshold:
            pairs.append((i + base['i'], j + base['j']))

        matrix = matrix[i + 1:, j + 1:]
        base['i'] += (i + 1)
        base['j'] += (j + 1)

        if min(matrix.shape) == 0:
            break
    return pairs


class ReplicaRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''
    cam = namedtuple('camera', 'fx fy cx cy scale')(
            600., 600., 599.5, 339.5, 6553.5)

    def __init__(self, path, register=True):
        path = os.path.expanduser(path)

        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'rgb', ext='.jpg')
            depth_ids, depth_timestamps = self.listdir(path, 'depth')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'rgb', ext='.jpg')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])

        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

    def sort(self, xs, st = 3):
        return sorted(xs, key=lambda x:float(x[st:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        st = 5
        for name in self.sort(files,st):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[st:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)


class NeuralSurfaceRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''
    cam = namedtuple('camera', 'fx fy cx cy scale')(
        554.2562584220408, 554.2562584220408, 320., 240., 1000)

    def __init__(self, path, register=True):
        path = os.path.expanduser(path)

        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'images')
            depth_ids, depth_timestamps = self.listdir(path, 'depth_filtered')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'images')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth_filtered')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])

        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

    def sort(self, xs, st = 3):
        return sorted(xs, key=lambda x:float(x[st:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        st = 3 if split == "images" else 5
        for name in self.sort(files,st):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[st:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)

class TUMRGBDDataset(object):
    '''
    path example: 'path/to/your/TUM R-GBD Dataset/rgbd_dataset_freiburg1_xyz'
    '''
    cam = namedtuple('camera', 'fx fy cx cy scale')(
        481.20, 480.0, 319.5, 239.5, 5000)
    '''

    cam = namedtuple('camera', 'fx fy cx cy scale')(
        525.0, 525.0, 319.5, 239.5, 5000)
    '''

    def __init__(self, path, register=True):
        path = os.path.expanduser(path)

        if not register:
            rgb_ids, rgb_timestamps = self.listdir(path, 'rgb')
            depth_ids, depth_timestamps = self.listdir(path, 'depth')
        else:
            rgb_imgs, rgb_timestamps = self.listdir(path, 'rgb')
            depth_imgs, depth_timestamps = self.listdir(path, 'depth')
            
            interval = (rgb_timestamps[1:] - rgb_timestamps[:-1]).mean() * 2/3
            matrix = np.abs(rgb_timestamps[:, np.newaxis] - depth_timestamps)
            pairs = make_pair(matrix, interval)

            rgb_ids = []
            depth_ids = []
            for i, j in pairs:
                rgb_ids.append(rgb_imgs[i])
                depth_ids.append(depth_imgs[j])

        self.rgb = ImageReader(rgb_ids, rgb_timestamps)
        self.depth = ImageReader(depth_ids, depth_timestamps)
        self.timestamps = rgb_timestamps

    def sort(self, xs):
        return sorted(xs, key=lambda x:float(x[:-4]))

    def listdir(self, path, split='rgb', ext='.png'):
        imgs, timestamps = [], []
        files = [x for x in os.listdir(os.path.join(path, split)) if x.endswith(ext)]
        for name in self.sort(files):
            imgs.append(os.path.join(path, split, name))
            timestamp = float(name[:-len(ext)].rstrip('.'))
            timestamps.append(timestamp)

        return imgs, np.array(timestamps)

    def __getitem__(self, idx):
        return self.rgb[idx], self.depth[idx]

    def __len__(self):
        return len(self.rgb)

# -------------------------------------------------------------------------------------
#                         KITTI
# -------------------------------------------------------------------------------------


def load_velo_scan(file):
    """Load and parse a velodyne binary file."""
    scan = np.fromfile(file, dtype=np.float32) 
    return scan.reshape((-1,4))

def prepare_kitti_data_v3(path0,valid_distance):
    p0_ = load_velo_scan(path0)
    p0 = p0_[:,:3]

    valid_mask = (p0**2).sum(1) < valid_distance**2
    p0 = p0[valid_mask,:]

    p0_src = o3d.geometry.PointCloud()
    p0_src.points = o3d.utility.Vector3dVector(p0[:,:])
    downpcd0 = p0_src.voxel_down_sample(voxel_size=0.35)
    
    return p0, np.asarray(downpcd0.points)


def prepare_kitti_data(path0, normal = False, use_range = False, pc=True):
    p0_ = load_velo_scan(path0)
    p0 = p0_[:,:3]

    if use_range:
        current_range, _, _, _= range_projection(p0_, fov_up=3, fov_down=-25.0, proj_H=64, proj_W=900, max_range=50)
        if pc == False:
            return None, None, current_range
    else:
        current_range = None

    p0_src = o3d.geometry.PointCloud()
    p0_src.points = o3d.utility.Vector3dVector(p0[:,:])
    downpcd0 = p0_src.voxel_down_sample(voxel_size=0.2)
    
    if normal:
        downpcd0.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                          max_nn=30))
        p0 = np.asarray(downpcd0.points)
        n0 = np.asarray(downpcd0.normals)
    else:

        p0 = np.asarray(downpcd0.points)
        n0 = None
    return p0, n0, current_range

def normal(p):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5,
                                                          max_nn=30))
    return  np.asarray(pcd.normals)


class KITTIDataset_v3(object):
    def __init__(self, basedir="/media/yijun/2021/dataset/kitti_odometry/dataset/sequences/",
                        sequence=0):
        #"/media/yijun/2021/dataset/kitti_odometry/dataset/sequences/00/velodyne/"
        self.path = os.path.join(basedir,'%02d'%sequence)
        self.files = glob.glob(os.path.join(self.path,'velodyne','*.bin'))
        self.files = sorted(self.files, key=lambda x:float(x[-10:-4]))

    def __getitem__(self, idx):
        filename = self.files[idx]
        pc = False
        pc, pc_sampled= prepare_kitti_data_v3(filename, valid_distance=1000)
        return pc, pc_sampled
    def __len__(self):
        return len(self.files)
    




class KITTIDataset_v2(object):
    def __init__(self, basedir="/media/yijun/2021/dataset/kitti_odometry/dataset/sequences/",
                        sequence=0):
        #"/media/yijun/2021/dataset/kitti_odometry/dataset/sequences/00/velodyne/"
        self.path = os.path.join(basedir,'%02d'%sequence)
        self.files = glob.glob(os.path.join(self.path,'velodyne','*.bin'))
        self.files = sorted(self.files, key=lambda x:float(x[-10:-4]))

    def __getitem__(self, idx):
        filename = self.files[idx]
        pc = False
        depth, normal, range_ = prepare_kitti_data(filename, normal=True, use_range=True, pc=pc)
        if pc == False:
            return None, None, range_
        valid = (depth**2).sum(1) < 40**2
        #valid = valid * (self.depth[idx][:,2] > -1)
        #draw(self.depth[idx][valid,:])
        return depth[valid,:], normal[valid,:], range_
    def __len__(self):
        return len(self.files)
    




class KITTIDataset(object):
    def __init__(self, basedir="/media/yijun/2021/dataset/kitti_odometry/dataset/sequences/",
                        sequence=0):
        #"/media/yijun/2021/dataset/kitti_odometry/dataset/sequences/00/velodyne/"
        self.path = os.path.join(basedir,'%02d'%sequence)
        self.files = glob.glob(os.path.join(self.path,'velodyne','*.bin'))
        self.files = sorted(self.files, key=lambda x:float(x[-10:-4]))
        self.depth = []
        self.pns = []
        for filename in tqdm(self.files[:]):
        #self.depth = [prepare_kitti_data(filename) for filename in self.files]
        #self.pns = [normal(p) for p in self.depth]
            depth, normal = prepare_kitti_data(filename, normal=True) 
            #self.depth.append(prepare_kitti_data(filename))
            #self.pns.append(normal(self.depth[-1]))
            #self.pns.append(self.depth[-1])

            self.depth.append(depth)
            self.pns.append(normal)

    def __getitem__(self, idx):
        valid = (self.depth[idx]**2).sum(1) < 40**2
        #valid = valid * (self.depth[idx][:,2] > -1)
        #draw(self.depth[idx][valid,:])
        return self.depth[idx][valid,:], self.pns[idx][valid,:]
    def __len__(self):
        return len(self.depth)
    
class KITTIDataset_onread(object):
    def __init__(self, basedir="/media/yijun/2021/dataset/kitti_odometry/dataset/sequences/",
                        sequence=0):
        #"/media/yijun/2021/dataset/kitti_odometry/dataset/sequences/00/velodyne/"
        self.path = os.path.join(basedir,'%02d'%sequence)
        self.files = glob.glob(os.path.join(self.path,'velodyne','*.bin'))
        self.files = sorted(self.files, key=lambda x:float(x[-10:-4]))
        self.depth = []
        self.pns = []
        '''
        for filename in tqdm(self.files[:]):
        #self.depth = [prepare_kitti_data(filename) for filename in self.files]
        #self.pns = [normal(p) for p in self.depth]
            depth, normal,_ = prepare_kitti_data(filename, normal=True) 
            #self.depth.append(prepare_kitti_data(filename))
            #self.pns.append(normal(self.depth[-1]))
            #self.pns.append(self.depth[-1])

            self.depth.append(depth)
            self.pns.append(normal)
        '''

    def __getitem__(self, idx):
        depth, normal, _ = prepare_kitti_data(self.files[idx], normal=True)

        valid = (depth**2).sum(1) < 40**2
        #valid = valid * (self.depth[idx][:,2] > -1)
        #draw(self.depth[idx][valid,:])
        return depth[valid,:], normal[valid,:]
    def __len__(self):
        return len(self.files)




