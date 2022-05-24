import importlib
import open3d as o3d
import argparse
import logging
import time
import torch
import numpy as np

from threading import Thread
#import system.map as mapping
import system.map_dictindexer as mapping

import pdb
import time


def encode(depth, normal, model, main_device, aux_device, args):
    if type(depth) == np.ndarray:
        depth = torch.from_numpy(depth.astype(np.float32)).cuda()
        normal = torch.from_numpy(normal.astype(np.float32)).cuda()
    st = time.time()
    local_map = mapping.DenseIndexedMap(model, args.mapping, args.model.code_length, main_device,
                                        args.run_async, aux_device) # first pose is important
    #print('init map', time.time()-st)
    local_map.integrate_keyframe(depth, normal, async_optimize=args.run_async,
                                         do_optimize=False)        
    #print('integrate', time.time()-st)
    return local_map

def decode(I_map, xyz:torch.Tensor):
    mean, sigma, mask, gd = I_map.get_sdf(xyz)

def loadcode(cold_var_file, model, main_device, aux_device, args):
    local_map = mapping.DenseIndexedMap(model, args.mapping, args.model.code_length, main_device,
                                        args.run_async, aux_device) 
    local_map.load(cold_var_file)
    return local_map


