import numpy as np
import open3d as o3d

import sys
if len(sys.argv) == 2:
    mesh = o3d.io.read_triangle_mesh(sys.argv[1])
    mesh.paint_uniform_color([1, 0.706, 0])
    o3d.visualization.draw_geometries([mesh])

else:
    mesh1 = o3d.io.read_triangle_mesh(sys.argv[1])
    mesh1.paint_uniform_color([1.,0,0])
    p1 = np.asarray(mesh1.vertices)

    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(p1)
    pcd1.colors = o3d.utility.Vector3dVector(np.array([[255,0,0]]*p1.shape[0]))


    mesh2 = o3d.io.read_triangle_mesh(sys.argv[2])
    mesh2.paint_uniform_color([0,0,1.])
    p2 = np.asarray(mesh2.vertices)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(p2)
    pcd2.colors = o3d.utility.Vector3dVector(np.array([[0,0,255]]*p2.shape[0]))


    o3d.visualization.draw_geometries([mesh1,mesh2])
    o3d.visualization.draw_geometries([pcd1,pcd2])


