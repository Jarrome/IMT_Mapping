from im2mesh.encoder import (
    conv, pix2mesh_cond, pointnet,
    psgn_cond, r2n2, voxels, vnn, vnn_tnet, vnn2
)


encoder_dict = {
    'simple_conv': conv.ConvEncoder,
    'resnet18': conv.Resnet18,
    'resnet34': conv.Resnet34,
    'resnet50': conv.Resnet50,
    'resnet101': conv.Resnet101,
    'r2n2_simple': r2n2.SimpleConv,
    'r2n2_resnet': r2n2.Resnet,
    'pointnet_simple': pointnet.SimplePointnet,
    'pointnet_resnet': pointnet.ResnetPointnet,
    'psgn_cond': psgn_cond.PCGN_Cond,
    'voxel_simple': voxels.VoxelEncoder,
    'pixel2mesh_cond': pix2mesh_cond.Pix2mesh_Cond,
    'vnn_dgcnn': vnn.VNN_DGCNN,
    'vnn_pointnet_simple': vnn.VNN_SimplePointnet,
    'vnn_pointnet_resnet': vnn.VNN_ResnetPointnet,
    'vnn_tnet_simple': vnn_tnet.SimplePointnet,
    'vnn_tnet_resnet': vnn_tnet.ResnetPointnet,
    'vnn2_dgcnn': vnn2.VNN_DGCNN,
    'vnn2_pointnet_simple': vnn2.VNN_SimplePointnet,
    'vnn2_pointnet_resnet': vnn2.VNN_ResnetPointnet
}
