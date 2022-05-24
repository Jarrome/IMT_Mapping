import sys
import numpy as np
from scipy.spatial import cKDTree
import transform
from transform import utils
import torch
from time import time

import pdb

def _transform(T, fm, fm_v):
    '''
        T: np(4,4)
        fm: np(v,c) # v voxels and c dims
        fm_v: np(v,3) # stores the center of that voxel
    '''
    v,c =  fm.shape
    T = T.to(torch.float32)
    # 1. transform the voxel center
    fm_v_post = (T[:3,:3].matmul(fm_v.T) + T[:3,(3,)]).T # v,3

    # 2. rotation of features
    fm_3 = fm.reshape((v,-1,3)).reshape((-1,3))
    fm_3_post = T[:3,:3].matmul(fm_3.T).T.reshape((v,-1,3))
    fm_post = fm_3_post.reshape((v,c))

    return fm_post, fm_v_post

def interpolate(fm,fm_v,fm_v_tgt,voxel_size):
    '''
        align fm_v to fm_v_tgt
    '''
    v,c = fm.shape
    # search tree
    tree = cKDTree(fm_v,k=8)
    dd,idxs = tree.query(fm_v_tgt) # (v_gtg,8)
    valid = ((dd>=voxel_size/2).sum(axis=1) == 0) # have to be inside of 8 points

    valid_nm = valid.shape[0]

    # weight
    w = np.exp(-dd[valid,:]**2/voxel_size)  
    w = w / np.sum(w,axis=1, keepdims=True) # v_gtg,8 

    # average
    fm_avg = fm[ii[valid,:].reshape(-1),:].reshape((valid_nm,8,c)) * w[...,np.newaxis]  #  valid_nm,8,c
    fm_avg = fm_avg.sum(-2)

    return fm_avg, fm_v_tgt[valid,:]

def average_hypothesis(fm,fm_x,fm_x_tgt,Jm,T,voxel_size,fm_obs_count):
    '''
        align fm_x to fm_x_tgt
    '''
    v,c = fm.shape
    # search tree
    tree = cKDTree(fm_x.cpu().detach().numpy())
    n_gtg ,_ = fm_x_tgt.shape
    k = 8
    dd,idxs = tree.query(fm_x_tgt.cpu().detach().numpy(),k=k) # (v_gtg,8)
    if k == 1:
        dd = dd[:,np.newaxis]
        idxs = idxs[:,np.newaxis]

    valid = ((dd>=voxel_size*.86).sum(axis=1) < k) # have to be near the fm_x
    valid_nm = valid.sum()


    idxs = idxs[valid,:]
    # find v, a vector from fm_v to target
    nb_xyz = fm_x[idxs.reshape((-1)),:] # valid_nm*8,3
    nb_xyz = nb_xyz.reshape(valid_nm,k,3)
    # fix bug, should be negative
    v = -(nb_xyz - fm_x_tgt[valid,:].unsqueeze(1))/voxel_size  # valid_nm,8,3



    # weight
    w_ = torch.exp(-torch.tensor(dd[valid,:]).to(fm)**2/voxel_size)  
    w = w_ / torch.sum(w_,axis=1, keepdim=True) # v_gtg,8 

    # average
    #fm_avg = fm[ii[valid,:].reshape(-1),:].reshape((valid_nm,8,c))  #  valid_nm,8,c

    # get 
    #T_xi = torch.eye(4)
    #T_xi = T_xi.unsqueeze(0).unsqueeze(0).expand((valid_nm, 8, 1, 1)) # (valid_nm,8,4,4)
    # T is 1,4,4
    T = T.unsqueeze(0).expand(valid_nm,4,4).to(torch.float32)
    T_xi = T[:,:3,:3].transpose(1,2).bmm(v.transpose(1,2)).transpose(1,2) # (valid_nm, 8, 3)
    delta_xi = T_xi

    #delta_xi = se3.log(T_xi) # (valid_nm, 8, 6)
    fm_8 = fm[idxs.reshape((-1)),:].reshape((valid_nm,k,c)) # valid_nm, 8, c
    Jm_8 = Jm[idxs.reshape((-1)),:,:].reshape((valid_nm,k,c,3))
    fm_hp = fm_8 + torch.einsum('nijk,nik->nij', Jm_8, delta_xi) # valid_nm, 8, c
    fm_avg = (fm_hp * w.unsqueeze(-1) ).sum(1)


    # count
    fm_obs_count_avg = fm_obs_count[idxs.reshape(-1)].reshape((valid_nm,k))
    fm_obs_count_avg = (fm_obs_count_avg * w).sum(1)



    return fm_avg, fm_x_tgt[valid,:], fm_obs_count_avg



def interpolate(fm,fm_v,fm_v_tgt,voxel_size,fm_obs_count):
    '''
        align fm_v to fm_v_tgt
    '''
    v,c = fm.shape
    # search tree
    tree = cKDTree(fm_v,k=8)
    dd,idxs = tree.query(fm_v_tgt) # (v_gtg,8)
    valid = ((dd>=voxel_size).sum(axis=1) == 0) # have to be inside of 8 points
    valid_nm = valid.shape[0]

    # weight
    w = np.exp(-dd[valid,:]**2/voxel_size)  
    w = w / np.sum(w,axis=1, keepdims=True) # v_gtg,8 

    # average
    fm_avg = fm[idxs[valid,:].reshape(-1),:].reshape((valid_nm,8,c)) * w[...,np.newaxis]  #  valid_nm,8,c
    fm_avg = fm_avg.sum(-2)

    #
    obs_count = fm_obs_count[idxs[valid]].reshape(-1).reshape((valid_nm,8)) * w
    obs_count = obs_count.sum(-1)

    return fm_avg, fm_v_tgt[valid,:], obs_count



def contour_gvs(fm_x, voxel_size, x2v, v2x, center):
    '''
       global voxel is with eye(4) 
    '''
    #vs = (fm_x-bound_min.unsqueeze(0))/voxel_size 
    vs = x2v(fm_x, set_int=False)
    center_vs = x2v(center, set_int = False)

    ceils = torch.ceil(vs) # v,3
    floor = torch.floor(vs) # v,3

    table = dict()
    table[0] = ceils
    table[1] = floor

    cts = []

    # first fill in the best_v_tb
    best_v_tb_f = (((vs - floor)**2).sum(1) < 1e-8 )
    best_v_tb_c = (((vs - ceils)**2).sum(1) < 1e-8 )




    cts.append(floor[best_v_tb_f,:])
    cts.append(ceils[best_v_tb_c,:])


    # add all neib cases
    for i in range(2):
        for j in range(2):
            for k in range(2):
                candi = torch.stack([table[i][:,0],table[j][:,1],table[k][:,2]],axis=1)
                #good = torch.sum((candi - vs)**2,axis=1) <= 0.25

                good = torch.sum((candi - center_vs)**2,axis=1) <= torch.sum((vs - center)**2,axis=1)#torch.sum((candi - vs)**2,axis=1) <= 0.75
                #good = torch.sum((candi-vs) * (vs-center_vs) ,axis=1) < 0
                #pdb.set_trace()
                #good =good* ~best_v_tb_f * ~best_v_tb_c
                good = ~best_v_tb_f * ~best_v_tb_c
            
                #pdb.set_trace()
                cts.append(candi) # list of v,3
    ct = torch.cat(cts) #8v,3
    fm_x_ct = v2x(torch.unique(ct,dim=0))
    #* voxel_size + bound_min.unsqueeze(0) # v_unique, 3
    return fm_x_ct

def test_plot(a,b):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()#figsize=(1, 1))
    #ax = fig.axes(projection='3d')
    pdb.set_trace()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(a[:,0],a[:,1],a[:,2], marker='o')
    ax.scatter(b[:,0],b[:,1],b[:,2], marker='^')
    plt.show()


def do_transform_interp(T, fm, fm_obs_count, fm_x, Jm, voxel_size, v2x,x2v, target_pose_T):
    '''
        T: np(4,4) # deltaT between old T and new T
        fm: np(v,c) # v voxels and c dims
        fm_x: np(v,3) # stores the center of that voxel
        Jm: np(v,c,3) # Jacobian on old T
        voxel_size:
    '''
    st = time()

    check_valid = fm_obs_count > 1e-1
    fm = fm[check_valid,:]
    fm_obs_count = fm_obs_count[check_valid]
    fm_x = fm_x[check_valid,:]
    Jm = Jm[check_valid,:,:]




    # 1. transform
    fm_post, fm_x_post = _transform(T, fm, fm_x)
    print(1,time()-st)
    # 2. contour grids
    fm_x_tgt = contour_gvs(fm_x_post, voxel_size, x2v,v2x, target_pose_T[:3,3].unsqueeze(0))
    print(2,time()-st)

    '''
    # 3. interpolate
    fm_itp, fm_v_itp = interpolate(fm_post,fm_v_post,fm_v_tgt,voxel_size)
    '''

    # 3. average hypothesises
    Jm_pointform = Jm.view(Jm.shape[0],-1,3,3) # dim 1,2 is the feature-points Lx3
    Jm_rotted = torch.einsum('ijkn,km->ijmn',Jm_pointform,T[:3,:3].transpose(0,1).to(torch.float))
    Jm_rotted = Jm_rotted.reshape((Jm.shape[0],-1,3))


    print(3,time()-st)

    fm_itp, fm_x_itp, fm_obs_count_itp = average_hypothesis(fm_post, fm_x_post, fm_x_tgt, Jm_rotted, T, voxel_size, fm_obs_count)
    print(4,time()-st)

    #pdb.set_trace()
    #test_plot(fm_x.cpu().detach().numpy(),fm_x_itp.cpu().detach().numpy())

    return fm_itp, fm_x_itp, fm_obs_count_itp




def cal_Jac(pa, grad_f0_pa):
    batch_size = pa.shape[0]
    # 1. get "warp Jacobian", warp => Identity matrix, can be pre-computed
    # grad_f0_pa is B, L, L, 3
    g_ = torch.zeros(batch_size, 6).to(pa)
    warp_jac = utils.compute_warp_jac(g_, pa, num_points=pa.shape[1])   # B x K x 3 x 6
    J = torch.einsum('iajk,ijkm->iam', grad_f0_pa, warp_jac) #B,L,6
    return J


def cal_point_Jac(pa, grad):
    batch_size = pa.shape[0]
    # 1. get "warp Jacobian", warp => Identity matrix, can be pre-computed
    # grad_f0_pa is B, c, N, 3
    g_ = torch.zeros(batch_size, 6).to(pa)
    warp_jac = transform.utils.compute_warp_jac(g_, pa, num_points=pa.shape[1])   # B x N x 3 x 6
    J = torch.einsum('iajk,ijkm->ijam', grad_f0_pa, warp_jac) #B,N,c,6
    return J.squeeze(0) # N,c,6

def point_encoder_w_J_graph(encoder, input, outdim):
    '''
        v4
        encoder_in: 1,((feature-x)+x+cross),3,N,K
        encoder_out: 1xc

        Jacobian is f_size,xi_size
    '''
    # Jacobian of feature to points: c,N,3
    c = outdim
    B,dim_cat, _, N, K = input.shape # 

    #bias = torch.zeros(input.shape).to(input)
    bias = torch.zeros((3,N))
    bias.requires_grad_(True)

    input[0,0,:,:,:] = input[0,0,:,:,:] + bias.unsqueeze(-1)
    f = encoder(input) # N,c # no pooling
    Js = []
    for i in range(c):
        iden = torch.zeros((N,c)).to(f)
        iden[:,i] = 1
        f.backward(iden,retain_graph=True)

        Ji = bias.grad.data # 3,N
        Js.append(Ji)

    J = torch.stack(Js) # c,3,N
    J = J.permute(2,0,1)# N,c,3
    return f, J


def point_encoder_w_J(encoder, input, outdim, delta=1e-2):
    ''' use approx_jacobian
        v4
        encoder_in: Nx3
        encoder_out: 1xc

        Jacobian is f_size,xi_size
    '''
    # Jacobian of feature to points: c,N,3
    c = outdim
    N,d = input.shape # d is 6 here
    #x = input.unsqueeze(0).expand(c, N, d) # c,N,3
    #x.requires_grad_(True)# outdim,outdim

    bias = torch.cat([torch.tensor([[0,0,0]]),torch.eye(3)*delta],axis=0).to(input) # 4,3
    input = input.unsqueeze(1) # N,1,c
    input = input.repeat(1,4,1)
    input[:,:,:3] = input[:,:,:3]+bias.unsqueeze(0) # N, 4, 3


    GPU_small = True
    with torch.no_grad():
        if GPU_small:
            fs = []
            for i in range(4):
                f = encoder(input[:,i,:]).unsqueeze(1)
                fs.append(f)
            f = torch.cat(fs,axis=1)

        else:
            f = encoder(input.view(N*4,6)).view(N,4,c) # N*3,c # no pooling
    
    J = (f[:,1:,:] - f[:,0,:].unsqueeze(1)) / delta # N,3,c

    J = J.transpose(1,2) # N,c,3
    return f[:,0,:], J

def point_encoder_w_J_v3(encoder, input, outdim):
    ''' better but slow
        v3
        encoder_in: Nx3
        encoder_out: 1xc

        Jacobian is f_size,xi_size
    '''
    # Jacobian of feature to points: c,N,3
    c = outdim
    N,d = input.shape # d is 6 here
    #x = input.unsqueeze(0).expand(c, N, d) # c,N,3
    #x.requires_grad_(True)# outdim,outdim

    bias = torch.zeros(N,3).to(input)
    bias.requires_grad_(True)
    f = encoder(input[:,:3]+bias) # N,c # no pooling
    Js = []
    for i in range(c):
        iden = torch.zeros((N,c)).to(f)
        iden[:,i] = 1
        f.backward(iden,retain_graph=True)

        Ji = bias.grad.data # N, 3
        Js.append(Ji)

    J = torch.stack(Js) # c,N,3

    J = J.transpose(0,1) # N,c,3
    return f, J




def point_encoder_w_J_v2(encoder, input, outdim):
    '''
        v2
        encoder_in: Nx3
        encoder_out: 1xc

        Jacobian is f_size,xi_size
    '''
    # Jacobian of feature to points: c,N,3
    c = outdim
    N,d = input.shape # d is 6 here
    #x = input.unsqueeze(0).expand(c, N, d) # c,N,3
    #x.requires_grad_(True)# outdim,outdim
    input.requires_grad_(True)

    f = encoder(input) # N,c # no pooling
    Js = []
    for i in range(c):
        iden = torch.zeros((N,c)).to(f)
        iden[:,i] = 1
        f.backward(iden,retain_graph=True)

        Ji = input.grad.data # N, 3
        Js.append(Ji)

    J = torch.stack(Js) # c,N,c

    # Jacobian
    J = J.unsqueeze(0)
    J = cal_point_Jac(input[:,:3],J)
    return f, J



def point_encoder_w_J_v1(encoder, input, outdim):
    '''
        encoder_in: Nx3
        encoder_out: 1xc

        Jacobian is f_size,xi_size
    '''
    # Jacobian of feature to points: c,N,3
    c = outdim
    N,d = input.shape # d is 6 here
    x = input.unsqueeze(0).expand(c, N, d) # c,N,3
    x.requires_grad_(True)# outdim,outdim
    pdb.set_trace()
    f = encoder(x) # outdim,N,c # no pooling
    f.backward(torch.eye(c).unsqueeze(1))

    J = x.grad.data # c, N, 3

    # Jacobian
    J = J.unsqueeze(0)
    J = cal_point_Jac(input,J)
    return f, J

     

def encoder_w_J(encoder, input, outdim):
    '''
        encoder_in: 1xNx3
        encoder_out: 1xc


        Jacobian is f_size,xi_size
    '''
    # Jacobian of feature to points: outdim, N,3
    B,N,_ = input.shape
    assert(B==1)
    x = input.expand(outdim, N, 3) # outdim,N,3
    x.requires_grad_(True)# outdim,outdim
    pdb.set_trace()

    f = encoder(x)
    f.backward(torch.eye(outdim))

    J = x.grad.data # outdim, N, 3

    # Jacobian
    J = J.unsqueeze(0)
    J = cal_Jac(input,J)
    return f, J

     
    











    

