""" part of source code from PointNetLK (https://github.com/hmgoforth/PointNetLK), 
Deep Closest Point (https://github.com/WangYueFt/dcp), modified. """

import numpy as np
import torch
import matplotlib.pyplot as plt


# functions for invmat
def batch_inverse(x):
    """ M(n) -> M(n); x -> x^-1 """
    batch_size, h, w = x.size()
    assert h == w
    y = torch.zeros_like(x)
    for i in range(batch_size):
        y[i, :, :] = x[i, :, :].inverse()
    return y


def batch_inverse_dx(y):
    """ backward """
    batch_size, h, w = y.size()
    assert h == w
    yl = y.repeat(1, 1, h).view(batch_size*h*h, h, 1)
    yr = y.transpose(1, 2).repeat(1, h, 1).view(batch_size*h*h, 1, h)
    dy = - yl.bmm(yr).view(batch_size, h, h, h, h)

    return dy


class InvMatrix(torch.autograd.Function):
    """ M(n) -> M(n); x -> x^-1.
    """
    @staticmethod
    def forward(ctx, x):
        y = batch_inverse(x)
        ctx.save_for_backward(y)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        y, = ctx.saved_tensors
        batch_size, h, w = y.size()
        assert h == w

        dy = batch_inverse_dx(y)   # dy(j,k,m,n) = dy(j,k)/dx(m,n)
        go = grad_output.contiguous().view(batch_size, 1, h*h)   # [1, (j*k)]
        ym = dy.view(batch_size, h*h, h*h)   # [(j*k), (m*n)]
        r = go.bmm(ym)  # [1, (m*n)]
        grad_input = r.view(batch_size, h, h)   # [m, n]

        return grad_input


# function for se3/so3 operations
def transform(g, a):
    # g : SE(3),  B x 4 x 4
    # a : R^3,    B x N x 3
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3].contiguous().view(*(g.size()[0:-2]), 3, 3)
    p = g_[:, 0:3, 3].contiguous().view(*(g.size()[0:-2]), 3)
    if len(g.size()) == len(a.size()):
        a = a.transpose(1,2)
        b = R.matmul(a) + p.unsqueeze(-1)
    else:
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p
    return b


# so3
def mat_so3(x):
    # x: [*, 3]
    # X: [*, 3, 3]
    x_ = x.view(-1, 3)
    x1, x2, x3 = x_[:, 0], x_[:, 1], x_[:, 2]
    O = torch.zeros_like(x1)

    X = torch.stack((
        torch.stack((O, -x3, x2), dim=1),
        torch.stack((x3, O, -x1), dim=1),
        torch.stack((-x2, x1, O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 3, 3)


def btrace(X):
    # batch-trace: [B, N, N] -> [B]
    n = X.size(-1)
    X_ = X.view(-1, n, n)
    tr = torch.zeros(X_.size(0)).to(X)
    for i in range(tr.size(0)):
        m = X_[i, :, :]
        tr[i] = torch.trace(m)
    return tr.view(*(X.size()[0:-2]))


def vec(X):
    X_ = X.view(-1, 3, 3)
    x1, x2, x3 = X_[:, 2, 1], X_[:, 0, 2], X_[:, 1, 0]
    x = torch.stack((x1, x2, x3), dim=1)
    return x.view(*X.size()[0:-2], 3)


def log_so3(g):
    eps = 1.0e-6
    R = g.view(-1, 3, 3)
    tr = btrace(R)
    c = (tr - 1) / 2
    t = torch.acos(c)
    sc = sinc1(t)
    idx0 = (torch.abs(sc) <= eps)
    idx1 = (torch.abs(sc) > eps)
    sc = sc.view(-1, 1, 1)

    X = torch.zeros_like(R)
    if idx1.any():
        X[idx1] = (R[idx1] - R[idx1].transpose(1, 2)) / (2*sc[idx1])

    if idx0.any():
        t2 = t[idx0] ** 2
        A = (R[idx0] + torch.eye(3).type_as(R).unsqueeze(0)) * t2.view(-1, 1, 1) / 2
        aw1 = torch.sqrt(A[:, 0, 0])
        aw2 = torch.sqrt(A[:, 1, 1])
        aw3 = torch.sqrt(A[:, 2, 2])
        sgn_3 = torch.sign(A[:, 0, 2])
        sgn_3[sgn_3 == 0] = 1
        sgn_23 = torch.sign(A[:, 1, 2])
        sgn_23[sgn_23 == 0] = 1
        sgn_2 = sgn_23 * sgn_3
        w1 = aw1
        w2 = aw2 * sgn_2
        w3 = aw3 * sgn_3
        w = torch.stack((w1, w2, w3), dim=-1)
        W = mat_so3(w)
        X[idx0] = W

    x = vec(X.view_as(g))
    return x


def inv_vecs_Xg_ig(x):
    """ H = inv(vecs_Xg_ig(x)) """
    t = x.view(-1, 3).norm(p=2, dim=1).view(-1, 1, 1)
    X = mat_so3(x)
    S = X.bmm(X)
    I = torch.eye(3).to(x)

    e = 0.01
    eta = torch.zeros_like(t)
    s = (t < e)
    c = (s == 0)
    t2 = t[s] ** 2
    eta[s] = ((t2/40 + 1)*t2/42 + 1)*t2/720 + 1/12   # O(t**8)
    eta[c] = (1 - (t[c]/2) / torch.tan(t[c]/2)) / (t[c]**2)

    H = I - 1/2*X + eta*S
    return H.view(*(x.size()[0:-1]), 3, 3)


def log(g):
    g_ = g.view(-1, 4, 4)
    R = g_[:, 0:3, 0:3]
    p = g_[:, 0:3, 3]

    w = log_so3(R)
    H = inv_vecs_Xg_ig(w)
    v = H.bmm(p.contiguous().view(-1, 3, 1)).view(-1, 3)

    x = torch.cat((w, v), dim=1)
    return x.view(*(g.size()[0:-2]), 6)


# se3
def mat_se3(x):
    # size: [*, 6] -> [*, 4, 4]
    x_ = x.view(-1, 6)
    w1, w2, w3 = x_[:, 0], x_[:, 1], x_[:, 2]
    v1, v2, v3 = x_[:, 3], x_[:, 4], x_[:, 5]
    O = torch.zeros_like(w1)

    X = torch.stack((
        torch.stack((  O, -w3,  w2, v1), dim=1),
        torch.stack(( w3,   O, -w1, v2), dim=1),
        torch.stack((-w2,  w1,   O, v3), dim=1),
        torch.stack((  O,   O,   O,  O), dim=1)), dim=1)
    return X.view(*(x.size()[0:-1]), 4, 4)


def sinc1(t):
    """ sinc1: t -> sin(t)/t """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = 1 - t2/6*(1 - t2/20*(1 - t2/42))   # Taylor series O(t^8)
    r[c] = torch.sin(t[c]) / t[c]

    return r


def sinc2(t):
    """ sinc2: t -> (1 - cos(t)) / (t**2) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t ** 2
    r[s] = 1/2*(1-t2[s]/12*(1-t2[s]/30*(1-t2[s]/56)))   # Taylor series O(t^8)
    r[c] = (1-torch.cos(t[c]))/t2[c]

    return r


def sinc3(t):
    """ sinc3: t -> (t - sin(t)) / (t**3) """
    e = 0.01
    r = torch.zeros_like(t)
    a = torch.abs(t)

    s = a < e
    c = (s == 0)
    t2 = t[s] ** 2
    r[s] = 1/6*(1-t2/20*(1-t2/42*(1-t2/72)))   # Taylor series O(t^8)
    r[c] = (t[c]-torch.sin(t[c]))/(t[c]**3)

    return r


# functions for exp map
def exp(x):
    x_ = x.view(-1, 6)
    w, v = x_[:, 0:3], x_[:, 3:6]
    t = w.norm(p=2, dim=1).view(-1, 1, 1)   # norm of rotation
    W = mat_so3(w)
    S = W.bmm(W)
    I = torch.eye(3).to(w)

    # Rodrigues' rotation formula.
    R = I + sinc1(t)*W + sinc2(t)*S
    V = I + sinc2(t)*W + sinc3(t)*S

    p = V.bmm(v.contiguous().view(-1, 3, 1))

    z = torch.Tensor([0, 0, 0, 1]).view(1, 1, 4).repeat(x_.size(0), 1, 1).to(x)
    Rp = torch.cat((R, p), dim=2)
    g = torch.cat((Rp, z), dim=1)

    return g.view(*(x.size()[0:-1]), 4, 4)


class ExpMap(torch.autograd.Function):
    """ Exp: se(3) -> SE(3)
    """
    @staticmethod
    def forward(ctx, x):
        """ Exp: R^6 -> M(4),
            size: [B, 6] -> [B, 4, 4],
              or  [B, 1, 6] -> [B, 1, 4, 4]
        """
        ctx.save_for_backward(x)
        g = exp(x)
        return g

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        g = exp(x)
        gen_k = mat_se3(torch.eye(6)).to(x)

        dg = gen_k.matmul(g.view(-1, 1, 4, 4))
        # (k, i, j)
        dg = dg.to(grad_output)

        go = grad_output.contiguous().view(-1, 1, 4, 4)
        dd = go * dg
        grad_input = dd.sum(-1).sum(-1)

        return grad_input


# explicitly compute the analytical feature jacobian
def feature_jac(M, A, Ax, BN, device):
    # M, A, Ax, BN: list
    A1, A2, A3 = A
    M1, M2, M3 = M
    Ax1, Ax2, Ax3 = Ax
    BN1, BN2, BN3 = BN

    # 1 x c_in x c_out x 1
    A1 = (A1.T).detach().unsqueeze(-1)
    A2 = (A2.T).detach().unsqueeze(-1)
    A3 = (A3.T).detach().unsqueeze(-1)

    # calculate gradient for batch normalization using autograd, 
    # since the dimension is small, and the actual computation is complex.
    # B x 1 x c_out x N
    dBN1 = torch.autograd.grad(outputs=BN1, inputs=Ax1, grad_outputs=torch.ones(BN1.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN2 = torch.autograd.grad(outputs=BN2, inputs=Ax2, grad_outputs=torch.ones(BN2.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()
    dBN3 = torch.autograd.grad(outputs=BN3, inputs=Ax3, grad_outputs=torch.ones(BN3.size()).to(device), retain_graph=True)[0].unsqueeze(1).detach()

    # B x 1 x c_out x N
    M1 = M1.detach().unsqueeze(1)
    M2 = M2.detach().unsqueeze(1)
    M3 = M3.detach().unsqueeze(1)

    # 1. using *, naturally broadcast --> B x c_in x c_out x N
    A1BN1M1 = A1 * dBN1 * M1
    A2BN2M2 = A2 * dBN2 * M2
    A3BN3M3 =  M3 * dBN3 * A3

    # using torch.einsum()
    A1BN1M1_A2BN2M2 = torch.einsum('ijkl,ikml->ijml', A1BN1M1, A2BN2M2)   # B x 3 x 64 x N
    A2BN2M2_A3BN3M3 = torch.einsum('ijkl,ikml->ijml', A1BN1M1_A2BN2M2, A3BN3M3)   # B x 3 x K x N
    
    feat_jac = A2BN2M2_A3BN3M3

    return feat_jac   # B x 3 x K x N


# explicitly compute the analytical warp Jacobian
def compute_warp_jac(t, xx, num_points):
    b = xx.shape[0]
    
    warp_jac = torch.zeros(b, num_points, 3, 6).to(xx)
    T = exp(t)
    rotm = T[:, :3, :3]   # Bx3x3
    warp_jac[..., 3:] = -rotm.transpose(1,2).unsqueeze(1).repeat(1, num_points, 1, 1)   # BxNx3x6
    
    x = xx[..., 0]
    y = xx[..., 1]
    z = xx[..., 2]
    d03 = T[:, 1, 0].unsqueeze(1) * z - T[:, 2, 0].unsqueeze(1) * y   # BxN
    d04 = -T[:, 0, 0].unsqueeze(1) * z + T[:, 2, 0].unsqueeze(1) * x
    d05 = T[:, 0, 0].unsqueeze(1) * y - T[:, 1, 0].unsqueeze(1) * x
    d13 = T[:, 1, 1].unsqueeze(1) * z - T[:, 2, 1].unsqueeze(1) * y
    d14 = -T[:, 0, 1].unsqueeze(1) * z + T[:, 2, 1].unsqueeze(1) * x
    d15 = T[:, 0, 1].unsqueeze(1) * y - T[:, 1, 1].unsqueeze(1) * x
    d23 = T[:, 1, 2].unsqueeze(1) * z - T[:, 2, 2].unsqueeze(1) * y
    d24 = -T[:, 0, 2].unsqueeze(1) * z + T[:, 2, 2].unsqueeze(1) * x
    d25 = T[:, 0, 2].unsqueeze(1) * y - T[:, 1, 2].unsqueeze(1) * x
    
    d0 = torch.cat([d03.unsqueeze(-1), d04.unsqueeze(-1), d05.unsqueeze(-1)], -1)   # BxNx3
    d1 = torch.cat([d13.unsqueeze(-1), d14.unsqueeze(-1), d15.unsqueeze(-1)], -1)
    d2 = torch.cat([d23.unsqueeze(-1), d24.unsqueeze(-1), d25.unsqueeze(-1)], -1)
    warp_jac[..., :3] = torch.cat([d0.unsqueeze(-2), d1.unsqueeze(-2), d2.unsqueeze(-2)], -2)

    return warp_jac


# explicitly compute the conditional warp Jacobian
def cal_conditioned_warp_jacobian(voxel_coords):
    # conditioned warp: see supplementary for detailed math.
    #               --                                        --  ^-1
    #               |   1  ,   0  ,   0  ,   0  ,   0  ,   0   |
    #               |   0  ,   1  ,   0  ,   0  ,   0  ,   0   |
    # xi_v / xi_g = |   0  ,   0  ,   1  ,   0  ,   0  ,   0   |
    #               |   0  , -xi_6,  xi_5,   1  ,   0  ,   0   |
    #               |  xi_6,   0  , -xi_4,   0  ,   1  ,   0   |
    #               | -xi_5,  xi_4,   0  ,   0  ,   0  ,   1   |
    #               --                                        --
    
    V = voxel_coords.shape[0]
    conditioned_jac = torch.eye(6).repeat(V, 1, 1)   # V x 6 x 6
    trans_twist_mat_00 = torch.zeros(V, 1).to(voxel_coords)
    trans_twist_mat_11 = torch.zeros(V, 1).to(voxel_coords)
    trans_twist_mat_22 = torch.zeros(V, 1).to(voxel_coords)
    trans_twist_mat_01 = -voxel_coords[:, 2].unsqueeze(1)
    trans_twist_mat_02 = voxel_coords[:, 1].unsqueeze(1)
    trans_twist_mat_10 = voxel_coords[:, 2].unsqueeze(1)
    trans_twist_mat_12 = -voxel_coords[:, 0].unsqueeze(1)
    trans_twist_mat_20 = -voxel_coords[:, 1].unsqueeze(1)
    trans_twist_mat_21 = voxel_coords[:, 0].unsqueeze(1)
    
    trans_twist_mat_0 = torch.cat([trans_twist_mat_00, trans_twist_mat_01, trans_twist_mat_02], 1).reshape(-1, 3)
    trans_twist_mat_1 = torch.cat([trans_twist_mat_10, trans_twist_mat_11, trans_twist_mat_12], 1).reshape(-1, 3)
    trans_twist_mat_2 = torch.cat([trans_twist_mat_20, trans_twist_mat_21, trans_twist_mat_22], 1).reshape(-1, 3)
    trans_twist_mat = torch.cat([trans_twist_mat_0, trans_twist_mat_1, trans_twist_mat_2], 1).reshape(-1, 3, 3)
    conditioned_jac[:, 3:, :3] = trans_twist_mat   # V x 6 x 6
    
    conditioned_jac = torch.inverse(conditioned_jac)
    
    return conditioned_jac
    

# functions for testing metrics
def test_metrics(rotations_gt, translation_gt, rotations_ab, translation_ab, filename):
    rotations_gt = np.concatenate(rotations_gt, axis=0).reshape(-1, 3)
    translation_gt = np.concatenate(translation_gt, axis=0).reshape(-1, 3)
    rotations_ab = np.concatenate(rotations_ab, axis=0).reshape(-1, 3)
    translation_ab = np.concatenate(translation_ab, axis=0).reshape(-1,3)

    # root square error
    rot_err = np.sqrt(np.mean((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2, axis=1))
    trans_err = np.sqrt(np.mean((translation_ab - translation_gt) ** 2, axis=1))

    suc_tab = np.zeros(11)
    
    # set the criteria
    rot_err_tab = np.arange(11) * 0.5
    trans_err_tab = np.arange(11) * 0.05
    
    err_count_tab = np.triu(np.ones((11, 11)))
    
    for i in range(rot_err.shape[0]):
        if rot_err[i] <= rot_err_tab[0] and trans_err[i] <= trans_err_tab[0]:
            suc_tab = suc_tab + err_count_tab[0]
        elif rot_err[i] <= rot_err_tab[1] and trans_err[i] <= trans_err_tab[1]:
            suc_tab = suc_tab + err_count_tab[1]
        elif rot_err[i] <= rot_err_tab[2] and trans_err[i] <= trans_err_tab[2]:
            suc_tab = suc_tab + err_count_tab[2]
        elif rot_err[i] <= rot_err_tab[3] and trans_err[i] <= trans_err_tab[3]:
            suc_tab = suc_tab + err_count_tab[3]
        elif rot_err[i] <= rot_err_tab[4] and trans_err[i] <= trans_err_tab[4]:
            suc_tab = suc_tab + err_count_tab[4]
        elif rot_err[i] <= rot_err_tab[5] and trans_err[i] <= trans_err_tab[5]:
            suc_tab = suc_tab + err_count_tab[5]
        elif rot_err[i] <= rot_err_tab[6] and trans_err[i] <= trans_err_tab[6]:
            suc_tab = suc_tab + err_count_tab[6]
        elif rot_err[i] <= rot_err_tab[7] and trans_err[i] <= trans_err_tab[7]:
            suc_tab = suc_tab + err_count_tab[7]
        elif rot_err[i] <= rot_err_tab[8] and trans_err[i] <= trans_err_tab[8]:
            suc_tab = suc_tab + err_count_tab[8]
        elif rot_err[i] <= rot_err_tab[9] and trans_err[i] <= trans_err_tab[9]:
            suc_tab = suc_tab + err_count_tab[9]
        elif rot_err[i] <= rot_err_tab[10] and trans_err[i] <= trans_err_tab[10]:
            suc_tab = suc_tab + err_count_tab[10]

    print('success cases are {}'.format(suc_tab))

    # 1. use mean error
    rot_mse_ab = np.mean((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2)
    rot_rmse_ab = np.sqrt(rot_mse_ab)
    rot_mae_ab = np.mean(np.abs(np.degrees(rotations_ab) - np.degrees(rotations_gt)))

    trans_mse_ab = np.mean((translation_ab - translation_gt) ** 2)
    trans_rmse_ab = np.sqrt(trans_mse_ab)
    trans_mae_ab = np.mean(np.abs(translation_ab - translation_gt))
    
    # 2. use median error
    rot_mse_ab_02 = np.median((np.degrees(rotations_ab) - np.degrees(rotations_gt)) ** 2)
    rot_rmse_ab_02 = np.sqrt(rot_mse_ab_02)
    rot_mae_ab_02 = np.median(np.abs(np.degrees(rotations_ab) - np.degrees(rotations_gt)))
    
    trans_mse_ab_02 = np.median((translation_ab - translation_gt) ** 2)
    trans_rmse_ab_02 = np.sqrt(trans_mse_ab_02)
    trans_mae_ab_02 = np.median(np.abs(translation_ab - translation_gt))

    print('Source to Template:')
    print(filename)
    print('********************mean********************')
    print('rot_MSE: {}, rot_RMSE: {}, rot_MAE: {}, trans_MSE: {}, trans_RMSE: {}, trans_MAE: {}'.format(rot_mse_ab, 
            rot_rmse_ab, rot_mae_ab, trans_mse_ab, trans_rmse_ab, trans_mae_ab))
    print('********************median********************')
    print('rot_MSE: {}, rot_RMSE: {}, rot_MAE: {}, trans_MSE: {}, trans_RMSE: {}, trans_MAE: {}'.format(rot_mse_ab_02, 
            rot_rmse_ab_02, rot_mae_ab_02, trans_mse_ab_02, trans_rmse_ab_02, trans_mae_ab_02))

    return

