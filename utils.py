# utils functions
import numpy as np
from numpy import ndarray as arr
from scipy.interpolate import RegularGridInterpolator
# INTERP_METHOD = 'linear' # fast, but less accurate
INTERP_METHOD = 'quintic' # fast, but less accurate

def sample_random_subgrid(rrG, zzG, nr=64, nz=64):
    rm, rM, zm, zM = rrG.min(), rrG.max(), zzG.min(), zzG.max()
    delta_r_min = .33*(rM-rm)
    delta_r_max = .75*(rM-rm)
    delta_z_min = .2*(zM-zm)
    delta_z_max = .75*(zM-zm)
    delta_r = np.random.uniform(delta_r_min, delta_r_max, 1)
    r0 = np.random.uniform(rm, rm+delta_r_max-delta_r, 1)
    delta_z = np.random.uniform(delta_z_min, delta_z_max, 1)
    z0 = np.random.uniform(zm,zm+delta_z_max-delta_z, 1)
    rr = np.linspace(r0, r0+delta_r, nr)
    zz = np.linspace(z0, z0+delta_z, nz)
    rrg, zzg = np.meshgrid(rr, zz, indexing='xy')
    return rrg, zzg

def get_box_from_grid(rrg, zzg):
    rm, rM, zm, zM = rrg.min(), rrg.max(), zzg.min(), zzg.max()
    return np.array([[rm,zm],[rM,zm],[rM,zM],[rm,zM],[rm,zm]])

def interp_fun(ψ, rrG, zzG, rrg, zzg, method=INTERP_METHOD):
    interp_func = RegularGridInterpolator((rrG[0,:], zzG[:,0]), ψ.T, method=method)
    pts = np.column_stack((rrg.flatten(), zzg.flatten()))
    f_int = interp_func(pts).reshape(rrg.shape)
    return f_int

def resample_on_new_subgrid(fs:list, rrG, zzG, nr=64, nz=64):
    rrg, zzg = sample_random_subgrid(rrG, zzG, nr, nz)
    fs_int = [interp_fun(ψ, rrG, zzG, rrg, zzg) for ψ in fs]
    return fs_int, rrg, zzg

# kernels
def calc_laplace_df_dr_ker(hr, hz):
    α = -2*(hr**2 + hz**2)
    laplace_ker = np.array(([0, hr**2/α, 0], [hz**2/α, 1, hz**2/α], [0, hr**2/α, 0]))
    dr_ker = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr*α)*(hr**2*hz**2)
    return laplace_ker, dr_ker

def gauss_kernel(size=3):
    if size == 3: return np.array(([1,2,1], [2,4,2], [1,2,1]), dtype='float32')/16
    elif size == 5: return np.array(([1,4,7,4,1],[4,16,26,16,4],[7,26,41,26,7],[4,16,26,16,4],[1,4,7,4,1]), dtype='float32')/273
    else: raise NotImplementedError 

#calculate the Grad-Shafranov operator
# pytorch version
import torch
import torch.nn.functional as F
def calc_gso(ψ:arr, rr:arr, zz:arr):
    assert ψ.shape == (64,64), f"ψ.shape = {ψ.shape}"
    # assert lap_ker.shape == (3,3), f"lap_ker.shape = {lap_ker.shape}"
    # assert dr_ker.shape == (3,3), f"dr_ker.shape = {dr_ker.shape}"
    # assert gauss_ker.shape == (3,3), f"gauss_ker.shape = {gauss_ker.shape}"
    assert rr.shape == (64,64), f"rr.shape = {rr.shape}"
    assert zz.shape == (64,64), f"zz.shape = {zz.shape}"
    # ψ, lap_ker, dr_ker, gauss_ker, rr, zz = map(torch.tensor, (ψ, lap_ker, dr_ker, gauss_ker, rr, zz))
    ψ, rr, zz = map(torch.tensor, (ψ, rr, zz))
    Ψ = Ψ
    Δr, Δz = rr[1,2]-rr[1,1], zz[2,1]-zz[1,1] 
    α = -2*(Δr**2 + Δz**2)  # constant
    β = (Δr**2 * Δz**2) / α # constant
    lap_ker = torch.tensor([[0, Δr**2/α, 0], [Δz**2/α, 1, Δz**2/α], [0, Δr**2/α, 0]])
    dr_ker = torch.tensor([[0,0,0],[+1,0,-1],[0,0,0]]) #* (Δr**2 * Δz**2) / (2*Δr*α)
    conv_lap = F.conv2d(ψ.view(1,1,64,64), lap_ker.view(1,1,3,3), padding=1)
    conv_dr = F.conv2d(ψ.view(1,1,64,64), dr_ker.view(1,1,3,3), padding=1)
    ΔΨ = (conv_lap - conv_dr/rr.view(1,1,64,64)) / β # grad-shafranov operator
    return (ΔΨ.view(64,64)).numpy()

# tensorflow version
import tensorflow as tf
def fun_GSoperator_NN_conv_smooth_batch_adaptive(ψ, lap_ker, dr_ker, gauss_ker, rr, zz):
    ψ = tf.transpose(ψ,[3,1,2,0])
    lap_psi = tf.nn.depthwise_conv2d(ψ,tf.transpose(tf.expand_dims(lap_ker, axis=-1),[1,2,0,3]),strides=[1, 1, 1, 1],padding='VALID')
    lap_psi = tf.transpose(lap_psi,[3,1,2,0]) # no need to be transposed becaused Laplacian filter is left/rigth symmetric
    dpsi_dr = tf.nn.depthwise_conv2d(ψ,tf.transpose(tf.expand_dims(dr_ker, axis=-1), [1,2,0,3]),strides=[1, 1, 1, 1],padding='VALID')
    dpsi_dr = - dpsi_dr # necessary because nn.depthwise_conv2d filters has to be transposed to perform real convolution (here [+h 0 -h] -> [-h 0 +h])
    dpsi_dr = tf.transpose(dpsi_dr,[3,1,2,0])
    RR_in = tf.expand_dims(rr[:,1:-1,1:-1],axis=-1)
    dpsi_dr = tf.math.divide(dpsi_dr,RR_in)
    GS_ope = lap_psi - dpsi_dr
    hr = rr[:,1,2] - rr[:,1,1]
    hz = zz[:,2,1] - zz[:,1,1]
    α = -2*(hr**2 + hz**2)
    hr = tf.expand_dims(tf.expand_dims(tf.expand_dims(hr,axis=-1),axis=-1),axis=-1)
    hz = tf.expand_dims(tf.expand_dims(tf.expand_dims(hz,axis=-1),axis=-1),axis=-1)
    α = tf.expand_dims(tf.expand_dims(tf.expand_dims(α,axis=-1),axis=-1),axis=-1)
    GS_ope = GS_ope*α/(hr**2*hz**2)
    GS_ope = tf.nn.conv2d(GS_ope,gauss_ker,strides=[1, 1, 1, 1],padding='SAME')
    GS_ope = tf.squeeze(GS_ope,axis = -1)
    return GS_ope