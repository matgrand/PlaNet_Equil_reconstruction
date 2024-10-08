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

#calculate the Grad-Shafranov operator pytorch
import torch
import torch.nn.functional as F
def Ϛ(x, ker): 
    assert x.ndim == 4, f"x.ndim = {x.ndim}"
    assert x.shape[1] == 1, f"x.shape = {x.shape}"
    assert ker.ndim == 4, f"ker.ndim = {ker.ndim}"
    assert ker.shape[1] == 1, f"ker.shape = {ker.shape}"
    b = x.shape[0] # batch size
    x = F.conv2d(x.view(1, b, x.shape[2], x.shape[3]), ker, padding=0, groups=b)
    x = F.pad(x, (1,1,1,1), mode='replicate')
    return x.view(b, 1, x.shape[2], x.shape[3])

def laplace_ker(Δr, Δz, α, dev=torch.device("cpu")): # [[0, Δr**2/α, 0], [Δz**2/α, 1, Δz**2/α], [0, Δr**2/α, 0]]
    kr, kz = Δr**2/α, Δz**2/α
    ker = torch.zeros(len(Δr),1, 3, 3, dtype=torch.float32, device=dev)
    ker[:,0,0,1], ker[:,0,1,0], ker[:,0,1,2], ker[:,0,2,1], ker[:,0,1,1] = kr, kz, kz, kr, 1
    return ker
   
def dr_ker(Δr, Δz, α, dev=torch.device("cpu")): # [[0,0,0],[-1,0,+1],[0,0,0]] * (Δr**2 * Δz**2) / (2*Δr*α)
    ker = torch.zeros(len(Δr),1, 3, 3, dtype=torch.float32, device=dev)
    k = (Δr**2 * Δz**2) / (2*Δr*α)
    ker[:,0,1,0], ker[:,0,1,2] = -k, k
    return ker

def calc_gso(ψ, rr, zz):
    assert ψ.shape == rr.shape == zz.shape == (64,64), f"ψ.shape = {ψ.shape}, rr.shape = {rr.shape}, zz.shape = {zz.shape}"
    Ψ, rr, zz = torch.tensor(ψ).view(1,1,64,64), torch.tensor(rr).view(1,1,64,64), torch.tensor(zz).view(1,1,64,64)
    return calc_gso_batch(Ψ, rr, zz).numpy()[0,0]

def calc_gso_batch(Ψ, rr, zz, dev=torch.device('cpu')):
    assert Ψ[0].shape == rr[0].shape == zz[0].shape == (1,64,64), f"Ψ.shape = {Ψ.shape}, rr.shape = {rr.shape}, zz.shape = {zz.shape}"
    Δr, Δz = rr[:,0,1,2]-rr[:,0,1,1], zz[:,0,2,1]-zz[:,0,1,1] 
    α = (-2*(Δr**2 + Δz**2))
    β = ((Δr**2 * Δz**2) / α)
    ΔΨ = (1/β.view(-1,1,1,1)) * (Ϛ(Ψ, laplace_ker(Δr, Δz, α, dev)) - Ϛ(Ψ, dr_ker(Δr, Δz, α, dev))/rr) # grad-shafranov operator
    return ΔΨ