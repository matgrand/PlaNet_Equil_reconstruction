# utils functions
import numpy as np
from scipy.interpolate import RegularGridInterpolator
INTERP_METHOD = 'linear' # fast, but less accurate
# INTERP_METHOD = 'quintic' # fast, but less accurate

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

def interp_fun(f, rrG, zzG, rrg, zzg, method=INTERP_METHOD):
  interp_func = RegularGridInterpolator((rrG[0,:], zzG[:,0]), f.T, method=method)
  pts = np.column_stack((rrg.flatten(), zzg.flatten()))
  f_int = interp_func(pts).reshape(rrg.shape)
  return f_int

def resample_on_new_subgrid(fs:list, rrG, zzG, nr=64, nz=64):
  rrg, zzg = sample_random_subgrid(rrG, zzG, nr, nz)
  fs_int = [interp_fun(f, rrG, zzG, rrg, zzg) for f in fs]
  return fs_int, rrg, zzg


# kernels
def calc_laplace_df_dr_ker(hr, hz):
    α = -2*(hr**2 + hz**2)
    laplace_ker = np.array(([0, hr**2/α, 0], [hz**2/α, 1, hz**2/α], [0, hr**2/α, 0]))
    df_dr_ker = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr*α)*(hr**2*hz**2)
    return laplace_ker, df_dr_ker