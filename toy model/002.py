from classy import Class
import numpy as np
import matplotlib.pyplot as plt

# === Exact Gaussian w(z) ===
A = 0.0833
sigma = 1.0
z_peak = 0.7

def w_exact_gaussian(z):
    """Exact Gaussian w(z) for CSGT model"""
    return -1.0 + A * np.exp(- (z - z_peak)**2 / (2*sigma**2))

# === CLASS では tabulated w(a) を渡す ===
# a = 1/(1+z)
z_vals = np.linspace(0, 3, 300)
a_vals = 1.0 / (1.0 + z_vals)
w_vals = w_exact_gaussian(z_vals)

# CLASS expects a tabulated array: [[a0, w0], [a1, w1], ...]
w_tab = np.column_stack((a_vals, w_vals))

# === パラメータ設定 ===
params = {
    'output': 'tCl,pCl,lCl',
    'l_max_scalars': 2500,
    'P_k_max_h/Mpc': 10.,
    'z_max_pk': 3.,
    'omega_b': 0.0224,
    'omega_cdm': 0.12,
    'H0': 67.4,
    'tau_reio': 0.054,
    'has_fld': 'yes',
    'Omega_fld': 0.7,
    'w_fld': 'external',
    'w_fld_tab': w_tab,  # Exact Gaussian tabulated
}

# === CLASS 実行 ===
cosmo = Class()
cosmo.set(params)
cosmo.compute()

# === CMB power spectra 取得 ===
cls = cosmo.lensed_cl(2500)  # l_max
ells = cls['ell']
Cl_TT_LCDM = cls['tt']       # TT spectrum

# === LCDM 比較用（w=-1） ===
params_lcdm = params.copy()
params_lcdm['w_fld'] = -1.0
params_lcdm.pop('w_fld_tab')  # Tabulated削除
cosmo_lcdm = Class()
cosmo_lcdm.set(params_lcdm)
cosmo_lcdm.compute()
cls_lcdm = cosmo_lcdm.lensed_cl(2500)
Cl_TT_LCDM_ref = cls_lcdm['tt']

# === 可視化 ===
plt.figure(figsize=(10,6))
plt.plot(ells, Cl_TT_LCDM_ref, color='gray', label='LCDM w=-1')
plt.plot(ells, Cl_TT_LCDM, color='crimson', linestyle='--', label='CSGT Exact Gaussian')
plt.xlabel(r'Multipole $\ell$')
plt.ylabel(r'$C_\ell^{TT}$')
plt.title('CMB Power Spectrum: LCDM vs CSGT')
plt.legend()
plt.yscale('log')
plt.grid(True)
plt.show()

# === 終了処理 ===
cosmo.struct_cleanup()
cosmo.empty()
cosmo_lcdm.struct_cleanup()
cosmo_lcdm.empty()
