import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import differential_evolution
from scipy.interpolate import interp1d

# =========================
# データ読み込み（Pantheon+）
# =========================
pantheon_file = r'Pantheon+SH0ES.dat'
try:
    sn_df = pd.read_csv(pantheon_file, sep=r'\s+', comment='#', header=None,
                        usecols=[2, 10, 11], names=['z', 'mu_obs', 'sigma_mu'], engine='python')
    sn_df = sn_df.apply(pd.to_numeric, errors='coerce').dropna().sort_values('z').reset_index(drop=True)
    print(f"Pantheon+ loaded: {len(sn_df)} points")
except:
    print("データ読み込み失敗。ダミー生成")
    sn_df = pd.DataFrame({'z': np.linspace(0.01, 2.2, 100), 'mu_obs': 30 + 10*np.linspace(0.01, 2.2, 100), 'sigma_mu': 0.1})

# =========================
# 理論関数（w_offsetを追加）
# =========================
def w_integrand_ext(zp_val, A, sigma, zp_peak, w_off):
    # 背景の w_off と ガウス型の山 A の統合
    # 1+w = (1+w_off) + A*exp(...)
    return ((1.0 + w_off) + A * np.exp(-(zp_val - zp_peak)**2 / (2 * sigma**2))) / (1.0 + zp_val)

def get_mu_theory_extended(z_array, A, sigma, zp_peak, M, H0, Om, w_off):
    c = 299792.458
    Ode = 1.0 - Om
    z_grid = np.linspace(0, 2.5, 120) # 精度をさらに向上
    
    comoving_grid = []
    curr_chi = 0
    for i in range(len(z_grid)):
        z_curr = z_grid[i]
        if z_curr == 0:
            comoving_grid.append(0)
            continue
        
        # 3 * ∫ (1+w)/(1+z) dz の計算
        w_int, _ = quad(w_integrand_ext, 0, z_curr, args=(A, sigma, zp_peak, w_off), epsabs=1e-8)
        E_z = np.sqrt(Om * (1 + z_curr)**3 + Ode * np.exp(3.0 * w_int))
        
        if i > 0:
            dz = z_grid[i] - z_grid[i-1]
            curr_chi += dz / E_z
        comoving_grid.append(curr_chi)
        
    dist_interp = interp1d(z_grid, comoving_grid, kind='cubic')
    chi = dist_interp(z_array)
    dL = (1 + z_array) * chi * (c / H0)
    return 5.0 * np.log10(dL * 1e6 / 10.0) + M

def chi2_final_extended(params, is_csgt):
    if is_csgt:
        A, sigma, zp_peak, M, H0, Om, w_off = params
    else:
        M, H0, Om = params
        A, sigma, zp_peak, w_off = 0.0, 1.0, 0.7, -1.0 # LCDMはw=-1固定
    
    try:
        mu_th = get_mu_theory_extended(sn_df['z'].values, A, sigma, zp_peak, M, H0, Om, w_off)
        return np.sum(((sn_df['mu_obs'].values - mu_th)**2 / sn_df['sigma_mu'].values**2))
    except:
        return 1e18

# =========================
# 最適化実行
# =========================
# [A, sigma, zp_peak, M, H0, Om, w_off]
bounds_csgt = [
    (0.01, 0.5),      # A
    (0.1, 1.5),       # sigma
    (0.4, 1.2),       # zp_peak
    (-19.38, -19.32), # M
    (72.5, 73.5),     # H0 (SH0ESの中心値を狙う)
    (0.25, 0.35),     # Om
    (-1.10, -0.90)    # w_off (ここが自由の翼よ)
]

bounds_lcdm = [
    (-19.38, -19.32), # M
    (72.5, 73.5),     # H0
    (0.25, 0.35)      # Om
]

print("Launching Final Evolution...")
res_csgt = differential_evolution(chi2_final_extended, bounds_csgt, args=(True,), workers=1, tol=0.001)
res_lcdm = differential_evolution(chi2_final_extended, bounds_lcdm, args=(False,), workers=1, tol=0.001)

delta_chi2 = res_lcdm.fun - res_csgt.fun

print(f"\n===== ULTIMATE RESULT =====")
print(f"Delta chi2 = {delta_chi2:.4f}")

if delta_chi2 > 0:
    print(f"Victory! CSGT has surpassed LCDM.")
    print(f"Optimal w_off: {res_csgt.x[6]:.4f}, Peak A: {res_csgt.x[0]:.4f}, z_p: {res_csgt.x[2]:.4f}")
else:
    print(f"Still close... Difference: {delta_chi2:.4f}")