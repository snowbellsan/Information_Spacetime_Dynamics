import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d

# =========================
# 前回の最適化結果（勝利パラメータ）
# =========================
res_csgt_x = [0.010, 0.100, 1.200, -19.34, 72.99, 0.35, -1.10] # A, sigma, zp, M, H0, Om, w_off
res_lcdm_x = [-19.34, 72.99, 0.35] # M, H0, Om

# =========================
# グラフ用計算関数
# =========================
def get_plot_data(params, is_csgt=True):
    if is_csgt:
        A, sigma, zp, M, H0, Om, w_off = params
    else:
        M, H0, Om = params
        A, sigma, zp, w_off = 0.0, 1.0, 0.7, -1.0
        
    Ode = 1.0 - Om
    z_range = np.linspace(0.001, 2.3, 200)
    
    # w(z) の計算
    w_z = w_off + A * np.exp(-(z_range - zp)**2 / (2 * sigma**2))
    
    # H(z) の計算
    H_z = []
    for z in z_range:
        w_int, _ = quad(lambda zp_val: ((1.0 + w_off) + A * np.exp(-(zp_val - zp)**2 / (2 * sigma**2))) / (1.0 + zp_val), 0, z)
        Ez = np.sqrt(Om * (1 + z)**3 + Ode * np.exp(3.0 * w_int))
        H_z.append(H0 * Ez)
        
    return z_range, w_z, np.array(H_z)

# データの取得
z_axis, w_csgt, H_csgt = get_plot_data(res_csgt_x, True)
_, w_lcdm, H_lcdm = get_plot_data(res_lcdm_x, False)

# =========================
# プロット作成
# =========================
fig, ax = plt.subplots(1, 2, figsize=(15, 6))

# 左図：状態方程式 w(z)
ax[0].plot(z_axis, w_csgt, 'r-', lw=2, label='CSGT (A=0.01, z_p=1.2)')
ax[0].plot(z_axis, w_lcdm, 'b--', lw=2, label='ΛCDM (w=-1)')
ax[0].axhline(-1, color='gray', linestyle=':', alpha=0.5)
ax[0].set_xlabel('Redshift z', fontsize=12)
ax[0].set_ylabel('Equation of State w(z)', fontsize=12)
ax[0].set_title('Dark Energy Evolution', fontsize=14)
ax[0].legend()
ax[0].grid(alpha=0.3)

# 右図：膨張率 H(z) の比較（偏差）
# 標準モデルからのズレをパーセントで表示すると分かりやすいわ
H_diff = (H_csgt - H_lcdm) / H_lcdm * 100
ax[1].plot(z_axis, H_diff, 'g-', lw=2, label='(H_CSGT - H_LCDM) / H_LCDM [%]')
ax[1].axhline(0, color='blue', linestyle='--', alpha=0.5)
ax[1].set_xlabel('Redshift z', fontsize=12)
ax[1].set_ylabel('Difference in H(z) (%)', fontsize=12)
ax[1].set_title('Expansion Rate Deviation from ΛCDM', fontsize=14)
ax[1].legend()
ax[1].grid(alpha=0.3)

plt.tight_layout()
plt.show()

print("Graphs generated successfully.")