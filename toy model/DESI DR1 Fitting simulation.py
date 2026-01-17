import numpy as np
import matplotlib.pyplot as plt

# 1. 宇宙論的背景の設定
z = np.linspace(0.0, 1.5, 100)

# 2. CSGTモデルによる w(z) の予測
# 未来（z=-1）からの情報勾配が z=0.7 付近で最大化すると仮定
def csgt_w_model(z_val):
    # ベースは ΛCDM (-1)
    base_w = -1.0
    # 未来からのバックプロパゲーションによる Phantom Crossing 項
    # z=0.7付近で w を押し下げる
    bridge_effect = -0.15 * np.exp(-(z_val - 0.7)**2 / 0.08)
    return base_w + bridge_effect

w_csgt = csgt_w_model(z)

# 3. 実際の観測データ (DESI DR1 を模した代表的なデータ点)
# redshift, w_value, error_bar
obs_z = np.array([0.15, 0.51, 0.71, 0.93, 1.15])
obs_w = np.array([-0.98, -1.05, -1.12, -1.02, -0.99]) # Phantom Crossing の傾向を反映
obs_err = np.array([0.05, 0.07, 0.08, 0.06, 0.05])

# 可視化
plt.figure(figsize=(10, 7))

# CSGTモデル曲線
plt.plot(z, w_csgt, color='crimson', label='CSGT Model (Information Gradient)', linewidth=2.5)

# 観測データ点
plt.errorbar(obs_z, obs_w, yerr=obs_err, fmt='o', color='black', 
             capsize=5, label='Mock DESI DR1 Data (Reflected Trends)')

# 境界線
plt.axhline(-1, color='gray', linestyle='--', alpha=0.6, label='$\Lambda$CDM Limit ($w=-1$)')
plt.fill_between(z, -1.5, -1, color='blue', alpha=0.05, label='Phantom Domain ($w < -1$)')

plt.title('Fitting CSGT to DESI DR1: The Phantom Crossing Hypothesis', fontsize=14)
plt.xlabel('Redshift $z$', fontsize=12)
plt.ylabel('Dark Energy Equation of State $w(z)$', fontsize=12)
plt.gca().invert_xaxis() # 左に向かって過去、右に向かって未来
plt.ylim(-1.3, -0.8)
plt.legend(loc='lower left')
plt.grid(True, which='both', linestyle=':', alpha=0.5)

plt.show()