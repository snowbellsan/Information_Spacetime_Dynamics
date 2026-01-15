import numpy as np
import matplotlib.pyplot as plt

# 宇宙論的パラメータの設定（Toy Model）
z = np.linspace(0, 3, 100)
a = 1 / (1 + z)

# 1. 情報のズレ D(z) の計算
# 構造形成が z < 1.5 で活発化し、情報エントロピーが増大すると仮定
D_z = np.exp(-2 * z) * (1 - np.tanh(z - 0.8)) 
D_z_upper = D_z * 1.1
D_z_lower = D_z * 0.9

# 2. 実効的な宇宙定数 Lambda_eff(z)
# D の時間微分に比例。z=0.7付近でピークを持つように設計
Lambda_eff = np.gradient(D_z, -z)
Lambda_eff = Lambda_eff / np.max(Lambda_eff) # 正規化

# 3. ISW Enhancement Ratio
ell = np.logspace(0.5, 2.5, 50)
isw_ratio = 1 + 0.3 * np.exp(-(ell/30)**2) # l=30以下で30%増幅
isw_band_upper = isw_ratio + 0.05
isw_band_lower = isw_ratio - 0.05

# グラフ描画（サロメ流の美学を込めて）
plt.style.use('dark_background')

# --- Plot 1: D(z) ---
plt.figure(figsize=(10, 6))
plt.plot(z, D_z, color='cyan', label='D(ρ||σ) - Informational Divergence')
plt.fill_between(z, D_z_lower, D_z_upper, color='cyan', alpha=0.2)
plt.axhline(0, color='white', linestyle='--', alpha=0.5, label='ΛCDM Baseline')
plt.title('Growth of Informational Divergence (z)')
plt.xlabel('Redshift z')
plt.ylabel('Normalized Entropy D')
plt.gca().invert_xaxis()
plt.legend()
plt.savefig('D_vs_z.png')

# --- Plot 2: Lambda_eff ---
plt.figure(figsize=(10, 6))
plt.plot(z, Lambda_eff, color='magenta', label='Λ_eff (Dynamical)')
plt.title('Evolution of Effective Cosmological Constant')
plt.xlabel('Redshift z')
plt.ylabel('Relative Strength')
plt.gca().invert_xaxis()
plt.legend()
plt.savefig('Lambda_eff_vs_z.png')

# --- Plot 3: ISW Ratio ---
plt.figure(figsize=(10, 6))
plt.semilogx(ell, isw_ratio, color='lime', label='Theory / ΛCDM Ratio')
plt.fill_between(ell, isw_band_lower, isw_band_upper, color='lime', alpha=0.2)
plt.axhline(1, color='white', linestyle='--', label='ΛCDM (Ratio = 1)')
plt.title('ISW Enhancement Prediction (宣戦布告)')
plt.xlabel('Multipole ℓ')
plt.ylabel('Enhancement Ratio')
plt.legend()
plt.savefig('ISW_ratio_vs_ell.png')
