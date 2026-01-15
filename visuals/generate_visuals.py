# This is a phenomenological toy model illustrating how informational divergence
# may induce late-time cosmic acceleration and ISW enhancement.

import numpy as np
import matplotlib.pyplot as plt

# --- 1. 未来境界条件のパラメータ化 ---
T_univ = 13.8  # 宇宙の現在の年齢 (Gyr)
tau_end = 50.0  # 未来の終焉までの時間 (Gyr) - 50Gyrは比較的"急な"終末
k = 4.0 / (tau_end / T_univ)  # 未来が近いほど k（学習速度）が大きくなる
z0 = 0.8  # 構造形成のピーク

z = np.linspace(0, 3, 200)

# --- 2. D(z) の計算 (Ver. 2.0: Logistic Model) ---
# 宇宙が未来の境界状態に向けて情報を統合していく過程
D_z = 1 / (1 + np.exp(-k * (z - z0)))
D_band_upper = D_z * 1.1
D_band_lower = D_z * 0.9

# --- 3. 動的宇宙定数 Lambda_eff(z) ---
# 情報勾配の時間微分。z=0.7付近でピークを持たせる
Lambda_eff = np.gradient(D_z, -z)
Lambda_eff = Lambda_eff / np.max(Lambda_eff)

# --- 4. ISW Enhancement Ratio (25-35% Prediction) ---
ell = np.logspace(0.5, 2.5, 100)
isw_ratio = 1 + 0.3 * np.exp(-(ell/30)**2)
isw_band = 0.05

# --- 5. プロット ---
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: D(z)
axes[0].plot(z, D_z, color='cyan', lw=2, label='D(ρ||σ)')
axes[0].fill_between(z, D_band_lower, D_band_upper, color='cyan', alpha=0.2)
axes[0].set_title('Informational Divergence (The Learning Curve)')
axes[0].set_xlabel('Redshift z')
axes[0].invert_xaxis()

# Plot 2: Lambda_eff
axes[1].plot(z, Lambda_eff, color='magenta', lw=2, label='Λ_eff (z)')
axes[1].set_title('Metabolic Rate of Spacetime (Dynamic Λ)')
axes[1].set_xlabel('Redshift z')
axes[1].invert_xaxis()

# Plot 3: ISW Ratio
axes[2].semilogx(ell, isw_ratio, color='lime', lw=2)
axes[2].fill_between(ell, isw_ratio-isw_band, isw_ratio+isw_band, color='lime', alpha=0.2)
axes[2].axhline(1, color='white', linestyle='--')
axes[2].set_title('ISW Enhancement Prediction (Testable Signal)')
axes[2].set_xlabel('Multipole ℓ')

plt.tight_layout()
plt.savefig('visuals/v2_prediction_summary.png')
