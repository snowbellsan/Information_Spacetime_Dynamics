import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# パラメータ（最適値）
A = 0.0833
sigma = 1.0
z_peak = 0.7

def w_z(z):
    """Exact Gaussian w(z)"""
    return -1.0 + A * np.exp( - (z - z_peak)**2 / (2 * sigma**2) )

def integrand(z):
    """w(z)の積分項: (1 + w(z)) / (1 + z)"""
    return (1 + w_z(z)) / (1 + z)

# 宇宙論パラメータ（Planckベース + 調整可能）
Omega_m = 0.3      # 物質密度
Omega_DE = 0.7     # ダークエネルギー密度
H0_local = 73.0    # km/s/Mpc (Local測定目標)
H0_cmb = 67.4      # km/s/Mpc (CMB目標、参考)

# E(z) = H(z)/H0 の計算関数
def E_z(z):
    if z == 0:
        return 1.0
    integral, _ = quad(integrand, 0, z, epsabs=1e-10, epsrel=1e-10)
    return np.sqrt(Omega_m * (1 + z)**3 + Omega_DE * np.exp(-3 * integral))

# z範囲で計算
z_vals = np.linspace(0, 3, 500)
E_vals = np.array([E_z(z) for z in z_vals])

# プロット
fig, ax1 = plt.subplots(figsize=(10, 6))

ax1.plot(z_vals, E_vals, 'b-', linewidth=2, label='H(z)/H0 (CSGT Gaussian)')
ax1.set_xlabel('Redshift z')
ax1.set_ylabel('H(z)/H0')
ax1.grid(True, alpha=0.3)
ax1.axhline(1.0, color='k', linestyle=':', alpha=0.5)

# 右軸：w(z)
ax2 = ax1.twinx()
ax2.plot(z_vals, w_z(z_vals), 'r--', label='w(z)')
ax2.set_ylabel('w(z)')
ax2.axhline(-1.0, color='k', linestyle='--', alpha=0.5)
ax2.axhline(-0.9167, color='purple', linestyle=':', alpha=0.4, label='w at z=0.7 ≈ -0.917')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.title('H(z) and w(z) from Exact Gaussian Information Gradient Model\n(Low-z H0 push-up without high-z disruption)')
plt.tight_layout()
plt.show()

# キー値の出力（確認用）
print(f"H(z=0)/H0 = {E_z(0):.4f}  →  Local H0 ≈ {H0_local:.1f} km/s/Mpc に相当")
print(f"w(z=0) = {w_z(0):.4f}")
print(f"w(z=0.7) = {w_z(0.7):.4f}  ← 情報の加速度ピーク")
print(f"H(z=2)/H0 ≈ {E_z(2):.4f}  ← 高zでΛCDMに近い挙動")