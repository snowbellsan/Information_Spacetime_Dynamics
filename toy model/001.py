import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === 基本パラメータ ===
H0 = 67.4               # km/s/Mpc
Omega_m = 0.3
Omega_r = 0.0            # 無視可能
Omega_DE = 0.7
A = 0.0833
sigma = 1.0
z_peak = 0.7

# === Exact Gaussian w(z) ===
def w(z):
    """
    Dark Energy equation of state with Exact Gaussian peak at z_peak
    """
    return -1.0 + A * np.exp(- (z - z_peak)**2 / (2*sigma**2))

# === Friedmann積分項 ===
def integrand(z_prime):
    return (1 + w(z_prime)) / (1 + z_prime)

# === H(z)計算 ===
def H(z):
    integral, _ = quad(integrand, 0, z)
    rho_DE_factor = np.exp(3 * integral)
    return H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_r * (1+z)**4 + Omega_DE * rho_DE_factor)

# === Information Acceleration dp/dz ===
def info_acceleration(z_vals):
    """
    dp/dz = d(w(z)+1)/dz
    情報の加速度として解釈
    """
    p_info = w(z_vals) + 1      # p_info = w(z)+1
    dpdz = np.gradient(p_info, z_vals)  # 数値微分で加速度
    return dpdz

# === z配列 ===
z_vals = np.linspace(0, 3, 300)
H_vals = np.array([H(z) for z in z_vals])
dpdz_vals = info_acceleration(z_vals)

# === プロット ===
fig, ax1 = plt.subplots(figsize=(10,6))

# H(z)
ax1.plot(z_vals, H_vals, color='blue', linewidth=2, label='H(z) with Exact Gaussian w(z)')
ax1.axhline(73, color='red', linestyle='--', label='Low-z H0 ~ 73')
ax1.axhline(67.4, color='green', linestyle='--', label='High-z H0 ~ 67.4')
ax1.set_xlabel('Redshift z')
ax1.set_ylabel('H(z) [km/s/Mpc]')
ax1.grid(True)

# dp/dz 二次軸
ax2 = ax1.twinx()
ax2.plot(z_vals, dpdz_vals, color='purple', linestyle='-.', label='Information Acceleration dp/dz')
ax2.set_ylabel('Information Acceleration (dp/dz)')

# 凡例まとめ
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

plt.title('H(z) and Information Acceleration dp/dz with Exact Gaussian w(z)')
plt.show()
