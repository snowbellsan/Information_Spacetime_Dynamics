import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# === パラメータ ===
H0 = 67.4               # km/s/Mpc
Omega_m = 0.3
Omega_r = 0.0            # 無視可能
Omega_DE = 0.7
A = 0.0833
sigma = 1.0
z_peak = 0.7

# === w(z) exact Gaussian ===
def w(z):
    return -1.0 + A * np.exp(- (z - z_peak)**2 / (2*sigma**2))

# === DE密度の積分項 ===
def integrand(z_prime):
    return (1 + w(z_prime)) / (1 + z_prime)

# === H(z) 計算 ===
def H(z):
    integral, _ = quad(integrand, 0, z)
    rho_DE_factor = np.exp(3 * integral)
    Hz = H0 * np.sqrt(Omega_m * (1+z)**3 + Omega_r * (1+z)**4 + Omega_DE * rho_DE_factor)
    return Hz

# === 情報圧力ピーク: p_info ~ w(z) + 1 ===
def info_pressure(z):
    return w(z) + 1  # ガウスのピーク = 追加圧力

# === z 配列 ===
z_vals = np.linspace(0, 3, 300)
H_vals = np.array([H(z) for z in z_vals])
p_info_vals = info_pressure(z_vals)
p_info_accel = np.gradient(p_info_vals, z_vals)  # dp/dz = 情報の「加速度」

# === プロット ===
plt.figure(figsize=(10,6))

# H(z)
plt.plot(z_vals, H_vals, label='H(z) with exact Gaussian w(z)', color='blue', linewidth=2)

# 低z/high-z の H0 を目安線
plt.axhline(73, color='red', linestyle='--', label='Low-z H0 ~ 73 km/s/Mpc')
plt.axhline(67.4, color='green', linestyle='--', label='High-z H0 ~ 67.4 km/s/Mpc')

# 情報圧力ピークを二次軸で表示
plt.twinx()
plt.plot(z_vals, p_info_accel, color='purple', linestyle='-.', label='Information Acceleration dp/dz')
plt.ylabel('Information Acceleration (dp/dz)')

plt.xlabel('Redshift z')
plt.title('H(z) and Information Acceleration with Exact Gaussian w(z)')
plt.grid(True)
plt.legend(loc='upper left')

plt.show()
