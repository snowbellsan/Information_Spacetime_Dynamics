import numpy as np
import matplotlib.pyplot as plt

# 宇宙のパラメータ設定
Z_START = 2.0        # 過去 (赤方偏移)
Z_END = -1.0         # 未来 (28.7 Gyr 先を想定)
STEPS = 500
TARGET_Z = 0.7       # 星乃と議論した Phantom Crossing の特異点

def simulate_csgt():
    z_axis = np.linspace(Z_START, Z_END, STEPS)
    
    # 1. 情報ダイバージェンス D(z) の定義
    # 未来の完成図 (P_future) と現在の分布 (P_present) の距離
    # z=2 (過去) では大きく、z=-1 (未来) で 0 に収束する
    d_z = np.exp(z_axis) / np.exp(Z_START) 
    
    # 2. 情報勾配による「復元力」 (Self-Correcting Potential)
    # 未来からのバックプロパゲーション。z=0.7付近で影響が最大化するよう設計
    restoring_force = -np.gradient(d_z) * np.exp(-(z_axis - TARGET_Z)**2 / 0.1)
    
    # 3. 暗エネルギーの状態方程式 w(z) の算出
    # 標準的な -1 に対し、情報勾配の復元力が加わることで w < -1 (Phantom) が出現
    w_z = -1.0 + restoring_force * 5.0  # 係数は可視化用の調整
    
    return z_axis, d_z, w_z

z, d, w = simulate_csgt()

# 可視化
fig, ax1 = plt.subplots(figsize=(10, 6))

color = 'tab:blue'
ax1.set_xlabel('Redshift (z) [Right is Future]')
ax1.set_ylabel('Information Divergence D(z)', color=color)
ax1.plot(z, d, color=color, label='Information Divergence (KLD)')
ax1.tick_params(axis='y', labelcolor=color)
ax1.invert_xaxis() # 過去から未来へ

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Equation of State w(z)', color=color)
ax2.plot(z, w, color=color, linewidth=2, label='Dark Energy w(z)')
ax2.axhline(-1, color='gray', linestyle='--', label='LCDM Limit (w=-1)')
ax2.fill_between(z, w, -1, where=(w < -1), color='red', alpha=0.3, label='Phantom Crossing')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('CSGT Simulation: Information Gradient as a Restoring Force')
fig.tight_layout()
plt.legend(loc='upper left')
plt.show()