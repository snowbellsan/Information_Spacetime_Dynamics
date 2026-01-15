import numpy as np
import matplotlib.pyplot as plt

# --- Cosmic Constants ---
T_UNIV = 13.8  # Gyr
Z0 = 0.8       # Information transition peak
Z_RANGE = np.linspace(3, 0, 300)  # From Past(z=3) to Present(z=0)

def simulate_dissipative_dynamics(tau_end, beta=0.15, gamma=1.2):
    """
    beta: Dissipation rate (Selective Forgetting)
    gamma: High-z suppression scale (Inflationary legacy)
    """
    k = 4.0 / (tau_end / T_UNIV)
    D = np.zeros_like(Z_RANGE)
    D[0] = 0.001  # Initial tiny seed of information
    
    # Numerical Integration of the Dissipative Logistic Equation
    # dD/dt = k*D*(1-D) - beta*exp(-gamma*z)*D
    for i in range(1, len(Z_RANGE)):
        dz = Z_RANGE[i-1] - Z_RANGE[i]
        dt = dz  # Simplified time-step mapping
        
        # Logistic Growth Term
        growth = k * D[i-1] * (1 - D[i-1])
        # Dissipation Term (The "Forgetting" effect)
        dissipation = beta * np.exp(-gamma * Z_RANGE[i-1]) * D[i-1]
        
        D[i] = D[i-1] + (growth - dissipation) * dt

    # Metabolic Rate (L_eff) and Equation of State (w_z)
    # Using dD/dt to derive the dynamic pressure of information
    L_eff = np.gradient(D, -Z_RANGE)
    L_norm = L_eff / np.max(L_eff)
    
    # Theoretical w(z) from Information Gradient
    # Standard: -1.0, Shifted by metabolic response
    w_z = -1.0 - 0.2 * (k / 1.1) * (L_norm - 0.2 * beta * np.exp(-gamma * Z_RANGE))
    
    return D, L_norm, w_z

# --- Visualization ---
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Case A: Pure Storage (beta=0.0) vs Case B: Dissipative Memory (beta=0.15)
D_pure, _, w_pure = simulate_dissipative_dynamics(50.0, beta=0.0)
D_diss, _, w_diss = simulate_dissipative_dynamics(50.0, beta=0.15)

ax1.plot(Z_RANGE, D_pure, 'w--', alpha=0.5, label='Pure Storage (Ver 2.1)')
ax1.plot(Z_RANGE, D_diss, color='#ff00ff', lw=3, label='Dissipative Memory (Ver 2.3)')
ax1.set_title("Information Accumulation D(z)")
ax1.invert_xaxis()
ax1.legend()



ax2.plot(Z_RANGE, w_pure, 'w--', alpha=0.5)
ax2.plot(Z_RANGE, w_diss, color='#00ffff', lw=3, label='Phantom-to-Quintessence Cross')
ax2.axhline(-1, color='white', linestyle=':', alpha=0.5)
ax2.set_title("Equation of State w(z)")
ax2.set_ylabel("w(z)")
ax2.set_ylim(-1.3, -0.9)
ax2.invert_xaxis()
ax2.legend()

plt.tight_layout()
plt.savefig('visuals/v2_3_dissipative_memory.png')
plt.show()