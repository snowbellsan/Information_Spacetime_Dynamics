# This is a phenomenological toy model illustrating how informational divergence
# may induce late-time cosmic acceleration and ISW enhancement.

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Constants & Core Parameters ---
T_univ = 13.8  # Current age of the universe (Gyr)
z0 = 0.8       # Peak of structure formation / information transition
z = np.linspace(0, 3, 200)

# Scenarios for the Fate of the Universe (tau_end in Gyr)
# tau_end: Time from Big Bang to the final information saturation (Singularity/Boundary)
scenarios = {
    'Rapid Integration (25 Gyr)': {'tau': 25.0, 'color': '#ff4b2b'}, # Red: Close to Big Rip
    'Standard Evolution (50 Gyr)': {'tau': 50.0, 'color': '#ff00ff'}, # Magenta: Ver 2.0 baseline
    'Slow Maturation (150 Gyr)': {'tau': 150.0, 'color': '#00ffff'}  # Cyan: Near-LCDM / Heat Death
}

def get_dynamics(tau_end, z_range):
    # k: Learning rate (Transition sharpness) linked to future boundary
    k = 4.0 / (tau_end / T_univ)
    # D(z): Informational Divergence (Logistic Model)
    D_z = 1 / (1 + np.exp(-k * (z_range - z0)))
    # Lambda_eff: Metabolic response (gradient of D)
    L_eff = np.gradient(D_z, -z_range)
    L_eff = L_eff / np.max(L_eff)
    # w(z): Effective Equation of State
    # Models the "Phantom Dive" (w < -1) as a response to information backpropagation
    w_z = -1.0 - (0.15 * (k/1.1)) * np.exp(-(z_range - 0.7)**2 / (2 * 0.2**2))
    return D_z, L_eff, w_z

# --- 2. Visualization Setup ---
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for label, params in scenarios.items():
    D, L, W = get_dynamics(params['tau'], z)
    
    # Plot 1: Informational Divergence (The "Learning" History)
    axes[0].plot(z, D, color=params['color'], lw=2, label=label)
    axes[0].set_title('Informational Divergence D(z)')
    axes[0].set_xlabel('Redshift z')
    axes[0].invert_xaxis()
    axes[0].legend(fontsize='small')

    # Plot 2: Dynamic Lambda_eff (The Metabolic Rate)
    axes[1].plot(z, L, color=params['color'], lw=2)
    axes[1].set_title('Metabolic Rate of Spacetime (Λ_eff)')
    axes[1].set_xlabel('Redshift z')
    axes[1].invert_xaxis()

    # Plot 3: Equation of State w(z) (The Destiny Signal)
    # w < -1 represents the "Phantom" region where information is back-propagating.
    axes[2].plot(z, W, color=params['color'], lw=2)
    axes[2].axhline(-1, color='white', linestyle='--', alpha=0.5)
    axes[2].set_title('Effective Equation of State w(z)')
    axes[2].set_xlabel('Redshift z')
    axes[2].set_ylabel('w(z)')
    axes[2].set_ylim(-1.3, -0.8)
    axes[2].invert_xaxis()

plt.tight_layout()
plt.savefig('v2_1_destiny_engine.png')
