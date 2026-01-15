# This is a phenomenological toy model illustrating how informational divergence
# may induce late-time cosmic acceleration and ISW enhancement.

import numpy as np
import matplotlib.pyplot as plt

# --- 1. Future Boundary Condition & Parameters ---
# T_univ: Current age of the universe (Gyr)
T_univ = 13.8  
# tau_end: Time until the cosmic horizon saturation/end (Gyr)
# A value of 50.0 Gyr implies a relatively "sharp" approach to the final state.
tau_end = 50.0  
# k: Learning rate (Transition sharpness)
# The closer the end (tau_end), the higher the rate of information processing (k).
k = 4.0 / (tau_end / T_univ)  
# z0: Center of the informational transition, aligned with structure formation peak.
z0 = 0.8  

z = np.linspace(0, 3, 200)

# --- 2. Informational Divergence D(z) (Logistic Growth Model) ---
# Modeling the process of information integration towards the future boundary.
D_z = 1 / (1 + np.exp(-k * (z - z0)))
D_band_upper = D_z * 1.1
D_band_lower = D_z * 0.9

# --- 3. Dynamic Cosmological Constant: Lambda_eff(z) ---
# Derived from the temporal gradient of information divergence.
# This represents the metabolic response of spacetime to structure growth.
Lambda_eff = np.gradient(D_z, -z)
Lambda_eff = Lambda_eff / np.max(Lambda_eff)

# --- 4. ISW Enhancement Ratio (25-35% Falsifiable Prediction) ---
# Predicting a significant boost in the Integrated Sachs-Wolfe effect at low multipoles.
ell = np.logspace(0.5, 2.5, 100)
isw_ratio = 1 + 0.3 * np.exp(-(ell/30)**2)
isw_band = 0.05

# --- 5. Visualization ---
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Informational Divergence
axes[0].plot(z, D_z, color='cyan', lw=2, label='D(ρ||σ)')
axes[0].fill_between(z, D_band_lower, D_band_upper, color='cyan', alpha=0.2)
axes[0].set_title('Informational Divergence (The Learning Curve)')
axes[0].set_xlabel('Redshift z')
axes[0].set_ylabel('Normalized D')
axes[0].invert_xaxis()
axes[0].grid(alpha=0.2)

# Plot 2: Dynamic Lambda_eff
axes[1].plot(z, Lambda_eff, color='magenta', lw=2, label='Λ_eff(z)')
axes[1].set_title('Metabolic Rate of Spacetime (Dynamic Λ)')
axes[1].set_xlabel('Redshift z')
axes[1].set_ylabel('Relative Intensity')
axes[1].invert_xaxis()
axes[1].grid(alpha=0.2)

# Plot 3: ISW Enhancement
axes[2].semilogx(ell, isw_ratio, color='lime', lw=2, label='Theory/ΛCDM')
axes[2].fill_between(ell, isw_ratio-isw_band, isw_ratio+isw_band, color='lime', alpha=0.2)
axes[2].axhline(1, color='white', linestyle='--', label='ΛCDM Baseline')
axes[2].set_title('ISW Enhancement Prediction (Testable Signal)')
axes[2].set_xlabel('Multipole ℓ')
axes[2].set_ylabel('Enhancement Ratio')
axes[2].legend()
axes[2].grid(alpha=0.2)

plt.tight_layout()
# Output for GitHub /visuals directory
plt.savefig('visuals/v2_metabolic_engine_summary.png')
