import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

# --- Configurations ---
T_UNIV = 13.8
Z0 = 0.8

def hubble_model(tau_end, z):
    k = 4.0 / (tau_end / T_UNIV)
    D_z = 1 / (1 + np.exp(-k * (z - Z0)))
    L_eff = np.gradient(D_z, -z) if isinstance(z, np.ndarray) else 0.5
    L_norm = L_eff / np.max(L_eff) if np.max(L_eff) > 0 else 1
    return 67.4 * np.sqrt(0.315 * (1+z)**3 + 0.685 * L_norm)

def run_analysis():
    print("--- Information-Geometric Destiny Engine Ver 2.2 ---")
    # Mock data reflecting DESI DR2 trends
    z_obs = np.array([0.15, 0.38, 0.51, 0.70, 0.85, 1.48])
    h_obs = np.array([68.0, 83.0, 90.0, 105.0, 115.0, 150.0])
    err = h_obs * 0.03

    res = minimize(lambda t: np.sum(((h_obs - hubble_model(t, z_obs))/err)**2), 
                   x0=50.0, bounds=[(15.0, 500.0)])
    
    best_tau = res.x[0]
    print(f"\nResult: Estimated Universe Lifespan (tau_end) = {best_tau:.2f} Gyr")
    
    if best_tau < 30:
        print("Status: Rapid Integration. The cosmic horizon is approaching saturation.")
    else:
        print("Status: Stable Evolution. The informational metabolism is balanced.")

    # Show Plot
    plt.style.use('dark_background')
    z_range = np.linspace(0, 2, 100)
    plt.plot(z_range, hubble_model(best_tau, z_range), color='magenta', lw=2)
    plt.errorbar(z_obs, h_obs, yerr=err, fmt='o', color='cyan', alpha=0.8)
    plt.title(f"Universal Destiny: tau_end = {best_tau:.1f} Gyr")
    plt.show()

    input("\nPress Enter to close the engine...")

if __name__ == "__main__":
    run_analysis()