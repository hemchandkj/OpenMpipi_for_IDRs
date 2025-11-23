import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ---------- User-editable paths / settings ----------
csv_path = '/scratch/hc675/Analysis_dir/ddx4/TcTemps/Tctemps300ddx4wt.csv'
set_lim = 280        # K, temperature threshold used in Tc fit
z_dim =            # nm, slab interface side,  (used to normalize densities)
point_size = 40
fit_linesize = 2.3
color = 'tab:green'
# ---------------------------------------------------

# Load data
df = pd.read_csv(csv_path)

# compute diff used in fits (only for temperatures >= set_lim)
mask = df['temperature'] >= set_lim
df.loc[:, 'diff'] = ((df.loc[mask, 'avg_rho_l'] - df.loc[mask, 'avg_rho_v']) / (z_dim**2))
# ensure arrays for fitting/plotting
T_all = df['temperature'].values
T_fit_mask = df.loc[mask, 'temperature'].values
rhoV_data = np.array(df['avg_rho_v']) / (z_dim**2)
rhoL_data = np.array(df['avg_rho_l']) / (z_dim**2)

# ---------- Fit Tc from (diff**3.06) vs T using model d*(1-T/Tc) ----------
def density_diff_model(T, Tc, d):
    return d * (1.0 - T / Tc)

y_Tc = np.array((df.loc[mask, 'diff'] ** 3.06).dropna())
T_for_Tc = np.array(df.loc[mask, 'temperature'].dropna())

popt_Tc, pcov_Tc = curve_fit(density_diff_model, T_for_Tc, y_Tc, maxfev=20000)
Tc = float(popt_Tc[0])
d_opt = float(popt_Tc[1]) if popt_Tc.size > 1 else None

# ---------- Fit critical density rho_c from diff vs T using linear form ----------
def crit_density_model(T, rho_c, A):
    return 2.0 * rho_c + 2.0 * A * (T - Tc)

y_rho = np.array(df.loc[mask, 'diff'].dropna())
T_for_rho = np.array(df.loc[mask, 'temperature'].dropna())

popt_rho, pcov_rho = curve_fit(crit_density_model, T_for_rho, y_rho, maxfev=20000)
rho_c = float(popt_rho[0])
A_opt = float(popt_rho[1]) if popt_rho.size > 1 else None

# ---------- Binodal model and parameter fit ----------
def binodal(T, a, s4, cc):
    # returns rhoV, rhoL arrays for input T (vector)
    R = (-T + Tc) / a
    S = (-T + Tc) / s4 + 2.0 * cc
    rhoL = (R) ** (1.0 / 3.06) / 2.0 + S / 2.0
    rhoV = S - rhoL
    return rhoV, rhoL

def binodal_flat(T, a, s4, cc):
    rhoV, rhoL = binodal(T, a, s4, cc)
    return np.concatenate((rhoV, rhoL))

# Prepare data for binodal fit: order must match binodal_flat (rhoV for all T, then rhoL for all T)
T_data_for_fit = T_all.astype(float)
y_binodal = np.concatenate((rhoV_data, rhoL_data))

initial_guess = [0.1, 30.0, 1.0]
popt_bin, pcov_bin = curve_fit(binodal_flat, T_data_for_fit, y_binodal, p0=initial_guess, maxfev=20000)
a_fit, s4_fit, cc_fit = popt_bin

# ---------- Prepare smooth fit curve up to Tc ----------
T_curve = np.linspace(np.min(T_all), Tc, 500)
rhoV_curve, rhoL_curve = binodal(T_curve, a_fit, s4_fit, cc_fit)

# ---------- Plot everything in one cell ----------
fig, ax = plt.subplots(figsize=(8, 4.5))

# simulated points (both phases, same color)
ax.scatter(rhoV_data, T_all, color=color, s=point_size, alpha=0.85, label='Simulated', zorder=3)
ax.scatter(rhoL_data, T_all, color=color, s=point_size, alpha=0.85, zorder=3)

# fitted binodal curves (same color)
ax.plot(rhoV_curve, T_curve, color=color, linewidth=fit_linesize, label='Fit', zorder=2)
ax.plot(rhoL_curve, T_curve, color=color, linewidth=fit_linesize, zorder=2)

# To ensure the hollow critical marker does not show the line through it:
# draw a small filled circle with the axes facecolor to mask the line, then draw the hollow edge on top.
ax_face = ax.get_facecolor()
mask_s = 130   # mask marker size (slightly larger than edge)
edge_s = 80    # visible hollow edge size

# mask (background-colored) plotted above lines but below the hollow edge
ax.scatter(rho_c, Tc, s=mask_s, color=ax_face, edgecolors=ax_face, zorder=6)

# hollow critical point edge (same color as points/fit)
ax.scatter(
    rho_c,
    Tc,
    facecolors='none',
    edgecolors=color,
    marker='o',
    s=edge_s,
    linewidths=2.3,
    label=rf'Critical Point ($\rho_c$={rho_c:.3f}, $T_c$={Tc:.2f}K)',
    zorder=7
)

ax.set_xlabel('Density (gm/cm$^3$)', fontsize=12)
ax.set_ylabel('Temperature (K)', fontsize=12)
ax.set_title('DDX4 WT Tc at 300mM NaCl', fontsize=14)

# compact legend: avoid duplicate 'Simulated' label
handles, labels = ax.get_legend_handles_labels()
# keep first occurrence of each label
seen = {}
new_handles = []
new_labels = []
for h, l in zip(handles, labels):
    if l not in seen:
        seen[l] = True
        new_handles.append(h)
        new_labels.append(l)
ax.legend(new_handles, new_labels, frameon=True, loc='upper right', fontsize=9)

ax.grid(False)

# print fitted values concisely
print(f"Tc = {Tc:.4f} K, rho_c = {rho_c:.6f}")
print(f"Binodal fit params: a={a_fit:.4g}, s4={s4_fit:.4g}, cc={cc_fit:.4g}")

plt.tight_layout()
plt.show()
