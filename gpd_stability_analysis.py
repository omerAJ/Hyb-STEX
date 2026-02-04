import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt
import os

# Paths
train_path = "/home/maincoder/Documents/Hyb-STEX/code/Hyb-STEX/data/ST-SSL_Dataset/NYCTaxi/train.npz"
output_dir = "/home/maincoder/Documents/Hyb-STEX/code/Hyb-STEX/data/ST-SSL_Dataset/NYCTaxi/evtdiag_quick"
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading train data...")
data = np.load(train_path)
y = data['y']
y_train = y[:, 0, :, :]  # Shape: (S, N, F)
S, N, F = y_train.shape
print(f"y_train shape: {y_train.shape}")

# Thresholds to test
quantiles = [0.90, 0.95, 0.97, 0.99]

# Store stability results
stability_results = []

print("\n" + "="*70)
print("STABILITY TABLE (Pooled GPD at different thresholds)")
print("="*70)

for f in range(min(2, F)):
    print(f"\n--- Feature {f} ---")

    # Aggregate all nodes
    ts_all = y_train[:, :, f].flatten()
    ts_all = ts_all[ts_all >= 5]  # Filter noise
    n_total = len(ts_all)
    print(f"Total samples after filtering: {n_total}")

    for q in quantiles:
        u = np.percentile(ts_all, q * 100)
        z = ts_all[ts_all > u] - u
        n_exc = len(z)

        # Fit GPD
        params = genpareto.fit(z, floc=0)
        xi, loc, sigma = params

        stability_results.append({
            'feature': f,
            'quantile': q,
            'threshold': u,
            'n_exceedances': n_exc,
            'xi': xi,
            'sigma': sigma
        })

        print(f"  q={q:.2f}: u={u:>8.2f}, n_exc={n_exc:>6}, xi={xi:>8.4f}, sigma={sigma:>8.2f}")

# Write stability table
results_path = os.path.join(output_dir, 'gpd_stability_table.txt')
with open(results_path, 'w') as f:
    f.write("GPD Stability Analysis - NYC Taxi Data (Pooled per Feature)\n")
    f.write("="*80 + "\n\n")

    f.write("STABILITY TABLE\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Feat':<6} {'Quantile':<10} {'Threshold':<12} {'N_exceed':<10} {'xi':<12} {'sigma':<12}\n")
    f.write("-"*80 + "\n")

    for r in stability_results:
        f.write(f"{r['feature']:<6} {r['quantile']:<10.2f} {r['threshold']:<12.2f} {r['n_exceedances']:<10} {r['xi']:<12.4f} {r['sigma']:<12.2f}\n")

    f.write("-"*80 + "\n\n")

    # Summary per feature
    f.write("STABILITY SUMMARY\n")
    f.write("-"*40 + "\n")
    for feat in [0, 1]:
        feat_results = [r for r in stability_results if r['feature'] == feat]
        xi_values = [r['xi'] for r in feat_results]
        f.write(f"Feature {feat}: xi range [{min(xi_values):.4f}, {max(xi_values):.4f}], ")
        f.write(f"mean xi = {np.mean(xi_values):.4f}\n")

    f.write("\n")
    all_xi = [r['xi'] for r in stability_results]
    if all([-0.3 <= xi <= 0.2 for xi in all_xi]):
        f.write("CONCLUSION: xi is STABLE across thresholds (-0.3 to 0.2 range) - GPD fit is reliable\n")
    elif all([-0.5 <= xi <= 0.3 for xi in all_xi]):
        f.write("CONCLUSION: xi is REASONABLY STABLE - GPD fit is acceptable\n")
    else:
        f.write(f"CONCLUSION: xi varies significantly [{min(all_xi):.3f}, {max(all_xi):.3f}] - GPD fit may be unreliable\n")

print(f"\nStability table written to: {results_path}")

# Create plots
print("\n" + "="*70)
print("Creating diagnostic plots...")
print("="*70)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for f in range(min(2, F)):
    # Get pooled data
    ts_all = y_train[:, :, f].flatten()
    ts_all = ts_all[ts_all >= 5]

    # Plot at q=0.95 and q=0.99
    for j, q in enumerate([0.95, 0.99]):
        u = np.percentile(ts_all, q * 100)
        z = ts_all[ts_all > u] - u

        # Fit GPD
        params = genpareto.fit(z, floc=0)
        xi, loc, sigma = params

        # QQ Plot
        ax_qq = axes[f, j*2]
        z_sorted = np.sort(z)
        n = len(z_sorted)
        theoretical_quantiles = genpareto.ppf(np.arange(1, n+1) / (n+1), xi, loc=0, scale=sigma)

        ax_qq.scatter(theoretical_quantiles, z_sorted, alpha=0.5, s=10, c='steelblue')
        max_val = max(theoretical_quantiles.max(), z_sorted.max())
        ax_qq.plot([0, max_val], [0, max_val], 'r--', lw=1.5, label='45° line')
        ax_qq.set_xlabel('Theoretical (GPD)', fontsize=10)
        ax_qq.set_ylabel('Empirical', fontsize=10)
        ax_qq.set_title(f'Feature {f}, q={q}\nQQ Plot (ξ={xi:.3f})', fontsize=11)
        ax_qq.legend(loc='lower right')
        ax_qq.grid(True, alpha=0.3)

        # Survival Plot
        ax_surv = axes[f, j*2+1]

        # Empirical survival function
        z_sorted = np.sort(z)
        emp_survival = 1 - np.arange(1, n+1) / (n+1)

        # Theoretical GPD survival
        x_theory = np.linspace(0, z_sorted.max(), 200)
        gpd_survival = 1 - genpareto.cdf(x_theory, xi, loc=0, scale=sigma)

        ax_surv.semilogy(z_sorted, emp_survival, 'o', markersize=3, alpha=0.5,
                        label='Empirical', color='steelblue')
        ax_surv.semilogy(x_theory, gpd_survival, 'r-', lw=2, label='GPD fit')
        ax_surv.set_xlabel('Exceedance', fontsize=10)
        ax_surv.set_ylabel('Survival P(X > x)', fontsize=10)
        ax_surv.set_title(f'Feature {f}, q={q}\nSurvival Plot (n={n})', fontsize=11)
        ax_surv.legend(loc='upper right')
        ax_surv.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gpd_diagnostic_plots.png'), dpi=150)
plt.close()
print(f"Diagnostic plots saved to: {os.path.join(output_dir, 'gpd_diagnostic_plots.png')}")

# Read and display stability table
print("\n" + "="*70)
print("RESULTS")
print("="*70)
with open(results_path, 'r') as f:
    print(f.read())
