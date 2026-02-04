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

# Quantiles to test
quantiles = [0.90, 0.95, 0.97, 0.99]

# Store results
results = []

print("\n" + "="*80)
print("NORMALIZED POOLING ANALYSIS")
print("Per-node threshold & scale, then pool normalized exceedances")
print("="*80)

for f in range(min(2, F)):
    print(f"\n{'='*60}")
    print(f"FEATURE {f}")
    print(f"{'='*60}")

    for q in quantiles:
        print(f"\n--- Quantile {q} ---")

        # Collect normalized exceedances from all nodes
        pooled_normalized = []
        nodes_contributing = 0
        total_exceedances = 0

        for n in range(N):
            ts = y_train[:, n, f]
            ts = ts[ts >= 5]  # Filter noise

            if len(ts) < 50:
                continue

            # Node-specific threshold
            u_n = np.percentile(ts, q * 100)

            # Node exceedances
            exc_mask = ts > u_n
            z_n = ts[exc_mask] - u_n

            if len(z_n) < 10:  # Need some exceedances
                continue

            # Normalize by mean exceedance (scale factor)
            scale_n = np.mean(z_n)
            if scale_n > 0:
                z_normalized = z_n / scale_n
                pooled_normalized.extend(z_normalized)
                nodes_contributing += 1
                total_exceedances += len(z_n)

        pooled_normalized = np.array(pooled_normalized)

        if len(pooled_normalized) < 100:
            print(f"  Too few pooled exceedances: {len(pooled_normalized)}")
            continue

        print(f"  Nodes contributing: {nodes_contributing}")
        print(f"  Total exceedances (raw): {total_exceedances}")
        print(f"  Pooled normalized exceedances: {len(pooled_normalized)}")

        # Fit GPD to normalized pooled exceedances
        # For normalized data, threshold is 0
        params = genpareto.fit(pooled_normalized, floc=0)
        xi, loc, sigma = params

        print(f"  GPD fit: xi = {xi:.4f}, sigma = {sigma:.4f}")

        # For exponential (xi=0), mean = sigma, so normalized data with mean=1 should give sigma≈1
        print(f"  (Expected sigma ≈ 1 for well-normalized data)")

        results.append({
            'feature': f,
            'quantile': q,
            'nodes': nodes_contributing,
            'n_pooled': len(pooled_normalized),
            'xi': xi,
            'sigma': sigma
        })

# Write results
results_path = os.path.join(output_dir, 'gpd_normalized_pooling.txt')
with open(results_path, 'w') as file:
    file.write("GPD Analysis with Normalized Pooling\n")
    file.write("="*80 + "\n")
    file.write("Method: Per-node threshold, normalize by mean exceedance, then pool\n")
    file.write("="*80 + "\n\n")

    file.write("STABILITY TABLE (Normalized Pooling)\n")
    file.write("-"*80 + "\n")
    file.write(f"{'Feat':<6} {'Quantile':<10} {'Nodes':<8} {'N_pooled':<12} {'xi':<12} {'sigma':<12}\n")
    file.write("-"*80 + "\n")

    for r in results:
        file.write(f"{r['feature']:<6} {r['quantile']:<10.2f} {r['nodes']:<8} {r['n_pooled']:<12} {r['xi']:<12.4f} {r['sigma']:<12.4f}\n")

    file.write("-"*80 + "\n\n")

    # Summary
    file.write("STABILITY SUMMARY\n")
    file.write("-"*40 + "\n")
    for feat in [0, 1]:
        feat_results = [r for r in results if r['feature'] == feat]
        if feat_results:
            xi_values = [r['xi'] for r in feat_results]
            file.write(f"Feature {feat}: xi range [{min(xi_values):.4f}, {max(xi_values):.4f}], ")
            file.write(f"mean xi = {np.mean(xi_values):.4f}\n")

    file.write("\n")
    all_xi = [r['xi'] for r in results]
    if all([-0.3 <= xi <= 0.2 for xi in all_xi]):
        file.write("CONCLUSION: xi is STABLE - GPD fit is reliable\n")
    elif all([-0.5 <= xi <= 0.3 for xi in all_xi]):
        file.write("CONCLUSION: xi is REASONABLY STABLE - GPD fit is acceptable\n")
    else:
        file.write(f"CONCLUSION: xi range [{min(all_xi):.3f}, {max(all_xi):.3f}]\n")

print(f"\nResults written to: {results_path}")

# Create diagnostic plots
print("\n" + "="*60)
print("Creating diagnostic plots...")
print("="*60)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for f in range(min(2, F)):
    for j, q in enumerate([0.95, 0.99]):
        # Collect normalized exceedances
        pooled_normalized = []
        for n in range(N):
            ts = y_train[:, n, f]
            ts = ts[ts >= 5]
            if len(ts) < 50:
                continue
            u_n = np.percentile(ts, q * 100)
            z_n = ts[ts > u_n] - u_n
            if len(z_n) < 10:
                continue
            scale_n = np.mean(z_n)
            if scale_n > 0:
                pooled_normalized.extend(z_n / scale_n)

        pooled_normalized = np.array(pooled_normalized)

        # Fit GPD
        params = genpareto.fit(pooled_normalized, floc=0)
        xi, loc, sigma = params

        # QQ Plot
        ax_qq = axes[f, j*2]
        z_sorted = np.sort(pooled_normalized)
        n_pts = len(z_sorted)
        theoretical_q = genpareto.ppf(np.arange(1, n_pts+1) / (n_pts+1), xi, loc=0, scale=sigma)

        ax_qq.scatter(theoretical_q, z_sorted, alpha=0.3, s=5, c='steelblue')
        max_val = max(theoretical_q.max(), z_sorted.max())
        ax_qq.plot([0, max_val], [0, max_val], 'r--', lw=1.5)
        ax_qq.set_xlabel('Theoretical (GPD)', fontsize=10)
        ax_qq.set_ylabel('Empirical (normalized)', fontsize=10)
        ax_qq.set_title(f'Feature {f}, q={q}\nQQ (ξ={xi:.3f}, σ={sigma:.3f})', fontsize=11)
        ax_qq.grid(True, alpha=0.3)

        # Survival Plot
        ax_surv = axes[f, j*2+1]
        emp_survival = 1 - np.arange(1, n_pts+1) / (n_pts+1)
        x_theory = np.linspace(0, z_sorted.max(), 200)
        gpd_survival = 1 - genpareto.cdf(x_theory, xi, loc=0, scale=sigma)

        ax_surv.semilogy(z_sorted, emp_survival, 'o', markersize=2, alpha=0.3, color='steelblue')
        ax_surv.semilogy(x_theory, gpd_survival, 'r-', lw=2)
        ax_surv.set_xlabel('Normalized exceedance', fontsize=10)
        ax_surv.set_ylabel('Survival P(X > x)', fontsize=10)
        ax_surv.set_title(f'Feature {f}, q={q}\nSurvival (n={n_pts})', fontsize=11)
        ax_surv.grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'gpd_normalized_plots.png'), dpi=150)
plt.close()
print(f"Plots saved to: {os.path.join(output_dir, 'gpd_normalized_plots.png')}")

# Print final results
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)
with open(results_path, 'r') as file:
    print(file.read())
