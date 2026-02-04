import numpy as np
from scipy.stats import genpareto
import os

# Paths
train_path = "/home/maincoder/Documents/Hyb-STEX/code/Hyb-STEX/data/ST-SSL_Dataset/NYCTaxi/train.npz"
test_path = "/home/maincoder/Documents/Hyb-STEX/code/Hyb-STEX/data/ST-SSL_Dataset/NYCTaxi/test.npz"

# Create output directory
output_dir = "/home/maincoder/Documents/Hyb-STEX/code/Hyb-STEX/data/ST-SSL_Dataset/NYCTaxi/evtdiag_quick"
os.makedirs(output_dir, exist_ok=True)

# Load train data
print("Loading train data...")
data = np.load(train_path)
y = data['y']
print(f"Original y shape: {y.shape}")

# Extract y_train = y[:, 0, :, :] - DON'T mask yet
y_train = y[:, 0, :, :]  # Shape: (S, N, F)
print(f"y_train shape: {y_train.shape}")
S, N, F = y_train.shape

# Store ALL detailed results
all_results = []

# Process each feature
for f in range(min(2, F)):  # Features 0 and 1
    print(f"\n{'='*60}")
    print(f"FEATURE {f}")
    print(f"{'='*60}")

    # Compute mean per node using raw data
    node_means = np.mean(y_train[:, :, f], axis=0)  # Shape: (N,)

    # Select ALL nodes
    top_indices = np.argsort(node_means)[::-1]  # All nodes sorted by mean
    print(f"\nAnalyzing all {len(top_indices)} nodes...")

    # Process each node
    for i, node_idx in enumerate(top_indices):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(top_indices)} nodes...")

        result = {
            'feature': f,
            'node': node_idx,
            'mean': node_means[node_idx],
            'n_samples_raw': len(y_train[:, node_idx, f]),
            'n_samples_filtered': None,
            'u95': None,
            'u99': None,
            'len_z95': None,
            'len_z99': None,
            'xi': None,
            'sigma': None,
            'fit_status': None
        }

        # Get time series for this node
        ts = y_train[:, node_idx, f]

        # Mask values < 5 (noise)
        ts = ts[ts >= 5]
        result['n_samples_filtered'] = len(ts)

        if len(ts) < 50:
            result['fit_status'] = f"SKIP: only {len(ts)} samples after filtering"
            all_results.append(result)
            continue

        # Compute thresholds
        u95 = np.percentile(ts, 95)
        u99 = np.percentile(ts, 99)
        result['u95'] = u95
        result['u99'] = u99

        # Compute exceedances
        z95 = ts[ts > u95] - u95
        z99 = ts[ts > u99] - u99
        result['len_z95'] = len(z95)
        result['len_z99'] = len(z99)

        # Fit GPD if enough exceedances
        if len(z95) >= 50:
            # Fit GPD (shape, loc=0, scale)
            params = genpareto.fit(z95, floc=0)
            xi, loc, sigma = params
            result['xi'] = xi
            result['sigma'] = sigma
            result['fit_status'] = "OK"
        else:
            result['fit_status'] = f"SKIP: only {len(z95)} exceedances"

        all_results.append(result)

# Feature-level analysis (all nodes aggregated)
print(f"\n{'='*60}")
print("FEATURE-LEVEL ANALYSIS (All Nodes Aggregated)")
print(f"{'='*60}")

feature_results = []
for f in range(min(2, F)):
    print(f"\nFeature {f}:")

    # Aggregate all nodes for this feature
    ts_all = y_train[:, :, f].flatten()

    # Mask values < 5
    ts_all = ts_all[ts_all >= 5]
    print(f"  Total samples after filtering: {len(ts_all)}")

    # Compute thresholds
    u95 = np.percentile(ts_all, 95)
    u99 = np.percentile(ts_all, 99)

    # Compute exceedances
    z95 = ts_all[ts_all > u95] - u95
    z99 = ts_all[ts_all > u99] - u99

    print(f"  u95 = {u95:.2f}, n_z95 = {len(z95)}")
    print(f"  u99 = {u99:.2f}, n_z99 = {len(z99)}")

    # Fit GPD
    params = genpareto.fit(z95, floc=0)
    xi, loc, sigma = params
    print(f"  GPD fit: xi = {xi:.4f}, sigma = {sigma:.2f}")

    feature_results.append({
        'feature': f,
        'n_samples': len(ts_all),
        'u95': u95,
        'u99': u99,
        'n_z95': len(z95),
        'n_z99': len(z99),
        'xi': xi,
        'sigma': sigma
    })

# Write comprehensive results file
print(f"\n{'='*60}")
print("Writing detailed results...")
results_path = os.path.join(output_dir, 'gpd_all_results.txt')

with open(results_path, 'w') as f:
    f.write("GPD Tail Fit Analysis - NYC Taxi Data (All Nodes, All Features)\n")
    f.write("="*80 + "\n\n")

    # Feature-level results first
    f.write("FEATURE-LEVEL ANALYSIS (All Nodes Aggregated)\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Feat':<6} {'N_samples':<12} {'u95':<10} {'n_z95':<10} {'u99':<10} {'n_z99':<10} {'xi':<12} {'sigma':<12}\n")
    f.write("-"*80 + "\n")
    for fr in feature_results:
        f.write(f"{fr['feature']:<6} {fr['n_samples']:<12} {fr['u95']:<10.2f} {fr['n_z95']:<10} {fr['u99']:<10.2f} {fr['n_z99']:<10} {fr['xi']:<12.4f} {fr['sigma']:<12.2f}\n")
    f.write("-"*80 + "\n\n")

    # Summary table header
    f.write("DETAILED RESULTS TABLE\n")
    f.write("-"*80 + "\n")
    f.write(f"{'Feat':<5} {'Node':<6} {'Mean':<10} {'N_filt':<8} {'u95':<10} {'n_z95':<7} {'u99':<10} {'n_z99':<7} {'xi':<10} {'sigma':<10} {'Status'}\n")
    f.write("-"*80 + "\n")

    for r in all_results:
        u95_str = f"{r['u95']:.2f}" if r['u95'] is not None else "N/A"
        u99_str = f"{r['u99']:.2f}" if r['u99'] is not None else "N/A"
        z95_str = str(r['len_z95']) if r['len_z95'] is not None else "N/A"
        z99_str = str(r['len_z99']) if r['len_z99'] is not None else "N/A"
        xi_str = f"{r['xi']:.4f}" if r['xi'] is not None else "N/A"
        sigma_str = f"{r['sigma']:.2f}" if r['sigma'] is not None else "N/A"
        n_filt_str = str(r['n_samples_filtered']) if r['n_samples_filtered'] is not None else "N/A"

        f.write(f"{r['feature']:<5} {r['node']:<6} {r['mean']:<10.2f} {n_filt_str:<8} {u95_str:<10} {z95_str:<7} {u99_str:<10} {z99_str:<7} {xi_str:<10} {sigma_str:<10} {r['fit_status']}\n")

    f.write("-"*80 + "\n\n")

    # Summary statistics
    valid_results = [r for r in all_results if r['xi'] is not None]
    f.write("SUMMARY STATISTICS\n")
    f.write("-"*40 + "\n")
    f.write(f"Total nodes analyzed: {len(all_results)}\n")
    f.write(f"Nodes with valid GPD fits: {len(valid_results)}\n")
    f.write(f"Nodes skipped: {len(all_results) - len(valid_results)}\n\n")

    if valid_results:
        xi_values = [r['xi'] for r in valid_results]
        sigma_values = [r['sigma'] for r in valid_results]

        f.write(f"Shape parameter (xi) statistics:\n")
        f.write(f"  Min:    {min(xi_values):.4f}\n")
        f.write(f"  Max:    {max(xi_values):.4f}\n")
        f.write(f"  Mean:   {np.mean(xi_values):.4f}\n")
        f.write(f"  Median: {np.median(xi_values):.4f}\n")
        f.write(f"  Std:    {np.std(xi_values):.4f}\n\n")

        f.write(f"Scale parameter (sigma) statistics:\n")
        f.write(f"  Min:    {min(sigma_values):.2f}\n")
        f.write(f"  Max:    {max(sigma_values):.2f}\n")
        f.write(f"  Mean:   {np.mean(sigma_values):.2f}\n")
        f.write(f"  Median: {np.median(sigma_values):.2f}\n\n")

        # Per-feature breakdown
        for feat in [0, 1]:
            feat_results = [r for r in valid_results if r['feature'] == feat]
            if feat_results:
                feat_xi = [r['xi'] for r in feat_results]
                f.write(f"Feature {feat}: {len(feat_results)} valid fits, xi range [{min(feat_xi):.4f}, {max(feat_xi):.4f}]\n")

        f.write("\n")

    # Conclusion
    f.write("CONCLUSION\n")
    f.write("-"*40 + "\n")
    if valid_results:
        xi_values = [r['xi'] for r in valid_results]
        if max(xi_values) < 0.5 and min(xi_values) > -0.5:
            f.write("GPD looks PLAUSIBLE: All xi values in reasonable range [-0.5, 0.5]\n")
        elif max(xi_values) < 1.0 and min(xi_values) > -1.0:
            f.write(f"GPD MARGINALLY PLAUSIBLE: xi range [{min(xi_values):.3f}, {max(xi_values):.3f}] slightly outside typical bounds\n")
        else:
            f.write(f"GPD QUESTIONABLE: Extreme xi values [{min(xi_values):.3f}, {max(xi_values):.3f}]\n")

        # Interpret the sign of xi
        mean_xi = np.mean(xi_values)
        if mean_xi < -0.1:
            f.write("Interpretation: Negative xi (light tails) - data has finite upper bound, typical for bounded demand\n")
        elif mean_xi > 0.1:
            f.write("Interpretation: Positive xi (heavy tails) - data has heavier-than-exponential tails\n")
        else:
            f.write("Interpretation: xi near zero - approximately exponential tail behavior\n")
    else:
        f.write("GPD NOT PLAUSIBLE: No valid fits obtained\n")

print(f"Results written to: {results_path}")
print("\nDone! Check the evtdiag_quick/ folder for plots and detailed results.")
