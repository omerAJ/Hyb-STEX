import numpy as np

# Load train data
train_path = "/home/maincoder/Documents/Hyb-STEX/code/Hyb-STEX/data/ST-SSL_Dataset/NYCTaxi/train.npz"
data = np.load(train_path)
y = data['y']

print(f"y shape: {y.shape}")
y_train = y[:, 0, :, :]
print(f"y_train shape: {y_train.shape}")

print(f"\nData statistics:")
print(f"Min: {np.min(y_train):.4f}")
print(f"Max: {np.max(y_train):.4f}")
print(f"Mean: {np.mean(y_train):.4f}")
print(f"Median: {np.median(y_train):.4f}")
print(f"Std: {np.std(y_train):.4f}")

print(f"\nPercentiles:")
for p in [50, 75, 90, 95, 99]:
    print(f"  {p}th: {np.percentile(y_train, p):.4f}")

print(f"\nValues < 5: {np.sum(y_train < 5)} / {y_train.size} ({100*np.sum(y_train < 5)/y_train.size:.1f}%)")
print(f"Values >= 5: {np.sum(y_train >= 5)} / {y_train.size} ({100*np.sum(y_train >= 5)/y_train.size:.1f}%)")
print(f"Values == 0: {np.sum(y_train == 0)} / {y_train.size} ({100*np.sum(y_train == 0)/y_train.size:.1f}%)")
