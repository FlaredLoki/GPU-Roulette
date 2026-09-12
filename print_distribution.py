import pandas as pd
import numpy as np
from src.synthetic_data import generate_dataset

print("Generating 20,000 synthetic rows...")
df = generate_dataset(20000)

speedup = df['speedup_log2']
print(f"Mean speedup_log2: {speedup.mean():.4f}")
print(f"Median speedup_log2: {speedup.median():.4f}")
print(f"Min: {speedup.min():.4f}, Max: {speedup.max():.4f}")
print("\nPercentiles:")
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    print(f"{p}%: {np.percentile(speedup, p):.4f}")

import math
# Assuming threshold is 0.0 (1x speedup)
# Let's check what it is in train.py
threshold = 0.0
profitable = (speedup >= threshold).sum()
print(f"\nTotal rows >= 0.0 (Profitable): {profitable} out of 20000")

# Basic histogram
hist, bins = np.histogram(speedup, bins=20)
print("\nDistribution Histogram (speedup_log2):")
for i in range(len(hist)):
    count = hist[i]
    bar = '#' * int(count / 20000 * 50)
    print(f"{bins[i]:.2f} to {bins[i+1]:.2f} | {count:5d} | {bar}")
