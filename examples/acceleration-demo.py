#!/usr/bin/env python3
"""
Example demonstrating GPU acceleration and parallel processing features.
This example shows how to use the new acceleration features in CDSGD.
"""

import pandas as pd
import torch
from cdsgd import DSClustering

# Load iris data
data_path = "../data/iris.csv"
data = pd.read_csv(data_path)
X = data[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]

print("=" * 70)
print("CDSGD Acceleration Features Demo")
print("=" * 70)

# Example 1: GPU Acceleration
print("\n1. GPU Acceleration")
print("-" * 70)
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    # Use GPU acceleration
    ds_gpu = DSClustering(X, cluster=3, use_cuda=True, debug_mode=False)
    print("Created DSClustering with GPU acceleration enabled")
    print(f"Device: {ds_gpu.device}")
else:
    # Graceful fallback to CPU
    ds_cpu = DSClustering(X, cluster=3, use_cuda=True, debug_mode=False)
    print("CUDA not available - automatically using CPU")
    print(f"Device: {ds_cpu.device}")

# Example 2: Parallel Clustering Selection
print("\n2. Parallel Clustering Selection")
print("-" * 70)
print("Using n_jobs=-1 to parallelize across all CPU cores")

# Auto-detect best algorithm with parallel evaluation
ds_parallel = DSClustering(X, n_jobs=-1, use_cuda=False, debug_mode=False)
ds_parallel.generate_categorical_rules()
labels = ds_parallel.predict()

print(f"Best algorithm: {ds_parallel.selector.get_best_name()}")
print(f"Best silhouette score: {ds_parallel.selector.best_score_total:.3f}")
print(f"Found {len(set(labels))} clusters")

# Example 3: Configurable Training Size
print("\n3. Configurable Training Data Size")
print("-" * 70)

# Use 80% of data for training (default is 40%)
ds_large = DSClustering(X, cluster=3, train_size=0.8,
                        use_cuda=False, debug_mode=False)
print(f"Using {ds_large.train_size*100:.0f}% of data for training")
print(f"Total samples: {len(X)}, Training samples: ~{int(len(X) * ds_large.train_size)}")

# Example 4: Combined Features
print("\n4. Combined: GPU + Parallel + Large Training Set")
print("-" * 70)

ds_optimized = DSClustering(
    X,
    cluster=3,
    use_cuda=True,      # GPU acceleration
    n_jobs=-1,          # Parallel clustering selection
    train_size=0.8,     # Use 80% for training
    debug_mode=False
)
ds_optimized.generate_categorical_rules()
labels_opt = ds_optimized.predict()

print(f"Device: {ds_optimized.device}")
print(f"Parallel jobs: {ds_optimized.selector.n_jobs}")
print(f"Training size: {ds_optimized.train_size}")
print(f"Predictions: {len(labels_opt)} labels")

# Example 5: Backward Compatibility
print("\n5. Backward Compatibility")
print("-" * 70)
print("Existing code works without any changes:")

# Old-style usage still works
ds_old = DSClustering(X)
ds_old2 = DSClustering(X, 3)
ds_old3 = DSClustering(X, most_voted=True)

print("✓ DSClustering(X) - works")
print("✓ DSClustering(X, 3) - works")
print("✓ DSClustering(X, most_voted=True) - works")

print("\n" + "=" * 70)
print("Demo Complete!")
print("=" * 70)
print("\nKey Takeaways:")
print("  • GPU acceleration is enabled by default (use_cuda=True)")
print("  • Parallel processing is enabled by default (n_jobs=-1)")
print("  • All features have graceful fallbacks")
print("  • 100% backward compatible with existing code")
