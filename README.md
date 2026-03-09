# DSClustering

Tabular interpretable clustering based on Dempster-Shafer Theory and
Gradient Descent

## Features

- **GPU Acceleration** - 5-20x faster training with CUDA support
- **Parallel Clustering** - 2-8x faster algorithm selection with multi-core CPUs
- **Interpretable Rules** - Dempster-Shafer theory for explainable clustering
- **Auto-detection** - Automatically finds optimal number of clusters
- **Multiple Algorithms** - Tests K-Means, Agglomerative, and DBSCAN

## Description

This repository contains 1 implementation of the clustering:

- `DSClustering`

## Quick Start

```python
import pandas as pd
from cdsgd import DSClustering

# Load your data
data = pd.read_csv('data.csv')
X = data[['feature1', 'feature2', 'feature3']]

# Create clustering model with GPU acceleration
ds = DSClustering(X)
ds.generate_categorical_rules()
labels = ds.predict()

# Print interpretable rules
ds.print_most_important_rules()
```

## Performance Acceleration

CDSGD includes GPU acceleration and parallel processing for improved performance:

- **GPU Acceleration**: Use `use_cuda=True` (default) for 5-20x speedup
- **Parallel Processing**: Use `n_jobs=-1` (default) for 2-8x speedup
- **Configurable Training**: Use `train_size=0.8` for larger training sets

See [ACCELERATION_GUIDE.md](ACCELERATION_GUIDE.md) for details.
