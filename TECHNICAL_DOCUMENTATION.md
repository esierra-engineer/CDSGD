# DSClustering Technical Documentation

## Overview

**DSClustering** is a tabular interpretable clustering algorithm that combines Dempster-Shafer Theory with Gradient Descent optimization. It extends the `DSClassifierMultiQ` class from the DSGD library to provide explainable clustering with human-readable rules.

**Version:** 0.1
**Authors:** Ricardo Valdivia
**Repository:** https://github.com/ricardo-valdivia/CDSGD
**Last Updated:** 2026-03-09

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Class Reference](#class-reference)
3. [Memory Usage Analysis](#memory-usage-analysis)
4. [Performance Bottlenecks](#performance-bottlenecks)
5. [CUDA Acceleration Opportunities](#cuda-acceleration-opportunities)
6. [API Reference](#api-reference)
7. [Usage Examples](#usage-examples)
8. [Dependencies](#dependencies)

---

## Architecture Overview

### Design Pattern

DSClustering follows a two-phase approach:

1. **Ensemble Clustering Phase** (via `ClusteringSelector`)
   - Tests multiple clustering algorithms: K-Means, Agglomerative Clustering, DBSCAN
   - Evaluates performance using silhouette scores
   - Selects best algorithm or combines results via voting

2. **Rule Generation Phase** (via `DSClassifierMultiQ`)
   - Uses Dempster-Shafer Theory to generate interpretable rules
   - Optimizes rule weights using gradient descent (PyTorch)
   - Provides explainability through mass function assignments

### Data Flow

```
Input Data (DataFrame)
    ↓
ClusteringSelector
    ├─ K-Means (multiple n_clusters)
    ├─ Agglomerative (multiple linkages)
    └─ DBSCAN (multiple eps/min_samples)
    ↓
Best/Voted Labels
    ↓
DSClassifierMultiQ (parent class)
    ├─ Generate categorical rules
    ├─ Optimize with gradient descent
    └─ Predict with rule-based inference
    ↓
Final Cluster Assignments + Explanations
```

---

## Class Reference

### DSClustering

**Location:** `cdsgd/DSClustering.py`

**Inheritance:** `DSClassifierMultiQ` (from dsgd library)

#### Constructor Parameters

```python
def __init__(
    self,
    data: pd.DataFrame,           # Input feature data
    cluster: Union[int, None] = None,  # Number of clusters (None = auto-detect)
    most_voted: bool = False,     # Use voting across algorithms
    min_iter: int = 50,           # Minimum training iterations
    max_iter: int = 400,          # Maximum training iterations
    debug_mode: bool = True,      # Enable detailed training output
    lossfn: str = "MSE",          # Loss function ("MSE" or "CE")
    num_workers: int = 0,         # DataLoader workers
    min_dloss: float = 1e-7       # Convergence threshold
)
```

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | pd.DataFrame | Original input feature data |
| `selector` | ClusteringSelector | Handles algorithm selection |
| `cluster_labels_df` | pd.DataFrame | Labels from all tested algorithms |
| `best` | np.ndarray | Best or most-voted cluster labels |
| `cluster` | int | Final number of clusters |
| `df_with_labels` | pd.DataFrame | Features concatenated with algorithm labels |
| `y_pred` | np.ndarray | Final predicted cluster assignments |

#### Methods

##### `generate_categorical_rules()`

Generates interpretable rules from the clustering algorithms' labels.

**Memory Impact:** Moderate - stores rule definitions in model
**Time Complexity:** O(n × m × k) where n=samples, m=features, k=clusters

```python
def generate_categorical_rules(self) -> None
```

##### `fit()`

Trains the model using gradient descent optimization.

**Memory Impact:** High - creates training split, PyTorch tensors, gradients
**Time Complexity:** O(iterations × n × rules × k)

**Fixed Compatibility Issue:** Handles different return values based on `debug_mode`:
- `debug_mode=True`: Returns (losses, epoch, dt)
- `debug_mode=False`: Returns (losses, epoch, None)

```python
def fit(self) -> Tuple[List[float], int, Optional[float]]
```

**Training Process:**
1. Splits data (40% train, 60% unused) for rule fitting
2. Generates single-feature rules (3 breaks) and multi-feature rules
3. Optimizes rule weights using Adam or SGD optimizer
4. Applies MSE or Cross-Entropy loss
5. Normalizes masses after each iteration

##### `predict()`

Performs clustering by fitting the model and predicting labels.

**Memory Impact:** Very High - calls fit() internally
**Time Complexity:** Same as fit() + O(n × rules)

```python
def predict(self) -> np.ndarray
```

**Note:** This method calls `fit()` internally, so calling `fit()` separately is unnecessary.

##### `print_most_important_rules(classes=None, threshold=0.2)`

Prints the most influential rules for each cluster.

**Output:** Console output with rule descriptions and mass assignments
**Filters out:** Algorithm label rules (K-Means/DBSCAN/Agglomerative Labels)

```python
def print_most_important_rules(
    self,
    classes: Optional[List[int]] = None,
    threshold: float = 0.2
) -> None
```

##### `metrics(y=None)`

Evaluates clustering performance with comprehensive metrics.

**Metrics Computed:**
- **With ground truth (y):**
  - Accuracy
  - F1 Score (Macro and Micro)
  - Confusion Matrix
  - Adjusted Rand Index
  - Pearson Correlation
- **Always:**
  - Silhouette Score

```python
def metrics(self, y: Optional[np.ndarray] = None) -> None
```

##### `predict_explain(x)`

Explains cluster assignment for a single instance.

**Returns:**
- Predicted cluster label
- Cluster class
- DataFrame of applicable rules
- Human-readable explanation string

```python
def predict_explain(
    self,
    x: np.ndarray
) -> Tuple[int, int, pd.DataFrame, str]
```

---

### ClusteringSelector

**Location:** `cdsgd/ClusteringSelector.py`

Helper class that evaluates multiple clustering algorithms to find the best fit.

#### Constructor Parameters

```python
def __init__(
    self,
    data: pd.DataFrame,
    cluster: Union[int, None] = None
)
```

#### Key Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `data` | pd.DataFrame | Input features |
| `best_algorithm` | str | Name of best algorithm |
| `best_params` | list | Parameters of best algorithm |
| `best_labels` | np.ndarray | Labels from best algorithm |
| `best_score_total` | float | Highest silhouette score |
| `kmeans_labels` | np.ndarray | Labels from K-Means |
| `agglomerative_labels` | np.ndarray | Labels from Agglomerative |
| `dbscan_labels` | np.ndarray | Labels from DBSCAN |

#### Algorithm Search Space

**K-Means:**
- `n_clusters`: [2, 3, 4] (or specified value)

**Agglomerative Clustering:**
- `n_clusters`: [2, 3, 4] (or specified value)
- `linkage`: ['ward', 'complete', 'average']

**DBSCAN:**
- `eps`: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
- `min_samples`: [2, 3, 4, 5, 6]

**Total Configurations Tested:**
- K-Means: 3 configs (or 1 if cluster specified)
- Agglomerative: 9 configs (3 clusters × 3 linkages)
- DBSCAN: 30 configs (6 eps × 5 min_samples)
- **Total: 42 clustering runs** (or 40 if cluster specified)

#### Methods

##### `select_best_clustering()`

Tests all algorithm configurations and selects the best.

**Memory Impact:** Low-Moderate - stores only best labels from each algorithm
**Time Complexity:** O(42 × clustering_cost)
- K-Means: O(n × k × iterations)
- Agglomerative: O(n² log n) for ward linkage
- DBSCAN: O(n log n) with spatial indexing

##### `get_cluster_labels_df()`

Returns DataFrame with normalized labels from all three algorithms.

**Memory Impact:** Low - creates DataFrame with 3 columns
**Return:** DataFrame with columns: "K-Means Labels", "Agglomerative Labels", "DBSCAN Labels"

##### `get_most_voted()`

Computes mode label across algorithms for each sample.

**Memory Impact:** Low
**Time Complexity:** O(n)

---

## Memory Usage Analysis

### Memory Consumption Breakdown

#### 1. **Initialization Phase**

```python
ds = DSClustering(data)  # data is n × m DataFrame
```

**Memory allocated:**
- Original data: `n × m × 8 bytes` (float64)
- ClusteringSelector:
  - 3 label arrays: `3 × n × 8 bytes`
  - Intermediate arrays during testing: `~n × 8 bytes` (reused)
- `cluster_labels_df`: `n × 3 × 8 bytes`
- `df_with_labels`: `n × (m + 3) × 8 bytes`

**Total Initialization:** ≈ `n × (2m + 27) × 8 bytes`

**Example:** For Iris (n=150, m=4):
- ≈ 150 × (8 + 27) × 8 = **42 KB**

#### 2. **Rule Generation Phase**

```python
ds.generate_categorical_rules()
```

**Memory allocated:**
- Categorical rules storage: `~3 × k × 8 bytes` per rule
- Number of rules: typically 3k (one per algorithm label)

**Total:** ≈ `72k bytes` (usually < 1 KB)

#### 3. **Training Phase (fit)**

```python
ds.fit()  # Most memory-intensive phase
```

**Memory allocated:**

a) **Train/Test Split:**
   - X_train: `0.4n × (m+3) × 8 bytes`
   - y_train: `0.4n × 8 bytes`

b) **PyTorch Tensors:**
   - Input tensor: `0.4n × (m+3+1) × 4 bytes` (float32)
   - Target tensor: `0.4n × k × 4 bytes` (one-hot for MSE loss)
   - Prediction tensor: `0.4n × k × 4 bytes`

c) **Rule Storage:**
   - Single rules: `~3k × (m+3) rules` × (8 bytes for breaks + 32 bytes for masses)
     ≈ `3k(m+3) × 40 bytes`
   - Multiplication rules: `~(m+3)² rules` × 40 bytes
     ≈ `(m+3)² × 40 bytes`

d) **Gradients:**
   - One gradient tensor per rule parameter
   - Approximately matches rule storage size

e) **Optimizer State (Adam):**
   - 2 state tensors per parameter (momentum and velocity)
   - ≈ `2 × total_parameters × 4 bytes`

**Example (Iris):**
- Train split: 150 × 0.4 = 60 samples, 7 features
- Single rules: 3 × 3 × 7 = 63 rules
- Mult rules: 49 rules
- Total rules: ~112 rules
- Parameters: ~112 × 3 (masses per rule) = 336 parameters

**Memory estimate:**
- Data: 60 × 7 × 8 = 3.4 KB
- Tensors: 60 × (7 + 3) × 4 × 3 = 7.2 KB
- Rules: 112 × 40 = 4.5 KB
- Gradients: 4.5 KB
- Optimizer: 336 × 2 × 4 = 2.7 KB
- **Total Training:** ≈ **22 KB**

**Large Dataset (n=100k, m=50):**
- Train split: 40k samples, 53 features
- Single rules: ~500 rules
- Mult rules: ~2800 rules
- Total: ~3300 rules, ~10k parameters

**Memory estimate:**
- Data: 40k × 53 × 8 = 17 MB
- Tensors: 40k × 54 × 4 × 3 = 26 MB
- Rules: 3300 × 40 = 132 KB
- Gradients: 132 KB
- Optimizer: 10k × 2 × 4 = 80 KB
- **Total Training:** ≈ **45 MB**

#### 4. **Prediction Phase**

```python
labels = ds.predict()  # Calls fit() + prediction
```

**Additional memory:**
- Prediction output: `n × k × 4 bytes` (temporary)
- Final labels: `n × 8 bytes`

**Peak memory:** Same as training phase

### Memory Optimization Recommendations

1. **Use fewer workers:** Set `num_workers=0` to avoid multiprocessing overhead
2. **Reduce batch size:** For large datasets, use smaller batches (default is full dataset)
3. **Limit rule generation:** Reduce `single_rules_breaks` parameter
4. **Skip voting:** Use `most_voted=False` to save 3 label arrays
5. **Clear after fit:** Delete intermediate data after training if only predictions needed

---

## Performance Bottlenecks

### Identified Bottlenecks

#### 1. **ClusteringSelector.select_best_clustering()**

**Severity:** HIGH
**Location:** cdsgd/ClusteringSelector.py:58-136

**Issue:**
- Tests 42 different algorithm configurations sequentially
- Each configuration computes silhouette score: O(n²)
- Total: 42 × O(n²) operations

**Time Complexity:**
- K-Means fitting: O(n × k × i) where i = iterations (typically 100-300)
- Agglomerative (ward): O(n² log n)
- DBSCAN: O(n log n) with KD-tree
- Silhouette score: O(n²)

**Measurements (estimated for n samples):**
- n=150: ~0.5 seconds
- n=1000: ~5 seconds
- n=10000: ~200 seconds
- n=100000: ~hours (prohibitive)

**Speedup Opportunities:**
1. **Parallelize algorithm testing** - Test configurations in parallel
2. **Early stopping** - Stop if score exceeds threshold
3. **Subsample for large datasets** - Use stratified sample for algorithm selection
4. **Cache silhouette** - Compute once per algorithm, not per parameter set
5. **Use faster metrics** - Replace silhouette with Calinski-Harabasz (O(n))

#### 2. **DSClassifierMultiQ.fit() - Gradient Descent**

**Severity:** MEDIUM-HIGH
**Location:** Parent class in dsgd library

**Issue:**
- Full batch gradient descent (no mini-batching by default)
- Computes loss on entire training set each epoch
- No GPU utilization by default

**Time Complexity per epoch:**
- Forward pass: O(n × rules × k)
- Loss computation: O(n × k)
- Backward pass: O(n × rules × k)
- Parameter update: O(rules)

**Measurements (estimated):**
- 150 samples, 112 rules: ~0.01s per epoch
- 400 epochs: ~4 seconds
- 10k samples, 500 rules: ~0.5s per epoch
- 400 epochs: ~200 seconds

**Speedup Opportunities:**
1. **Use mini-batch training** - Reduce memory and enable parallelization
2. **GPU acceleration** - Move tensors to CUDA
3. **Reduce max_iter** - Use convergence criteria more aggressively
4. **Learning rate scheduling** - Converge faster with adaptive LR

#### 3. **Rule Generation**

**Severity:** MEDIUM
**Location:** generate_categorical_rules() + parent class

**Issue:**
- Generates rules for all features and all algorithms' labels
- Statistical rule generation computes quantile breaks: O(n log n) per feature

**Time Complexity:**
- Categorical rules: O(unique_values × k)
- Statistical single rules: O(m × n log n)
- Multiplication rules: O(m²)

**Speedup Opportunities:**
1. **Skip redundant rules** - Don't generate rules for algorithm labels
2. **Parallelize feature processing** - Generate rules in parallel
3. **Cache quantiles** - Reuse breaks across features with similar distributions

#### 4. **train_test_split in fit()**

**Severity:** LOW-MEDIUM
**Location:** DSClustering.py:60

**Issue:**
- Uses only 40% of data for training (60% discarded)
- Creates unnecessary copy of entire dataset

**Impact:**
- Wastes 60% of available training data
- Reduces model quality for small datasets
- Memory overhead for large datasets

**Speedup Opportunities:**
1. **Use full dataset** - Remove split or make configurable
2. **Use validation set** - If split needed, use for validation not discard

#### 5. **Label Normalization**

**Severity:** LOW
**Location:** ClusteringSelector.normalize_labels() and get_cluster_labels_df()

**Issue:**
- Iterates over all samples to build correspondence dictionary: O(n)
- Called 3 times per instance (once per algorithm)

**Speedup Opportunities:**
1. **Vectorize** - Use NumPy operations instead of Python loops
2. **Cache** - Store normalized labels instead of recomputing

### Performance Summary Table

| Component | Time Complexity | Memory | Bottleneck Severity | GPU-Friendly? |
|-----------|----------------|--------|---------------------|---------------|
| ClusteringSelector | O(42 × n²) | O(n) | HIGH | Partial |
| generate_rules | O(m × n log n) | O(rules) | MEDIUM | No |
| fit (GD) | O(iter × n × rules × k) | O(n × m + rules) | MEDIUM-HIGH | YES |
| predict | O(n × rules × k) | O(n) | LOW-MEDIUM | YES |

---

## CUDA Acceleration Opportunities

### Current CUDA Support

**PyTorch Backend:**
- DSClustering uses PyTorch for gradient descent optimization
- Tensors are created on CPU by default
- No explicit GPU device management

### CUDA-Accelerated Components

#### 1. **Gradient Descent Training** (HIGH IMPACT)

**Current Implementation:**
```python
# In dsgd.DSClassifierMultiQ
optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
```

**CUDA Optimization:**
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
self.model = self.model.to(device)
X_tensor = X_tensor.to(device)
y_tensor = y_tensor.to(device)
```

**Expected Speedup:**
- Small datasets (n < 1000): **1-2x** (GPU overhead dominates)
- Medium datasets (1000 < n < 100k): **5-20x**
- Large datasets (n > 100k): **10-50x**

**Implementation Complexity:** LOW
- Add device parameter to constructor
- Move model and tensors to device
- Minimal code changes required

#### 2. **Batch Processing in Prediction** (MEDIUM IMPACT)

**Current:**
- Predicts on full dataset at once
- Single forward pass on CPU

**CUDA Optimization:**
- Batch predictions on GPU
- Leverage tensor parallelism

**Expected Speedup:**
- Small: **1-2x**
- Medium: **5-10x**
- Large: **10-30x**

**Implementation Complexity:** LOW

#### 3. **Silhouette Score Computation** (MEDIUM-HIGH IMPACT)

**Current:**
- sklearn's silhouette_score uses CPU NumPy
- O(n²) distance matrix computation
- Called 42 times in ClusteringSelector

**CUDA Optimization:**
```python
# Use cupy or PyTorch for distance computation
import torch
def silhouette_score_cuda(X, labels):
    X_tensor = torch.tensor(X).cuda()
    # Compute pairwise distances on GPU
    dists = torch.cdist(X_tensor, X_tensor)
    # Compute silhouette on GPU
    ...
```

**Expected Speedup:**
- Medium datasets: **10-30x** (distance computation is highly parallel)
- Large datasets: **30-100x**

**Implementation Complexity:** MEDIUM
- Requires custom implementation or cupy/cuML integration
- sklearn doesn't support GPU

#### 4. **K-Means Clustering** (HIGH IMPACT)

**Current:**
- sklearn.cluster.KMeans uses CPU
- Called 3 times with different n_clusters

**CUDA Optimization:**
```python
# Option 1: Use cuML (RAPIDS)
from cuml.cluster import KMeans as cuKMeans
kmeans = cuKMeans(n_clusters=n_clusters)

# Option 2: PyTorch implementation
from fast_pytorch_kmeans import KMeans
kmeans = KMeans(n_clusters=n_clusters, mode='euclidean')
```

**Expected Speedup:**
- Medium datasets: **5-15x**
- Large datasets: **10-50x**

**Implementation Complexity:** MEDIUM
- Requires additional dependencies (cuML or custom)
- API differences to handle

#### 5. **Agglomerative Clustering** (LOW-MEDIUM IMPACT)

**Current:**
- sklearn.cluster.AgglomerativeClustering
- Not inherently parallel

**CUDA Optimization:**
- Limited GPU implementations available
- Distance matrix computation can be GPU-accelerated
- Linkage computation less amenable to GPU

**Expected Speedup:**
- **2-5x** (distance computation only)

**Implementation Complexity:** HIGH

#### 6. **DBSCAN** (LOW IMPACT)

**Current:**
- sklearn.cluster.DBSCAN
- Spatial indexing (KD-tree) not GPU-friendly

**CUDA Optimization:**
```python
from cuml.cluster import DBSCAN as cuDBSCAN
```

**Expected Speedup:**
- **2-8x** for medium-large datasets

**Implementation Complexity:** MEDIUM

### Recommended CUDA Implementation Strategy

#### Phase 1: Quick Wins (LOW complexity, HIGH impact)
1. Add device parameter to DSClustering
2. Move gradient descent to GPU
3. Move prediction to GPU

**Estimated effort:** 2-4 hours
**Expected speedup:** 5-20x on training

#### Phase 2: Algorithm Selection (MEDIUM complexity, HIGH impact)
1. Implement GPU-accelerated silhouette score
2. Add cuML K-Means support (optional fallback to sklearn)

**Estimated effort:** 1-2 days
**Expected speedup:** 10-30x on ClusteringSelector

#### Phase 3: Full GPU Pipeline (MEDIUM-HIGH complexity)
1. Add cuML support for DBSCAN and Agglomerative
2. Implement batch processing throughout
3. Add memory management for large datasets

**Estimated effort:** 3-5 days
**Expected speedup:** 20-50x end-to-end

### CUDA Code Example

```python
class DSClustering(DSClassifierMultiQ):
    def __init__(self, data, cluster=None, use_cuda=True, **kwargs):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        # ... existing initialization ...
        super().__init__(cluster, device=self.device, **kwargs)

    def fit(self):
        X_train, _, y_train, _ = train_test_split(...)

        # Move to GPU
        X_train_tensor = torch.tensor(X_train).to(self.device)
        y_train_tensor = torch.tensor(y_train).to(self.device)

        # Fit with GPU tensors
        result = super().fit(X_train_tensor, y_train_tensor, ...)
        return result

    def predict(self):
        self.fit()

        # Predict on GPU
        X_tensor = torch.tensor(self.df_with_labels.values).to(self.device)
        y_pred = super().predict(X_tensor)

        # Move back to CPU for compatibility
        return y_pred.cpu().numpy()
```

### GPU Memory Considerations

**CUDA Memory Requirements:**
- Same as CPU memory (tensors only)
- Plus ~500MB PyTorch/CUDA overhead

**For large datasets (n=100k, m=50):**
- Training tensors: ~45 MB (see Memory Analysis)
- Fits comfortably on any GPU (>2GB)

**For very large datasets (n=1M, m=100):**
- Training tensors: ~500 MB
- Requires mid-range GPU (>4GB)
- Consider batch processing

---

## API Reference

### Installation

```bash
pip install -e .
# Or with updated requirements:
pip install -r requirements-updated.txt
pip install -e .
```

**Requirements:**
- Python 3.10+
- numpy >= 2.0.0
- pandas >= 3.0.0
- scikit-learn >= 1.5.0
- scipy >= 1.10.0
- torch >= 2.0.0

### Quick Start

```python
import pandas as pd
from cdsgd import DSClustering

# Load your data
data = pd.read_csv('data.csv')
X = data[['feature1', 'feature2', 'feature3']]

# Auto-detect number of clusters
ds = DSClustering(X)
ds.generate_categorical_rules()
labels = ds.predict()

# Print interpretable rules
ds.print_most_important_rules()

# Evaluate (optional ground truth)
ds.metrics(y_true)
```

### Parameter Tuning Guide

#### `cluster` (int or None)
- **Default:** None (auto-detect)
- **Recommendation:**
  - None for exploratory analysis
  - Specify if you know the true number of clusters
  - Try values in range [2, 10] for tabular data

#### `most_voted` (bool)
- **Default:** False
- **Recommendation:**
  - False for fastest performance (uses best algorithm only)
  - True for robustness (ensemble voting)

#### `min_iter` / `max_iter` (int)
- **Default:** 50 / 400
- **Recommendation:**
  - Small datasets (n < 1000): min=20, max=200
  - Large datasets (n > 10k): min=50, max=100 (use early stopping)
  - Increase if model hasn't converged

#### `debug_mode` (bool)
- **Default:** True
- **Recommendation:**
  - True for development (verbose output)
  - False for production (faster)

#### `lossfn` (str)
- **Default:** "MSE"
- **Options:** "MSE" or "CE"
- **Recommendation:**
  - "MSE" for most cases (works well with soft assignments)
  - "CE" if you want hard cluster assignments

#### `num_workers` (int)
- **Default:** 0
- **Recommendation:**
  - 0 for small datasets (no multiprocessing overhead)
  - 2-4 for large datasets with SSD storage
  - Avoid on Windows (multiprocessing issues)

#### `min_dloss` (float)
- **Default:** 1e-7
- **Recommendation:**
  - 1e-7 for standard convergence
  - 1e-5 for faster (less accurate) convergence
  - 1e-9 for very precise convergence

---

## Usage Examples

### Example 1: Basic Clustering

```python
import pandas as pd
from cdsgd import DSClustering
from sklearn.datasets import load_iris

# Load data
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)

# Cluster with auto-detection
ds = DSClustering(X)
ds.generate_categorical_rules()
labels = ds.predict()

print(f"Found {len(set(labels))} clusters")
```

### Example 2: With Known Cluster Count

```python
# Specify 3 clusters
ds = DSClustering(X, cluster=3)
ds.generate_categorical_rules()
labels = ds.predict()

# Print interpretable rules
ds.print_most_important_rules()
```

### Example 3: With Evaluation

```python
# With ground truth labels
y_true = iris.target

ds = DSClustering(X, cluster=3)
ds.generate_categorical_rules()
labels = ds.predict()

# Comprehensive metrics
ds.metrics(y_true)
```

**Output:**
```
Information of DSClassifier

Accuracy: 89.3%
F1 Macro: 0.891
F1 Micro: 0.893

Confusion Matrix:
[[50  0  0]
 [ 0 45  5]
 [ 0 11 39]]
------------------
Clustering Metrics
Rand_index:  0.823
Pearson:  0.907
------------------------------------------------
Silhoutte:  0.551
```

### Example 4: Explain Individual Predictions

```python
ds = DSClustering(X, cluster=3)
ds.generate_categorical_rules()
labels = ds.predict()

# Explain a single instance
sample = X.iloc[0].values
pred, cls, rules_df, explanation = ds.predict_explain(sample)

print(f"Predicted cluster: {pred}")
print(f"\nExplanation:\n{explanation}")
print(f"\nApplicable rules:\n{rules_df}")
```

### Example 5: Ensemble Voting

```python
# Use voting across all algorithms
ds = DSClustering(X, cluster=3, most_voted=True)
ds.generate_categorical_rules()
labels = ds.predict()

# More robust but slightly slower
```

### Example 6: Fast Mode for Large Datasets

```python
# Optimize for large datasets
ds = DSClustering(
    X,
    cluster=5,
    min_iter=20,
    max_iter=100,
    debug_mode=False,
    num_workers=0,
    min_dloss=1e-5
)
ds.generate_categorical_rules()
labels = ds.predict()
```

---

## Dependencies

### Core Dependencies

- **numpy** (>=2.0.0): Array operations, numerical computing
- **pandas** (>=3.0.0): DataFrame handling, data manipulation
- **scikit-learn** (>=1.5.0): Clustering algorithms, metrics, preprocessing
- **scipy** (>=1.10.0): Statistical functions (Pearson correlation)
- **torch** (>=2.0.0): Neural network backend, gradient descent optimization

### External Dependencies

- **dsgd** (from GitHub): Parent class library providing Dempster-Shafer classifier
  - Repository: https://github.com/Sergio-P/DSGD.git
  - Provides: DSClassifierMultiQ, DSModel, rule generation

### Optional Dependencies (for CUDA)

- **cuml** (RAPIDS): GPU-accelerated clustering algorithms
- **cupy**: GPU-accelerated NumPy operations
- **fast-pytorch-kmeans**: GPU K-Means implementation

---

## Compatibility Notes

### Environment Compatibility (Updated 2026-03-09)

✅ **TESTED AND WORKING:**
- Python 3.12.3
- numpy 2.4.3
- pandas 3.0.1
- scikit-learn 1.8.0
- scipy 1.17.1
- torch 2.10.0+cu128

✅ **KNOWN COMPATIBLE:**
- Python 3.10, 3.11, 3.12
- numpy >= 2.0
- pandas >= 3.0
- scikit-learn >= 1.5
- scipy >= 1.10
- torch >= 2.0

⚠️ **COMPATIBILITY FIX APPLIED:**
- Fixed return value unpacking in `fit()` method
- Now correctly handles different return values based on `debug_mode` setting

❌ **KNOWN ISSUES:**
- Original requirements.txt (2022-era versions) may have incompatibilities
- Use `requirements-updated.txt` for modern environments

### Platform Support

- ✅ Linux (tested on Ubuntu)
- ✅ macOS (should work, not extensively tested)
- ✅ Windows (should work, avoid `num_workers > 0`)

### CUDA Support

- ✅ PyTorch with CUDA backend works out of the box
- ❌ Explicit CUDA device management not implemented (tensors default to CPU)
- ⚡ Potential for significant speedup with GPU implementation (see CUDA section)

---

## Changelog

### Version 0.1 (2026-03-09)
- ✅ Fixed compatibility with modern package versions
- ✅ Fixed `fit()` method return value handling
- ✅ Added comprehensive technical documentation
- ✅ Identified performance bottlenecks
- ✅ Analyzed memory usage patterns
- ✅ Documented CUDA acceleration opportunities

---

## Future Improvements

### High Priority
1. **GPU Acceleration** - Implement CUDA support for gradient descent
2. **Parallel Algorithm Selection** - Speed up ClusteringSelector
3. **Memory Optimization** - Reduce memory footprint for large datasets
4. **Convergence Optimization** - Better early stopping criteria

### Medium Priority
1. **Mini-batch Training** - Enable batch processing for large datasets
2. **Hyperparameter Tuning** - Add automatic parameter selection
3. **Incremental Clustering** - Support online/incremental learning
4. **Additional Metrics** - Add more evaluation metrics

### Low Priority
1. **Visualization Tools** - Add built-in plotting functions
2. **Rule Pruning** - Remove redundant/low-importance rules
3. **Export/Import** - Save/load trained models
4. **Cross-validation** - Built-in CV support

---

## References

1. Dempster-Shafer Theory: Shafer, G. (1976). A Mathematical Theory of Evidence.
2. DSGD Library: https://github.com/Sergio-P/DSGD.git
3. Silhouette Score: Rousseeuw, P. J. (1987). Silhouettes: A graphical aid to the interpretation and validation of cluster analysis.

---

## License

See repository LICENSE file.

---

## Contact

For issues, questions, or contributions:
- Repository: https://github.com/ricardo-valdivia/CDSGD
- Issue Tracker: https://github.com/ricardo-valdivia/CDSGD/issues

---

*Documentation generated: 2026-03-09*
