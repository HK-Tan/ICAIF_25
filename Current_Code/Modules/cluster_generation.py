"""
GPU-Accelerated Sliding Window Clustering and Centroid Calculation

This script performs two main tasks:
1.  Sliding Window Clustering: For each window in a time-series of asset returns,
    it computes a correlation matrix and assigns a cluster label to each asset.
    This produces a time-series of cluster assignments. The expensive eigenvector
    calculation is accelerated using PyTorch's LOBPCG algorithm.
2.  Centroid Return Calculation: Using the cluster assignments from the most recent
    window, it calculates a representative time-series for each cluster.

The entire process is accelerated on a CUDA-enabled NVIDIA GPU using CuPy and PyTorch.

Requirements:
- A CUDA-enabled NVIDIA GPU
- CuPy: (pip install cupy-cudaXXX, where XXX matches your CUDA version)
- PyTorch: (pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX)
- pandas, numpy, tqdm, pyarrow
"""
import pandas as pd
import numpy as np
import cupy as cp
import cupyx
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm
import os
import shutil
import torch


# --- CONFIGURATION PARAMETERS ---

# GPU device to use
GPU_DEVICE_ID = 0

# BATCH_SIZE controls how many sliding windows are processed on the GPU at once.
BATCH_SIZE = 5

# Sigma for Gaussian weighting in centroid calculation.
SIGMA_FOR_WEIGHTS = 1.0

# Use PyTorch's LOBPCG for faster eigenvector calculation.
# Requires PyTorch to be installed. Will fall back to CuPy eigh if False or unavailable.
USE_LOBPCG = True

# ==============================================================================
# PART 1: CORE BATCHED ALGORITHMS (Implemented in CuPy + PyTorch)
# ==============================================================================

def KMeans_batched(x, K, Niter=50):
    """
    Implements Lloyd's algorithm with K-Means++ initialization for the Euclidean metric.
    This version is fully vectorized to run on a batch of CuPy arrays on the GPU.
    """
    B, N, D = x.shape  # Batch size, number of samples, dimension

    # K-Means++ initialization
    centroids = cp.empty((B, K, D), dtype=x.dtype)
    first_centroid_idx = cp.random.randint(0, N, size=(B,))
    centroids[:, 0, :] = x[cp.arange(B), first_centroid_idx, :]

    for k_idx in range(1, K):
        x_expanded = x.reshape(B, N, 1, D)
        centroids_subset = centroids[:, :k_idx, :].reshape(B, 1, k_idx, D)
        dist_sq = cp.sum((x_expanded - centroids_subset) ** 2, axis=3)
        min_dist_sq = dist_sq.min(axis=2)
        probs = min_dist_sq / (min_dist_sq.sum(axis=1, keepdims=True) + 1e-12)
        cum_probs = cp.cumsum(probs, axis=1)
        r = cp.random.rand(B, 1)
        next_centroid_idx = (r < cum_probs).argmax(axis=1)
        centroids[:, k_idx, :] = x[cp.arange(B), next_centroid_idx, :]

    c = centroids
    for i in range(Niter):
        c_old = c.copy()
        x_bni = x.reshape(B, N, 1, D)
        c_bkj = c.reshape(B, 1, K, D)
        D_b_nk = cp.sum((x_bni - c_bkj) ** 2, axis=3)
        cl = D_b_nk.argmin(axis=2)
        cl_one_hot = cp.eye(K, dtype=x.dtype)[cl]
        Ncl = cl_one_hot.sum(axis=1)
        c_sum = cp.einsum('bnk,bnd->bkd', cl_one_hot, x)
        c = c_sum / (Ncl.reshape(B, K, 1) + 1e-8)
        empty_clusters_mask = (Ncl == 0)
        c[empty_clusters_mask] = c_old[empty_clusters_mask]

        if cp.allclose(c, c_old, atol=1e-6, rtol=0):
            break

    return cl, c

def diag_embed_batched(x: cp.ndarray) -> cp.ndarray:
    """OPTIMIZED: Creates a batch of diagonal matrices using broadcasting."""
    B, N = x.shape
    # Broadcasting x[..., None] (B, N, 1) with an identity matrix (N, N) is faster
    # than creating a zero matrix and filling the diagonal.
    return cp.eye(N, dtype=x.dtype) * x[..., None]

def sqrtinvdiag_batched(M: cp.ndarray) -> cp.ndarray:
    """OPTIMIZED: Calculates the inverse square root of the diagonal of a batch of matrices."""
    diag = cp.diagonal(M, axis1=-2, axis2=-1)
    inv_sqrt_diag = 1.0 / cp.sqrt(cp.maximum(diag, 1e-12))
    return diag_embed_batched(inv_sqrt_diag)

def SPONGE_sym_batched(p, n, k, tau=1.0, Niter_kmeans=50):
    """Batched version of SPONGE_sym, with optional PyTorch LOBPCG acceleration."""
    B, N, _ = p.shape
    eye = cp.eye(N, dtype=p.dtype).reshape(1, N, N)

    D_p = diag_embed_batched(p.sum(axis=-1))
    D_n = diag_embed_batched(n.sum(axis=-1))

    d_p_inv_sqrt = sqrtinvdiag_batched(D_p)
    d_n_inv_sqrt = sqrtinvdiag_batched(D_n)

    L_p = eye - d_p_inv_sqrt @ p @ d_p_inv_sqrt
    L_n = eye - d_n_inv_sqrt @ n @ d_n_inv_sqrt

    matrix1 = L_n + tau * eye
    matrix2 = L_p + tau * eye

    # --- Eigen-decomposition using Generalized Eigensolver ---
    # Convert to PyTorch tensors for lobpcg (zero-copy)
    # Note: lobpcg requires inputs to be on the GPU for GPU execution
    matrix1_torch = torch.as_tensor(matrix1, device=f'cuda:{GPU_DEVICE_ID}')
    matrix2_torch = torch.as_tensor(matrix2, device=f'cuda:{GPU_DEVICE_ID}')

    # Solves the generalized eigenproblem: matrix1 * v = w * matrix2 * v
    w_torch, v_torch = torch.lobpcg(matrix1_torch, k=k, B=matrix2_torch, largest=False)

    # Convert back to CuPy arrays (zero-copy)
    v = cp.asarray(v_torch) # Shape: (B, N, k)
    w = cp.asarray(w_torch) # Shape: (B, k)

    # --- FIX: Scale eigenvectors before KMeans ---
    # Reshape 'w' to (B, 1, k) to enable broadcasting for the division.
    # Add a small epsilon for numerical stability.
    epsilon = 1e-12
    w_reshaped = w.reshape(B, 1, k)

    # 'v / w' is now broadcast correctly: (B, N, k) / (B, 1, k)
    eigenvectors_to_cluster = v / (w_reshaped + epsilon)

    # Perform KMeans on the scaled eigenvectors
    cl, _ = KMeans_batched(eigenvectors_to_cluster, k, Niter=Niter_kmeans)

    return cl


# ==============================================================================
# PART 2: USER-FACING WRAPPER FUNCTIONS
# ==============================================================================

def gpu_batched_clustering_sponge(corr_matrices_gpu, n_clusters):
    p = cp.maximum(corr_matrices_gpu, 0)
    n = cp.maximum(-corr_matrices_gpu, 0)
    labels_cupy = SPONGE_sym_batched(p, n, k=n_clusters)
    return labels_cupy.astype(cp.int32)

def fast_corrcoef_gpu_batched(A):
    n_samples = A.shape[1]
    means = cp.mean(A, axis=1, keepdims=True)
    A_demeaned = A - means
    cov_matrix = cp.einsum('bij,bik->bjk', A_demeaned, A_demeaned) / (n_samples - 1)
    variances = cp.einsum('bii->bi', cov_matrix)
    stds = cp.sqrt(variances)
    denominator = cp.einsum('bi,bj->bij', stds, stds)
    cp.place(denominator, denominator == 0, 1)
    corr_matrix = cov_matrix / denominator
    return corr_matrix

def calculate_centroid_returns_gpu(asset_returns_gpu, labels_gpu, n_clusters, sigma):
    print("Calculating centroid returns on GPU...")
    T, N = asset_returns_gpu.shape
    epsilon = 1e-12
    one_hot_labels = cp.eye(n_clusters, dtype=asset_returns_gpu.dtype)[labels_gpu]
    cluster_return_sums = asset_returns_gpu @ one_hot_labels
    cluster_counts = one_hot_labels.sum(axis=0)
    centroids_gpu = cluster_return_sums / (cluster_counts + epsilon)
    aligned_centroids_gpu = centroids_gpu[:, labels_gpu]
    squared_distances = cp.sum((asset_returns_gpu - aligned_centroids_gpu)**2, axis=0)
    exponent_vals = -squared_distances / (2 * (sigma**2 + epsilon))
    max_exp_per_cluster = cp.full(n_clusters, -cp.inf, dtype=exponent_vals.dtype)
    cupyx.scatter_max(max_exp_per_cluster, labels_gpu, exponent_vals)
    stable_exponent_vals = exponent_vals - max_exp_per_cluster[labels_gpu]
    unnormalized_weights = cp.exp(stable_exponent_vals)
    sum_weights_per_cluster = cp.zeros(n_clusters, dtype=unnormalized_weights.dtype)
    cupyx.scatter_add(sum_weights_per_cluster, labels_gpu, unnormalized_weights)
    normalized_weights = unnormalized_weights / (sum_weights_per_cluster[labels_gpu] + epsilon)
    weighted_simple_returns = cp.expm1(asset_returns_gpu) * normalized_weights
    cluster_simple_returns = weighted_simple_returns @ one_hot_labels
    cluster_log_returns = cp.log1p(cluster_simple_returns)
    return cluster_log_returns

def process_windows_on_gpu(asset_returns_df, lookback_period, n_clusters, batch_size):
    print("Preparing data for GPU...")
    all_returns_np = np.ascontiguousarray(asset_returns_df.values, dtype=np.float32)
    num_timesteps, num_assets = all_returns_np.shape
    num_windows = num_timesteps - lookback_period + 1
    if num_windows <= 0:
        raise ValueError(f"lookback_period ({lookback_period}) is larger than the number of timesteps ({num_timesteps}).")
    shape = (num_windows, lookback_period, num_assets)
    strides = (all_returns_np.strides[0], all_returns_np.strides[0], all_returns_np.strides[1])
    all_windows_np = as_strided(all_returns_np, shape=shape, strides=strides)
    print(f"Created {num_windows} sliding windows of size {lookback_period}.")
    print(f"Processing in batches of {batch_size} on GPU {cp.cuda.runtime.getDevice()}...")
    all_labels_list = []
    for i in tqdm(range(0, num_windows, batch_size), desc="GPU Batches (Sliding Window)"):
        batch_end = min(i + batch_size, num_windows)
        windows_np_batch = all_windows_np[i:batch_end]
        windows_gpu = cp.asarray(windows_np_batch)
        corr_matrices_gpu = fast_corrcoef_gpu_batched(windows_gpu)
        labels_gpu = gpu_batched_clustering_sponge(corr_matrices_gpu, n_clusters)
        labels_np = cp.asnumpy(labels_gpu)
        all_labels_list.append(labels_np)
    print("GPU processing complete. Assembling final DataFrame...")
    final_labels_np = np.vstack(all_labels_list)
    window_end_dates = asset_returns_df.index[lookback_period - 1:]
    labels_df = pd.DataFrame(final_labels_np, index=window_end_dates, columns=asset_returns_df.columns)
    return labels_df


# --- Main Execution Block ---


# Setup GPU and libraries
cp.cuda.runtime.setDevice(GPU_DEVICE_ID)
torch.cuda.set_device(GPU_DEVICE_ID)

# --- PARAMETER SWEEP CONFIGURATION ---
LOOKBACK_WINDOWS_TO_RUN = [252]
N_CLUSTERS_TO_RUN = [5, 8, 12, 16]
# Local temporary output directory in Colab
LOCAL_OUTPUT_DIR = "clustering_outputs_local"
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)

# --- DATA LOADING ---
print("\n--> Loading data...")
# Use the path to the data downloaded in the previous cell
df = pd.read_parquet('log_returns_by_ticker.parquet')
df = df.astype(np.float32)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} assets.")

count = 0
# --- MAIN PROCESSING LOOP ---
for lookback in LOOKBACK_WINDOWS_TO_RUN:
    for n_clusters in N_CLUSTERS_TO_RUN:
        print(f"\n{'='*60}")
        print(f"  STARTING RUN: Lookback = {lookback}, Clusters = {n_clusters}")
        print(f"{'='*60}\n")
        try:
            # Part 1: Sliding Window Clustering
            labels_df = process_windows_on_gpu(df, lookback, n_clusters, BATCH_SIZE)
            labels_fname = os.path.join(LOCAL_OUTPUT_DIR, f"cluster_labels_{n_clusters}c_{lookback}d.parquet")
            labels_df.to_parquet(labels_fname, engine='pyarrow')
            print(f"\nGenerated local file: {labels_fname}")

            # Part 2: Centroid Return Calculation
            final_labels = cp.asarray(labels_df.iloc[-1].values.astype(np.int32))
            asset_returns_gpu = cp.asarray(df.values)
            centroid_returns = calculate_centroid_returns_gpu(asset_returns_gpu, final_labels, n_clusters, SIGMA_FOR_WEIGHTS)
            centroids_df = pd.DataFrame(cp.asnumpy(centroid_returns), index=df.index, columns=[f'Cluster_{i}' for i in range(n_clusters)])
            centroids_fname = os.path.join(LOCAL_OUTPUT_DIR, f"centroid_returns_{n_clusters}c_{lookback}d.parquet")
            centroids_df.to_parquet(centroids_fname, engine='pyarrow')
            print(f"Generated local file: {centroids_fname}")

            # Part 3: Copy results to Google Drive
            print("\n--> Copying results to Google Drive...")
            shutil.copy(labels_fname, GDRIVE_OUTPUT_PATH)
            shutil.copy(centroids_fname, GDRIVE_OUTPUT_PATH)
            print(f"✅ Successfully copied files to {GDRIVE_OUTPUT_PATH}")

        except ValueError as e:
            print(f"\nSKIPPING RUN (Lookback={lookback}, Clusters={n_clusters}): {e}")
        except Exception as e:
            print(f"\n❌ UNEXPECTED ERROR (Lookback={lookback}, Clusters={n_clusters}): {e}")
            import traceback
            traceback.print_exc()

print(f"\n\n{'='*60}")
print("✅ ALL PARAMETER SWEEP RUNS ARE COMPLETE.")
print(f"Check '{GDRIVE_OUTPUT_PATH}' in your Google Drive for all output files.")
print(f"{'='*60}")