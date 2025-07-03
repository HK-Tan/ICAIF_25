"""
GPU-Accelerated Sliding Window Clustering and Centroid Calculation

This script performs two main tasks:
1.  Sliding Window Clustering: For each window in a time-series of asset returns,
    it computes a correlation matrix and assigns a cluster label to each asset.
    This produces a time-series of cluster assignments. The expensive eigenvector
    calculation is accelerated using the elezar/gpu-arpack library via ctypes.
2.  Centroid Return Calculation: Using the cluster assignments from the most recent
    window, it calculates a representative time-series for each cluster.

The entire process is accelerated on a CUDA-enabled NVIDIA GPU using CuPy and cuML.

Requirements:
- A CUDA-enabled NVIDIA GPU
- A compiled shared library from https://github.com/elezar/gpu-arpack
- CuPy: (pip install cupy-cudaXXX, where XXX matches your CUDA version)
- cuML: (conda install -c rapidsai -c conda-forge -c nvidia cuml)
- pandas, numpy, tqdm, pyarrow
"""
import pandas as pd
import numpy as np
import cupy as cp
import cupyx
import ctypes as ct
from numpy.lib.stride_tricks import as_strided
from tqdm import tqdm
import os
import shutil
from cuml.cluster import KMeans as cuMLKMeans


# --- CONFIGURATION PARAMETERS ---

# GPU device to use
GPU_DEVICE_ID = 0

# BATCH_SIZE controls how many sliding windows are processed on the GPU at once.
BATCH_SIZE = 4

# Sigma for Gaussian weighting in centroid calculation.
SIGMA_FOR_WEIGHTS = 1.0

# --- IMPORTANT: Set this path to your compiled gpu-arpack shared library ---
# For example: 'gpu-arpack/lib/libsingle_dense_gpu.so'
# This path MUST be correct for the script to run.
GPU_ARPACK_LIB_PATH = 'libsingle_dense_gpu.so'


# ==============================================================================
# PART 1: CORE BATCHED ALGORITHMS (Implemented in CuPy + cuML + gpu-arpack)
# ==============================================================================

def eigh_generalized_gpu_arpack(A, B, k):
    """
    Solves the generalized eigenvalue problem A*v = w*B*v for k smallest eigenvalues
    using the elezar/gpu-arpack library via ctypes.

    Args:
        A (cp.ndarray): Symmetric matrix A on the GPU.
        B (cp.ndarray): Symmetric positive-definite matrix B on the GPU.
        k (int): The number of smallest eigenvalues to compute.

    Returns:
        tuple[cp.ndarray, cp.ndarray]: A tuple containing eigenvalues and eigenvectors.
    """
    if not os.path.exists(GPU_ARPACK_LIB_PATH):
        raise FileNotFoundError(
            f"gpu-arpack library not found at: {GPU_ARPACK_LIB_PATH}\n"
            "Please compile the library from https://github.com/elezar/gpu-arpack "
            "and set the GPU_ARPACK_LIB_PATH variable correctly."
        )

    lib = ct.CDLL(GPU_ARPACK_LIB_PATH)
    # Assuming the function for single-precision, symmetric generalized eigenproblem is 'dense_ssygvx'
    # You may need to change this name based on the exact library version you compiled.
    func = lib['dense_ssygvx']
    func.restype = None
    func.argtypes = [
        ct.c_int,      # N (matrix size)
        ct.c_void_p,   # A (device pointer)
        ct.c_int,      # LDA
        ct.c_void_p,   # B (device pointer)
        ct.c_int,      # LDB
        ct.c_char,     # 'I' for range, 'A' for all, 'V' for values
        ct.c_float,    # vl
        ct.c_float,    # vu
        ct.c_int,      # il (index lower bound)
        ct.c_int,      # iu (index upper bound)
        ct.c_int,      # M (number of eigenvalues found) - output
        ct.c_int,      # NEV (number of eigenvalues to find)
        ct.c_void_p,   # W (eigenvalues) - output
        ct.c_void_p,   # Z (eigenvectors) - output
        ct.c_int,      # LDZ
    ]

    N = A.shape[0]
    # ARPACK/LAPACK routines expect column-major (Fortran) arrays.
    A_fortran = cp.asfortranarray(A, dtype=cp.float32)
    B_fortran = cp.asfortranarray(B, dtype=cp.float32)

    # Allocate output arrays on the GPU
    eigenvalues = cp.zeros(k, dtype=cp.float32)
    eigenvectors = cp.zeros((N, k), dtype=cp.float32, order='F')

    # Get device pointers
    ptr_A = A_fortran.data.ptr
    ptr_B = B_fortran.data.ptr
    ptr_W = eigenvalues.data.ptr
    ptr_Z = eigenvectors.data.ptr

    m_found = ct.c_int(0)

    # Call the C function. We ask for eigenvalues by index range (1 to k).
    func(N, ct.c_void_p(ptr_A), N,
         ct.c_void_p(ptr_B), N,
         b'I', 0.0, 0.0, 1, k,
         ct.byref(m_found), k,
         ct.c_void_p(ptr_W),
         ct.c_void_p(ptr_Z), N)

    return eigenvalues, eigenvectors


def diag_embed_batched(x: cp.ndarray) -> cp.ndarray:
    """OPTIMIZED: Creates a batch of diagonal matrices using broadcasting."""
    B, N = x.shape
    return cp.eye(N, dtype=x.dtype) * x[..., None]

def sqrtinvdiag_batched(M: cp.ndarray) -> cp.ndarray:
    """OPTIMIZED: Calculates the inverse square root of the diagonal of a batch of matrices."""
    diag = cp.diagonal(M, axis1=-2, axis2=-1)
    inv_sqrt_diag = 1.0 / cp.sqrt(cp.maximum(diag, 1e-12))
    return diag_embed_batched(inv_sqrt_diag)

def SPONGE_sym_batched(p, n, k, tau=1.0, Niter_kmeans=50):
    """
    Batched version of SPONGE_sym, using gpu-arpack (via ctypes) for eigensolvers
    and cuML KMeans for clustering.
    """
    B, N, _ = p.shape
    eye = cp.eye(N, dtype=p.dtype).reshape(1, N, N)

    D_p = diag_embed_batched(p.sum(axis=-1))
    D_n = diag_embed_batched(n.sum(axis=-1))

    d_p_inv_sqrt = sqrtinvdiag_batched(D_p)
    d_n_inv_sqrt = sqrtinvdiag_batched(D_n)

    L_p = eye - d_p_inv_sqrt @ p @ d_p_inv_sqrt
    L_n = eye - d_n_inv_sqrt @ n @ d_n_inv_sqrt

    matrix1_batch = L_n + tau * eye
    matrix2_batch = L_p + tau * eye

    all_labels = []

    # The C library does not support batches, so we loop through them.
    for i in range(B):
        # Solve the generalized eigenproblem for the i-th item in the batch
        w_item, v_item = eigh_generalized_gpu_arpack(matrix1_batch[i], matrix2_batch[i], k)

        epsilon = 1e-12
        w_reshaped = w_item.reshape(1, k)
        eigenvectors_to_cluster = v_item / (w_reshaped + epsilon)

        kmeans = cuMLKMeans(n_clusters=k, n_init=1, max_iter=Niter_kmeans, random_state=i) # Use loop index for seed
        labels_item = kmeans.fit_predict(eigenvectors_to_cluster)
        all_labels.append(labels_item)

    cl = cp.stack(all_labels)
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

# Setup GPU
cp.cuda.runtime.setDevice(GPU_DEVICE_ID)

# --- PARAMETER SWEEP CONFIGURATION ---
LOOKBACK_WINDOWS_TO_RUN = [60, 120, 252]
N_CLUSTERS_TO_RUN = [5, 8, 12, 16]
LOCAL_OUTPUT_DIR = "clustering_outputs_local"
os.makedirs(LOCAL_OUTPUT_DIR, exist_ok=True)
if 'GDRIVE_OUTPUT_PATH' not in locals():
    GDRIVE_OUTPUT_PATH = LOCAL_OUTPUT_DIR

# --- DATA LOADING ---
print("\n--> Loading data...")
# Dummy DataFrame for demonstration. Replace with your actual data.
num_samples = 500
num_assets = 100
data = np.random.randn(num_samples, num_assets) * 0.01
dates = pd.to_datetime(pd.date_range(start='2020-01-01', periods=num_samples, freq='B'))
df = pd.DataFrame(data, index=dates, columns=[f'Asset_{i}' for i in range(num_assets)])
df = df.astype(np.float32)
print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} assets.")

if __name__ == '__main__':

    count = 0
    # --- MAIN PROCESSING LOOP ---
    for lookback in LOOKBACK_WINDOWS_TO_RUN:
        for n_clusters in N_CLUSTERS_TO_RUN:
            if count <= 3:
                count += 1
                continue
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

                # Part 3: Copy results
                if GDRIVE_OUTPUT_PATH != LOCAL_OUTPUT_DIR:
                    print("\n--> Copying results to target directory...")
                    shutil.copy(labels_fname, GDRIVE_OUTPUT_PATH)
                    shutil.copy(centroids_fname, GDRIVE_OUTPUT_PATH)
                    print(f"✅ Successfully copied files to {GDRIVE_OUTPUT_PATH}")

            except (ValueError, FileNotFoundError) as e:
                print(f"\nSKIPPING RUN (Lookback={lookback}, Clusters={n_clusters}): {e}")
            except Exception as e:
                print(f"\n❌ UNEXPECTED ERROR (Lookback={lookback}, Clusters={n_clusters}): {e}")
                import traceback
                traceback.print_exc()

    print(f"\n\n{'='*60}")
    print("✅ ALL PARAMETER SWEEP RUNS ARE COMPLETE.")
    print(f"Check '{GDRIVE_OUTPUT_PATH}' for all output files.")
    print(f"{'='*60}")