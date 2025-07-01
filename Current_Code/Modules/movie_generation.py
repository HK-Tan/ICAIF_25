import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
import numba
import multiprocessing
import os
from tqdm.auto import tqdm

# --- Setup Paths and Parameters ---
# It's good practice to set these paths at the top
# and handle directory changes carefully.
# Let's assume the data is in a subfolder relative to the script.
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Gets the directory of the script
# DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'Data')
# os.chdir(DATA_DIR)
# For now, we'll keep your original setup.
os.chdir('C:/Users/james/ICAIF_25')
from signet.cluster import Cluster
os.chdir('C:/Users/james/ICAIF_25/Current_Code/Data')

dpi = 64
plt.rcParams['figure.dpi'] = dpi

@numba.njit(parallel=True, fastmath=True)
def fast_corrcoef_numba(A):
    """
    Fully-jitted and parallelized implementation of corrcoef with nan_to_num logic.
    Assumes rowvar=False.
    """
    n_rows, n_cols = A.shape
    means = np.empty(n_cols, dtype=A.dtype)
    for j in numba.prange(n_cols):
        means[j] = A[:, j].mean()
    A_demeaned = A - means
    cov_matrix = A_demeaned.T @ A_demeaned / (n_rows - 1)
    variances = np.diag(cov_matrix)
    stds = np.sqrt(variances)
    corr_matrix = np.empty_like(cov_matrix)
    for i in numba.prange(n_cols):
        for j in range(n_cols):
            denominator = stds[i] * stds[j]
            if denominator == 0:
                corr_matrix[i, j] = 0.0
            else:
                corr_matrix[i, j] = cov_matrix[i, j] / denominator
    return corr_matrix

@numba.njit
def set_upper_triangle_nan(arr):
    rows, cols = arr.shape
    for i in range(rows):
        for j in range(i, cols):
            arr[i, j] = np.nan
    return arr

@numba.njit(fastmath=True, parallel=True)
def reorder_matrix_numba(matrix, order):
    n = matrix.shape[0]
    reordered_matrix = np.empty_like(matrix)
    for i in numba.prange(n):
        for j in range(n):
            reordered_matrix[i, j] = matrix[order[i], order[j]]
    return reordered_matrix

@numba.njit(parallel=True)
def get_sparse_pos_neg_parts(matrix):
    rows, cols = matrix.shape
    n_elements = matrix.size
    n_pos = 0
    n_neg = 0
    for i in numba.prange(n_elements):
        row = i // cols
        col = i % cols
        val = matrix[row, col]
        if val > 0:
            n_pos += 1
        elif val < 0:
            n_neg += 1
    pos_data = np.empty(n_pos, dtype=matrix.dtype)
    pos_rows = np.empty(n_pos, dtype=np.int32)
    pos_cols = np.empty(n_pos, dtype=np.int32)
    neg_data = np.empty(n_neg, dtype=matrix.dtype)
    neg_rows = np.empty(n_neg, dtype=np.int32)
    neg_cols = np.empty(n_neg, dtype=np.int32)
    pos_idx = 0
    neg_idx = 0
    for r in range(rows):
        for c in range(cols):
            val = matrix[r, c]
            if val > 0:
                pos_data[pos_idx] = val
                pos_rows[pos_idx] = r
                pos_cols[pos_idx] = c
                pos_idx += 1
            elif val < 0:
                neg_data[neg_idx] = -val
                neg_rows[neg_idx] = r
                neg_cols[neg_idx] = c
                neg_idx += 1
    return (pos_data, pos_rows, pos_cols), (neg_data, neg_rows, neg_cols)

# --- WORKER FUNCTION ---
def create_animation_chunk(asset_returns_df,
        lookback_period,
        n_clusters_to_form,
        start_frame,
        num_frames_in_chunk,
        chunk_index,
        output_prefix):
    """
    Worker function to render a single chunk of the animation.
    """
    asset_returns_np = asset_returns_df.values.astype(np.float32)
    asset_returns_index = asset_returns_df.index
    num_assets = asset_returns_df.shape[1]

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout(pad=3.0)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    initial_frame_index = start_frame
    initial_window = asset_returns_np[initial_frame_index : initial_frame_index + lookback_period]
    initial_corr = fast_corrcoef_numba(initial_window)
    (pos_data, pos_rows, pos_cols), (neg_data, neg_rows, neg_cols) = get_sparse_pos_neg_parts(initial_corr)
    pos_corr = sparse.csc_matrix((pos_data, (pos_rows, pos_cols)), shape=initial_corr.shape)
    neg_corr = sparse.csc_matrix((neg_data, (neg_rows, neg_cols)), shape=initial_corr.shape)
    cluster_obj = Cluster((pos_corr, neg_corr))
    labels = cluster_obj.SPONGE_sym(min(n_clusters_to_form, num_assets))
    order_indices = np.argsort(labels)
    reordered_matrix = reorder_matrix_numba(initial_corr, order_indices)
    initial_matrix_to_plot = set_upper_triangle_nan(reordered_matrix)

    im = ax.imshow(
        initial_matrix_to_plot, cmap='bwr', vmin=-1, vmax=1,
        interpolation='none', animated=True
    )

    start_date = asset_returns_index[initial_frame_index].strftime('%Y-%m-%d')
    end_date = asset_returns_index[initial_frame_index + lookback_period - 1].strftime('%Y-%m-%d')
    title = ax.set_title(
        f"Clustered Asset Correlation\n{start_date} to {end_date}", fontsize=16
    )

    def update(frame_num):
      start_index = frame_num
      end_index = frame_num + lookback_period
      window_np = asset_returns_np[start_index:end_index]
      correlation_matrix_np = fast_corrcoef_numba(window_np)
      (pos_data, pos_rows, pos_cols), (neg_data, neg_rows, neg_cols) = get_sparse_pos_neg_parts(correlation_matrix_np)
      pos_corr_numba = sparse.csc_matrix((pos_data, (pos_rows, pos_cols)), shape=correlation_matrix_np.shape)
      neg_corr_numba = sparse.csc_matrix((neg_data, (neg_rows, neg_cols)), shape=correlation_matrix_np.shape)
      signet_data = (pos_corr_numba, neg_corr_numba)
      effective_n_clusters = min(n_clusters_to_form, num_assets)
      cluster_obj = Cluster(signet_data)
      labels = cluster_obj.SPONGE_sym(effective_n_clusters)
      order_indices = np.argsort(labels)
      reordered_matrix_np = reorder_matrix_numba(correlation_matrix_np, order_indices)
      matrix_to_plot = set_upper_triangle_nan(reordered_matrix_np)
      im.set_data(matrix_to_plot)
      start_date = asset_returns_index[start_index].strftime('%Y-%m-%d')
      end_date = asset_returns_index[end_index-1].strftime('%Y-%m-%d')
      title.set_text(f"Clustered Asset Correlation\n{start_date} to {end_date}")
      return im, title

    frame_iterator = range(start_frame, start_frame + num_frames_in_chunk)
    ani = FuncAnimation(fig, update, frames=frame_iterator, blit=True, repeat=False)
    output_filename = f"{output_prefix}_part_{chunk_index}.mp4"

    with tqdm(total=num_frames_in_chunk, desc=f"Chunk {chunk_index}", position=chunk_index + 1, leave=False) as pbar:
        def progress_callback(current_frame, total_frames):
            pbar.update(1)
        ani.save(
            output_filename, writer='ffmpeg', dpi=dpi, fps=15,
            progress_callback=progress_callback,
            extra_args=['-vcodec', 'h264_qsv', '-preset', 'veryfast']
        )
    plt.close(fig)
    return output_filename

# *** FIX: MOVED THIS FUNCTION TO THE TOP LEVEL ***
def worker_wrapper(args):
    """
    A simple top-level wrapper that unpacks arguments for use with
    pool.imap_unordered, which requires a single-argument function.
    This function must be at the top level to be "picklable".
    """
    return create_animation_chunk(*args)

# --- ORCHESTRATOR FUNCTION ---
def create_parallel_animations(
    asset_returns_df: pd.DataFrame,
    lookback_period: int,
    n_clusters_to_form: int,
    num_threads: int,
    output_prefix: str = "correlation_movie_chunk"
):
    """
    Splits the animation rendering task across multiple processes and
    displays a high-level progress bar tracking the completion of chunks.
    """
    total_frames = len(asset_returns_df) - lookback_period
    if total_frames <= 0:
        print("Not enough data for the given lookback period.")
        return

    frames_per_thread = total_frames // num_threads
    tasks = []
    current_start_frame = 0
    for i in range(num_threads):
        start = current_start_frame
        if i == num_threads - 1:
            end = total_frames
        else:
            end = start + frames_per_thread
        num_frames_in_chunk = end - start
        if num_frames_in_chunk > 0:
            task_args = (asset_returns_df, lookback_period, n_clusters_to_form, start, num_frames_in_chunk, i, output_prefix)
            tasks.append(task_args)
        current_start_frame = end

    print(f"Total frames: {total_frames}. Splitting into {len(tasks)} chunks to be processed by {num_threads} threads.")
    print("Starting parallel rendering. This may take a while...")

    with multiprocessing.Pool(processes=num_threads, maxtasksperchild=1) as pool:
        # Use the top-level worker_wrapper, which is now picklable.
        results_iterator = pool.imap_unordered(worker_wrapper, tasks)
        results = list(tqdm(results_iterator, total=len(tasks), desc="Overall Progress", position=0))

    print("\nAll animation chunks have been created:")
    for filename in results:
        print(f"- {filename}")

    file_list_path = "file_list.txt"
    with open(file_list_path, "w") as f:
        for filename in sorted(results):
            f.write(f"file '{filename}'\n")

    print(f"\nGenerated '{file_list_path}' for video stitching.")
    print("\n--- TO STITCH THE VIDEOS ---")
    print("Run the following command in your terminal:")
    print(f"ffmpeg -f concat -safe 0 -i {file_list_path} -c copy final_correlation_movie.mp4")

# --- Main Execution Block ---
if __name__ == '__main__':
    multiprocessing.freeze_support()
    lookback = 252
    num_clusters = 5
    num_threads_to_use = max(1, os.cpu_count() - 2)

    df = pd.read_parquet('log_returns_by_ticker.parquet')

    create_parallel_animations(
        asset_returns_df=df,
        lookback_period=lookback,
        n_clusters_to_form=num_clusters,
        num_threads=num_threads_to_use,
    )