import pandas as pd
import pandas as pd
import numpy as np
import multiprocessing
import os
import shutil
import pickle
import numba
from scipy import sparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- FIX FOR MKL/KMEANS MEMORY LEAK ON WINDOWS ---
# This must be set BEFORE any library that uses MKL (like numpy or sklearn).
os.environ['OMP_NUM_THREADS'] = '1'

# --- CONFIGURATION PARAMETER ---
# This controls the trade-off between RAM usage and CPU utilization.
# Higher value = higher CPU usage and higher RAM per worker.
MICRO_BATCH_SIZE = 10

# --- Setup Paths and Parameters ---
try:
    import sys
    sys.path.append('C:/Users/james/ICAIF_25')
    from signet.cluster import Cluster
except (ModuleNotFoundError, ImportError):
    print("CRITICAL: 'signet' library not found. Please install it.")
    sys.exit(1)

try:
    # Set this to the directory containing your data file.
    os.chdir('C:/Users/james/ICAIF_25/Current_Code/Data')
except FileNotFoundError:
    print("Warning: Could not change to the data directory. Ensure paths are correct.")

# --- Numba Optimized Functions (Unchanged) ---
@numba.njit(parallel=True, fastmath=True)
def fast_corrcoef_numba(A):
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
    n_pos = 0; n_neg = 0
    for i in numba.prange(n_elements):
        val = matrix.ravel()[i]
        if val > 0: n_pos += 1
        elif val < 0: n_neg += 1
    pos_data = np.empty(n_pos, dtype=matrix.dtype); pos_rows = np.empty(n_pos, dtype=np.int32); pos_cols = np.empty(n_pos, dtype=np.int32)
    neg_data = np.empty(n_neg, dtype=matrix.dtype); neg_rows = np.empty(n_neg, dtype=np.int32); neg_cols = np.empty(n_neg, dtype=np.int32)
    pos_idx = 0; neg_idx = 0
    for r in range(rows):
        for c in range(cols):
            val = matrix[r, c]
            if val > 0:
                pos_data[pos_idx] = val; pos_rows[pos_idx] = r; pos_cols[pos_idx] = c
                pos_idx += 1
            elif val < 0:
                neg_data[neg_idx] = -val; neg_rows[neg_idx] = r; neg_cols[neg_idx] = c
                neg_idx += 1
    return (pos_data, pos_rows, pos_cols), (neg_data, neg_rows, neg_cols)

# --- WORKER FUNCTION (Unchanged) ---
def process_and_stream_chunk(
    asset_returns_chunk_np, lookback_period, n_clusters_to_form,
    num_frames_in_chunk, chunk_index, output_dir, micro_batch_size):
    chunk_filename = os.path.join(output_dir, f"chunk_{chunk_index:03d}.bin")
    asset_dim = asset_returns_chunk_np.shape[1]
    matrix_batch_preallocated = np.empty(
        (micro_batch_size, asset_dim, asset_dim), dtype=np.float32)
    batch_idx = 0
    with open(chunk_filename, 'wb') as f_out:
        for i in range(num_frames_in_chunk):
            window_np = asset_returns_chunk_np[i : i + lookback_period]
            correlation_matrix_np = fast_corrcoef_numba(window_np)
            (pos_data, pos_rows, pos_cols), (neg_data, neg_rows, neg_cols) = get_sparse_pos_neg_parts(correlation_matrix_np)
            pos_corr_numba = sparse.csc_matrix((pos_data, (pos_rows, pos_cols)), shape=correlation_matrix_np.shape)
            neg_corr_numba = sparse.csc_matrix((neg_data, (neg_rows, neg_cols)), shape=correlation_matrix_np.shape)
            signet_data = (pos_corr_numba, neg_corr_numba)
            effective_n_clusters = min(n_clusters_to_form, correlation_matrix_np.shape[0])
            cluster_obj = Cluster(signet_data)
            labels = cluster_obj.SPONGE_sym(effective_n_clusters)
            order_indices = np.argsort(labels)
            reordered_matrix_np = reorder_matrix_numba(correlation_matrix_np, order_indices)
            matrix_to_plot = set_upper_triangle_nan(reordered_matrix_np).astype(np.float32)
            matrix_batch_preallocated[batch_idx] = matrix_to_plot
            batch_idx += 1
            if batch_idx == micro_batch_size:
                f_out.write(matrix_batch_preallocated.tobytes())
                batch_idx = 0
        if batch_idx > 0:
            f_out.write(matrix_batch_preallocated[:batch_idx].tobytes())

# --- PARALLEL GENERATION (Unchanged) ---
def parallel_generate_matrices(
    asset_returns_df: pd.DataFrame, lookback_period: int, n_clusters_to_form: int,
    num_threads: int, output_dir: str):
    total_frames = len(asset_returns_df) - lookback_period
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"Created temp directory for matrix chunks: {output_dir}")
    all_returns_np = asset_returns_df.values.astype(np.float32)
    original_index = asset_returns_df.index
    asset_dim = asset_returns_df.shape[1]
    metadata = {
        'total_frames': total_frames, 'asset_dim': asset_dim, 'dtype': 'float32',
        'date_strings': [
            f"{original_index[i].strftime('%Y-%m-%d')} to {original_index[i + lookback_period - 1].strftime('%Y-%m-%d')}"
            for i in range(total_frames)
        ]
    }
    with open(os.path.join(output_dir, 'metadata.pkl'), 'wb') as f: pickle.dump(metadata, f)
    frames_per_thread = (total_frames + num_threads - 1) // num_threads
    tasks = []
    for i in range(num_threads):
        start_frame_global = i * frames_per_thread
        end_frame_global = min(start_frame_global + frames_per_thread, total_frames)
        num_frames_in_chunk = end_frame_global - start_frame_global
        if num_frames_in_chunk > 0:
            slice_start = start_frame_global
            slice_end = end_frame_global + lookback_period - 1
            chunk_np = all_returns_np[slice_start:slice_end]
            task_args = (chunk_np, lookback_period, n_clusters_to_form,
                         num_frames_in_chunk, i, output_dir, MICRO_BATCH_SIZE)
            tasks.append(task_args)
    print(f"\nGenerating {total_frames} matrices across {len(tasks)} chunks...")
    with multiprocessing.Pool(processes=num_threads) as pool:
        pool.starmap(process_and_stream_chunk, tasks)
    print("\nAll matrix chunks generated successfully.")
    return output_dir

# --- NEW: RENDERING FUNCTION WITH MINI-CHUNK CACHING ---
def render_video_from_streamed_chunks(
    matrix_dir: str,
    output_filename: str,
    n_clusters: int,
    render_ram_budget_gb: float = 4.0
):
    """
    Renders the video by reading matrices in small, fixed-size "mini-chunks"
    from the large .bin files on disk. This keeps RAM usage low and predictable.
    """
    with open(os.path.join(matrix_dir, 'metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)

    total_frames = metadata['total_frames']
    asset_dim = metadata['asset_dim']
    dtype = np.dtype(metadata['dtype'])
    date_strings = metadata['date_strings']

    chunk_files = sorted([os.path.join(matrix_dir, f) for f in os.listdir(matrix_dir) if f.endswith('.bin')])

    if not chunk_files:
        print("Error: No matrix chunk files (.bin) found in the directory.")
        return

    # --- Memory-Efficient "Mini-Chunk" Caching Logic ---
    matrix_size_bytes = asset_dim * asset_dim * dtype.itemsize
    ram_budget_bytes = render_ram_budget_gb * 1024**3

    # Calculate how many full matrices we can fit in our RAM budget
    matrix_cache_size_frames = max(1, int(ram_budget_bytes // matrix_size_bytes))

    chunk_frame_counts = [os.path.getsize(f) // matrix_size_bytes for f in chunk_files]
    chunk_start_frames = np.cumsum([0] + chunk_frame_counts[:-1]).astype(np.int64)

    print(f"\nPreparing to render {total_frames} frames from {len(chunk_files)} large chunk files.")
    print(f"Rendering RAM budget set to {render_ram_budget_gb} GB.")
    print(f"Reading data in mini-chunks of {matrix_cache_size_frames} matrices (~{matrix_cache_size_frames * matrix_size_bytes / 1024**2:.2f} MB each).")

    # --- Animation Setup ---
    fig, ax = plt.subplots(figsize=(10, 8.8), dpi=64)
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
    fig.patch.set_facecolor('#F0F0F0'); ax.set_facecolor('#F0F0F0')

    # --- Caching variables for the update function ---
    # We use a mutable dict to hold state across calls to the update function.
    cache = {
        'data': None,          # Will hold the numpy array of the current mini-chunk
        'start_frame': -1,     # Global start frame of the cached data
        'end_frame': -1        # Global end frame (exclusive) of the cached data
    }

    # Function to load a new mini-chunk into the cache
    def load_minicache(global_frame_to_load):
        # Find which large .bin file the frame belongs to
        target_chunk_idx = np.searchsorted(chunk_start_frames, global_frame_to_load, side='right') - 1

        # Calculate where to start reading within that large file
        local_start_frame = global_frame_to_load - chunk_start_frames[target_chunk_idx]

        filepath = chunk_files[target_chunk_idx]

        print(f"  CACHE MISS: Loading mini-chunk starting at frame {global_frame_to_load} from file {os.path.basename(filepath)}...")

        with open(filepath, 'rb') as f:
            # Seek to the exact byte location of the desired matrix
            byte_offset = local_start_frame * matrix_size_bytes
            f.seek(byte_offset)

            # Read exactly one mini-chunk's worth of data (or less if at EOF)
            elements_to_read = matrix_cache_size_frames * asset_dim * asset_dim
            raw_data = np.fromfile(f, dtype=dtype, count=elements_to_read)

        # Update cache content and its global frame range
        num_frames_read = len(raw_data) // (asset_dim * asset_dim)
        cache['data'] = raw_data.reshape((num_frames_read, asset_dim, asset_dim))
        cache['start_frame'] = global_frame_to_load
        cache['end_frame'] = global_frame_to_load + num_frames_read

    # Initialize by loading the first mini-chunk
    load_minicache(0)

    im = ax.imshow(cache['data'][0], cmap='bwr', vmin=-1, vmax=1, interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cbar.set_label('Correlation')
    ax.set_xticks([]); ax.set_yticks([])
    main_title = ax.set_title(f"Clustered Asset Correlation ({n_clusters} Clusters)", fontsize=16, pad=30)
    date_subtitle = fig.text(0.5, 0.92, "Date Range", ha='center', fontsize=12, color='gray')

    def update(frame_num):
        # Check if the requested frame is outside our current cached mini-chunk
        if not (cache['start_frame'] <= frame_num < cache['end_frame']):
            load_minicache(frame_num)

        # Calculate the frame's index *within* the currently loaded mini-chunk
        local_frame_idx = frame_num - cache['start_frame']

        im.set_data(cache['data'][local_frame_idx])
        date_subtitle.set_text(date_strings[frame_num])

        return im, date_subtitle

    ani = animation.FuncAnimation(fig, update, frames=total_frames, blit=True, interval=50)
    writer = animation.writers['ffmpeg'](fps=15, metadata=dict(artist='MyScript'), bitrate=4000,
                                         extra_args=['-vcodec', 'h264_qsv', '-preset', 'fast'])

    print(f"Stitching video: {output_filename}...")
    ani.save(output_filename, writer=writer)
    plt.close(fig)
    print("\nVideo created successfully.")

# --- Main Execution Block (MODIFIED FOR A QUICK TEST) ---
if __name__ == '__main__':
    multiprocessing.freeze_support()

    # --- Parameters for a QUICK test run ---
    # 1. Use a short lookback period to reduce work per frame.
    lookback = 30
    # 2. Use fewer threads if you want to see clearer log output for debugging.
    num_threads_to_use = os.cpu_count() - 1
    # 3. Only run for ONE cluster value to avoid repeating the whole process.
    cluster_values_to_run = [3]

    print(f"--- RUNNING IN QUICK TEST MODE ---")
    print(f"Using {num_threads_to_use} threads for parallel processing.")
    print("Loading data...")
    df_full = pd.read_parquet('log_returns_by_ticker.parquet')

    # --- CRITICAL CHANGE: Slice the DataFrame to a very small size ---
    # This reduces BOTH the number of frames and the complexity of each matrix.
    # We take 100 rows and 20 columns.
    # This will generate 100 - 30 = 70 frames.
    df = df_full.iloc[:100, :20]
    # -------------------------------------------------------------------

    print(f"Data loaded and SLICED to: {df.shape[0]} rows, {df.shape[1]} assets for the test.")

    for num_clusters in cluster_values_to_run:
        print(f"\n{'='*30}\n  STARTING TEST RUN FOR {num_clusters} CLUSTERS  \n{'='*30}\n")
        matrix_output_dir = f"temp_matrices_TEST_{num_clusters}clusters"

        temp_dir = parallel_generate_matrices(
            asset_returns_df=df, lookback_period=lookback, n_clusters_to_form=num_clusters,
            num_threads=num_threads_to_use, output_dir=matrix_output_dir)

        if not temp_dir:
            print("Matrix generation failed. Skipping run.")
            continue

        final_movie_name = f"final_correlation_movie_TEST_{num_clusters}clusters.mp4"
        success = render_video_from_streamed_chunks(
            matrix_dir=temp_dir,
            output_filename=final_movie_name,
            n_clusters=num_clusters,
            render_ram_budget_gb=15.0  # Set the rendering RAM budget here. 4GB is a safe default.
        )

        if success:
            print(f"Cleaning up temporary matrix chunks in {temp_dir}...")
            shutil.rmtree(temp_dir)
            print("Cleanup complete.")

    print(f"\n{'='*30}\n  QUICK TEST RUN COMPLETED  \n{'='*30}\n")