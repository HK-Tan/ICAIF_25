import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy import sparse
import multiprocessing
import os
os.chdir('C:/Users/james/ICAIF_25')
from signet.cluster import Cluster
os.chdir('C:/Users/james/ICAIF_25/Current_Code/Data')

# --- WORKER FUNCTION ---
# This function is executed by each parallel process.
# It creates one chunk of the full animation.
def create_animation_chunk(asset_returns_df,
        lookback_period,
        n_clusters_to_form,
        start_frame,
        num_frames_in_chunk,
        chunk_index,
        output_prefix):
    """
    Worker function to render a single chunk of the animation without a progress bar.
    Designed to be called by multiprocessing.Pool.
    """
    asset_names = asset_returns_df.columns

    fig, ax = plt.subplots(figsize=(10, 8))
    fig.tight_layout(pad=3.0)

    def update(local_frame_num):
        global_frame_num = start_frame + local_frame_num
        ax.clear()

        start_index = global_frame_num
        end_index = global_frame_num + lookback_period
        window_df = asset_returns_df.iloc[start_index:end_index]

        correlation_matrix_np = np.corrcoef(window_df.values, rowvar=False)
        correlation_matrix_df = pd.DataFrame(correlation_matrix_np, index=asset_names, columns=asset_names).fillna(0)

        pos_corr = np.maximum(correlation_matrix_df.values, 0)
        neg_corr = np.maximum(-correlation_matrix_df.values, 0)
        signet_data = (sparse.csc_matrix(pos_corr), sparse.csc_matrix(neg_corr))
        effective_n_clusters = min(n_clusters_to_form, window_df.shape[1])
        cluster_obj = Cluster(signet_data)
        labels = cluster_obj.SPONGE_sym(effective_n_clusters)

        order_indices = np.argsort(labels)
        reordered_matrix = correlation_matrix_df.iloc[order_indices, order_indices]
        mask = np.triu(np.ones_like(reordered_matrix.values, dtype=bool))
        matrix_to_plot = reordered_matrix.copy()
        matrix_to_plot.values[mask] = np.nan

        ax.imshow(matrix_to_plot, cmap='viridis', vmin=-1, vmax=1, interpolation='none')
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

        start_date = window_df.index[0].strftime('%Y-%m-%d')
        end_date = window_df.index[-1].strftime('%Y-%m-%d')
        ax.set_title(f"Clustered Asset Correlation\n{start_date} to {end_date}", fontsize=16)

    # Create the animation object for this chunk
    ani = FuncAnimation(fig, update, frames=num_frames_in_chunk, repeat=False)

    # --- SAVE WITHOUT PROGRESS BAR ---
    output_filename = f"{output_prefix}_part_{chunk_index}.mp4"
    print(f"Starting to render chunk {chunk_index} ({num_frames_in_chunk} frames)...")

    # Save this chunk to a unique file without the progress_callback
    ani.save(
        output_filename,
        writer='ffmpeg',
        dpi=150
    )

    print(f"Finished rendering chunk {chunk_index} -> {output_filename}")

    plt.close(fig)
    return output_filename

# --- ORCHESTRATOR FUNCTION ---
# This function splits the work and manages the parallel processes.
def create_parallel_animations(
    asset_returns_df: pd.DataFrame,
    lookback_period: int,
    n_clusters_to_form: int,
    num_threads: int,
    output_prefix: str = "correlation_movie_chunk"
):
    """
    Splits the animation rendering task across multiple processes.
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

    # --- RUN TASKS WITHOUT PROGRESS BAR ---
    with multiprocessing.Pool(processes=num_threads) as pool:
        # pool.starmap blocks until all results are ready.
        # The tqdm wrapper has been removed.
        results = pool.starmap(create_animation_chunk, tasks)

    print("\nAll animation chunks have been created:")
    for filename in results:
        print(f"- {filename}")

    # --- Create file_list.txt for easy ffmpeg concatenation ---
    file_list_path = "file_list.txt"
    with open(file_list_path, "w") as f:
        for filename in sorted(results): # Sort to ensure correct order
            f.write(f"file '{filename}'\n")

    print(f"\nGenerated '{file_list_path}' for video stitching.")
    print("\n--- TO STITCH THE VIDEOS ---")
    print("Run the following command in your terminal:")
    print(f"ffmpeg -f concat -safe 0 -i {file_list_path} -c copy final_correlation_movie.mp4")


# --- Main Execution Block ---
if __name__ == '__main__':
    # This check is crucial for multiprocessing to work correctly
    multiprocessing.freeze_support()

    lookback = 252
    num_clusters = 5

    # USER-DEFINED: SET THE NUMBER OF THREADS/PROCESSES
    num_threads_to_use = os.cpu_count() or 4

    df = pd.read_parquet('log_returns_by_ticker.parquet')

    # 2. Run the parallel animation creation process
    create_parallel_animations(
        asset_returns_df=df,
        lookback_period=lookback,
        n_clusters_to_form=num_clusters,
        num_threads=num_threads_to_use,
    )