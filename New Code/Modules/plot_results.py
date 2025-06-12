import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

def plot_returns(results_dict, convert_to_linear=False):
    # Your data list
    data = results_dict['cluster_avg_pnl_list']

    # Extract fields
    window_ids = [d['Window_ID'] for d in data]
    avg_pnls_log = [d['Avg_Window_PNL'] for d in data]
    n_clusters = [d['N_Clusters'] for d in data]
    var_orders = [d['VAR_Order'] for d in data]

    if convert_to_linear:
        # Convert log returns to linear returns
        avg_pnls = [np.exp(r) - 1 for r in avg_pnls_log]
        cumulative_returns = np.cumprod([1 + r for r in avg_pnls]) - 1

        # Report linear returns in basis points
        avg_pnls_display = [r * 10000 for r in avg_pnls]
        cumulative_returns_display = cumulative_returns * 10000

        y_label_window = "Window Return (bps)"
        y_label_cumulative = "Cumulative Return (bps)"
    else:
        # Keep log returns directly
        avg_pnls = avg_pnls_log
        cumulative_returns = np.cumsum(avg_pnls_log)

        # Report log returns as-is (unitless)
        avg_pnls_display = avg_pnls
        cumulative_returns_display = cumulative_returns

        y_label_window = "Window Log Return"
        y_label_cumulative = "Cumulative Log Return"

    # Create plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})

    # Plot 1: Avg return and cumulative return
    ax1 = axs[0]
    line1 = ax1.plot(window_ids, avg_pnls_display, marker='o', label="Window Return")
    ax1.set_ylabel(y_label_window)
    ax1.set_title("Window Return and Cumulative Return")

    # Second y-axis for cumulative return
    ax2 = ax1.twinx()
    line2 = ax2.plot(window_ids, cumulative_returns_display, color='green', linestyle='--', label="Cumulative Return")
    ax2.set_ylabel(y_label_cumulative)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right")

    # Plot 2: Bar chart of N_Clusters and VAR_Order
    bar_width = 0.4
    x = range(len(window_ids))
    axs[1].bar([i - bar_width / 2 for i in x], n_clusters, width=bar_width, label='N_Clusters')
    axs[1].bar([i + bar_width / 2 for i in x], var_orders, width=bar_width, label='VAR_Order')
    axs[1].set_ylabel("Value")
    axs[1].set_xlabel("Window ID")
    axs[1].set_title("N_Clusters and VAR_Order per Window")
    axs[1].legend(loc="upper right")

    plt.tight_layout()
    plt.show()
    return




def plot_interwindow_errors(results_dict):
    # Extract data from results_dict
    forecast_list = results_dict['per_cluster_forecasted_return']
    actual_list = results_dict['per_cluster_actual_return']
    pnl_list_data = results_dict['cluster_avg_pnl_list']

    # Extract cluster counts and VAR orders
    n_clusters = [d['N_Clusters'] for d in pnl_list_data]
    var_orders = [d['VAR_Order'] for d in pnl_list_data]

    # Storage for RMSEs and relative MAEs
    window_cluster_rmse = []
    window_cluster_relative_mae = []
    window_cluster_relative_rmse = []

    # Loop over each window
    for forecast_df, actual_df in zip(forecast_list, actual_list):
        forecast_df = forecast_df[actual_df.columns]  # ensure same columns
        
        # Compute RMSE per cluster
        cluster_rmses = np.sqrt(((forecast_df - actual_df) ** 2).mean())
        window_cluster_rmse.append(cluster_rmses.values)
        
        # Compute MAE per cluster
        cluster_maes = (forecast_df - actual_df).abs().mean()
        
        # Compute relative MAE (normalize by mean absolute actual value)
        mean_actual = actual_df.abs().mean()
        relative_mae = cluster_maes / mean_actual
        relative_rmse = cluster_rmses / mean_actual
        window_cluster_relative_mae.append(relative_mae.values)
        window_cluster_relative_rmse.append(relative_rmse.values)

    window_ids = np.arange(len(window_cluster_rmse))

    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})

    flier_properties = dict(marker='o', markerfacecolor='red', markersize=3)

    # Subplot 1: RMSE boxplot
    axs[0].boxplot(window_cluster_rmse, positions=window_ids, widths=0.6, flierprops=flier_properties)
    axs[0].set_ylabel("Cluster RMSE")
    axs[0].set_title("Cluster-wise Forecast RMSE per Window")
    axs[0].grid(True)

    # # Subplot 2: Relative MAE boxplot
    # axs[1].boxplot(window_cluster_relative_mae, positions=window_ids, widths=0.6)
    # axs[1].set_ylabel("Relative MAE")
    # axs[1].set_title("Cluster-wise Normalized MAE per Window")
    # axs[1].grid(True)

    # Subplot 2: Relative RMSE boxplot
    axs[1].boxplot(window_cluster_relative_rmse, positions=window_ids, widths=0.6, flierprops=flier_properties)
    axs[1].set_ylabel("Relative RMSE")
    axs[1].set_title("Cluster-wise Normalized RMSE per Window")
    axs[1].grid(True)

    # Subplot 3: N_Clusters and VAR_Order bar chart
    bar_width = 0.4
    x = range(len(window_ids))
    axs[2].bar([i - bar_width / 2 for i in x], n_clusters, width=bar_width, label='N_Clusters')
    axs[2].bar([i + bar_width / 2 for i in x], var_orders, width=bar_width, label='VAR_Order')
    axs[2].set_ylabel("Value")
    axs[2].set_xlabel("Window ID")
    axs[2].set_title("N_Clusters and VAR_Order per Window")
    axs[2].legend(loc="upper right")

    # Adaptive xticks: show every 10th window ID
    axs[2].set_xticks(window_ids[::10])
    axs[2].set_xticklabels(window_ids[::10])

    plt.tight_layout()
    plt.show()
    return

def plot_inwindow_errors(results_dict):
    # Load data
    forecast_list = results_dict['per_cluster_forecasted_return']
    actual_list = results_dict['per_cluster_actual_return']
    pnl_list_data = results_dict['cluster_avg_pnl_list']

    # Extract cluster count for each window
    n_clusters_list = [d['N_Clusters'] for d in pnl_list_data]

    # Compute in-window RMSE for each window
    window_rmse_series = []
    cluster_count_series = []

    for forecast_df, actual_df, n_clusters in zip(forecast_list, actual_list, n_clusters_list):
        forecast_df = forecast_df[actual_df.columns]  # align columns
        errors = (forecast_df - actual_df) ** 2
        rmse_by_day = np.sqrt(errors.mean(axis=1))
        window_rmse_series.append(rmse_by_day.values)
        cluster_count_series.append(n_clusters)

    # Group by cluster count
    rmse_by_cluster_count = defaultdict(list)

    for rmse_series, n_clusters in zip(window_rmse_series, cluster_count_series):
        rmse_by_cluster_count[n_clusters].append(rmse_series)

    # Compute means
    mean_rmse_by_cluster_count = {}
    window_count_by_cluster = {}

    for n_clusters, series_list in rmse_by_cluster_count.items():
        stacked = np.vstack(series_list)
        mean_rmse_by_cluster_count[n_clusters] = stacked.mean(axis=0)
        window_count_by_cluster[n_clusters] = len(series_list)

    # Compute overall mean RMSE across all windows
    all_rmse = np.vstack(window_rmse_series)
    overall_mean_rmse = all_rmse.mean(axis=0)

    # Determine shared y-limits for boxplots
    # Use np.nanmin and np.nanmax to safely handle NaN values
    global_ymin = np.nanmin([np.nanmin(np.vstack(rmse_by_cluster_count[k])) for k in rmse_by_cluster_count])
    global_ymax = np.nanmax([np.nanmax(np.vstack(rmse_by_cluster_count[k])) for k in rmse_by_cluster_count])

    # ---------------------------------------------------------
    # Plot combined comparison + box-and-whiskers

    num_clusters = len(mean_rmse_by_cluster_count)
    total_subplots = num_clusters + 1
    height_ratios = [1.5] + [0.8] * num_clusters

    fig, axs = plt.subplots(
        total_subplots, 1, figsize=(14, 3.5 * total_subplots),
        gridspec_kw={'height_ratios': height_ratios}, sharex=True
    )

    # Top plot: Mean curves
    ax_top = axs[0]
    for n_clusters in sorted(mean_rmse_by_cluster_count.keys()):
        mean_rmse = mean_rmse_by_cluster_count[n_clusters]
        label = f"{n_clusters} Clusters (N={window_count_by_cluster[n_clusters]})"
        ax_top.plot(mean_rmse, label=label)

    # Overall average curve
    ax_top.plot(overall_mean_rmse, color='black', linewidth=2.5, linestyle='--',
                label=f'Overall Avg (N={len(window_rmse_series)})')

    ax_top.set_ylabel("Mean In-Window RMSE")
    ax_top.set_title("Mean In-Window RMSE by Cluster Count")
    ax_top.legend()
    ax_top.grid(True)

    # Bottom plots: per-cluster box-and-whisker subplots
    for ax, n_clusters in zip(axs[1:], sorted(rmse_by_cluster_count.keys())):
        series_list = rmse_by_cluster_count[n_clusters]
        stacked = np.vstack(series_list)
        
        ax.boxplot([stacked[:, day] for day in range(stacked.shape[1])],
                positions=np.arange(stacked.shape[1]),
                widths=0.6)
        
        ax.set_ylim(global_ymin, global_ymax)
        ax.set_ylabel(f"In-Window RMSE\n{n_clusters} Clusters (N={window_count_by_cluster[n_clusters]})")
        ax.grid(True)

    # Bottom axis
    axs[-1].set_xlabel("Day in Window")

    plt.tight_layout()
    plt.show()
