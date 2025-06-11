import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_returns(results_dict):
    # Your data list
    data = results_dict['cluster_avg_pnl_list']

    # Extract fields
    window_ids = [d['Window_ID'] for d in data]
    avg_pnls = [d['Avg_Window_PNL'] for d in data]
    n_clusters = [d['N_Clusters'] for d in data]
    var_orders = [d['VAR_Order'] for d in data]

    # Convert per-window returns to basis points
    avg_pnls_bps = [r * 10_000 for r in avg_pnls]

    # Compute cumulative return and convert to bps
    cumulative_returns = np.cumsum([r for r in avg_pnls])
    cumulative_returns_bps = (cumulative_returns - 1) * 10_000

    # Create plot
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # Plot 1: Avg PNL in basis points and cumulative return in bps
    ax1 = axs[0]
    line1 = ax1.plot(window_ids, avg_pnls, marker='o', label="Window Log Return")
    ax1.set_ylabel("Window Log Return")
    ax1.set_title("Log Return per Window and Cumulative Log Return")

    # Second y-axis for cumulative return (also in bps now)
    ax2 = ax1.twinx()
    line2 = ax2.plot(window_ids, cumulative_returns, color='green', linestyle='--', label="Cumulative Log Return")
    ax2.set_ylabel("Cumulative Log Return")

    # Combine legends from both axes
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


def plot_errors(results_dict):
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
        window_cluster_relative_mae.append(relative_mae.values)

    window_ids = np.arange(len(window_cluster_rmse))

    # Create figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(14, 12), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1]})

    # Subplot 1: RMSE boxplot
    axs[0].boxplot(window_cluster_rmse, positions=window_ids, widths=0.6)
    axs[0].set_ylabel("Cluster RMSE")
    axs[0].set_title("Cluster-wise Forecast RMSE per Window")
    axs[0].grid(True)

    # Subplot 2: Relative MAE boxplot
    axs[1].boxplot(window_cluster_relative_mae, positions=window_ids, widths=0.6)
    axs[1].set_ylabel("Relative MAE")
    axs[1].set_title("Cluster-wise Normalized MAE per Window")
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



# def plot_rmse(results_dict):
