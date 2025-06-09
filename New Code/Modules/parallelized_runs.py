import pandas as pd
import numpy as np
import multiprocessing
from ClusterVARForecast import ClusterVARForecaster, NaiveVARForecaster

def calculate_pnl(forecast_df, actual_df):
    # Assumes forecast_df and actual_df are valid, have common columns, and alignable indices
    common_cols = forecast_df.columns.intersection(actual_df.columns)
    f_aligned, a_aligned = forecast_df[common_cols].copy(), actual_df[common_cols].copy()

    f_aligned.reset_index(drop=True, inplace=True)
    a_aligned.reset_index(drop=True, inplace=True)

    min_len = min(len(f_aligned), len(a_aligned))
    f_aligned = f_aligned.iloc[:min_len]
    a_aligned = a_aligned.iloc[:min_len]


    # STRATEGY 1: Go long $1 on clusters with positive forecast return, go short $1 on clusters with negative forecast return
    # positions = np.sign(f_aligned)

    # STRATEGY 2:
    positions = f_aligned / f_aligned.abs().sum()
    # print(positions)

    pnl_per_period_per_asset = positions * a_aligned
    total_pnl_per_asset_or_cluster = pnl_per_period_per_asset.sum(axis=0)
    return total_pnl_per_asset_or_cluster

def _process_single_hyper_eval_task(args_bundle):
    """
    Processes a single hyperparameter evaluation task by performing:
    1. Data preparation (unpacking and reconstructing DataFrames).
    2. Cluster-based VAR forecasting and evaluation.
    3. PNL calculation.
    """
    window_idx, hyper_train_df_tuple, hyper_eval_df_tuple, \
    L_hyper, E_hyper, asset_columns_list, \
    k_val, cluster_method_param, p_val, sigma_param, \
    rep_idx, k_idx, p_idx = args_bundle

    hyper_train_df = pd.DataFrame(
        hyper_train_df_tuple[0],
        index=hyper_train_df_tuple[1],
        columns=hyper_train_df_tuple[2]
    )
    hyper_eval_df = pd.DataFrame(
        hyper_eval_df_tuple[0],
        index=hyper_eval_df_tuple[1],
        columns=hyper_eval_df_tuple[2]
    )

    cluster_forecaster = ClusterVARForecaster(
        n_clusters=k_val, # n_clusters_param
        cluster_method=cluster_method_param,
        var_order=p_val, # var_order_param
        sigma_for_weights=sigma_param
    )

    cluster_forecaster.lookback_start_idx_ = 0
    cluster_forecaster.lookback_end_idx_ = L_hyper # L from original
    cluster_forecaster._define_clusters_and_centroids(hyper_train_df) # lookback_df from original

    lookback_cluster_returns = cluster_forecaster._calculate_weighted_cluster_returns(
        hyper_train_df, # lookback_df
        (0, L_hyper)    # (0, L)
    )

    forecast_horizon_for_cluster = E_hyper + 1 # E + 1
    forecasted_returns_cluster = cluster_forecaster._forecast(
        lookback_cluster_returns,
        forecast_horizon_for_cluster,
        cross_val=True
    )
    true_eval_returns_cluster = cluster_forecaster._calculate_weighted_cluster_returns(
        hyper_eval_df,  # eval_df
        (0, E_hyper)    # (0, E)
    )

    pnl_series_cluster = calculate_pnl(forecasted_returns_cluster, true_eval_returns_cluster)
    avg_pnl_cluster = pnl_series_cluster.mean() # Assumes pnl_series_cluster is not empty

    pnl = avg_pnl_cluster # This was the return of _hyperparameter_search_worker

    print(f"Rep: {rep_idx}, K: {k_idx}, P: {p_idx}, PNL: {pnl}")
    return window_idx, rep_idx, k_idx, p_idx, pnl

def _perform_final_evaluation_for_window_task(args_bundle):
    window_idx, lookback_df_tuple, eval_df_tuple, L_window, E_window, asset_columns_list, \
    best_n_clusters, best_var_order, cluster_method_param, sigma_param, \
    run_naive_var_comparison_flag, store_sample_forecasts_flag_for_this_window = args_bundle

    lookback_df = pd.DataFrame(lookback_df_tuple[0], index=lookback_df_tuple[1], columns=lookback_df_tuple[2])
    eval_df = pd.DataFrame(eval_df_tuple[0], index=eval_df_tuple[1], columns=eval_df_tuple[2])

    # --- ClusterVARForecaster part ---
    # Assumes L_window > best_var_order
    cluster_forecaster = ClusterVARForecaster(best_n_clusters, cluster_method_param, best_var_order, sigma_param)
    cluster_forecaster.lookback_start_idx_ = 0
    cluster_forecaster.lookback_end_idx_ = L_window
    cluster_forecaster._define_clusters_and_centroids(lookback_df)

    lookback_cluster_returns = cluster_forecaster._calculate_weighted_cluster_returns(lookback_df, (0, L_window))
    cluster_forecaster._fit(lookback_cluster_returns)
    forecast_horizon = E_window + 1
    true_eval_returns_cluster = cluster_forecaster._calculate_weighted_cluster_returns(eval_df, (0, E_window))
    forecasted_returns_cluster = cluster_forecaster._forecast(true_eval_returns_cluster, forecast_horizon, cross_val=False)
    pnl_series_cluster = calculate_pnl(forecasted_returns_cluster, true_eval_returns_cluster)
    avg_pnl_cluster = pnl_series_cluster.mean()

    forecast_data_cluster_sample, actual_data_cluster_sample = None, None
    if store_sample_forecasts_flag_for_this_window:
        forecast_data_cluster_sample = forecasted_returns_cluster
        actual_data_cluster_sample = true_eval_returns_cluster
        forecast_data_cluster_sample.index = pd.RangeIndex(len(forecast_data_cluster_sample))
        actual_data_cluster_sample.index = pd.RangeIndex(len(actual_data_cluster_sample))

    # --- NaiveVARForecaster part ---
    avg_pnl_naive = np.nan # Default if not run
    if run_naive_var_comparison_flag:
        # Assumes L_window > best_var_order
        naive_forecaster = NaiveVARForecaster(best_var_order)
        valid_lookback_df = lookback_df.astype(float).dropna(axis=1, how='all')
        naive_forecaster._fit(valid_lookback_df)
        forecast_horizon = E_window + 1
        forecasted_returns_naive = naive_forecaster._forecast(eval_df, forecast_horizon, cross_val=False)
        forecasted_returns_naive = forecasted_returns_naive.reindex(columns=eval_df.columns, fill_value=np.nan)
        pnl_series_naive = calculate_pnl(forecasted_returns_naive, eval_df)
        avg_pnl_naive = pnl_series_naive.mean()

    return window_idx, avg_pnl_cluster, avg_pnl_naive, float(best_n_clusters), float(best_var_order), \
           forecast_data_cluster_sample, actual_data_cluster_sample

def run_sliding_window_var_evaluation_vectorized(
    asset_returns_df, initial_lookback_len, eval_len, repetitions, n_clusters_config,
    cluster_method, var_order_config, sigma_intra_cluster, num_windows_config,
    store_sample_forecasts=True, run_naive_var_comparison=True, max_threads=4
):
    # Assumes asset_returns_df is valid, and total_T is sufficient for num_windows_config
    total_T, S = asset_returns_df.shape
    asset_columns_list = asset_returns_df.columns.tolist()
    asset_returns_np = asset_returns_df.to_numpy()
    num_actual_windows = num_windows_config # Simplified: Assumes num_windows_config is valid

    all_hyper_eval_tasks = []
    window_data_map = {}

    print("Phase 1: Preparing hyperparameter evaluation tasks...")
    # Assumes initial_lookback_len > eval_len for hyperparameter search to be meaningful
    for i in range(num_actual_windows):
        lb_start = i * eval_len
        lb_end = lb_start + initial_lookback_len
        eval_start = lb_end
        eval_end = eval_start + eval_len

        lookback_np_window = asset_returns_np[lb_start:lb_end]
        eval_np_window = asset_returns_np[eval_start:eval_end]
        lookback_df_idx = asset_returns_df.index[lb_start:lb_end]
        eval_df_idx = asset_returns_df.index[eval_start:eval_end]
        window_data_map[i] = {
            'lookback_tuple': (lookback_np_window, lookback_df_idx, asset_columns_list),
            'eval_tuple': (eval_np_window, eval_df_idx, asset_columns_list)
        }

        hyper_train_len = initial_lookback_len - eval_len
        hyper_eval_len = eval_len
        hyper_train_np = lookback_np_window[:hyper_train_len]
        hyper_eval_np = lookback_np_window[hyper_train_len:]
        hyper_train_idx = lookback_df_idx[:hyper_train_len]
        hyper_eval_idx = lookback_df_idx[hyper_train_len:]
        hyper_train_df_tuple = (hyper_train_np, hyper_train_idx, asset_columns_list)
        hyper_eval_df_tuple = (hyper_eval_np, hyper_eval_idx, asset_columns_list)

        for rep_idx in range(repetitions):
            for k_idx, k_val in enumerate(n_clusters_config):
                for p_idx, p_val in enumerate(var_order_config):
                    # Assumes hyper_train_len > p_val
                    all_hyper_eval_tasks.append((
                        i, hyper_train_df_tuple, hyper_eval_df_tuple,
                        hyper_train_len, hyper_eval_len, asset_columns_list,
                        k_val, cluster_method, p_val, sigma_intra_cluster,
                        rep_idx, k_idx, p_idx
                    ))

    print(f"Phase 1: Running {len(all_hyper_eval_tasks)} hyperparameter PNL calculations in parallel...")
    with multiprocessing.Pool(processes=max_threads) as pool:
        hyper_search_results = pool.map(_process_single_hyper_eval_task, all_hyper_eval_tasks)
        # [result.wait() for result in hyper_search_results]
    print("Phase 1: Hyperparameter PNL calculations completed.")

    all_final_eval_tasks = []
    window_best_hyperparams = {}

    print("Phase 2: Determining best hyperparameters and preparing final evaluation tasks...")
    for i in range(num_actual_windows):
        results_for_window_i = [res for res in hyper_search_results if res[0] == i]
        # Assumes results_for_window_i is not empty and contains valid PNLs
        hyper_scores_cube = np.full((repetitions, len(n_clusters_config), len(list(var_order_config))), -np.inf) # Keep -np.inf for nanargmax
        for _, rep_idx, k_idx, p_idx, pnl_val in results_for_window_i:
            if not np.isnan(pnl_val): # Minimal check for safety with nanargmax
                 hyper_scores_cube[rep_idx, k_idx, p_idx] = pnl_val

        aggregated_hyper_scores = np.nanmax(hyper_scores_cube, axis=0) # nanmax is robust to NaNs
        # Assumes aggregated_hyper_scores is not all -inf or nan
        best_k_idx, best_p_idx = np.unravel_index(np.nanargmax(aggregated_hyper_scores), aggregated_hyper_scores.shape)
        best_n_clusters = n_clusters_config[best_k_idx]
        best_var_order = list(var_order_config)[best_p_idx]

        window_best_hyperparams[i] = {'n_clusters': best_n_clusters, 'var_order': best_var_order}

        lookback_tuple = window_data_map[i]['lookback_tuple']
        eval_tuple = window_data_map[i]['eval_tuple']
        store_sample_flag = store_sample_forecasts and (i == num_actual_windows - 1)

        all_final_eval_tasks.append((
            i, lookback_tuple, eval_tuple,
            initial_lookback_len, eval_len, asset_columns_list,
            best_n_clusters, best_var_order, cluster_method, sigma_intra_cluster,
            run_naive_var_comparison, store_sample_flag
        ))

    print(f"Phase 2: Running {len(all_final_eval_tasks)} final window evaluations in parallel...")
    with multiprocessing.Pool(processes=max_threads) as pool:
        final_results_list = pool.map(_perform_final_evaluation_for_window_task, all_final_eval_tasks)
        # [result.wait() for result in final_results_list]
    print("Phase 2: Final window evaluations completed.")

    all_window_pnl_cluster_list, all_window_pnl_naive_list = [], []
    sample_forecast_data_cluster, sample_actual_data_cluster, sample_window_idx_cluster = None, None, None

    for result_tuple in final_results_list:
        win_idx, pnl_c, pnl_n, n_c_sel, vo_sel, fc_samp, ac_samp = result_tuple

        all_window_pnl_cluster_list.append({ # Assumes pnl_c is a valid number
            'Window_ID': win_idx, 'Avg_Window_PNL': pnl_c,
            'N_Clusters': int(n_c_sel), 'VAR_Order': int(vo_sel), 'Method': 'Clustered VAR'
        })
        if run_naive_var_comparison: # Assumes pnl_n is a valid number if run
            all_window_pnl_naive_list.append({
                'Window_ID': win_idx, 'Avg_Window_PNL': pnl_n,
                'VAR_Order': int(vo_sel), 'Method': 'Naive VAR'
            })

        if store_sample_forecasts and (win_idx == num_actual_windows - 1):
            # Assumes fc_samp and ac_samp are valid if this condition is met
             sample_forecast_data_cluster = fc_samp
             sample_actual_data_cluster = ac_samp
             sample_window_idx_cluster = win_idx

    results_dict = {
        'cluster_avg_pnl_list': all_window_pnl_cluster_list,
        'sample_forecast_cluster': sample_forecast_data_cluster,
        'sample_actual_cluster': sample_actual_data_cluster,
        'sample_window_idx_cluster': sample_window_idx_cluster
    }
    if run_naive_var_comparison:
        results_dict['naive_avg_pnl_list'] = all_window_pnl_naive_list

    print("All processing finished.")
    return results_dict