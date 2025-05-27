import numpy as np
import pandas as pd
import sys
from scipy import sparse
import warnings
from sklearn.cluster import KMeans, SpectralClustering

warnings.filterwarnings('ignore')

# Attempt to import Signet and install if not found
try:
    from signet.cluster import Cluster
except ImportError:
    print("Signet package not found. Attempting to install from GitHub...")
    try:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/alan-turing-institute/SigNet.git"]
        )
        from signet.cluster import Cluster
        print("Signet package installed successfully.")
    except Exception as e:
        print(f"Error installing Signet package: {e}")
        print("Please install it manually: pip install git+https://github.com/alan-turing-institute/SigNet.git")
        sys.exit(1)

# ----------------------------------------------------------------

class PyFolio:
    """
    Performs clustering on asset returns, fits a VAR model to cluster returns,
    forecasts future cluster returns, and calculates forecast errors.
    """
    def __init__(self, n_clusters, cluster_method, var_order):
        """
        Initializes the ClusterVARForecaster.

        Args:
            n_clusters (int): Number of clusters to form.
            cluster_method (str): Method for clustering assets ('SPONGE', 'signed_laplacian',
                                  'SPONGE_sym', 'Kmeans', 'spectral_clustering').
            var_order (int): Order of the VAR model.
        """
        self.n_clusters = max(1, int(n_clusters))
        self.cluster_method = cluster_method
        self.var_order = max(1, int(var_order))
        self.corr_matrix_ = None
        self.cluster_definitions_ = None # Stores {'Cluster_1': [tickers], ...}

    def _calculate_correlation_matrix(self, asset_returns_lookback_df):
        """
        Calculates the Pearson correlation matrix from asset returns.

        Args:
            asset_returns_lookback_df (pandas.DataFrame): DataFrame of asset returns for the lookback period.

        Returns:
            pandas.DataFrame: The correlation matrix.
        """
        if asset_returns_lookback_df.empty:
            return pd.DataFrame()

        mean_vals = asset_returns_lookback_df.mean(axis=0)
        std_vals = asset_returns_lookback_df.std(axis=0)
        std_vals[std_vals < 1e-9] = 1.0 # Avoid division by zero for constant series

        normalized_data = (asset_returns_lookback_df - mean_vals) / std_vals
        correlation_matrix = normalized_data.corr(method='pearson').fillna(0)
        return correlation_matrix

    def _apply_clustering_algorithm(self, correlation_matrix_df, num_clusters_to_form):
        """
        Applies the specified clustering algorithm.

        Args:
            correlation_matrix_df (pandas.DataFrame): The correlation matrix of assets.
            num_clusters_to_form (int): The target number of clusters.

        Returns:
            numpy.ndarray: Array of cluster labels for each asset.
        """
        # Helper for Signet input
        def _get_signet_data(corr_df):
            pos_corr = corr_df.applymap(lambda x: x if x >= 0 else 0)
            neg_corr = corr_df.applymap(lambda x: abs(x) if x < 0 else 0)
            return (sparse.csc_matrix(pos_corr.values), sparse.csc_matrix(neg_corr.values))

        if correlation_matrix_df.empty:
            return np.array([])

        num_assets = correlation_matrix_df.shape[0]
        effective_n_clusters = min(num_clusters_to_form, num_assets) if num_assets > 0 else 1
        if effective_n_clusters <= 0: return np.array([])


        if self.cluster_method == 'SPONGE':
            signet_data = _get_signet_data(correlation_matrix_df)
            cluster_obj = Cluster(signet_data)
            labels = cluster_obj.SPONGE(effective_n_clusters)
        elif self.cluster_method == 'signed_laplacian':
            signet_data = _get_signet_data(correlation_matrix_df)
            cluster_obj = Cluster(signet_data)
            labels = cluster_obj.spectral_cluster_laplacian(effective_n_clusters)
        elif self.cluster_method == 'SPONGE_sym':
            signet_data = _get_signet_data(correlation_matrix_df)
            cluster_obj = Cluster(signet_data)
            labels = cluster_obj.SPONGE_sym(effective_n_clusters)
        elif self.cluster_method == 'Kmeans':
            data_for_kmeans = correlation_matrix_df.fillna(0)
            kmeans = KMeans(n_clusters=effective_n_clusters, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(data_for_kmeans)
        elif self.cluster_method == 'spectral_clustering':
            affinity_matrix = np.abs(correlation_matrix_df.fillna(0).values)
            affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2
            np.fill_diagonal(affinity_matrix, 1)
            try:
                sc = SpectralClustering(n_clusters=effective_n_clusters, affinity='precomputed',
                                        random_state=0, assign_labels='kmeans')
                labels = sc.fit_predict(affinity_matrix)
            except Exception: # Fallback to KMeans
                data_for_kmeans = correlation_matrix_df.fillna(0)
                kmeans = KMeans(n_clusters=effective_n_clusters, random_state=0, n_init='auto')
                labels = kmeans.fit_predict(data_for_kmeans)
        else:
            raise ValueError(f"Unknown clustering method: {self.cluster_method}")

        if not isinstance(labels, np.ndarray) or labels.size == 0:
            if num_assets > 0:
                return np.zeros(num_assets, dtype=int) # Default to single cluster
            return np.array([])
        return labels

    def _define_clusters(self, asset_returns_lookback_df):
        """
        Performs clustering and defines cluster compositions.

        Args:
            asset_returns_lookback_df (pandas.DataFrame): Asset returns for the lookback period.

        Sets:
            self.corr_matrix_ (pandas.DataFrame): Calculated correlation matrix.
            self.cluster_definitions_ (dict): Maps cluster names to lists of ticker symbols.
                                             e.g., {'Cluster_1': ['AAPL', 'MSFT'], ...}
        """
        self.corr_matrix_ = self._calculate_correlation_matrix(asset_returns_lookback_df)

        num_assets = self.corr_matrix_.shape[0]
        if num_assets == 0 :
            self.cluster_definitions_ = {}
            return

        labels = self._apply_clustering_algorithm(self.corr_matrix_, self.n_clusters)

        if labels.size == 0 and num_assets > 0 : # Clustering failed, put all in one
             labels = np.zeros(num_assets, dtype=int)
        elif labels.size == 0 and num_assets == 0:
            self.cluster_definitions_ = {}
            return


        asset_names = list(self.corr_matrix_.columns)
        labeled_assets = pd.DataFrame({'ClusterLabel': labels}, index=asset_names)
        # Ensure actual number of clusters is reflected if Signet changes it
        actual_n_clusters = len(np.unique(labels))
        self.n_clusters = actual_n_clusters


        cluster_definitions = {}
        for label_val in np.unique(labels):
            cluster_name = f'Cluster_{label_val + 1}' # 1-based naming
            tickers_in_cluster = list(labeled_assets[labeled_assets['ClusterLabel'] == label_val].index)
            if tickers_in_cluster: # Only add if cluster is not empty
                cluster_definitions[cluster_name] = tickers_in_cluster

        self.cluster_definitions_ = cluster_definitions

    def _calculate_equal_weighted_cluster_returns(self, asset_returns_df, period_indices):
        """
        Calculates equal-weighted returns for each defined cluster over a given period.

        Args:
            asset_returns_df (pandas.DataFrame): Full historical asset returns.
            period_indices (tuple): (start_idx, end_idx) for slicing asset_returns_df.

        Returns:
            pandas.DataFrame: DataFrame of cluster returns, columns are cluster names.
                              Returns empty DataFrame if no clusters or data.
        """
        if not self.cluster_definitions_:
            # print("Warning: Cluster definitions not available. Cannot calculate cluster returns.")
            return pd.DataFrame()

        start_idx, end_idx = period_indices
        if not (0 <= start_idx < end_idx <= len(asset_returns_df)):
            # print("Warning: Invalid period indices for calculating cluster returns.")
            return pd.DataFrame(index=pd.Index([]), columns=list(self.cluster_definitions_.keys()))

        data_slice = asset_returns_df.iloc[start_idx:end_idx]

        cluster_returns_dict = {}
        for cluster_name, tickers in self.cluster_definitions_.items():
            # Filter for tickers present in the current data_slice
            valid_tickers = [ticker for ticker in tickers if ticker in data_slice.columns]
            if not valid_tickers:
                # If no valid tickers for this cluster in this slice, assign zeros or NaNs
                cluster_returns_dict[cluster_name] = np.zeros(len(data_slice))
                # Alternatively, np.full(len(data_slice), np.nan) if preferred
                continue

            # Calculate mean returns for the valid tickers in the cluster
            cluster_returns_dict[cluster_name] = data_slice[valid_tickers].mean(axis=1).values

        if not cluster_returns_dict: # No clusters had valid tickers
            return pd.DataFrame(index=data_slice.index)

        return pd.DataFrame(cluster_returns_dict, index=data_slice.index)

    def _fit_var_and_forecast(self, lookback_cluster_returns_df, forecast_horizon):
        """
        Fits a VAR model to lookback cluster returns and forecasts for a given horizon.

        Args:
            lookback_cluster_returns_df (pandas.DataFrame): Time series of cluster returns for fitting VAR.
            forecast_horizon (int): Number of steps to forecast.

        Returns:
            pandas.DataFrame: Forecasted cluster returns. Empty if model cannot be fit.
        """
        data_for_var = lookback_cluster_returns_df.astype(float).dropna(axis=1, how='all')

        if data_for_var.empty or data_for_var.shape[0] <= self.var_order or data_for_var.shape[1] == 0:
            # print("Warning: Not enough data or series for VAR model. Returning empty forecast.")
            return pd.DataFrame(columns=lookback_cluster_returns_df.columns if not lookback_cluster_returns_df.empty else ["dummy_var_col"])

        num_obs, num_series = data_for_var.shape

        Y_matrix = data_for_var.iloc[self.var_order:].values

        X_regressors_list = [np.ones((num_obs - self.var_order, 1))] # Constant
        for lag in range(1, self.var_order + 1):
            X_regressors_list.append(data_for_var.iloc[self.var_order - lag : num_obs - lag].values)
        X_matrix = np.hstack(X_regressors_list)

        if X_matrix.shape[0] != Y_matrix.shape[0] or X_matrix.shape[0] == 0 :
             return pd.DataFrame(columns=data_for_var.columns)


        try:
            coefficients = np.linalg.solve(X_matrix.T @ X_matrix, X_matrix.T @ Y_matrix)
        except np.linalg.LinAlgError:
            try: # Use pseudo-inverse as fallback
                coefficients = np.linalg.pinv(X_matrix.T @ X_matrix) @ (X_matrix.T @ Y_matrix)
            except np.linalg.LinAlgError:
                return pd.DataFrame(columns=data_for_var.columns) # Cannot solve

        forecasts_array = np.zeros((forecast_horizon, num_series))
        history_for_fcst = data_for_var.iloc[num_obs - self.var_order : num_obs].values

        for i in range(forecast_horizon):
            lagged_vals_flat = history_for_fcst[::-1].ravel()
            current_X_fcst = np.hstack(([1.0], lagged_vals_flat))

            next_forecast = current_X_fcst @ coefficients
            forecasts_array[i, :] = next_forecast
            history_for_fcst = np.vstack((history_for_fcst[1:], next_forecast))

        # Create index for the forecast DataFrame (assuming daily, can be adapted)
        forecast_index = pd.RangeIndex(start=0, stop=forecast_horizon, step=1) # Generic index
        if not data_for_var.empty and hasattr(data_for_var.index, 'freq') and data_for_var.index.freq is not None:
            try:
                forecast_index = pd.date_range(start=data_for_var.index[-1] + data_for_var.index.freq,
                                            periods=forecast_horizon, freq=data_for_var.index.freq)
            except Exception: # Fallback to generic if date range fails
                 pass


        forecast_df = pd.DataFrame(forecasts_array, columns=data_for_var.columns, index=forecast_index)
        return forecast_df.reindex(columns=lookback_cluster_returns_df.columns, fill_value=0.0)


    def process_step(self, asset_returns_df, lookback_indices, eval_len):
        """
        Processes one step of clustering, VAR fitting, forecasting, and true value calculation.

        Args:
            asset_returns_df (pandas.DataFrame): Full historical asset returns.
            lookback_indices (tuple): (start_idx, end_idx) for the lookback period.
            eval_len (int): Length of the evaluation period (forecast horizon).

        Returns:
            tuple: (forecasted_cluster_returns_df, true_eval_cluster_returns_df)
                   Returns (empty_df, empty_df) if processing fails at any stage.
        """
        lb_start, lb_end = lookback_indices

        # 1. Define clusters based on lookback data
        lookback_asset_returns = asset_returns_df.iloc[lb_start:lb_end]
        self._define_clusters(lookback_asset_returns)

        if not self.cluster_definitions_:
            # print("No clusters defined, cannot proceed with VAR.")
            empty_df = pd.DataFrame()
            return empty_df, empty_df

        # 2. Calculate cluster returns for the lookback period
        lookback_cluster_returns = self._calculate_equal_weighted_cluster_returns(asset_returns_df, lookback_indices)
        if lookback_cluster_returns.empty or lookback_cluster_returns.isnull().all().all():
            # print("Lookback cluster returns are empty or all NaN.")
            empty_df = pd.DataFrame(columns=list(self.cluster_definitions_.keys()))
            return empty_df, empty_df.copy()


        # 3. Fit VAR and forecast
        forecasted_returns = self._fit_var_and_forecast(lookback_cluster_returns, eval_len)

        # 4. Calculate true cluster returns for the evaluation period
        eval_start_idx = lb_end
        eval_end_idx = lb_end + eval_len

        # Ensure eval_end_idx does not exceed data length
        eval_end_idx = min(eval_end_idx, len(asset_returns_df))
        actual_eval_len = eval_end_idx - eval_start_idx

        if actual_eval_len <= 0: # No evaluation period possible
            # print("Evaluation period has zero or negative length.")
            true_eval_returns = pd.DataFrame(columns=forecasted_returns.columns) # Match columns if forecast exists
            # Adjust forecast to match actual evaluation length if needed (though forecast is already for eval_len)
            return forecasted_returns.head(0), true_eval_returns # Return empty structure matching columns


        true_eval_returns = self._calculate_equal_weighted_cluster_returns(asset_returns_df,
                                                                         (eval_start_idx, eval_end_idx))

        # Align forecast index with true evaluation data index for easier comparison later
        if not forecasted_returns.empty and not true_eval_returns.empty:
            if len(forecasted_returns) == len(true_eval_returns): # Only if lengths match
                 forecasted_returns.index = true_eval_returns.index
            else: # Lengths mismatch, could be due to forecast_horizon vs actual_eval_len
                  # Truncate or pad forecast to match true_eval_returns length for direct comparison
                  # For now, we return as is, error calculation needs to handle this.
                  # Simplest for now: truncate forecast if longer.
                  if len(forecasted_returns) > len(true_eval_returns):
                      forecasted_returns = forecasted_returns.iloc[:len(true_eval_returns)]
                  # If forecast is shorter, that's an issue - VAR produced fewer forecasts than actual eval days.
                  # This can happen if eval_len > forecast_horizon used in _fit_var_and_forecast
                  # (but they should be the same: forecast_horizon=eval_len).
                  # Or if true_eval_returns has fewer days than eval_len due to data end.

        # Ensure columns match between forecast and true returns
        if not forecasted_returns.empty and not true_eval_returns.empty:
            common_cols = forecasted_returns.columns.intersection(true_eval_returns.columns)
            forecasted_returns = forecasted_returns[common_cols]
            true_eval_returns = true_eval_returns[common_cols]
        elif forecasted_returns.empty and not true_eval_returns.empty: # No forecast, but true values exist
            forecasted_returns = pd.DataFrame(index=true_eval_returns.index, columns=true_eval_returns.columns).fillna(0) # Or NaN
        elif not forecasted_returns.empty and true_eval_returns.empty: # Forecast, but no true values (e.g. eval period was empty)
            true_eval_returns = pd.DataFrame(index=forecasted_returns.index, columns=forecasted_returns.columns).fillna(0) # Or NaN


        return forecasted_returns, true_eval_returns


def calculate_forecast_errors(forecast_df, actual_df, metric='mse'):
    """
    Calculates forecast error between forecasted and actual values.

    Args:
        forecast_df (pandas.DataFrame): DataFrame of forecasted values.
        actual_df (pandas.DataFrame): DataFrame of actual values.
                                      Must have same shape and index/columns as forecast_df.
        metric (str): Error metric to use ('mse', 'mae', 'rmse').

    Returns:
        pandas.Series: Series of error scores, one for each column (cluster).
                       Returns empty Series if inputs are incompatible.
    """
    if forecast_df.empty or actual_df.empty or forecast_df.shape != actual_df.shape:
        # print("Warning: Forecast and actual DataFrames are incompatible for error calculation.")
        return pd.Series(dtype=float)

    # Ensure columns are aligned for subtraction
    common_cols = forecast_df.columns.intersection(actual_df.columns)
    if len(common_cols) == 0:
        return pd.Series(dtype=float)

    f_aligned = forecast_df[common_cols]
    a_aligned = actual_df[common_cols]

    errors = f_aligned - a_aligned

    if metric.lower() == 'mse':
        scores = (errors ** 2).mean()
    elif metric.lower() == 'mae':
        scores = errors.abs().mean()
    elif metric.lower() == 'rmse':
        scores = np.sqrt((errors ** 2).mean())
    else:
        raise ValueError(f"Unknown error metric: {metric}. Supported: 'mse', 'mae', 'rmse'.")

    return scores


def run_sliding_window_var_evaluation(
    asset_returns_df,
    initial_lookback_len,
    eval_len,
    n_clusters,
    cluster_method,
    var_order,
    num_windows,
    error_metric='mse'
):
    """
    Runs a sliding window evaluation of VAR forecasts for cluster returns.

    Args:
        asset_returns_df (pandas.DataFrame): Full historical asset returns.
        initial_lookback_len (int): Length of the first lookback window.
        eval_len (int): Length of the evaluation/forecast window.
        n_clusters (int): Number of clusters.
        cluster_method (str): Clustering method.
        var_order (int): VAR model order.
        num_windows (int): Number of sliding windows to evaluate.
        error_metric (str): Error metric for evaluation ('mse', 'mae', 'rmse').

    Returns:
        list: A list of pandas.Series, where each Series contains the error scores
              (one per cluster) for that window. List will be empty if no windows run.
    """
    all_window_errors = []

    forecaster = ClusterVARForecaster(
        n_clusters=n_clusters,
        cluster_method=cluster_method,
        var_order=var_order
    )

    for i in range(num_windows):
        lb_start = i * eval_len # Assumes non-overlapping lookback, or rather, lookback slides by eval_len
                                # More typical: lb_start = i * step_size, lookback_len is fixed.
                                # Let's assume lookback window slides by eval_len for simplicity here.
        lb_end = lb_start + initial_lookback_len # Lookback length is fixed

        if lb_end + eval_len > len(asset_returns_df):
            # print(f"Window {i+1}: Not enough data to form full lookback and evaluation periods. Stopping.")
            break

        print(f"Processing window {i+1}/{num_windows}...")

        forecast_df, actual_df = forecaster.process_step(
            asset_returns_df=asset_returns_df,
            lookback_indices=(lb_start, lb_end),
            eval_len=eval_len
        )

        if forecast_df.empty and actual_df.empty:
            print(f"  Window {i+1}: Processing step failed to produce results. Skipping error calculation.")
            # Optionally append None or a Series of NaNs to mark this window's failure
            # all_window_errors.append(pd.Series(dtype=float))
            continue

        # Handle cases where one is empty but not the other, if process_step doesn't fully align them
        if forecast_df.empty or actual_df.empty or forecast_df.shape[0] != actual_df.shape[0] or forecast_df.shape[1] != actual_df.shape[1]:
             print(f"  Window {i+1}: Forecast and actual data shapes mismatch post-processing. F:{forecast_df.shape}, A:{actual_df.shape}. Skipping error calculation.")
             # Attempt to align columns if shapes are otherwise compatible (same number of rows)
             if not forecast_df.empty and not actual_df.empty and forecast_df.shape[0] == actual_df.shape[0]:
                common_cols = forecast_df.columns.intersection(actual_df.columns)
                if common_cols.empty:
                    print(f"    No common columns for error calculation.")
                    continue
                forecast_df = forecast_df[common_cols]
                actual_df = actual_df[common_cols]
                if forecast_df.empty: # if common_cols was empty after all
                    print(f"    No common columns after intersection led to empty df.")
                    continue
             else: # Shapes are fundamentally incompatible
                continue


        window_errors = calculate_forecast_errors(forecast_df, actual_df, metric=error_metric)

        if not window_errors.empty:
            print(f"  Window {i+1} {error_metric.upper()}: {window_errors.mean():.4f} (avg across clusters)")
            all_window_errors.append(window_errors)
        else:
            print(f"  Window {i+1}: Error calculation resulted in empty scores.")

    return all_window_errors

