import numpy as np
import pandas as pd
import sys
from scipy import sparse
import warnings
from sklearn.cluster import KMeans, SpectralClustering

# Note: The Signet import and installation block remains the same as your previous code.
# I'll omit it here for brevity but assume it's present at the top of the file.
# try:
#     from signet.cluster import Cluster
# except ImportError:
# ... (Signet import/install logic) ...

warnings.filterwarnings('ignore')

class ClusterVARForecaster:
    """
    Performs clustering on asset returns, fits a VAR model to cluster returns,
    forecasts future cluster returns, and calculates forecast errors.
    Uses Gaussian weighting for assets within clusters.
    """
    def __init__(self, n_clusters, cluster_method, var_order, sigma_for_weights): # Added sigma_for_weights
        """
        Initializes the ClusterVARForecaster.

        Args:
            n_clusters (int): Number of clusters to form.
            cluster_method (str): Method for clustering assets.
            var_order (int): Order of the VAR model.
            sigma_for_weights (float): Sigma parameter for Gaussian weighting within clusters.
        """
        self.n_clusters = max(1, int(n_clusters))
        self.cluster_method = cluster_method
        self.var_order = max(1, int(var_order))
        self.sigma_for_weights = sigma_for_weights # Store sigma
        if self.sigma_for_weights <= 0:
            raise ValueError("sigma_for_weights must be positive.")

        self.corr_matrix_ = None
        # self.cluster_definitions_ will store {'Cluster_Name': {'tickers': [...], 'centroid_ts': np.array}}
        self.cluster_definitions_ = None
        # self.intra_cluster_asset_weights_ will store {'Cluster_Name': {tickerA: weightA, ...}}
        self.intra_cluster_asset_weights_ = None


    def _calculate_correlation_matrix(self, asset_returns_lookback_df):
        # This method remains the same as your previous version
        if asset_returns_lookback_df.empty:
            return pd.DataFrame()
        mean_vals = asset_returns_lookback_df.mean(axis=0)
        std_vals = asset_returns_lookback_df.std(axis=0)
        std_vals[std_vals < 1e-9] = 1.0
        normalized_data = (asset_returns_lookback_df - mean_vals) / std_vals
        correlation_matrix = normalized_data.corr(method='pearson').fillna(0)
        return correlation_matrix

    def _apply_clustering_algorithm(self, correlation_matrix_df, num_clusters_to_form):
        # This method remains the same as your previous version
        def _get_signet_data(corr_df):
            pos_corr = corr_df.applymap(lambda x: x if x >= 0 else 0)
            neg_corr = corr_df.applymap(lambda x: abs(x) if x < 0 else 0)
            return (sparse.csc_matrix(pos_corr.values), sparse.csc_matrix(neg_corr.values))

        if correlation_matrix_df.empty: return np.array([])
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
            except Exception:
                data_for_kmeans = correlation_matrix_df.fillna(0)
                kmeans = KMeans(n_clusters=effective_n_clusters, random_state=0, n_init='auto')
                labels = kmeans.fit_predict(data_for_kmeans)
        else:
            raise ValueError(f"Unknown clustering method: {self.cluster_method}")

        if not isinstance(labels, np.ndarray) or labels.size == 0:
            if num_assets > 0: return np.zeros(num_assets, dtype=int)
            return np.array([])
        return labels

    def _define_clusters_and_weights(self, asset_returns_lookback_df): # Renamed and expanded
        """
        Performs clustering, defines cluster compositions (tickers and centroids),
        and calculates intra-cluster Gaussian weights based on the lookback data.
        """
        self.corr_matrix_ = self._calculate_correlation_matrix(asset_returns_lookback_df)

        num_assets = self.corr_matrix_.shape[0]
        if num_assets == 0 :
            self.cluster_definitions_ = {}
            self.intra_cluster_asset_weights_ = {}
            return

        labels = self._apply_clustering_algorithm(self.corr_matrix_, self.n_clusters)

        if labels.size == 0 and num_assets > 0 :
             labels = np.zeros(num_assets, dtype=int)
        elif labels.size == 0 and num_assets == 0:
            self.cluster_definitions_ = {}
            self.intra_cluster_asset_weights_ = {}
            return

        asset_names = list(self.corr_matrix_.columns)
        labeled_assets = pd.DataFrame({'ClusterLabel': labels}, index=asset_names)

        actual_n_clusters = len(np.unique(labels))
        self.n_clusters = actual_n_clusters # Update n_clusters if clustering changes it

        cluster_definitions_temp = {}
        intra_cluster_weights_temp = {}

        for label_val in np.unique(labels):
            cluster_name = f'Cluster_{label_val + 1}'
            tickers_in_cluster = list(labeled_assets[labeled_assets['ClusterLabel'] == label_val].index)

            if not tickers_in_cluster:
                cluster_definitions_temp[cluster_name] = {'tickers': [], 'centroid_ts': np.array([])}
                intra_cluster_weights_temp[cluster_name] = {}
                continue

            valid_tickers_for_centroid = [t for t in tickers_in_cluster if t in asset_returns_lookback_df.columns]

            if not valid_tickers_for_centroid:
                centroid_ts = np.zeros(len(asset_returns_lookback_df)) # Or handle as appropriate
                cluster_definitions_temp[cluster_name] = {'tickers': tickers_in_cluster, 'centroid_ts': centroid_ts}
                intra_cluster_weights_temp[cluster_name] = {} # No valid tickers to weight
                continue

            centroid_ts = asset_returns_lookback_df[valid_tickers_for_centroid].mean(axis=1).values
            cluster_definitions_temp[cluster_name] = {'tickers': tickers_in_cluster, 'centroid_ts': centroid_ts}

            # Calculate Gaussian weights for this cluster
            current_cluster_asset_weights = {}
            exp_decay_values = []
            asset_series_for_weights = []

            for ticker in tickers_in_cluster: # Use all assigned tickers for weight calc if they exist in data
                if ticker in asset_returns_lookback_df.columns:
                    asset_return_series_np = asset_returns_lookback_df[ticker].values
                    if centroid_ts.shape == asset_return_series_np.shape:
                        squared_distance = np.sum((centroid_ts - asset_return_series_np)**2)
                        exp_decay = np.exp(-squared_distance / (2 * (self.sigma_for_weights**2)))
                        exp_decay_values.append(exp_decay)
                        asset_series_for_weights.append(ticker) # Track tickers for which weights are calculated
                    # Else: shape mismatch, ticker won't get a weight contribution here

            if exp_decay_values:
                sum_of_exp_decay = np.sum(exp_decay_values)
                if sum_of_exp_decay > 1e-9:
                    normalized_weights = np.array(exp_decay_values) / sum_of_exp_decay
                    for ticker, weight in zip(asset_series_for_weights, normalized_weights):
                        current_cluster_asset_weights[ticker] = weight
                else: # Fallback to equal weights among these tickers if sum is too small
                    num_w_tickers = len(asset_series_for_weights)
                    for ticker in asset_series_for_weights:
                        current_cluster_asset_weights[ticker] = 1.0 / num_w_tickers if num_w_tickers > 0 else 0.0

            intra_cluster_weights_temp[cluster_name] = current_cluster_asset_weights

        self.cluster_definitions_ = cluster_definitions_temp
        self.intra_cluster_asset_weights_ = intra_cluster_weights_temp


    def _calculate_weighted_cluster_returns(self, asset_returns_df, period_indices): # RENAMED
        """
        Calculates Gaussian-weighted returns for each defined cluster over a given period,
        using pre-calculated intra-cluster asset weights.
        """
        if not self.intra_cluster_asset_weights_: # Check if weights were calculated
            # print("Warning: Intra-cluster asset weights not available. Cannot calculate cluster returns.")
            # Try to get column names from cluster_definitions_ if it exists
            cols = list(self.cluster_definitions_.keys()) if self.cluster_definitions_ else []
            idx_len = period_indices[1] - period_indices[0] if period_indices[1] > period_indices[0] else 0
            idx = asset_returns_df.index[period_indices[0]:period_indices[1]] if idx_len > 0 else pd.Index([])
            return pd.DataFrame(np.zeros((len(idx), len(cols))), index=idx, columns=cols)


        start_idx, end_idx = period_indices
        if not (0 <= start_idx < end_idx <= len(asset_returns_df)):
            idx_len = period_indices[1] - period_indices[0] if period_indices[1] > period_indices[0] else 0
            idx = asset_returns_df.index[start_idx:end_idx] if idx_len > 0 else pd.Index([])
            return pd.DataFrame(index=idx, columns=list(self.intra_cluster_asset_weights_.keys()))

        data_slice = asset_returns_df.iloc[start_idx:end_idx]

        cluster_returns_dict = {}
        for cluster_name, ticker_weights_map in self.intra_cluster_asset_weights_.items():
            cluster_return_series = np.zeros(len(data_slice))
            if not ticker_weights_map: # No weighted assets for this cluster
                cluster_returns_dict[cluster_name] = cluster_return_series
                continue

            for ticker, weight in ticker_weights_map.items():
                if ticker in data_slice.columns:
                    cluster_return_series += data_slice[ticker].values * weight
            cluster_returns_dict[cluster_name] = cluster_return_series

        if not cluster_returns_dict:
            return pd.DataFrame(index=data_slice.index)

        return pd.DataFrame(cluster_returns_dict, index=data_slice.index)

    def _fit_var_and_forecast(self, lookback_cluster_returns_df, forecast_horizon):
        # This method remains the same as your previous version
        data_for_var = lookback_cluster_returns_df.astype(float).dropna(axis=1, how='all')
        if data_for_var.empty or data_for_var.shape[0] <= self.var_order or data_for_var.shape[1] == 0:
            return pd.DataFrame(columns=lookback_cluster_returns_df.columns if not lookback_cluster_returns_df.empty else ["dummy_var_col"])

        num_obs, num_series = data_for_var.shape
        Y_matrix = data_for_var.iloc[self.var_order:].values
        X_regressors_list = [np.ones((num_obs - self.var_order, 1))]
        for lag in range(1, self.var_order + 1):
            X_regressors_list.append(data_for_var.iloc[self.var_order - lag : num_obs - lag].values)
        X_matrix = np.hstack(X_regressors_list)

        if X_matrix.shape[0] != Y_matrix.shape[0] or X_matrix.shape[0] == 0 :
             return pd.DataFrame(columns=data_for_var.columns)

        try:
            coefficients = np.linalg.solve(X_matrix.T @ X_matrix, X_matrix.T @ Y_matrix)
        except np.linalg.LinAlgError:
            try:
                coefficients = np.linalg.pinv(X_matrix.T @ X_matrix) @ (X_matrix.T @ Y_matrix)
            except np.linalg.LinAlgError:
                return pd.DataFrame(columns=data_for_var.columns)

        forecasts_array = np.zeros((forecast_horizon, num_series))
        history_for_fcst = data_for_var.iloc[num_obs - self.var_order : num_obs].values

        for i in range(forecast_horizon):
            lagged_vals_flat = history_for_fcst[::-1].ravel()
            current_X_fcst = np.hstack(([1.0], lagged_vals_flat))
            next_forecast = current_X_fcst @ coefficients
            forecasts_array[i, :] = next_forecast
            history_for_fcst = np.vstack((history_for_fcst[1:], next_forecast))

        forecast_index = pd.RangeIndex(start=0, stop=forecast_horizon, step=1)
        if not data_for_var.empty and isinstance(data_for_var.index, pd.DatetimeIndex) and data_for_var.index.freq is not None:
            try:
                forecast_index = pd.date_range(start=data_for_var.index[-1] + data_for_var.index.freq,
                                            periods=forecast_horizon, freq=data_for_var.index.freq)
            except Exception:
                 pass

        forecast_df = pd.DataFrame(forecasts_array, columns=data_for_var.columns, index=forecast_index)
        return forecast_df.reindex(columns=lookback_cluster_returns_df.columns, fill_value=0.0)


    def process_step(self, asset_returns_df, lookback_indices, eval_len):
        """
        Processes one step of clustering, VAR fitting, forecasting, and true value calculation.
        """
        lb_start, lb_end = lookback_indices

        lookback_asset_returns = asset_returns_df.iloc[lb_start:lb_end]
        self._define_clusters_and_weights(lookback_asset_returns) # MODIFIED CALL

        if not self.intra_cluster_asset_weights_: # Check if weights were successfully defined
            empty_df = pd.DataFrame()
            return empty_df, empty_df

        lookback_cluster_returns = self._calculate_weighted_cluster_returns(asset_returns_df, lookback_indices) # MODIFIED CALL
        if lookback_cluster_returns.empty or lookback_cluster_returns.isnull().all().all():
            empty_df = pd.DataFrame(columns=list(self.intra_cluster_asset_weights_.keys()))
            return empty_df, empty_df.copy()

        forecasted_returns = self._fit_var_and_forecast(lookback_cluster_returns, eval_len)

        eval_start_idx = lb_end
        eval_end_idx = lb_end + eval_len
        eval_end_idx = min(eval_end_idx, len(asset_returns_df))
        actual_eval_len = eval_end_idx - eval_start_idx

        if actual_eval_len <= 0:
            true_eval_returns = pd.DataFrame(columns=forecasted_returns.columns if not forecasted_returns.empty else list(self.intra_cluster_asset_weights_.keys()))
            return forecasted_returns.head(0) if not forecasted_returns.empty else pd.DataFrame(columns=list(self.intra_cluster_asset_weights_.keys())), true_eval_returns

        true_eval_returns = self._calculate_weighted_cluster_returns(asset_returns_df, # MODIFIED CALL
                                                                     (eval_start_idx, eval_end_idx))

        # Alignment logic (same as before, ensure it handles potentially different column sets robustly)
        if not forecasted_returns.empty and not true_eval_returns.empty:
            if len(forecasted_returns) == len(true_eval_returns):
                 forecasted_returns.index = true_eval_returns.index
            else:
                  if len(forecasted_returns) > len(true_eval_returns):
                      forecasted_returns = forecasted_returns.iloc[:len(true_eval_returns)]
                      if not true_eval_returns.empty : forecasted_returns.index = true_eval_returns.index


        if not forecasted_returns.empty and not true_eval_returns.empty:
            all_cols = forecasted_returns.columns.union(true_eval_returns.columns)
            # Ensure true_eval_returns has an index before reindexing forecast
            ref_index = true_eval_returns.index if not true_eval_returns.empty else forecasted_returns.index

            forecasted_returns = forecasted_returns.reindex(columns=all_cols, index=ref_index).fillna(0)
            true_eval_returns = true_eval_returns.reindex(columns=all_cols, index=ref_index).fillna(0)

        elif forecasted_returns.empty and not true_eval_returns.empty:
            forecasted_returns = pd.DataFrame(0, index=true_eval_returns.index, columns=true_eval_returns.columns)
        elif not forecasted_returns.empty and true_eval_returns.empty:
             # This implies actual_eval_len was likely > 0 but _calculate_weighted_cluster_returns returned empty
             # e.g. no valid tickers in data_slice for any cluster weights
            true_eval_returns = pd.DataFrame(0, index=forecasted_returns.index, columns=forecasted_returns.columns)

        return forecasted_returns, true_eval_returns

# calculate_forecast_errors function remains the same

def run_sliding_window_var_evaluation(
    asset_returns_df,
    initial_lookback_len,
    eval_len,
    n_clusters,
    cluster_method,
    var_order,
    sigma_for_weights, # Added
    num_windows,
    error_metric='mse'
):
    """
    Runs a sliding window evaluation of VAR forecasts for cluster returns.
    """
    all_window_errors = []
    all_forecasts = []
    all_actuals = []

    forecaster = ClusterVARForecaster(
        n_clusters=n_clusters,
        cluster_method=cluster_method,
        var_order=var_order,
        sigma_for_weights=sigma_for_weights # Pass sigma
    )

    # ... (rest of the function is the same as your previous version) ...
    for i in range(num_windows):
        lb_start = i * eval_len
        lb_end = lb_start + initial_lookback_len

        if lb_end + eval_len > len(asset_returns_df):
            print(f"Window {i+1}: Not enough data. Stopping.")
            break

        print(f"Processing window {i+1}/{num_windows}...")

        forecast_df, actual_df = forecaster.process_step(
            asset_returns_df=asset_returns_df,
            lookback_indices=(lb_start, lb_end),
            eval_len=eval_len
        )

        all_forecasts.append(forecast_df)
        all_actuals.append(actual_df)

        if forecast_df.empty and actual_df.empty:
            print(f"  Window {i+1}: Processing step failed. Skipping error calculation.")
            all_window_errors.append(pd.Series(dtype=float, name=f"Window_{i+1}_Errors"))
            continue

        if forecast_df.shape[0] != actual_df.shape[0] or forecast_df.shape[1] == 0 or actual_df.shape[1] == 0:
             print(f"  Window {i+1}: Forecast ({forecast_df.shape}) and actual ({actual_df.shape}) data shapes are incompatible. Skipping error calculation.")
             all_window_errors.append(pd.Series(dtype=float, name=f"Window_{i+1}_Errors"))
             continue

        # Use the existing calculate_forecast_errors function
        window_errors = calculate_forecast_errors(forecast_df, actual_df, metric=error_metric)
        window_errors.name = f"Window_{i+1}_Errors" # Name the series for better DataFrame column name later

        if not window_errors.empty:
            print(f"  Window {i+1} {error_metric.upper()}: {window_errors.mean():.6f} (avg across clusters)")
            all_window_errors.append(window_errors)
        else:
            print(f"  Window {i+1}: Error calculation resulted in empty scores.")
            all_window_errors.append(pd.Series(dtype=float, name=f"Window_{i+1}_Errors")) # Append empty series if errors were empty

    return all_window_errors, all_forecasts, all_actuals