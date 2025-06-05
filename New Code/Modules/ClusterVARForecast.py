import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from signet.cluster import Cluster

class ClusterVARForecaster:
    def __init__(self, n_clusters, cluster_method, var_order, sigma_for_weights=0.01):
        self.n_clusters = max(1, int(n_clusters))
        self.cluster_method = cluster_method
        self.var_order = max(1, int(var_order))
        self.sigma_for_weights = max(float(sigma_for_weights), 1e-9) # Keep basic sanity
        self.corr_matrix_ = None
        self.cluster_definitions_ = None
        self.lookback_start_idx_ = None
        self.lookback_end_idx_ = None
        self.current_lookback_data_for_weights_ = None
        self.results = None
        self.lag_order_used = 0

    def _calculate_correlation_matrix(self, asset_returns_lookback_df):
        # Assumes asset_returns_lookback_df is valid and has >= 2 assets
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(asset_returns_lookback_df)
        normalized_df = pd.DataFrame(normalized_data, index=asset_returns_lookback_df.index, columns=asset_returns_lookback_df.columns)
        return normalized_df.corr(method='pearson').fillna(0) # Keep fillna for stability

    def _apply_clustering_algorithm(self, correlation_matrix_df, num_clusters_to_form):
        def _get_signet_data(corr_df):
            pos_corr = corr_df.applymap(lambda x: x if x >= 0 else 0)
            neg_corr = corr_df.applymap(lambda x: abs(x) if x < 0 else 0)
            return (sparse.csc_matrix(pos_corr.values), sparse.csc_matrix(neg_corr.values))

        num_assets = correlation_matrix_df.shape[0]
        effective_n_clusters = min(num_clusters_to_form, num_assets) # Assumes num_assets > 0

        if self.cluster_method == 'Kmeans':
            data_for_kmeans = correlation_matrix_df.fillna(0) # Keep fillna
            kmeans = KMeans(n_clusters=effective_n_clusters, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(data_for_kmeans)
        elif self.cluster_method in ['SPONGE', 'signed_laplacian', 'SPONGE_sym']:
            signet_data = _get_signet_data(correlation_matrix_df)
            cluster_obj = Cluster(signet_data)
            if self.cluster_method == 'SPONGE': labels = cluster_obj.SPONGE(effective_n_clusters)
            elif self.cluster_method == 'signed_laplacian': labels = cluster_obj.spectral_cluster_laplacian(effective_n_clusters)
            else: labels = cluster_obj.SPONGE_sym(effective_n_clusters)
        elif self.cluster_method == 'spectral_clustering':
            affinity_matrix = np.abs(correlation_matrix_df.fillna(0).values) # Keep fillna
            affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2
            np.fill_diagonal(affinity_matrix, 1)
            sc = SpectralClustering(n_clusters=effective_n_clusters, affinity='precomputed', random_state=0, assign_labels='kmeans')
            labels = sc.fit_predict(affinity_matrix)
        else: # This is an actual error, not an edge case for data.
            raise ValueError(f"Unknown clustering method: {self.cluster_method}")
        return labels

    def _define_clusters_and_centroids(self, asset_returns_lookback_df):
        self.corr_matrix_ = self._calculate_correlation_matrix(asset_returns_lookback_df)
        self.current_lookback_data_for_weights_ = asset_returns_lookback_df
        labels = self._apply_clustering_algorithm(self.corr_matrix_, self.n_clusters)

        asset_names = list(self.corr_matrix_.columns)
        labeled_assets = pd.DataFrame({'ClusterLabel': labels}, index=asset_names)

        unique_labels = np.unique(labels)
        cluster_definitions = {}
        for label_val in unique_labels:
            cluster_name = f'Cluster_{label_val + 1}'
            tickers_in_cluster = list(labeled_assets[labeled_assets['ClusterLabel'] == label_val].index)
            # Assumes tickers_in_cluster and valid_tickers_for_centroid are not empty
            valid_tickers_for_centroid = [t for t in tickers_in_cluster if t in asset_returns_lookback_df.columns]
            centroid_ts = asset_returns_lookback_df[valid_tickers_for_centroid].mean(axis=1)
            cluster_definitions[cluster_name] = {'tickers': tickers_in_cluster, 'centroid_ts': centroid_ts}
        self.cluster_definitions_ = cluster_definitions

    def _calculate_weighted_cluster_returns(self, asset_returns_df_slice, period_indices_on_slice):
        # Assumes self.cluster_definitions_, self.current_lookback_data_for_weights_ are set.
        # Assumes self.lookback_start_idx_, self.lookback_end_idx_ are set.
        start_idx_slice, end_idx_slice = period_indices_on_slice
        lookback_data_for_dist_calc = self.current_lookback_data_for_weights_.iloc[self.lookback_start_idx_:self.lookback_end_idx_]
        data_slice_current_period = asset_returns_df_slice.iloc[start_idx_slice:end_idx_slice]

        cluster_returns_dict = {}
        for cluster_name, info in self.cluster_definitions_.items():
            tickers_in_cluster = info['tickers']
            centroid_ts_for_dist_calc = info['centroid_ts'].iloc[self.lookback_start_idx_:self.lookback_end_idx_]

            asset_gaussian_weights = {}
            total_gaussian_weight_sum = 0.0
            for ticker in tickers_in_cluster:
                # Assumes ticker is in lookback_data_for_dist_calc.columns
                asset_returns_for_dist_calc = lookback_data_for_dist_calc[ticker]
                # Assumes lengths match and not empty
                squared_distance = np.sum((centroid_ts_for_dist_calc.values - asset_returns_for_dist_calc.values)**2)
                weight = np.exp(-squared_distance / (2 * (self.sigma_for_weights**2)))
                asset_gaussian_weights[ticker] = weight
                total_gaussian_weight_sum += weight

            current_cluster_period_returns = pd.Series(0.0, index=data_slice_current_period.index)
            # Assumes total_gaussian_weight_sum > 0
            for ticker, unnormalized_weight in asset_gaussian_weights.items():
                 # Assumes ticker is in data_slice_current_period.columns
                normalized_weight = unnormalized_weight / total_gaussian_weight_sum
                current_cluster_period_returns += data_slice_current_period[ticker] * normalized_weight
            cluster_returns_dict[cluster_name] = current_cluster_period_returns.values

        return pd.DataFrame(cluster_returns_dict, index=data_slice_current_period.index)

    def _fit(self, cluster_returns_df):
        whole_data = cluster_returns_df.astype(float).dropna(axis=1, how='all') # Keep dropna
        # Assumes whole_data is not empty and has enough rows for self.var_order
        data_for_var = whole_data.values
        model = VAR(data_for_var)
        self.results = model.fit(self.var_order) # Assumes fit succeeds
        self.lag_order_used = self.results.k_ar

    def _forecast(self, cluster_returns_df, forecast_horizon, cross_val):
        whole_data = cluster_returns_df.astype(float).dropna(axis=1, how='all') # Keep dropna
        output_columns = cluster_returns_df.columns
        num_forecast_steps = forecast_horizon - 1 # Assumes forecast_horizon > 0

        # Assumes whole_data is not empty and self.results is not None if not cross_val
        forecast_list = []

        for i in range(num_forecast_steps):
            current_results_iter = self.results
            current_lag_order_used_iter = self.lag_order_used
            slice_end_point_for_var_fit = len(whole_data) - (num_forecast_steps - i - current_lag_order_used_iter)
            # Assumes slice_end_point_for_var_fit > self.var_order
            data_for_var_model_fit_iter = whole_data.iloc[:slice_end_point_for_var_fit]

            if cross_val:
                # Assumes data_for_var_model_fit_iter is not empty and has enough rows
                model_cv = VAR(data_for_var_model_fit_iter.values)
                current_results_iter = model_cv.fit(self.var_order) # Assumes fit succeeds
                current_lag_order_used_iter = current_results_iter.k_ar

            # Assumes current_results_iter is not None and data_for_var_model_fit_iter has enough history
            forecast_input = data_for_var_model_fit_iter.values[-current_lag_order_used_iter:]
            forecast_result = current_results_iter.forecast(y=forecast_input, steps=1)
            forecast_list.append(forecast_result) # Assumes forecast succeeds

        # Assumes forecast_list is not empty
        forecast_array = np.concatenate(forecast_list, axis=0)
        forecast_index = pd.RangeIndex(start=0, stop=len(forecast_array), step=1)
        forecast_df = pd.DataFrame(forecast_array, columns=whole_data.columns, index=forecast_index)
        return forecast_df.reindex(columns=output_columns, fill_value=np.nan) # Keep reindex for consistent output

class NaiveVARForecaster:
    def __init__(self, var_order):
        self.var_order = max(1, int(var_order))
        self.results = None
        self.lag_order_used = 0

    def _fit(self, asset_returns_df):
        whole_data = asset_returns_df.astype(float).dropna(axis=1, how='all')
        data_for_var = whole_data.values
        model = VAR(data_for_var)
        self.results = model.fit(self.var_order)
        self.lag_order_used = self.results.k_ar

    def _forecast(self, asset_returns_df, forecast_horizon, cross_val):
        whole_data = asset_returns_df.astype(float).dropna(axis=1, how='all')
        output_columns = asset_returns_df.columns
        num_forecast_steps = forecast_horizon - 1

        forecast_list = []
        data_for_var_model_fit_outer = whole_data

        for i in range(num_forecast_steps):
            current_results_iter = self.results
            current_lag_order_used_iter = self.lag_order_used
            slice_end_point_for_var_fit = len(whole_data) - (num_forecast_steps - i - current_lag_order_used_iter)
            data_for_var_model_fit_iter = whole_data.iloc[:slice_end_point_for_var_fit]

            if cross_val:
                model_cv = VAR(data_for_var_model_fit_iter.values)
                current_results_iter = model_cv.fit(self.var_order)
                current_lag_order_used_iter = current_results_iter.k_ar

            forecast_input = data_for_var_model_fit_iter.values[-current_lag_order_used_iter:]
            forecast_list.append(current_results_iter.forecast(y=forecast_input, steps=1))


        forecast_array = np.concatenate(forecast_list, axis=0)
        forecast_index = pd.RangeIndex(start=0, stop=len(forecast_array), step=1)
        forecast_df = pd.DataFrame(forecast_array, columns=whole_data.columns, index=forecast_index)
        return forecast_df.reindex(columns=output_columns, fill_value=np.nan)
