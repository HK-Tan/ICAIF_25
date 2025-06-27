import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from signet.cluster import Cluster
from sktime.forecasting.var_reduce import VARReduce
from statsmodels.tsa.api import VAR
from cpp_rls_filter import CppExpL1L2Regression

# class ExpL1L2Regression:

#     def __init__(
#         self,
#         n: int,
#         w: np.ndarray,
#         lam: float = 0.1,
#         halflife: float = 20.0,
#         gamma: float = 0.01,
#         epsilon: float = 1e-6,
#     ):
#         self.n = n
#         self.lam = lam
#         self.gamma = gamma
#         self.epsilon = epsilon
#         self.beta = np.exp(np.log(0.5) / halflife)
#         self.w = w
#         self.P = np.diag(np.ones(self.n) * self.lam)

#     def update(self, x: np.ndarray, y: float) -> None:
#         r = 1 + (x.T @ self.P @ x) / self.beta
#         k = (self.P @ x) / (r * self.beta)
#         e = y - x @ self.w

#         k = k.reshape(-1, 1)

#         extra = (
#             self.gamma * ((self.beta - 1) / self.beta)
#             * (np.eye(self.n) - k @ self.w.reshape(1, -1))
#             @ self.P @ (np.sign(self.w) / (np.abs(self.w) + self.epsilon))
#         )

#         self.w = self.w + k.flatten() * e + extra

#         self.P = self.P / self.beta - (k @ k.T) * r

#     def predict(self, x: np.ndarray) -> float:
#         return self.w @ x

class NaiveVARForecaster:
    """
    Fits a Vector Autoregression (VAR) model directly on asset returns.
    """
    def __init__(self, var_order):
        self.var_order = max(1, int(var_order))
        self.results = None
        self.lag_order_used = 0

    def _fit(self, asset_returns_df):
        """
        Fits the VAR model on the provided dataframe.
        """
        whole_data = asset_returns_df.astype(float).dropna(axis=1, how='all')
        data_for_var = whole_data.values  # df -> np.ndarray
        model = VAR(data_for_var)
        self.results = model.fit(self.var_order)
        self.lag_order_used = self.var_order

    def _forecast(self, asset_returns_df, forecast_horizon, cross_val):
        """
        This forecast function should be inherited to perform forecasting for any VAR model.
        I feel like (think about if this is true) that the _forecast method should be the same for all VAR models (including ClusterVAR)
        """
        whole_data = asset_returns_df.astype(float).dropna(axis=1, how='all')
        output_columns = asset_returns_df.columns # stores column names for output
        num_forecast_steps = forecast_horizon - 1

        forecast_list = []

        ###################################################################################
        ## See comments below -> Don't we need to update and do a "rolling window" fit? (yes, I think it's implemented)
        ###################################################################################

        for i in range(num_forecast_steps):
            # I think we forgot to "refit the data"
            current_results_iter = self.results # Fitted model instance
            current_lag_order_used_iter = self.lag_order_used
            slice_end_point_for_var_fit = len(whole_data) - (num_forecast_steps - i - current_lag_order_used_iter)
            data_for_var_model_fit_iter = whole_data.iloc[:slice_end_point_for_var_fit]
            forecast_input = data_for_var_model_fit_iter.values[-current_lag_order_used_iter:]
            forecast_list.append(current_results_iter.forecast(y=forecast_input, steps=1))


        forecast_array = np.concatenate(forecast_list, axis=0)
        forecast_index = pd.RangeIndex(start=0, stop=len(forecast_array), step=1)
        forecast_df = pd.DataFrame(forecast_array, columns=whole_data.columns, index=forecast_index)
        return forecast_df.reindex(columns=output_columns, fill_value=np.nan)

class ClusterVARForecaster(NaiveVARForecaster):
    """
    Groups assets into clusters and fits a VAR model on the cluster returns.
    Inherits from NaiveVARForecaster to reuse the core VAR fitting mechanism.
    """
    def __init__(self, n_clusters, cluster_method, var_order, sigma_for_weights=0.01):
        super().__init__(var_order)
        self.n_clusters = max(1, int(n_clusters))
        self.cluster_method = cluster_method
        self.current_lookback_data_for_weights_ = None
        self.sigma_for_weights = max(float(sigma_for_weights), 1e-9)
        self.corr_matrix_ = None
        self.cluster_definitions_ = None

    def _calculate_correlation_matrix(self, asset_returns_lookback_df):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(asset_returns_lookback_df)
        normalized_df = pd.DataFrame(normalized_data, index=asset_returns_lookback_df.index, columns=asset_returns_lookback_df.columns)
        return normalized_df.corr(method='pearson').fillna(0)

    def convert_sktime_var_coeffs_to_statsmodels(self, sktime_model, series_names=None):
        sk_coeffs = sktime_model.coefficients_
        sk_intercept = sktime_model.intercept_
        lags, k, _ = sk_coeffs.shape
        swapped_coeffs = sk_coeffs.swapaxes(1, 2)
        sm_lag_coeffs = swapped_coeffs.reshape(lags * k, k)
        sm_full_coeffs = np.vstack([sk_intercept, sm_lag_coeffs])
        if series_names is None:
            series_names = [f'Var{i+1}' for i in range(k)]
        regressor_index = ['const']
        for i in range(1, lags + 1):
            for var_name in series_names:
                regressor_index.append(f'L{i}.{var_name}')
        params_df = pd.DataFrame(sm_full_coeffs, index=regressor_index, columns=series_names)
        return params_df

    def _apply_clustering_algorithm(self, correlation_matrix_df, num_clusters_to_form):
        def _get_signet_data(corr_df):
            pos_corr = corr_df.applymap(lambda x: x if x >= 0 else 0)
            neg_corr = corr_df.applymap(lambda x: abs(x) if x < 0 else 0)
            return (sparse.csc_matrix(pos_corr.values), sparse.csc_matrix(neg_corr.values))
        num_assets = correlation_matrix_df.shape[0]
        effective_n_clusters = min(num_clusters_to_form, num_assets)
        if self.cluster_method == 'Kmeans':
            kmeans = KMeans(n_clusters=effective_n_clusters, random_state=0, n_init='auto')
            labels = kmeans.fit_predict(correlation_matrix_df.fillna(0))
        elif self.cluster_method in ['SPONGE', 'signed_laplacian', 'SPONGE_sym']:
            signet_data = _get_signet_data(correlation_matrix_df)
            cluster_obj = Cluster(signet_data)
            if self.cluster_method == 'SPONGE': labels = cluster_obj.SPONGE(effective_n_clusters)
            elif self.cluster_method == 'signed_laplacian': labels = cluster_obj.spectral_cluster_laplacian(effective_n_clusters)
            else: labels = cluster_obj.SPONGE_sym(effective_n_clusters)
        elif self.cluster_method == 'spectral_clustering':
            affinity_matrix = np.abs(correlation_matrix_df.fillna(0).values)
            affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2
            np.fill_diagonal(affinity_matrix, 1)
            sc = SpectralClustering(n_clusters=effective_n_clusters, affinity='precomputed', random_state=0, assign_labels='kmeans')
            labels = sc.fit_predict(affinity_matrix)
        else:
            raise ValueError(f"Unknown clustering method: {self.cluster_method}")
        return labels

    # =========================================================================
    # === REVISED AND MORE ROBUST VECTORIZED FUNCTION =========================
    # =========================================================================
    def _define_clusters_and_centroids(self, asset_returns_lookback_df):
        """
        Vectorized version using positional groupby to define clusters and centroids.
        This is more robust against potential index/label mismatches.
        """
        self.corr_matrix_ = self._calculate_correlation_matrix(asset_returns_lookback_df)
        self.current_lookback_data_for_weights_ = asset_returns_lookback_df

        # labels is a NumPy array whose order corresponds to the columns of self.corr_matrix_
        labels = self._apply_clustering_algorithm(self.corr_matrix_, self.n_clusters)

        # 1. Calculate all centroid time series in one operation.
        # We group the returns DataFrame's COLUMNS by the raw `labels` array.
        # Pandas groups positionally: the 1st column is grouped by the 1st label, etc.
        # This is safe because the column order of asset_returns_lookback_df was preserved
        # when creating the correlation matrix and thus the labels.
        all_centroids_df = asset_returns_lookback_df.groupby(by=labels, axis=1).mean()

        # 2. Get all ticker lists for each cluster.
        # To do this, we create an explicit mapping between tickers and their labels.
        asset_names = self.corr_matrix_.columns
        ticker_cluster_map = pd.DataFrame({'Ticker': asset_names, 'ClusterLabel': labels})
        tickers_by_cluster = ticker_cluster_map.groupby('ClusterLabel')['Ticker'].apply(list)

        # 3. Assemble the final dictionary using a comprehension.
        # The columns of `all_centroids_df` are now the unique cluster labels (e.g., 0, 1, 2...).
        self.cluster_definitions_ = {
            label: {
                'tickers': tickers_by_cluster[label],
                'centroid_ts': all_centroids_df[label]
            }
            for label in all_centroids_df.columns
        }
    # =========================================================================
    # === END OF REVISED FUNCTION =============================================
    # =========================================================================

    def _calculate_weighted_cluster_returns(self, asset_returns_df_slice, period_indices_on_slice):
        start_idx_slice, end_idx_slice = period_indices_on_slice
        lookback_data_for_dist_calc = self.current_lookback_data_for_weights_
        data_slice_current_period = asset_returns_df_slice.iloc[start_idx_slice:end_idx_slice]
        cluster_returns_dict = {}
        for cluster_name, info in self.cluster_definitions_.items():
            tickers_in_cluster = info['tickers']
            if not tickers_in_cluster or not all(t in lookback_data_for_dist_calc.columns for t in tickers_in_cluster):
                continue
            centroid_ts_for_dist_calc = info['centroid_ts'].iloc[self.lookback_start_idx_:self.lookback_end_idx_]
            asset_returns_for_dist_calc = lookback_data_for_dist_calc[tickers_in_cluster]
            squared_distances = (asset_returns_for_dist_calc.subtract(centroid_ts_for_dist_calc, axis=0)**2).sum(axis=0)
            epsilon = 1e-12
            exponent_vals = -squared_distances / (2 * (self.sigma_for_weights**2 + epsilon))
            stable_exponent_vals = exponent_vals - exponent_vals.max()
            unnormalized_weights = np.exp(stable_exponent_vals)
            total_weight_sum = unnormalized_weights.sum()
            normalized_weights = unnormalized_weights / total_weight_sum
            returns_to_weight = data_slice_current_period[tickers_in_cluster]
            current_cluster_period_returns = np.log(1+((np.exp(returns_to_weight)-1) * normalized_weights).sum(axis=1))
            cluster_returns_dict[cluster_name] = current_cluster_period_returns
        return pd.DataFrame(cluster_returns_dict, index=data_slice_current_period.index)

    def _fit(self, returns_df):
        whole_data = returns_df.astype(float).dropna(axis=1, how='all')
        data_for_var = whole_data.values
        self.results = VARReduce(self.var_order, regressor=ElasticNet())
        self.results.fit(data_for_var)
        self.lag_order_used = self.var_order

    def _forecast(self, returns_df, forecast_horizon, cross_val):
        whole_data = returns_df.astype(float).dropna(axis=1, how='all')
        output_columns = returns_df.columns
        num_forecast_steps = forecast_horizon - 1
        lag_order = self.var_order
        forecast_list = []
        if cross_val:
            initial_model = VARReduce(lags=lag_order, regressor=ElasticNet())
            initial_model.fit(whole_data[:-num_forecast_steps])
            forecast_start_idx = len(whole_data) - forecast_horizon - lag_order
            X_lags = np.hstack([whole_data.values[lag_order - i : -i or None, :] for i in range(1, lag_order+1)])
            num_rows = X_lags.shape[0]
            ones_column = np.ones((num_rows, 1))
            full_regressor_matrix = np.hstack([ones_column, X_lags])
            model = initial_model
        else:
            X_lags = np.hstack([whole_data.values[lag_order - i : -i or None, :] for i in range(1, lag_order+1)])
            num_rows = X_lags.shape[0]
            ones_column = np.ones((num_rows, 1))
            full_regressor_matrix = np.hstack([ones_column, X_lags])
            forecast_start_idx = 0
            model = self.results
        params_matrix = self.convert_sktime_var_coeffs_to_statsmodels(model)
        n_rls_inputs = params_matrix.shape[0]
        rls_filter = CppExpL1L2Regression(
            initial_w=params_matrix.T,
            n_features=n_rls_inputs,
            halflife=2000,
            lam=0.5,
            gamma=0.5,
        )
        for t in range(forecast_start_idx, forecast_start_idx + num_forecast_steps - self.lag_order_used):
            input_x = full_regressor_matrix[t]
            forecasts_at_t = rls_filter.predict(input_x)
            forecast_list.append(forecasts_at_t)
            target_d_vector = whole_data.iloc[t].values
            rls_filter.update(input_x, target_d_vector)
        forecast_array = np.array(forecast_list)
        forecast_index = pd.RangeIndex(start=0, stop=len(forecast_array), step=1)
        forecast_df = pd.DataFrame(forecast_array, columns=whole_data.columns, index=forecast_index)
        return forecast_df.reindex(columns=output_columns, fill_value=np.nan)