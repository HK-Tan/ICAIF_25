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
        # Initialize the parent class with the VAR order
        super().__init__(var_order)

        # Initialize clustering-specific parameters
        self.n_clusters = max(1, int(n_clusters))
        self.cluster_method = cluster_method
        self.current_lookback_data_for_weights_ = None
        self.sigma_for_weights = max(float(sigma_for_weights), 1e-9)

        # Attributes to store clustering results
        self.corr_matrix_ = None
        self.cluster_definitions_ = None # Will store tickers, centroids, and weights

    # --- Helper methods (from original implementation) ---

    def _calculate_correlation_matrix(self, asset_returns_lookback_df):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(asset_returns_lookback_df)
        normalized_df = pd.DataFrame(normalized_data, index=asset_returns_lookback_df.index, columns=asset_returns_lookback_df.columns)
        # normalized_df step converts numpy -> pandas DataFrame
        return normalized_df.corr(method='pearson').fillna(0)

    def convert_sktime_var_coeffs_to_statsmodels(self, sktime_model, series_names=None):
        """
        Converts coefficients from a fitted sktime VAR/VARReduce model
        to the format used by statsmodels VAR.

        Parameters
        ----------
        sktime_model : sktime.forecasting.var.VAR
            A fitted sktime VAR or VARReduce model instance.
        series_names : list of str, optional
            Names of the time series variables, used for creating the
            final DataFrame index and columns.

        Returns
        -------
        pandas.DataFrame
            A DataFrame with coefficients in the statsmodels format,
            having shape (lags*k + 1, k).
        """
        # 1. Extract coefficients and dimensions from the sktime model
        sk_coeffs = sktime_model.coefficients_
        sk_intercept = sktime_model.intercept_
        lags, k, _ = sk_coeffs.shape

        # 2. Perform the "swap and reshape" for the lag coefficients
        # Original axes: (lag, target_var, regressor_var)
        # Swap to: (lag, regressor_var, target_var)
        swapped_coeffs = sk_coeffs.swapaxes(1, 2)
        # Reshape to combine lag and regressor_var into a single dimension
        # New shape: (lags * k, k)
        sm_lag_coeffs = swapped_coeffs.reshape(lags * k, k)

        # 3. Stack the intercepts on top of the lag coefficients
        # Final shape: (lags * k + 1, k)
        sm_full_coeffs = np.vstack([sk_intercept, sm_lag_coeffs])

        # 4. (Optional but recommended) Create a pandas DataFrame with proper labels
        if series_names is None:
            series_names = [f'Var{i+1}' for i in range(k)]

        # Create the index for the regressors
        regressor_index = ['const']
        for i in range(1, lags + 1):
            for var_name in series_names:
                regressor_index.append(f'L{i}.{var_name}')

        # Create the final DataFrame
        params_df = pd.DataFrame(
            sm_full_coeffs,
            index=regressor_index,
            columns=series_names
        )
        return params_df

    # --- cluster return logic ---

    def _apply_clustering_algorithm(self, correlation_matrix_df, num_clusters_to_form):
        """
        Applies the specified clustering algorithm to the correlation matrix.
        Args:
            correlation_matrix_df (pd.DataFrame): The correlation matrix of asset returns.
            num_clusters_to_form (int): The number of clusters to form.
        Returns:
            labels (np.ndarray): Cluster labels for each asset.
        """
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
            valid_tickers_for_centroid = [t for t in tickers_in_cluster if t in asset_returns_lookback_df.columns]
            centroid_ts = asset_returns_lookback_df[valid_tickers_for_centroid].mean(axis=1)
            cluster_definitions[cluster_name] = {'tickers': tickers_in_cluster, 'centroid_ts': centroid_ts}
        self.cluster_definitions_ = cluster_definitions

    def _calculate_weighted_cluster_returns(self, asset_returns_df_slice, period_indices_on_slice):
        """
        Calculates the weighted returns for each cluster based on the Gaussian distance from the centroid.
        Args:
            asset_returns_df_slice (pd.DataFrame): DataFrame slice of asset returns for the current period.
            period_indices_on_slice (tuple): Tuple containing start and end indices for the current period slice.
        Returns:
            pd.DataFrame: DataFrame containing the weighted returns for each cluster.
        """
        start_idx_slice, end_idx_slice = period_indices_on_slice
        lookback_data_for_dist_calc = self.current_lookback_data_for_weights_#.iloc[self.lookback_start_idx_:self.lookback_end_idx_]
        data_slice_current_period = asset_returns_df_slice.iloc[start_idx_slice:end_idx_slice]

        cluster_returns_dict = {}
        for cluster_name, info in self.cluster_definitions_.items():
            # Info contains 'tickers' and 'centroid_ts'
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

    # --- Overridden Core Methods ---

    def _fit(self, returns_df):
        """
        Fits the VAR model on the provided dataframe.
        """
        whole_data = returns_df.astype(float).dropna(axis=1, how='all')
        data_for_var = whole_data.values  # df -> np.ndarray
        self.results = VARReduce(self.var_order, regressor=ElasticNet())
        self.results.fit(data_for_var)
        self.lag_order_used = self.var_order

    def _forecast(self, returns_df, forecast_horizon, cross_val):

        whole_data = returns_df.astype(float).dropna(axis=1, how='all') # Keep dropna
        output_columns = returns_df.columns
        num_forecast_steps = forecast_horizon - 1 # Assumes forecast_horizon > 0
        lag_order = self.var_order

        forecast_list = []

        if cross_val:
            initial_model = VARReduce(lags=lag_order, regressor=ElasticNet())
            initial_model.fit(whole_data[:-num_forecast_steps])

            # The first time point we forecast is the one immediately after the initial training data.
            forecast_start_idx = len(whole_data) - forecast_horizon - lag_order

            # Create a list to hold the lagged arrays
            # The regressors are ordered [lag1_vars, lag2_vars, ...]
            # For each lag i, we select the appropriate slice of the original data.
            # The slice data_np[lag_order-i : -i] correctly aligns the lagged
            # observations with the endogenous variables.
            # Stack the arrays horizontally to form the final matrix
            X_lags = np.hstack([whole_data.values[lag_order - i : -i or None, :] for i in range(1, lag_order+1)])

            # Get the number of rows needed for the ones column
            num_rows = X_lags.shape[0]

            # Create a column vector of ones: shape (num_rows, 1)
            ones_column = np.ones((num_rows, 1))

            # Horizontally stack the ones column and the lags matrix
            full_regressor_matrix = np.hstack([ones_column, X_lags])

            model = initial_model

        else:
            X_lags = np.hstack([whole_data.values[lag_order - i : -i or None, :] for i in range(1, lag_order+1)])
            num_rows = X_lags.shape[0]
            ones_column = np.ones((num_rows, 1))
            full_regressor_matrix = np.hstack([ones_column, X_lags])
            forecast_start_idx = 0
            model = self.results

        params_matrix = self.convert_sktime_var_coeffs_to_statsmodels(model) # Shape: (n_regressors, n_assets)
        n_rls_inputs = params_matrix.shape[0]  # Number of regressors (constant + p*m)
        num_assets = params_matrix.shape[1]    # Number of assets/variables

        # rls_filters = [ExpL1L2Regression(n=n_rls_inputs, w=params_matrix.iloc[:, i].values) for i in range(num_assets)]

        # # We will loop for `num_forecast_steps` or until we run out of data.
        # for t in range(forecast_start_idx, forecast_start_idx + num_forecast_steps - self.lag_order_used):
        #     input_x = full_regressor_matrix[t]

        #     # Predict first
        #     forecast_list.append([rls_filters[j].predict(input_x) for j in range(num_assets)])

        #     # then adapt
        #     target_d_vector = whole_data.iloc[t].values
        #     for j in range(num_assets):
        #         # Adapt the j-th filter with the j-th target value
        #         rls_filters[j].update(input_x, target_d_vector[j])
        rls_filter = CppExpL1L2Regression(
            initial_w=params_matrix.T, # Pass transposed weights (assets x features)
            n_features=n_rls_inputs
            # other params like lam, halflife, etc., can be passed here
        )

        for t in range(forecast_start_idx, forecast_start_idx + num_forecast_steps - self.lag_order_used):
            input_x = full_regressor_matrix[t]
            # Predict for ALL assets in one go (calls C++ `predict`)
            forecasts_at_t = rls_filter.predict(input_x)
            forecast_list.append(forecasts_at_t)
            # Adapt the filter for ALL assets in one go (calls C++ `update`)
            target_d_vector = whole_data.iloc[t].values
            rls_filter.update(input_x, target_d_vector)

        forecast_array = np.array(forecast_list)
        forecast_index = pd.RangeIndex(start=0, stop=len(forecast_array), step=1)
        forecast_df = pd.DataFrame(forecast_array, columns=whole_data.columns, index=forecast_index)
        return forecast_df.reindex(columns=output_columns, fill_value=np.nan)