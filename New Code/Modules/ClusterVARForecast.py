import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.vector_ar.var_model import VAR
from signet.cluster import Cluster
import padasip as pa # For RLS filters

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
        whole_data = asset_returns_df.astype(float).dropna(axis=1, how='all')
        output_columns = asset_returns_df.columns # stores column names for output
        num_forecast_steps = forecast_horizon - 1 

        forecast_list = []

        ###################################################################################
        ## See comments below -> Don't we need to update and do a "rolling window" fit?
        ###################################################################################

        for i in range(num_forecast_steps):
            # I think we forgot to "refit the data" 
            current_results_iter = self.results # Fitted model instance
            current_lag_order_used_iter = self.lag_order_used
            slice_end_point_for_var_fit = len(whole_data) - (num_forecast_steps - i - current_lag_order_used_iter)
            data_for_var_model_fit_iter = whole_data.iloc[:slice_end_point_for_var_fit] # Expanding window?
            # That is, pick a "rolling window" instead here and update the data for each iteration?
            forecast_input = data_for_var_model_fit_iter.values[-current_lag_order_used_iter:] # Needed to make a prediction
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
        self.sigma_for_weights = max(float(sigma_for_weights), 1e-9)

        # Attributes to store clustering results
        self.corr_matrix_ = None
        self.cluster_definitions_ = None # Will store tickers, centroids, and weights

    # --- Helper methods for clustering (from original implementation) ---

    def _calculate_correlation_matrix(self, asset_returns_lookback_df):
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(asset_returns_lookback_df)
        normalized_df = pd.DataFrame(normalized_data, index=asset_returns_lookback_df.index, columns=asset_returns_lookback_df.columns)
        # normalized_df step converts numpy -> pandas DataFrame
        return normalized_df.corr(method='pearson').fillna(0)

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

    # --- cluster return logic ---

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
        lookback_data_for_dist_calc = self.current_lookback_data_for_weights_.iloc[self.lookback_start_idx_:self.lookback_end_idx_]
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

    def _forecast(self, cluster_returns_df, forecast_horizon, cross_val):
        """
        Forecasts returns for each cluster using a VAR model.
        Args:
            cluster_returns_df (pd.DataFrame): DataFrame of cluster returns (from _calculate_weighted_cluster_returns).
            forecast_horizon (int): Number of steps to forecast.
            cross_val (bool): Whether to use cross-validation.
        Returns:
            pd.DataFrame: Forecasted returns for each cluster.
        """
        whole_data = cluster_returns_df.astype(float).dropna(axis=1, how='all') # Keep dropna
        output_columns = cluster_returns_df.columns
        num_forecast_steps = forecast_horizon - 1 # Assumes forecast_horizon > 0

        forecast_list = []

        if cross_val:
            initial_model = VAR(whole_data.values[:-num_forecast_steps])
            initial_results = initial_model.fit(self.var_order)
            lag_order = self.var_order
            initial_training_rows = initial_results.nobs - lag_order

            model_end = VAR(whole_data.values)
            results_end = model_end.fit(lag_order)
            # results_end = VARReduce(lags=lag_order, regressor=ElasticNetCV(cv=5, random_state=0, max_iter=10000)).fit(whole_data.values
            full_regressor_matrix = results_end.endog_lagged  # Concatenated matrix coefficients
            # Shape: (n_samples, n_regressors) where n_regressors = p * (m + 1 (constant))

#            # Extract the parameters from the initial results (the initialized VAR model)
            params_matrix = initial_results.params # Shape: (n_regressors, n_assets)
            n_rls_inputs = params_matrix.shape[0]  # Number of regressors (constant + p*m) ?????
            num_assets = params_matrix.shape[1]    # Number of assets/variables

            rls_filters = [pa.filters.FilterRLS(n=n_rls_inputs, mu = 1, w=params_matrix[:, i]) for i in range(num_assets)]
            # This reads: From the padasip library, we create RLS filters for each (clustered) asset
            # Each RLS filter is an instance of FilterRLS.  
            # mu is the forgetting factor, w is the initial weights (parameters) for each filter.
            # Usually, mu = 1.0, which means no forgetting (plain "RLS").
            # Initialize RLS filters for each asset with the initial parameters from the VAR model.
            # Hence, rls_filters[0] corresponds to coefficients for the first (cluster) asset, rls_filters[1] for the second (cluster), etc.

            # The first time point we forecast is the one immediately after the initial training data.
            forecast_start_idx = initial_training_rows

            # We will loop for `num_forecast_steps` or until we run out of data.
            for t in range(forecast_start_idx, forecast_start_idx + num_forecast_steps):

                # t is a scalar, so x_matrix_idx is too (ie shifted indices of the matrix).
                x_matrix_idx = t - lag_order
                input_x = full_regressor_matrix[x_matrix_idx]
                # As t progresses, we are shifting the input_x "window" to the right.
                # This is equivalent to what we do in a rolling window forecast.

                # Predict first
                forecast_list.append([rls_filters[j].predict(input_x) for j in range(num_assets)])

                # then adapt
                target_d_vector = whole_data.iloc[t].values
                for j in range(num_assets):
                    # Adapt the j-th filter (ie for jth asset) with the incoming target value for the j-th asset.
                    rls_filters[j].adapt(target_d_vector[j], input_x)

            forecast_array = np.array(forecast_list)

        else:
            # This is saying that the forecast function of the parent class will be used,
            # using just the cluster_returns_df as input. 
            return super()._forecast(cluster_returns_df, forecast_horizon, cross_val)

        forecast_index = pd.RangeIndex(start=0, stop=len(forecast_array), step=1)
        forecast_df = pd.DataFrame(forecast_array, columns=whole_data.columns, index=forecast_index)
        return forecast_df.reindex(columns=output_columns, fill_value=np.nan)