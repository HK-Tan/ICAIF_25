import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from scipy import sparse
import warnings
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering

warnings.filterwarnings('ignore')

try:
    from signet.cluster import Cluster
except ImportError:
    print("Signet package not found. Installing...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/alan-turing-institute/SigNet.git"])
        from signet.cluster import Cluster
    except Exception as e:
        print(f"Error installing Signet package: {e}")
        sys.exit(1)

# ----------------------------------------------------------------



def get_sp500_PnL(start_date, end_date):

    '''
    ----------------------------------------------------------------
    GENERAL IDEA : get the S&P500 index daily PnL between the star
                   and end dates
    ----------------------------------------------------------------

    ----------------------------------------------------------------
    PARAMS :

    - start_date, end_date : strings, corresponding to start and end
                             dates. The format is the datetime format
                             "YYYY-MM-DD"

    ----------------------------------------------------------------

    ----------------------------------------------------------------
    OUTPUT : pandas.DataFrame containing the S&P500 index daily
             between the star and end dates
    ----------------------------------------------------------------
    '''

    # Specify the ticker symbol for S&P 500
    ticker_symbol = "^GSPC"

    # Fetch historical data
    sp500_data = yf.download(ticker_symbol, start=start_date, end=end_date)
    sp500_data['Daily PnL'] = (sp500_data['Close'] - sp500_data['Open']) / sp500_data['Open'][0] ## /100 because we initially invest 1 dollar in our portfolio?
    sp500_PnL = sp500_data['Daily PnL'].transpose() ## we remove the -2 values to have matching values

    return sp500_PnL

def calculate_mean_correlation(df):
    if df.empty or df.shape[1] < 2: # Need at least 2 columns for correlation
        return 0.0

    correlation_matrix = df.corr().fillna(0) # Fill NaNs (e.g. from constant columns)
    correlation_values = correlation_matrix.values
    n = correlation_values.shape[0]

    # Use np.triu_indices to get upper triangle indices excluding diagonal
    # This is more efficient than nested loops for large n, though for typical cluster sizes, loop is fine.
    # For consistency with original, keeping loop.
    total_correlation = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total_correlation += correlation_values[i, j]
            count += 1

    return total_correlation / count if count > 0 else 0.0

def get_most_corr_cluster(portfolio, lookback_window, df_cleaned, number=1, strat='correlation', cache=None):
    # Cache key: A tuple uniquely identifying this portfolio state for correlation calculation
    # Using portfolio object id can be tricky if portfolio object itself isn't hashable or changes.
    # A simpler cache key could be tuple(lookback_window) if df_cleaned and portfolio.cluster_composition are fixed for that window.
    # For now, let's assume cache is managed externally for each `current_pf_c_instance` call context.
    # If cache is provided and this specific 'number' is in cache, return it.

    # A more robust cache key if using within a loop for different 'number'
    # but same portfolio instance and lookback_window:
    # cache_key_base = (id(portfolio.cluster_composition), tuple(lookback_window))
    # cache_key_full = (cache_key_base, number)
    # if cache is not None and cache_key_full in cache:
    #     return cache[cache_key_full]

    # Simpler caching: If we are caching all results for a portfolio+lookback,
    # the cache passed would store a list of sorted clusters.
    # Let's assume cache stores the sorted list for the current context.

    sorted_cluster_corr_list = None
    if cache is not None and 'sorted_clusters' in cache:
        sorted_cluster_corr_list = cache['sorted_clusters']
    else:
        mean_corr_list = []
        if not portfolio.cluster_composition:
            # print("Warning: portfolio.cluster_composition is empty in get_most_corr_cluster.")
            if cache is not None: cache['sorted_clusters'] = []
            return (None, 0.0, [])

        for name, cluster_info in portfolio.cluster_composition.items():
            tickers = cluster_info['tickers']
            if not tickers:
                mean_corr_list.append((name, 0.0, tickers))
                continue

            lw_start, lw_end = lookback_window[0], lookback_window[1]
            if not (0 <= lw_start < lw_end <= len(df_cleaned)):
                # print(f"Warning: Invalid lookback_window {lookback_window} for df_cleaned.")
                mean_corr_list.append((name, 0.0, tickers))
                continue

            tickers_df = df_cleaned[tickers].iloc[lw_start:lw_end, :]
            mean_corr = calculate_mean_correlation(tickers_df)
            mean_corr_list.append((name, mean_corr, tickers))

        if not mean_corr_list:
            if cache is not None: cache['sorted_clusters'] = []
            return (None, 0.0, [])

        sorted_cluster_corr_list = sorted(mean_corr_list, key=lambda x: x[1])
        if cache is not None:
            cache['sorted_clusters'] = sorted_cluster_corr_list

    if not sorted_cluster_corr_list: # If list is empty after all
        return (None, 0.0, [])

    try:
        # 'number' is 1-indexed for k-th most correlated. List is 0-indexed, sorted ascending.
        # -number gives from the end (highest correlation).
        if 1 <= number <= len(sorted_cluster_corr_list):
            result = sorted_cluster_corr_list[-number]
        else:
            # print(f"Warning: 'number' {number} out of bounds. Returning most correlated.")
            result = sorted_cluster_corr_list[-1]
    except IndexError:
        # print("Warning: IndexError in get_most_corr_cluster. Defaulting.")
        result = (None, 0.0, [])

    return result


def most_corr_returns(portfolio, lookback_window, evaluation_window, df_cleaned, number=1, cache_for_get_cluster=None):
    most_corr_cluster_info = get_most_corr_cluster(portfolio, lookback_window, df_cleaned, number, cache=cache_for_get_cluster)

    cluster_name, _, ticker_list = most_corr_cluster_info # Unpack correlation value as well

    eval_start = lookback_window[1]
    eval_end = lookback_window[1] + evaluation_window

    # Adjust eval_end if it exceeds data length
    if eval_end > len(df_cleaned):
        eval_end = len(df_cleaned)

    if cluster_name is None or not ticker_list or not (0 <= eval_start < eval_end):
        # print(f"Warning: No valid cluster/tickers or invalid eval window for number {number} in most_corr_returns.")
        # Ensure index is valid even for empty return
        idx = df_cleaned.index[eval_start:eval_end] if 0 <= eval_start < eval_end else pd.Index([])
        return pd.DataFrame(index=idx, columns=['empty_cluster'], data=0.0)

    actual_eval_window_len = eval_end - eval_start

    most_corr_cluster_returns = pd.DataFrame(
        index=df_cleaned.index[eval_start:eval_end],
        columns=[cluster_name],
        data=np.zeros((actual_eval_window_len, 1))
    )

    if not hasattr(portfolio, 'consolidated_weight') or portfolio.consolidated_weight.empty:
        # print("Warning: portfolio.consolidated_weight not found/empty. Assuming zero weights.")
        return most_corr_cluster_returns

    # Ensure consolidated_weight has 'weight' row if it's a DataFrame as constructed in PyFolioC
    weights_series = None
    if 'weight' in portfolio.consolidated_weight.index:
        weights_series = portfolio.consolidated_weight.loc['weight']
    else: # Fallback if structure is different, e.g. Series directly or single column DataFrame
        if isinstance(portfolio.consolidated_weight, pd.Series):
            weights_series = portfolio.consolidated_weight
        elif isinstance(portfolio.consolidated_weight, pd.DataFrame) and portfolio.consolidated_weight.shape[0] == 1:
            weights_series = portfolio.consolidated_weight.iloc[0] # Assume first row if not named 'weight'

    if weights_series is None:
        # print("Warning: Could not extract weights_series from portfolio.consolidated_weight.")
        return most_corr_cluster_returns


    for ticker in ticker_list:
        if ticker in weights_series.index: # Check against Series index
            weight = weights_series[ticker]
            most_corr_cluster_returns[cluster_name] += df_cleaned[ticker].iloc[eval_start:eval_end].values * weight # Use .values for alignment
        # else:
            # print(f"Warning: Ticker {ticker} from cluster {cluster_name} not in consolidated_weight series.")

    return most_corr_cluster_returns

def most_corr_PnL(consolidated_portfolio, lookback_window, evaluation_window, df_cleaned, number=1, cache_for_get_cluster=None):
    most_corr_return_df = most_corr_returns(consolidated_portfolio, lookback_window, evaluation_window, df_cleaned, number, cache_for_get_cluster=cache_for_get_cluster)

    # Get cluster info (potentially from cache via most_corr_returns -> get_most_corr_cluster)
    # This call ensures we have the cluster_info, even if PnL is zero.
    cluster_info = get_most_corr_cluster(consolidated_portfolio, lookback_window, df_cleaned, number, cache=cache_for_get_cluster)

    if most_corr_return_df.empty or most_corr_return_df.iloc[:,0].isnull().all() or most_corr_return_df.shape[0] == 0:
        # print(f"Warning: most_corr_returns for number {number} is empty/all NaN. PnL will be 0.")
        return 0.0, cluster_info

    returns_series = most_corr_return_df.iloc[:, 0].fillna(0)
    if returns_series.empty: # Double check after fillna if original was completely empty structure
        return 0.0, cluster_info

    cumulative_returns = (1 + returns_series).cumprod() - 1

    final_pnl = cumulative_returns.iloc[-1] if not cumulative_returns.empty else 0.0
    return final_pnl, cluster_info


class PyFolio:
    def __init__(self, historical_data, lookback_window, evaluation_window, number_of_clusters, sigma, beta, EWA_cov = False, short_selling=False, cov_method='SPONGE', var_order=1):
        self.historical_data = historical_data
        self.lookback_window = lookback_window
        self.evaluation_window = evaluation_window
        self.number_of_clusters = number_of_clusters
        self.cov_method = cov_method
        self.sigma = sigma
        self.beta = beta
        self.EWA_cov = EWA_cov
        self.short_selling = short_selling
        self.var_order = max(1, int(var_order))

        self.correlation_matrix = self.corr_matrix()
        self.cluster_composition = self.cluster_composition_and_centroid()
        self.constituent_weights_res = self.constituent_weights()
        self.cluster_returns = self.cluster_return(lookback_window)
        self.cluster_level_weights = self._calculate_cluster_weights()
        self.final_weights = self.final_W()

    def apply_SPONGE(self):
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)
        data = (sparse.csc_matrix(A_pos.values), sparse.csc_matrix(A_neg.values))
        cluster = Cluster(data)
        return cluster.SPONGE(self.number_of_clusters)

    def apply_signed_laplacian(self):
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)
        A_pos_sparse = sparse.csc_matrix(A_pos.values)
        A_neg_sparse = sparse.csc_matrix(A_neg.values)
        data = (A_pos_sparse, A_neg_sparse)
        cluster = Cluster(data)
        return cluster.spectral_cluster_laplacian(self.number_of_clusters)

    def apply_SPONGE_sym(self):
        A_pos, A_neg = self.correlation_matrix.applymap(lambda x: x if x >= 0 else 0), self.correlation_matrix.applymap(lambda x: abs(x) if x < 0 else 0)
        A_pos_sparse = sparse.csc_matrix(A_pos.values)
        A_neg_sparse = sparse.csc_matrix(A_neg.values)
        data = (A_pos_sparse, A_neg_sparse)
        cluster = Cluster(data)
        return cluster.SPONGE_sym(self.number_of_clusters)

    def apply_kmeans(self):
        data = self.correlation_matrix.fillna(0) # Ensure no NaNs for KMeans
        if data.empty: return np.array([])
        kmeans = KMeans(n_clusters=self.number_of_clusters, random_state=0, n_init='auto') # n_init='auto' for future sklearn
        return kmeans.fit_predict(data) # Use fit_predict

    def apply_spectral_clustering(self):
        # Sklearn's SpectralClustering is generally more standard for this unless Signet has specific advantages.
        # Using affinity matrix from absolute correlation:
        if self.correlation_matrix.empty: return np.array([])
        affinity_matrix = np.abs(self.correlation_matrix.fillna(0).values)
        # Ensure matrix is symmetric and non-negative
        affinity_matrix = (affinity_matrix + affinity_matrix.T) / 2
        np.fill_diagonal(affinity_matrix, 1) # Self-similarity is 1

        # Handle cases with few samples or clusters for SpectralClustering
        n_samples = affinity_matrix.shape[0]
        if self.number_of_clusters > n_samples:
            # print(f"Warning: n_clusters ({self.number_of_clusters}) > n_samples ({n_samples}). Reducing n_clusters for SpectralClustering.")
            effective_n_clusters = max(1, n_samples) # Must be at least 1
        else:
            effective_n_clusters = self.number_of_clusters

        if effective_n_clusters <= 0 : return np.array([]) # Should not happen if n_samples > 0

        try:
            sc = SpectralClustering(n_clusters=effective_n_clusters, affinity='precomputed', random_state=0, assign_labels='kmeans')
            labels = sc.fit_predict(affinity_matrix)
        except Exception as e:
            # print(f"SpectralClustering failed: {e}. Falling back to KMeans.")
            return self.apply_kmeans() # Fallback to KMeans if SpectralClustering fails
        return labels

    def corr_matrix(self):
        data_slice = self.historical_data.iloc[self.lookback_window[0]:self.lookback_window[1], :]
        if data_slice.empty: return pd.DataFrame() # Handle empty slice

        # Normalize data: (X - mean) / std. Handle std=0 by replacing with a small number.
        mean_vals = data_slice.mean(axis=0)
        std_vals = data_slice.std(axis=0)
        std_vals[std_vals < 1e-9] = 1e-9 # Avoid division by zero or very small std

        # Element-wise operations are efficient on DataFrames
        normalized_data = (data_slice - mean_vals) / std_vals

        # .corr() is efficient and handles pairwise NaNs by default (though normalized_data shouldn't have many if std is handled)
        correlation_matrix = normalized_data.corr(method='pearson')
        return correlation_matrix.fillna(0) # Fill any remaining NaNs (e.g., if a col was perfectly constant)

    def cluster_composition_and_centroid(self):
        if self.number_of_clusters <=0: # No clusters to form
            return {}

        if self.correlation_matrix.empty or self.correlation_matrix.shape[0] < self.number_of_clusters:
            # Not enough assets to form the requested number of clusters, or no correlation matrix
            # print("Warning: Correlation matrix empty or too few assets for clustering. Reducing number_of_clusters.")
            # Fallback: try to cluster all assets into one cluster if possible, or return empty.
            if self.correlation_matrix.empty or self.correlation_matrix.shape[0] == 0:
                return {}
            effective_n_clusters = max(1, self.correlation_matrix.shape[0]) # At least 1 cluster, or all assets if fewer than requested
            if effective_n_clusters < self.number_of_clusters : self.number_of_clusters = effective_n_clusters
            # if self.number_of_clusters becomes 0, handle above.


        # ... (rest of the clustering method calls using self.number_of_clusters)
        if self.cov_method == 'SPONGE': labels = self.apply_SPONGE()
        elif self.cov_method == 'signed_laplacian': labels = self.apply_signed_laplacian()
        elif self.cov_method == 'SPONGE_sym': labels = self.apply_SPONGE_sym()
        elif self.cov_method == 'Kmeans': labels = self.apply_kmeans()
        elif self.cov_method == 'spectral_clustering': labels = self.apply_spectral_clustering()
        else: raise ValueError(f"Unknown clustering method: {self.cov_method}")

        if not isinstance(labels, np.ndarray) or labels.size == 0: # If clustering failed or returned empty
            # print(f"Warning: Clustering method {self.cov_method} returned no labels. Defaulting to single cluster for all assets.")
            if self.correlation_matrix.empty: return {}
            labels = np.zeros(self.correlation_matrix.shape[0], dtype=int) # All in cluster 0
            self.number_of_clusters = 1 # Adjust effective number of clusters

        result = pd.DataFrame(index=list(self.correlation_matrix.columns), columns=['Cluster label'], data=labels)
        result['Cluster label'] += 1 # 1-based indexing

        cluster_composition = {}
        hist_data_lookback = self.historical_data.iloc[self.lookback_window[0]:self.lookback_window[1], :]

        for i in range(1, result['Cluster label'].max() + 1): # Iterate up to max label found
            cluster_tickers_series = result[result['Cluster label'] == i]
            if cluster_tickers_series.empty: continue

            tickers = list(cluster_tickers_series.index)
            valid_tickers = [t for t in tickers if t in hist_data_lookback.columns]

            if not valid_tickers:
                centroid_val = np.zeros(self.lookback_window[1] - self.lookback_window[0])
            else:
                centroid_val = hist_data_lookback.loc[:, valid_tickers].mean(axis=1).values # .values for numpy array

            cluster_composition[f'cluster {i}'] = {'tickers': valid_tickers, 'centroid': centroid_val}
        return cluster_composition

    def constituent_weights(self):
        constituent_weights = {}
        hist_data_lookback = self.historical_data.iloc[self.lookback_window[0]:self.lookback_window[1], :]

        for cluster_name, cluster_data_info in self.cluster_composition.items():
            tickers = cluster_data_info['tickers']
            centroid_np = cluster_data_info['centroid']

            exp_vals = []
            valid_tickers_for_weights = []

            if not tickers:
                constituent_weights[cluster_name] = {}
                continue

            for elem_ticker in tickers:
                if elem_ticker not in hist_data_lookback.columns: continue
                valid_tickers_for_weights.append(elem_ticker)
                elem_returns_np = hist_data_lookback[elem_ticker].values

                if centroid_np.shape != elem_returns_np.shape:
                    distance_to_centroid_sq = np.inf # Penalize heavily if shapes mismatch
                else:
                    distance_to_centroid_sq = np.sum((centroid_np - elem_returns_np)**2)

                exp_vals.append(np.exp(-distance_to_centroid_sq / (2 * (self.sigma**2))))

            exp_vals_np = np.array(exp_vals)
            total_cluster_weight_exp = exp_vals_np.sum()

            current_cluster_weights = {}
            if total_cluster_weight_exp > 1e-9:
                normalized_exp_weights = exp_vals_np / total_cluster_weight_exp
                for ticker, weight in zip(valid_tickers_for_weights, normalized_exp_weights):
                    current_cluster_weights[ticker] = weight
            else: # Fallback to equal weights if sum is too small
                num_v_tickers = len(valid_tickers_for_weights)
                for ticker in valid_tickers_for_weights:
                    current_cluster_weights[ticker] = 1.0 / num_v_tickers if num_v_tickers > 0 else 0.0

            constituent_weights[cluster_name] = current_cluster_weights
        return constituent_weights

    def cluster_return(self, lookback_window):
        start_idx, end_idx = lookback_window[0], lookback_window[1]

        if not (0 <= start_idx < end_idx <= len(self.historical_data)) or not self.constituent_weights_res:
            return pd.DataFrame(index=pd.Index([]), columns=["dummy_cluster"])

        num_days = end_idx - start_idx
        cluster_names = list(self.constituent_weights_res.keys())
        num_clusters = len(cluster_names)

        cluster_returns_np = np.zeros((num_days, num_clusters))
        hist_data_slice = self.historical_data.iloc[start_idx:end_idx, :]

        for i, cluster_name in enumerate(cluster_names):
            for ticker, weight in self.constituent_weights_res[cluster_name].items():
                if ticker in hist_data_slice.columns:
                    cluster_returns_np[:, i] += hist_data_slice[ticker].values * weight

        return pd.DataFrame(cluster_returns_np, index=hist_data_slice.index, columns=cluster_names)

    def _fit_and_forecast_var(self):
        data_for_var = self.cluster_returns.astype(float).dropna(axis=1, how='all')
        var_order = self.var_order

        if data_for_var.empty or data_for_var.shape[0] <= var_order or data_for_var.shape[1] == 0:
            return pd.Series(0.0, index=self.cluster_returns.columns)

        N_obs, N_series = data_for_var.shape
        X_lags_list = [data_for_var.iloc[var_order - lag : N_obs - lag].values for lag in range(1, var_order + 1)]
        X_lags = np.hstack(X_lags_list)
        X_const = np.ones((X_lags.shape[0], 1))
        X = np.hstack((X_const, X_lags))
        Y = data_for_var.iloc[var_order:].values
        coefficients = np.linalg.solve(X.T @ X, X.T @ Y)

        # try:
        #     XTX = X.T @ X
        #     XTY = X.T @ Y
        #     # Add small regularization term to XTX diagonal for stability if XTX is ill-conditioned
        #     # This is a form of Ridge regression for VAR.
        #     # lambda_reg = 1e-6
        #     # coefficients = np.linalg.solve(XTX + lambda_reg * np.eye(XTX.shape[0]), XTY)
        #     coefficients = np.linalg.solve(XTX, XTY)

        # except np.linalg.LinAlgError:
        #     # print(f"XTX is singular or near-singular. Falling back to pseudo-inverse for VAR.")
        #     try:
        #         coefficients = np.linalg.pinv(X.T @ X) @ (X.T @ Y)
        #     except np.linalg.LinAlgError:
        #         # print(f"Pseudo-inverse also failed. Returning zero forecasts.")
        #         return pd.Series(0.0, index=self.cluster_returns.columns)

        forecasts = np.zeros((self.evaluation_window, N_series))
        history_for_forecast = data_for_var.iloc[N_obs - var_order : N_obs].values

        for i in range(self.evaluation_window):
            lagged_values_flat = history_for_forecast[::-1].ravel() # Flattens in reverse order of lags
            forecast_X_vec = np.hstack(([1.0], lagged_values_flat))
            next_forecast = forecast_X_vec @ coefficients
            forecasts[i, :] = next_forecast
            history_for_forecast = np.vstack((history_for_forecast[1:], next_forecast))

        forecasts_df = pd.DataFrame(forecasts, columns=data_for_var.columns)
        mean_forecasted_returns = forecasts_df.mean()
        return mean_forecasted_returns.reindex(self.cluster_returns.columns, fill_value=0.0)

    def _calculate_cluster_weights(self):
        forecasted_returns = self._fit_and_forecast_var()

        if forecasted_returns.empty or forecasted_returns.isnull().all():
            num_clusters = len(self.cluster_returns.columns) if self.cluster_returns is not None and not self.cluster_returns.empty else 1
            num_clusters = max(1, num_clusters) # Ensure not zero
            return pd.Series(1.0 / num_clusters, index=self.cluster_returns.columns if self.cluster_returns is not None and not self.cluster_returns.empty else ["dummy"])

        if not self.short_selling:
            positive_forecasts = forecasted_returns.clip(lower=0)
            sum_positive = positive_forecasts.sum()
            if sum_positive < 1e-9:
                num_clusters = len(forecasted_returns)
                cluster_weights = pd.Series(1.0 / num_clusters if num_clusters > 0 else 0.0, index=forecasted_returns.index)
            else:
                cluster_weights = positive_forecasts / sum_positive
        else:
            abs_sum_forecasts = forecasted_returns.abs().sum()
            if abs_sum_forecasts < 1e-9:
                num_clusters = len(forecasted_returns)
                cluster_weights = pd.Series(1.0 / num_clusters if num_clusters > 0 else 0.0, index=forecasted_returns.index)
            else:
                cluster_weights = forecasted_returns / abs_sum_forecasts
        return cluster_weights.reindex(self.cluster_returns.columns, fill_value=0.0)


    def final_W(self):
        W_dict = {}
        if self.cluster_level_weights.empty or self.cluster_level_weights.isnull().all():
            # print("Warning: cluster_level_weights is empty/NaN. Defaulting final weights.")
            # Fallback to equal weight among all original historical assets
            num_total_assets = len(self.historical_data.columns)
            if num_total_assets > 0:
                return pd.DataFrame({'weights': np.ones(num_total_assets) / num_total_assets},
                                    index=self.historical_data.columns)
            else:
                return pd.DataFrame(columns=['weights'], index=pd.Index([], name='ticker'))

        for cluster_name, constituent_dict in self.constituent_weights_res.items():
            cluster_w = self.cluster_level_weights.get(cluster_name, 0.0)
            if abs(cluster_w) > 1e-9:
                for ticker, constituent_w in constituent_dict.items():
                    # W_dict[ticker] = W_dict.get(ticker, 0.0) + constituent_w * cluster_w # This could lead to double counting if a ticker is in multiple clusters (not typical for hard clustering)
                    W_dict[ticker] = constituent_w * cluster_w # Assuming hard clustering, ticker belongs to one cluster effectively or its weight is sum from "soft" assignment by constituent_weights

        if not W_dict:
            num_total_assets = len(self.historical_data.columns)
            if num_total_assets > 0:
                return pd.DataFrame({'weights': np.ones(num_total_assets) / num_total_assets}, index=self.historical_data.columns)
            else: return pd.DataFrame(columns=['weights'], index=pd.Index([], name='ticker'))

        # Normalize final weights if short selling is false, to sum to 1 and be non-negative
        final_weights_series = pd.Series(W_dict)
        if not self.short_selling:
            final_weights_series = final_weights_series.clip(lower=0)
            sum_final_weights = final_weights_series.sum()
            if sum_final_weights > 1e-9:
                final_weights_series = final_weights_series / sum_final_weights
            else: # If all clipped weights are zero, distribute equally
                non_zero_count = (final_weights_series > 1e-9).sum() # Should be 0 here
                if len(final_weights_series) > 0:
                    final_weights_series = pd.Series(1.0/len(final_weights_series), index=final_weights_series.index)
                # else it remains empty series

        return pd.DataFrame(final_weights_series, columns=['weights'])


class PyFolioC(PyFolio):
    def __init__(self, number_of_repetitions, historical_data, lookback_window, evaluation_window, number_of_clusters, sigma, beta, EWA_cov = False, short_selling=False, cov_method='SPONGE', var_order=1, transaction_cost_rate=0.0001):
        self.number_of_repetitions = number_of_repetitions
        super().__init__(historical_data, lookback_window, evaluation_window, number_of_clusters, sigma, beta, EWA_cov, short_selling, cov_method, var_order)
        self.transaction_cost_rate = transaction_cost_rate
        self.consolidated_weight = self.consolidated_W()
        self.portfolio_return = self.portfolio_returns()

    def consolidated_W(self):
        if self.number_of_repetitions == 1: # If only one repetition, use weights from super init
            return self.final_weights.T.rename(index={'weights':'weight'}) # Match expected output format

        all_weights_dfs = []
        for _ in range(self.number_of_repetitions):
            try:
                portfolio_run = PyFolio(historical_data=self.historical_data, lookback_window=self.lookback_window, evaluation_window=self.evaluation_window, number_of_clusters=self.number_of_clusters, sigma=self.sigma, beta=self.beta, EWA_cov=self.EWA_cov, short_selling=self.short_selling, cov_method=self.cov_method, var_order=self.var_order)
                if not portfolio_run.final_weights.empty:
                    all_weights_dfs.append(portfolio_run.final_weights)
            except Exception as e:
                print(f"Error in PyFolio run for consolidated_W: {e}. Skipping.")

        if not all_weights_dfs:
            all_tickers = self.historical_data.columns
            return pd.DataFrame(np.zeros((1, len(all_tickers))), columns=all_tickers, index=['weight'])

        # Each df in all_weights_dfs has 'weights' column and ticker index
        # Concatenate along axis=1, then mean, then transpose.
        # Example: df1 = pd.DataFrame({'weights': [0.5,0.5]}, index=['A','B'])
        #          df2 = pd.DataFrame({'weights': [0.6,0.4]}, index=['A','B'])
        # panel = pd.concat([df1,df2], axis=1) -> columns are ('weights',0), ('weights',1) or similar if names are same
        # Need to handle column names carefully if they are all 'weights'.
        # Renaming columns before concat:
        for i, df in enumerate(all_weights_dfs):
            df.columns = [f'weights_rep_{i}']

        consolidated_panel = pd.concat(all_weights_dfs, axis=1)
        average_weights_series = consolidated_panel.mean(axis=1)
        return pd.DataFrame(average_weights_series, columns=['weight']).T


    def portfolio_returns(self):
        eval_start_idx, eval_end_idx = self.lookback_window[1], self.lookback_window[1] + self.evaluation_window
        eval_end_idx = min(eval_end_idx, len(self.historical_data))

        if eval_start_idx >= eval_end_idx:
            return pd.DataFrame(columns=['return'], index=pd.Index([]))

        hist_data_eval_slice = self.historical_data.iloc[eval_start_idx:eval_end_idx, :]
        portfolio_returns_np = np.zeros(len(hist_data_eval_slice))

        if self.consolidated_weight.empty or 'weight' not in self.consolidated_weight.index:
            return pd.DataFrame(portfolio_returns_np, index=hist_data_eval_slice.index, columns=['return'])

        weights_series = self.consolidated_weight.loc['weight']
        for ticker, weight in weights_series.items():
            if abs(weight) > 1e-9 and ticker in hist_data_eval_slice.columns:
                portfolio_returns_np += hist_data_eval_slice[ticker].values * weight

        return pd.DataFrame(portfolio_returns_np, index=hist_data_eval_slice.index, columns=['return'])

    def sliding_window_past_dep(self, number_of_window, include_transaction_costs=True):
        overall_return_list = []
        portfolio_values_over_time = [1.0]
        Turnovers=[]
        previous_weights_df = None # Stores consolidated_weight (DataFrame with 'weight' row)

        for i in range(number_of_window):
            current_portfolio_value_for_block_pnl = portfolio_values_over_time[-1]
            try:
                current_lookback_window = [self.lookback_window[0] + self.evaluation_window * i,
                                           self.lookback_window[1] + self.evaluation_window * i]
                if not (current_lookback_window[1] <= len(self.historical_data) and
                        current_lookback_window[0] < current_lookback_window[1]):
                    break

                current_pf_c_instance = PyFolioC(number_of_repetitions=self.number_of_repetitions, historical_data=self.historical_data, lookback_window=current_lookback_window, evaluation_window=self.evaluation_window, number_of_clusters=self.number_of_clusters, sigma=self.sigma, beta=self.beta, EWA_cov=self.EWA_cov, short_selling=self.short_selling, cov_method=self.cov_method, var_order=self.var_order, transaction_cost_rate=self.transaction_cost_rate)
                current_consolidated_weights_df = current_pf_c_instance.consolidated_weight

                current_weights_series = current_consolidated_weights_df.loc['weight'] if not current_consolidated_weights_df.empty and 'weight' in current_consolidated_weights_df.index else pd.Series(0.0, index=self.historical_data.columns)

                if previous_weights_df is None: Turnover = 1.0
                else:
                    prev_w_series = previous_weights_df.loc['weight']
                    all_tkrs = prev_w_series.index.union(current_weights_series.index)
                    Turnover = np.sum(np.abs(prev_w_series.reindex(all_tkrs).fillna(0) - current_weights_series.reindex(all_tkrs).fillna(0)))
                Turnovers.append(Turnover)
                previous_weights_df = current_consolidated_weights_df

                transaction_costs_per_day = (Turnover * self.transaction_cost_rate) / self.evaluation_window if include_transaction_costs and self.evaluation_window > 0 else 0.0
                block_portfolio_returns_df = current_pf_c_instance.portfolio_return

                if block_portfolio_returns_df.empty:
                    portfolio_values_over_time.append(current_portfolio_value_for_block_pnl)
                    continue

                overall_return_list.append(block_portfolio_returns_df - transaction_costs_per_day)
                adjusted_block_returns = block_portfolio_returns_df['return'] - transaction_costs_per_day
                block_cumulative_pnl_values = (np.cumprod(1 + adjusted_block_returns) - 1) * current_portfolio_value_for_block_pnl

                if not block_cumulative_pnl_values.empty:
                    portfolio_values_over_time.append(current_portfolio_value_for_block_pnl + block_cumulative_pnl_values.iloc[-1])
                else: portfolio_values_over_time.append(current_portfolio_value_for_block_pnl)
                print(f'step {i+1}/{number_of_window}, portfolio value: {portfolio_values_over_time[-1]:.4f}')
            except Exception as e:
                print(f"Error in sliding_window_past_dep step {i+1}: {e}")
                break

        overall_return_final_df = pd.concat(overall_return_list) if overall_return_list else pd.DataFrame(columns=['return'])
        continuous_cumulative_pnl_from_start = ((1 + overall_return_final_df['return']).cumprod() - 1) if not overall_return_final_df.empty else pd.Series([])

        # For daily_PnL, it was originally block-wise from start of block. Reconstruct similar idea if needed or use overall_return_final_df
        # For now, daily_PnL is not explicitly reconstructed here as it was for block-relative PnL.
        # The primary PnL metrics are overall_return_final_df (daily strategy returns) and continuous_cumulative_pnl_from_start (total PnL).
        return overall_return_final_df, continuous_cumulative_pnl_from_start.values, portfolio_values_over_time, None, Turnovers # daily_PnL placeholder


    def sliding_window_past_indep(self, number_of_window, include_transaction_costs=True):
        overall_return_list = []
        portfolio_value_tracker = [1.0]
        previous_weights_df = None
        Turnovers = []
        # Ensure number_of_clusters is positive for initializing this list
        num_contrib_lists = max(1, self.number_of_clusters)
        most_corr_contribution_lists = [[] for _ in range(num_contrib_lists)]

        # Cache for get_most_corr_cluster results per sliding window step
        # Key: tuple(lookback_window), Value: { 'sorted_clusters': [...] }
        # This cache is reset for each main sliding window step (i)

        for i in range(1, number_of_window + 1):
            # Reset cache for this new portfolio instance context
            correlation_cache_for_this_step = {}
            try:
                current_lookback_window = [self.lookback_window[0] + self.evaluation_window * (i-1),
                                           self.lookback_window[1] + self.evaluation_window * (i-1)]
                if not (current_lookback_window[1] <= len(self.historical_data) and
                        current_lookback_window[0] < current_lookback_window[1]):
                    break

                current_pf_c_instance = PyFolioC(number_of_repetitions=self.number_of_repetitions, historical_data=self.historical_data, lookback_window=current_lookback_window, evaluation_window=self.evaluation_window, number_of_clusters=self.number_of_clusters, sigma=self.sigma, beta=self.beta, EWA_cov=self.EWA_cov, short_selling=self.short_selling, cov_method=self.cov_method, var_order=self.var_order, transaction_cost_rate=self.transaction_cost_rate)
                current_consolidated_weights_df = current_pf_c_instance.consolidated_weight
                current_weights_series = current_consolidated_weights_df.loc['weight'] if not current_consolidated_weights_df.empty and 'weight' in current_consolidated_weights_df.index else pd.Series(0.0, index=self.historical_data.columns)

                if previous_weights_df is None: Turnover = 1.0
                else:
                    prev_w_series = previous_weights_df.loc['weight']
                    all_tkrs = prev_w_series.index.union(current_weights_series.index)
                    Turnover = np.sum(np.abs(prev_w_series.reindex(all_tkrs).fillna(0) - current_weights_series.reindex(all_tkrs).fillna(0)))
                Turnovers.append(Turnover)
                previous_weights_df = current_consolidated_weights_df

                transaction_costs_per_day = (Turnover * self.transaction_cost_rate) / self.evaluation_window if include_transaction_costs and self.evaluation_window > 0 else 0.0
                block_portfolio_returns_df = current_pf_c_instance.portfolio_return

                if block_portfolio_returns_df.empty:
                    if overall_return_list: portfolio_value_tracker.append(portfolio_value_tracker[-1])
                    else: portfolio_value_tracker.append(1.0)
                    continue

                overall_return_list.append(block_portfolio_returns_df - transaction_costs_per_day)

                # Most correlated cluster contribution
                if not block_portfolio_returns_df.empty:
                    non_adjusted_block_cum_returns = (1 + block_portfolio_returns_df['return']).cumprod() -1
                    last_non_adjusted_return_for_block = non_adjusted_block_cum_returns.iloc[-1] if not non_adjusted_block_cum_returns.empty else 0.0

                    if abs(last_non_adjusted_return_for_block) > 1e-9 and current_pf_c_instance.cluster_composition:
                        num_clusters_to_check = min(self.number_of_clusters, len(current_pf_c_instance.cluster_composition))
                        num_clusters_to_check = max(1, num_clusters_to_check)

                        for k_idx in range(num_clusters_to_check): # k_idx from 0 to num_clusters_to_check-1
                            # most_corr_PnL takes (k_idx+1) for 1-based 'number'
                            try:
                                # Pass the cache for this step
                                profit_k, cluster_info_k = most_corr_PnL(current_pf_c_instance, current_lookback_window, self.evaluation_window, self.historical_data, k_idx+1, cache_for_get_cluster=correlation_cache_for_this_step)

                                if cluster_info_k[0] is None : contribution = np.nan
                                else:
                                    num_tickers_in_k = len(cluster_info_k[2])
                                    total_hist_assets = self.historical_data.shape[1]
                                    baseline = (num_tickers_in_k / total_hist_assets) if total_hist_assets > 0 else 0.0
                                    contribution = (profit_k / last_non_adjusted_return_for_block) - baseline

                                if k_idx < len(most_corr_contribution_lists):
                                     most_corr_contribution_lists[k_idx].append(contribution)
                            except Exception as most_corr_e:
                                # print(f"Err most_corr_PnL k_idx={k_idx}: {most_corr_e}")
                                if k_idx < len(most_corr_contribution_lists):
                                    most_corr_contribution_lists[k_idx].append(np.nan)

                if overall_return_list:
                    temp_overall_df = pd.concat(overall_return_list)
                    current_total_pnl = (1 + temp_overall_df['return']).cumprod().iloc[-1] - 1 if not temp_overall_df.empty else 0.0
                    portfolio_value_tracker.append(1 + current_total_pnl)
                    print(f'step {i}/{number_of_window}, portfolio value (cumulative): {portfolio_value_tracker[-1]:.4f}')
                else: # Should not happen if block_portfolio_returns_df was not empty
                    portfolio_value_tracker.append(1.0)
                    print(f'step {i}/{number_of_window}, portfolio value (cumulative): 1.0000')

            except Exception as e:
                print(f"Error in sliding_window_past_indep step {i}: {e}")
                break

        overall_return_final_df = pd.concat(overall_return_list) if overall_return_list else pd.DataFrame(columns=['return'])
        continuous_cumulative_pnl_from_start = ((1 + overall_return_final_df['return']).cumprod() - 1) if not overall_return_final_df.empty else pd.Series([])

        # block_daily_PnL was about per-block PnL, not directly useful for final reporting without context.
        # Return None or an empty array for it for now as its independent calculation isn't critical for overall PnL.
        return overall_return_final_df, continuous_cumulative_pnl_from_start.values, portfolio_value_tracker, None, Turnovers, most_corr_contribution_lists