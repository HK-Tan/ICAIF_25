import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone
from sklearn.linear_model import Lasso
import scipy.stats as st
from scipy.stats import norm
from scipy import sparse
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from tqdm import tqdm

try:
    from statsmodels.stats.multitest import multipletests
except ImportError:
    print("statsmodels package not found. Attempting to install...")
    import subprocess
    import sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "statsmodels"]
    )
    # This part of the code should go first since importing parallelized_runs already requires the signet package
    from statsmodels.stats.multitest import multipletests

## Clustering packages etc
try:
    from signet.cluster import Cluster
except ImportError:
    print("Signet package not found. Attempting to install from GitHub...")
    import subprocess
    import sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "git+https://github.com/alan-turing-institute/SigNet.git"]
    )
    # This part of the code should go first since importing parallelized_runs already requires the signet package
    from signet.cluster import Cluster

## EconML packages
try:
    from econml.inference import StatsModelsInference
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
    from econml.dml import LinearDML, SparseLinearDML
except ImportError:
    print("econml package not found. Attempting to install...")
    import subprocess
    import sys
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "econml"]
    )
    # This part of the code should go first since importing parallelized_runs already requires the signet package
    from econml.inference import StatsModelsInference
    from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
    from econml.dml import LinearDML, SparseLinearDML

# from parallelized_runs import calculate_pnl

# Helper function to make lagged copies of a DataFrame
def make_lags(df, p):
    """
    Create lagged copies of a DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    p (int): The number of lags to create.

    Returns:
    pd.DataFrame: A DataFrame with lagged columns.
    """
    return pd.concat([df.shift(k).add_suffix(f'_lag{k}') for k in range(1, p+1)], axis=1)

def make_lags_with_orginal(df, p):
    """
    Create lagged copies of a DataFrame and include the original columns.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    p (int): The number of lags to create.

    Returns:
    pd.DataFrame: A DataFrame with the original columns and the lagged columns.
    """
    lagged_df = make_lags(df, p)
    return pd.concat([df, lagged_df], axis=1)

def realign(Y,T,W):
    # Remind me to check again
    full = pd.concat([Y, T, W], axis=1).dropna()
    Y_cols = Y.columns
    T_cols = T.columns
    W_cols = W.columns
    return full[Y_cols], full[T_cols], full[W_cols]

def calculate_weighted_cluster_portfolio_returns(
    asset_returns_lookback_df: pd.DataFrame,
    n_clusters_to_form: int,
    sigma_for_gaussian_weights: float
) -> pd.DataFrame:
    """
    Calculates weighted returns for asset clusters using a streamlined, vectorized approach.

    NOTE: This version has no error handling and assumes perfect input data.

    Args:
        asset_returns_lookback_df (pd.DataFrame): DataFrame of asset returns.
            Rows are timestamps, columns are asset tickers.
        n_clusters_to_form (int): The desired number of clusters to form.
        sigma_for_gaussian_weights (float): Sigma value used in the Gaussian
            weight calculation (controls the spread of weights).

    Returns:
        pd.DataFrame: DataFrame containing the weighted returns for each cluster.
    """
    # --- 1. Calculate Correlation Matrix (More Efficiently) ---
    # Pearson correlation is invariant to scaling, so StandardScaler is redundant.
    correlation_matrix_df = asset_returns_lookback_df.corr(method='pearson').fillna(0)

    # --- 2. Apply Clustering Algorithm (with Vectorized SIGNET Prep) ---
    # Vectorize the creation of positive and negative correlation matrices.
    pos_corr = np.maximum(correlation_matrix_df.values, 0)
    neg_corr = np.maximum(-correlation_matrix_df.values, 0)
    signet_data = (sparse.csc_matrix(pos_corr), sparse.csc_matrix(neg_corr))

    num_assets = correlation_matrix_df.shape[0]
    effective_n_clusters = min(n_clusters_to_form, num_assets)

    # Assuming 'Cluster' is an external class with a 'SPONGE_sym' method
    cluster_obj = Cluster(signet_data)
    labels = cluster_obj.SPONGE_sym(effective_n_clusters)

    # --- 3. Calculate Centroids, Weights, and Returns (Vectorized) ---

    # Calculate all cluster centroids at once using groupby
    # T -> Transpose so assets are rows for easy grouping
    centroids_df = asset_returns_lookback_df.T.groupby(labels).mean().T

    # Create a DataFrame aligned with original returns, where each column is the
    # appropriate centroid for that asset. This prepares for vectorized subtraction.
    aligned_centroids_df = centroids_df.iloc[:, labels]
    aligned_centroids_df.columns = asset_returns_lookback_df.columns

    # Calculate squared Euclidean distance between each asset and its centroid (all at once)
    squared_distances = ((asset_returns_lookback_df - aligned_centroids_df)**2).sum(axis=0)

    # Calculate unnormalized Gaussian weights from distances (all at once)
    unnormalized_weights = np.exp(-squared_distances / (2 * sigma_for_gaussian_weights**2))

    # Normalize weights within each cluster using groupby and transform
    # This ensures weights for assets in the same cluster sum to 1.
    normalized_weights = unnormalized_weights.groupby(labels).transform(lambda x: x / x.sum())

    # Apply normalized weights to asset returns via broadcasting
    weighted_asset_returns = asset_returns_lookback_df * normalized_weights

    # Sum the weighted returns for each cluster using groupby
    cluster_returns_df = weighted_asset_returns.T.groupby(labels).sum().T

    # Rename columns for clarity, matching the original function's output format
    # cluster_returns_df.columns = [f"Cluster_{i + 1}" for i in sorted(np.unique(labels))]

    return cluster_returns_df

#### A library of regressors that can be used with DML
"""
Here, we make use of regressors that automatically processes multiple outputs as running it
through MultiOutputRegressor might be costly in terms of time.
"""
def get_regressor(regressor_name, force_multioutput=False, **kwargs):
    """Factory function to create different regressors with MultiOutput wrapper if needed"""
    base_regressors = {
        'extra_trees': ExtraTreesRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            min_samples_split=kwargs.get('min_samples_split', 20),
            random_state=kwargs.get('random_state', 0)
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=kwargs.get('n_estimators', 100),
            max_depth=kwargs.get('max_depth', 5),
            min_samples_split=kwargs.get('min_samples_split', 20),
            random_state=kwargs.get('random_state', 0)
        ),
        'lasso': Lasso(
            alpha=kwargs.get('alpha', 1.0),
            max_iter=kwargs.get('max_iter', 1000),
            fit_intercept=kwargs.get('fit_intercept', True),
            random_state=kwargs.get('random_state', 0)
        )
    }

    base_model = base_regressors[regressor_name]

    # For Y model (multiple outputs), we might need MultiOutputRegressor
    if force_multioutput:
        return MultiOutputRegressor(base_model)
    else:
        return base_model

def fit_and_predict(Y_df_lagged, W_df_lagged, p, model_y_name, model_y_params, model_t_name, model_t_params, cv_folds):
    T_df_lagged = make_lags(Y_df_lagged, p)
    Y_df_lagged, T_df_lagged, W_df_lagged = realign(Y_df_lagged, T_df_lagged, W_df_lagged)
    Y_df_train, T_df_train, W_df_train = Y_df_lagged.iloc[:-1,:], T_df_lagged.iloc[:-1,:], W_df_lagged.iloc[:-1,:]
    est = LinearDML(
        model_y=get_regressor(model_y_name, force_multioutput=False, **model_y_params),
        model_t=get_regressor(model_t_name, force_multioutput=False, **model_t_params),
        cv=TimeSeriesSplit(n_splits=cv_folds),
        discrete_treatment=False,
        random_state=0
    )
    est.fit(Y_df_train, T_df_train, X=None, W=W_df_train)

    # Prediction step: Y_hat = Y_base (from confounding) + T_next @ theta.T (from the "treatment effect")

    # The structure is: est.models_y[0] contains the 5 CV fold models
    Y_base_folds = []
    for model in est.models_y[0]:
        # Note: iterate through est.models_y[0] (each fold of the CV model), not est.models_y (the CV model)
        pred = model.predict(W_df_test)
        Y_base_folds.append(pred)
    Y_base = np.mean(np.array(Y_base_folds), axis = 0) # Average estimators over the folds
    theta = est.const_marginal_ate()
    return Y_base + T_df_test @ theta.T

"""
To-do: Push the starting days, check if the indices make sense
"""

def rolling_window_OR_VAR_w_para_search(lookback_df, confound_df,
                                        p_max=5,  # maximum number of lags
                                        k = 20, # number of clusters
                                        model_y_name='extra_trees',
                                        model_t_name='extra_trees',
                                        model_y_params=None,
                                        model_t_params=None,
                                        cv_folds=5,
                                        lookback_days=252*4,  # 4 years of daily data
                                        days_valid=20,  # 1 month validation set
                                        error_metric='rmse'):
    """
    THe purpose of this function is to run a rolling window evaluation
    using OR-VAR. The idea is that we will always run this via orthongalized regression
    framework under DML. However, to determine the optimal value of p, we will run a
    "hyperparameter"-like search over the lookback window (to prevent look-ahead bias).

    This is best illustrated with a concrete example:

    Suppose we have a lookback window of 252*4 days (i.e. 4 years of daily data). For a fixed p,
    the base model between the "treatment" and the "outcome" follows a linear relationship (under LinearDML),
    and hence would be compututationally cheap to run. Coupled with machine learning methods for non-linear
    relationships that runs fast (ie ExtraTreesRegressor), it makes it computationally feasible to
    refresh our DML coefficients every day.

    Henceforth, we now split the lookback window into a training and a validation set. For instance, we can set
    aside one month (~20 days) worth of data as a validation set, and use the rest as training data.
    This means that for each value of p, we train on the training set, obtain the validation errors on the validation set,
    and pick the value of p that minimizes the validation error. The validation error used here could either be
    the RSME or the PnL, depending on the use case.

    Remark 1: As part of a potential extension, it is possible to also incorporate an extra parameter/hyperparameter
    for the number of clusters if we are using clustering methods as a dimensionality reduction technique.
    Alternatively, one could also have k to represent the top-k assets instead (top by market cap, etc).

    Remark 2: We are assuming the absence of a hyperparameter here. One could technically add more hyperparameters
    like the size of lookback window, days_valid, etc. However, this would make the search space too large and
    computationally expensive to run while overfitting the model. Hence, we will not do that here.

    Inputs:

    asset_df: DataFrame of asset returns (outcome variable)
    confound_df: DataFrame of confounding variables (confounding variables)
    p_max: A hyperparameter representing the maximum number of lags to consider (e.g. 5).
    model_y_name: Name of the regressor to use for the outcome variable (e.g. 'extra_trees').
    model_t_name: Name of the regressor to use for the treatment variable (e.g. 'extra_trees').
    model_y_params: Dictionary of parameters for the outcome regressor.
    model_t_params: Dictionary of parameters for the treatment regressor.
    cv_folds: Number of cross-validation folds to use.
    lookback_days: Number of days to use for the lookback window
        (e.g. 252*4; assume that this is valid ie data set has more than 4 years worth of daily data).
    days_valid: Number of days to use for validation (e.g. 20; assume that this is less than lookback_days).
    error_metric: Metric to use for validation (e.g. 'rmse' or 'pnl').

    Outputs:

    test_start: The starting index of the test set.
    num_days: The total number of days in the dataset.
    p_optimal: The optimal value of p that minimizes the validation error at the end of the lookback window,
        for each day (hence be a vector of length num_days - test_start).
    """

    if model_y_params is None:
        model_y_params = {}
    if model_t_params is None:
        model_t_params = {}


    test_start = lookback_days  # Start of the test set after training and validation
    num_days = lookback_df.shape[0] - 1  # Total number of days in the dataset,
                                  # minus one day off since we cannot train on the last day
    p_optimal = np.zeros(num_days - test_start)  # Store optimal p for each day in the test set
    Y_hat_next_store = np.zeros((num_days - test_start, k))
    #print("Size of Y_hat_next_store:", Y_hat_next_store.shape)

    asset_df = calculate_weighted_cluster_portfolio_returns(lookback_df, k, .01)

    for day_idx in tqdm(range(test_start, num_days)):
        # The ccomments indicate what happens at day_idx = test_start = 1008, so the train set is w/ index 0 to 1007.
        print("It is day", day_idx, "out of", num_days, "days in the dataset.")
        # First, we perform a parameter search for the optimal p.
        train_start = max(0, day_idx - lookback_days)   # e.g. 0
        train_end = day_idx - days_valid - 1            # e.g. 1008 - 20 - 1 = 987
        valid_start = train_end + 1                     # e.g. 988
        valid_end = valid_start + days_valid - 1        # e.g. 988 + 20 - 1 = 1007; total length = 20

        valid_errors = []

        for p in range(1, p_max + 1):
            current_error = 0
            for valid_shift in range(days_valid):
                # e.g. valid_shift = 0, 19
                start_idx = train_start + valid_shift    # e.g. 0 + 0, 0 + 19 = 19
                end_idx = train_end + valid_shift + 2    # e.g. 987 + 0 + 2, 987 + 19 + 2 = 1008
                # + 2 is to account for the fact that python slicing excludes the last element, and
                # we need to set aside the element at the last index of the train set for validation.

                # Create lagged treatment variables
                # Recall that columns are days, and rows are tickers
                Y_df_lagged = asset_df.iloc[start_idx:end_idx,:].copy() # 0:989 but 989 is excluded, 19:1008 but 1008 excluded
                W_df_lagged = make_lags(confound_df.iloc[start_idx:end_idx,:], p)
                Y_hat_next = fit_and_predict(
                    Y_df_lagged,
                    W_df_lagged,
                    p,
                    model_y_name,
                    model_y_params,
                    model_t_name,
                    model_t_params,
                    cv_folds
                )
                Y_df_test = Y_df_lagged.iloc[-1:,:]

                # Obtain error in the desired metric and accumulate it over the validation window
                if error_metric == 'rmse':
                        current_error += root_mean_squared_error(Y_df_test, Y_hat_next)
                else:
                    raise ValueError("Unsupported error metric.")
            valid_errors.append( (p,current_error) )
        print("Validation errors for different p values:", valid_errors)
        p_opt = min(valid_errors, key=lambda x: x[1])[0]  # Get the p with the minimum validation error
        p_optimal[day_idx - test_start] = p_opt  # Store the optimal p for this day

        # Once we have determined the optimal p value, we now fit with "today's" data set
        # Terminal start_idx = 19
        # Terminal end_idx = 1008 (from results of previous loop at the end of the loop for valid)
        Y_df_lagged = asset_df.iloc[start_idx:end_idx+1,:].copy()  # 19:1009, but 1009 is excluded, so 1008 is the "test" element
        W_df_lagged = make_lags(confound_df.iloc[start_idx:end_idx+1,:], p_opt)
        Y_hat_next_store[day_idx-test_start,:] = fit_and_predict(
            Y_df_lagged,
            W_df_lagged,
            p_opt,
            model_y_name,
            model_y_params,
            model_t_name,
            model_t_params,
            cv_folds
        )

    result = {
        'test_start': test_start,
        'num_days': num_days,
        'p_optimal': p_optimal,
        'Y_hat_next_store': Y_hat_next_store,
    }

    return result


def calculate_pnl(forecast_df, actual_df, pnl_strategy="weighted", contrarian=False):
    """
    This function calculates the PnL based on the forecasted returns and actual returns.

    Inputs:
    forecast_df: DataFrame containing the forecasted returns for each asset/cluster.
    actual_df: DataFrame containing the actual returns for each asset/cluster (in terms of log returns).
    pnl_strategy: Strategy for calculating PnL. Options are:
        - "naive": Go long $1 on clusters with positive forecast return, go short $1 on clusters with negative forecast return.
        - "weighted": Weight based on the predicted return of each cluster.
        - "top": Only choose clusters with absolute returns above average.
    contrarian: If True, inverts the trading signals (bets against forecasts).

    Remark:
    The dataframes keep daily data as rows, with columns as different assets or clusters.
    We also assume that these df are aligned.

    Output:
    Returns a Series with total PnL for each asset/cluster over the entire period.
    """

    # Convert log returns to simple returns for a "factor"
    simple_returns = np.exp(actual_df)

    # Set trading direction: -1 for contrarian, 1 for normal
    direction = -1 if contrarian else 1

    if pnl_strategy == "naive":
        raw_positions = direction * np.sign(forecast_df)
        # Normalize so absolute positions sum to 1 each day
        row_abs_sum = raw_positions.abs().sum(axis=1).replace(0, 1)
        positions = raw_positions.div(row_abs_sum, axis=0)

    elif pnl_strategy == "weighted":
        row_abs_sum = forecast_df.abs().sum(axis=1).replace(0, 1)
        positions = direction * forecast_df.div(row_abs_sum, axis=0)

    elif pnl_strategy == "top":
        positions = pd.DataFrame(0, index=forecast_df.index, columns=forecast_df.columns)

        for col in forecast_df.columns:
            threshold = forecast_df[col].abs().mean()
            positions.loc[forecast_df[col] > threshold, col] = direction
            positions.loc[forecast_df[col] < -threshold, col] = -direction

        row_sums = positions.abs().sum(axis=1).replace(0, 1)
        positions = positions.div(row_sums, axis=0)

    # Calculate daily portfolio returns
    daily_pnl = positions * simple_returns
    daily_portfolio_returns = daily_pnl.sum(axis=1)

    # CORRECT: Compound the returns
    #cumulative_returns = daily_portfolio_returns.cumprod() - 1
    cumulative_returns = (daily_portfolio_returns.cumsum())/ len(daily_portfolio_returns)

    return cumulative_returns