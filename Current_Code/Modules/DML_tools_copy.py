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
from multiprocessing import Pool
import time
from datetime import datetime
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
    asset_returns_df: pd.DataFrame,
    lookback_period: int,
    n_clusters_to_form: int,
    sigma_for_gaussian_weights: float
) -> pd.DataFrame:
    """
    Calculates weighted returns for asset clusters using a streamlined, vectorized approach.

    NOTE: This version has no error handling and assumes perfect input data.

    Args:
        asset_returns_df (pd.DataFrame): DataFrame of asset returns.
            Rows are timestamps, columns are asset tickers.
        n_clusters_to_form (int): The desired number of clusters to form.
        sigma_for_gaussian_weights (float): Sigma value used in the Gaussian
            weight calculation (controls the spread of weights).

    Returns:
        pd.DataFrame: DataFrame containing the weighted returns for each cluster.
    """
    # --- 1. Calculate Correlation Matrix (More Efficiently) ---
    # Pearson correlation is invariant to scaling, so StandardScaler is redundant.
    correlation_matrix_df = asset_returns_df[:lookback_period].corr(method='pearson').fillna(0)

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
    centroids_df = asset_returns_df.groupby(labels, axis=1).mean()

    # Create a DataFrame aligned with original returns, where each column is the
    # appropriate centroid for that asset. This prepares for vectorized subtraction.
    aligned_centroids_df = centroids_df.iloc[:, labels]
    aligned_centroids_df.columns = asset_returns_df.columns

    # Calculate squared Euclidean distance between each asset and its centroid (all at once)
    squared_distances = ((asset_returns_df - aligned_centroids_df)**2).sum(axis=0)

    # To prevent numerical underflow when sigma is very small, we stabilize the
    # exponent calculation. This is a standard trick (similar to the log-sum-exp trick).

    # 1. Calculate the raw exponent values. Add a small epsilon to sigma to avoid division by zero if sigma is exactly 0.
    epsilon = 1e-12
    exponent_vals = -squared_distances / (2 * (sigma_for_gaussian_weights**2 + epsilon))

    # 2. Subtract the maximum exponent *within each cluster*.
    # This shifts the exponent range for each cluster so that the max value is 0.
    # The `transform` function broadcasts the result back to the original shape.
    max_exponent_per_cluster = exponent_vals.groupby(labels).transform('max')
    stable_exponent_vals = exponent_vals - max_exponent_per_cluster

    # 3. Calculate unnormalized weights using the stabilized exponents.
    # Now, the largest weight in each cluster will be exp(0) = 1.
    # This guarantees the sum of weights per cluster is always >= 1, preventing division by zero.
    unnormalized_weights = np.exp(stable_exponent_vals)

    # Normalize weights within each cluster using groupby and transform
    # This ensures weights for assets in the same cluster sum to 1.
    normalized_weights = unnormalized_weights.groupby(labels).transform(lambda x: x / x.sum())

    # Apply normalized weights to asset returns via broadcasting
    weighted_asset_returns = (np.exp(asset_returns_df) - 1) * normalized_weights

    # Sum the weighted returns for each cluster using groupby
    cluster_returns_df = np.log(1 + weighted_asset_returns.groupby(labels, axis=1).sum())

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
    W_df_test, T_df_test = W_df_lagged.iloc[-1:,:], T_df_lagged.iloc[-1:,:]
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

def rolling_window_OR_VAR_w_para_search(whole_df, confound_df,
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
    num_days = whole_df.shape[0] - 1  # Total number of days in the dataset,
                                  # minus one day off since we cannot train on the last day
    p_optimal = np.zeros(num_days - test_start)  # Store optimal p for each day in the test set
    Y_hat_next_store = np.zeros((num_days - test_start, k))
    #print("Size of Y_hat_next_store:", Y_hat_next_store.shape)

    asset_df = calculate_weighted_cluster_portfolio_returns(whole_df, lookback_days, k, .05)

    if len(asset_df) < lookback_days + 1 or lookback_days <= days_valid:
        raise ValueError("Dataset is too small for the specified lookback_days and days_valid.")

    for day_idx in range(test_start, num_days):
        # The comments indicate what happens at day_idx = test_start = 1008, so the train set is w/ index 0 to 1007.
        print("It is day", day_idx, "out of", num_days, "days in the dataset.")
        # First, we perform a parameter search for the optimal p.
        train_start = max(0, day_idx - lookback_days)   # e.g. 0
        train_end = day_idx - days_valid                # e.g. 1008 - 20 = 988
        valid_start = train_end + 1                     # e.g. 989
        valid_end = valid_start + days_valid - 1        # e.g. 989 + 20 - 1 = 1008; total length = 20

        valid_errors = []
        for p in range(1, p_max + 1):
            current_error = 0
            for valid_shift in range(days_valid):
                # e.g. valid_shift = 0, 19
                start_idx = train_start + valid_shift    # e.g. 0 + 0, 0 + 19 = 19
                end_idx = train_end + valid_shift + 2    # e.g. 988 + 0 + 2, 988 + 19 + 2 = 1009 (usually excluded)
                # + 2 is to account for the fact that python slicing excludes the last element, and
                # we need to set aside the element at the last index of the train set for validation.

                # Create lagged treatment variables
                # Recall that columns are days, and rows are tickers
                Y_df_lagged = asset_df.iloc[start_idx:end_idx,:].copy() # 0:989 but 989 is excluded, 19:1009 but 1009 excluded
                W_df_lagged = make_lags(confound_df.iloc[start_idx:end_idx,:], p)
                Y_df_test = Y_df_lagged.iloc[-1:,:]
                # In the last value of valid_shift = 19, then 19:1008 (1008 included) but we took out the 1008th element for

                #  validation (hence :-1 in train), so we have 19:1007 (1007 included) for training.

                # For validation, note that the predictive features are from T_df_pred and W_df_pred, to predict a value of Y_df_pred

                #  To test/evaluate the error, Y_df_pred is "true" (ie the correct Y at index 1008), and we predict this

                #  with the prediction part of this ie Y_hat_next, using T_df_lagged at 1008 and W_df_lagged at 1008.

                #  Here, this is correct since T_df_lagged and W_df_lagged contains minimially lag 1 variables.
                Y_hat_next = fit_and_predict(Y_df_lagged, W_df_lagged, p, model_y_name, model_y_params, model_t_name, model_t_params, cv_folds)

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
        # Recalculate indices for the full lookback window
        final_start_idx = max(0, day_idx - lookback_days)  # Use full lookback window  # 1008 - 1008 = 0
        final_end_idx = day_idx + 2

        # Max value of day_idx is num_days - 1, ie 1298, so we are allowed to get up

        #   to day_idx + 2 = 1300 (exclusive)
        Y_df_lagged = asset_df.iloc[final_start_idx:final_end_idx+1,:].copy()  # Include current day for prediction
        W_df_lagged = make_lags(confound_df.iloc[final_start_idx:final_end_idx+1,:], p_opt)
        # Note that the full data sets (up till train_end) are used for training

        # For prediction, the corresponding lagged variables as obtained in the last row of the

        #   individual dataframe will be used to predict the outcome (ie "next day") since a time

        #   series prediction model is written as a function of the previous values (time steps).

        # Since these are lagged, obtaining the last row of it (apart from the outcome) even along the row

        #    at which we are predicting, would still be the lagged values!
        Y_hat_next_store[day_idx-test_start,:] = fit_and_predict(Y_df_lagged, W_df_lagged, p_opt, model_y_name, model_y_params, model_t_name, model_t_params, cv_folds)
        # For day_idx = 1008, final_end_idx = 1008 + 2 = 1010, slicing includes the index = 1009 slice, which we

        #    remove so that we are training up till index 1008 (inclusive), and updating the predicted Y_hat

        #    which corresponds to day with index 1009.

        # 1009 to 1299, so 291 days; last index is 1298 - 1008 = 290 (hence consistent).
    result = {
        'test_start': test_start,
        'num_days': num_days,
        'p_optimal': p_optimal,
        'Y_hat_next_store': Y_hat_next_store,
    }

    return result


def create_index_mapping(T_names_prev, T_names_post, d_y):
    """
    Create index mapping as 2-tuples (pre_idx, post_idx) for coefficient comparison.

    Inputs:
    - T_names_prev: List of treatment variable names from the previous coefficients.
    - T_names_post: List of treatment variable names from the post coefficients.
    - d_y: Number of assets (ie outcome) in the model, placed as the outcome variables.

    Returns:
    - idx_pairs: List of (pre_idx, post_idx) tuples for existing coefficients
        i.e. Of the previous coefficients, which ones do they map to in the post coefficients?
    - idx_new: List of post_idx for new coefficients
    """
    d_T_prev = len(T_names_prev)
    d_T_post = len(T_names_post)

    idx_pairs = []
    idx_new = []

    # Create mapping for all coefficients
    for y in range(d_y):
        for j_post, name_post in enumerate(T_names_post):
            post_idx = y * d_T_post + j_post
            if name_post in T_names_prev:
                # This is an existing coefficient
                j_prev = T_names_prev.index(name_post)
                pre_idx = y * d_T_prev + j_prev
                idx_pairs.append((pre_idx, post_idx))
            else:
                # If not, then this is a new coefficient
                idx_new.append(post_idx)

    return idx_pairs, idx_new

def rolling_window_ORACLE_VAR(asset_df, confound_df,
                                        p_max=5,  # maximum number of lags
                                        model_y_name='extra_trees',
                                        model_t_name='extra_trees',
                                        model_y_params=None,
                                        model_t_params=None,
                                        cv_folds=5,
                                        lookback_days=252*4,  # 4 years of daily data
                                        significance_level=0.15,  # Significance level for p-value testing
                                        error_metric='rmse'):
    """
    THe purpose of this function is to run a rolling window evaluation
    using ORACLE-VAR. The idea is that we will always run this via orthongalized regression
    framework under DML. To determine the optimal value of p, we perform a multiple-hypothesis testing
    approach iteratively as described in the paper. This removes the need to obtain validation error, and
    instead, we will be looking at the p-values and specify a given significance level.

    This is best illustrated with a concrete example:

    Suppose we have a lookback window of 252*4 days (i.e. 4 years of daily data). We do this process iteratively
    by first assuming that p = 1, so we assign the following variables:

    Treatment = asset_lag1
    Outcome = asset_lag0
    Confounding = confound_lag1, confound_lag2

    Next, we start by appending asset_lag2 to Treatment, and look at the p-value of the new coefficient block
    between asset_lag2 and asset_lag0 (for testing if these coefficients are zero). If the p-value is less than the
    significance level, we keep asset_lag2 in the model. Otherwise, we test to see if the coefficients for the
    previous treatment variable (asset_lag1) has been changed. If so, we instead move asset_lag2 to the confounding variables.

    Hence, if p_value > significance_level, for both cases, the algorithm terminates and we keep the current p value
    with the relevant assignment of treatment, and confounding variables. If not, repeat the process by appending
    confound_lag3 to the confounding variables, and testing the p-values of the coefficients for testing asset_lag3
    in the treatment variable (note that asset_lag0 is always the outcome variable).

    Remark 1: significance_level is a hyperparameter that we can set to control the false discovery rate/family-wise error rate.
        By default, we set it to 0.15, which is a common value used in practice for "discovery"-style/feature selection tasks.

    Remark 2: The code will assume that no validation set is needed, and that the entire lookback window is used for training.

    Inputs:

    asset_df: DataFrame of asset returns (outcome variable)
    confound_df: DataFrame of confounding variables (confounding variables)
    p_max: A hyperparameter representing the maximum number of lags to consider (e.g. 5) before we
        terminate the iterative algorithm (though it is possible for one iteration to terminate before 5).
    model_y_name: Name of the regressor to use for the outcome variable (e.g. 'extra_trees').
    model_t_name: Name of the regressor to use for the treatment variable (e.g. 'extra_trees').
    model_y_params: Dictionary of parameters for the outcome regressor.
    model_t_params: Dictionary of parameters for the treatment regressor.
    cv_folds: Number of cross-validation folds to use.
    lookback_days: Number of days to use for the lookback window
        (e.g. 252*4; assume that this is valid ie data set has more than 4 years worth of daily data).
    significance_level: Significance level for p-value testing (e.g. 0.15).
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
    num_days = asset_df.shape[0] - 1  # Total number of days in the dataset,
                                  # minus one day off since we cannot train on the last day
    p_optimal = np.zeros(num_days - test_start)  # Store optimal p for each day in the test set
    which_lag_control = np.zeros((num_days - test_start, p_max+1))  # Store which lag is in the control set for each day
    which_lag_treatment = np.zeros((num_days - test_start, p_max+1))  # Store which lag is in the treatment set for each day
    which_lag_treatment[:,1] = 1  # The first lag is always in treatment variable  (rows = days, columns = lags)
    # Here, to match the column index with the lag value, we set the column with index 1 (ie second) to be 1.
    Y_hat_next_store = np.zeros((num_days - test_start, asset_df.shape[1]))

    if len(asset_df) < lookback_days + 1:
        raise ValueError("Dataset is too small for the specified lookback_days and days_valid.")

    for day_idx in range(test_start, num_days):
        # The comments indicate what happens at day_idx = test_start = 1008, so the train set is w/ index 0 to 1007.
        print("It is day", day_idx, "out of", num_days, "days in the dataset.")

        ###########################################################
        ### Training Part
        ###########################################################
        train_start = max(0, day_idx - lookback_days)      # e.g. 0
        train_end = day_idx + 1                            # e.g. 1009
        final_end_idx = train_end + 1                      # e.g. 1010

        # Outcome; same for both pre and post models
        Y_df_lagged = asset_df.iloc[train_start:final_end_idx,:].copy()
            # Exclusive of the last day, so 0:1009 (1009 is excluded) for training
            # This makes sense since "today" is day index of 1008, and we already know the value of the asset
            #   and hence, are allowed to train on it.
            # However, we pre-append the next day (1009) so that we can predict the value of the asset
            #   at the next day (1009) using the output configuration from the algorithm

        # Confounding variables; same for both pre and post models
        W_df_lagged = make_lags(confound_df.iloc[train_start:final_end_idx,:], 1)

        ### Initializing the pre model outside of the loop (before p = 2)
        T_df_pre_lagged = make_lags(asset_df.iloc[train_start:final_end_idx,:].copy(), 1)
        Y_df_pre_lagged, T_df_pre_lagged, W_df_pre_lagged = realign(Y_df_lagged, T_df_pre_lagged, W_df_lagged)

        """
        At this stage, (p = 1), we have the following variables:

        Pre Model:
        Treatment = asset_lag1
        Outcome = asset_lag0
        Control/Confounding = confound_lag1
        """

        last_valid_p = 1
        for p in range(2, p_max + 1):

            ##### Starting value, p = 2 (though this loops till p = 5 unless terminated early)

            """
            Pre Model (Asumming p = 2):
            Treatment = asset_lag1
            Outcome = asset_lag0
            Control/Confounding = confound_lag1, confound_lag2

            Post Model (Asumming p = 2):
            Treatment = asset_lag1, asset_lag2
            Outcome = asset_lag0
            Control/Confounding = confound_lag1, confound_lag2
            """

            W_df_pre_lagged = pd.concat([
                W_df_pre_lagged,
                confound_df.iloc[train_start:final_end_idx,:].shift(p).add_suffix(f'_lag{p}')
            ], axis=1)

            # Confounding variables for post model are the same as pre-model initially
            W_df_post_lagged = W_df_pre_lagged.copy()

            # Note that T_df_pre_lagged is already created at the end of the hypothesis testing loops
            #   and was designed to be carried over to the next iteration here.

            # Create post-model treatment variables (pre + current lag p)
            T_df_post_lagged = pd.concat([
                T_df_pre_lagged,
                asset_df.iloc[train_start:final_end_idx,:].shift(p).add_suffix(f'_lag{p}')
            ], axis=1)

            # Realign models
            Y_df_pre_lagged, T_df_pre_lagged, W_df_pre_lagged = realign(Y_df_lagged, T_df_pre_lagged, W_df_pre_lagged)
            Y_df_post_lagged, T_df_post_lagged, W_df_post_lagged = realign(Y_df_lagged, T_df_post_lagged, W_df_post_lagged)
            # Note that the first argument is Y_df_lagged, no pre or post so that we can realign the data
            #   properly with the NaN introduced by the shift operation.

            ### Pre Model
            est_pre = LinearDML(
                model_y=get_regressor(model_y_name, force_multioutput=False, **model_y_params),
                model_t=get_regressor(model_t_name, force_multioutput=False, **model_t_params),
                cv=TimeSeriesSplit(n_splits=cv_folds),
                discrete_treatment=False,
                random_state=0
            )

            est_pre.fit(Y_df_pre_lagged.iloc[:-1,:], T_df_pre_lagged.iloc[:-1,:], X=None,
                        W=W_df_pre_lagged.iloc[:-1,:], inference=StatsModelsInference())
            # The -1 one here is to exclude the last row, which is the prediction row (ie data for the next day)
            est_pre_inf = est_pre.const_marginal_ate_inference()
            theta_pre = est_pre_inf.mean_point.ravel()  # Flatten to 1-D vector
            cov_pre = est_pre_inf.mean_pred_stderr.ravel()

            ### Post Model
            est_post = LinearDML(
                model_y=get_regressor(model_y_name, force_multioutput=False, **model_y_params),
                model_t=get_regressor(model_t_name, force_multioutput=False, **model_t_params),
                cv=TimeSeriesSplit(n_splits=cv_folds),
                discrete_treatment=False,
                random_state=0,
            )

            est_post.fit(Y_df_post_lagged.iloc[:-1,:], T_df_post_lagged.iloc[:-1,:], X=None,
                         W=W_df_post_lagged.iloc[:-1,:], inference=StatsModelsInference())
            est_post_inf = est_post.const_marginal_ate_inference()
            theta_post = est_post_inf.mean_point.ravel()  # Flatten to 1-D vector
            cov_post = est_post_inf.mean_pred_stderr.ravel()

            T_names_pre = T_df_pre_lagged.columns.tolist()
            T_names_post = T_df_post_lagged.columns.tolist()
            d_y   = asset_df.shape[1]                    # number of outcome series
            # Columns are not affected by the pre-append trick

            idx_pairs, idx_post = create_index_mapping(T_names_pre, T_names_post, d_y)

            # Within each test, we perform a Benjamini–Hochberg FDR test
            # p-values of signifinance
            p_sig_store = []
            for i in idx_post:
                z_stat = theta_post[i] / cov_post[i]
                p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
                p_sig_store.append(p_value)
            p_sig = min(multipletests(p_sig_store, alpha=significance_level, method="fdr_bh")[1])
            print(p_sig)

            # p-value of drift without significance
            p_drift_store = []
            for i,j in idx_pairs:
                z_stat = (theta_post[j] - theta_pre[i])/ np.sqrt(cov_post[j]**2 + cov_pre[i]**2)
                p_value = 2 * (1 - norm.cdf(np.abs(z_stat)))
                p_drift_store.append(p_value)
            p_drift = min(multipletests(p_drift_store, alpha=significance_level, method="fdr_bh")[1])
            print(p_drift)

            # Case 1: Significance of new coefficients
            if p_sig < significance_level:
                print(f"✓ Significance detected for lag {p} (p-value: {p_sig:.4f})")
                which_lag_treatment[day_idx-test_start, p] = 1  # Mark this lag as treatment
                T_df_pre_lagged = T_df_post_lagged
                # Y_df_pre_lagged = Y_df_post_lagged  (true but not necessary)
                W_df_pre_lagged = W_df_post_lagged
                last_valid_p = p
                continue

            # Case 2: Drift without Significance
            elif p_drift < significance_level:
                print(f"✓ Drift detected for lag {p} (p-value: {p_drift:.4f})")
                which_lag_control[day_idx-test_start, p] = 1  # Mark this lag as control/confounding
                # Shift the treatment variable to the confounding
                # First, create that asset_lagged variable with just lag p assets
                asset_lag_p = asset_df.iloc[train_start:final_end_idx,:].shift(p).add_suffix(f'_lag{p}')
                W_df_pre_lagged = pd.concat([W_df_pre_lagged, asset_lag_p], axis=1)
                Y_df_pre_lagged, T_df_pre_lagged, W_df_pre_lagged = realign(Y_df_lagged, T_df_pre_lagged, W_df_pre_lagged)
                # Note that the first argument is Y_df_lagged, no pre or post so that we can realign the data
                #   properly with the NaN introduced by the shift operation.
                # Furthermore, we do not need to update the T_df_pre_lagged since that was before we added
                #   the asset_lag_p variable (as seen from above, we have added it to W_df_pre_lagged instead).
                last_valid_p = p
                continue

            # Case 3: Neither
            else:
                print(f"✗ No significance or drift for lag {p} (p_sig: {p_sig:.4f}, p_drift: {p_drift:.4f})")
                last_valid_p = p - 1  # Store the optimal p for this day (this p didn't pass so p - 1))
                break

        p_optimal[day_idx - test_start] = last_valid_p  # Store the optimal p for this day
        ###########################################################
        ### Prediction Part
        ###########################################################

        # In all the cases, we have the final pre-model with the optimal p value. Hence, these will be used
        #   to make the predictions.

        # Re-initialize the final pre-model with the optimal p value
        # Note that there is no need to rebuild pre, even though we are predicting for the next day.
        # The trick of tracking the variables in the pre-model is that we always have this last row (next day)
        #   already included (and when we are training, we just slice that out), so that the full unsliced
        #   dataframe contains the last row (ie next day lagged values) that we can use to predict the next day.

        est = LinearDML(
            model_y=get_regressor(model_y_name, force_multioutput=False, **model_y_params),
            model_t=get_regressor(model_t_name, force_multioutput=False, **model_t_params),
            cv=TimeSeriesSplit(n_splits=cv_folds),
            discrete_treatment=False,
            random_state=0
        )

        # Note that the full data sets (up till train_end) are used for training
        # For prediction, the corresponding lagged variables as obtained in the last row of the
        #   individual dataframe will be used to predict the outcome (ie "next day") since a time
        #   series prediction model is written as a function of the previous values (time steps).
        est.fit(Y_df_pre_lagged.iloc[:-1,:], T_df_pre_lagged.iloc[:-1,:], X=None, W=W_df_pre_lagged.iloc[:-1,:])
        # Remember that we can only train on data without information on the next day, which is
        #    the [-1,:] slice of the dataframe.

        # Prediction step: Y_hat = Y_base (from confounding) + T_next @ theta.T (from the "treatment effect")
        Y_df_pred, T_df_pred, W_df_pred = Y_df_pre_lagged.iloc[-1:,:], T_df_pre_lagged.iloc[-1:,:], W_df_pre_lagged.iloc[-1:,:]
        # The structure is: est.models_y[0] contains the 5 CV fold models
        Y_base_folds = []
        for model in est.models_y[0]:
            # Note: iterate through est.models_y[0] (each fold of the CV model), not est.models_y (the CV model)
            pred = model.predict(W_df_pred)
            Y_base_folds.append(pred)
        Y_base = np.mean(np.array(Y_base_folds), axis = 0) # Average estimators over the folds
        theta = est.const_marginal_ate()
        Y_hat_next_store[day_idx-test_start,:] = Y_base + T_df_pred @ theta.T

    result = {
        'test_start': test_start,
        'num_days': num_days,
        'p_optimal': p_optimal,
        'Y_hat_next_store': Y_hat_next_store,
        'which_lag_control': which_lag_control,
        'which_lag_treatment': which_lag_treatment,
    }

    return result

# IZ: Function definition for a single training error evaluation, used for parallelization
def evaluate_training_run(curr_cfg, asset_df, confound_df, lookback_days, days_valid,
                    model_y_name, model_y_params, model_t_name, model_t_params,
                    cv_folds, error_metric):

    (day_idx, p, valid_shift) = curr_cfg

    # The comments indicate what happens at day_idx = test_start = 1008, so the train set is w/ index 0 to 1007.
    train_start = max(0, day_idx - lookback_days)   # e.g. 0
    train_end = day_idx - days_valid - 1            # e.g. 1008 - 20 - 1 = 987
    valid_start = train_end + 1                     # e.g. 988
    valid_end = valid_start + days_valid - 1        # e.g. 988 + 20 - 1 = 1007; total length = 20

    # e.g. valid_shift = 0, 19
    start_idx = train_start + valid_shift    # e.g. 0 + 0, 0 + 19 = 19
    end_idx = train_end + valid_shift + 2    # e.g. 987 + 0 + 2, 987 + 19 + 2 = 1008 (usually excluded)
    # + 2 is to account for the fact that python slicing excludes the last element, and
    # we need to set aside the element at the last index of the train set for validation.

    # Create lagged treatment variables
    # Recall that columns are days, and rows are tickers
    Y_df_lagged = asset_df.iloc[start_idx:end_idx,:].copy() # 0:989 but 989 is excluded, 19:1008 but 1008 excluded
    W_df_lagged = make_lags(confound_df.iloc[start_idx:end_idx,:], p)
    Y_df_test = Y_df_lagged.iloc[-1:,:]
    # In the last value of valid_shift = 19, then 19:1007 (1007 included) but we took out the 1007th element for
    #  validation, so we have 19:1006 (1006 included) for training and 1007 for "internal validation".
    Y_hat_next = fit_and_predict(Y_df_lagged, W_df_lagged, p, model_y_name, model_y_params, model_t_name, model_t_params, cv_folds)

    if error_metric == 'rmse':
        # IZ: This should be a returned value in the function - will get automatically pushed to a results list by Pool
        return root_mean_squared_error(Y_df_test, Y_hat_next)
    else:
        raise ValueError("Unsupported error metric.")


# IZ: Function definition for a single future prediction (using found optimal p), used for parallelization
def evaluate_prediction(day_idx, asset_df, confound_df, lookback_days, p_opt,
                    model_y_name, model_y_params, model_t_name, model_t_params, cv_folds):

    # IZ: Force p_opt to an integer to prevent any issues with indexing in the make_lag function
    p_opt = int(p_opt)

    # IZ: Once we have determined the optimal p value, we now fit with "today's" data set
    # IZ: Recalculate indices for the full lookback window
    final_start_idx = max(0, day_idx - lookback_days)  # Use full lookback window  # 1008 - 1008 = 0
    final_end_idx = day_idx  # Up to current day (exclusive in slicing), ie 1008

    Y_df_lagged = asset_df.iloc[final_start_idx:final_end_idx+1,:].copy()  # Include current day for prediction
    W_df_lagged = make_lags(confound_df.iloc[final_start_idx:final_end_idx+1,:], p_opt)
    return fit_and_predict(Y_df_lagged, W_df_lagged, p_opt, model_y_name, model_y_params, model_t_name, model_t_params, cv_folds)


def parallel_rolling_window_OR_VAR_w_para_search(whole_df, confound_df,
                                                 p_max=5,  # maximum number of lags
                                                 k = 20, # number of clusters
                                                 model_y_name='extra_trees',
                                                 model_t_name='extra_trees',
                                                 model_y_params=None,
                                                 model_t_params=None,
                                                 cv_folds=5,
                                                 lookback_days=252*4,  # 4 years of daily data
                                                 days_valid=20,  # 1 month validation set
                                                 error_metric='rmse',
                                                 max_threads=1):

    start_exec_time = time.time()
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if model_y_params is None:
        model_y_params = {}
    if model_t_params is None:
        model_t_params = {}


    test_start = lookback_days  # Start of the test set after training and validation
    num_days = whole_df.shape[0] - 1  # Total number of days in the dataset,
                                  # minus one day off since we cannot train on the last day

    p_optimal = np.zeros(num_days - test_start, dtype=int)  # Store optimal p for each day in the test set
    Y_hat_next_store = np.zeros((num_days - test_start, k))
    #print("Size of Y_hat_next_store:", Y_hat_next_store.shape)

    asset_df = calculate_weighted_cluster_portfolio_returns(whole_df, lookback_days, k, .05)

    if len(asset_df) < lookback_days + 1 or lookback_days <= days_valid:
        raise ValueError("Dataset is too small for the specified lookback_days and days_valid.")

    runs_configs_list = []
    for day_idx in range(test_start, num_days):
        for p in range(1, p_max + 1):
            for valid_shift in range(days_valid):
                runs_configs_list.append((day_idx, p, valid_shift))


    # IZ: This main loop can be parallelized across the list of run configurations, i.e. (day, p, shift) tuples
    # It should return a list of results of the error metric for each of those
    # TODO: Parallelize this
    print("Beginning search for optimal VAR order for each day")
    with Pool(max_threads) as pool:
        error_metric_results = pool.starmap(
            evaluate_training_run,
            [(curr_cfg, asset_df, confound_df, lookback_days, days_valid,
            model_y_name, model_y_params, model_t_name, model_t_params,
            cv_folds, error_metric) for curr_cfg in runs_configs_list]
        )

    # IZ: Load the run configurations and error metric results into a dataframe for aggregation
    runs_df = pd.DataFrame([(*cfg,err) for cfg,err in zip(runs_configs_list,error_metric_results)],
                           columns=["day_idx", "p", "valid_shift", "error_metric"])
    # Group by day_index and p to get the cumulative errors per day/p combination over the training set
    runs_df_sum_error = runs_df.groupby(['day_idx', 'p']).sum()
    # Now group by the day index again to find the p value that gives minimum train error
    # First find the df indices corresponding to the minimum error rows per day_idx
    p_opt_indices = runs_df_sum_error.groupby('day_idx')['error_metric'].idxmin()
    # Take those indices from the original dataframe, and convert to a list of tuples of (day_idx, p_opt)
    runs_df_p_opt = runs_df_sum_error.loc[p_opt_indices]
    day_p_opt_tuples = runs_df_p_opt.index.tolist()

    print("Per-day optimal VAR order (p) and corresponding validation error:")
    print(runs_df_p_opt.reset_index().values.tolist())

    for (day_idx, p_opt) in day_p_opt_tuples:
        # print((day_idx, p_opt))
        p_optimal[day_idx-test_start] = p_opt

    print("Completed VAR order search")
    print(f"Total elapsed time: {time.time()-start_exec_time:.4f} seconds")


    # IZ: Now we test the optimal found p's on the test sets, this can be parallelized over the day indices as well
    print("Computing daily predictions using the observed optimal VAR order")
    with Pool(max_threads) as pool:
        Y_hat_next_store = pool.starmap(
            evaluate_prediction,
            [(day_idx, asset_df, confound_df, lookback_days, p_optimal[day_idx-test_start],
            model_y_name, model_y_params, model_t_name, model_t_params, cv_folds)
            for day_idx in range(test_start, num_days)]
    )

    result = {
        'test_start': test_start,
        'num_days': num_days,
        'p_optimal': p_optimal,
        'Y_hat_next_store': Y_hat_next_store,
    }

    print("Completed predictions")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Elapsed time: {time.time()-start_exec_time:.4f} seconds")

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
    # Percentage change is given by exp(log_return) - 1
    simple_returns = np.exp(actual_df) - 1

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
    #print(positions)
    daily_pnl = positions * simple_returns
    daily_portfolio_returns_per = daily_pnl.sum(axis=1)
    print("Daily portfolio returns (percentage change):", daily_portfolio_returns_per)
    daily_portfolio_returns = daily_portfolio_returns_per + 1

    cumulative_returns = daily_portfolio_returns.cumprod() - 1

    return cumulative_returns, daily_portfolio_returns_per
