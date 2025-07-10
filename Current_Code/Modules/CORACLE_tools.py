import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

## EconML packages
import scipy.stats as st
from scipy.stats import norm
from econml.inference import StatsModelsInference
from econml.sklearn_extensions.linear_model import StatsModelsLinearRegression
from statsmodels.stats.multitest import multipletests
from econml.dml import LinearDML, SparseLinearDML
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.metrics import root_mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.base import clone
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
from multiprocessing import Pool
import time
from datetime import datetime
import gc

# from parallelized_runs import calculate_pnl

# Helper function to make lagged copies of a DataFrame
def make_lags(df, p):
    """
    Create lagged copies of a DataFrame (withtout the original columns; ie starting from lag 1).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    p (int): The number of lags to create.

    Returns:
    pd.DataFrame: A DataFrame with lagged columns.
    """
    if not isinstance(p, int): raise ValueError(f"Value of p for computing lags must be an integer, acutal value is p={p}")
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


def rolling_window_OR_VAR_w_para_search(asset_df, confound_df,
                                        p_max=5,  # maximum number of lags
                                        model_y_name='extra_trees',
                                        model_t_name='extra_trees',
                                        model_y_params=None,
                                        model_t_params=None,
                                        cv_folds=5,
                                        lookback_days=252*4,  # 4 years of daily data
                                        days_valid=20,  # 1 month validation set
                                        error_metric='rmse'):
    """
    The purpose of this function is to run a rolling window evaluation
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
    num_days = asset_df.shape[0] - 1  # Total number of days in the dataset,
                                      # minus one day off since we cannot train on the last day; ie 1299
    p_optimal = np.zeros(num_days - test_start)  # Store optimal p for each day in the test set
    Y_hat_next_store = np.zeros((num_days - test_start, asset_df.shape[1]))

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
                # we need to set aside the element at the last row for validation.

                # Create lagged treatment variables
                # Recall that columns are days, and rows are tickers
                Y_df_lagged = asset_df.iloc[start_idx:end_idx,:] # 0:989 but 989 is excluded, 19:1009 but 1009 excluded
                W_df_lagged = make_lags(confound_df.iloc[start_idx:end_idx,:], p)
                T_df_lagged = make_lags(Y_df_lagged, p)
                Y_df_lagged, T_df_lagged, W_df_lagged = realign(Y_df_lagged, T_df_lagged, W_df_lagged)
                Y_df_train, T_df_train, W_df_train = Y_df_lagged.iloc[:-1,:], T_df_lagged.iloc[:-1,:], W_df_lagged.iloc[:-1,:]
                Y_df_pred, T_df_pred, W_df_pred = Y_df_lagged.iloc[-1:,:], T_df_lagged.iloc[-1:,:], W_df_lagged.iloc[-1:,:]
                # In the last value of valid_shift = 19, then 19:1008 (1008 included) but we took out the 1008th element for
                #  validation (hence :-1 in train), so we have 19:1007 (1007 included) for training.
                # For validation, note that the predictive features are from T_df_pred and W_df_pred, to predict a value of Y_df_pred
                #  To test/evaluate the error, Y_df_pred is "true" (ie the correct Y at index 1008), and we predict this
                #  with the prediction part of this ie Y_hat_next, using T_df_lagged at 1008 and W_df_lagged at 1008.
                #  Here, this is correct since T_df_lagged and W_df_lagged contains minimially lag 1 variables.

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
                    pred = model.predict(W_df_pred)
                    Y_base_folds.append(pred)

                Y_base = np.mean(np.array(Y_base_folds), axis = 0) # Average estimators over the folds
                theta = est.const_marginal_ate()
                Y_hat_next = Y_base + T_df_pred @ theta.T

                # Obtain error in the desired metric and accumulate it over the validation window
                if error_metric == 'rmse':
                        current_error += root_mean_squared_error(Y_df_pred, Y_hat_next)
                else:
                    raise ValueError("Unsupported error metric.")
                
                # Memory optimization: cleanup model
                del est
                gc.collect()
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

        Y_df_lagged = asset_df.iloc[final_start_idx:final_end_idx,:]  # Include current day for prediction
        W_df_lagged = make_lags(confound_df.iloc[final_start_idx:final_end_idx,:], p_opt)
        T_df_lagged = make_lags(Y_df_lagged, p_opt)
        Y_df_lagged, T_df_lagged, W_df_lagged = realign(Y_df_lagged, T_df_lagged, W_df_lagged)
        # Note that the full data sets (up till train_end) are used for training
        # For prediction, the corresponding lagged variables as obtained in the last row of the 
        #   individual dataframe will be used to predict the outcome (ie "next day") since a time
        #   series prediction model is written as a function of the previous values (time steps).
        # Since these are lagged, obtaining the last row of it (apart from the outcome) even along the row
        #    at which we are predicting, would still be the lagged values!
        Y_df_train, T_df_train, W_df_train = Y_df_lagged.iloc[:-1,:], T_df_lagged.iloc[:-1,:], W_df_lagged.iloc[:-1,:]
        Y_df_pred, T_df_pred, W_df_pred = Y_df_lagged.iloc[-1:,:], T_df_lagged.iloc[-1:,:], W_df_lagged.iloc[-1:,:]
        # For day_idx = 1008, final_end_idx = 1008 + 2 = 1010, slicing includes the index = 1009 slice, which we 
        #    remove so that we are training up till index 1008 (inclusive), and updating the predicted Y_hat
        #    which corresponds to day with index 1009.
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
            pred = model.predict(W_df_pred)
            Y_base_folds.append(pred)
        Y_base = np.mean(np.array(Y_base_folds), axis = 0) # Average estimators over the folds
        theta = est.const_marginal_ate()
        Y_hat_next_store[day_idx-test_start,:] = Y_base + T_df_pred @ theta.T
        # 0th row -> day_idx = 1008, 1st row -> day_idx = 1009, etc.
        # ... last = 1298 - 1008 = 290th  row -> day_idx = 1298
        # Note that the Y_hat_next_store is a matrix w/ num_days - test_start = 1299 - 1008 = 291 rows
        #   so this is consistent!
        
        # Memory optimization: cleanup model
        del est
        gc.collect()
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
