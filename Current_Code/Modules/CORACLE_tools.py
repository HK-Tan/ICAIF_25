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

def make_original_with_forecast_row(df):
    """Create lagged DataFrame with an additional forecasting row"""
    # Add forecast row with NaN values
    forecast_row = pd.DataFrame({col: [np.nan] for col in df.columns})
    return pd.concat([df, forecast_row], ignore_index=True) 

def make_lags(df, p):
    """Create lagged DataFrame with p lags"""
    # Does not contain the original columns, only the lagged ones
    lagged_df = pd.concat([df.shift(i) for i in range(p + 1)], axis=1)
    lagged_df.columns = [f"{col}_lag{i}" for i in range(p + 1) for col in df.columns]
    return lagged_df

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

def predict_ahead_with_lag(asset_df, confound_df, p,
                           model_y_name='extra_trees', model_t_name='extra_trees',
                          model_y_params=None, model_t_params=None,
                          cv_folds=5):
    """
    Predict (just) the next day's returns using a lagged VAR model with orthogonalized regression.
    We will basically do OR and train VAR entirely on df and confound_df to predict the next day
    for a given lag value p and cluster assignment (asset_df)

    This will return a 1 x n array, where n is the number of assets in df representing the 
    predicted returns for the next day for each of the assets in df.
    """
    if model_y_params is None:
        model_y_params = {}
    if model_t_params is None:
        model_t_params = {}
    
    Y_df_lagged = make_original_with_forecast_row(asset_df)  # Include current day for prediction
    T_df_lagged = make_lags(Y_df_lagged, p)
    W_df_lagged = make_lags(make_original_with_forecast_row(confound_df), p)
    
    Y_df_train, T_df_train, W_df_train = Y_df_lagged.iloc[p:-1,:], T_df_lagged.iloc[p:-1,:], W_df_lagged.iloc[p:-1,:]
    Y_df_pred, T_df_pred, W_df_pred = Y_df_lagged.iloc[-1:,:], T_df_lagged.iloc[-1:,:], W_df_lagged.iloc[-1:,:]

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

    # Memory optimization
    del est, Y_base_folds
    del Y_df_lagged, T_df_lagged, W_df_lagged
    del Y_df_train, T_df_train, W_df_train
    del Y_df_pred, T_df_pred, W_df_pred
    gc.collect()

    return Y_hat_next



def memoized_evaluate_csv(clustered_df, confound_df, p_max=5,
                          model_y_name='extra_trees', model_t_name='extra_trees',
                          model_y_params=None, model_t_params=None,
                          cv_folds=5, lookback_days=252, days_valid=5):
    """
    Note that clustered_df should have a total of lookback_days + days_valid rows + 1
        (the +1 is to account for the fact that we are predicting the next day).
    Note that the number of data points used to train VAR here would be lookback_days - days_valid
        as part of the memoization technique here.
    
    With such a structure, we always assume that index lookback_days - 1 represents the current day,
        and we would like to roll forward with the same "clustered_df" (recall that each of these df
        corresponds to a particular cluster assignment obtained by using data from the lookback window
        up till today). In other words, the assignment of clusters is frozen and we proceed to use the
        same assignment to roll forward. 

    The output dataframe will have:

        > A total of k(p_max + 1) columns:
            First k columns are predictions using p = 1
            Second k columns are predictions using p = 2
            ...
            Second last k columns are predictions using p = p_max
            Last k column are the actual returns for the next day (i.e. the "true" values).

        > 2*days_valid rows (representing (days_valid - 1) days before the current day,
            + current day + (days_valid - 1) ahead + additional day in the future of that to predict).
    """
    if model_y_params is None:
        model_y_params = {}
    if model_t_params is None:
        model_t_params = {}
    
    train_days = lookback_days - days_valid # Number of days used for training
    # If lookback_days = 252, days_valid = 5, then train_days = 252 - 5 = 247, then the first
    #  247 rows will be used for training (thus 0:247, excluding index 247).
    num_clusters = clustered_df.shape[1]  # Number of clusters (or assets)
    Y_hat_next_store = np.zeros((2 * days_valid, num_clusters * (p_max + 1)))

    # Roll forward and predict!
    for i in range(2 * days_valid): # If days_valid = 5, we take index 0, 1, ..., 9
        # Get the training set
        Y_train_df = clustered_df.iloc[i:i + train_days,:]
        W_train_df = confound_df.iloc[i:i + train_days,:]
        for p in range(1, p_max+1):
            Y_hat_next_store[i, (p - 1) * num_clusters:p * num_clusters] = predict_ahead_with_lag(Y_train_df, W_train_df, p,
                        model_y_name=model_y_name, model_t_name=model_t_name,
                        model_y_params=model_y_params, model_t_params=model_t_params,
                        cv_folds=cv_folds)
        Y_hat_next_store[i, -num_clusters:] = clustered_df.iloc[i + train_days,:].values

    return Y_hat_next_store

"""
Missing: Saving it as a csv? Please help to implement this.
"""

def predict_tomorrow(memoized_clustered_dir, target_dir,
                     todays_date,
                     p_max=5,  # maximum number of lags
                     model_y_name='extra_trees',
                     model_t_name='extra_trees',
                     model_y_params=None,
                     model_t_params=None,
                     cv_folds=5,
                     lookback_days=252,  # 1 year of daily data
                     days_valid=5):
    """
    The purpose of this function is to perform a hyperparameter tuning-like search
    to determine the optimal value of p and cluster assignments to predict for next day's returns
    using CORACLE-VAR using the memoized csv files. 

    This is best illustrated with a concrete example:

    Let's say we have a lookback window of 252 days (i.e. 1 year of daily data), with 5 days of validation. Suppose that we
    start off on some todays_date, ie '2001-01-05', which corresponds to the 257th day (index 256), 
    with the 252nd to 256th days (inclusive) being the validation set.
    (Note that there is a "one-off" issue as compared to the implementation for ORACLE-VAR, depending on what 
    is defined as the length of the lookback window.)

    We also assume that we have a dataset with daily data from 2000-12-29 to 2021-12-01, which gives us a total of ~ 5,000 days of data.
    For the csv file titled '2000-12-29.csv', the assumption is that this corresponds to a cluster assignment determined on the
    lookback window of 252 days ending on 2000-12-29, and the cluster assignment is frozen for the next 20 (trading) days and 
    one extra testing day to end on 2001-01-31 (inclusive). This also implies that we can only choose: days_valid <= 20.
    These should be available in the memoized_clustered_dir directory.

    With "+5 days of validation", the relevant csv files would then be
    - '2000-12-29.csv' (for the 252 days ending on 2000-12-29),
    - '2001-01-02.csv' (ending on the 253rd day, i.e. 2001-01-02),
    - '2001-01-03.csv' (ending on the 254th day, i.e. 2001-01-03),
    - '2001-01-04.csv' (ending on the 255th day, i.e. 2001-01-04),
    - '2001-01-05.csv' (ending on the 256th day, i.e. 2001-01-05, 
                        ie the supposed "today", which is also the last day of the validation set)

    Now, let's say that we are looking at the folder reprsenting a fixed k-cluster assignment.
    The hyperparameter search stage will then be done by looking over all possible values of p and their predictions,
    plus all possible cluster assignments (i.e. all csv files in the folder - recall that each csv file is an assignment of clusters!)
    The optimal cluster assignment and lag size p would be the one that minimizes the validation error ('rmse') 
    over their corresponding validation set. How would this be done? 

    Recall that each csv file is being outputted by the memoized_evaluate_csv function, which for each row, consists of
    the predictions for the next day using p = 1, 2, ..., p_max, and the actual returns for the next day. Hence, it
    suffices to find the index of the row corresponding to todays_date, and then the look backwards for the validation set,
    and for each lag value (corresponding to k columns in their repsective column indices), calculate the rmse betwen the
    prediction "row vector" and the actural returns "row vector" (the last k columns in the row), and then average them up
    over the validation set (5 rows).

    To optimize over the number of clusters, we will assume that the csv files are stored in memoized_clustered_dir as follows:
    - memoized_clustered_dir/
        - k5/
            - 2000-12-29.csv
            - 2001-01-02.csv
            - 2001-01-03.csv
            - 2001-01-04.csv
            - 2001-01-05.csv
        - k8/
            - 2000-12-29.csv
            - 2001-01-02.csv
            - 2001-01-03.csv
            - 2001-01-04.csv
            - 2001-01-05.csv
        - k12/
            - 2000-12-29.csv
            - 2001-01-02.csv
            - 2001-01-03.csv
            - 2001-01-04.csv
            - 2001-01-05.csv

    where k5, k8, k12 are the number of clusters (k) used to generate the cluster assignments.

    We now obtain the optimal p and lag assignment for k clusters. Assuming that we have a total of K combinations of
    the number of clusters, say K = 3 for k in {5,8,12}. Then, the total number of combinations to optimize over is given by
    K * p_max * days_valid. 

    The output of this function will be a DataFrame with the following structure:
    - The first column is the date (i.e. todays_date).
    - The second column is the optimal p value.
    - The third column is the optimal number of clusters (ie 5 for k = 5)
    - The fourth column is the assignment in terms of the name of the csv file corresponding to the cluster assignment.
        i.e. '2001-01-04.csv' for the 256th day (i.e. the day before the current day)
    - The next k columns are the predicted returns for the next day using the optimal p and cluster assignment.
    - The next (last) k columns are the actual returns for the next day.
    
    
    Inputs:
    memoized_clustered_dir: Directory containing the memoized clustered CSV files.
    target_dir: Directory to save the results.
    todays_date: The date for which to make the prediction (e.g. '2001-01-05').
    p_max: Maximum number of lags to consider (e.g. 5).
    model_y_name: Name of the regressor to use for the outcome variable (e.g. 'extra_trees').
    model_t_name: Name of the regressor to use for the treatment variable (e.g. 'extra_trees').
    model_y_params: Dictionary of parameters for the outcome regressor.
    model_t_params: Dictionary of parameters for the treatment regressor.
    cv_folds: Number of cross-validation folds to use.
    lookback_days: Number of days to use for the lookback window (e.g. 252).
    days_valid: Number of days to use for validation (e.g. 5).
    
    Outputs:
    A DataFrame with the optimal p, cluster assignment, predicted returns, and actual returns for the next day.
    Refer to the details above.
    """

    pass # TODO: Implement this function

def rolling_window_CORACLE_w_para_search(memoized_clustered_dir, full_confound_df, target_dir,
                                        p_max=5,  # maximum number of lags
                                        model_y_name='extra_trees',
                                        model_t_name='extra_trees',
                                        model_y_params=None,
                                        model_t_params=None,
                                        cv_folds=5,
                                        lookback_days=252,  # 1 year of daily data
                                        days_valid=5,  
                                        error_metric='rmse'):
    """
    The purpose of this function is to run a rolling window evaluation
    using CORACLE-VAR. The idea is that we will always run this via orthongalized regression
    framework under DML. However, to determine the optimal value of p and cluster assignments, we will run a
    "hyperparameter"-like search over the lookback window (to prevent look-ahead bias).

    This is best illustrated with a concrete example:

    Let's say we have a lookback window of 252 days (i.e. 1 year of daily data), with 5 days of validation. Hence, we will
    start off on the 257th day (index 256), with the 252nd to 256th days (inclusive) being the validation set.
    (Note that there is a "one-off" issue as compared to the implementation for ORACLE-VAR, depending on what 
    is defined as the length of the lookback window.)

    We also assume that we have a dataset with daily data from 2000-12-29 to 2021-12-01, which gives us a total of ~ 5,000 days of data.
    For the csv file titled '2000-12-29.csv', the assumption is that this corresponds to a cluster assignment determined on the
    lookback window of 252 days ending on 2000-12-29, and the cluster assignment is frozen for the next 20 (trading) days and 
    one extra testing day to end on 2001-01-31 (inclusive). This also implies that we can only choose: days_valid <= 20.
    These should be available in the memoized_clustered_dir directory.

    With "+5 days of validation", the relevant csv files would then be
    - '2000-12-29.csv' (for the 252 days ending on 2000-12-29),
    - '2001-01-02.csv' (ending on the 253rd day, i.e. 2001-01-02),
    - '2001-01-03.csv' (ending on the 254th day, i.e. 2001-01-03),
    - '2001-01-04.csv' (ending on the 255th day, i.e. 2001-01-04),
    - '2001-01-05.csv' (ending on the 256th day, i.e. 2001-01-05, 
                        ie the supposed "today", which is also the last day of the validation set)

    Now, let's say that we are looking at the folder reprsenting a fixed cluster assignment size k.
    The hyperparameter search stage will then be done by looking over all possible values of p and their predictions,
    plus all possible cluster assignments (i.e. all csv files in the folder - recall that each csv file is an assignment of clusters!)
    The optimal cluster assignment and lag size p would be the one that minimizes the validation error averaged over the validation set.




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

