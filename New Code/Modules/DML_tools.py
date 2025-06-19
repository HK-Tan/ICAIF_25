import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
warnings.filterwarnings("ignore", category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

## Clustering packages etc
try:
    from signet.cluster import Cluster
except ImportError:
    print("Signet package not found. Attempting to install from GitHub...")
    try:
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "git+https://github.com/alan-turing-institute/SigNet.git"]
        )
        # This part of the code should go first since importing parallelized_runs already requires the signet package
        from signet.cluster import Cluster
        print("Signet package installed successfully.")
    except Exception as e:
        print(f"Error installing Signet package: {e}")
        print("Please install it manually: pip install git+https://github.com/alan-turing-institute/SigNet.git")

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
%matplotlib inline

from parallelized_runs import calculate_pnl

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
    
def rolling_window_OR_VAR_w_para_search(Y_df, confound_df,
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

    Y_df: DataFrame of asset returns (outcome variable)
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
    num_days = Y_df.shape[0]  # Total number of days in the dataset
    p_optimal = np.zeros(num_days - test_start)  # Store optimal p for each day in the test set

    if len(Y_df) < lookback_days + 1 or lookback_days <= days_valid:
        raise ValueError("Dataset is too small for the specified lookback_days and days_valid.")

    for day_idx in range(test_start, num_days):
        # First, we perform a parameter search for the optimal p.
        train_start = max(0, day_idx - lookback_days)
        train_end = day_idx - days_valid 
        valid_start = train_end + 1
        valid_end = day_idx - 1

        valid_errors = []
        for p in range(1, p_max + 1):
            current_error = 0
            for valid_move in range(valid_start, valid_end + 1):
                start_idx = train_start + valid_move
                end_idx = start_idx + days_valid

                # Create lagged treatment variables
                # Recall that columns are days, and rows are tickers
                T_df_lagged = Y_df.iloc[:,start_idx:end_idx+1].copy()
                W_df_lagged = make_lags(confound_df.iloc[:,start_idx:end_idx+1], p)
                Y_df_lagged = make_lags(T_df_lagged.iloc[:,start_idx:end_idx+1], p)
                Y_df_lagged, T_df_lagged, W_df_lagged = realign(Y_df_lagged, T_df_lagged, W_df_lagged)
                Y_df_train, T_df_train, W_df_train = Y_df_lagged.iloc[:,:-1], T_df_lagged.iloc[:,:-1], W_df_lagged.iloc[:,:-1]
                Y_df_test , T_df_test, W_df_test = Y_df_lagged.iloc[:,-1], T_df_lagged.iloc[:,-1], W_df_lagged.iloc[:,-1]
                Y_df_test, T_df_test, W_df_test = Y_df_test.reshape(-1,1), T_df_test.reshape(-1,1), W_df_test.reshape(-1,1)
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
                Y_hat_next = Y_base + T_df_test @ theta.T
                
                # Obtain error in the desired metric and accumulate it over the validation window


    result = {
        'test_start': test_start,
        'num_days': num_days,
        'p_optimal': p_optimal,
    }

    return result

def rolling_window_evaluation_lookback_with_lags(Y_df, W_df,
                                                p=3,  # number of lags
                                                window_size=20,
                                                lookback_days=252,
                                                model_y_name='extra_trees',
                                                model_t_name='extra_trees',
                                                model_y_params=None,
                                                model_t_params=None,
                                                cv_folds=5,
                                                train_ratio=0.7,
                                                verbose=True):
    """
    Rolling window evaluation with proper lagged treatment variables
    """
    
    if model_y_params is None:
        model_y_params = {}
    if model_t_params is None:
        model_t_params = {}
    
    # CREATE LAGGED TREATMENT VARIABLES
    T_df = make_lags(Y_df, p)
    
    # Create full dataset and drop NAs
    full = pd.concat([Y_df, T_df, W_df], axis=1).dropna()
    
    # Adjust for the fact that we lose p rows due to lagging
    if train_ratio is None:
        initial_train_size = lookback_days
    else:
        initial_train_size = int(max(train_ratio*full.shape[0], lookback_days))
    
    # Get column names
    Y_cols = Y_df.columns
    T_cols = T_df.columns
    W_cols = W_df.columns
    
    if verbose:
        print(f"Created {len(T_cols)} treatment variables from {p} lags of {len(Y_cols)} assets")
        print(f"Sample treatment columns: {T_cols[:5].tolist()}")
    
    # Storage for predictions and actuals
    all_predictions = []
    all_actuals = []
    prediction_dates = []
    training_times = []
    
    # Start from initial_train_size and go through each day
    test_start = initial_train_size
    test_end = len(full)
    
    if verbose:
        print(f"Starting rolling window evaluation with {p} lags...")
        print(f"Lookback window: {lookback_days} days")
        print(f"Initial training size: {initial_train_size}")
        print(f"Test period: {test_end - test_start} days")
        print(f"RMSE window size: {window_size} days")
    
    import time
    
    for current_day in range(test_start, test_end):
        if verbose and (current_day - test_start) % 10 == 0:
            progress = (current_day - test_start) / (test_end - test_start) * 100
            print(f"Processing day {current_day - test_start + 1}/{test_end - test_start} ({progress:.1f}%)")
        
        start_time = time.time()
        
        # Use only the last lookback_days for training
        train_start_idx = max(0, current_day - lookback_days)
        train_data = full.iloc[train_start_idx:current_day]
        test_data = full.iloc[current_day:current_day+1]  # Just one day
        
        # Extract arrays
        Y_tr = train_data[Y_cols].values
        T_tr = train_data[T_cols].values
        W_tr = train_data[W_cols].values
        
        Y_te = test_data[Y_cols].values[0]  # Get 1D array for single day
        T_te = test_data[T_cols].values[0]
        W_te = test_data[W_cols].values[0]
        
        # Create and fit model
        model_y = get_regressor(model_y_name, force_multioutput=False, **model_y_params)
        model_t = get_regressor(model_t_name, force_multioutput=False, **model_t_params)
        
        tscv = TimeSeriesSplit(cv_folds)
        
        est = LinearDML(
            model_y=model_y,
            model_t=model_t,
            cv=tscv,
            discrete_treatment=False,
            random_state=0
        )
        
        # Fit model
        est.fit(Y_tr, T_tr, X=None, W=W_tr)
        
        # Make prediction for current day
        prediction = corrected_predict_single_day(est, T_te, W_te)
        
        # Store results
        all_predictions.append(prediction)
        all_actuals.append(Y_te)
        prediction_dates.append(full.index[current_day])
        
        # Track timing
        training_times.append(time.time() - start_time)
    
    # Convert to arrays
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
    
    # Calculate rolling window RMSE
    window_rmses = []
    window_dates = []
    per_asset_window_rmses = []
    
    for i in range(len(all_predictions) - window_size + 1):
        window_pred = all_predictions[i:i+window_size]
        window_actual = all_actuals[i:i+window_size]
        
        # Calculate RMSE for this window
        mse = np.mean((window_actual - window_pred) ** 2)
        rmse = np.sqrt(mse)
        
        # Also calculate per-asset RMSE for this window
        mse_per_asset = np.mean((window_actual - window_pred) ** 2, axis=0)
        rmse_per_asset = np.sqrt(mse_per_asset)
        
        window_rmses.append(rmse)
        per_asset_window_rmses.append(rmse_per_asset)
        window_dates.append((prediction_dates[i], prediction_dates[i+window_size-1]))
    
    # Calculate overall metrics
    overall_mse_per_asset = np.mean((all_actuals - all_predictions) ** 2, axis=0)
    overall_rmse_per_asset = np.sqrt(overall_mse_per_asset)
    overall_mse = np.mean((all_actuals - all_predictions) ** 2)
    overall_rmse = np.sqrt(overall_mse)
    
    results = {
        'predictions': all_predictions,
        'actuals': all_actuals,
        'prediction_dates': prediction_dates,
        'window_rmses': np.array(window_rmses),
        'per_asset_window_rmses': np.array(per_asset_window_rmses),
        'window_dates': window_dates,
        'rmse_per_asset': overall_rmse_per_asset,
        'overall_rmse': overall_rmse,
        'asset_names': Y_cols,
        'treatment_names': T_cols,
        'window_size': window_size,
        'lookback_days': lookback_days,
        'n_lags': p,
        'avg_training_time': np.mean(training_times),
        'total_training_time': np.sum(training_times)
    }
    
    if verbose:
        print(f"\nEvaluation complete!")
        print(f"Overall RMSE: {overall_rmse:.6f}")
        print(f"Average window RMSE: {np.mean(window_rmses):.6f}")
        print(f"Min window RMSE: {np.min(window_rmses):.6f}")
        print(f"Max window RMSE: {np.max(window_rmses):.6f}")
        print(f"Average training time per day: {np.mean(training_times):.2f} seconds")
        print(f"Total training time: {np.sum(training_times)/60:.2f} minutes")
    
    return results

