import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import itertools

def make_lags(df, p):
    """
    Create lagged copies of a DataFrame (without the original columns; starting from lag 1).

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    p (int): The number of lags to create.

    Returns:
    pd.DataFrame: A DataFrame with lagged columns.
    """
    if not isinstance(p, int):
        raise ValueError(f"Value of p for computing lags must be an integer, actual value is p={p}")
    return pd.concat([df.shift(k).add_suffix(f'_lag{k}') for k in range(1, p+1)], axis=1)

def realign(*dfs):
    """
    Realign multiple DataFrames by dropping rows with NaN values.
    
    Parameters:
    *dfs: Variable number of DataFrames to realign
    
    Returns:
    tuple: Tuple of realigned DataFrames in the same order as input
    """
    full = pd.concat(dfs, axis=1).dropna()
    result = []
    col_start = 0
    for df in dfs:
        ## Iteratively extract each df in dfs from the full aligned DataFrame
        ## Indexing is done based on the original order of columns
        col_end = col_start + len(df.columns)
        result.append(full.iloc[:, col_start:col_end])
        col_start = col_end
    return tuple(result)

def rolling_window_VAR(asset_df, confound_df=None,
                      p_max=5,
                      model_type='ols',
                      alpha_grid=None,
                      lookback_days=252*4,
                      days_valid=20,
                      error_metric='rmse'):
    """
    Perform rolling window VAR/VARX estimation with hyperparameter tuning.
    
    This function implements either:
    1. Vector Autoregression (VAR) model when confound_df=None
    2. Vector Autoregression with exogenous variables (VARX) when confound_df is provided
    
    Model equations:
    VAR:  Y_t = c + A_1*Y_{t-1} + ... + A_p*Y_{t-p} + error_t
    VARX: Y_t = c + A_1*Y_{t-1} + ... + A_p*Y_{t-p} + B_1*W_{t-1} + ... + B_p*W_{t-p} + error_t
    
    Parameters:
    -----------
    asset_df : pd.DataFrame
        The primary time series data (asset returns), treated as endogenous variables (Y).
    confound_df : pd.DataFrame, optional (default=None)
        The exogenous confounding variables (macro indicators), treated as control variables (W).
        If None: Runs standard VAR model using only asset_df
        If provided: Runs VARX model including confounding variables
    p_max : int, default=5
        Maximum number of lags to consider.
    model_type : str, default='ols'
        Type of model to fit. Options: 'ols', 'lasso'.
    alpha_grid : list, optional
        Grid of regularization parameters for LASSO. If None, uses default grid.
    lookback_days : int, default=252*4
        Number of days to use for the lookback window (4 years of daily data).
    days_valid : int, default=20
        Number of days to use for validation within each lookback window.
    error_metric : str, default='rmse'
        Metric to use for validation. Currently supports 'rmse'.
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'test_start': Starting index of the test set
        - 'num_days': Total number of days in the dataset
        - 'p_optimal': Array of optimal lag orders for each day
        - 'alpha_optimal': Array of optimal alpha values for each day (LASSO only)
        - 'Y_hat_next_store': Predicted values for each day
        - 'validation_errors': Validation errors for each day's hyperparameter search
    """
    
    # Set default alpha grid for LASSO
    if alpha_grid is None:
        alpha_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    # Validate inputs
    if len(asset_df) < lookback_days + 1 or lookback_days <= days_valid:
        raise ValueError("Dataset is too small for the specified lookback_days and days_valid.")
    
    if model_type not in ['ols', 'lasso']:
        raise ValueError("model_type must be either 'ols' or 'lasso'")
    
    if error_metric != 'rmse':
        raise ValueError("Currently only 'rmse' error metric is supported")
    
    # Initialize variables
    test_start = lookback_days
    num_days = asset_df.shape[0] - 1  # Total days minus one (can't train on last day)
    p_optimal = np.zeros(num_days - test_start)
    alpha_optimal = np.zeros(num_days - test_start) if model_type == 'lasso' else None
    Y_hat_next_store = np.zeros((num_days - test_start, asset_df.shape[1]))
    
    model_variant = "VAR" if confound_df is None else "VARX"
    print(f"Starting {model_variant} analysis with {model_type.upper()} model")
    print(f"Dataset: {len(asset_df)} days, Test period: {num_days - test_start} days")
    
    # Rolling window loop
    for day_idx in range(test_start, num_days):
        print(f"Processing day {day_idx} out of {num_days} days in the dataset.")
        
        # Define training and validation windows
        train_start = max(0, day_idx - lookback_days)
        train_end = day_idx - days_valid
        valid_start = train_end + 1
        valid_end = valid_start + days_valid - 1
        
        # Hyperparameter search
        best_error = float('inf')
        best_p = 1
        best_alpha = alpha_grid[0] if model_type == 'lasso' else None
        
        # Grid search over hyperparameters
        param_combinations = []
        if model_type == 'ols':
            param_combinations = [(p, None) for p in range(1, p_max + 1)]
        else:  # lasso
            param_combinations = [(p, alpha) for p in range(1, p_max + 1) for alpha in alpha_grid]
        
        for p, alpha in param_combinations:
            current_error = 0
            valid_count = 0
            
            # Validate over rolling windows within validation period
            for valid_shift in range(days_valid):
                start_idx = train_start + valid_shift
                end_idx = train_end + valid_shift + 2  # +2 for prediction setup
                
                # Prepare data for this validation fold
                Y_data = asset_df.iloc[start_idx:end_idx].copy()
                Y_lagged = make_lags(Y_data, p)
                
                # Conditional feature preparation based on model type
                if confound_df is not None:
                    # VARX model: include confounding variables
                    W_lagged = make_lags(confound_df.iloc[start_idx:end_idx], p)
                    X_features = pd.concat([Y_lagged, W_lagged], axis=1)
                else:
                    # Standard VAR model: only lagged Y variables
                    X_features = Y_lagged
                
                # Realign data (remove NaN rows)
                Y_aligned, X_aligned = realign(Y_data, X_features)
                
                # Split into train/validation for this fold
                Y_train = Y_aligned.iloc[:-1,:]
                X_train = X_aligned.iloc[:-1,:]
                Y_val = Y_aligned.iloc[-1:,:].values
                X_val = X_aligned.iloc[-1:,:].values
                
                # Fit model
                if model_type == 'ols':
                    model = LinearRegression(fit_intercept=True)
                else:  # lasso
                    model = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
                
                model.fit(X_train.values, Y_train.values)
                
                # Predict and calculate error
                Y_pred = model.predict(X_val)
                fold_error = np.sqrt(mean_squared_error(Y_val, Y_pred))
                current_error += fold_error
                valid_count += 1
                    
            
            if valid_count > 0:
                avg_error = current_error / valid_count
                
                # Track best parameters as it is being appended
                if avg_error < best_error:
                    best_error = avg_error
                    best_p = p
                    best_alpha = alpha
        
        # Store optimal parameters
        p_optimal[day_idx - test_start] = best_p
        if model_type == 'lasso':
            alpha_optimal[day_idx - test_start] = best_alpha
        
        print(f"Optimal parameters: p={best_p}" + 
              (f", alpha={best_alpha}" if model_type == 'lasso' else "") + 
              f", validation RMSE={best_error:.4f}")
        
        # Fit final model on full lookback window
        final_start_idx = max(0, day_idx - lookback_days)
        final_end_idx = day_idx + 2  # + 2 for prediction setup
        
        # Prepare final training data
        Y_data = asset_df.iloc[final_start_idx:final_end_idx].copy()
        Y_lagged = make_lags(Y_data, best_p)
        
        # Conditional feature preparation for final model
        if confound_df is not None:
            # VARX model: include confounding variables
            W_lagged = make_lags(confound_df.iloc[final_start_idx:final_end_idx], best_p)
            X_features = pd.concat([Y_lagged, W_lagged], axis=1)
        else:
            # Standard VAR model: only lagged Y variables
            X_features = Y_lagged
        
        # Realign data
        Y_aligned, X_aligned = realign(Y_data, X_features)
        
        # Split into train/predict
        Y_train = Y_aligned.iloc[:-1,:]
        X_train = X_aligned.iloc[:-1,:]
        X_pred = X_aligned.iloc[-1:,:].values
        
        # Fit final model
        if model_type == 'ols':
            final_model = LinearRegression(fit_intercept=True)
        else:  # lasso
            final_model = Lasso(alpha=best_alpha, fit_intercept=True, max_iter=1000)
        
        final_model.fit(X_train.values, Y_train.values)
        
        # Make prediction
        Y_hat_next = final_model.predict(X_pred)
        Y_hat_next_store[day_idx - test_start] = Y_hat_next.flatten()
        
    
    # Prepare results
    result = {
        'test_start': test_start,
        'num_days': num_days,
        'p_optimal': p_optimal,
        'Y_hat_next_store': Y_hat_next_store,
    }
    
    if model_type == 'lasso':
        result['alpha_optimal'] = alpha_optimal
    
    print(f"{model_variant} analysis completed. Processed {len(p_optimal)} days.")
    return result
