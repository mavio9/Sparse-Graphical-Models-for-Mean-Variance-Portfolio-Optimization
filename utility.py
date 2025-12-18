
import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.preprocessing import StandardScaler
from sklearn.covariance import graphical_lasso, ledoit_wolf


def get_data(tickers, start_date, end_date, min_coverage, fill_nan, save):
    """
    Downloads historical price data via Yahoo Finance, filters assets by 
    data availability, and returns a cleaned DataFrame of daily returns.

    Args:
        tickers (list): List of ticker names.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        min_coverage (float): Minimum ratio of non-null data points required.

    Returns:
        pd.DataFrame: Cleaned daily returns for all qualified assets.
    """
    print("Downloading data from Yahoo Finance...")
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']

    if isinstance(data, pd.Series):
        data = data.to_frame()

    coverage = data.notna().mean(axis=0)
    keep = coverage[coverage >= min_coverage].index
    data = data[keep]

    print(f"Kept {data.shape[1]} / {len(tickers)} tickers with >= {min_coverage:.0%} coverage")

    # Forward fill NaN values with the last available price
    if fill_nan:
        data = data.fillna(method='ffill')
        print("Filled missing data with forward fill.")

    returns = data.pct_change(fill_method=None)
    days_before = len(returns)
    returns = returns.dropna(how="any")
    days_after = len(returns)
    print(f"Dropped {days_before - days_after} days with missing data; {days_after} days remain")
    
    if save:
        returns.to_csv("returns_data.csv")

    print(f"✅ Retrieved {returns.shape[1]} assets over {returns.shape[0]} days")
    return returns



def glasso(alpha, returns_df, save_outputs, max_iter):
    
    X = returns_df.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    emp_cov, _ = ledoit_wolf(X_scaled, assume_centered=True)

    cov_z, prec_z = graphical_lasso(
        emp_cov,
        alpha=alpha,
        tol=1e-4,
        max_iter=max_iter
    )

    # --- Un-standardize back to raw-return units ---
    s = scaler.scale_                 # per-asset std
    D = np.diag(s)
    Dinv = np.diag(1.0 / s)

    cov_raw = D @ cov_z @ D
    prec_raw = Dinv @ prec_z @ Dinv

    if save_outputs:
        pd.DataFrame(cov_raw, index=returns_df.columns, columns=returns_df.columns).to_csv(f"covariance_raw_alpha={alpha}.csv")
        pd.DataFrame(prec_raw, index=returns_df.columns, columns=returns_df.columns).to_csv(f"precision_raw_alpha={alpha}.csv")
        

    return cov_raw, prec_raw
