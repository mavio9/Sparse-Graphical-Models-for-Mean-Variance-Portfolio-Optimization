import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import ledoit_wolf, graphical_lasso
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_matrices(cov_file, prec_file):
    # Load the data
    cov_df = pd.read_csv(cov_file, index_col=0)
    prec_df = pd.read_csv(prec_file, index_col=0)

    # Plot Covariance Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cov_df, cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
    plt.title('Covariance Matrix')
    plt.savefig('covariance_heatmap.png')
    plt.show()

    # Plot Precision Matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(prec_df, cmap='coolwarm', center=0, xticklabels=False, yticklabels=False)
    plt.title('Precision Matrix (Inverse Covariance)')
    plt.savefig('precision_heatmap.png')
    plt.show()

# Example usage with your filenames
plot_matrices(
    'covariance_using_return_alpha0.4.csv', 
    'precision_using_return_alpha=0.4.csv'
)
def glasso(alpha, dataset):
    # --- LOAD DATA ---
    X = pd.read_csv(dataset).dropna()
    assets = X['Ticker'].unique()
    print("Using dataset with", len(assets), "assets")

    # --- PREPROCESS ---
    X['Date'] = pd.to_datetime(X['Date'])

    # Pivot to form a matrix of Close prices (Date x Ticker)
    price_df = X.pivot(index='Date', columns='Ticker', values='Close')

    # Calculate returns and drop days with missing data
    returns = price_df.pct_change().dropna(how="any") 
    
    print(f"✅ Using {returns.shape[1]} assets over {returns.shape[0]} days")

    # --- STANDARDIZE ---
    # It is important to standardize because Graphical Lasso assumes data is centered
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(returns)
    print("X_scaled shape:", X_scaled.shape)

    # --- COMPUTE INPUT COVARIANCE ---
    # You are using Ledoit-Wolf shrinkage. This is a valid robust estimator, 
    # but note that Graphical Lasso also applies regularization. 
    # Standard GLasso usually takes the empirical covariance (np.cov) as input.
    emp_cov, _ = ledoit_wolf(X_scaled, assume_centered=True)

    # --- GRAPHICAL LASSO ---
    cov_matrix, prec_matrix = graphical_lasso(emp_cov, alpha=alpha, tol=1e-4)
    print("Covariance matrix shape:", cov_matrix.shape)

    # --- SAVE OUTPUTS ---
    # FIX: Use returns.columns to get the actual ticker names for the index/columns
    tickers = returns.columns
    
    cov_df = pd.DataFrame(cov_matrix, index=tickers, columns=tickers)
    prec_df = pd.DataFrame(prec_matrix, index=tickers, columns=tickers)
    
    cov_filename = f"covariance_using_return_alpha{alpha}.csv"
    prec_filename = f"precision_using_return_alpha={alpha}.csv"

    cov_df.to_csv(cov_filename)
    prec_df.to_csv(prec_filename)
    plot_matrices(cov_filename, prec_filename)
    print(f"✅ Saved covariance to: {cov_filename}")
    print(f"✅ Saved precision to: {prec_filename}")
    
    return cov_matrix, prec_matrix 

alpha = 0
dataset = "cleaned_assets_price_volume_data_6_months_no_std.csv"
cov, prec = glasso(alpha, dataset)