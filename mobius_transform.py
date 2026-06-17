import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def compute_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def mobius_transform(eigvals, a=1.0, b=0.0, c=1.0, d=1.0):
    """
    Apply Möbius transformation to eigenvalues: λ' = (a*λ + b) / (c*λ + d)
    Default: λ' = λ / (λ + 1) (maps [0,∞) to [0,1))
    """
    return (a * eigvals + b) / (c * eigvals + d + 1e-8)

def mobius_sensitivity(returns_df, macro_factor, epsilon=0.01):
    """
    Compute sensitivity of the transformed spectral norm to macro factor.
    returns_df: DataFrame of returns for all ETFs in the universe (T x n)
    """
    if returns_df.shape[1] < 2:
        return {ticker: 0.0 for ticker in returns_df.columns}
    # Compute correlation matrix
    corr = returns_df.corr().values
    eigvals, _ = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0)  # ensure non-negative
    # Define function to compute transformed spectral norm given c parameter
    def transformed_norm(c_param):
        eig_trans = mobius_transform(eigvals, a=1.0, b=0.0, c=c_param, d=1.0)
        return np.max(eig_trans)
    # Baseline
    c_base = 1.0 + macro_factor
    spectral_base = transformed_norm(c_base)
    # Perturbed macro
    macro_plus = min(1.0, macro_factor + epsilon)
    macro_minus = max(0.0, macro_factor - epsilon)
    c_plus = 1.0 + macro_plus
    c_minus = 1.0 + macro_minus
    spectral_plus = transformed_norm(c_plus)
    spectral_minus = transformed_norm(c_minus)
    # Derivative
    derivative = (spectral_plus - spectral_minus) / (macro_plus - macro_minus)
    # Per-ETF score: we assign the derivative to all ETFs (since it's a global metric)
    # To get per-ETF variation, we can compute each ETF's contribution to the spectral norm
    # using the eigenvector components.
    # Compute per-ETF contribution to the largest eigenvalue
    eigvals, eigvecs = np.linalg.eigh(corr)
    max_idx = np.argmax(eigvals)
    max_eigvec = eigvecs[:, max_idx]
    # Per-ETF contribution to the spectral norm = squared eigenvector component
    per_etf_contrib = max_eigvec ** 2
    # Scale by derivative to get per-ETF sensitivity
    sensitivity = derivative * per_etf_contrib
    tickers = returns_df.columns
    return {tickers[i]: float(sensitivity[i]) for i in range(len(tickers))}
