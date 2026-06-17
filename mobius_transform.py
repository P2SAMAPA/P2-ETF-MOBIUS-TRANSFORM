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

def mobius_sensitivity(returns, macro_factor, epsilon=0.01):
    """
    Compute sensitivity of the transformed spectral norm to macro factor.
    """
    # Compute correlation matrix
    corr = returns.corr().values
    eigvals, _ = np.linalg.eigh(corr)
    eigvals = np.maximum(eigvals, 0)  # ensure non-negative
    # Baseline Möbius transform
    transformed = mobius_transform(eigvals)
    spectral_norm_base = np.max(transformed)
    # Perturb macro factor
    macro_plus = min(1.0, macro_factor + epsilon)
    macro_minus = max(0.0, macro_factor - epsilon)
    # For perturbation, we need to adjust correlation matrix? 
    # We'll use a simplified approach: the Möbius parameter a,b,c,d are functions of macro.
    # We'll vary the transformation parameter directly.
    # Simpler: we adjust the "c" parameter as a function of macro.
    # c = 1 + macro_factor (so higher macro -> more aggressive transform)
    def transformed_norm(c_param):
        eig_trans = mobius_transform(eigvals, a=1.0, b=0.0, c=c_param, d=1.0)
        return np.max(eig_trans)
    # Baseline
    c_base = 1.0 + macro_factor
    spectral_base = transformed_norm(c_base)
    # Perturbed macro
    c_plus = 1.0 + macro_plus
    c_minus = 1.0 + macro_minus
    spectral_plus = transformed_norm(c_plus)
    spectral_minus = transformed_norm(c_minus)
    # Derivative
    derivative = (spectral_plus - spectral_minus) / (macro_plus - macro_minus)
    return derivative
