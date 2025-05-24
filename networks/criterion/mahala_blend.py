import numpy as np
from sklearn.covariance import LedoitWolf

def compute_latent_stats(X, use_ledoit_wolf=False):
    """
    Compute mean and covariance matrix for latent features.
    
    Args:
        X (np.ndarray): Latent features
        use_ledoit_wolf (bool): Whether to use Ledoit-Wolf covariance estimation
        
    Returns:
        tuple: (mean, inv_cov)
    """
    if len(X) == 0:
        return None, None
        
    mean = np.mean(X, axis=0)
    if use_ledoit_wolf:
        cov = LedoitWolf().fit(X).covariance_
    else:
        cov = np.cov(X, rowvar=False)
        # Add small diagonal term for numerical stability
        cov = cov + 1e-6 * np.eye(X.shape[1])
    
    try:
        inv_cov = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        inv_cov = np.linalg.pinv(cov)
    
    return mean, inv_cov

def blend_stats(mu_src, cov_src, mu_tgt, cov_tgt, alpha=0.7):
    """
    Blend source and target statistics.
    
    Args:
        mu_src (np.ndarray): Source mean
        cov_src (np.ndarray): Source inverse covariance
        mu_tgt (np.ndarray): Target mean
        cov_tgt (np.ndarray): Target inverse covariance
        alpha (float): Weight for source statistics (0-1)
        
    Returns:
        tuple: (blended_mean, blended_inv_cov)
    """
    # Handle cases where either source or target stats are None
    if mu_src is None or cov_src is None:
        return mu_tgt, cov_tgt
    if mu_tgt is None or cov_tgt is None:
        return mu_src, cov_src
        
    # Blend means
    mu_blend = alpha * mu_src + (1 - alpha) * mu_tgt
    
    # Blend inverse covariances
    cov_blend = alpha * cov_src + (1 - alpha) * cov_tgt
    
    return mu_blend, cov_blend

def mahalanobis_distance(X, mean, inv_cov):
    """
    Compute Mahalanobis distance between features and a distribution.
    
    Args:
        X (np.ndarray): Features
        mean (np.ndarray): Distribution mean
        inv_cov (np.ndarray): Inverse covariance matrix
        
    Returns:
        np.ndarray: Mahalanobis distances
    """
    if mean is None or inv_cov is None:
        return np.zeros(len(X))
        
    diff = X - mean
    return np.sqrt(np.sum(diff.dot(inv_cov) * diff, axis=1))

def compute_blended_mahalanobis(X_src, X_tgt, X_test, alpha=0.7, use_ledoit_wolf=False):
    """
    Compute blended Mahalanobis distance for test samples.
    
    Args:
        X_src (np.ndarray): Source domain features
        X_tgt (np.ndarray): Target domain features
        X_test (np.ndarray): Test features
        alpha (float): Weight for source statistics (0-1)
        use_ledoit_wolf (bool): Whether to use Ledoit-Wolf covariance estimation
        
    Returns:
        np.ndarray: Mahalanobis distances
    """
    # Compute statistics for source and target
    mu_src, inv_cov_src = compute_latent_stats(X_src, use_ledoit_wolf)
    mu_tgt, inv_cov_tgt = compute_latent_stats(X_tgt, use_ledoit_wolf)
    
    # Blend statistics
    mu_blend, inv_cov_blend = blend_stats(mu_src, inv_cov_src, mu_tgt, inv_cov_tgt, alpha)
    
    # Compute distances
    return mahalanobis_distance(X_test, mu_blend, inv_cov_blend) 