import torch
import numpy as np
from tqdm import tqdm

def cov_v_diff(in_v):
    in_v_tmp = in_v.clone()
    mu = torch.mean(in_v_tmp.t(), 1)
    diff = torch.sub(in_v, mu)

    return diff, mu

def cov_v(diff, num):
    var = torch.matmul(diff.t(), diff) / num
    return var

def mahalanobis(u, v, cov_x, use_precision=False, reduction=True):
    num, dim = v.size()
    if use_precision == True:
        inv_cov = cov_x
    else:
        inv_cov = torch.inverse(cov_x)
    delta = torch.sub(u, v)
    m_loss = torch.matmul(torch.matmul(delta, inv_cov), delta.t())

    if reduction:
        return torch.sum(m_loss)/num
    else:
        return m_loss, num

def loss_function_mahala(recon_x, x, block_size, cov=None, is_source_list=None, is_target_list=None, update_cov=False, use_precision=False, reduction=True):
    ### Modified mahalanobis loss###
    if update_cov == False:
        loss = mahalanobis(recon_x.view(-1, block_size), x.view(-1, block_size), cov, use_precision, reduction=reduction)
        return loss
    else:
        diff = x - recon_x
        cov_diff_source, _ = cov_v_diff(in_v=diff[is_source_list].view(-1, block_size))

        cov_diff_target = None
        is_calc_cov_target = any(is_target_list)
        if is_calc_cov_target:
            cov_diff_target, _ = cov_v_diff(in_v=diff[is_target_list].view(-1, block_size))

        loss = diff**2
        if reduction:
            loss = torch.mean(loss, dim=1)
        
        return loss, cov_diff_source, cov_diff_target

def loss_reduction_mahala(loss):
    return torch.mean(loss)

def calc_inv_cov(model, device="cpu"):
    inv_cov_source=None
    inv_cov_target=None
    
    cov_x_source = model.cov_source.data
    cov_x_source = cov_x_source.to(device).float()
    inv_cov_source = torch.inverse(cov_x_source)
    inv_cov_source = inv_cov_source.to(device).float()
    
    cov_x_target = model.cov_target.data
    cov_x_target = cov_x_target.to(device).float()
    inv_cov_target = torch.inverse(cov_x_target)
    inv_cov_target = inv_cov_target.to(device).float()
    
    return inv_cov_source, inv_cov_target

def calc_inv_cov_target_aware(model, target_loader, device, use_latent=False):
    """
    Calculate mean and inverse covariance of train_target reconstruction errors or latent features.
    Set use_latent=True to operate in latent space.
    """
    model.eval()
    features = []

    for batch in tqdm(target_loader, desc="Extracting features for target-aware Mahala"):
        x = batch[0].to(device).float()  # Ensure float32
        with torch.no_grad():
            if use_latent:
                feat = model.encoder(x)  # latent features
            else:
                x_hat = model(x)
                if isinstance(x_hat, tuple):
                    x_hat = x_hat[0]
                feat = (x - x_hat).view(x.shape[0], -1)  # recon error
        features.append(feat.cpu().numpy().astype(np.float32))  # Ensure float32

    X = np.vstack(features)
    mean = X.mean(axis=0).astype(np.float32)  # Ensure float32
    cov = np.cov(X, rowvar=False).astype(np.float32) + 1e-6 * np.eye(X.shape[1], dtype=np.float32)  # regularization
    inv_cov = np.linalg.inv(cov).astype(np.float32)  # Ensure float32

    return mean, inv_cov

def mahalanobis_distance(X, mean, inv_cov):
    """
    Compute Mahalanobis distance between features X and target mean/cov.
    """
    delta = X - mean
    return np.sum(delta @ inv_cov * delta, axis=1)

def loss_function_mahala_target_aware(recon_x, x, mean, inv_cov, use_latent=False):
    """
    Compute target-aware Mahalanobis distance loss.
    """
    if use_latent:
        feat = recon_x.view(recon_x.shape[0], -1)  # latent features
    else:
        feat = (x - recon_x).view(x.shape[0], -1)  # recon error
    
    # Convert to numpy for computation
    feat_np = feat.cpu().numpy().astype(np.float32)  # Ensure float32
    
    # Compute Mahalanobis distance
    distances = mahalanobis_distance(feat_np, mean, inv_cov)
    
    # Convert back to torch tensor with float32
    return torch.from_numpy(distances).to(feat.device).float()  # Ensure float32

def extract_latent_features(model, data_loader, device):
    """
    Extract latent features from the encoder for all samples in the data loader.
    """
    model.eval()
    latent_list = []
    filenames = []

    for batch in tqdm(data_loader, desc="Extracting latent features"):
        x = batch[0].to(device).float()
        with torch.no_grad():
            z = model.encoder(x)
        latent_list.append(z.cpu().numpy().astype(np.float32))
        filenames.extend(batch[3])  # Store filenames

    return np.vstack(latent_list), filenames

def compute_latent_stats(X, use_ledoit_wolf=False):
    """
    Compute mean and inverse covariance matrix for latent features.
    
    Args:
        X (np.ndarray): Latent features
        use_ledoit_wolf (bool): Whether to use Ledoit-Wolf covariance estimation
        
    Returns:
        tuple: (mean, inv_cov)
    """
    if len(X) == 0:
        return None, None
        
    mean = np.mean(X, axis=0).astype(np.float32)
    if use_ledoit_wolf:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(X).covariance_.astype(np.float32)
    else:
        cov = np.cov(X, rowvar=False).astype(np.float32)
        # Add small diagonal term for numerical stability
        cov = cov + 1e-6 * np.eye(X.shape[1], dtype=np.float32)
    
    try:
        inv_cov = np.linalg.inv(cov).astype(np.float32)
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        inv_cov = np.linalg.pinv(cov).astype(np.float32)
    
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

def calc_inv_cov_target_aware_latent(model, target_loader, device, alpha=0.7, use_ledoit_wolf=False):
    """
    Calculate blended mean and inverse covariance of latent features.
    
    Args:
        model: The autoencoder model
        target_loader: DataLoader for training data
        device: Device to run computations on
        alpha (float): Weight for source statistics (0-1)
        use_ledoit_wolf (bool): Whether to use Ledoit-Wolf covariance estimation
        
    Returns:
        tuple: (mean, inv_cov) for blended statistics
    """
    # Extract latent features from training data
    train_features, train_filenames = extract_latent_features(model, target_loader, device)
    
    # Split training data into source and target
    is_target = np.array(["target" in name for name in train_filenames])
    is_source = np.logical_not(is_target)
    
    # Get source and target features
    X_src = train_features[is_source]
    X_tgt = train_features[is_target]
    
    # Compute statistics for source and target
    mu_src, inv_cov_src = compute_latent_stats(X_src, use_ledoit_wolf)
    mu_tgt, inv_cov_tgt = compute_latent_stats(X_tgt, use_ledoit_wolf)
    
    # Blend statistics
    mu_blend, inv_cov_blend = blend_stats(mu_src, inv_cov_src, mu_tgt, inv_cov_tgt, alpha)
    
    return mu_blend, inv_cov_blend

def loss_function_mahala_target_aware_latent(recon_x, x, mean, inv_cov):
    """
    Compute blended Mahalanobis distance loss in latent space.
    
    Args:
        recon_x: Latent features from encoder
        x: Input features (not used, kept for compatibility)
        mean: Blended mean vector
        inv_cov: Blended inverse covariance matrix
        
    Returns:
        torch.Tensor: Mahalanobis distances
    """
    # Get latent features
    feat = recon_x  # recon_x is actually the latent features z
    
    # Convert to numpy for computation
    feat_np = feat.detach().cpu().numpy().astype(np.float32)
    
    # Compute Mahalanobis distance
    diff = feat_np - mean
    distances = np.sqrt(np.sum(diff.dot(inv_cov) * diff, axis=1))
    
    # Convert back to torch tensor
    return torch.from_numpy(distances).to(feat.device).float()
