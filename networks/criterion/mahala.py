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

    for batch in tqdm(data_loader, desc="Extracting latent features"):
        x = batch[0].to(device).float()
        with torch.no_grad():
            z = model.encoder(x)
        latent_list.append(z.cpu().numpy().astype(np.float32))

    return np.vstack(latent_list)

def compute_mahala_stats(X):
    """
    Compute mean and inverse covariance matrix for Mahalanobis distance.
    """
    mean = np.mean(X, axis=0).astype(np.float32)
    cov = np.cov(X, rowvar=False).astype(np.float32) + 1e-6 * np.eye(X.shape[1], dtype=np.float32)
    inv_cov = np.linalg.inv(cov).astype(np.float32)
    return mean, inv_cov

def calc_inv_cov_target_aware_latent(model, target_loader, device):
    """
    Calculate mean and inverse covariance of train_target latent features.
    """
    # Extract latent features
    latent_features = extract_latent_features(model, target_loader, device)
    
    # Compute Mahalanobis statistics
    mean, inv_cov = compute_mahala_stats(latent_features)
    
    return mean, inv_cov

def loss_function_mahala_target_aware_latent(recon_x, x, mean, inv_cov):
    """
    Compute target-aware Mahalanobis distance loss in latent space.
    """
    # Get latent features - no need to reshape since they're already in the right shape
    feat = recon_x  # recon_x is actually the latent features z
    
    # Convert to numpy for computation
    feat_np = feat.cpu().numpy().astype(np.float32)
    
    # Compute Mahalanobis distance
    distances = mahalanobis_distance(feat_np, mean, inv_cov)
    
    # Convert back to torch tensor
    return torch.from_numpy(distances).to(feat.device).float()
