import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy.stats import wasserstein_distance

from models import ProCFair, irm_penalty

def compute_mmd(x, y, sigma=1.0):
    x_i = x.unsqueeze(1)
    y_j = y.unsqueeze(0)
    dist = torch.pow(x_i - y_j, 2)
    return (torch.exp(-dist / (2 * sigma ** 2))).mean()

def compute_mmd_loss(x, y, sigma=1.0):
    xx = compute_mmd(x, x, sigma)
    yy = compute_mmd(y, y, sigma)
    xy = compute_mmd(x, y, sigma)
    return (xx + yy - 2 * xy).item()

def train_procfair(X_miss, S_miss, Y, input_dim, hidden_dim=50, latent_dim=10, 
                   epochs=50, batch_size=256, lr=0.001, lambda_irm=1.0,
                   progress_bar=None, status_text=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Masks
    mask_x = ~np.isnan(X_miss)
    mask_s = ~np.isnan(S_miss)
    
    # Impute X for initial input to encoder
    X_in = np.copy(X_miss)
    for j in range(X_in.shape[1]):
        col_mean = np.nanmean(X_in[:, j])
        X_in[np.isnan(X_in[:, j]), j] = col_mean

    # Impute S temporarily with 0 for tensor creation
    S_in = np.copy(S_miss)
    S_in[np.isnan(S_in)] = 0.0

    X_in_t = torch.tensor(X_in, dtype=torch.float32)
    mask_x_t = torch.tensor(mask_x, dtype=torch.float32)
    S_t = torch.tensor(S_in, dtype=torch.float32).unsqueeze(1)
    mask_s_t = torch.tensor(mask_s, dtype=torch.float32).unsqueeze(1)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_in_t, mask_x_t, S_t, mask_s_t, Y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = ProCFair(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'total_loss': [], 'recon_loss': [], 'kld_loss': [], 
        'elbo': [], 'irm_penalty': [], 'proxy_s_loss': []
    }
    
    def compute_loss_for_s(s_assumed, x_b, m_b, y_b):
        x_hat, mu, logvar, z = model.forward_ciwae(x_b, m_b, s_assumed)
        recon_loss_batch = F.mse_loss(x_hat * m_b, x_b * m_b, reduction='none').sum(dim=1, keepdim=True) / m_b.sum(dim=1, keepdim=True).clamp(min=1)
        kld_loss_batch = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1, keepdim=True)
        
        x_imputed = x_b * m_b + x_hat * (1 - m_b)
        s_cf = 1 - s_assumed
        x_cf_hat = model.decoder(z, s_cf)
        x_cf = x_b * m_b + x_cf_hat * (1 - m_b)
        
        logits_f = model.predictor(x_imputed)
        loss_pred_f_batch = F.binary_cross_entropy_with_logits(logits_f, y_b, reduction='none')
        
        logits_cf = model.predictor(x_cf)
        loss_pred_cf_batch = F.binary_cross_entropy_with_logits(logits_cf, y_b, reduction='none')
        
        total_loss_batch = recon_loss_batch + 0.1 * kld_loss_batch + loss_pred_f_batch + loss_pred_cf_batch
        return total_loss_batch, recon_loss_batch, kld_loss_batch, logits_f

    model.train()
    for epoch in range(epochs):
        epoch_losses = {'total':0, 'recon':0, 'kld':0, 'irm':0, 'proxy_s':0}
        
        for x_b, m_x_b, s_b, m_s_b, y_b in dataloader:
            x_b, m_x_b, s_b, m_s_b, y_b = x_b.to(device), m_x_b.to(device), s_b.to(device), m_s_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            
            # Proxy S Predictor
            logits_s_proxy = model.proxy_s(x_b) # Use X_in as proxy
            prob_s = torch.sigmoid(logits_s_proxy)
            
            # Supervised proxy loss on observed S
            loss_proxy_s = F.binary_cross_entropy_with_logits(logits_s_proxy, s_b, reduction='none')
            loss_proxy_s = (loss_proxy_s * m_s_b).sum() / m_s_b.sum().clamp(min=1)
            
            # Loss for observed S
            loss_obs, recon_obs, kld_obs, logits_f_obs = compute_loss_for_s(s_b, x_b, m_x_b, y_b)
            
            # Marginalized Loss for missing S
            s0 = torch.zeros_like(s_b)
            s1 = torch.ones_like(s_b)
            loss0, recon0, kld0, _ = compute_loss_for_s(s0, x_b, m_x_b, y_b)
            loss1, recon1, kld1, _ = compute_loss_for_s(s1, x_b, m_x_b, y_b)
            
            loss_marg = prob_s * loss1 + (1 - prob_s) * loss0
            recon_marg = prob_s * recon1 + (1 - prob_s) * recon0
            kld_marg = prob_s * kld1 + (1 - prob_s) * kld0
            
            # Combine based on m_s_b
            final_loss = loss_obs * m_s_b + loss_marg * (1 - m_s_b)
            final_recon = recon_obs * m_s_b + recon_marg * (1 - m_s_b)
            final_kld = kld_obs * m_s_b + kld_marg * (1 - m_s_b)
            
            # Calculate IRM Penalty for observed S
            irm_pen = 0
            for s_val in [0, 1]:
                mask_s_val = ((s_b == s_val) & (m_s_b == 1)).squeeze()
                if mask_s_val.sum() > 1:
                    irm_pen += irm_penalty(logits_f_obs[mask_s_val], y_b[mask_s_val])
            
            total_loss = final_loss.mean() + lambda_irm * irm_pen + loss_proxy_s
            
            total_loss.backward()
            optimizer.step()
            
            epoch_losses['total'] += total_loss.item()
            epoch_losses['recon'] += final_recon.mean().item()
            epoch_losses['kld'] += final_kld.mean().item()
            epoch_losses['proxy_s'] += loss_proxy_s.item()
            epoch_losses['irm'] += irm_pen.item() if isinstance(irm_pen, torch.Tensor) else irm_pen
            
        N = len(dataloader)
        elbo = -(epoch_losses['recon']/N + epoch_losses['kld']/N)
        
        history['total_loss'].append(epoch_losses['total']/N)
        history['recon_loss'].append(epoch_losses['recon']/N)
        history['kld_loss'].append(epoch_losses['kld']/N)
        history['elbo'].append(elbo)
        history['proxy_s_loss'].append(epoch_losses['proxy_s']/N)
        history['irm_penalty'].append(epoch_losses['irm']/N)
            
        if progress_bar is not None:
            progress_bar.progress((epoch + 1) / epochs)
        if status_text is not None:
            status_text.text(f"ProCFair Epoch {epoch+1}/{epochs} | Loss: {epoch_losses['total']/N:.4f} | ELBO: {elbo:.4f} | Proxy-S Loss: {epoch_losses['proxy_s']/N:.4f}")
            
    return model, history

def evaluate_procfair(model, X_miss, S_miss, Y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    mask_x = ~np.isnan(X_miss)
    mask_s = ~np.isnan(S_miss)
    
    X_in = np.copy(X_miss)
    for j in range(X_in.shape[1]):
        col_mean = np.nanmean(X_in[:, j])
        X_in[np.isnan(X_in[:, j]), j] = col_mean

    S_in = np.copy(S_miss)
    S_in[np.isnan(S_in)] = 0.0

    X_in_t = torch.tensor(X_in, dtype=torch.float32).to(device)
    mask_x_t = torch.tensor(mask_x, dtype=torch.float32).to(device)
    S_t = torch.tensor(S_in, dtype=torch.float32).unsqueeze(1).to(device)
    mask_s_t = torch.tensor(mask_s, dtype=torch.float32).unsqueeze(1).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        # Infer missing S
        logits_s_proxy = model.proxy_s(X_in_t)
        preds_s_prob = torch.sigmoid(logits_s_proxy)
        preds_s_binary = (preds_s_prob > 0.5).float()
        
        # Use inferred S where true S is missing
        S_final = S_t * mask_s_t + preds_s_binary * (1 - mask_s_t)
        
        # CIWAE
        x_hat, _, _, z = model.forward_ciwae(X_in_t, mask_x_t, S_final)
        x_imputed = X_in_t * mask_x_t + x_hat * (1 - mask_x_t)
        
        logits_f = model.predictor(x_imputed)
        preds_f = torch.sigmoid(logits_f)
        acc = ((preds_f > 0.5) == Y_t).float().mean().item()
        
        s_cf = 1 - S_final
        x_cf_hat = model.decoder(z, s_cf)
        x_cf = X_in_t * mask_x_t + x_cf_hat * (1 - mask_x_t)
        
        logits_cf = model.predictor(x_cf)
        preds_cf = torch.sigmoid(logits_cf)
        
        # Calculate W1-dist and MMD
        preds_f_np = preds_f.cpu().numpy().flatten()
        preds_cf_np = preds_cf.cpu().numpy().flatten()
        
        w1_dist = wasserstein_distance(preds_f_np, preds_cf_np)
        mmd_val = compute_mmd_loss(preds_f, preds_cf)
        
        # Accuracy of Proxy S predictor (on observed S)
        observed_s_mask = (mask_s_t == 1).squeeze()
        if observed_s_mask.sum() > 0:
            s_acc = ((preds_s_binary[observed_s_mask] == S_t[observed_s_mask]).float().mean().item())
        else:
            s_acc = 1.0 # If no S observed
            
    return acc, w1_dist, mmd_val, s_acc, preds_f_np, preds_cf_np

def train_cfairmd_baseline(X_miss, S_miss, Y, input_dim, hidden_dim=50, latent_dim=10,  
                           epochs=50, batch_size=256, lr=0.001, lambda_irm=1.0,
                           progress_bar=None, status_text=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    mask_x = ~np.isnan(X_miss)
    
    X_in = np.copy(X_miss)
    for j in range(X_in.shape[1]):
        col_mean = np.nanmean(X_in[:, j])
        X_in[np.isnan(X_in[:, j]), j] = col_mean

    # Baseline CFairMD: simply impute missing S with 0 (Mode imputation)
    S_in = np.copy(S_miss)
    S_in[np.isnan(S_in)] = 0.0

    X_in_t = torch.tensor(X_in, dtype=torch.float32)
    mask_x_t = torch.tensor(mask_x, dtype=torch.float32)
    S_t = torch.tensor(S_in, dtype=torch.float32).unsqueeze(1)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_in_t, mask_x_t, S_t, Y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    from models import ProCFair
    # Reusing ProCFair architecture but ignoring the proxy output
    model = ProCFair(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_b, m_x_b, s_b, y_b in dataloader:
            x_b, m_x_b, s_b, y_b = x_b.to(device), m_x_b.to(device), s_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            
            x_hat, mu, logvar, z = model.forward_ciwae(x_b, m_x_b, s_b)
            recon_loss = F.mse_loss(x_hat * m_x_b, x_b * m_x_b, reduction='none').sum(dim=1).mean()
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            x_imputed = x_b * m_x_b + x_hat * (1 - m_x_b)
            s_cf = 1 - s_b
            x_cf_hat = model.decoder(z, s_cf)
            x_cf = x_b * m_x_b + x_cf_hat * (1 - m_x_b)
            
            logits_f = model.predictor(x_imputed)
            loss_pred_f = F.binary_cross_entropy_with_logits(logits_f, y_b)
            
            logits_cf = model.predictor(x_cf)
            loss_pred_cf = F.binary_cross_entropy_with_logits(logits_cf, y_b)
            
            irm_pen = 0
            for s_val in [0, 1]:
                mask_s_val = (s_b == s_val).squeeze()
                if mask_s_val.sum() > 1:
                    irm_pen += irm_penalty(logits_f[mask_s_val], y_b[mask_s_val])
            
            total_loss = recon_loss + 0.1 * kld_loss + loss_pred_f + loss_pred_cf + lambda_irm * irm_pen
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        if progress_bar is not None:
            progress_bar.progress((epoch + 1) / epochs)
        if status_text is not None:
            status_text.text(f"CFairMD (Baseline) Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    return model

def evaluate_cfairmd_baseline(model, X_miss, S_miss, Y):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    mask_x = ~np.isnan(X_miss)
    
    X_in = np.copy(X_miss)
    for j in range(X_in.shape[1]):
        col_mean = np.nanmean(X_in[:, j])
        X_in[np.isnan(X_in[:, j]), j] = col_mean

    S_in = np.copy(S_miss)
    S_in[np.isnan(S_in)] = 0.0 # Baseline assumption

    X_in_t = torch.tensor(X_in, dtype=torch.float32).to(device)
    mask_x_t = torch.tensor(mask_x, dtype=torch.float32).to(device)
    S_t = torch.tensor(S_in, dtype=torch.float32).unsqueeze(1).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1).to(device)
    
    with torch.no_grad():
        x_hat, _, _, z = model.forward_ciwae(X_in_t, mask_x_t, S_t)
        x_imputed = X_in_t * mask_x_t + x_hat * (1 - mask_x_t)
        
        logits_f = model.predictor(x_imputed)
        preds_f = torch.sigmoid(logits_f)
        acc = ((preds_f > 0.5) == Y_t).float().mean().item()
        
        s_cf = 1 - S_t
        x_cf_hat = model.decoder(z, s_cf)
        x_cf = X_in_t * mask_x_t + x_cf_hat * (1 - mask_x_t)
        
        logits_cf = model.predictor(x_cf)
        preds_cf = torch.sigmoid(logits_cf)
        
        preds_f_np = preds_f.cpu().numpy().flatten()
        preds_cf_np = preds_cf.cpu().numpy().flatten()
        
        w1_dist = wasserstein_distance(preds_f_np, preds_cf_np)
        mmd_val = compute_mmd_loss(preds_f, preds_cf)
            
    return acc, w1_dist, mmd_val, preds_f_np, preds_cf_np

def train_claire(X_miss, S_miss, Y, input_dim, hidden_dim=50, latent_dim=10, 
                 epochs=50, batch_size=256, lr=0.001, lambda_irm=1.0, alpha_mmd=1.0, beta_cf=1.0,
                 progress_bar=None, status_text=None):
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_x = ~np.isnan(X_miss)
    
    X_in = np.copy(X_miss)
    for j in range(X_in.shape[1]):
        col_mean = np.nanmean(X_in[:, j])
        X_in[np.isnan(X_in[:, j]), j] = col_mean

    S_in = np.copy(S_miss)
    S_in[np.isnan(S_in)] = 0.0

    X_in_t = torch.tensor(X_in, dtype=torch.float32)
    mask_x_t = torch.tensor(mask_x, dtype=torch.float32)
    S_t = torch.tensor(S_in, dtype=torch.float32).unsqueeze(1)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_in_t, mask_x_t, S_t, Y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    from models import ProCFair
    model = ProCFair(input_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for x_b, m_x_b, s_b, y_b in dataloader:
            x_b, m_x_b, s_b, y_b = x_b.to(device), m_x_b.to(device), s_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            
            x_hat, mu, logvar, z = model.forward_ciwae(x_b, m_x_b, s_b)
            recon_loss = F.mse_loss(x_hat * m_x_b, x_b * m_x_b, reduction='none').sum(dim=1).mean()
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # Latent distribution matching (MMD on Z)
            z_s0 = z[(s_b == 0).squeeze()]
            z_s1 = z[(s_b == 1).squeeze()]
            loss_mmd_z = 0
            if len(z_s0) > 0 and len(z_s1) > 0:
                loss_mmd_z = compute_mmd_loss(z_s0, z_s1)
            
            # Counterfactual Representation Penalty
            s_cf = 1 - s_b
            x_cf_hat = model.decoder(z, s_cf)
            x_cf = x_b * m_x_b + x_cf_hat * (1 - m_x_b)
            # Re-encode x_cf to get z_cf
            _, _, _, z_cf = model.forward_ciwae(x_cf, m_x_b, s_cf) # pass s_cf but actually encoder only uses x_cf in ProCFair
            loss_cf_rep = F.mse_loss(z, z_cf)
            
            # Predictor Loss and IRM
            x_imputed = x_b * m_x_b + x_hat * (1 - m_x_b)
            logits_f = model.predictor(x_imputed)
            loss_pred_f = F.binary_cross_entropy_with_logits(logits_f, y_b)
            
            irm_pen = 0
            for s_val in [0, 1]:
                mask_s_val = (s_b == s_val).squeeze()
                if mask_s_val.sum() > 1:
                    irm_pen += irm_penalty(logits_f[mask_s_val], y_b[mask_s_val])
            
            total_loss = recon_loss + 0.1 * kld_loss + alpha_mmd * loss_mmd_z + beta_cf * loss_cf_rep + loss_pred_f + lambda_irm * irm_pen
            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item()
            
        if progress_bar is not None:
            progress_bar.progress((epoch + 1) / epochs)
        if status_text is not None:
            status_text.text(f"CLAIRE Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(dataloader):.4f}")
            
    return model

def evaluate_claire(model, X_miss, S_miss, Y):
    # Same evaluation pipeline as baseline
    return evaluate_cfairmd_baseline(model, X_miss, S_miss, Y)

def train_adversarial_cf(X_miss, S_miss, Y, input_dim, hidden_dim=50, latent_dim=10, 
                         epochs=50, batch_size=256, lr=0.001, lambda_irm=1.0, beta_cf=1.0,
                         progress_bar=None, status_text=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mask_x = ~np.isnan(X_miss)
    
    X_in = np.copy(X_miss)
    for j in range(X_in.shape[1]):
        col_mean = np.nanmean(X_in[:, j])
        X_in[np.isnan(X_in[:, j]), j] = col_mean

    S_in = np.copy(S_miss)
    S_in[np.isnan(S_in)] = 0.0

    X_in_t = torch.tensor(X_in, dtype=torch.float32)
    mask_x_t = torch.tensor(mask_x, dtype=torch.float32)
    S_t = torch.tensor(S_in, dtype=torch.float32).unsqueeze(1)
    Y_t = torch.tensor(Y, dtype=torch.float32).unsqueeze(1)
    
    dataset = TensorDataset(X_in_t, mask_x_t, S_t, Y_t)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    from models import ProCFair, DiscriminatorS
    model = ProCFair(input_dim, hidden_dim, latent_dim).to(device)
    discriminator = DiscriminatorS(latent_dim).to(device)
    
    opt_g = optim.Adam(model.parameters(), lr=lr)
    opt_d = optim.Adam(discriminator.parameters(), lr=lr)
    
    model.train()
    discriminator.train()
    for epoch in range(epochs):
        epoch_loss_g = 0
        for x_b, m_x_b, s_b, y_b in dataloader:
            x_b, m_x_b, s_b, y_b = x_b.to(device), m_x_b.to(device), s_b.to(device), y_b.to(device)
            
            # Step 1: Train Discriminator
            opt_d.zero_grad()
            with torch.no_grad():
                _, _, _, z_det = model.forward_ciwae(x_b, m_x_b, s_b)
            pred_s = discriminator(z_det.detach())
            loss_d = F.binary_cross_entropy_with_logits(pred_s, s_b)
            loss_d.backward()
            opt_d.step()
            
            # Step 2: Train Generator/Predictor
            opt_g.zero_grad()
            x_hat, mu, logvar, z = model.forward_ciwae(x_b, m_x_b, s_b)
            recon_loss = F.mse_loss(x_hat * m_x_b, x_b * m_x_b, reduction='none').sum(dim=1).mean()
            kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
            
            # Adversarial Penalty
            pred_s_fake = discriminator(z)
            loss_adv = F.binary_cross_entropy_with_logits(pred_s_fake, torch.full_like(s_b, 0.5))
            
            # Counterfactual Penalty
            s_cf = 1 - s_b
            x_cf_hat = model.decoder(z, s_cf)
            x_cf = x_b * m_x_b + x_cf_hat * (1 - m_x_b)
            _, _, _, z_cf = model.forward_ciwae(x_cf, m_x_b, s_cf)
            loss_cf_rep = F.mse_loss(z, z_cf)
            
            # Predictor Loss
            x_imputed = x_b * m_x_b + x_hat * (1 - m_x_b)
            logits_f = model.predictor(x_imputed)
            loss_pred_f = F.binary_cross_entropy_with_logits(logits_f, y_b)
            
            irm_pen = 0
            for s_val in [0, 1]:
                mask_s_val = (s_b == s_val).squeeze()
                if mask_s_val.sum() > 1:
                    irm_pen += irm_penalty(logits_f[mask_s_val], y_b[mask_s_val])
            
            total_loss_g = recon_loss + 0.1 * kld_loss + 1.0 * loss_adv + beta_cf * loss_cf_rep + loss_pred_f + lambda_irm * irm_pen
            total_loss_g.backward()
            opt_g.step()
            epoch_loss_g += total_loss_g.item()
            
        if progress_bar is not None:
            progress_bar.progress((epoch + 1) / epochs)
        if status_text is not None:
            status_text.text(f"Adversarial CF Epoch {epoch+1}/{epochs} | Loss G: {epoch_loss_g/len(dataloader):.4f}")
            
    return model

def evaluate_adversarial_cf(model, X_miss, S_miss, Y):
    return evaluate_cfairmd_baseline(model, X_miss, S_miss, Y)
