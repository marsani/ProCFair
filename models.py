import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Input dim includes covariates X + missing mask M
        self.fc1 = nn.Linear(input_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, mask):
        x_in = torch.cat([x, mask], dim=1)
        h = F.elu(self.fc1(x_in))
        h = F.elu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, n_sensitive_groups=2):
        super().__init__()
        # TAR structure: separate heads for sensitive attribute S
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Heads
        self.heads = nn.ModuleList([
            nn.Linear(hidden_dim, output_dim) for _ in range(n_sensitive_groups)
        ])

    def forward(self, z, s):
        h = F.elu(self.fc1(z))
        h = F.elu(self.fc2(h))
        
        # Select head based on s
        # s is binary (0 or 1)
        out = torch.zeros(z.size(0), self.heads[0].out_features).to(z.device)
        s = s.long()
        for i in range(len(self.heads)):
            mask = (s == i).squeeze()
            if mask.sum() > 0:
                out[mask] = self.heads[i](h[mask])
        return out

class Predictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Predicts Y from X (without S to be unaware, but uses invariant penalty)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.elu(self.fc1(x))
        h = F.elu(self.fc2(h))
        logits = self.out(h)
        return logits

def irm_penalty(logits, y):
    """
    Compute IRMv1 penalty.
    """
    scale = torch.tensor(1.).to(logits.device).requires_grad_()
    loss = F.binary_cross_entropy_with_logits(logits * scale, y)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return torch.sum(grad**2)

class ProxyPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Predicts prob(S=1 | X_P)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        logits = self.out(h)
        return logits

class DiscriminatorS(nn.Module):
    # Predicts S from Latent Space Z for Adversarial Learning (Grari et al.)
    def __init__(self, latent_dim, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Linear(hidden_dim//2, 1)
        )
        
    def forward(self, z):
        return self.net(z)

class ProCFair(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, n_importance_samples=5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.n_samples = n_importance_samples
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.predictor = Predictor(input_dim, hidden_dim)
        # We use observed X as proxy variables
        self.proxy_s = ProxyPredictor(input_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_ciwae(self, x_obs, mask, s):
        # Encode
        mu, logvar = self.encoder(x_obs, mask)
        
        # Sample Z
        # For simplicity in this demo, we use single sample for loss but we can average predictions
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_hat = self.decoder(z, s)
        
        return x_hat, mu, logvar, z

    def generate_counterfactuals(self, x_obs, mask, s_factual):
        # Generate counterfactual by flipping S
        s_cf = 1 - s_factual
        mu, logvar = self.encoder(x_obs, mask)
        z = self.reparameterize(mu, logvar)
        
        x_factual = self.decoder(z, s_factual)
        x_counterfactual = self.decoder(z, s_cf)
        
        return x_factual, x_counterfactual

