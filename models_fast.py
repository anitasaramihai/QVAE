import numpy as np
import torch
import torch.nn as nn

# Versiune rapidă FĂRĂ circuit cuantic - doar MLP clasic
# Pentru a testa viteza și compara cu hybrid VAE

class FastVAE(nn.Module):  # MLP puro, fără quantum
  
    def __init__(self, input_size, hidden_size=128, latent_size=4):
        super(FastVAE, self).__init__()

        # Encoder - clasic
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        
        # Latent distribution
        self.fc_mu = nn.Linear(hidden_size // 2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size // 2, latent_size)
        
        # Decoder - clasic
        self.fc3 = nn.Linear(latent_size, hidden_size // 2)
        self.fc4 = nn.Linear(hidden_size // 2, hidden_size)
        self.fc_mu_x = nn.Linear(hidden_size, input_size)
        self.fc_logvar_x = nn.Linear(hidden_size, input_size)

    def encode(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu_z = self.fc_mu(h)
        logvar_z = self.fc_logvar(h)
        return mu_z, logvar_z

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = torch.relu(self.fc3(z))
        h = torch.relu(self.fc4(h))
        mu_x = self.fc_mu_x(h)
        logvar_x = self.fc_logvar_x(h)
        return mu_x, logvar_x

    def forward(self, x):
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x, logvar_x = self.decode(z)
        return mu_x, logvar_x, mu_z, logvar_z


def vae_loss(x, mu_x, logvar_x, mu_z, logvar_z, beta=1.0):
    recon_loss = 0.5 * torch.sum(
        logvar_x
        + ((x - mu_x) ** 2) / torch.exp(logvar_x)
        + np.log(2 * np.pi),
        dim=1
    )

    kl_loss = -0.5 * torch.sum(
        1 + logvar_z - mu_z.pow(2) - logvar_z.exp(),
        dim=1
    )

    total_loss = torch.mean(recon_loss + beta * kl_loss)

    return (
        total_loss,
        torch.mean(recon_loss),
        torch.mean(kl_loss)
    )
