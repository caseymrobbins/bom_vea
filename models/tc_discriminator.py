# models/tc_discriminator.py
# Total correlation discriminator for latent separability

import torch.nn as nn


class TCDiscriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


def create_tc_discriminator(input_dim, device='cuda', hidden_dim=256):
    disc = TCDiscriminator(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    print(f"TC Discriminator ({input_dim} dims) params: {sum(p.numel() for p in disc.parameters()):,}")
    return disc
