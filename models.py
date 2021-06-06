import torch
import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, data_shape, latent_dim=None):
        super(Generator, self).__init__()
        self.data_shape = data_shape

        self.latent_dim = int(np.prod(self.data_shape)) if latent_dim is None else latent_dim

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat))
            layers.append(nn.PReLU())
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.data_shape))),
        )

    def forward(self, z):
        out = self.model(z)
        return out

    def save(self, path):
        save_dict = {
            'latent_dim': self.latent_dim,
            'model': self.model.state_dict(),
            'data_shape': self.data_shape,
        }
        torch.save(save_dict, path)

        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        G = Generator(save_dict['data_shape'], latent_dim=save_dict['latent_dim'])
        G.model.load_state_dict(save_dict["model"])

        return G


class Discriminator(nn.Module):
    def __init__(self, data_shape):
        super(Discriminator, self).__init__()
        self.data_shape = data_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.data_shape)), 512),
            nn.PReLU(),
            nn.Linear(512, 256),
            nn.PReLU(),
            nn.Linear(256, 128),
            nn.PReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, mol):
        validity = self.model(mol)
        return validity

    def save(self, path):
        save_dict = {
            'model': self.model.state_dict(),
            'data_shape': self.data_shape,
        }
        torch.save(save_dict, path)
        return

    @staticmethod
    def load(path):
        save_dict = torch.load(path)
        D = Discriminator(save_dict['data_shape'])
        D.model.load_state_dict(save_dict["model"])

        return D