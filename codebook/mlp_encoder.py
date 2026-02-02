from torch import nn
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_shape):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_shape = latent_shape
        self.latent_dim_flat = np.prod(latent_shape)  # 64 * 32 = 2048

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim_flat),
            nn.SiLU()
        )

    def forward(self, x):
        # [batch, 2048] -> [batch, 2048]
        x = self.model(x)
        # [batch, 2048] -> [batch, 64, 32]
        return x.view(-1, *self.latent_shape)

class FP_MLP(nn.Module):
    def __init__(self, input_dim, latent_shape, num_embeddings, embedding_dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder = Encoder(input_dim, latent_shape)

        self.module1 = nn.Sequential(
            nn.Linear(latent_shape[-1], embedding_dim // 4),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim // 4, embedding_dim),
            nn.SiLU()
        )

    def quantize(self, x):
        z_e = self.encoder(x)  # -> [batch, 64, 32]
        output = self.module1(z_e)
        return output  # -> [batch, 64, 4096]