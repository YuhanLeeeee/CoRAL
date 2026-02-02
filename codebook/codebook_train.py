import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
import time


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, input_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.commitment_cost = commitment_cost

        self.encoder = nn.Linear(self.input_dim, self.embedding_dim)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / self.num_embeddings, 1.0 / self.num_embeddings)

        self.register_buffer('cluster_counts', torch.zeros(num_embeddings, dtype=torch.long))
        self.register_buffer('total_vectors_processed', torch.tensor(0, dtype=torch.long))

    def get_usage_stats(self):
        if self.total_vectors_processed == 0:
            perplexity = torch.tensor(0.0)
            dead_codes = self.num_embeddings
        else:
            prob = self.cluster_counts.bfloat16() / self.total_vectors_processed

            perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-10)))

            dead_codes = torch.sum(self.cluster_counts == 0).item()

        return {
            "perplexity": perplexity.item(),
            "dead_codes": dead_codes,
            "usage_counts": self.cluster_counts.detach().cpu().numpy()
        }

    def reset_counters(self):
        self.cluster_counts.zero_()
        self.total_vectors_processed.zero_()

    def reset_dead_codes(self, encoder_outputs):
        dead_indices = torch.where(self.cluster_counts == 0)[0]
        if dead_indices.numel() == 0:
            print("No dead codes to reset.")
            return
        print(f"Resetting {dead_indices.numel()} dead codes.")
        encoder_outputs_flat = encoder_outputs.contiguous().view(-1, self.embedding_dim)
        num_dead = dead_indices.numel()
        num_candidates = encoder_outputs_flat.size(0)
        replacement_indices = torch.randint(0, num_candidates, (num_dead,))
        replacement_vectors = encoder_outputs_flat[replacement_indices]
        with torch.no_grad():
            self.embedding.weight.data[dead_indices] = replacement_vectors

    def forward(self, latents):
        # [batch_size, 64, 32] -> [batch_size, 64, 4096]
        latents_projected = self.encoder(latents)

        flat_latents = latents_projected.reshape(-1, self.embedding_dim)

        distances = (torch.sum(flat_latents ** 2, dim=1, keepdim=True)
                     - 2 * torch.matmul(flat_latents, self.embedding.weight.t())
                     + torch.sum(self.embedding.weight ** 2, dim=1, keepdim=False))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        if self.training:
            current_counts = torch.bincount(encoding_indices.flatten(), minlength=self.num_embeddings)

            self.cluster_counts.data.add_(current_counts)
            self.total_vectors_processed.data.add_(encoding_indices.numel())

        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=latents.device,
                                dtype=torch.bfloat16)
        encodings.scatter_(1, encoding_indices, 1)

        quantized_flat = torch.matmul(encodings, self.embedding.weight)
        quantized = quantized_flat.view(latents_projected.shape)

        codebook_loss = F.mse_loss(quantized, latents_projected.detach())
        commit_loss = F.mse_loss(latents_projected, quantized.detach())
        loss = codebook_loss + self.commitment_cost * commit_loss

        quantized = latents_projected + (quantized - latents_projected).detach()

        return {
            "quantized": quantized,  # [batch_size, 64, 4096]
            "loss": loss,
            "encoding_indices": encoding_indices.view(latents.shape[0], -1),
            "z_e": latents_projected
        }


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_shape):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_shape = latent_shape
        self.latent_dim_flat = np.prod(latent_shape)  # 64 * 32 = 2048

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.latent_dim_flat),
            nn.ReLU()
        )

    def forward(self, x):
        # [batch, 2048] -> [batch, 2048]
        x = self.model(x)
        # [batch, 2048] -> [batch, 64, 32]
        return x.view(-1, *self.latent_shape)


class Decoder(nn.Module):
    def __init__(self, output_dim, quantized_shape):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.quantized_shape = quantized_shape
        self.quantized_dim_flat = np.prod(quantized_shape)  # 64 * 4096

        self.model = nn.Sequential(
            nn.Linear(self.quantized_dim_flat, 4096),
            nn.ReLU(),
            nn.Linear(4096, self.output_dim)
        )

    def forward(self, z_q):
        # [batch, 64, 4096] -> [batch, 64 * 4096]
        z_q_flat = z_q.view(-1, self.quantized_dim_flat)
        # [batch, 64 * 4096] -> [batch, 2048]
        x_recon = self.model(z_q_flat)
        return x_recon


class VQVAE(nn.Module):
    def __init__(self, input_dim, latent_shape, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, latent_shape)

        self.vq_layer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            input_dim=latent_shape[-1],
            commitment_cost=commitment_cost
        )

        quantized_shape = (latent_shape[0], embedding_dim)
        self.decoder = Decoder(input_dim, quantized_shape)

    def quantize(self, x):
        z_e = self.encoder(x)  # -> [batch, 64, 32]

        vq_output = self.vq_layer(z_e)
        return vq_output['quantized']  # -> [batch, 64, 4096]

    def forward(self, x):
        z_e = self.encoder(x)  # -> [batch, 64, 32]

        vq_output = self.vq_layer(z_e)
        z_q = vq_output['quantized']  # -> [batch, 64, 4096]
        vq_loss = vq_output['loss']
        ze = vq_output['z_e']

        x_recon = self.decoder(z_q)  # -> [batch, 2048]

        return {
            "reconstruction": x_recon,
            "vq_loss": vq_loss,
            'z_e': ze
        }


# train
INPUT_DIM = 2048
TOKEN_NUM = 64
LATENT_SHAPE = (TOKEN_NUM, 32)
NUM_EMBEDDINGS = 1024
EMBEDDING_DIM = 4096
COMMITMENT_COST = 0.25
PATH = f'./checkpoints/model_{time.time()}.pt'

LEARNING_RATE = 3e-4
BATCH_SIZE = 512
NUM_EPOCHS = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = VQVAE(
    input_dim=INPUT_DIM,
    latent_shape=LATENT_SHAPE,
    num_embeddings=NUM_EMBEDDINGS,
    embedding_dim=EMBEDDING_DIM,
    commitment_cost=COMMITMENT_COST
)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

if __name__ == '__main__':
    from codebook_dataloader import CodeBookDataset

    train_dataset = CodeBookDataset()
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=8
    )
    print(f"Dataset created with {len(train_dataset)} samples.")

    print("\n--- Starting Training ---")
    model.train()
    for epoch in range(NUM_EPOCHS):
        model.vq_layer.reset_counters()
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_vq_loss = 0.0
        last_z_e = None

        for i, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
            batch_data = batch_data

            optimizer.zero_grad()

            outputs = model(batch_data)
            x_recon = outputs['reconstruction']
            vq_loss = outputs['vq_loss']
            z_e = outputs['z_e']

            last_z_e = z_e

            recon_loss = F.mse_loss(x_recon, batch_data)
            total_loss = recon_loss + vq_loss

            total_loss.backward()
            optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()

        num_batches = len(train_loader)
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] | "
              f"Total Loss: {epoch_total_loss / num_batches:.4f} | "
              f"Recon Loss: {epoch_recon_loss / num_batches:.4f} | "
              f"VQ Loss: {epoch_vq_loss / num_batches:.4f}")
        stats = model.vq_layer.get_usage_stats()
        print("-" * 50)
        print(f"  Codebook Perplexity: {stats['perplexity']:.4f}")
        print(f"  Number of Dead Codes: {stats['dead_codes']} / {model.vq_layer.num_embeddings}")
        print("-" * 50)
        if stats['dead_codes'] > 0:
            with torch.no_grad():
                model.vq_layer.reset_dead_codes(last_z_e)

    torch.save(model.state_dict(), PATH)
    print("\n--- Training Finished ---")

    print("\n--- Verifying the result ---")
    model.eval()
    with torch.no_grad():
        sample_batch, = next(iter(train_loader))
        sample_batch = sample_batch

        original_vector = sample_batch[0]
        reconstructed_vector = model(original_vector.unsqueeze(0))['reconstruction'].squeeze(0)

        mse = F.mse_loss(reconstructed_vector, original_vector)

        print(f"Original vector (first 5 dims):    {original_vector[:5].cpu().numpy()}")
        print(f"Reconstructed vector (first 5 dims): {reconstructed_vector[:5].cpu().numpy()}")
        print(f"\nMSE between original and reconstructed vector: {mse.item():.4f}")
