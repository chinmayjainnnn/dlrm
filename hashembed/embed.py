import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from embedding import *


class EmbeddingDataset(Dataset):
    def __init__(self, indices, embeddings):
        self.indices = indices
        self.embeddings = embeddings

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        index = self.indices[idx]
        embedding = self.embeddings[index]
        return torch.tensor(index, dtype=torch.long), embedding

    i=13
    
    
    pretrained_embeddings = np.load(f'hashembed/sparse_embeddings/emb_l.{i}.weight.npy')  # Shape: (num_embeddings, embedding_dim)
    pretrained_embeddings = torch.from_numpy(pretrained_embeddings)  # Convert to torch tensor
    pretrained_embedding_layer = nn.Embedding.from_pretrained(pretrained_embeddings, freeze=True)
    
    num_embeddings, embedding_dim = pretrained_embeddings.shape
    print(f'{num_embeddings = }')
    print(f'{embedding_dim = }')
    num_hashes = 4 
    num_buckets = (num_embeddings * num_hashes) // embedding_dim
    hash_embedding = HashEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_buckets=num_buckets,
        num_hashes=num_hashes,
        train_sharedEmbed=True,
        train_weight=True,
        append_weight=False,
        aggregation_mode='sum',
        mask_zero=False,
        seed=42  # Optional seed for reproducibility
    )
    print(hash_embedding.shared_embeddings.weight.shape)
    print(hash_embedding.importance_weights.weight.shape)
    indices = np.arange(num_embeddings)

    class EmbeddingDataset(Dataset):
        def __init__(self, indices):
            self.indices = indices

        def __len__(self):
            return len(self.indices)

        def __getitem__(self, idx):
            index = self.indices[idx]
            return torch.tensor(index, dtype=torch.long)

    batch_size = 512
    num_epochs = 100
    criterion = nn.MSELoss()
    optimizer = optim.Adam(hash_embedding.parameters(), lr=0.002)



    dataset = EmbeddingDataset(indices)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if torch.cuda.is_available():
        print("yeah")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    hash_embedding.to(device)
    pretrained_embedding_layer.to(device)

    # for epoch in range(num_epochs):
    for epoch in tqdm(range(num_epochs), desc=f"Training Embedding File {i}", unit="epoch"):
        hash_embedding.train()
        total_loss = 0
        cj=0
        for batch_indices in dataloader:
            batch_indices = batch_indices.to(device)
            
            # print(cj)
            # if cj%1000==0:
            #     print(f'{cj = }')
            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Get embeddings from HashEmbedding
            outputs = hash_embedding(batch_indices)

            # Get target embeddings from the pretrained embedding layer
            with torch.no_grad():
                target_embeddings = pretrained_embedding_layer(batch_indices)

            # Compute loss
            loss = criterion(outputs, target_embeddings)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_indices.size(0)

        avg_loss = total_loss / len(dataset)
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}')

    shared_embeddings = hash_embedding.shared_embeddings.weight
    importance_weights = hash_embedding.importance_weights.weight

    # 2. Convert to NumPy arrays (optional)
    shared_embeddings_np = shared_embeddings.cpu().detach().numpy()
    importance_weights_np = importance_weights.cpu().detach().numpy()

    # 3. Save to .npy files
    np.save(f'hashembed/shared_embeddings/shared_embeddings{i}.npy', shared_embeddings_np)
    np.save(f'hashembed/importance_weights/importance_weights{i}.npy', importance_weights_np)


# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# use this to load back the embeddings
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\



"""# Load the embeddings from .npy files
shared_embeddings_np = np.load('shared_embeddings.npy')
importance_weights_np = np.load('importance_weights.npy')

# Convert NumPy arrays to tensors
shared_embeddings_tensor = torch.tensor(shared_embeddings_np)
importance_weights_tensor = torch.tensor(importance_weights_np)

# Load the tensors into the model's parameters
hash_embedding.shared_embeddings.weight.data = shared_embeddings_tensor
hash_embedding.importance_weights.weight.data = importance_weights_tensor

# Ensure that the parameters require gradients if you plan to continue training
hash_embedding.shared_embeddings.weight.requires_grad = True
hash_embedding.importance_weights.weight.requires_grad = True
"""