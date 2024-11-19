import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from embedding import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# index=9  #yet to be done  #embedding table no.
#screen s2 -21
#screen s1 -20
#screen s3 -19
#screen s4 -0
# screen s5 -9


for index in range(9,10):
    target_embedding_table = torch.tensor(np.load(f'../criteo_pretrain_model/embedding_sparse/emb_l.{index}.weight.npy'), dtype=torch.float32).to(device)
    num_embeddings, embedding_dim = target_embedding_table.shape
    print(num_embeddings,embedding_dim)

    num_hashes = 4
    num_buckets = (num_embeddings * num_hashes) // embedding_dim

    hash_embedding = HashEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_buckets=None,  # Adjust based on your requirements
        num_hashes=num_hashes,
        train_sharedEmbed=True,
        train_weight=True,
        aggregation_mode='sum',
        seed=42
    ).to(device)

    # indices = np.arange(num_embeddings)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(hash_embedding.parameters(), lr=0.001)

    # Training loop
    num_epochs = 500  
    batch_size = 512 

    for epoch in range(num_epochs):
        epoch_loss = 0
        for i in range(0, num_embeddings, batch_size):
            batch_indices = torch.arange(i, min(i + batch_size, num_embeddings), device=device).long()
            # print(f'{i=}{batch_indices=}')
            batch_target = target_embedding_table[batch_indices]
            batch_output = hash_embedding(batch_indices)

            loss = criterion(batch_output, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

    weights = {
        'shared_embeddings': hash_embedding.shared_embeddings.weight.detach().cpu(),
        'importance_weights': hash_embedding.importance_weights.weight.detach().cpu()
    }
    filename=f"weights/hash_embedding_weights{index}.pth"
    torch.save(weights, filename)
    print("Weights saved to hash_embedding_weights.pth")
