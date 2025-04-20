import numpy as np
import faiss
import pandas as pd
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# Loading Projection Head Model
# ------------------------------------------------------------
class ProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim=256):
        super(ProjectionHead, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, output_dim),
        )
    
    def forward(self, x):
        return self.model(x)
    
model = ProjectionHead(input_dim=1280, output_dim=256)
model.load_state_dict(torch.load("/scratch/gpfs/jr8867/main/db/models/triplets-hard/triplets-hard-model-large.pth"))
model.to(device)
model.eval() # Set model to eval mode, since we are not training, and just need to project the embeddings

# ------------------------------------------------------------
# Loading Embeddings
# ------------------------------------------------------------  

train_subset_dir = '/scratch/gpfs/jr8867/main/db/family-split-train-test/train_subset'
test_dir = '/scratch/gpfs/jr8867/main/db/family-split-train-test/test'

train_embeddings = np.load(f'{train_subset_dir}/train_subset_embeddings.npy')
test_embeddings = np.load(f'{test_dir}/test_embeddings.npy')

train_metadata = pd.read_csv(f'{train_subset_dir}/train_subset_metadata.csv')
test_metadata = pd.read_csv(f'{test_dir}/test_metadata.csv')

print(f"Train subset embeddings shape: {train_embeddings.shape}")
print(f"Test embeddings shape: {test_embeddings.shape}")
print(f"Train subset metadata shape: {train_metadata.shape}")
print(f"Test metadata shape: {test_metadata.shape}")

# ------------------------------------------------------------
# Projecting Embeddings
# ------------------------------------------------------------

with torch.no_grad():
    train_projected_embeddings = model(torch.tensor(train_embeddings, dtype=torch.float32).to(device)).cpu().numpy()
    test_projected_embeddings = model(torch.tensor(test_embeddings, dtype=torch.float32).to(device)).cpu().numpy()

# ------------------------------------------------------------
# FAISS Indexing
# ------------------------------------------------------------

# Train Index
index = faiss.IndexFlatL2(train_projected_embeddings.shape[1])
index.add(train_projected_embeddings)
faiss.write_index(index, "/scratch/gpfs/jr8867/main/db/indices/triplets-hard/triplets-hard_train.index")

np.save("/scratch/gpfs/jr8867/main/db/indices/triplets-hard/triplets-hard_train_embeddings.npy", train_projected_embeddings)
train_metadata.to_csv("/scratch/gpfs/jr8867/main/db/indices/triplets-hard/triplets-hard_train_metadata.csv", index=False)

# Test Index
index = faiss.IndexFlatL2(test_projected_embeddings.shape[1])
index.add(test_projected_embeddings)
faiss.write_index(index, "/scratch/gpfs/jr8867/main/db/indices/triplets-hard/triplets-hard_test.index")

np.save("/scratch/gpfs/jr8867/main/db/indices/triplets-hard/triplets-hard_test_embeddings.npy", test_projected_embeddings)
test_metadata.to_csv("/scratch/gpfs/jr8867/main/db/indices/triplets-hard/triplets-hard_test_metadata.csv", index=False)

print("Saved Embeddings, Metadata, and FAISS index!")
