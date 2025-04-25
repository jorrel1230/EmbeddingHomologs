import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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
model.load_state_dict(torch.load("/scratch/gpfs/jr8867/main/db/models/triplets-random/triplets-random-model-large.pth", map_location=device))
model.to(device)
model.eval() # Set model to eval mode, since we are not training, and just need to project the embeddings

# ------------------------------------------------------------
# Loading Embeddings
# ------------------------------------------------------------  

test_dir = '/scratch/gpfs/jr8867/main/db/family-split-train-test/test'

test_embeddings = np.load(f'{test_dir}/test_embeddings.npy')
test_metadata = pd.read_csv(f'{test_dir}/test_metadata.csv')

print(test_metadata.shape)
print(test_embeddings.shape)

# ------------------------------------------------------------
# Projecting Embeddings
# ------------------------------------------------------------

with torch.no_grad():
    test_projected_embeddings = model(torch.tensor(test_embeddings, dtype=torch.float32).to(device)).cpu().numpy()

chosen_superfamilies = [3001879, 3001060, 3002098, 3001843, 3000571, 3000116, 3002019]

# Filter the metadata to include only the chosen superfamilies
filtered_indices = test_metadata[test_metadata['sf'].isin(chosen_superfamilies)].index
filtered_test_embeddings = test_projected_embeddings[filtered_indices]
filtered_test_metadata = test_metadata.loc[filtered_indices]

print(filtered_test_metadata.shape)
print(filtered_test_embeddings.shape)

sf_counts = filtered_test_metadata['sf'].value_counts()
print(sf_counts)

# Perform linear projection of the embeddings to R^2 using PCA
pca = PCA(n_components=2)
projected_embeddings_2d = pca.fit_transform(filtered_test_embeddings)

# Create a scatter plot of the projected embeddings
plt.figure(figsize=(10, 8))
plt.scatter(projected_embeddings_2d[:, 0], projected_embeddings_2d[:, 1], alpha=0.7)

# Create a color map for unique superfamily labels
unique_superfamilies = filtered_test_metadata['sf'].unique()
colors = plt.get_cmap('tab10')(np.arange(len(unique_superfamilies)))

# Create a legend for the superfamily labels
for i, sf in enumerate(unique_superfamilies):
    plt.scatter([], [], color=colors[i], label=sf)  # Empty scatter for legend

# Plot points with unique colors for each superfamily label
for i, txt in enumerate(filtered_test_metadata['sf']):
    plt.scatter(projected_embeddings_2d[i, 0], projected_embeddings_2d[i, 1], color=colors[np.where(unique_superfamilies == txt)[0][0]], alpha=0.7)

plt.legend(title='sf')  # Add legend with title 'sf'

plt.title(r'Triplet Loss w/ Random Triplets: PCA Projection to $\mathbb{R}^2$')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid()
plt.savefig('triplets-random-proj-space.png')
plt.close()


