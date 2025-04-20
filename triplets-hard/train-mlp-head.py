import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import faiss
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys
import os
import concurrent.futures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# Loading Data
# ------------------------------------------------------------

train_embeddings = np.load("/scratch/gpfs/jr8867/main/db/family-split-train-test/train/train_embeddings.npy")
train_metadata = pd.read_csv("/scratch/gpfs/jr8867/main/db/family-split-train-test/train/train_metadata.csv")

test_embeddings = np.load("/scratch/gpfs/jr8867/main/db/family-split-train-test/test/test_embeddings.npy")
test_metadata = pd.read_csv("/scratch/gpfs/jr8867/main/db/family-split-train-test/test/test_metadata.csv")

train_index = faiss.read_index("/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_train.index")
test_index = faiss.read_index("/scratch/gpfs/jr8867/main/db/indices/baseline/baseline_test.index")

# Dataset class
class ProteinDataset(Dataset):
    def __init__(self, embeddings, superfamilies, families):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.superfamilies = torch.tensor(superfamilies, dtype=torch.long)
        self.families = torch.tensor(families, dtype=torch.long)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.superfamilies[idx], self.families[idx]

# Triplet Sampling Function
def get_triplets(embeddings, index, superfamilies, families, num_random_triplets=150000, num_hard_triplets=50000, k=500):
    total_triplets = num_random_triplets + num_hard_triplets
    triplets = np.zeros((total_triplets, 3), dtype=np.int64)
    label_dict = {}
    
    for i, label in enumerate(superfamilies):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(i)
    
    # Generate random triplets
    for i in tqdm(range(num_random_triplets), desc="Getting random triplets", unit="triplet"):
        anchor_idx = np.random.randint(0, len(superfamilies))
        anchor_label = superfamilies[anchor_idx]
        
        # Ensure positive is different from anchor if possible
        possible_positives = [p for p in label_dict[anchor_label] if p != anchor_idx]
        if not possible_positives: # Handle case where label has only one member
             positive_idx = anchor_idx # Use anchor itself if no other positive available
        else:
             positive_idx = np.random.choice(possible_positives)

        negative_label = np.random.choice([l for l in label_dict.keys() if l != anchor_label])
        negative_idx = np.random.choice(label_dict[negative_label])
        
        triplets[i] = [anchor_idx, positive_idx, negative_idx]
    
    # Generate hard triplets
    hard_triplet_count = 0
    pbar_hard = tqdm(total=num_hard_triplets, desc="Getting hard triplets", unit="triplet")
    
    while hard_triplet_count < num_hard_triplets:
        anchor_idx = np.random.randint(0, len(superfamilies))
        anchor_superfamily = superfamilies[anchor_idx]
        anchor_family = families[anchor_idx]
        
        # Ensure positive is different from anchor if possible
        possible_positives = [p for p in label_dict[anchor_superfamily] if p != anchor_idx]
        if not possible_positives:
             positive_idx = anchor_idx # Use anchor itself if no other positive available
        else:
             positive_idx = np.random.choice(possible_positives)
    
        # Select negative from different superfamily, by searching for the nearest neighbor in the baseline index
        # This has the intended effect of encouraging the model to learn with harder training examples.
        positive_embedding = embeddings[positive_idx]
        positive_embedding = positive_embedding.reshape(1, -1) # have to reshape to be compatible with faiss
        D, I = index.search(positive_embedding, k+1)
        neighbor_indices = I[0][1:]  # Skip the first result (self)
        neighbor_distances = D[0][1:]
        
        # Get superfamilies and families for each neighbor
        neighbor_superfamilies = superfamilies[neighbor_indices]
        neighbor_families = families[neighbor_indices]
        
        # Select all neighbors that share neither superfamily nor family with the anchor
        negative_indices_mask = (neighbor_superfamilies != anchor_superfamily) # | (neighbor_families != anchor_family) # Original logic commented out - focus on superfamily difference first
        
        # Filter valid negative indices from the neighbors
        potential_negative_indices = neighbor_indices[negative_indices_mask]

        if len(potential_negative_indices) == 0:
            continue # No suitable hard negative found for this anchor/positive, try again

        # Add found hard triplets
        for neg_idx in potential_negative_indices:
            if hard_triplet_count < num_hard_triplets:
                current_triplet_idx = num_random_triplets + hard_triplet_count
                triplets[current_triplet_idx] = [anchor_idx, positive_idx, neg_idx]
                hard_triplet_count += 1
                pbar_hard.update(1)
            else:
                break # Stop adding if we have enough hard triplets
        
        if hard_triplet_count >= num_hard_triplets:
            break # Exit the while loop once enough hard triplets are found

    pbar_hard.close()
    
    # If not enough hard triplets were found, we might have empty rows at the end.
    # Optionally, trim the array or handle it downstream. For now, return as is.
    print(f"Generated {num_random_triplets} random and {hard_triplet_count} hard triplets.")
    return triplets[:num_random_triplets + hard_triplet_count] # Return potentially trimmed array


# Projection Head Model
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

# Training Loop
def train_projection_head(train_embeddings, train_index, train_superfamilies, train_families, 
                         test_embeddings, test_index, test_superfamilies, test_families, 
                         output_dir, model_name, epochs=10, lr=0.001, triplet_margin=0.2, 
                         batch_size=32, initial_model_path=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model = ProjectionHead(input_dim=train_embeddings.shape[1], output_dim=256).to(device)
    
    # Load initial model if provided
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading initial model from {initial_model_path}")
        model.load_state_dict(torch.load(initial_model_path))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.TripletMarginLoss(margin=triplet_margin)

    train_dataset = ProteinDataset(train_embeddings, train_superfamilies, train_families)
    test_dataset = ProteinDataset(test_embeddings, test_superfamilies, test_families)

    best_loss = float('inf')

    for epoch in range(epochs):

        # Get triplets
        train_triplets = get_triplets(train_embeddings, train_index, train_superfamilies, train_families)
        
        model.train()
        total_loss = 0

        for i in tqdm(range(0, len(train_triplets), batch_size), desc=f"Epoch {epoch+1}", unit="batch"):
            batch_triplets = train_triplets[i:i + batch_size]
            anchors = torch.stack([train_dataset[anchor_idx][0] for anchor_idx, _, _ in batch_triplets]).to(device)
            positives = torch.stack([train_dataset[pos_idx][0] for _, pos_idx, _ in batch_triplets]).to(device)
            negatives = torch.stack([train_dataset[neg_idx][0] for _, _, neg_idx in batch_triplets]).to(device)

            anchor_out = model(anchors)
            positive_out = model(positives)
            negative_out = model(negatives)

            loss = criterion(anchor_out, positive_out, negative_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_triplets)

        # Save model if it has the best loss so far
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model_path = os.path.join(output_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with loss: {best_loss:.4e}")

        # Evaluation on test set
        test_triplets = get_triplets(test_embeddings, test_index, test_superfamilies, test_families, num_random_triplets=20000, num_hard_triplets=0, k=0)

        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for anchor_idx, pos_idx, neg_idx in test_triplets:
                anchor = test_dataset[anchor_idx][0].to(device)
                positive = test_dataset[pos_idx][0].to(device)
                negative = test_dataset[neg_idx][0].to(device)

                anchor_out = model(anchor)
                positive_out = model(positive)
                negative_out = model(negative)

                t_loss = criterion(anchor_out, positive_out, negative_out)
                total_test_loss += t_loss.item()
        avg_test_loss = total_test_loss / len(test_triplets)

        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4e} | Test Loss: {avg_test_loss:.4e}")

    return model, best_loss

# Main execution
if __name__ == "__main__":
    print("Starting training process...")
    
    output_dir = "/scratch/gpfs/jr8867/main/db/models/triplets-hard"

    # Get superfamilies and families from metadata
    train_superfamilies = train_metadata['sf'].values
    train_families = train_metadata['fa'].values
    test_superfamilies = test_metadata['sf'].values
    test_families = test_metadata['fa'].values
    
    # Train the model from scratch
    projection_model, best_loss = train_projection_head(
        train_embeddings, train_index, train_superfamilies, train_families,
        test_embeddings, test_index, test_superfamilies, test_families,
        output_dir=output_dir, model_name="triplets-hard-model-large",
        epochs=50, lr=0.001, triplet_margin=0.2, 
        batch_size=8192
    )

    # Save the model
    model_path = os.path.join(output_dir, "triplets-hard-model-large-end.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print(f"Training completed. Best loss achieved: {best_loss:.4e}")
