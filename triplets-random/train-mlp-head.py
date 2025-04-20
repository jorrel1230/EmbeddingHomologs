import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------------------------------------
# Loading Data
# ------------------------------------------------------------

train_embeddings = np.load("/scratch/gpfs/jr8867/main/db/family-split-train-test/train/train_embeddings.npy")
train_metadata = pd.read_csv("/scratch/gpfs/jr8867/main/db/family-split-train-test/train/train_metadata.csv")

test_embeddings = np.load("/scratch/gpfs/jr8867/main/db/family-split-train-test/test/test_embeddings.npy")
test_metadata = pd.read_csv("/scratch/gpfs/jr8867/main/db/family-split-train-test/test/test_metadata.csv")

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
def get_triplets(embeddings, superfamilies, families, num_triplets=10000):
    triplets = np.zeros((num_triplets, 3), dtype=np.int64)
    label_dict = {}
    
    for i, label in enumerate(superfamilies):
        if label not in label_dict:
            label_dict[label] = []
        label_dict[label].append(i)
    
    for i in tqdm(range(num_triplets), desc="Getting triplets", unit="triplet"):
        anchor_idx = np.random.randint(0, len(superfamilies))
        anchor_label = superfamilies[anchor_idx]
        
        positive_idx = np.random.choice(label_dict[anchor_label])
        
        negative_label = np.random.choice([l for l in label_dict.keys() if l != anchor_label])
        negative_idx = np.random.choice(label_dict[negative_label])
        
        triplets[i] = [anchor_idx, positive_idx, negative_idx]
    
    return triplets


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
def train_projection_head(train_embeddings, train_superfamilies, train_families, 
                         test_embeddings, test_superfamilies, test_families, 
                         output_dir, model_name, epochs=10, lr=0.001, triplet_margin=0.2, 
                         train_triplet_count=200000, test_triplet_count=10000, batch_size=32,
                         initial_model_path=None):
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
        train_triplets = get_triplets(train_embeddings, train_superfamilies, train_families, num_triplets=train_triplet_count)
        
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
        test_triplets = get_triplets(test_embeddings, test_superfamilies, test_families, num_triplets=test_triplet_count)

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
    
    output_dir = "/scratch/gpfs/jr8867/main/db/models/triplets-random"

    # Get superfamilies and families from metadata
    train_superfamilies = train_metadata['sf'].values
    train_families = train_metadata['fa'].values
    test_superfamilies = test_metadata['sf'].values
    test_families = test_metadata['fa'].values
    
    # Train the model from scratch
    projection_model, best_loss = train_projection_head(
        train_embeddings, train_superfamilies, train_families,
        test_embeddings, test_superfamilies, test_families,
        output_dir=output_dir, model_name="triplets-random-model-large",
        epochs=50, lr=0.001, triplet_margin=0.2, 
        train_triplet_count=200000, test_triplet_count=10000, 
        batch_size=8192
    )

    # Save the model
    model_path = os.path.join(output_dir, "triplets-random-model-large-end.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print(f"Training completed. Best loss achieved: {best_loss:.4e}")
