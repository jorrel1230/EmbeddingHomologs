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

# ---------------------------------------------------------------------
# Pair Sampling Function for N-pair Loss
# ---------------------------------------------------------------------
def get_pairs(embeddings, superfamilies, families, num_pairs=10000):
    """
    Samples pairs of indices such that both samples in a pair come from the same superfamily.
    
    Args:
        embeddings (np.array): Embeddings array.
        superfamilies (np.array): Superfamily labels.
        families (np.array): Family labels.
        num_pairs (int): Number of pairs to sample.
    
    Returns:
        np.array: Array of shape (num_pairs, 2) where each row is [anchor_idx, positive_idx]
    """
    pairs = []
    label_dict = {}
    
    # Build dictionary mapping label to indices.
    for i, label in enumerate(superfamilies):
        label_dict.setdefault(label, []).append(i)
    
    # Only use labels with at least 2 samples.
    valid_labels = [label for label, indices in label_dict.items() if len(indices) > 1]
    if not valid_labels:
        raise ValueError("No labels with at least two samples found.")
    
    for i in tqdm(range(num_pairs), desc="Getting pairs", unit="pair"):
        # Randomly select a valid label.
        anchor_label = np.random.choice(valid_labels)
        indices = label_dict[anchor_label]
        # Randomly sample two distinct indices from the same label.
        anchor_idx, positive_idx = np.random.choice(indices, size=2, replace=False)
        pairs.append([anchor_idx, positive_idx])
    
    return np.array(pairs)

# ---------------------------------------------------------------------
# N-pair Loss Implementation
# ---------------------------------------------------------------------
class NPairLossDirect(nn.Module):
    """
    Implements the N-pair loss as described in Sohn (2016).

    For a batch of anchors and positives:
        L = 1/N * sum_i { log [ 1 + sum_{j != i} exp( (anchor_i · pos_j) - (anchor_i · pos_i) ) ] }

    Args:
        l2_reg (float): Optional L2 regularization weight.
    """
    def __init__(self, l2_reg=0.0):
        super(NPairLossDirect, self).__init__()
        self.l2_reg = l2_reg

    def forward(self, anchors, positives):
        """
        Args:
            anchors (Tensor): Tensor of shape (batch_size, embed_dim).
            positives (Tensor): Tensor of shape (batch_size, embed_dim).
        
        Returns:
            Tensor: Computed N-pair loss.
        """
        batch_size = anchors.size(0)
        # Compute similarity matrix (dot product between every anchor and positive).
        sim_matrix = torch.matmul(anchors, positives.t())  # (B, B)

        # Correct positive scores are on the diagonal.
        correct_scores = sim_matrix.diag().unsqueeze(1)  # (B, 1)

        # Compute difference between every score and the correct score.
        diff_matrix = sim_matrix - correct_scores

        # Exclude the diagonal elements (correct pairs) from the negatives.
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=anchors.device)
        diff_matrix_no_diag = diff_matrix[mask].view(batch_size, batch_size - 1)

        # Compute loss for each anchor.
        loss_per_anchor = torch.log1p(torch.exp(diff_matrix_no_diag).sum(dim=1))
        loss = loss_per_anchor.mean()

        # Optionally add L2 regularization on embeddings.
        if self.l2_reg > 0:
            l2_loss = (anchors.norm(dim=1).mean() + positives.norm(dim=1).mean()) / 2.0
            loss += self.l2_reg * l2_loss

        return loss


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
                         output_dir, model_name, epochs=10, lr=0.001,
                         train_pair_count=200000, test_pair_count=10000, 
                         batch_size=32, initial_model_path=None):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    model = ProjectionHead(input_dim=train_embeddings.shape[1], output_dim=256).to(device)
    
    # Load initial model if provided
    if initial_model_path and os.path.exists(initial_model_path):
        print(f"Loading initial model from {initial_model_path}")
        model.load_state_dict(torch.load(initial_model_path))
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = NPairLossDirect(l2_reg=0.002)

    train_dataset = ProteinDataset(train_embeddings, train_superfamilies, train_families)
    test_dataset = ProteinDataset(test_embeddings, test_superfamilies, test_families)

    best_loss = float('inf')

    for epoch in range(epochs):

        # Sample training pairs.
        train_pairs = get_pairs(train_embeddings, train_superfamilies, train_families, num_pairs=train_pair_count)
        model.train()
        total_loss = 0.0

        # Process training pairs in mini-batches.
        for i in tqdm(range(0, len(train_pairs), batch_size), desc=f"Epoch {epoch+1}", unit="batch"):
            batch_pairs = train_pairs[i:i + batch_size]
            anchors = torch.stack([train_dataset[pair[0]][0] for pair in batch_pairs]).to(device)
            positives = torch.stack([train_dataset[pair[1]][0] for pair in batch_pairs]).to(device)
            
            anchor_out = model(anchors)
            positive_out = model(positives)
            
            loss = criterion(anchor_out, positive_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_train_loss = total_loss / (len(train_pairs) / batch_size)
        

        # Save model if it has the best loss so far
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            model_path = os.path.join(output_dir, f"{model_name}.pth")
            torch.save(model.state_dict(), model_path)
            print(f"New best model saved to {model_path} with loss: {best_loss:.4e}")

        # Evaluation on test set.
        test_pairs = get_pairs(test_embeddings, test_superfamilies, test_families, num_pairs=test_pair_count)
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for pair in test_pairs:
                anchor = test_dataset[pair[0]][0].unsqueeze(0).to(device)
                positive = test_dataset[pair[1]][0].unsqueeze(0).to(device)
                anchor_out = model(anchor)
                positive_out = model(positive)
                t_loss = criterion(anchor_out, positive_out)
                total_test_loss += t_loss.item()
        avg_test_loss = total_test_loss / len(test_pairs)
        
        print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4e} | Test Loss: {avg_test_loss:.4e}")

    return model, best_loss

# Main execution
if __name__ == "__main__":
    print("Starting training process...")
    
    output_dir = "/scratch/gpfs/jr8867/main/db/models/npair-random"

    # Get superfamilies and families from metadata
    train_superfamilies = train_metadata['sf'].values
    train_families = train_metadata['fa'].values
    test_superfamilies = test_metadata['sf'].values
    test_families = test_metadata['fa'].values

    # Train the model from scratch
    projection_model, best_loss = train_projection_head(
        train_embeddings, train_superfamilies, train_families,
        test_embeddings, test_superfamilies, test_families,
        output_dir=output_dir, model_name="npair-random-model-large",
        epochs=50, lr=0.001, batch_size=8192
    )

    # Save the model
    model_path = os.path.join(output_dir, "npair-random-model-large-end.pth")
    torch.save(projection_model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    print(f"Training completed. Best loss achieved: {best_loss:.4e}")
