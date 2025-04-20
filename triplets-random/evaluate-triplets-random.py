import torch
import torch.nn as nn
import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
import gc
import os
import pandas as pd


print("Loading Caches Embeddings and Metadata")

# ------------------------------------------------------------
# Loading Train Embeddings and Metadata
# ------------------------------------------------------------

triplets_train_embeddings = np.load("/scratch/gpfs/jr8867/main/db/indices/triplets-random/triplets-random_train_embeddings.npy")
triplets_train_metadata = pd.read_csv("/scratch/gpfs/jr8867/main/db/indices/triplets-random/triplets-random_train_metadata.csv")

# Load the FAISS index
triplets_train_index = faiss.read_index("/scratch/gpfs/jr8867/main/db/indices/triplets-random/triplets-random_train.index")

# Extract relevant columns for evaluation
triplets_train_superfamilies = triplets_train_metadata['sf'].values
triplets_train_families = triplets_train_metadata['fa'].values
triplets_train_indices = triplets_train_metadata.index.values

# ------------------------------------------------------------
# Loading Test Embeddings and Metadata
# ------------------------------------------------------------

triplets_test_embeddings = np.load("/scratch/gpfs/jr8867/main/db/indices/triplets-random/triplets-random_test_embeddings.npy")
triplets_test_metadata = pd.read_csv("/scratch/gpfs/jr8867/main/db/indices/triplets-random/triplets-random_test_metadata.csv")

# Load the FAISS index
triplets_test_index = faiss.read_index("/scratch/gpfs/jr8867/main/db/indices/triplets-random/triplets-random_test.index")

# Extract relevant columns for evaluation
triplets_test_superfamilies = triplets_test_metadata['sf'].values
triplets_test_families = triplets_test_metadata['fa'].values
triplets_test_indices = triplets_test_metadata.index.values

def search_similar_proteins(query_embedding, index, superfamilies, families, k=50):
    """
    Search for similar proteins and return detailed results.
    
    Args:
    - query_embedding: Embedding of the query protein
    - index: FAISS index to search
    - superfamilies: Array of superfamily labels
    - families: Array of family labels
    - k: Number of nearest neighbors to retrieve
    
    Returns:
    - results: Dictionary with similarity details
    """
    # Reshape query embedding for FAISS search
    query_embedding = query_embedding.reshape(1, -1)
    
    # Search in FAISS index
    D, I = index.search(query_embedding, k+1)
    
    # Get the indices and distances for the first query
    neighbor_indices = I[0][1:]  # Skip the first result (self)
    neighbor_distances = D[0][1:]
    
    # Get superfamilies and families for each neighbor
    neighbor_superfamilies = superfamilies[neighbor_indices]
    neighbor_families = families[neighbor_indices]
    
    # Determine homology status
    query_superfamily = superfamilies[I[0][0]]
    query_family = families[I[0][0]]
    
    results = {
        'distances': neighbor_distances,
        'superfamilies': neighbor_superfamilies,
        'families': neighbor_families,
        'query_superfamily': query_superfamily,
        'query_family': query_family
    }
    
    return results

def evaluate_similarity_search(index, embeddings, superfamilies, families, k=50):
    """
    Collect similarity scores and labels for ROC curve generation.
    
    Args:
    - index: FAISS index
    - embeddings: Full embedding matrix
    - superfamilies: Array of superfamily labels
    - families: Array of family labels
    - k: Number of nearest neighbors to retrieve
    
    Returns:
    - all_scores: Numpy array of similarity scores
    - all_labels: Numpy array of ground truth labels
    """
    all_scores = []
    all_labels = []

    for i in tqdm(range(len(embeddings)), desc="Collecting similarity scores"):
        # Get the query embedding
        query_embedding = embeddings[i]
        
        # Search for similar proteins
        results = search_similar_proteins(query_embedding, index, superfamilies, families, k)
        
        # Process each result
        for dist, neighbor_superfamily, neighbor_family in zip(
            results['distances'], 
            results['superfamilies'], 
            results['families']
        ):
            # Determine if this is a homolog (same family or superfamily)
            is_homolog = 0
            if (neighbor_family == results['query_family'] or 
                neighbor_superfamily == results['query_superfamily']):
                is_homolog = 1
            
            # Add to our results
            all_scores.append(dist)
            all_labels.append(is_homolog)

    return np.array(all_scores), np.array(all_labels)

def generate_roc_curve(scores, labels, output_path=None):
    """
    Generate ROC curve from similarity scores and labels.
    
    Args:
    - scores: Numpy array of similarity scores
    - labels: Numpy array of ground truth labels
    - output_path: Optional path to save ROC curve plot
    
    Returns:
    - fpr: False Positive Rate
    - tpr: True Positive Rate
    - auc_score: Area Under Curve
    """
    # Invert scores since smaller distances indicate higher similarity
    fpr, tpr, thresholds = roc_curve(labels, -scores)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve if output path is provided
    if output_path:
        plt.figure(figsize=(8,6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Baseline Embeddings')
        plt.legend(loc="lower right")
        plt.savefig(output_path)
        plt.close()
    
    return fpr, tpr, roc_auc

# Example usage in main script
if __name__ == "__main__":

    # ------------------------------------------------------------
    # Train ROC Curve
    # ------------------------------------------------------------

    # Collect similarity scores and labels
    all_scores, all_labels = evaluate_similarity_search(
        triplets_train_index, 
        triplets_train_embeddings, 
        triplets_train_superfamilies, 
        triplets_train_families, 
        k=50
    )
    
    # Generate ROC curve
    fpr, tpr, roc_auc = generate_roc_curve(
        all_scores, 
        all_labels, 
        output_path='/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_train_roc_curve.png'
    )
    
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Optionally, save scores and labels for future analysis
    np.save('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_train_scores.npy', all_scores)
    np.save('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_train_labels.npy', all_labels)

    # ------------------------------------------------------------
    # Test ROC Curve
    # ------------------------------------------------------------

    # Collect similarity scores and labels
    all_scores, all_labels = evaluate_similarity_search(
        triplets_test_index, 
        triplets_test_embeddings, 
        triplets_test_superfamilies, 
        triplets_test_families, 
        k=50
    )
    
    # Generate ROC curve
    fpr, tpr, roc_auc = generate_roc_curve(
        all_scores, 
        all_labels, 
        output_path='/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_test_roc_curve.png'
    )
    
    print(f"ROC AUC: {roc_auc:.4f}")
    
    # Optionally, save scores and labels for future analysis
    np.save('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_test_scores.npy', all_scores)
    np.save('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_test_labels.npy', all_labels)

