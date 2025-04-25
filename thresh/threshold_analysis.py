import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import seaborn as sns

vanilla_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/baseline/evals/baseline_scores.npy')
vanilla_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/baseline/evals/baseline_labels.npy')

npair_train_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_train_scores.npy')
npair_train_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_train_labels.npy')
npair_test_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_test_scores.npy')
npair_test_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_test_labels.npy')

triplets_random_train_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_train_scores.npy')
triplets_random_train_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_train_labels.npy')
triplets_random_test_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_test_scores.npy')
triplets_random_test_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_test_labels.npy')

triplets_hard_train_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-hard/evals/triplets-hard_train_scores.npy')
triplets_hard_train_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-hard/evals/triplets-hard_train_labels.npy')
triplets_hard_test_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-hard/evals/triplets-hard_test_scores.npy')
triplets_hard_test_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-hard/evals/triplets-hard_test_labels.npy')

# ------------------------------------
# Remove indices where scores are 0
# ------------------------------------

non_zero_indices = vanilla_scores != 0
vanilla_scores = vanilla_scores[non_zero_indices]
vanilla_labels = vanilla_labels[non_zero_indices]

non_zero_indices = npair_train_scores != 0
npair_train_scores = npair_train_scores[non_zero_indices]
npair_train_labels = npair_train_labels[non_zero_indices]

non_zero_indices = npair_test_scores != 0
npair_test_scores = npair_test_scores[non_zero_indices]
npair_test_labels = npair_test_labels[non_zero_indices]

non_zero_indices = triplets_random_train_scores != 0
triplets_random_train_scores = triplets_random_train_scores[non_zero_indices]
triplets_random_train_labels = triplets_random_train_labels[non_zero_indices]

non_zero_indices = triplets_random_test_scores != 0
triplets_random_test_scores = triplets_random_test_scores[non_zero_indices]
triplets_random_test_labels = triplets_random_test_labels[non_zero_indices]

non_zero_indices = triplets_hard_train_scores != 0
triplets_hard_train_scores = triplets_hard_train_scores[non_zero_indices]
triplets_hard_train_labels = triplets_hard_train_labels[non_zero_indices]

non_zero_indices = triplets_hard_test_scores != 0
triplets_hard_test_scores = triplets_hard_test_scores[non_zero_indices]
triplets_hard_test_labels = triplets_hard_test_labels[non_zero_indices]

print("Removed indices where scores are 0.")

# ------------------------------------
# Compute Optimal Threshold For Similarity Scores
# ------------------------------------
def compute_confusion_matrix(labels, scores, threshold):
    """
    Computes the confusion matrix for a given threshold.

    Args:
        labels (np.array): Ground truth labels (0 or 1).
        scores (np.array): Similarity scores.
        threshold (float): Threshold for classifying scores.

    Returns:
        np.array: Confusion matrix (2x2).
    """
    predictions = scores < threshold  # Changed >= to < because scores are distances
    tp = np.sum((labels == 1) & (predictions == 1))
    fp = np.sum((labels == 0) & (predictions == 1))
    tn = np.sum((labels == 0) & (predictions == 0))
    fn = np.sum((labels == 1) & (predictions == 0))
    return np.array([[tp, fp], [fn, tn]])

def plot_confusion_matrix(confusion_matrix, title, filename):
    """Plots the confusion matrix as a heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Positive', 'Negative'], 
                yticklabels=['Positive', 'Negative'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    plt.savefig(filename)
    plt.close()


def analyze_threshold_train(labels, scores, num_thresholds=100):
    """
    Analyzes different thresholds to find the optimal one based on F1-score on the training set.

    Args:
        labels (np.array): Ground truth labels.
        scores (np.array): Similarity scores.
        num_thresholds (int): Number of thresholds to test.

    Returns:
        tuple: Optimal threshold.
    """
    thresholds = np.linspace(np.min(scores), np.max(scores), num_thresholds)
    best_threshold = None
    best_f1 = 0

    for threshold in thresholds:
        confusion_matrix = compute_confusion_matrix(labels, scores, threshold)
        tp = confusion_matrix[0, 0]
        fp = confusion_matrix[0, 1]
        fn = confusion_matrix[1, 0]

        if tp + fp == 0 or tp + fn == 0:
            f1 = 0  # Avoid division by zero
        else:
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold


def evaluate_threshold(labels, scores, threshold):
    """
    Evaluates a given threshold and returns the confusion matrix.

    Args:
        labels (np.array): Ground truth labels.
        scores (np.array): Similarity scores.
        threshold (float): Threshold to evaluate.

    Returns:
        np.array: Confusion matrix.
    """
    confusion_matrix = compute_confusion_matrix(labels, scores, threshold)
    return confusion_matrix

# Analyze N-Pair Random
print("Analyzing N-Pair Random...")
npair_best_threshold = analyze_threshold_train(npair_train_labels, npair_train_scores)
npair_confusion_matrix = evaluate_threshold(npair_test_labels, npair_test_scores, npair_best_threshold)
print(f"N-Pair Random Best Threshold: {npair_best_threshold}")
print(f"N-Pair Random Confusion Matrix:\n{npair_confusion_matrix}")
plot_confusion_matrix(npair_confusion_matrix, 'N-Pair Random Confusion Matrix', 'npair_confusion_matrix.png')

# Analyze Triplets Random
print("Analyzing Triplets Random...")
triplets_random_best_threshold = analyze_threshold_train(triplets_random_train_labels, triplets_random_train_scores)
triplets_random_confusion_matrix = evaluate_threshold(triplets_random_test_labels, triplets_random_test_scores, triplets_random_best_threshold)
print(f"Triplets Random Best Threshold: {triplets_random_best_threshold}")
print(f"Triplets Random Confusion Matrix:\n{triplets_random_confusion_matrix}")
plot_confusion_matrix(triplets_random_confusion_matrix, 'Triplets Random Confusion Matrix', 'triplets_random_confusion_matrix.png')

# Analyze Triplets Hard
print("Analyzing Triplets Hard...")
triplets_hard_best_threshold = analyze_threshold_train(triplets_hard_train_labels, triplets_hard_train_scores)
triplets_hard_confusion_matrix = evaluate_threshold(triplets_hard_test_labels, triplets_hard_test_scores, triplets_hard_best_threshold)
print(f"Triplets Hard Best Threshold: {triplets_hard_best_threshold}")
print(f"Triplets Hard Confusion Matrix:\n{triplets_hard_confusion_matrix}")
plot_confusion_matrix(triplets_hard_confusion_matrix, 'Triplets Hard Confusion Matrix', 'triplets_hard_confusion_matrix.png')

# Analyze Vanilla
print("Analyzing Vanilla...")
vanilla_best_threshold = analyze_threshold_train(vanilla_labels, vanilla_scores)
vanilla_confusion_matrix = evaluate_threshold(vanilla_labels, vanilla_scores, vanilla_best_threshold)
print(f"Vanilla Best Threshold: {vanilla_best_threshold}")
print(f"Vanilla Confusion Matrix:\n{vanilla_confusion_matrix}")
plot_confusion_matrix(vanilla_confusion_matrix, 'Vanilla Confusion Matrix', 'vanilla_confusion_matrix.png')

print("Analysis complete. Confusion matrices saved.")
