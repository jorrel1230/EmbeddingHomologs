import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

scores = np.load(f'/scratch/gpfs/jr8867/main/db/indices/baseline/evals/baseline_scores.npy')
labels = np.load(f'/scratch/gpfs/jr8867/main/db/indices/baseline/evals/baseline_labels.npy')

print("Loaded scores and labels.")

# Remove indices where scores are 0
non_zero_indices = scores != 0
scores = scores[non_zero_indices]
labels = labels[non_zero_indices]

print(f"Removed {len(non_zero_indices) - len(scores)} zero-score entries.")


# ----------------------
# Generate ROC curve
# ----------------------

# Compute ROC curve and ROC area
fpr, tpr, _ = roc_curve(labels, -scores)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Baseline - ROC Curve', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.savefig('roc_curve.png')
plt.close()

print("Generated ROC curve.")

# ----------------------
# Generate PR curve
# ----------------------

# Compute PR curve and PR area
precision, recall, _ = precision_recall_curve(labels, -scores)
pr_auc = auc(recall, precision)
pr_base = 1/2816

# Plot PR curve
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Baseline PR curve (AUC = {pr_auc:.2f})')
plt.plot([0, 1], [pr_base, pr_base], color='gray', linestyle='--', label='Random Classifier')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Baseline - Precision-Recall Curve', fontsize=14)
plt.legend(loc="upper right", fontsize=12)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.grid(True)
plt.savefig('pr_curve.png')
plt.close()

print(f"Generated PR curve with AUC = {pr_auc:.4f}")

# ------------------------------------------------------------------------------------------------
# Generate Histogram of scores by label
# ------------------------------------------------------------------------------------------------  

# Generate histogram of scores by label
plt.figure(figsize=(6, 6))
h1 = plt.hist(scores[labels == 1], bins=100, color='blue', alpha=0.5, label='Positive Scores', density=True)
h2 = plt.hist(scores[labels == 0], bins=100, color='orange', alpha=0.5, label='Negative Scores', density=True)
plt.xlabel('Scores', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Baseline - Distribution of Scores by Label', fontsize=14)
plt.legend(fontsize=12)

# Trace lines through the top of the histogram
max_height = max(max(h1[0]), max(h2[0]))
plt.plot(h1[1][:-1], h1[0], color='blue', linewidth=2)
plt.plot(h2[1][:-1], h2[0], color='orange', linewidth=2)

plt.savefig('score_histogram.png')
plt.close()

print("Generated score histogram.")