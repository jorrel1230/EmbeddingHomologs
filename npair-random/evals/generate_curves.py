import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

baseline_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/baseline/evals/baseline_scores.npy')
baseline_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/baseline/evals/baseline_labels.npy')

npair_train_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_train_scores.npy')
npair_train_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_train_labels.npy')
npair_test_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_test_scores.npy')
npair_test_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_test_labels.npy')

temp = npair_train_scores
npair_train_scores = npair_test_scores
npair_test_scores = temp

temp = npair_train_labels
npair_train_labels = npair_test_labels
npair_test_labels = temp

print("Loaded scores and labels.")

# ------------------------------------
# Remove indices where scores are 0
# ------------------------------------

non_zero_indices = baseline_scores != 0
baseline_scores = baseline_scores[non_zero_indices]
baseline_labels = baseline_labels[non_zero_indices]

non_zero_indices = npair_train_scores != 0
npair_train_scores = npair_train_scores[non_zero_indices]
npair_train_labels = npair_train_labels[non_zero_indices]

non_zero_indices = npair_test_scores != 0
npair_test_scores = npair_test_scores[non_zero_indices]
npair_test_labels = npair_test_labels[non_zero_indices]

# ----------------------
# Generate ROC curve
# ----------------------

# Compute ROC curve and ROC area
train_fpr, train_tpr, _ = roc_curve(npair_train_labels, -npair_train_scores)
train_roc_auc = auc(train_fpr, train_tpr)

test_fpr, test_tpr, _ = roc_curve(npair_test_labels, -npair_test_scores)
test_roc_auc = auc(test_fpr, test_tpr)

baseline_fpr, baseline_tpr, _ = roc_curve(baseline_labels, -baseline_scores)
baseline_roc_auc = auc(baseline_fpr, baseline_tpr)

# Plot Train ROC curve
plt.figure(figsize=(6, 6))
plt.plot(train_fpr, train_tpr, color='blue', lw=2, label=f'Train ROC curve (AUC = {train_roc_auc:.2f})')
plt.plot(baseline_fpr, baseline_tpr, color='red', lw=2, label=f'Baseline ROC curve (AUC = {baseline_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('N-Pair Random (Train) - ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve_train.png')
plt.close()

# Plot Test ROC curve
plt.figure(figsize=(6, 6))
plt.plot(test_fpr, test_tpr, color='green', lw=2, label=f'Test ROC curve (AUC = {test_roc_auc:.2f})')
plt.plot(baseline_fpr, baseline_tpr, color='red', lw=2, label=f'Baseline ROC curve (AUC = {baseline_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('N-Pair Random (Test) - ROC Curve')
plt.legend(loc="lower right")
plt.savefig('roc_curve_test.png')
plt.close()

print("Generated ROC curves.")

# ----------------------
# Generate PR curve
# ----------------------

# Compute PR curve and PR area
train_precision, train_recall, _ = precision_recall_curve(npair_train_labels, -npair_train_scores)
train_pr_auc = auc(train_recall, train_precision)

test_precision, test_recall, _ = precision_recall_curve(npair_test_labels, -npair_test_scores)
test_pr_auc = auc(test_recall, test_precision)

# Plot PR curve
plt.figure(figsize=(6, 6))
plt.plot(train_recall, train_precision, color='blue', lw=2, label=f'Train PR curve (AUC = {train_pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('N-Pair Random (Train) - PR Curve')
plt.legend(loc="upper right")
plt.savefig('pr_curve_train.png')
plt.close()

# Plot Test PR curve
plt.figure(figsize=(6, 6))
plt.plot(test_recall, test_precision, color='green', lw=2, label=f'Test PR curve (AUC = {test_pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('N-Pair Random (Test) - PR Curve')
plt.legend(loc="upper right")
plt.savefig('pr_curve_test.png')
plt.close()

print("Generated PR curves.")

# ------------------------------------------------------------------------------------------------
# Generate Histogram of scores by label
# ------------------------------------------------------------------------------------------------  

# Generate histogram of scores by label
plt.figure(figsize=(6, 6))
h1 = plt.hist(npair_train_scores[npair_train_labels == 1], bins=100, color='blue', alpha=0.5, label='Positive Scores', density=True)
h2 = plt.hist(npair_train_scores[npair_train_labels == 0], bins=100, color='orange', alpha=0.5, label='Negative Scores', density=True)
plt.xlabel('Scores')
plt.ylabel('Density')
plt.title('N-Pair Random (Train) - Distribution of Scores by Label')
plt.legend()

# Trace lines through the top of the histogram
max_height = max(max(h1[0]), max(h2[0]))
plt.plot(h1[1][:-1], h1[0], color='blue', linewidth=2)
plt.plot(h2[1][:-1], h2[0], color='orange', linewidth=2)

plt.savefig('score_histogram_train.png')
plt.close()

# Generate histogram of scores by label
plt.figure(figsize=(6, 6))
h1 = plt.hist(npair_test_scores[npair_test_labels == 1], bins=100, color='blue', alpha=0.5, label='Positive Scores', density=True)
h2 = plt.hist(npair_test_scores[npair_test_labels == 0], bins=100, color='orange', alpha=0.5, label='Negative Scores', density=True)
plt.xlabel('Scores')
plt.ylabel('Density')
plt.title('N-Pair Random (Test) - Distribution of Scores by Label')
plt.legend()

# Trace lines through the top of the histogram
max_height = max(max(h1[0]), max(h2[0]))
plt.plot(h1[1][:-1], h1[0], color='blue', linewidth=2)
plt.plot(h2[1][:-1], h2[0], color='orange', linewidth=2)

plt.savefig('score_histogram_test.png')
plt.close()

print("Generated score histogram.")