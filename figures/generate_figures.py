import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve

vanilla_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/baseline/evals/baseline_scores.npy')
vanilla_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/baseline/evals/baseline_labels.npy')

npair_test_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_test_scores.npy')
npair_test_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/npair-random/evals/npair-random_test_labels.npy')

triplets_random_test_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_test_scores.npy')
triplets_random_test_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-random/evals/triplets-random_test_labels.npy')

triplets_hard_test_scores = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-hard/evals/triplets-hard_test_scores.npy')
triplets_hard_test_labels = np.load('/scratch/gpfs/jr8867/main/db/indices/triplets-hard/evals/triplets-hard_test_labels.npy')

print("Loaded scores and labels.")

# ------------------------------------
# Remove indices where scores are 0
# ------------------------------------

non_zero_indices = vanilla_scores != 0
vanilla_scores = vanilla_scores[non_zero_indices]
vanilla_labels = vanilla_labels[non_zero_indices]

non_zero_indices = npair_test_scores != 0
npair_test_scores = npair_test_scores[non_zero_indices]
npair_test_labels = npair_test_labels[non_zero_indices]

non_zero_indices = triplets_random_test_scores != 0
triplets_random_test_scores = triplets_random_test_scores[non_zero_indices]
triplets_random_test_labels = triplets_random_test_labels[non_zero_indices]

non_zero_indices = triplets_hard_test_scores != 0
triplets_hard_test_scores = triplets_hard_test_scores[non_zero_indices]
triplets_hard_test_labels = triplets_hard_test_labels[non_zero_indices]

print("Removed indices where scores are 0.")

# ----------------------
# Generate ROC curve
# ----------------------

# Compute ROC curve and ROC area
test_fpr, test_tpr, _ = roc_curve(npair_test_labels, -npair_test_scores)
test_roc_auc = auc(test_fpr, test_tpr)

vanilla_fpr, vanilla_tpr, _ = roc_curve(vanilla_labels, -vanilla_scores)
vanilla_roc_auc = auc(vanilla_fpr, vanilla_tpr)

# Plot Test ROC curve
plt.figure()
plt.plot(test_fpr, test_tpr, color='green', lw=2, label=f'Test ROC curve (AUC = {test_roc_auc:.2f})')
plt.plot(vanilla_fpr, vanilla_tpr, color='red', lw=2, label=f'Vanilla ROC curve (AUC = {vanilla_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('N-Pair Random (Test) - ROC Curve')
plt.legend(loc="lower right")
plt.savefig('npair_random_roc_curve.png')
plt.close()

# Compute ROC curve and ROC area
test_fpr, test_tpr, _ = roc_curve(triplets_random_test_labels, -triplets_random_test_scores)
test_roc_auc = auc(test_fpr, test_tpr)

vanilla_fpr, vanilla_tpr, _ = roc_curve(vanilla_labels, -vanilla_scores)
vanilla_roc_auc = auc(vanilla_fpr, vanilla_tpr)

# Plot Test ROC curve
plt.figure()
plt.plot(test_fpr, test_tpr, color='green', lw=2, label=f'Test ROC curve (AUC = {test_roc_auc:.2f})')
plt.plot(vanilla_fpr, vanilla_tpr, color='red', lw=2, label=f'Vanilla ROC curve (AUC = {vanilla_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Triplets Random (Test) - ROC Curve')
plt.legend(loc="lower right")
plt.savefig('triplets_random_roc_curve.png')
plt.close()

# Compute ROC curve and ROC area
test_fpr, test_tpr, _ = roc_curve(triplets_hard_test_labels, -triplets_hard_test_scores)
test_roc_auc = auc(test_fpr, test_tpr)

vanilla_fpr, vanilla_tpr, _ = roc_curve(vanilla_labels, -vanilla_scores)
vanilla_roc_auc = auc(vanilla_fpr, vanilla_tpr)

# Plot Test ROC curve
plt.figure()
plt.plot(test_fpr, test_tpr, color='green', lw=2, label=f'Test ROC curve (AUC = {test_roc_auc:.2f})')
plt.plot(vanilla_fpr, vanilla_tpr, color='red', lw=2, label=f'Vanilla ROC curve (AUC = {vanilla_roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Triplets Hard (Test) - ROC Curve')
plt.legend(loc="lower right")
plt.savefig('triplets_hard_roc_curve.png')
plt.close()



print("Generated ROC curves.")

# ----------------------
# Generate PR curve
# ----------------------


# Calculate the positive rate for the baseline
positive_rate = 1 / 2816

# Compute PR curve and PR area
test_precision, test_recall, _ = precision_recall_curve(npair_test_labels, -npair_test_scores)
test_pr_auc = auc(test_recall, test_precision)

vanilla_precision, vanilla_recall, _ = precision_recall_curve(vanilla_labels, -vanilla_scores)
vanilla_pr_auc = auc(vanilla_recall, vanilla_precision)

# Plot PR curve
plt.figure()
plt.plot(test_recall, test_precision, color='green', lw=2, label=f'Test PR curve')
plt.plot(vanilla_recall, vanilla_precision, color='red', lw=2, label=f'Vanilla PR curve')
plt.plot([0, 1], [positive_rate, positive_rate], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('N-Pair Random (Test) - PR Curve')
plt.legend(loc="upper right")
plt.savefig('npair_random_pr_curve.png')
plt.close()

# Compute PR curve and PR area
test_precision, test_recall, _ = precision_recall_curve(triplets_random_test_labels, -triplets_random_test_scores)
test_pr_auc = auc(test_recall, test_precision)

vanilla_precision, vanilla_recall, _ = precision_recall_curve(vanilla_labels, -vanilla_scores)
vanilla_pr_auc = auc(vanilla_recall, vanilla_precision)

# Plot PR curve
plt.figure()
plt.plot(test_recall, test_precision, color='green', lw=2, label=f'Test PR curve')
plt.plot(vanilla_recall, vanilla_precision, color='red', lw=2, label=f'Vanilla PR curve')
plt.plot([0, 1], [positive_rate, positive_rate], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('Triplets Random (Test) - PR Curve')
plt.legend(loc="upper right")
plt.savefig('triplets_random_pr_curve.png')
plt.close()

# Compute PR curve and PR area
test_precision, test_recall, _ = precision_recall_curve(triplets_hard_test_labels, -triplets_hard_test_scores)
test_pr_auc = auc(test_recall, test_precision)

vanilla_precision, vanilla_recall, _ = precision_recall_curve(vanilla_labels, -vanilla_scores)
vanilla_pr_auc = auc(vanilla_recall, vanilla_precision)

# Plot PR curve
plt.figure()
plt.plot(test_recall, test_precision, color='green', lw=2, label=f'Test PR curve')
plt.plot(vanilla_recall, vanilla_precision, color='red', lw=2, label=f'Vanilla PR curve')
plt.plot([0, 1], [positive_rate, positive_rate], color='gray', lw=2, linestyle='--', label='Random')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('Triplets Hard (Test) - PR Curve')
plt.legend(loc="upper right")
plt.savefig('triplets_hard_pr_curve.png')
plt.close()

print("Generated PR curves.")