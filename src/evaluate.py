"""
Evaluation utilities: classification report, confusion matrix, softmax
probability collection, and ROC/AUC curves (overall + per-class).
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.preprocessing import label_binarize

from . import config


def evaluate_model(model, test_loader, class_names, label: str = "Model"):
    """Run inference on the test set and print a classification report + confusion matrix."""
    model.eval()
    all_preds, all_labels = [], []

    print(f"{label} Evaluation")

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    print(classification_report(all_labels, all_preds, target_names=class_names))
    print("Confusion Matrix")
    print(confusion_matrix(all_labels, all_preds))

    return all_labels, all_preds


def collect_softmax_probs(model, test_loader):
    """Run inference and collect softmax probabilities + true labels (for ROC curves)."""
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(config.DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_probs)


def plot_roc_comparison(all_labels, probs_by_model: dict, num_classes: int = 4):
    """
    probs_by_model: {"Baseline CNN": probs_array, "Fine-Tuned CNN": probs_array, ...}
    Plots one micro-averaged ROC curve per model on the same axes.
    """
    y_bin = label_binarize(all_labels, classes=list(range(num_classes)))

    plt.figure(figsize=(7, 6))
    for name, scores in probs_by_model.items():
        fpr, tpr, _ = roc_curve(y_bin.ravel(), scores.ravel())
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison of Models")
    plt.legend()
    plt.show()


def plot_per_class_roc(all_labels, probs, class_names):
    """Per-class ROC curves for a single model (used on the fine-tuned CNN in the report)."""
    y_true_bin = label_binarize(all_labels, classes=list(range(len(class_names))))

    plt.figure(figsize=(8, 6))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cls} (AUC = {roc_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Per-Class ROC Curves (Fine-Tuned CNN)")
    plt.legend()
    plt.show()
