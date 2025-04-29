import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from improved_model import (
    ImprovedCommonFeatureExtractor,
    ImprovedSubjectSpecificMapper,
    ImprovedSubjectSpecificClassifier
)
from improved_training import DEAPDataset

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def load_improved_model(checkpoint_path="checkpoints/best_model_improved.pt"):
    """Load the trained improved model"""
    cfe = ImprovedCommonFeatureExtractor().to(device)
    sfe = ImprovedSubjectSpecificMapper().to(device)
    ssc = ImprovedSubjectSpecificClassifier().to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfe.load_state_dict(checkpoint['cfe'])
    sfe.load_state_dict(checkpoint['sfe'])
    ssc.load_state_dict(checkpoint['ssc'])

    cfe.eval()
    sfe.eval()
    ssc.eval()
    return cfe, sfe, ssc

def test_improved_model(feature_path, label_path, checkpoint_path="checkpoints/best_model.pt", batch_size=128):
    """Test the improved model and generate detailed metrics"""
    print("🔍 Testing improved model...")

    # Load the model
    cfe, sfe, ssc = load_improved_model(checkpoint_path)

    # Prepare dataset
    dataset = DEAPDataset(feature_path, label_path, normalize=True)
    indices = list(range(len(dataset)))
    labels = [label for _, label in dataset]
    _, test_idx, _, _ = train_test_split(indices, labels, test_size=0.2, stratify=labels, random_state=42)
    test_set = Subset(dataset, test_idx)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    all_preds, all_labels = [], []
    all_probs = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            z = sfe(cfe(xb))
            logits = ssc(z)
            probs = F.softmax(logits, dim=1)

            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate accuracy
    acc = accuracy_score(all_labels, all_preds)
    print(f"✅ Test Accuracy: {acc:.4f}")

    # Generate detailed classification report
    class_names = ["LVLA (Sad)", "LVHA (Fear)", "HVLA (Calm)", "HVHA (Happy)"]
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("\nClassification Report:")
    print(report)

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1, 2, 3])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Test Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig("plots/test_confmat_improved.png")
    plt.close()

    # Plot ROC curves for each class
    all_probs = np.array(all_probs)
    all_labels_onehot = np.zeros((len(all_labels), 4))
    for i, label in enumerate(all_labels):
        all_labels_onehot[i, label] = 1

    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(all_labels_onehot[:, i], all_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plots/roc_curves_improved.png")
    plt.close()

    # Plot prediction confidence distribution
    plt.figure(figsize=(10, 6))

    # Get the confidence (max probability) for each prediction
    confidences = np.max(all_probs, axis=1)
    correct = (np.array(all_preds) == np.array(all_labels))

    # Plot confidence distribution for correct and incorrect predictions
    sns.histplot(confidences[correct], bins=20, alpha=0.5, label='Correct Predictions', color='green')
    sns.histplot(confidences[~correct], bins=20, alpha=0.5, label='Incorrect Predictions', color='red')

    plt.xlabel('Confidence (Max Probability)')
    plt.ylabel('Count')
    plt.title('Prediction Confidence Distribution')
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/confidence_distribution.png")
    plt.close()

    return acc, all_labels, all_preds

def analyze_per_subject_performance(feature_path, label_path, checkpoint_path="checkpoints/best_model_improved.pt"):
    """Analyze model performance for each subject separately"""
    print("🔍 Analyzing per-subject performance...")

    # Load the model
    cfe, sfe, ssc = load_improved_model(checkpoint_path)

    # Prepare full dataset
    full_dataset = DEAPDataset(feature_path, label_path, normalize=True)

    # Dictionary to store per-subject accuracies
    subject_accuracies = {}

    # Process each subject
    for subject in range(32):
        subject_dataset = DEAPDataset(feature_path, label_path, only_subject=subject, normalize=True)
        if len(subject_dataset) == 0:
            continue

        subject_loader = DataLoader(subject_dataset, batch_size=64, shuffle=False, num_workers=0)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in subject_loader:
                xb = xb.to(device)
                z = sfe(cfe(xb))
                logits = ssc(z)
                preds = logits.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(yb.numpy())

        # Calculate accuracy for this subject
        if len(all_labels) > 0:
            acc = accuracy_score(all_labels, all_preds)
            subject_accuracies[subject] = acc
            print(f"Subject {subject+1}: Accuracy = {acc:.4f}")

    # Plot per-subject accuracies
    plt.figure(figsize=(12, 6))
    subjects = list(subject_accuracies.keys())
    accs = list(subject_accuracies.values())

    plt.bar(range(len(subjects)), accs)
    plt.axhline(y=np.mean(accs), color='r', linestyle='-', label=f'Mean Accuracy: {np.mean(accs):.4f}')
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Subject')
    plt.xticks(range(len(subjects)), [s+1 for s in subjects])
    plt.legend()
    plt.tight_layout()
    plt.savefig("plots/subject_accuracies.png")
    plt.close()

    return subject_accuracies

if __name__ == "__main__":
    # Use relative paths for better portability
    feature_path = "../../de_features.npy"
    label_path = "../../de_labels.npy"

    # Test the improved model
    test_improved_model(feature_path, label_path, batch_size=64)

    # Analyze per-subject performance
    analyze_per_subject_performance(feature_path, label_path)
