import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path
# Import your model and dataset classes
from model import EmotionClassifier
from enhanced_dataset import EnhancedDEAPDataset

def load_model(model_path, device):
    """Load trained model from checkpoint"""
    print(f"Loading model from {model_path}")
    
    # Initialize model
    model = EmotionClassifier().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Print model info
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown'):.4f}")
    
    return model, checkpoint

def extract_features(model, dataloader, device):
    """Extract features and predictions from the model"""
    model.eval()
    all_eeg_features = []
    all_face_features = []
    all_fused_features = []
    all_labels = []
    all_preds = []
    all_probs = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Move data to device
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            subject_ids = batch.get('subject_id', torch.zeros_like(labels)).to(device)
            
            # Forward pass
            logits, projections = model(eeg, face)
            eeg_features = model.eeg_encoder(eeg)
            face_features = model.face_encoder(face)
            
            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            # Store features and predictions
            all_eeg_features.append(eeg_features.cpu().numpy())
            all_face_features.append(face_features.cpu().numpy())
            all_fused_features.append(projections.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_subject_ids.append(subject_ids.cpu().numpy())
    
    # Concatenate all batches
    return {
        'eeg_features': np.vstack(all_eeg_features),
        'face_features': np.vstack(all_face_features),
        'fused_features': np.vstack(all_fused_features),
        'labels': np.concatenate(all_labels),
        'predictions': np.concatenate(all_preds),
        'probabilities': np.vstack(all_probs),
        'subject_ids': np.concatenate(all_subject_ids)
    }

def plot_tsne(features, labels, subject_ids, title, output_path, color_by='labels'):
    """Plot t-SNE visualization of features"""
    print(f"Generating t-SNE plot for {title}...")
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    tsne_result = tsne.fit_transform(features)
    
    # Create DataFrame for easy plotting
    df = pd.DataFrame({
        'x': tsne_result[:, 0],
        'y': tsne_result[:, 1],
        'label': labels,
        'subject': subject_ids
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    
    if color_by == 'labels':
        # Color by emotion labels
        sns.scatterplot(
            x='x', y='y',
            hue='label',
            palette=sns.color_palette("hls", len(np.unique(labels))),
            data=df,
            legend="full",
            alpha=0.7
        )
        plt.title(f't-SNE Visualization of {title} (Colored by Emotion)')
        
    elif color_by == 'subjects':
        # Color by subject IDs
        sns.scatterplot(
            x='x', y='y',
            hue='subject',
            palette=sns.color_palette("tab20", len(np.unique(subject_ids))),
            data=df,
            legend="full",
            alpha=0.7
        )
        plt.title(f't-SNE Visualization of {title} (Colored by Subject)')
    
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix"""
    print("Generating confusion matrix...")
    
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig(output_path)
    plt.close()

def plot_roc_curve(y_true, y_prob, output_path):
    """Plot ROC curve"""
    print("Generating ROC curve...")
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(output_path)
    plt.close()

def plot_feature_distributions(features, labels, title, output_path):
    """Plot feature distributions by class"""
    print(f"Generating feature distribution plots for {title}...")
    
    # Use PCA to reduce dimensionality for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    pca_result = pca.fit_transform(features)
    
    # Plot distributions of top 5 principal components
    plt.figure(figsize=(15, 10))
    for i in range(5):
        plt.subplot(2, 3, i+1)
        for label in np.unique(labels):
            sns.kdeplot(pca_result[labels == label, i], label=f'Class {label}')
        plt.title(f'PC{i+1} Distribution')
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_subject_performance(features, labels, predictions, subject_ids, output_path):
    """Analyze performance by subject"""
    print("Analyzing subject-level performance...")
    
    # Calculate accuracy per subject
    subjects = np.unique(subject_ids)
    subject_accuracies = []
    
    for subject in subjects:
        mask = subject_ids == subject
        if np.sum(mask) > 0:
            acc = np.mean(predictions[mask] == labels[mask])
            subject_accuracies.append((subject, acc, np.sum(mask)))
    
    # Sort by accuracy
    subject_accuracies.sort(key=lambda x: x[1])
    
    # Plot
    plt.figure(figsize=(12, 8))
    subjects = [str(s[0]) for s in subject_accuracies]
    accuracies = [s[1] for s in subject_accuracies]
    sample_counts = [s[2] for s in subject_accuracies]
    
    # Create bar plot with sample count as color intensity
    bars = plt.bar(subjects, accuracies, color='skyblue')
    
    # Color bars based on sample count
    max_count = max(sample_counts)
    for i, (bar, count) in enumerate(zip(bars, sample_counts)):
        # Normalize count to range [0.3, 1.0] for color intensity
        intensity = 0.3 + 0.7 * (count / max_count)
        bar.set_color(plt.cm.Blues(intensity))
    
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--', label=f'Average: {np.mean(accuracies):.2f}')
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Subject')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    
    # Save detailed results as CSV
    df = pd.DataFrame({
        'Subject': [s[0] for s in subject_accuracies],
        'Accuracy': [s[1] for s in subject_accuracies],
        'Sample_Count': [s[2] for s in subject_accuracies]
    })
    df.to_csv(output_path.replace('.png', '.csv'), index=False)

def analyze_feature_importance(model, output_path):
    """Analyze feature importance from model weights"""
    print("Analyzing feature importance...")
    
    # This is a simplified approach - for a real model you'd need to adapt this
    # based on your specific architecture
    try:
        # Get weights from the first layer of EEG encoder
        eeg_weights = model.eeg_encoder[0].weight.cpu().detach().numpy()
        
        # Compute importance as the L1 norm of weights
        importance = np.abs(eeg_weights).mean(axis=0)
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance)), importance)
        plt.xlabel('EEG Feature Index')
        plt.ylabel('Importance Score')
        plt.title('EEG Feature Importance')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Could not analyze feature importance: {e}")

def generate_classification_report(y_true, y_pred, output_path):
    """Generate and save classification report"""
    print("Generating classification report...")
    
    # Get report
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Convert to DataFrame for better formatting
    df = pd.DataFrame(report).transpose()
    
    # Save to CSV
    df.to_csv(output_path)
    
    # Print summary
    print("\nClassification Report Summary:")
    print(f"Accuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted F1: {report['weighted avg']['f1-score']:.4f}")

def main():
    parser = argparse.ArgumentParser(description='Analyze trained emotion recognition model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the processed data directory')
    parser.add_argument('--output_dir', type=str, default='./analysis_results', help='Directory to save analysis results')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for feature extraction')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(args.model_path, device)
    
    # Load dataset
    print(f"Loading dataset from {args.data_dir}")
    dataset = EnhancedDEAPDataset(
        processed_dir=args.data_dir,
        target='valence',
        apply_augmentation=False,  # No augmentation for analysis
        balance_classes=False,     # Use original distribution
        cache_features=True
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Extract features
    features = extract_features(model, dataloader, device)
    
    # Generate t-SNE visualizations
    plot_tsne(
        features['fused_features'], 
        features['labels'], 
        features['subject_ids'], 
        'Fused Features', 
        os.path.join(args.output_dir, 'tsne_fused_by_emotion.png'),
        color_by='labels'
    )
    
    plot_tsne(
        features['fused_features'], 
        features['labels'], 
        features['subject_ids'], 
        'Fused Features', 
        os.path.join(args.output_dir, 'tsne_fused_by_subject.png'),
        color_by='subjects'
    )
    
    plot_tsne(
        features['eeg_features'], 
        features['labels'], 
        features['subject_ids'], 
        'EEG Features', 
        os.path.join(args.output_dir, 'tsne_eeg_by_emotion.png'),
        color_by='labels'
    )
    
    plot_tsne(
        features['face_features'], 
        features['labels'], 
        features['subject_ids'], 
        'Face Features', 
        os.path.join(args.output_dir, 'tsne_face_by_emotion.png'),
        color_by='labels'
    )
    
    # Generate confusion matrix
    plot_confusion_matrix(
        features['labels'], 
        features['predictions'], 
        os.path.join(args.output_dir, 'confusion_matrix.png')
    )
    
    # Generate ROC curve
    plot_roc_curve(
        features['labels'], 
        features['probabilities'], 
        os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # Generate feature distributions
    plot_feature_distributions(
        features['fused_features'], 
        features['labels'], 
        'Fused Features', 
        os.path.join(args.output_dir, 'feature_distributions.png')
    )
    
    # Analyze subject performance
    analyze_subject_performance(
        features['fused_features'], 
        features['labels'], 
        features['predictions'], 
        features['subject_ids'], 
        os.path.join(args.output_dir, 'subject_performance.png')
    )
    
    # Analyze feature importance
    analyze_feature_importance(
        model, 
        os.path.join(args.output_dir, 'feature_importance.png')
    )
    
    # Generate classification report
    generate_classification_report(
        features['labels'], 
        features['predictions'], 
        os.path.join(args.output_dir, 'classification_report.csv')
    )
    
    # Save model summary
    try:
        from torchinfo import summary
        with open(os.path.join(args.output_dir, 'model_summary.txt'), 'w') as f:
            # Assuming input shapes based on your dataset
            model_summary = summary(model, input_size=[(args.batch_size, 32), (args.batch_size, 768)], device=device)
            print(model_summary, file=f)
    except ImportError:
        print("torchinfo not installed. Skipping model summary.")
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}")
    print("\nSummary of visualizations generated:")
    print("1. t-SNE plots of features (colored by emotion and subject)")
    print("2. Confusion matrix")
    print("3. ROC curve")
    print("4. Feature distributions by class")
    print("5. Subject-level performance analysis")
    print("6. Feature importance analysis")
    print("7. Detailed classification report")
    print("8. Model architecture summary")

if __name__ == "__main__":
    main()