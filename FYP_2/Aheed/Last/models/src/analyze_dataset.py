import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from tqdm import tqdm
import logging

# Import your dataset and model
from dataset import DEAPDataset
from model import EmotionClassifier

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path, device='cuda'):
    """Load a trained model"""
    model = EmotionClassifier().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model

def extract_features(model, dataset, device='cuda', batch_size=32):
    """Extract features from the model for visualization"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    all_eeg_features = []
    all_face_features = []
    all_fused_features = []
    all_labels = []
    all_preds = []
    all_subject_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            subject_ids = batch['subject_id']
            
            # Forward pass
            logits, eeg_proj, face_proj = model(eeg, face)
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
            
            # Get intermediate features
            eeg_features = model.get_eeg_encoder_output(eeg)
            face_features = model.get_face_encoder_output(face)
            
            # Store features and metadata
            all_eeg_features.append(eeg_features.cpu().numpy())
            all_face_features.append(face_features.cpu().numpy())
            all_fused_features.append(torch.cat([eeg_proj, face_proj], dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_subject_ids.append(subject_ids.numpy())
    
    # Concatenate results
    return {
        'eeg_features': np.concatenate(all_eeg_features),
        'face_features': np.concatenate(all_face_features),
        'fused_features': np.concatenate(all_fused_features),
        'labels': np.concatenate(all_labels),
        'predictions': np.concatenate(all_preds),
        'subject_ids': np.concatenate(all_subject_ids)
    }

def visualize_features(features, labels, title, output_path):
    """Visualize features using t-SNE"""
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_2d[:, 0], 
        features_2d[:, 1], 
        c=labels, 
        cmap='coolwarm', 
        alpha=0.7
    )
    plt.colorbar(scatter, label='Class')
    plt.title(f't-SNE Visualization of {title}')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def analyze_subject_performance(predictions, labels, subject_ids):
    """Analyze performance by subject"""
    unique_subjects = np.unique(subject_ids)
    subject_accuracies = []
    
    for subject in unique_subjects:
        mask = subject_ids == subject
        if np.sum(mask) > 0:  # Ensure we have samples for this subject
            subject_preds = predictions[mask]
            subject_labels = labels[mask]
            accuracy = np.mean(subject_preds == subject_labels)
            subject_accuracies.append((subject, accuracy, np.sum(mask)))
    
    # Sort by accuracy
    subject_accuracies.sort(key=lambda x: x[1], reverse=True)
    
    # Create DataFrame for better visualization
    df = pd.DataFrame(subject_accuracies, columns=['Subject', 'Accuracy', 'Samples'])
    
    # Plot subject accuracies
    plt.figure(figsize=(12, 6))
    plt.bar(df['Subject'], df['Accuracy'])
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Subject')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('subject_accuracies.png')
    plt.close()
    
    return df

def analyze_errors(features, predictions, labels, output_dir):
    """Analyze error patterns"""
    # Identify correct and incorrect predictions
    correct_mask = predictions == labels
    incorrect_mask = ~correct_mask
    
    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Plot correct vs incorrect predictions
    plt.figure(figsize=(10, 8))
    plt.scatter(
        features_2d[correct_mask, 0], 
        features_2d[correct_mask, 1], 
        c='green', 
        alpha=0.5,
        label='Correct'
    )
    plt.scatter(
        features_2d[incorrect_mask, 0], 
        features_2d[incorrect_mask, 1], 
        c='red', 
        alpha=0.5,
        label='Incorrect'
    )
    plt.title('PCA Visualization of Correct vs Incorrect Predictions')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'error_analysis.png'))
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Classification report
    report = classification_report(labels, predictions, output_dict=True)
    return report

def main():
    # Create output directory
    output_dir = 'analysis_results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    dataset = DEAPDataset(
        processed_dir=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
        target='valence',
        apply_augmentation=False,
        balance_classes=False  # Use original distribution for analysis
    )
    
    # Load your best model
    model_path = r'C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\models\src\outputs\run_20250502_110624\best_model.pth'  # Update with your model path
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)
    
    # Extract features
    logger.info("Extracting features from the model...")
    features = extract_features(model, dataset, device)
    
    # Visualize features
    logger.info("Visualizing features...")
    visualize_features(
        features['eeg_features'], 
        features['labels'], 
        'EEG Features', 
        os.path.join(output_dir, 'eeg_features_tsne.png')
    )
    
    visualize_features(
        features['face_features'], 
        features['labels'], 
        'Face Features', 
        os.path.join(output_dir, 'face_features_tsne.png')
    )
    
    visualize_features(
        features['fused_features'], 
        features['labels'], 
        'Fused Features', 
        os.path.join(output_dir, 'fused_features_tsne.png')
    )
    
    # Analyze subject performance
    logger.info("Analyzing performance by subject...")
    subject_df = analyze_subject_performance(
        features['predictions'], 
        features['labels'], 
        features['subject_ids']
    )
    subject_df.to_csv(os.path.join(output_dir, 'subject_performance.csv'), index=False)
    
    # Analyze errors
    logger.info("Analyzing error patterns...")
    error_report = analyze_errors(
        features['fused_features'], 
        features['predictions'], 
        features['labels'], 
        output_dir
    )
    
    # Save error report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        for class_id, metrics in error_report.items():
            if class_id in ['0', '1']:
                f.write(f"Class {class_id}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value:.4f}\n")
        f.write(f"\nAccuracy: {error_report['accuracy']:.4f}\n")
    
    logger.info(f"Analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
