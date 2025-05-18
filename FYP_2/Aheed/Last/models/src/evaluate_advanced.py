import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.manifold import TSNE
import pandas as pd
import argparse
import logging
from tqdm import tqdm

# Import custom modules
from enhanced_dataset import EnhancedDEAPDataset, create_stratified_subject_splits
from advanced_model import MultiSourceContrastiveModel
from advanced_losses import AdvancedLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('evaluation.log')
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path, device='cuda'):
    """Load a trained model"""
    model = MultiSourceContrastiveModel().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('val_metrics', None)

def extract_embeddings(model, dataloader, device='cuda'):
    """Extract embeddings from the model"""
    all_embeddings = []
    all_labels = []
    all_subject_ids = []
    all_predictions = []
    all_is_synthetic = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].cpu().numpy()
            subject_ids = batch['subject_id'].cpu().numpy()
            is_synthetic = batch.get('is_synthetic', torch.zeros(len(labels))).cpu().numpy()
            
            # Get embeddings
            embeddings = model.get_embeddings(eeg, face)
            
            # Get predictions
            outputs = model(eeg, face)
            logits = outputs['logits']
            preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()
            
            # Store results
            all_embeddings.append({
                'eeg': embeddings['eeg_features'].cpu().numpy(),
                'face': embeddings['face_features'].cpu().numpy(),
                'fused': embeddings['fused_features'].cpu().numpy()
            })
            all_labels.append(labels)
            all_subject_ids.append(subject_ids)
            all_predictions.append(preds)
            all_is_synthetic.append(is_synthetic)
    
    # Concatenate results
    embeddings = {
        'eeg': np.concatenate([e['eeg'] for e in all_embeddings]),
        'face': np.concatenate([e['face'] for e in all_embeddings]),
        'fused': np.concatenate([e['fused'] for e in all_embeddings])
    }
    labels = np.concatenate(all_labels)
    subject_ids = np.concatenate(all_subject_ids)
    predictions = np.concatenate(all_predictions)
    is_synthetic = np.concatenate(all_is_synthetic)
    
    return embeddings, labels, subject_ids, predictions, is_synthetic

def visualize_embeddings(embeddings, labels, subject_ids, predictions, is_synthetic, output_dir):
    """Visualize embeddings using t-SNE"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create colormap for subjects
    unique_subjects = np.unique(subject_ids)
    subject_cmap = plt.cm.get_cmap('tab20', len(unique_subjects))
    
    # Create colormap for binary labels
    label_cmap = plt.cm.get_cmap('coolwarm', 2)
    
    # Process each embedding type
    for name, data in embeddings.items():
        logger.info(f"Visualizing {name} embeddings...")
        
        # Apply t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(data)
        
        # 1. Visualize by label
        plt.figure(figsize=(10, 8))
        for i in range(2):  # Binary classification
            mask = labels == i
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[label_cmap(i) for _ in range(np.sum(mask))],
                label=f"Class {i}",
                alpha=0.7,
                edgecolors='none'
            )
        plt.title(f't-SNE Visualization of {name.capitalize()} Embeddings by Label')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_embeddings_by_label.png'))
        plt.close()
        
        # 2. Visualize by subject
        plt.figure(figsize=(12, 10))
        for i, subject in enumerate(unique_subjects):
            if subject < 0:  # Skip synthetic samples
                continue
            mask = subject_ids == subject
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=[subject_cmap(i % 20) for _ in range(np.sum(mask))],
                label=f"Subject {subject}" if i < 10 else None,  # Limit legend entries
                alpha=0.7,
                edgecolors='none'
            )
        plt.title(f't-SNE Visualization of {name.capitalize()} Embeddings by Subject')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_embeddings_by_subject.png'))
        plt.close()
        
        # 3. Visualize correct vs incorrect predictions
        plt.figure(figsize=(10, 8))
        correct_mask = predictions == labels
        incorrect_mask = ~correct_mask
        
        plt.scatter(
            embeddings_2d[correct_mask, 0],
            embeddings_2d[correct_mask, 1],
            c='green',
            label="Correct",
            alpha=0.7,
            edgecolors='none'
        )
        plt.scatter(
            embeddings_2d[incorrect_mask, 0],
            embeddings_2d[incorrect_mask, 1],
            c='red',
            label="Incorrect",
            alpha=0.7,
            edgecolors='none'
        )
        plt.title(f't-SNE Visualization of {name.capitalize()} Embeddings by Prediction')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_embeddings_by_prediction.png'))
        plt.close()
        
        # 4. Visualize real vs synthetic samples
        if np.any(is_synthetic):
            plt.figure(figsize=(10, 8))
            real_mask = ~is_synthetic.astype(bool)
            synthetic_mask = is_synthetic.astype(bool)
            
            plt.scatter(
                embeddings_2d[real_mask, 0],
                embeddings_2d[real_mask, 1],
                c='blue',
                label="Real",
                alpha=0.7,
                edgecolors='none'
            )
            plt.scatter(
                embeddings_2d[synthetic_mask, 0],
                embeddings_2d[synthetic_mask, 1],
                c='orange',
                label="Synthetic",
                alpha=0.7,
                edgecolors='none'
            )
            plt.title(f't-SNE Visualization of {name.capitalize()} Embeddings by Sample Type')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{name}_embeddings_by_sample_type.png'))
            plt.close()

def analyze_subject_performance(predictions, labels, subject_ids, output_dir):
    """Analyze and visualize performance by subject"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate metrics for each subject
    unique_subjects = np.unique(subject_ids)
    subject_metrics = []
    
    for subject in unique_subjects:
        mask = subject_ids == subject
        if np.sum(mask) > 0:
            subj_preds = predictions[mask]
            subj_labels = labels[mask]
            
            # Calculate metrics
            accuracy = np.mean(subj_preds == subj_labels)
            
            # Calculate class distribution
            class_counts = np.bincount(subj_labels, minlength=2)
            class_0_count = class_counts[0]
            class_1_count = class_counts[1]
            
            # Calculate class-specific accuracy
            class_0_acc = np.mean(subj_preds[subj_labels == 0] == 0) if class_0_count > 0 else 0
            class_1_acc = np.mean(subj_preds[subj_labels == 1] == 1) if class_1_count > 0 else 0
            
            subject_metrics.append({
                'subject_id': subject,
                'accuracy': accuracy,
                'samples': np.sum(mask),
                'class_0_count': class_0_count,
                'class_1_count': class_1_count,
                'class_0_acc': class_0_acc,
                'class_1_acc': class_1_acc
            })
    
    # Create DataFrame for analysis
    df = pd.DataFrame(subject_metrics)
    df = df.sort_values('accuracy', ascending=False)
    
    # Save to CSV
    df.to_csv(os.path.join(output_dir, 'subject_performance.csv'), index=False)
    
    # Plot subject accuracies
    plt.figure(figsize=(12, 6))
    bars = plt.bar(df['subject_id'], df['accuracy'])
    
    # Color bars by performance
    for i, bar in enumerate(bars):
        if df.iloc[i]['accuracy'] >= 0.8:
            bar.set_color('green')
        elif df.iloc[i]['accuracy'] >= 0.6:
            bar.set_color('blue')
        elif df.iloc[i]['accuracy'] >= 0.5:
            bar.set_color('orange')
        else:
            bar.set_color('red')
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
    plt.axhline(y=df['accuracy'].mean(), color='g', linestyle='--', label=f'Mean Accuracy: {df["accuracy"].mean():.2f}')
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy by Subject')
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_accuracies.png'))
    plt.close()
    
    # Plot class distribution by subject
    plt.figure(figsize=(12, 6))
    width = 0.35
    x = np.arange(len(df))
    
    plt.bar(x - width/2, df['class_0_count'], width, label='Class 0 (Negative)')
    plt.bar(x + width/2, df['class_1_count'], width, label='Class 1 (Positive)')
    
    plt.xlabel('Subject ID')
    plt.ylabel('Sample Count')
    plt.title('Class Distribution by Subject')
    plt.xticks(x, df['subject_id'])
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_class_distribution.png'))
    plt.close()
    
    # Plot class-specific accuracy by subject
    plt.figure(figsize=(12, 6))
    
    plt.bar(x - width/2, df['class_0_acc'], width, label='Class 0 Accuracy')
    plt.bar(x + width/2, df['class_1_acc'], width, label='Class 1 Accuracy')
    
    plt.axhline(y=0.5, color='r', linestyle='--', label='Random Chance')
    plt.xlabel('Subject ID')
    plt.ylabel('Accuracy')
    plt.title('Class-Specific Accuracy by Subject')
    plt.xticks(x, df['subject_id'])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_class_accuracy.png'))
    plt.close()
    
    return df

def plot_confusion_matrix(predictions, labels, output_dir):
    """Plot and save confusion matrix"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate confusion matrix
    cm = confusion_matrix(labels, predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot normalized confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_normalized.png'))
    plt.close()

def plot_roc_curve(predictions_prob, labels, output_dir):
    """Plot and save ROC curve"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate ROC curve
    fpr, tpr, _ = roc_curve(labels, predictions_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    return roc_auc

def main(args):
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    model, val_metrics = load_model(args.model_path, device)
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = EnhancedDEAPDataset(
        processed_dir=args.data_dir,
        target='valence',
        apply_augmentation=False,  # No augmentation for evaluation
        balance_classes=False,     # Use original distribution
        balance_subjects=False
    )
    
    # Create data splits
    train_set, val_set, test_set = create_stratified_subject_splits(
        dataset,
        test_size=0.15,
        val_size=0.15,
        random_state=42
    )
    
    # Create data loader for the specified split
    if args.split == 'train':
        eval_set = train_set
    elif args.split == 'val':
        eval_set = val_set
    else:  # test
        eval_set = test_set
    
    logger.info(f"Evaluating on {args.split} split with {len(eval_set)} samples...")
    
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )
    
    # Extract embeddings and predictions
    logger.info("Extracting embeddings and predictions...")
    embeddings, labels, subject_ids, predictions, is_synthetic = extract_embeddings(model, eval_loader, device)
    
    # Get prediction probabilities
    logger.info("Getting prediction probabilities...")
    all_probs = []
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Getting probabilities"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            
            outputs = model(eeg, face)
            logits = outputs['logits']
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
    
    predictions_prob = np.concatenate(all_probs)
    
    # Visualize embeddings
    logger.info("Visualizing embeddings...")
    visualize_embeddings(
        embeddings,
        labels,
        subject_ids,
        predictions,
        is_synthetic,
        os.path.join(args.output_dir, 'embeddings')
    )
    
    # Analyze subject performance
    logger.info("Analyzing subject performance...")
    subject_df = analyze_subject_performance(
        predictions,
        labels,
        subject_ids,
        os.path.join(args.output_dir, 'subject_analysis')
    )
    
    # Plot confusion matrix
    logger.info("Plotting confusion matrix...")
    plot_confusion_matrix(
        predictions,
        labels,
        os.path.join(args.output_dir, 'metrics')
    )
    
    # Plot ROC curve
    logger.info("Plotting ROC curve...")
    roc_auc = plot_roc_curve(
        predictions_prob,
        labels,
        os.path.join(args.output_dir, 'metrics')
    )
    
    # Calculate overall metrics
    accuracy = np.mean(predictions == labels)
    
    # Generate classification report
    report = classification_report(labels, predictions, output_dict=True)
    
    # Save metrics to file
    metrics_file = os.path.join(args.output_dir, 'evaluation_metrics.txt')
    with open(metrics_file, 'w') as f:
        f.write(f"Evaluation on {args.split} split\n")
        f.write(f"Model: {args.model_path}\n\n")
        
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n\n")
        
        f.write("Classification Report:\n")
        for cls in ['0', '1']:
            f.write(f"Class {cls}:\n")
            f.write(f"  Precision: {report[cls]['precision']:.4f}\n")
            f.write(f"  Recall: {report[cls]['recall']:.4f}\n")
            f.write(f"  F1-score: {report[cls]['f1-score']:.4f}\n")
            f.write(f"  Support: {report[cls]['support']}\n\n")
        
        f.write(f"Macro Avg F1: {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"Weighted Avg F1: {report['weighted avg']['f1-score']:.4f}\n\n")
        
        f.write("Subject Performance Summary:\n")
        f.write(f"  Best Subject: {subject_df.iloc[0]['subject_id']} (Acc: {subject_df.iloc[0]['accuracy']:.4f})\n")
        f.write(f"  Worst Subject: {subject_df.iloc[-1]['subject_id']} (Acc: {subject_df.iloc[-1]['accuracy']:.4f})\n")
        f.write(f"  Mean Subject Accuracy: {subject_df['accuracy'].mean():.4f}\n")
        f.write(f"  Std Dev Subject Accuracy: {subject_df['accuracy'].std():.4f}\n")
    
    logger.info(f"Evaluation completed. Results saved to {args.output_dir}")
    logger.info(f"Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and visualize advanced EEG emotion recognition model")
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model')
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed", 
                        help='Directory with processed data')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'], 
                        help='Dataset split to evaluate on')
    
    args = parser.parse_args()
    
    main(args)
