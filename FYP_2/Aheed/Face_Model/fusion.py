import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import logging
import json
import argparse
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from pathlib import Path
import pickle

from face_model import FaceEmotionClassifier, FusionModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fusion.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EEGFaceFusionDataset(Dataset):
    """Dataset for late fusion of EEG and Face embeddings"""
    
    def __init__(self, processed_dir, target='valence', eeg_model_path=None, face_model_path=None, mode='test'):
        """
        Args:
            processed_dir: Directory with processed data
            target: Target emotion ('valence' or 'arousal')
            eeg_model_path: Path to EEG model
            face_model_path: Path to Face model
            mode: 'train', 'val', or 'test'
        """
        self.processed_dir = Path(processed_dir)
        self.target = target
        self.mode = mode
        
        # Paths to models
        self.eeg_model_path = eeg_model_path
        self.face_model_path = face_model_path
        
        # Load samples with both EEG and face data
        self.samples = self._load_samples()
        logger.info(f"Loaded {len(self.samples)} samples for {mode} set")
    
    def _load_samples(self):
        """Load samples with both EEG and face data"""
        all_samples = []
        
        for subj_dir in sorted(self.processed_dir.glob("s*")):
            if not subj_dir.is_dir():
                continue
                
            # Only subjects 1-22 have video data
            if int(subj_dir.name[1:]) > 22:
                continue
                
            for trial_dir in sorted(subj_dir.glob("trial*")):
                if not trial_dir.is_dir():
                    continue
                    
                metadata_file = trial_dir / "metadata.json"
                eeg_file = trial_dir / "eeg.npy"
                face_file = trial_dir / "face.npy"
                
                if not all(f.exists() for f in [metadata_file, eeg_file, face_file]):
                    continue
                    
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                    # Skip if target label is not available
                    if metadata.get(self.target, -1) < 0:
                        continue
                        
                    # Skip if no video features
                    if not metadata.get('has_video', False):
                        continue
                        
                    sample = {
                        'eeg_path': str(eeg_file),
                        'face_path': str(face_file),
                        'label': metadata.get(self.target),
                        'subject_id': int(subj_dir.name[1:]),
                        'trial_id': int(trial_dir.name[5:])
                    }
                    
                    all_samples.append(sample)
                except Exception as e:
                    logger.warning(f"Error loading {metadata_file}: {e}")
        
        return all_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample with both EEG and face data"""
        sample = self.samples[idx]
        
        # Load features
        eeg_features = np.load(sample['eeg_path'])
        face_features = np.load(sample['face_path'])
        
        return {
            'eeg_features': torch.tensor(eeg_features, dtype=torch.float32),
            'face_features': torch.tensor(face_features, dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'subject_id': sample['subject_id'],
            'trial_id': sample['trial_id']
        }

def load_pretrained_models(args):
    """Load pretrained EEG and Face models"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Load Face model
    face_model = FaceEmotionClassifier(
        num_classes=2 if args.target in ['valence', 'arousal'] else 4
    ).to(device)
    
    face_checkpoint = torch.load(args.face_model_path, map_location=device)
    face_model.load_state_dict(face_checkpoint['model_state_dict'])
    face_model.eval()
    logger.info(f"Loaded face model from {args.face_model_path}")
    
    # For EEG model, we'll just use embeddings from numpy files
    # This is a placeholder as we're using the saved model from the Jupyter notebook
    
    return face_model

def extract_embeddings(data_loader, face_model, device, output_dir, split):
    """Extract embeddings from both modalities"""
    embeddings = {
        'eeg_embeddings': [],
        'face_embeddings': [],
        'labels': [],
        'subject_ids': [],
        'trial_ids': []
    }
    
    face_model.eval()
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting embeddings for {split}"):
            # Get data
            eeg_features = batch['eeg_features'].to(device)
            face_features = batch['face_features'].to(device)
            labels = batch['label'].cpu().numpy()
            subject_ids = batch['subject_id']
            trial_ids = batch['trial_id']
            
            # Get face embeddings
            outputs = face_model(face_features)
            face_embeddings = outputs['embeddings_normalized'].cpu().numpy()
            
            # Store embeddings
            embeddings['eeg_embeddings'].append(eeg_features.cpu().numpy())
            embeddings['face_embeddings'].append(face_embeddings)
            embeddings['labels'].append(labels)
            embeddings['subject_ids'].extend(subject_ids)
            embeddings['trial_ids'].extend(trial_ids)
    
    # Concatenate embeddings
    for key in ['eeg_embeddings', 'face_embeddings', 'labels']:
        embeddings[key] = np.concatenate(embeddings[key], axis=0)
    
    # Save embeddings
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{split}_embeddings.pkl"), 'wb') as f:
        pickle.dump(embeddings, f)
    
    logger.info(f"Saved {split} embeddings with shape: {embeddings['eeg_embeddings'].shape}, {embeddings['face_embeddings'].shape}")
    
    return embeddings

def train_fusion_model(train_embeddings, val_embeddings, args):
    """Train the fusion model"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Create fusion model
    fusion_model = FusionModel(
        embedding_dim=train_embeddings['eeg_embeddings'].shape[1],
        num_classes=2 if args.target in ['valence', 'arousal'] else 4
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(fusion_model.parameters(), lr=args.learning_rate)
    
    # Create criterion
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0
    best_epoch = 0
    patience_counter = 0
    
    # Convert to torch tensors
    train_eeg = torch.tensor(train_embeddings['eeg_embeddings'], dtype=torch.float32).to(device)
    train_face = torch.tensor(train_embeddings['face_embeddings'], dtype=torch.float32).to(device)
    train_labels = torch.tensor(train_embeddings['labels'], dtype=torch.long).to(device)
    
    val_eeg = torch.tensor(val_embeddings['eeg_embeddings'], dtype=torch.float32).to(device)
    val_face = torch.tensor(val_embeddings['face_embeddings'], dtype=torch.float32).to(device)
    val_labels = torch.tensor(val_embeddings['labels'], dtype=torch.long).to(device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': []
    }
    
    for epoch in range(args.epochs):
        # Training
        fusion_model.train()
        
        optimizer.zero_grad()
        
        # Forward pass
        train_logits = fusion_model(train_face, train_eeg)
        
        # Compute loss
        loss = criterion(train_logits, train_labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        _, train_preds = torch.max(train_logits, dim=1)
        train_acc = accuracy_score(train_labels.cpu().numpy(), train_preds.cpu().numpy())
        train_f1 = f1_score(train_labels.cpu().numpy(), train_preds.cpu().numpy(), average='macro')
        
        # Validation
        fusion_model.eval()
        
        with torch.no_grad():
            # Forward pass
            val_logits = fusion_model(val_face, val_eeg)
            
            # Compute loss
            val_loss = criterion(val_logits, val_labels)
            
            # Calculate metrics
            _, val_preds = torch.max(val_logits, dim=1)
            val_acc = accuracy_score(val_labels.cpu().numpy(), val_preds.cpu().numpy())
            val_f1 = f1_score(val_labels.cpu().numpy(), val_preds.cpu().numpy(), average='macro')
        
        # Update metrics
        train_metrics['train_loss'].append(loss.item())
        train_metrics['val_loss'].append(val_loss.item())
        train_metrics['train_acc'].append(train_acc)
        train_metrics['val_acc'].append(val_acc)
        train_metrics['train_f1'].append(train_f1)
        train_metrics['val_f1'].append(val_f1)
        
        # Log
        logger.info(f"Epoch {epoch+1}/{args.epochs}:")
        logger.info(f"  Train Loss: {loss.item():.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"  Val Loss: {val_loss.item():.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Check for best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': fusion_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(args.output_dir, 'best_fusion_model.pt'))
            
            logger.info(f"  New best model saved with accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs (best: {best_val_acc:.4f} at epoch {best_epoch+1})")
            
            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Plot metrics
    plot_metrics(train_metrics, args.output_dir)
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'fusion_metrics.json'), 'w') as f:
        json.dump(train_metrics, f, indent=2)
    
    # Load best model for testing
    checkpoint = torch.load(os.path.join(args.output_dir, 'best_fusion_model.pt'))
    fusion_model.load_state_dict(checkpoint['model_state_dict'])
    
    return fusion_model

def evaluate_fusion_model(fusion_model, test_embeddings, args):
    """Evaluate fusion model on test set"""
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    
    # Convert to torch tensors
    test_eeg = torch.tensor(test_embeddings['eeg_embeddings'], dtype=torch.float32).to(device)
    test_face = torch.tensor(test_embeddings['face_embeddings'], dtype=torch.float32).to(device)
    test_labels = torch.tensor(test_embeddings['labels'], dtype=torch.long).to(device)
    
    # Evaluate
    fusion_model.eval()
    
    with torch.no_grad():
        # Forward pass
        test_logits = fusion_model(test_face, test_eeg)
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        test_loss = criterion(test_logits, test_labels)
        
        # Calculate metrics
        _, test_preds = torch.max(test_logits, dim=1)
        test_acc = accuracy_score(test_labels.cpu().numpy(), test_preds.cpu().numpy())
        test_f1 = f1_score(test_labels.cpu().numpy(), test_preds.cpu().numpy(), average='macro')
        test_cm = confusion_matrix(test_labels.cpu().numpy(), test_preds.cpu().numpy())
    
    # Log results
    logger.info(f"Test Results:")
    logger.info(f"  Loss: {test_loss.item():.4f}")
    logger.info(f"  Accuracy: {test_acc:.4f}")
    logger.info(f"  F1 Score: {test_f1:.4f}")
    logger.info(f"  Confusion Matrix:\n{test_cm}")
    
    # Save results
    with open(os.path.join(args.output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'test_loss': test_loss.item(),
            'test_accuracy': test_acc,
            'test_f1_score': test_f1,
            'confusion_matrix': test_cm.tolist()
        }, f, indent=2)
    
    return {
        'loss': test_loss.item(),
        'accuracy': test_acc,
        'f1_score': test_f1,
        'confusion_matrix': test_cm
    }

def plot_metrics(metrics, output_dir):
    """Plot training metrics"""
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'fusion_loss.png'))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'fusion_accuracy.png'))
    plt.close()
    
    # Plot F1 score
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_f1'], label='Train F1 Score')
    plt.plot(metrics['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'fusion_f1_score.png'))
    plt.close()

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Late fusion of EEG and Face models")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed", 
                        help="Path to processed data directory")
    parser.add_argument('--face_model_path', type=str, required=True,
                        help="Path to pretrained face model")
    parser.add_argument('--target', type=str, default='valence', choices=['valence', 'arousal'],
                        help="Emotion target to predict")
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--patience', type=int, default=20, help="Early stopping patience")
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--output_dir', type=str, default='outputs/fusion', help="Output directory")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'fusion_args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Load models
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    face_model = load_pretrained_models(args)
    
    # Create datasets
    train_dataset = EEGFaceFusionDataset(args.data_dir, args.target, mode='train')
    val_dataset = EEGFaceFusionDataset(args.data_dir, args.target, mode='val')
    test_dataset = EEGFaceFusionDataset(args.data_dir, args.target, mode='test')
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Extract embeddings
    logger.info("Extracting embeddings...")
    train_embeddings = extract_embeddings(train_loader, face_model, device, args.output_dir, 'train')
    val_embeddings = extract_embeddings(val_loader, face_model, device, args.output_dir, 'val')
    test_embeddings = extract_embeddings(test_loader, face_model, device, args.output_dir, 'test')
    
    # Train fusion model
    logger.info("Training fusion model...")
    fusion_model = train_fusion_model(train_embeddings, val_embeddings, args)
    
    # Evaluate fusion model
    logger.info("Evaluating fusion model...")
    test_results = evaluate_fusion_model(fusion_model, test_embeddings, args)
    
    logger.info("Late fusion completed!")
    logger.info(f"Test accuracy: {test_results['accuracy']:.4f}")
    logger.info(f"Test F1 score: {test_results['f1_score']:.4f}")

if __name__ == "__main__":
    main() 