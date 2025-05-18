import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import numpy as np
import os
import logging
from pathlib import Path
from datetime import datetime
import json
import argparse
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
try:
    import wandb
    wandb_available = True
except ImportError:
    wandb_available = False
    print("Warning: wandb not installed. Logging to wandb will be disabled.")
from tqdm import tqdm

from face_model import FaceEmotionClassifier
from face_dataset import FaceDataset, create_subject_aware_splits

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("face_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)

def plot_and_save(metrics, output_dir):
    """Plot training history and save figures"""
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
    plt.savefig(os.path.join(plots_dir, 'loss.png'))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_acc'], label='Train Accuracy')
    plt.plot(metrics['val_acc'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'accuracy.png'))
    plt.close()
    
    # Plot F1 score
    plt.figure(figsize=(10, 5))
    plt.plot(metrics['train_f1'], label='Train F1 Score')
    plt.plot(metrics['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('Training and Validation F1 Score')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'f1_score.png'))
    plt.close()

class WarmupCosineScheduler:
    """Warmup cosine learning rate scheduler"""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

def evaluate(model, data_loader, criterion, device):
    """Evaluate model on given data loader"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in data_loader:
            face_features = batch['face_features'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(face_features)
            logits = outputs['logits']
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            num_batches += 1
            
            # Get predicted classes
            _, preds = torch.max(logits, dim=1)
            
            # Convert tensors to numpy arrays before extending lists
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print actual predictions distribution for debugging
    unique_preds, pred_counts = np.unique(all_preds, return_counts=True)
    pred_distribution = dict(zip(unique_preds, pred_counts))
    logger.info(f"Prediction distribution: {pred_distribution}")
    
    return {
        'loss': total_loss / max(num_batches, 1),  # Avoid division by zero
        'accuracy': acc,
        'f1_score': f1,
        'confusion_matrix': cm,
        'pred_distribution': pred_distribution
    }

def train(args):
    """Main training function"""
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"face_model_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save args
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize dataset
    dataset = FaceDataset(
        processed_dir=args.data_dir,
        target=args.target,
        balance_classes=args.balance_classes,
        augment=args.augment,
        test_mode=False
    )
    
    # Create subject-aware splits
    train_indices, val_indices, test_indices = create_subject_aware_splits(
        dataset, 
        test_size=args.test_size, 
        val_size=args.val_size,
        random_state=args.seed
    )
    
    # Create data loaders
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Initialize model
    model = FaceEmotionClassifier(
        num_classes=2 if args.target in ['valence', 'arousal'] else 4,
        pretrained=True,
        frozen_layers=args.frozen_layers
    ).to(device)
    
    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    
    # Initialize optimizer with differential learning rates for backbone vs. heads
    params = [
        {'params': [p for n, p in model.named_parameters() if 'vision_model' in n], 'lr': args.learning_rate * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'vision_model' not in n]}
    ]
    
    optimizer = torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Initialize scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        min_lr=args.min_lr
    )
    
    # Initialize criterion
    criterion = nn.CrossEntropyLoss()
    
    # Initialize WandB
    if args.use_wandb and wandb_available:
        wandb.init(
            project="face-emotion-recognition",
            config=vars(args),
            name=f"face_model_{timestamp}"
        )
        wandb.watch(model)
    elif args.use_wandb and not wandb_available:
        logger.warning("WandB logging requested but wandb is not installed.")
    
    # Training loop
    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Initialize metrics dictionary
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'train_f1': [],
        'val_f1': [],
        'lr': []
    }
    
    # Mixup function
    def mixup(x, y, alpha=0.2):
        """Applies mixup to the given batch of samples and labels"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in progress_bar:
            # Get batch data
            face_features = batch['face_features'].to(device)
            labels = batch['label'].to(device)
            
            # Apply mixup with probability 0.5
            do_mixup = np.random.random() < 0.5 and epoch > 5
            if do_mixup:
                face_features, labels_a, labels_b, lam = mixup(face_features, labels, alpha=0.2)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(face_features)
            logits = outputs['logits']
            aux_logits = outputs['aux_logits']
            
            # Compute main loss
            if do_mixup:
                loss_main = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
                loss_aux = lam * criterion(aux_logits, labels_a) + (1 - lam) * criterion(aux_logits, labels_b)
            else:
                loss_main = criterion(logits, labels)
                loss_aux = criterion(aux_logits, labels)
            
            # Combine losses with auxiliary classifier
            loss = loss_main + 0.3 * loss_aux
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Update parameters
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            num_batches += 1
            
            # Get predictions
            _, preds = torch.max(logits, dim=1)
            if do_mixup:
                # For metric calculation, use the original non-mixed labels
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels_a.cpu().numpy())
            else:
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix(loss=train_loss/num_batches)
        
        # Calculate train metrics
        train_acc = accuracy_score(train_labels, train_preds)
        train_f1 = f1_score(train_labels, train_preds, average='macro')
        train_loss = train_loss / num_batches
        
        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update learning rate
        current_lr = scheduler.step(epoch)
        
        # Update metrics
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_metrics['loss'])
        metrics['train_acc'].append(train_acc)
        metrics['val_acc'].append(val_metrics['accuracy'])
        metrics['train_f1'].append(train_f1)
        metrics['val_f1'].append(val_metrics['f1_score'])
        metrics['lr'].append(current_lr)
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs}:")
        logger.info(f"  Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_score']:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        
        # Log to WandB
        if args.use_wandb and wandb_available:
            wandb.log({
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'train_accuracy': train_acc,
                'val_accuracy': val_metrics['accuracy'],
                'train_f1': train_f1,
                'val_f1': val_metrics['f1_score'],
                'learning_rate': current_lr,
                'epoch': epoch+1
            })
        
        # Check for best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': best_val_acc,
                'val_f1': val_metrics['f1_score'],
                'args': vars(args)
            }, os.path.join(output_dir, 'best_model.pt'))
            
            logger.info(f"  New best model saved with accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement for {patience_counter} epochs (best: {best_val_acc:.4f} at epoch {best_epoch+1})")
            
            # Early stopping
            if patience_counter >= args.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Save checkpoint periodically
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }, os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, os.path.join(output_dir, 'final_model.pt'))
    
    # Plot metrics
    plot_and_save(metrics, output_dir)
    
    # Test best model
    logger.info("Loading best model for testing...")
    checkpoint = torch.load(os.path.join(output_dir, 'best_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_metrics = evaluate(model, test_loader, criterion, device)
    
    logger.info(f"Test Results:")
    logger.info(f"  Loss: {test_metrics['loss']:.4f}")
    logger.info(f"  Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"  F1 Score: {test_metrics['f1_score']:.4f}")
    logger.info(f"  Confusion Matrix:\n{test_metrics['confusion_matrix']}")
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1_score': test_metrics['f1_score'],
            'confusion_matrix': test_metrics['confusion_matrix'].tolist()
        }, f, indent=2)
    
    # Log to WandB
    if args.use_wandb and wandb_available:
        wandb.log({
            'test_loss': test_metrics['loss'],
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1_score']
        })
        wandb.finish()
    
    return test_metrics, output_dir

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train face emotion classifier model")
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed", 
                        help="Path to processed data directory")
    parser.add_argument('--target', type=str, default='valence', choices=['valence', 'arousal'],
                        help="Emotion target to predict")
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size")  # Reduced batch size
    parser.add_argument('--epochs', type=int, default=200, help="Number of epochs")
    parser.add_argument('--learning_rate', type=float, default=5e-5, help="Learning rate")  # Reduced learning rate
    parser.add_argument('--weight_decay', type=float, default=2e-5, help="Weight decay")  # Increased weight decay
    parser.add_argument('--min_lr', type=float, default=5e-7, help="Minimum learning rate")
    parser.add_argument('--warmup_epochs', type=int, default=20, help="Number of warmup epochs")  # Increased warmup
    parser.add_argument('--frozen_layers', type=int, default=4, help="Number of frozen layers in backbone")  # Fewer frozen layers
    
    # Dataset arguments
    parser.add_argument('--balance_classes', action='store_true', default=True, help="Whether to balance classes")
    parser.add_argument('--augment', action='store_true', default=True, help="Whether to use augmentations")
    parser.add_argument('--test_size', type=float, default=0.15, help="Test split size")
    parser.add_argument('--val_size', type=float, default=0.15, help="Validation split size")
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--device', type=str, default='cuda', help="Device to use")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for data loading")
    parser.add_argument('--output_dir', type=str, default='outputs', help="Output directory")
    parser.add_argument('--save_every', type=int, default=30, help="Save model every N epochs")
    parser.add_argument('--patience', type=int, default=90, help="Early stopping patience")
    parser.add_argument('--use_wandb', action='store_true', default=True, help="Whether to use WandB")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    test_metrics, output_dir = train(args) 