import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import random
import torch.nn.functional as F
import gc  # For garbage collection
import argparse

# Import custom modules
from enhanced_dataset import EnhancedDEAPDataset, create_stratified_subject_splits
from advanced_model import MultiSourceContrastiveModel
from advanced_losses import AdvancedLoss, MultiSourceContrastiveLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_advanced.log')
    ]
)
logger = logging.getLogger(__name__)

def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Worker initialization function for DataLoader
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def evaluate(model, loader, criterion=None, device="cuda"):
    """Enhanced evaluation function"""
    model.eval()
    all_preds = []
    all_labels = []
    all_subject_ids = []
    total_loss = 0.0
    steps = 0

    loss_components = {
        'total_loss': 0.0,
        'cls_loss': 0.0,
        'cont_loss': 0.0,
        'domain_loss': 0.0,
        'consistency_loss': 0.0
    }

    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            subject_ids = batch['subject_id'].to(device)

            # Forward pass
            outputs = model(eeg, face, subject_ids, lambda_val=0.0)  # No gradient reversal during evaluation
            logits = outputs['logits']

            # Get predictions
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            # Calculate loss if criterion provided
            if criterion:
                loss, loss_dict = criterion(outputs, labels, subject_ids)
                total_loss += loss.item()

                # Accumulate loss components
                for k, v in loss_dict.items():
                    if k in loss_components:
                        loss_components[k] += v.item()

                steps += 1

            # Collect predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_subject_ids.extend(subject_ids.cpu().numpy())

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_subject_ids = np.array(all_subject_ids)

    # Overall metrics
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Per-subject metrics
    subject_metrics = {}
    for subject_id in np.unique(all_subject_ids):
        mask = all_subject_ids == subject_id
        if np.sum(mask) > 0:
            subj_preds = all_preds[mask]
            subj_labels = all_labels[mask]
            subj_acc = accuracy_score(subj_labels, subj_preds)
            subj_f1 = f1_score(subj_labels, subj_preds, average='macro', zero_division=0)
            subject_metrics[int(subject_id)] = {
                'accuracy': subj_acc,
                'f1': subj_f1,
                'samples': np.sum(mask)
            }

    # Calculate average loss components
    if steps > 0:
        for k in loss_components:
            loss_components[k] /= steps

    # Return comprehensive metrics
    metrics = {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': conf_matrix,
        'loss': total_loss / steps if steps > 0 else 0,
        'loss_components': loss_components,
        'subject_metrics': subject_metrics
    }

    return metrics

def train_advanced(
    model,
    optimizer,
    train_dataset,
    val_dataset,
    device="cuda",
    epochs=150,
    batch_size=32,
    early_stopping_patience=20,
    scheduler=None,
    output_dir="./outputs_advanced",
    use_wandb=True,
    save_every=10,
    lambda_schedule='linear',
    mixup_alpha=0.2,
    label_smoothing=0.1
):
    """Advanced training function with domain adaptation and contrastive learning"""
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset)} samples")

    # Initialize loss function
    criterion = AdvancedLoss(
        num_classes=2,
        num_subjects=32,
        temperature=0.5,
        alpha=0.5,  # Weight for contrastive loss
        beta=0.3,   # Weight for domain adaptation loss
        gamma=0.2,  # Weight for cross-modal consistency loss
        delta=0.1,  # Weight for focal loss
        class_weights=None  # Will be set dynamically
    )

    # Setup data loaders with worker initialization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == "cuda" else False,
        worker_init_fn=worker_init_fn,
        drop_last=True  # Drop last batch to ensure consistent batch size
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )

    # Initialize tracking variables
    best_val_acc = 0.0
    best_val_f1 = 0.0
    best_epoch = 0
    early_stopping_counter = 0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': [],
        'lr': []
    }

    # Initialize wandb
    if use_wandb:
        wandb.init(project="emotion-classification-advanced", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "model": "MultiSourceContrastiveModel",
            "mixup_alpha": mixup_alpha,
            "label_smoothing": label_smoothing,
            "lambda_schedule": lambda_schedule
        })

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        train_steps = 0

        # Calculate lambda for domain adaptation (gradually increase)
        if lambda_schedule == 'linear':
            lambda_val = min(1.0, (epoch / (epochs * 0.5)))
        elif lambda_schedule == 'exp':
            lambda_val = 2.0 / (1.0 + np.exp(-10.0 * epoch / epochs)) - 1.0
        else:  # constant
            lambda_val = 1.0

        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            subject_ids = batch['subject_id'].to(device)

            # Apply mixup augmentation with probability
            if mixup_alpha > 0 and random.random() < 0.5:
                # Generate mixup parameters
                alpha = mixup_alpha
                lam = np.random.beta(alpha, alpha)

                # Create shuffled indices
                batch_size = eeg.size(0)
                index = torch.randperm(batch_size).to(device)

                # Mixup data
                mixed_eeg = lam * eeg + (1 - lam) * eeg[index]
                mixed_face = lam * face + (1 - lam) * face[index]

                # Forward pass with mixed data
                outputs = model(mixed_eeg, mixed_face, subject_ids, lambda_val=lambda_val)

                # Compute loss with label smoothing
                if label_smoothing > 0:
                    # Create soft labels
                    y_a = F.one_hot(labels, num_classes=2).float()
                    y_b = F.one_hot(labels[index], num_classes=2).float()

                    # Apply label smoothing
                    y_a = y_a * (1 - label_smoothing) + label_smoothing / 2
                    y_b = y_b * (1 - label_smoothing) + label_smoothing / 2

                    # Mixed soft labels
                    mixed_labels = lam * y_a + (1 - lam) * y_b

                    # Custom loss calculation for soft labels
                    logits = outputs['logits']
                    cls_loss = -torch.sum(F.log_softmax(logits, dim=1) * mixed_labels, dim=1).mean()

                    # Other loss components
                    _, loss_dict = criterion(outputs, labels, subject_ids)
                    loss_dict['cls_loss'] = cls_loss

                    # Combine losses
                    loss = cls_loss + criterion.alpha * loss_dict['cont_loss'] - criterion.beta * loss_dict['domain_loss'] + criterion.gamma * loss_dict['consistency_loss']
                else:
                    # Standard mixup loss
                    loss1, _ = criterion(outputs, labels, subject_ids)
                    loss2, _ = criterion(outputs, labels[index], subject_ids)
                    loss = lam * loss1 + (1 - lam) * loss2
            else:
                # Standard forward pass
                outputs = model(eeg, face, subject_ids, lambda_val=lambda_val)
                loss, loss_dict = criterion(outputs, labels, subject_ids)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Calculate training accuracy
            with torch.no_grad():
                logits = outputs['logits']
                preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
                acc = (preds == labels).float().mean().item()

            # Update metrics
            epoch_loss += loss.item()
            epoch_acc += acc
            train_steps += 1

            # Clean up to prevent memory leaks
            del eeg, face, labels, subject_ids, outputs, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate average training metrics
        avg_train_loss = epoch_loss / train_steps if train_steps > 0 else 0
        avg_train_acc = epoch_acc / train_steps if train_steps > 0 else 0

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_acc)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f}")

        # Validation phase
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Update history
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_f1'].append(val_metrics['f1'])

        # Log validation metrics
        logger.info(f"Validation | Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f}")
        logger.info(f"Confusion Matrix:\n{val_metrics['confusion_matrix']}")

        # Log to wandb
        if use_wandb:
            wandb_log = {
                'train_loss': avg_train_loss,
                'train_acc': avg_train_acc,
                'val_loss': val_metrics['loss'],
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_precision': val_metrics['precision'],
                'val_recall': val_metrics['recall'],
                'lr': optimizer.param_groups[0]['lr'],
                'lambda': lambda_val,
                'epoch': epoch
            }

            # Add loss components
            for k, v in val_metrics['loss_components'].items():
                wandb_log[f'val_{k}'] = v

            wandb.log(wandb_log)

        # Save best model (by accuracy)
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            early_stopping_counter = 0

            model_path = os.path.join(output_dir, "best_model_acc.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_metrics': val_metrics
            }, model_path)
            logger.info(f"Saved best accuracy model: {val_metrics['accuracy']:.4f}")

        # Save best model (by F1 score)
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']

            model_path = os.path.join(output_dir, "best_model_f1.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_metrics': val_metrics
            }, model_path)
            logger.info(f"Saved best F1 model: {val_metrics['f1']:.4f}")
        else:
            early_stopping_counter += 1

        # Early stopping check
        if early_stopping_counter >= early_stopping_patience:
            logger.info(f"Early stopping triggered after {epoch+1} epochs")
            break

        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
                'history': history
            }, checkpoint_path)

        # Step the learning rate scheduler if provided
        if scheduler:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_metrics['loss'])
            else:
                scheduler.step()

    # Close wandb
    if use_wandb:
        wandb.finish()

    # Log final results
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")

    # Load best model before returning
    best_model_path = os.path.join(output_dir, "best_model_acc.pth")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model, history

def plot_training_history(history, output_dir):
    """Plot training history and save figures"""
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'loss.png'))
    plt.close()

    # Plot accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Training and Validation Metrics')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'metrics.png'))
    plt.close()

    # Plot learning rate
    plt.figure(figsize=(10, 5))
    plt.plot(history['lr'])
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.savefig(os.path.join(plots_dir, 'lr.png'))
    plt.close()

def main(args):
    # Set seeds for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = f'./outputs/advanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset with advanced augmentations
    logger.info("Loading dataset...")
    full_dataset = EnhancedDEAPDataset(
        processed_dir=args.data_dir,
        target='valence',
        apply_augmentation=True,
        augmentation_prob=0.7,
        balance_classes=True,
        balance_method='smote',
        balance_subjects=True,
        cache_features=True
    )

    # Create stratified subject-aware splits
    logger.info("Creating dataset splits...")
    train_set, val_set, test_set = create_stratified_subject_splits(
        full_dataset,
        test_size=0.15,
        val_size=0.15,
        random_state=args.seed
    )

    # Initialize model
    logger.info("Initializing model...")
    model = MultiSourceContrastiveModel(
        eeg_input_dim=32,
        face_input_dim=768,
        hidden_dim=128,
        output_dim=2,
        num_subjects=32,
        temperature=0.5,
        dropout=0.5
    ).to(device)

    # Initialize optimizer with different learning rates for different components
    param_groups = [
        {'params': model.eeg_encoder.parameters(), 'lr': args.lr * 0.5},
        {'params': model.face_encoder.parameters(), 'lr': args.lr},
        {'params': model.eeg_domain_adapter.parameters(), 'lr': args.lr * 0.1},
        {'params': model.face_domain_adapter.parameters(), 'lr': args.lr * 0.1},
        {'params': model.cross_attn.parameters(), 'lr': args.lr * 0.5},
        {'params': model.classifier.parameters(), 'lr': args.lr},
        {'params': model.eeg_projection.parameters(), 'lr': args.lr * 0.5},
        {'params': model.face_projection.parameters(), 'lr': args.lr * 0.5}
    ]

    optimizer = optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g['lr'] * 5 for g in param_groups],
        epochs=args.epochs,
        steps_per_epoch=len(train_set) // args.batch_size + 1,
        pct_start=0.3,
        div_factor=25,
        final_div_factor=1000
    )

    # Train model
    logger.info("Starting training...")
    trained_model, history = train_advanced(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        early_stopping_patience=args.patience,
        scheduler=scheduler,
        output_dir=output_dir,
        use_wandb=args.use_wandb,
        save_every=10,
        lambda_schedule=args.lambda_schedule,
        mixup_alpha=args.mixup_alpha,
        label_smoothing=args.label_smoothing
    )

    # Plot training history
    plot_training_history(history, output_dir)

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )

    # Clear memory before evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Create criterion for evaluation
    criterion = AdvancedLoss(
        num_classes=2,
        num_subjects=32,
        temperature=0.5,
        alpha=0.5,
        beta=0.3,
        gamma=0.2,
        delta=0.1
    )

    test_metrics = evaluate(trained_model, test_loader, criterion, device)

    # Log test results
    logger.info(f"Test Results | Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}")

    # Save test results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Test F1 Score: {test_metrics['f1']:.4f}\n")
        f.write(f"Test Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"Test Recall: {test_metrics['recall']:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_metrics['confusion_matrix']}\n\n")

        # Write per-subject metrics
        f.write("Per-Subject Metrics:\n")
        for subject_id, metrics in sorted(test_metrics['subject_metrics'].items()):
            f.write(f"Subject {subject_id}: Acc={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}, Samples={metrics['samples']}\n")

    # Log to wandb
    if args.use_wandb:
        wandb.log({
            'test_accuracy': test_metrics['accuracy'],
            'test_f1': test_metrics['f1'],
            'test_precision': test_metrics['precision'],
            'test_recall': test_metrics['recall']
        })

    logger.info(f"Training and evaluation completed. Results saved to {output_dir}")
    return test_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced EEG Emotion Recognition Training")

    # Dataset parameters
    parser.add_argument('--data_dir', type=str, default=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
                        help='Directory with processed data')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=20, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    # Advanced parameters
    parser.add_argument('--lambda_schedule', type=str, default='linear', choices=['linear', 'exp', 'constant'],
                        help='Schedule for domain adaptation lambda')
    parser.add_argument('--mixup_alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--label_smoothing', type=float, default=0.1, help='Label smoothing factor')

    # Output parameters
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases for logging')

    args = parser.parse_args()

    main(args)
