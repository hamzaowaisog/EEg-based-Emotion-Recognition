import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import logging
from datetime import datetime
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import random
import torch.nn.functional as F
import gc  # For garbage collection

# Import your dataset and models
from dataset import DEAPDataset
from simple_model import SimpleEmotionClassifier
from modified_losses import SimplifiedLoss

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training_simplified.log')
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

def evaluate(model, loader, criterion=None, device="cuda"):
    """Evaluation function"""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    steps = 0

    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)

            logits, eeg_proj, face_proj = model(eeg, face)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            # Calculate loss if criterion provided
            if criterion:
                loss = criterion(logits, eeg_proj, face_proj, labels)
                total_loss += loss.item()
                steps += 1

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    # Calculate average loss
    avg_loss = total_loss / steps if steps > 0 else 0

    return avg_loss, acc, f1, conf_matrix

def train_simplified(
    model,
    optimizer,
    train_dataset,
    device="cuda",
    epochs=100,
    val_dataset=None,
    output_dir="./outputs_simplified",
    batch_size=32,
    early_stopping_patience=15,  # Increased patience
    scheduler=None,
    alpha=0.1,  # Reduced auxiliary loss weights
    beta=0.1,
    class_weights=None,
    use_wandb=True
):
    """Simplified training function with reduced complexity"""
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset) if val_dataset else 0} samples")
    
    # Initialize loss function with reduced weights for auxiliary losses
    criterion = SimplifiedLoss(
        alpha=alpha,
        beta=beta,
        class_weights=class_weights
    )
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device == "cuda" else False
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True if device == "cuda" else False
        )

    # Initialize tracking variables
    best_val_acc = 0.0
    best_epoch = 0
    early_stopping_counter = 0
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }

    # Initialize wandb
    if use_wandb:
        wandb.init(project="emotion-classification-simplified", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "alpha": alpha,
            "beta": beta,
            "model": "SimpleEmotionClassifier"
        })

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        train_steps = 0

        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)

            optimizer.zero_grad()

            logits, eeg_proj, face_proj = model(eeg, face)

            loss = criterion(logits, eeg_proj, face_proj, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item()
            train_steps += 1

            # Clean up to prevent memory leaks
            del eeg, face, labels, logits, eeg_proj, face_proj, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Calculate average training loss
        avg_train_loss = epoch_loss / train_steps if train_steps > 0 else 0
        history['train_loss'].append(avg_train_loss)
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f}")

        # Validation phase
        if val_loader:
            val_loss, val_acc, val_f1, conf_matrix = evaluate(model, val_loader, criterion, device)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_f1'].append(val_f1)

            logger.info(f"Validation | Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            logger.info(f"Confusion Matrix:\n{conf_matrix}")

            # Log to wandb
            if use_wandb:
                wandb.log({
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'val_f1': val_f1,
                    'lr': optimizer.param_groups[0]['lr']
                })

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                early_stopping_counter = 0

                model_path = os.path.join(output_dir, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1': val_f1
                }, model_path)
                logger.info(f"Saved best model with validation accuracy: {val_acc:.4f}")
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break

        # Step the learning rate scheduler if provided
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

    # Close wandb
    if use_wandb:
        wandb.finish()

    # Log final results
    logger.info(f"Training completed. Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")

    # Load best model before returning
    if val_loader:
        best_model_path = os.path.join(output_dir, "best_model.pth")
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, history

def main():
    # Set seeds for reproducibility
    set_seed(42)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize simplified model
    model = SimpleEmotionClassifier().to(device)

    # Initialize optimizer with reduced learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-6
    )

    # Learning rate scheduler with longer warmup
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=150,
        steps_per_epoch=29,  # Adjust based on your batch size and dataset size
        pct_start=0.3,  # Longer warmup
        div_factor=25,
        final_div_factor=1000
    )

    # Load dataset with reduced augmentation
    full_dataset = DEAPDataset(
        processed_dir=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
        target='valence',
        apply_augmentation=False,  # Disable augmentation initially
        balance_classes=True
    )

    class_weights = full_dataset.get_class_weights().to(device)
    
    # Split train/val/test (70/15/15)
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # Create splits using indices
    train_indices, val_indices, test_indices = random_split(
        range(len(full_dataset)),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create subsets
    train_set = Subset(full_dataset, train_indices)
    val_set = Subset(full_dataset, val_indices)
    test_set = Subset(full_dataset, test_indices)

    # Run training with simplified approach
    output_dir = './outputs/simplified_' + datetime.now().strftime("%Y%m%d_%H%M%S")
    trained_model, history = train_simplified(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        device=device,
        epochs=150,
        batch_size=32,
        early_stopping_patience=20,  # Increased patience
        scheduler=scheduler,
        alpha=0.1,  # Reduced auxiliary loss weights
        beta=0.1,
        class_weights=class_weights,
        output_dir=output_dir,
        use_wandb=True
    )

    # Evaluate on test set
    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Clear memory before evaluation
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    test_loss, test_acc, test_f1, test_conf_matrix = evaluate(
        trained_model,
        test_loader,
        criterion=SimplifiedLoss(alpha=0.1, beta=0.1, class_weights=class_weights),
        device=device
    )

    logger.info(f"Test Results | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_conf_matrix}")

    # Save test results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_conf_matrix}\n")

if __name__ == "__main__":
    main()
