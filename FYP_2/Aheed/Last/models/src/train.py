import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import logging
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
import random

from dataset import DEAPDataset
from model import EmotionClassifier
from losses import HybridLoss

def set_seed(seed=42):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(output_dir):
    """Setup logging configuration"""
    log_file = os.path.join(output_dir, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

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
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.plot(history['val_f1'], label='Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Validation Metrics')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, 'metrics.png'))
    plt.close()

def train(
    model, 
    optimizer, 
    train_dataset, 
    device="cuda", 
    epochs=100, 
    val_dataset=None,
    output_dir="./outputs",
    batch_size=64,
    early_stopping_patience=10,
    scheduler=None,
    alpha=0.3,
    beta=0.3,
    use_wandb=True,
    save_every=10
):
    """Enhanced training function with validation support and additional features"""
    # Setup output directory and logging
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logging(output_dir)
    logger.info(f"Training with {len(train_dataset)} samples, validating with {len(val_dataset) if val_dataset else 0} samples")
    logger.info(f"Device: {device}, Batch size: {batch_size}, Epochs: {epochs}")
    
    # Initialize loss function
    criterion = HybridLoss(alpha=alpha, beta=beta)
    logger.info(f"Using HybridLoss with alpha={alpha}, beta={beta}")
    
    # Setup data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if device == "cuda" else False
    )
    
    # Validation setup
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=4,
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
        wandb.init(project="emotion-classification", config={
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer.__class__.__name__,
            "learning_rate": optimizer.param_groups[0]['lr'],
            "alpha": alpha,
            "beta": beta,
        })
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        train_steps = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            
            optimizer.zero_grad()
            
            logits, projections = model(eeg, face)
            eeg_proj = model.eeg_encoder(eeg)
            face_proj = model.face_encoder(face)
            
            loss = criterion((logits, projections), eeg_proj, face_proj, labels)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            train_steps += 1
        
        avg_train_loss = epoch_loss / train_steps
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
        
        # Save checkpoint periodically
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_acc': history['val_acc'][-1] if val_loader else None,
            }, checkpoint_path)
        
        # Step the learning rate scheduler if provided
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
            logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Plot and save training history
    plot_training_history(history, output_dir)
    
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

def evaluate(model, loader, criterion=None, device="cuda"):
    """Enhanced evaluation function with loss and confusion matrix"""
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
            
            logits, projections = model(eeg, face)
            preds = torch.argmax(logits, dim=1)
            
            # Calculate loss if criterion provided
            if criterion:
                eeg_proj = model.eeg_encoder(eeg)
                face_proj = model.face_encoder(face)
                loss = criterion((logits, projections), eeg_proj, face_proj, labels)
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

def main():
    # Set seeds for reproducibility
    set_seed(42)
    
    # Configuration
    config = {
        'batch_size': 64,
        'learning_rate': 1e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'early_stopping_patience': 10,
        'alpha': 0.3,  # HybridLoss parameter
        'beta': 0.3,   # HybridLoss parameter
        'output_dir': './outputs/run_' + datetime.now().strftime("%Y%m%d_%H%M%S"),
        'use_wandb': True,  # Set to True to use wandb logging
    }
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model
    model = EmotionClassifier().to(device)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'], 
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Load datasets
    full_dataset = DEAPDataset(processed_dir=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed")
    
    # Split train/val/test (70/15/15)
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_set, val_set, test_set = torch.utils.data.random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Run training
    trained_model, history = train(
        model=model,
        optimizer=optimizer,
        train_dataset=train_set,
        val_dataset=val_set,
        device=device,
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        early_stopping_patience=config['early_stopping_patience'],
        scheduler=scheduler,
        alpha=config['alpha'],
        beta=config['beta'],
        output_dir=config['output_dir'],
        use_wandb=config['use_wandb']
    )
    
    # Evaluate on test set
    test_loader = DataLoader(test_set, batch_size=config['batch_size'], shuffle=False, num_workers=4)
    test_loss, test_acc, test_f1, test_conf_matrix = evaluate(
        trained_model, 
        test_loader, 
        criterion=HybridLoss(alpha=config['alpha'], beta=config['beta']),
        device=device
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Test Results | Loss: {test_loss:.4f} | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_conf_matrix}")
    
    # Save test results
    with open(os.path.join(config['output_dir'], 'test_results.txt'), 'w') as f:
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_conf_matrix}\n")

if __name__ == "__main__":
    from datetime import datetime
    main()