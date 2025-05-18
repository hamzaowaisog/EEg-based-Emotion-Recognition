import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import gc

# Import your dataset and models
from dataset import DEAPDataset
from advanced_model import MultiSourceContrastiveModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

def evaluate(model, loader, device="cuda"):
    """Evaluation function"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            
            # Forward pass
            outputs = model(eeg, face)
            logits = outputs['logits']
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)
    
    return acc, f1, conf_matrix

def main():
    # Set seeds for reproducibility
    set_seed(42)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = f'./outputs/simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = DEAPDataset(
        processed_dir=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
        target='valence',
        apply_augmentation=True,
        balance_classes=True
    )
    
    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size
    
    train_set, val_set, test_set = random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    logger.info(f"Dataset split: {len(train_set)} train, {len(val_set)} val, {len(test_set)} test")
    
    # Create data loaders
    train_loader = DataLoader(
        train_set,
        batch_size=32,
        shuffle=True,
        num_workers=0  # No multiprocessing to avoid issues
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=32,
        shuffle=False,
        num_workers=0
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
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-5
    )
    
    # Initialize loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info("Starting training...")
    epochs = 50
    best_val_acc = 0.0
    patience = 10
    counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            
            # Forward pass
            outputs = model(eeg, face)
            logits = outputs['logits']
            
            # Calculate loss
            loss = criterion(logits, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        val_acc, val_f1, conf_matrix = evaluate(model, val_loader, device)
        
        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(output_dir, "best_model.pth"))
            
            logger.info(f"Saved best model with validation accuracy: {val_acc:.4f}")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    checkpoint = torch.load(os.path.join(output_dir, "best_model.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test phase
    test_acc, test_f1, test_conf_matrix = evaluate(model, test_loader, device)
    
    logger.info(f"Test Results | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_conf_matrix}")
    
    # Save test results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_conf_matrix}\n")
    
    logger.info(f"Training and evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
