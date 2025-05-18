import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import logging
from dataset import DEAPDataset
from advanced_model import MultiSourceContrastiveModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load a small subset of the dataset
    logger.info("Loading dataset...")
    dataset = DEAPDataset(
        processed_dir=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
        target='valence',
        apply_augmentation=False,
        balance_classes=False
    )
    
    # Create a small dataloader
    loader = DataLoader(
        dataset,
        batch_size=4,
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
    
    # Test forward pass
    logger.info("Testing forward pass...")
    model.eval()
    
    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            
            # Forward pass
            outputs = model(eeg, face)
            logits = outputs['logits']
            
            # Print shapes
            logger.info(f"EEG shape: {eeg.shape}")
            logger.info(f"Face shape: {face.shape}")
            logger.info(f"Logits shape: {logits.shape}")
            
            # Print predictions
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)
            
            logger.info(f"Predictions: {preds.cpu().numpy()}")
            logger.info(f"Labels: {labels.cpu().numpy()}")
            
            # Only process one batch
            break
    
    logger.info("Model test completed successfully!")

if __name__ == "__main__":
    main()
