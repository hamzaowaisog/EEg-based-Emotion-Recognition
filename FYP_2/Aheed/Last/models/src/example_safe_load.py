"""
Example script demonstrating how to use the safe_load utility
"""

import os
import torch
import logging
from safe_load import safe_load, setup_safe_globals

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model_example(model_path, device=None):
    """
    Example function to load a model using the safe_load utility
    
    Args:
        model_path (str): Path to the model checkpoint
        device (torch.device, optional): Device to load the model to
        
    Returns:
        model: The loaded model
    """
    logger.info(f"Loading model from {model_path}")
    
    try:
        # Method 1: Using safe_load utility
        checkpoint = safe_load(model_path, device=device)
        logger.info("Successfully loaded model using safe_load utility")
        
        # Method 2: Setting up safe globals and using torch.load directly
        setup_safe_globals()
        checkpoint_direct = torch.load(model_path, weights_only=True)
        logger.info("Successfully loaded model using setup_safe_globals + torch.load")
        
        # Method 3: Without weights_only (fallback)
        checkpoint_fallback = torch.load(model_path)
        logger.info("Successfully loaded model without weights_only")
        
        return checkpoint
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    model_dir = "./outputs/balanced_20250502_155347"  # Replace with your actual output directory
    model_path = os.path.join(model_dir, "best_model_acc.pth")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_model_example(model_path, device=device)
    
    if checkpoint:
        logger.info(f"Model was trained for {checkpoint.get('epoch', 'unknown')} epochs")
        logger.info(f"Validation accuracy: {checkpoint.get('val_acc', 'unknown')}")
        logger.info(f"Validation F1 score: {checkpoint.get('val_f1', 'unknown')}")
