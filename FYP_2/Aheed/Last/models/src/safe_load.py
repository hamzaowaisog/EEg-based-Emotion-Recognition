"""
Utility module for safely loading PyTorch models with weights_only=True
"""

import torch
import numpy as np
from torch.serialization import add_safe_globals

def setup_safe_loading():
    """
    Add numpy.core.multiarray.scalar to the safe globals list
    to allow loading models with weights_only=True
    """
    # Get the scalar type from numpy
    scalar = np.array(0).item().__class__
    
    # Add it to PyTorch's safe globals list
    add_safe_globals([scalar])
    
    return True

def safe_load(path, device=None, weights_only=True):
    """
    Safely load a PyTorch model with weights_only=True
    
    Args:
        path (str): Path to the model checkpoint
        device (torch.device, optional): Device to load the model to
        weights_only (bool, optional): Whether to load only weights. Defaults to True.
        
    Returns:
        dict: The loaded checkpoint
    """
    # Setup safe loading
    setup_safe_loading()
    
    # Load the model
    if device:
        return torch.load(path, map_location=device, weights_only=weights_only)
    else:
        return torch.load(path, weights_only=weights_only)
