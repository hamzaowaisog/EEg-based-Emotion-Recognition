import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SimplifiedLoss(nn.Module):
    """Simplified loss function focusing primarily on classification"""
    def __init__(self, class_weights=None, alpha=0.1, beta=0.1):
        super().__init__()
        self.alpha = alpha  # Reduced weight for contrastive loss
        self.beta = beta    # Reduced weight for consistency loss
        self.class_weights = class_weights
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
    def forward(self, logits, eeg_proj, face_proj, targets):
        """Calculate primarily classification loss with minimal auxiliary losses"""
        # Classification loss (with class weighting)
        cls_loss = self.ce_loss(logits, targets)
        
        # Simple L2 normalization for projections
        eeg_norm = F.normalize(eeg_proj, dim=1)
        face_norm = F.normalize(face_proj, dim=1)
        
        # Simple contrastive loss (cosine similarity for same-class samples)
        batch_size = eeg_norm.size(0)
        if batch_size > 1:
            # Create mask for same-class pairs
            target_matrix = targets.unsqueeze(1) == targets.unsqueeze(0)
            mask = target_matrix.float() - torch.eye(batch_size, device=targets.device)
            mask = mask.clamp(min=0)  # Only keep same-class pairs (excluding self)
            
            # Calculate similarities
            sim_matrix = torch.matmul(eeg_norm, face_norm.t())
            
            # Contrastive loss: maximize similarity for same-class pairs
            pos_pairs = mask * sim_matrix
            contrastive_loss = -torch.sum(pos_pairs) / (torch.sum(mask) + 1e-6)
        else:
            contrastive_loss = torch.tensor(0.0, device=targets.device)
        
        # Simple consistency loss
        consistency_loss = 1.0 - F.cosine_similarity(eeg_norm, face_norm, dim=1).mean()
        
        # Combined loss with reduced weights for auxiliary losses
        total_loss = cls_loss + self.alpha * contrastive_loss + self.beta * consistency_loss
        
        return total_loss
