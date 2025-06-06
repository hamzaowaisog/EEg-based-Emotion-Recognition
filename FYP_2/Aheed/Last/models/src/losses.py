import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, temp=0.07):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.alpha = alpha  # SupCon weight
        self.beta = beta    # Cross-modal weight
        self.temp = temp
        self.eps = 1e-8  # Small constant for numerical stability

    def supervised_contrastive(self, projections, labels):
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(projections, projections.T) / self.temp
        
        # Create mask for positive pairs
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        
        # Remove self-similarities
        mask = mask - torch.eye(mask.size(0), device=mask.device)
        
        # Compute logits
        logits = sim_matrix - torch.log(torch.exp(sim_matrix).sum(1, keepdim=True) + self.eps)
        
        # Compute loss only for positive pairs
        pos_pairs = mask * logits
        num_pos_pairs = mask.sum() + self.eps
        
        return -pos_pairs.sum() / num_pos_pairs

    def cross_modal_contrastive(self, eeg_proj, face_proj):
        # Normalize projections
        eeg_proj = F.normalize(eeg_proj, dim=1)
        face_proj = F.normalize(face_proj, dim=1)
        
        # Compute similarity matrix
        logits = torch.mm(eeg_proj, face_proj.T) / self.temp
        
        # Create labels (diagonal elements are positive pairs)
        labels = torch.arange(eeg_proj.size(0), device=eeg_proj.device)
        
        # Compute cross-entropy loss
        return F.cross_entropy(logits, labels)

    def forward(self, outputs, eeg_proj, face_proj, labels):
        logits, projections = outputs
        
        # Classification loss
        ce_loss = self.ce(logits, labels)
        
        # Supervised contrastive loss
        supcon_loss = self.supervised_contrastive(projections, labels)
        
        # Cross-modal contrastive loss
        crossmodal_loss = self.cross_modal_contrastive(eeg_proj, face_proj)
        
        # Combine losses with weights
        total_loss = ce_loss + self.alpha * supcon_loss + self.beta * crossmodal_loss
        
        return total_loss