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

    def supervised_contrastive(self, projections, labels):
        projections = F.normalize(projections, dim=1)
        sim_matrix = torch.mm(projections, projections.T) / self.temp
        
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        logits = sim_matrix - torch.log(torch.exp(sim_matrix).sum(1, keepdim=True))
        
        return -(mask * logits).sum() / mask.sum()

    def cross_modal_contrastive(self, eeg_proj, face_proj):
        eeg_proj = F.normalize(eeg_proj, dim=1)
        face_proj = F.normalize(face_proj, dim=1)
        logits = torch.mm(eeg_proj, face_proj.T) / self.temp
        labels = torch.arange(eeg_proj.size(0), device=eeg_proj.device)
        return F.cross_entropy(logits, labels)

    def forward(self, outputs, eeg_proj, face_proj, labels):
        logits, projections = outputs
        ce_loss = self.ce(logits, labels)
        supcon_loss = self.supervised_contrastive(projections, labels)
        crossmodal_loss = self.cross_modal_contrastive(eeg_proj, face_proj)
        return ce_loss + self.alpha*supcon_loss + self.beta*crossmodal_loss