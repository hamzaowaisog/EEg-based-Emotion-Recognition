import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Learning Loss
    Adapted from: https://arxiv.org/abs/2004.11362
    """
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """
        Args:
            features: hidden vector of shape [batch_size, n_views, ...].
            labels: ground truth of shape [batch_size].
            mask: contrastive mask of shape [batch_size, batch_size], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = features.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [batch_size, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class MultiSourceContrastiveLoss(nn.Module):
    """
    Multi-Source Contrastive Loss with Domain Adaptation
    Inspired by the paper: "A Novel Multi-Source Contrastive Learning Approach for Robust Cross-Subject Emotion Recognition in EEG Data"
    """
    def __init__(self, temperature=0.5, alpha=0.5, beta=0.3, gamma=0.2, class_weights=None):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for contrastive loss
        self.beta = beta    # Weight for domain adaptation loss
        self.gamma = gamma  # Weight for cross-modal consistency loss
        
        # Classification loss
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        
        # Contrastive loss
        self.contrastive_loss = SupConLoss(temperature=temperature)
        
        # Domain classification loss
        self.domain_loss = nn.CrossEntropyLoss()
        
    def forward(self, model_outputs, targets, subject_ids):
        """
        Calculate multi-source contrastive loss with domain adaptation
        
        Args:
            model_outputs: Dictionary containing model outputs
                - logits: Classification logits [batch_size, num_classes]
                - eeg_proj: EEG projections [batch_size, hidden_dim]
                - face_proj: Face projections [batch_size, hidden_dim]
                - eeg_domain_pred: EEG domain predictions [batch_size, num_subjects]
                - face_domain_pred: Face domain predictions [batch_size, num_subjects]
            targets: Target labels [batch_size]
            subject_ids: Subject IDs [batch_size]
        """
        # Extract model outputs
        logits = model_outputs['logits']
        eeg_proj = model_outputs['eeg_proj']
        face_proj = model_outputs['face_proj']
        eeg_domain_pred = model_outputs['eeg_domain_pred']
        face_domain_pred = model_outputs['face_domain_pred']
        
        # 1. Classification loss
        cls_loss = self.ce_loss(logits, targets)
        
        # 2. Contrastive loss
        # Prepare features for contrastive loss
        batch_size = eeg_proj.size(0)
        
        # Normalize projections
        eeg_proj_norm = F.normalize(eeg_proj, dim=1)
        face_proj_norm = F.normalize(face_proj, dim=1)
        
        # Stack features for contrastive loss
        features = torch.stack([eeg_proj_norm, face_proj_norm], dim=1)  # [batch_size, 2, hidden_dim]
        
        # Calculate contrastive loss
        cont_loss = self.contrastive_loss(features, labels=targets)
        
        # 3. Domain adversarial loss
        domain_loss_eeg = self.domain_loss(eeg_domain_pred, subject_ids)
        domain_loss_face = self.domain_loss(face_domain_pred, subject_ids)
        domain_loss = (domain_loss_eeg + domain_loss_face) / 2.0
        
        # 4. Cross-modal consistency loss
        # Encourage consistency between modalities for the same sample
        consistency_loss = 1.0 - F.cosine_similarity(eeg_proj_norm, face_proj_norm, dim=1).mean()
        
        # Combine all losses
        total_loss = cls_loss + self.alpha * cont_loss - self.beta * domain_loss + self.gamma * consistency_loss
        
        # Return individual losses for monitoring
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'cont_loss': cont_loss,
            'domain_loss': domain_loss,
            'consistency_loss': consistency_loss
        }
        
        return total_loss, loss_dict

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    Adapted from: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        
    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class AdvancedLoss(nn.Module):
    """
    Advanced loss function combining multiple loss components
    """
    def __init__(self, num_classes=2, num_subjects=32, temperature=0.5, 
                 alpha=0.5, beta=0.3, gamma=0.2, delta=0.1, class_weights=None):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha  # Weight for contrastive loss
        self.beta = beta    # Weight for domain adaptation loss
        self.gamma = gamma  # Weight for cross-modal consistency loss
        self.delta = delta  # Weight for focal loss
        
        # Classification losses
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.focal_loss = FocalLoss(gamma=2.0, alpha=class_weights)
        
        # Contrastive loss
        self.contrastive_loss = SupConLoss(temperature=temperature)
        
        # Domain classification loss
        self.domain_loss = nn.CrossEntropyLoss()
        
    def forward(self, model_outputs, targets, subject_ids):
        """
        Calculate advanced loss with multiple components
        """
        # Extract model outputs
        logits = model_outputs['logits']
        eeg_proj = model_outputs['eeg_proj']
        face_proj = model_outputs['face_proj']
        eeg_domain_pred = model_outputs['eeg_domain_pred']
        face_domain_pred = model_outputs['face_domain_pred']
        
        # 1. Classification loss (combination of CE and Focal)
        ce_loss = self.ce_loss(logits, targets)
        focal_loss = self.focal_loss(logits, targets)
        cls_loss = (1 - self.delta) * ce_loss + self.delta * focal_loss
        
        # 2. Contrastive loss
        # Normalize projections
        eeg_proj_norm = F.normalize(eeg_proj, dim=1)
        face_proj_norm = F.normalize(face_proj, dim=1)
        
        # Stack features for contrastive loss
        features = torch.stack([eeg_proj_norm, face_proj_norm], dim=1)
        
        # Calculate contrastive loss
        cont_loss = self.contrastive_loss(features, labels=targets)
        
        # 3. Domain adversarial loss
        domain_loss_eeg = self.domain_loss(eeg_domain_pred, subject_ids)
        domain_loss_face = self.domain_loss(face_domain_pred, subject_ids)
        domain_loss = (domain_loss_eeg + domain_loss_face) / 2.0
        
        # 4. Cross-modal consistency loss
        consistency_loss = 1.0 - F.cosine_similarity(eeg_proj_norm, face_proj_norm, dim=1).mean()
        
        # Combine all losses
        total_loss = cls_loss + self.alpha * cont_loss - self.beta * domain_loss + self.gamma * consistency_loss
        
        # Return individual losses for monitoring
        loss_dict = {
            'total_loss': total_loss,
            'cls_loss': cls_loss,
            'ce_loss': ce_loss,
            'focal_loss': focal_loss,
            'cont_loss': cont_loss,
            'domain_loss': domain_loss,
            'consistency_loss': consistency_loss
        }
        
        return total_loss, loss_dict
