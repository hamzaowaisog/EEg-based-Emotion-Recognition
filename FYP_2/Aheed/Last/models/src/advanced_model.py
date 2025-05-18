import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GraphConvLayer(nn.Module):
    """Graph Convolutional Layer for EEG data"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        # x: [batch_size, num_nodes, in_features]
        # adj: [batch_size, num_nodes, num_nodes]
        support = torch.matmul(x, self.weight)  # [batch_size, num_nodes, out_features]
        output = torch.matmul(adj, support)  # [batch_size, num_nodes, out_features]
        return output + self.bias

class EEGGraphConvNet(nn.Module):
    """Graph Convolutional Network for EEG data"""
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=128, dropout=0.5):
        super().__init__()
        self.input_dim = input_dim

        # Use simpler architecture to avoid shape issues
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        # x: [batch_size, input_dim]
        return self.encoder(x)

class DomainAdaptiveEncoder(nn.Module):
    """Domain-adaptive encoder with gradient reversal for subject invariance"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_subjects=32, alpha=1.0):
        super().__init__()
        self.alpha = alpha

        # Feature encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )

        # Domain classifier (subject classifier)
        self.domain_classifier = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_subjects)
        )

    def forward(self, x, lambda_val=1.0):
        # Forward pass through encoder
        features = self.encoder(x)

        # Gradient reversal for domain adaptation
        reverse_features = GradientReversalFunction.apply(features, lambda_val * self.alpha)

        # Domain classification
        domain_pred = self.domain_classifier(reverse_features)

        return features, domain_pred

class GradientReversalFunction(torch.autograd.Function):
    """Gradient Reversal Layer for domain adaptation"""
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class MultiSourceContrastiveModel(nn.Module):
    """Advanced model with multi-source contrastive learning and domain adaptation"""
    def __init__(self,
                 eeg_input_dim=32,
                 face_input_dim=768,
                 hidden_dim=128,
                 output_dim=2,
                 num_subjects=32,
                 temperature=0.5,
                 dropout=0.5):
        super().__init__()

        # EEG encoder with graph convolution
        self.eeg_encoder = EEGGraphConvNet(
            input_dim=eeg_input_dim,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )

        # Face encoder
        self.face_encoder = nn.Sequential(
            nn.Linear(face_input_dim, hidden_dim*2),
            nn.BatchNorm1d(hidden_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Domain adaptation for EEG
        self.eeg_domain_adapter = DomainAdaptiveEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim//2,
            output_dim=hidden_dim,
            num_subjects=num_subjects
        )

        # Domain adaptation for face
        self.face_domain_adapter = DomainAdaptiveEncoder(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim//2,
            output_dim=hidden_dim,
            num_subjects=num_subjects
        )

        # Cross-modal attention fusion
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

        # Projection heads for contrastive learning
        self.eeg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.face_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Temperature parameter for contrastive loss
        self.temperature = temperature

    def forward(self, eeg, face, subject_ids=None, lambda_val=1.0):
        # Extract initial features
        eeg_features = self.eeg_encoder(eeg)
        face_features = self.face_encoder(face)

        # Apply domain adaptation
        eeg_adapted, eeg_domain_pred = self.eeg_domain_adapter(eeg_features, lambda_val)
        face_adapted, face_domain_pred = self.face_domain_adapter(face_features, lambda_val)

        # Simplified fusion - just concatenate and use a linear layer
        fused_features = torch.cat([eeg_adapted, face_adapted], dim=1)
        fused_features = F.relu(fused_features)
        fused_features = torch.mean(fused_features.view(fused_features.size(0), 2, -1), dim=1)

        # Classification
        logits = self.classifier(fused_features)

        # Projections for contrastive learning
        eeg_proj = self.eeg_projection(eeg_adapted)
        face_proj = self.face_projection(face_adapted)

        # Return all outputs needed for training
        return {
            'logits': logits,
            'eeg_proj': eeg_proj,
            'face_proj': face_proj,
            'eeg_domain_pred': eeg_domain_pred,
            'face_domain_pred': face_domain_pred,
            'fused_features': fused_features
        }

    def get_embeddings(self, eeg, face):
        """Get embeddings for visualization or analysis"""
        with torch.no_grad():
            eeg_features = self.eeg_encoder(eeg)
            face_features = self.face_encoder(face)

            eeg_adapted, _ = self.eeg_domain_adapter(eeg_features, 0.0)  # No gradient reversal
            face_adapted, _ = self.face_domain_adapter(face_features, 0.0)

            # Simplified fusion
            fused_features = torch.cat([eeg_adapted, face_adapted], dim=1)
            fused_features = F.relu(fused_features)
            fused_features = torch.mean(fused_features.view(fused_features.size(0), 2, -1), dim=1)

            return {
                'eeg_features': eeg_adapted,
                'face_features': face_adapted,
                'fused_features': fused_features
            }
