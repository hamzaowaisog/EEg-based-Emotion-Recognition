import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleEmotionClassifier(nn.Module):
    """Simplified emotion classifier for baseline comparison"""
    def __init__(self, 
                 eeg_input_dim=32, 
                 face_input_dim=768, 
                 hidden_dim=128, 
                 output_dim=2,
                 dropout=0.5):
        super().__init__()
        
        # EEG encoder (simpler)
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Face encoder (simpler)
        self.face_encoder = nn.Sequential(
            nn.Linear(face_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple fusion (concatenation + MLP)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, eeg, face):
        # Encode each modality
        eeg_features = self.eeg_encoder(eeg)
        face_features = self.face_encoder(face)
        
        # Concatenate features
        combined = torch.cat([eeg_features, face_features], dim=1)
        
        # Classification
        logits = self.fusion(combined)
        
        # For compatibility with existing code
        return logits, eeg_features, face_features
