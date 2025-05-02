import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEncoder(nn.Module):
    def __init__(self, input_dim=32, latent_dim=128):  # 32 channels
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, latent_dim))
    
    def forward(self, x):
        return self.net(x)

class FaceEncoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, latent_dim))
    
    def forward(self, x):
        return self.net(x)

class EmotionClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.eeg_encoder = EEGEncoder()
        self.face_encoder = FaceEncoder()
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5))
        
        # Projection heads for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64))
        
        # Final classifier
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, eeg, face):
        eeg_z = self.eeg_encoder(eeg)
        face_z = self.face_encoder(face)
        
        # Mid-level fusion
        fused = torch.cat([eeg_z, face_z], dim=1)
        fused = self.fusion(fused)
        
        # Projections for contrastive loss
        projections = self.projection(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits, F.normalize(projections, dim=1)