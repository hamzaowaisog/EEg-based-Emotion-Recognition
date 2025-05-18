import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPVisionModel, CLIPImageProcessor
import logging

logger = logging.getLogger(__name__)

class FaceEmotionClassifier(nn.Module):
    """SOTA Face Emotion Classifier using CLIP ViT-L/14 backbone"""
    
    def __init__(self, num_classes=2, pretrained=True, frozen_layers=4):
        super().__init__()
        
        # Initialize CLIP Vision Model (ViT-L/14)
        self.backbone_name = "openai/clip-vit-large-patch14"
        self.vision_model = CLIPVisionModel.from_pretrained(self.backbone_name)
        self.hidden_dim = self.vision_model.config.hidden_size  # 1024 for ViT-L/14
        
        # Freeze parts of the backbone if specified
        if frozen_layers > 0:
            self._freeze_layers(frozen_layers)
        
        # MLP head for emotion classification with improved regularization
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.4),  # Increased dropout
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),  # Increased dropout
            nn.Linear(256, num_classes)
        )
        
        # Projection head for contrastive learning and late fusion
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),  # Added dropout
            nn.Linear(512, 256)
        )
        
        # Additional auxiliary classifier for regularization
        self.aux_classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        logger.info(f"Initialized FaceEmotionClassifier with {self.backbone_name}")
        logger.info(f"Feature dimension: {self.hidden_dim}")
        
    def _freeze_layers(self, num_layers):
        """Freeze the first n layers of the vision model"""
        # Freeze encoder.embed_tokens and first num_layers blocks
        blocks_to_freeze = ["embeddings"] + [f"encoder.layers.{i}" for i in range(num_layers)]
        
        for name, param in self.vision_model.named_parameters():
            for block in blocks_to_freeze:
                if block in name:
                    param.requires_grad = False
                    break
        
        logger.info(f"Froze {num_layers} layers of the vision backbone")
    
    def get_features(self, features_or_pixels):
        """Extract features from the vision model or use pre-extracted features"""
        # Check if input is pre-extracted features or raw pixel values
        if len(features_or_pixels.shape) == 2:
            # Input is already extracted features
            return features_or_pixels
        else:
            # Input is pixel values, pass through vision model
            outputs = self.vision_model(pixel_values=features_or_pixels)
            # Return the [CLS] token features (first token)
            return outputs.last_hidden_state[:, 0, :]
    
    def forward(self, features_or_pixels):
        """Forward pass through the model"""
        features = self.get_features(features_or_pixels)
        
        # Detect and adapt to the actual input feature dimension
        actual_dim = features.shape[1]
        if actual_dim != self.hidden_dim:
            # Only log this warning once to avoid cluttering logs
            if not hasattr(self, 'dimension_warning_logged'):
                logger.warning(f"Input feature dimension ({actual_dim}) doesn't match expected dimension ({self.hidden_dim})")
                logger.warning(f"Reconstructing classifier and projection heads to match input dimension")
                self.dimension_warning_logged = True
            
            # Reconstruct classifier with actual dimension if needed
            device = features.device
            if not hasattr(self, 'adapted_to_dimension') or self.adapted_to_dimension != actual_dim:
                self.hidden_dim = actual_dim
                self.classifier = nn.Sequential(
                    nn.Linear(actual_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(0.4),
                    nn.Linear(512, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, self.classifier[-1].out_features)
                ).to(device)
                
                # Reconstruct projection head
                self.projection = nn.Sequential(
                    nn.Linear(actual_dim, 512),
                    nn.LayerNorm(512),
                    nn.GELU(),
                    nn.Dropout(0.2),
                    nn.Linear(512, 256)
                ).to(device)
                
                # Reconstruct auxiliary classifier
                self.aux_classifier = nn.Sequential(
                    nn.Linear(actual_dim, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.3),
                    nn.Linear(256, self.classifier[-1].out_features)
                ).to(device)
                
                self.adapted_to_dimension = actual_dim
            
        # Main classification path
        logits = self.classifier(features)
        
        # Auxiliary classification path
        aux_logits = self.aux_classifier(features)
        
        # Get embeddings for late fusion
        embeddings = self.projection(features)
        embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
        
        return {
            'logits': logits,
            'aux_logits': aux_logits,
            'embeddings': embeddings,
            'embeddings_normalized': embeddings_normalized,
            'features': features
        }
    
    def get_embeddings(self, features_or_pixels):
        """Utility method to get only embeddings for fusion"""
        with torch.no_grad():
            features = self.get_features(features_or_pixels)
            embeddings = self.projection(features)
            return F.normalize(embeddings, p=2, dim=1)


class FusionModel(nn.Module):
    """Late fusion model combining face and EEG embeddings"""
    
    def __init__(self, embedding_dim=256, num_classes=2):
        super().__init__()
        
        # Combined dimension is 2*embedding_dim (concatenated)
        self.classifier = nn.Sequential(
            nn.Linear(2*embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, face_embeddings, eeg_embeddings):
        """Forward pass with both modalities"""
        # Concatenate embeddings from both modalities
        combined = torch.cat([face_embeddings, eeg_embeddings], dim=1)
        
        # Get classification prediction
        logits = self.classifier(combined)
        
        return logits 