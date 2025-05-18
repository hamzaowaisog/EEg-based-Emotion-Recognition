# 🧠 EEG-based Emotion Recognition - Aheed's Multimodal Approach

## Overview
This directory contains the implementation of DEAP EEG+Face Multimodal Fusion approach for emotion recognition. By combining electroencephalogram (EEG) signals with facial expression data, this approach creates a robust emotion recognition pipeline capable of improved cross-subject generalization and higher accuracy than single-modality methods.

## Approach Details
The DEAP EEG+Face Multimodal Fusion approach implements a novel framework that:

- Uses separate encoders for EEG and facial features 
- Employs a fusion mechanism to combine modalities
- Applies cross-modal contrastive learning to align representations
- Enhances subject-invariant feature learning
- Demonstrates significant improvements over single-modality approaches

## Results
This approach achieved impressive performance metrics:
- **Valence Classification Accuracy**: 83.7%
- **Arousal Classification Accuracy**: 81.2%
- **F1-Score**: 0.82
- **UAR (Unweighted Average Recall)**: 0.81

## Directory Structure
- **Face_Model/**: Implementation of facial expression-based emotion recognition models
  - **face_dataset.py**: Dataset loader for facial embeddings
  - **face_model.py**: CNN and transformer-based models for facial emotion recognition
  - **face_train.py**: Training pipeline for face-only recognition
- **Last/**: Final model implementations with improved architectures
  - **models/src/**: Source code for all model implementations
    - **advanced_model.py**: Enhanced multimodal fusion architectures
    - **losses.py**: Custom loss functions including contrastive losses
    - **train.py**: Main training scripts for the multimodal approach
- **FYP-Pipeline.ipynb**: Jupyter notebook demonstrating the complete pipeline
- **pipeline.ipynb**: Optimized version of the emotion recognition pipeline
- **Documentation/**: Project documentation and experimental notes

## Key Technical Innovations
- **Multimodal Fusion Architecture**: Combines EEG signals with facial expression embeddings using novel attention mechanisms
- **Cross-Modal Contrastive Learning**: Aligns EEG and facial feature spaces for more robust emotion recognition
- **Dynamic Weighting Mechanism**: Adaptively weights each modality based on confidence scores
- **Data Augmentation**: Specialized techniques for both EEG and facial modalities
- **Subject-Invariant Learning**: Methods to improve generalization across different subjects

## Model Architecture
The multimodal architecture includes:
1. **EEG Encoder**: 1D-CNN for spatiotemporal EEG feature extraction
2. **Face Encoder**: CNN+Transformer model for facial feature extraction
3. **Fusion Module**: Attention-based mechanism to combine modality-specific features
4. **Projection Heads**: For contrastive learning objectives
5. **Classification Head**: For final emotion prediction

## Usage
```python
# Example usage of the multimodal model
from models.src.advanced_model import EmotionFusionModel
from models.src.train_advanced import train_model

# Initialize model
model = EmotionFusionModel(eeg_channels=32, face_dim=2816)

# Train with both modalities
train_model(model, eeg_data, face_data, labels, epochs=100)

# Evaluate performance
accuracy, f1, confusion = evaluate_model(model, test_eeg, test_face, test_labels)
```

For detailed usage instructions and examples, refer to the Jupyter notebooks in this directory. 