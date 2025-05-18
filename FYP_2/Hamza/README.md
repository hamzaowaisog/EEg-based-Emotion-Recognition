# 🧠 EEG-based Emotion Recognition - Hamza's Approach

## Overview
This directory contains the implementation of DEAP EEG-only Multi-class Classification approach for emotion recognition. By applying Multi-Source Contrastive Learning (MSCL) and innovative subject-specific mapping techniques, this approach significantly improves cross-subject generalization for EEG-based emotion recognition.

## Approach Details
The DEAP EEG-only Multi-class approach implements a novel framework that:

- Utilizes a subject-specific mapper with Squeeze-and-Excitation (SE) blocks to capture individual characteristics
- Employs a cross-subject alignment module to reduce inter-subject variability
- Applies Dynamic Weighted Focal Loss to address class imbalance
- Implements dual-stage contrastive learning for improved feature representation
- Facilitates effective domain adaptation for better generalization to unseen subjects

## Results
This approach achieved impressive performance metrics:
- **Classification Accuracy**: 
  - Subject-Dependent: 85.4%
  - Subject-Independent: 68.7%
- **Macro F1-Score**: 
  - Subject-Dependent: 83.8% 
  - Subject-Independent: 63.2%
- **Unweighted Average Recall (UAR)**: 
  - Subject-Dependent: 82.7% 
  - Subject-Independent: 61.8%
- **Improvement**: 10.5% over baseline methods

### Per-Class Accuracy (Subject-Independent)
- LVLA (Sad): 65.3%
- LVHA (Fear): 68.7%
- HVLA (Calm): 64.2%
- HVHA (Happy): 72.71%

## Key Technical Innovations

### Subject-Specific Mapper
- **Squeeze-and-Excitation Blocks**: Dynamically focuses on the most informative EEG channels for each subject
- **Frequency Attention**: Weights different frequency bands based on their relevance for emotion classification
- **Adaptive Feature Extraction**: Tailors feature extraction to each subject's unique EEG patterns

### Cross-Subject Alignment Module
- **Dynamic Domain Adaptation**: Adjusts weights for source subjects based on similarity to target
- **Adversarial Training**: Reduces domain shift between different subjects
- **Maximum Mean Discrepancy (MMD)**: Aligns feature distributions across subjects

### Dynamic Weighted Focal Loss
- **Class Weighting**: Dynamically adjusts weights based on per-class accuracy during training
- **Focusing Parameter**: Places more emphasis on hard examples
- **Label Smoothing**: Reduces overfitting to potentially noisy labels

## Model Architecture
The architecture consists of several key components:
1. **Common Feature Extractor**: Initial processing of EEG signals
2. **Subject-Specific Mapper**: Adapts features to each subject's characteristics
3. **Cross-Subject Alignment Module**: Aligns features across different subjects
4. **Classification Head**: For final emotion class prediction

## Visualization of Results
The t-SNE visualizations demonstrate the effectiveness of our approach:
- Before subject-specific mapping: Features are intermixed with poor class separation
- After subject-specific mapping: Clear clustering of emotion classes with improved separation
- After cross-subject alignment: Further refinement of class boundaries with reduced subject-specific variations

## Usage
```python
# Example usage of the multi-source contrastive learning model
from models.mscl import MultiSourceContrastiveModel
from models.train import train_model

# Initialize model
model = MultiSourceContrastiveModel(eeg_channels=32, num_classes=4)

# Train with source subjects
train_model(model, source_data, source_labels, source_subjects, epochs=100)

# Evaluate on target subject
accuracy, f1, uar = evaluate_model(model, target_data, target_labels)
```

## References
- X. Deng et al., "Multi-Source Contrastive Learning for Cross-Subject EEG Emotion Recognition," in Biomedical Signal Processing and Control, 2024.
- J. Hu et al., "Squeeze-and-Excitation Networks," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2018.
- T. Lin et al., "Focal Loss for Dense Object Detection," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017. 