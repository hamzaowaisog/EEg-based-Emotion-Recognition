# 🧠 EEG-based Emotion Recognition - Haseeb's Approach

## Overview
This directory contains the implementation of SEED EEG-only Multi-class Classification approach for emotion recognition. By combining Prototype Contrastive Learning with Maximum Mean Discrepancy (MMD) Loss, this approach achieves superior noise robustness and cross-subject generalization on the SEED dataset.

## Approach Details
The SEED EEG-only Multi-class approach implements an advanced framework that:

- Leverages prototype-based contrastive learning for more stable and discriminative feature representations
- Employs MMD Loss for effective domain adaptation across different subjects
- Combines multiple loss functions for optimal performance
- Provides enhanced robustness to noisy labels
- Maintains high accuracy in subject-independent evaluation scenarios

## Results
This approach achieved state-of-the-art performance:
- **Classification Accuracy**: 86.48% (3-class emotion classification)
- **Weighted F1-Score**: 86.56%
- **Improvement**: Significant performance gain over standard cross-entropy and other loss combinations

### Ablation Study Results
The effectiveness of our combined loss approach is demonstrated by the ablation study:
- CE Loss only: 75.2%
- CE + SupCon Loss: 79.8%
- CE + L_con2 Loss: 82.3%
- CE + MMD Loss: 80.9%
- **CE + L_con2 + MMD Loss**: 86.48%

## Key Technical Innovations

### Prototype Contrastive Loss
- **Class Prototypes**: Maintains momentum-updated class centroids in embedding space
- **Prototype-based Learning**: Encourages samples to align with their class prototypes
- **Noise Robustness**: Less sensitive to label noise than standard contrastive learning
- **Efficient Training**: More stable convergence compared to pair-based contrastive approaches

### Maximum Mean Discrepancy (MMD) Loss
- **Distribution Alignment**: Reduces the discrepancy between source and target domain distributions
- **Multi-kernel Implementation**: Uses multiple Gaussian kernels with different bandwidths
- **Cross-subject Adaptation**: Facilitates knowledge transfer across different subjects
- **Non-adversarial Approach**: More stable than adversarial domain adaptation methods

### Feature Engineering
- **Differential Entropy (DE)**: Extracts powerful features from different frequency bands
- **Asymmetry Features**: Incorporates DASM and RASM features to capture hemispheric asymmetry
- **Multi-band Processing**: Extracts features from delta, theta, alpha, beta, and gamma bands

## Model Architecture
The architecture consists of several key components:
1. **Feature Extractor**: Processes DE and asymmetry features
2. **Encoder Network**: Maps features to a latent representation space
3. **Projection Head**: For contrastive learning objectives
4. **Prototype Layer**: Maintains class prototypes for prototype-based learning
5. **Classification Head**: For three-class emotion prediction (negative, neutral, positive)

## Visualization of Results
The t-SNE visualization of the learned representations shows:
- Clear separation between the three emotion classes (negative, neutral, positive)
- Minimal inter-subject variability within each emotion class
- Compact and well-defined clusters for each emotion category

## Usage
```python
# Example usage of the prototype contrastive learning model
from models.prototype import PrototypeContrastiveModel
from models.train import train_model

# Initialize model
model = PrototypeContrastiveModel(
    input_dim=310,  # DE features (5 bands × 62 channels)
    proj_dim=128,
    num_classes=3
)

# Train with combined losses
train_model(
    model, 
    train_data, train_labels, 
    lambda_proto=0.5,  # Weight for prototype contrastive loss
    lambda_mmd=0.3,    # Weight for MMD loss
    epochs=100
)

# Evaluate model
accuracy, f1 = evaluate_model(model, test_data, test_labels)
```

## References
- X. Shen et al., "Contrastive Learning of Subject-Invariant EEG Representations for Cross-Subject Emotion Recognition," in IEEE Transactions on Affective Computing, 2022.
- J. Snell et al., "Prototypical Networks for Few-shot Learning," in Advances in Neural Information Processing Systems, 2017.
- A. Gretton et al., "A Kernel Two-Sample Test," Journal of Machine Learning Research, 2012.
- W. L. Zheng et al., "Investigating Critical Frequency Bands and Channels for EEG-based Emotion Recognition with Deep Neural Networks," in IEEE Transactions on Autonomous Mental Development, 2015. 