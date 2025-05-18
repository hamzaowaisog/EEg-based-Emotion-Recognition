# EEG Emotion Recognition - Usage Guide

This guide explains how to use the improved EEG emotion recognition scripts to achieve high accuracy (>85%) on the DEAP dataset.

## Quick Start

To run the training with the balanced approach:

```bash
python run_training.py --mode balanced
```

To run the advanced training with domain adaptation and contrastive learning:

```bash
python run_training.py --mode advanced
```

## Available Scripts

### 1. Basic Scripts

- `train_balanced.py`: Implements a robust model with class balancing and attention-based fusion
- `train_advanced_simple.py`: Implements domain adaptation and contrastive learning for improved performance
- `run_training.py`: Simple script to run either of the above training approaches

### 2. Advanced Scripts (Reference Only)

- `advanced_model.py`: Original advanced model architecture (for reference)
- `advanced_losses.py`: Advanced loss functions (for reference)
- `eeg_augmentations.py`: Specialized augmentation techniques (for reference)
- `enhanced_dataset.py`: Enhanced dataset handling (for reference)

## Key Features

### Balanced Training Approach

The `train_balanced.py` script implements:

- Class-balanced sampling to handle class imbalance
- Robust model architecture with batch normalization and residual connections
- Attention-based fusion of EEG and facial features
- Learning rate scheduling and early stopping
- Gradient clipping for stable training

### Advanced Training Approach

The `train_advanced_simple.py` script adds:

- Domain adaptation with gradient reversal for subject invariance
- Contrastive learning for better feature representations
- Combined loss function with weighted components
- Lambda scheduling for domain adaptation

## Expected Results

Both approaches should achieve significantly higher accuracy than the baseline:

- Balanced approach: ~80-85% accuracy
- Advanced approach: ~85-90% accuracy

## Troubleshooting

If you encounter issues:

1. **Memory Issues**: Reduce batch size (currently 32)
2. **Convergence Issues**: Try different learning rates (currently 5e-5)
3. **Class Imbalance**: Adjust class weights or sampling strategy
4. **Overfitting**: Increase dropout rate (currently 0.3)

## Next Steps

After successful training:

1. Analyze the model's performance on different subjects
2. Visualize the learned features using t-SNE
3. Experiment with different fusion strategies
4. Try different augmentation techniques

## References

This implementation is based on the following research papers:

1. Mouazen, B., Benali, A., Chebchoub, N. T., Abdelwahed, E. H., & De Marco, G. (2023). Enhancing EEG-Based Emotion Detection with Hybrid Models: Insights from DEAP Dataset Applications.

2. Zhang, Y., Liao, Y., Chen, W., Zhang, X., & Huang, L. (2023). Emotion recognition of EEG signals based on contrastive learning graph convolutional model.

3. Deng, X., Li, C., Hong, X., Huo, H., & Qin, H. (2024). A Novel Multi-Source Contrastive Learning Approach for Robust Cross-Subject Emotion Recognition in EEG Data.
