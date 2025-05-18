# Advanced EEG Emotion Recognition Framework

This framework implements state-of-the-art techniques for EEG-based emotion recognition using the DEAP dataset, incorporating domain adaptation, contrastive learning, and graph convolutional networks.

## Key Features

- **Graph Convolutional Networks** for EEG data processing
- **Domain Adaptation** for subject-invariant feature learning
- **Multi-Source Contrastive Learning** for improved feature representations
- **Advanced Data Augmentation** specifically designed for EEG and facial data
- **Enhanced Dataset Handling** with subject-aware balancing and stratification
- **Comprehensive Evaluation** with detailed visualizations and metrics

## Files Overview

- `advanced_model.py`: Implements the advanced model architecture with GCNs and domain adaptation
- `advanced_losses.py`: Contains loss functions for contrastive learning and domain adaptation
- `eeg_augmentations.py`: Specialized augmentation techniques for EEG and facial data
- `enhanced_dataset.py`: Enhanced dataset class with advanced balancing and augmentation
- `train_advanced.py`: Advanced training script with domain adaptation and contrastive learning
- `evaluate_advanced.py`: Comprehensive evaluation and visualization script

## How to Use

### 1. Training the Advanced Model

```bash
python train_advanced.py --data_dir "path/to/processed/data" --output_dir "./outputs/advanced" --use_wandb
```

Additional training options:
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 150)
- `--lr`: Learning rate (default: 1e-4)
- `--weight_decay`: Weight decay for regularization (default: 1e-5)
- `--patience`: Early stopping patience (default: 20)
- `--lambda_schedule`: Schedule for domain adaptation lambda (choices: 'linear', 'exp', 'constant')
- `--mixup_alpha`: Mixup alpha parameter (default: 0.2)
- `--label_smoothing`: Label smoothing factor (default: 0.1)

### 2. Evaluating the Model

```bash
python evaluate_advanced.py --model_path "path/to/model.pth" --output_dir "./evaluation_results" --split test
```

Evaluation options:
- `--model_path`: Path to the trained model (required)
- `--data_dir`: Directory with processed data
- `--output_dir`: Output directory for results (default: './evaluation_results')
- `--batch_size`: Batch size for evaluation (default: 32)
- `--split`: Dataset split to evaluate on (choices: 'train', 'val', 'test')

## Implementation Details

### Advanced Model Architecture

The model architecture combines several state-of-the-art techniques:

1. **EEG Graph Convolutional Network**: Processes EEG data as a graph, capturing spatial relationships between channels
2. **Domain Adaptation**: Uses gradient reversal layers to learn subject-invariant features
3. **Cross-Modal Attention**: Fuses EEG and facial features using multi-head attention
4. **Contrastive Learning**: Projects features into a shared embedding space for improved discrimination

### Loss Functions

The framework uses a combination of loss functions:

1. **Classification Loss**: Cross-entropy with focal weighting for imbalanced data
2. **Contrastive Loss**: Supervised contrastive loss for improved feature learning
3. **Domain Adversarial Loss**: Encourages subject-invariant features
4. **Cross-Modal Consistency Loss**: Ensures consistency between EEG and facial representations

### Data Augmentation

Specialized augmentation techniques for EEG data:

1. **Temporal Shifts**: Small shifts in time domain
2. **Spectral Transforms**: Frequency domain augmentations
3. **Channel Dropout**: Randomly masking channels
4. **Magnitude Warping**: Amplitude variations
5. **Cross-Modal Augmentations**: Correlated augmentations across modalities

## Expected Performance

This advanced framework is designed to achieve significantly higher accuracy (>85%) compared to traditional approaches by addressing the key challenges in EEG-based emotion recognition:

1. **Subject Variability**: Domain adaptation reduces subject-specific biases
2. **Limited Data**: Contrastive learning and augmentation improve generalization
3. **Noisy Signals**: Graph convolution and specialized augmentations handle noise effectively
4. **Multimodal Fusion**: Cross-attention effectively combines EEG and facial information

## References

This implementation is based on the following research papers:

1. Mouazen, B., Benali, A., Chebchoub, N. T., Abdelwahed, E. H., & De Marco, G. (2023). Enhancing EEG-Based Emotion Detection with Hybrid Models: Insights from DEAP Dataset Applications.

2. Zhang, Y., Liao, Y., Chen, W., Zhang, X., & Huang, L. (2023). Emotion recognition of EEG signals based on contrastive learning graph convolutional model.

3. Deng, X., Li, C., Hong, X., Huo, H., & Qin, H. (2024). A Novel Multi-Source Contrastive Learning Approach for Robust Cross-Subject Emotion Recognition in EEG Data.
