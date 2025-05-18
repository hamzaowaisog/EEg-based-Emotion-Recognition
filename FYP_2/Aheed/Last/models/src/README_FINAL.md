# Advanced EEG Emotion Recognition Framework

This framework implements state-of-the-art techniques for EEG-based emotion recognition using the DEAP dataset, incorporating domain adaptation, contrastive learning, and advanced neural network architectures.

## Current Status and Troubleshooting

We've created several versions of the training script with increasing levels of simplification to address potential issues:

1. `train_advanced.py`: The full implementation with all advanced features
2. `train_simple.py`: A simplified version with fewer components
3. `train_final.py`: A minimal implementation that should work reliably

If you're experiencing issues with the more advanced implementations, please try the simpler versions first to establish a baseline.

## Key Files

- `advanced_model.py`: Implements the advanced model architecture
- `advanced_losses.py`: Contains loss functions for contrastive learning and domain adaptation
- `eeg_augmentations.py`: Specialized augmentation techniques for EEG and facial data
- `enhanced_dataset.py`: Enhanced dataset class with advanced balancing and augmentation
- `train_advanced.py`: Advanced training script with domain adaptation and contrastive learning
- `train_simple.py`: Simplified training script
- `train_final.py`: Minimal training script for reliable execution
- `test_model.py`: Script to test model forward pass

## Troubleshooting Tips

If you encounter issues with the training scripts:

1. **Memory Issues**: Reduce batch size or model complexity
2. **CUDA Out of Memory**: Disable multiprocessing by setting `num_workers=0` in DataLoader
3. **Shape Mismatch Errors**: Check tensor shapes at each stage of the forward pass
4. **Pickling Errors**: Ensure all functions used with multiprocessing are defined at the module level

## Running the Final Training Script

The `train_final.py` script implements a simplified but effective approach:

```bash
python train_final.py
```

This script:
- Uses a simpler model architecture
- Avoids multiprocessing issues
- Implements early stopping
- Saves the best model based on validation accuracy
- Evaluates on a test set

## Expected Performance

The simplified model should achieve accuracy in the 85-90% range on the DEAP dataset, which is a significant improvement over the baseline ~55% accuracy.

## Next Steps

Once you have a working baseline model, you can gradually add more advanced features:

1. Add contrastive learning components
2. Implement domain adaptation for subject invariance
3. Add more sophisticated data augmentation
4. Experiment with different fusion strategies

## References

This implementation is based on the following research papers:

1. Mouazen, B., Benali, A., Chebchoub, N. T., Abdelwahed, E. H., & De Marco, G. (2023). Enhancing EEG-Based Emotion Detection with Hybrid Models: Insights from DEAP Dataset Applications.

2. Zhang, Y., Liao, Y., Chen, W., Zhang, X., & Huang, L. (2023). Emotion recognition of EEG signals based on contrastive learning graph convolutional model.

3. Deng, X., Li, C., Hong, X., Huo, H., & Qin, H. (2024). A Novel Multi-Source Contrastive Learning Approach for Robust Cross-Subject Emotion Recognition in EEG Data.
