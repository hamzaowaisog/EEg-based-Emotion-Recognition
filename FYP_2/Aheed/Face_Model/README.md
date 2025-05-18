# EEG-based Emotion Recognition with CLIP Face Model

This project implements a SOTA face emotion recognition model using the CLIP ViT-L/14 architecture for late fusion with an existing EEG-based emotion model. The implementation allows for standalone face model training and evaluation, as well as late fusion with EEG features.

## Project Structure

```
FYP_2/Aheed/Face_Model/
├── face_model.py       # Face model architecture using CLIP ViT-L/14
├── face_dataset.py     # Dataset class for loading face data
├── face_train.py       # Training script for face model
├── fusion.py           # Late fusion of EEG and face models
├── README.md           # This file
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- transformers
- scikit-learn
- matplotlib
- wandb (optional for logging)
- tqdm

## Face Model Training

The face model is built on CLIP ViT-L/14 and can be trained using the provided script:

```bash
python face_train.py --data_dir "path/to/processed/data" --output_dir "outputs/face_model" --target "valence"
```

### Key Arguments

- `--data_dir`: Path to processed data directory
- `--target`: Emotion target to predict ("valence" or "arousal")
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 100)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--frozen_layers`: Number of frozen layers in backbone (default: 8)
- `--use_wandb`: Whether to use WandB for logging (default: True)

## Late Fusion with EEG Model

After training the face model, you can perform late fusion with the existing EEG model:

```bash
python fusion.py --data_dir "path/to/processed/data" --face_model_path "outputs/face_model/best_model.pt" --output_dir "outputs/fusion"
```

### Key Arguments

- `--data_dir`: Path to processed data directory
- `--face_model_path`: Path to pretrained face model
- `--target`: Emotion target to predict ("valence" or "arousal")
- `--epochs`: Number of epochs for fusion model (default: 100)
- `--learning_rate`: Learning rate for fusion model (default: 1e-3)

## Implementation Details

### Face Model Architecture

The face model uses CLIP ViT-L/14 as the backbone feature extractor:

1. CLIP ViT-L/14 backbone extracts features from facial images
2. A projection head transforms these features for late fusion
3. A classification head produces binary valence/arousal predictions

### Fusion Strategy

The late fusion strategy combines embeddings from both modalities:

1. Extract embeddings from the trained face model
2. Use EEG features from the processed data
3. Concatenate both embeddings and pass through a simple MLP classifier

### Dataset Organization

The dataset organization follows the structure of the processed data:

```
processed/
├── s01/               # Subject 1
│   ├── trial01/       # Trial 1
│   │   ├── eeg.npy    # EEG features
│   │   ├── face.npy   # Face features
│   │   └── metadata.json  # Trial metadata
│   ├── trial02/
│   └── ...
├── s02/
└── ...
```

## Expected Results

With proper training, the face model should achieve:
- ~70-75% accuracy on the valence classification task
- ~65-70% accuracy on the arousal classification task

The late fusion model should improve results by:
- +5-10% increase in accuracy over individual modalities
- Improved F1 score and class balance

## Citation

If you use this code, please cite our work:

```
@article{eeg-emotion-recognition,
  title={Enhancing EEG-based Emotion Recognition with Late Fusion of CLIP Face Models},
  author={Your Name},
  journal={Your Journal},
  year={2023}
}
``` 