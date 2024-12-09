from torcheeg.datasets import SEEDDataset

# Load the cached dataset
dataset = SEEDDataset(
    io_path='C:/Users/tahir/Documents/EEg-based-Emotion-Recognition/.torcheeg/datasets_1733174610032_5iJyS',
    online_transform=None,
    label_transform=None,
    num_worker=6
)

# Inspect the first sample
sample = dataset[0]
eeg_data, label = sample[0], sample[1]

print(f"EEG Data Shape: {eeg_data.shape}")  # Expected: [n_channels, n_timepoints]
print(f"Label: {label}")
