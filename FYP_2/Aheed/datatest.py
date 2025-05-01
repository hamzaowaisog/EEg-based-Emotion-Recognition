import pickle
import numpy as np

# Load a sample participant (e.g., s01)
file_path = "data_preprocessed_python/s01.dat"
with open(file_path, 'rb') as f:
    subject_data = pickle.load(f, encoding='latin1')

# Extract EEG data and labels
eeg_data = subject_data['data']  # Shape: (40, 40, 8064)
labels = subject_data['labels']  # Shape: (40, 4)

# Print structure
print(f"EEG Data Shape: {eeg_data.shape}")
print(f"Labels Shape: {labels.shape}")

# Example: Extract EEG for first trial (video 1)
trial_1_eeg = eeg_data[0]  # Shape: (40, 8064)
trial_1_labels = labels[0]  # [Valence, Arousal, Dominance, Liking]
print(f"Trial 1 Labels: {trial_1_labels}")
