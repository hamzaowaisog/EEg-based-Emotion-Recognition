import pickle
import numpy as np
import os

# Only loading first 22 participants as the DEAP video dataset has 22 participants
eeg_data_dict = {}
for i in range(1, 23):  
    file_path = f"data_preprocessed_python/s{i:02d}.dat"
    with open(file_path, 'rb') as f:
        eeg_data_dict[f"s{i:02d}"] = pickle.load(f, encoding='latin1')

# Emotion categories based on trial numbers
emotion_labels = ["Excited_Happy"] * 10 + ["Calm_Content"] * 10 + ["Sad_Bored"] * 10 + ["Angry_Fearful"] * 10

# Dictionary to store EEG features by emotion class
eeg_features_by_emotion = {key: [] for key in set(emotion_labels)}

def differential_entropy(signal):
    """Compute Differential Entropy (DE) for an EEG trial"""
    return np.log(np.var(signal)) / 2  

# Process EEG data
for participant, data in eeg_data_dict.items():
    eeg_trials = data["data"]  # Shape: (40, 40, 8064)

    for trial_idx in range(40):
        emotion_label = emotion_labels[trial_idx]  # Directly assign based on trial index
        de_features = np.array([differential_entropy(eeg_trials[trial_idx, ch]) for ch in range(40)])
        eeg_features_by_emotion[emotion_label].append(de_features)

# Debugging: Verify the number of trials per category
for label, features in eeg_features_by_emotion.items():
    print(f"{label}: {len(features)} trials (Expected: 220)")

# Save extracted EEG features
output_dir = "output_eeg_features_based_on_emotion"
os.makedirs(output_dir, exist_ok=True)

for label, features in eeg_features_by_emotion.items():
    np.save(f"{output_dir}/{label}.npy", np.array(features))
