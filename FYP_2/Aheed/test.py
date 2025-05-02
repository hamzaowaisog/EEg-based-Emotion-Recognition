import numpy as np

# Load all emotion categories
for emotion in ["Excited_Happy", "Calm_Content", "Sad_Bored", "Angry_Fearful"]:
    features = np.load(f"output_eeg_features_based_on_emotion/{emotion}.npy", allow_pickle=True)
    print(f"{emotion}: {features.shape}")
# Save extracted EEG features
