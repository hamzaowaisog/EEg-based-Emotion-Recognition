import numpy as np

# Load extracted facial features
facial_features_by_emotion = {
    "Excited_Happy": np.load("Pytorch_Retinaface-master/output_facial_features/Excited_Happy_features.npy", allow_pickle=True),
    "Calm_Content": np.load("Pytorch_Retinaface-master/output_facial_features/Calm_Content_features.npy", allow_pickle=True),
    "Sad_Bored": np.load("Pytorch_Retinaface-master/output_facial_features/Sad_Bored_features.npy", allow_pickle=True),
    "Angry_Fearful": np.load("Pytorch_Retinaface-master/output_facial_features/Angry_Fearful_features.npy", allow_pickle=True)
}

# Load extracted EEG features
eeg_features_by_emotion = {
    "Excited_Happy": np.load("output_eeg_features_based_on_emotion/Excited_Happy.npy", allow_pickle=True),
    "Calm_Content": np.load("output_eeg_features_based_on_emotion/Calm_Content.npy", allow_pickle=True),
    "Sad_Bored": np.load("output_eeg_features_based_on_emotion/Sad_Bored.npy", allow_pickle=True),
    "Angry_Fearful": np.load("output_eeg_features_based_on_emotion/Angry_Fearful.npy", allow_pickle=True)
}

# Ensure shape consistency
for emotion in eeg_features_by_emotion:
    eeg_data = eeg_features_by_emotion[emotion]  # Shape: (N, 40)
    face_data = facial_features_by_emotion[emotion]  # Shape: (N, 512)

    # Concatenate EEG and Facial Features
    fused_features = np.concatenate((eeg_data, np.squeeze(face_data)), axis=1)
    print(f"{emotion} Fused Shape: {fused_features.shape}")

    # Save fused features
    np.save(f"fused_{emotion}.npy", fused_features)
