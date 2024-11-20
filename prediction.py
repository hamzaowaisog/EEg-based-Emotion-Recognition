import numpy as np
import torch

def compute_de(encoded_features):
    """Compute Differential Entropy (DE) features from encoded features."""
    return np.log(np.var(encoded_features, axis=-1) + 1e-6)  # Add small value for numerical stability

def predict_clisa(encoder, eeg_data, device):
    """Perform prediction using the trained CLISA encoder."""
    predictions = {}
    encoder.eval()

    for subject, trials_by_label in eeg_data.items():
        predictions[subject] = {}
        for label, trials in trials_by_label.items():
            for trial in trials:
                trial = torch.tensor(trial, dtype=torch.float32).unsqueeze(0).to(device)
                encoded_features = encoder(trial).detach().cpu().numpy()
                h_de = compute_de(encoded_features)

                # Mocked classifier: Use the mean DE feature to predict
                predicted_label = np.argmax(h_de.mean())
                predictions[subject][label] = predictions[subject].get(label, []) + [predicted_label]

    return predictions
