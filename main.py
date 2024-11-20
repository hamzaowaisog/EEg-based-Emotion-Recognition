from data_preparation import load_preprocessed_data
from training import train_clisa
from prediction import predict_clisa
import torch

# Paths and parameters
preprocessed_dir = "./processed"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
eeg_data = load_preprocessed_data(preprocessed_dir)

# Train model
encoder, projector = train_clisa(
    eeg_data=eeg_data,
    epochs=10,
    batch_size=32,
    learning_rate=0.001,
    device=device
)

# Save trained models
torch.save(encoder.state_dict(), "encoder.pth")
torch.save(projector.state_dict(), "projector.pth")

# Predict and evaluate
predictions = predict_clisa(encoder, eeg_data, device)
print("Predictions:", predictions)
