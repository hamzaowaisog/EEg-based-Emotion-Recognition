import torch
from model import EmotionClassifier

def test_shapes():
    # Initialize model
    model = EmotionClassifier()
    
    # Create test batch (batch_size=4)
    test_eeg = torch.randn(4, 32)  # [batch, eeg_features]
    test_face = torch.randn(4, 768)  # [batch, face_features]
    
    print("=== Starting Shape Check ===")
    _ = model(test_eeg, test_face)
    print("=== Shape Check Completed ===")

if __name__ == "__main__":
    test_shapes()