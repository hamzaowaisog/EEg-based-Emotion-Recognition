import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import numpy as np
import logging
from datetime import datetime
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define a more robust model
class RobustModel(nn.Module):
    def __init__(self, eeg_dim=32, face_dim=768, hidden_dim=128, output_dim=2):
        super().__init__()

        # EEG encoder with batch normalization and residual connections
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Face encoder with batch normalization and residual connections
        self.face_encoder = nn.Sequential(
            nn.Linear(face_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Attention-based fusion
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )

    def attention_fusion(self, eeg_features, face_features):
        # Self-attention mechanism
        q_eeg = self.query(eeg_features)
        k_face = self.key(face_features)
        v_face = self.value(face_features)

        # Compute attention scores
        scores = torch.matmul(q_eeg.unsqueeze(1), k_face.unsqueeze(2)) / np.sqrt(k_face.size(1))
        attention = F.softmax(scores, dim=-1)

        # Apply attention
        context = torch.matmul(attention, v_face.unsqueeze(2)).squeeze(2)

        # Concatenate with original features
        fused = torch.cat([eeg_features, context], dim=1)
        return fused

    def forward(self, eeg, face):
        eeg_features = self.eeg_encoder(eeg)
        face_features = self.face_encoder(face)

        # Simple concatenation as fallback
        fused = torch.cat([eeg_features, face_features], dim=1)

        # Classification
        logits = self.classifier(fused)

        return logits

# Define a dataset class with better handling
class BalancedDataset:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.samples = []
        self.class_counts = {}

        # Load data
        logger.info(f"Loading data from {data_dir}")

        # Find all subject directories
        subject_dirs = [d for d in os.listdir(data_dir) if d.startswith('s') and os.path.isdir(os.path.join(data_dir, d))]

        for subject_dir in subject_dirs:
            subject_id = int(subject_dir[1:])
            subject_path = os.path.join(data_dir, subject_dir)

            # Find all trial directories
            trial_dirs = [d for d in os.listdir(subject_path) if d.startswith('trial') and os.path.isdir(os.path.join(subject_path, d))]

            for trial_dir in trial_dirs:
                trial_path = os.path.join(subject_path, trial_dir)

                # Check if required files exist
                eeg_path = os.path.join(trial_path, 'eeg.npy')
                face_path = os.path.join(trial_path, 'face.npy')
                metadata_path = os.path.join(trial_path, 'metadata.json')

                if os.path.exists(eeg_path) and os.path.exists(metadata_path):
                    # Load metadata to get label
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    # Get valence label (0 or 1)
                    valence = metadata.get('valence', -1)
                    if valence >= 0:
                        self.samples.append({
                            'eeg_path': eeg_path,
                            'face_path': face_path if os.path.exists(face_path) else None,
                            'valence': valence,
                            'subject_id': subject_id
                        })

                        # Update class counts
                        if valence not in self.class_counts:
                            self.class_counts[valence] = 0
                        self.class_counts[valence] += 1

        logger.info(f"Loaded {len(self.samples)} samples")

        # Count class distribution
        logger.info(f"Class distribution: {self.class_counts}")

        # Calculate class weights for balanced sampling
        self.class_weights = {
            cls: len(self.samples) / (len(self.class_counts) * count)
            for cls, count in self.class_counts.items()
        }
        logger.info(f"Class weights: {self.class_weights}")

    def get_sample_weights(self):
        """Get weights for each sample based on class weights"""
        weights = [self.class_weights[sample['valence']] for sample in self.samples]
        return weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load EEG data
        try:
            eeg_data = np.load(sample['eeg_path'])
            # Ensure consistent shape
            if eeg_data.shape != (32,):
                eeg_data = eeg_data.flatten()[:32]
                if len(eeg_data) < 32:
                    eeg_data = np.pad(eeg_data, (0, 32 - len(eeg_data)))
        except Exception as e:
            logger.warning(f"Error loading EEG data: {e}")
            eeg_data = np.zeros(32)

        # Load face data or use zeros
        if sample['face_path'] and os.path.exists(sample['face_path']):
            try:
                face_data = np.load(sample['face_path'])
                if face_data.shape != (768,):
                    face_data = face_data.flatten()[:768]
                    if len(face_data) < 768:
                        face_data = np.pad(face_data, (0, 768 - len(face_data)))
            except Exception as e:
                logger.warning(f"Error loading face data: {e}")
                face_data = np.zeros(768)
        else:
            face_data = np.zeros(768)

        # Apply simple augmentation with 30% probability
        if random.random() < 0.3:
            # Add small Gaussian noise
            eeg_data = eeg_data + np.random.normal(0, 0.01, eeg_data.shape)
            face_data = face_data + np.random.normal(0, 0.01, face_data.shape)

        # Convert to tensors
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        face_tensor = torch.tensor(face_data, dtype=torch.float32)
        label_tensor = torch.tensor(sample['valence'], dtype=torch.long)

        return {
            'eeg': eeg_tensor,
            'face': face_tensor,
            'valence': label_tensor,
            'subject_id': sample['subject_id']
        }

def set_seed(seed):
    """Set seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def evaluate(model, loader, device="cuda"):
    """Evaluation function"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)

            # Forward pass
            logits = model(eeg, face)
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_matrix = confusion_matrix(all_labels, all_preds)

    return acc, f1, conf_matrix

def main():
    # Set seeds for reproducibility
    set_seed(42)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create output directory
    output_dir = f'./outputs/balanced_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    data_dir = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed"
    full_dataset = BalancedDataset(data_dir)

    # Split dataset
    dataset_size = len(full_dataset)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_indices, val_indices, test_indices = random_split(
        range(dataset_size),
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Get sample weights for balanced sampling
    sample_weights = full_dataset.get_sample_weights()
    train_weights = [sample_weights[i] for i in train_indices]

    # Create weighted sampler for training
    train_sampler = WeightedRandomSampler(
        weights=train_weights,
        num_samples=len(train_indices),
        replacement=True
    )

    logger.info(f"Dataset split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    # Create data loaders
    train_loader = DataLoader(
        full_dataset,
        batch_size=32,
        sampler=train_sampler,  # Use weighted sampler
        num_workers=0  # No multiprocessing to avoid issues
    )

    val_loader = DataLoader(
        full_dataset,
        batch_size=32,
        sampler=val_indices,  # Use validation indices
        num_workers=0
    )

    test_loader = DataLoader(
        full_dataset,
        batch_size=32,
        sampler=test_indices,  # Use test indices
        num_workers=0
    )

    # Initialize model
    logger.info("Initializing model...")
    model = RobustModel().to(device)

    # Initialize optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5,
        verbose=True
    )

    # Initialize loss function with class weights
    class_weights = torch.tensor(
        [full_dataset.class_weights[0], full_dataset.class_weights[1]],
        dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Training loop
    logger.info("Starting training...")
    epochs = 100
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience = 15
    counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)

            # Forward pass
            logits = model(eeg, face)

            # Calculate loss
            loss = criterion(logits, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update metrics
            train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(logits.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        val_acc, val_f1, conf_matrix = evaluate(model, val_loader, device)

        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")

        # Update learning rate
        scheduler.step(val_acc)

        # Save best model by accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(output_dir, "best_model_acc.pth"))

            logger.info(f"Saved best accuracy model with validation accuracy: {val_acc:.4f}")

        # Save best model by F1 score
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1
            }, os.path.join(output_dir, "best_model_f1.pth"))

            logger.info(f"Saved best F1 model with validation F1: {val_f1:.4f}")
        else:
            counter += 1
            if counter >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model by accuracy
    # Remove weights_only=True to avoid numpy.core.multiarray.scalar error
    checkpoint = torch.load(os.path.join(output_dir, "best_model_acc.pth"))
    model.load_state_dict(checkpoint['model_state_dict'])

    # Test phase
    test_acc, test_f1, test_conf_matrix = evaluate(model, test_loader, device)

    logger.info(f"Test Results | Acc: {test_acc:.4f} | F1: {test_f1:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_conf_matrix}")

    # Save test results
    with open(os.path.join(output_dir, 'test_results.txt'), 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test F1 Score: {test_f1:.4f}\n")
        f.write(f"Test Confusion Matrix:\n{test_conf_matrix}\n")

    logger.info(f"Training and evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
