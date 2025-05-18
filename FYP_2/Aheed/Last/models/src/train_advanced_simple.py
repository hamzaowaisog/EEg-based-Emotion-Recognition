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

# Gradient Reversal Layer for domain adaptation
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None

class GradientReversal(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

# Advanced model with domain adaptation and contrastive learning
class AdvancedModel(nn.Module):
    def __init__(self, eeg_dim=32, face_dim=768, hidden_dim=128, output_dim=2, num_subjects=32):
        super().__init__()

        # EEG encoder
        self.eeg_encoder = nn.Sequential(
            nn.Linear(eeg_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Face encoder
        self.face_encoder = nn.Sequential(
            nn.Linear(face_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )

        # Domain classifier (subject classifier) with gradient reversal
        self.gradient_reversal = GradientReversal(alpha=1.0)
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_subjects)
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        # Emotion classifier
        self.classifier = nn.Linear(hidden_dim, output_dim)

        # Projection heads for contrastive learning
        self.eeg_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.face_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, eeg, face, lambda_val=1.0):
        # Set gradient reversal strength
        self.gradient_reversal.alpha = lambda_val

        # Extract features
        eeg_features = self.eeg_encoder(eeg)
        face_features = self.face_encoder(face)

        # Concatenate features
        combined_features = torch.cat([eeg_features, face_features], dim=1)

        # Domain classification with gradient reversal
        domain_logits = self.domain_classifier(self.gradient_reversal(combined_features))

        # Fusion
        fused_features = self.fusion(combined_features)

        # Emotion classification
        emotion_logits = self.classifier(fused_features)

        # Projections for contrastive learning
        eeg_proj = self.eeg_projection(eeg_features)
        face_proj = self.face_projection(face_features)

        return {
            'emotion_logits': emotion_logits,
            'domain_logits': domain_logits,
            'eeg_proj': eeg_proj,
            'face_proj': face_proj,
            'fused_features': fused_features
        }

# Contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_1, features_2, labels=None):
        # Normalize features
        features_1 = F.normalize(features_1, dim=1)
        features_2 = F.normalize(features_2, dim=1)

        # Concatenate features from both modalities
        batch_size = features_1.size(0)
        features = torch.cat([features_1, features_2], dim=0)

        # Compute similarity matrix
        similarity = torch.matmul(features, features.T) / self.temperature

        # Mask for positive pairs
        mask = torch.zeros_like(similarity)
        mask[:batch_size, batch_size:] = torch.eye(batch_size)
        mask[batch_size:, :batch_size] = torch.eye(batch_size)

        # Compute loss
        similarity = similarity - torch.eye(2 * batch_size, device=similarity.device) * 1e9  # Mask diagonal
        exp_sim = torch.exp(similarity)

        # Compute log_prob
        log_prob = similarity - torch.log(exp_sim.sum(dim=1, keepdim=True))

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # Loss
        loss = -mean_log_prob_pos.mean()

        return loss

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.3, temperature=0.5, class_weights=None):
        super().__init__()
        self.alpha = alpha  # Weight for contrastive loss
        self.beta = beta    # Weight for domain loss
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.domain_loss = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, subject_ids):
        # Classification loss
        cls_loss = self.ce_loss(outputs['emotion_logits'], targets)

        # Contrastive loss
        cont_loss = self.contrastive_loss(outputs['eeg_proj'], outputs['face_proj'])

        # Domain adversarial loss
        domain_loss = self.domain_loss(outputs['domain_logits'], subject_ids)

        # Combined loss
        total_loss = cls_loss + self.alpha * cont_loss - self.beta * domain_loss

        return total_loss, {
            'cls_loss': cls_loss,
            'cont_loss': cont_loss,
            'domain_loss': domain_loss
        }

# Dataset class with better handling
class AdvancedDataset:
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
        subject_tensor = torch.tensor(sample['subject_id'], dtype=torch.long)

        return {
            'eeg': eeg_tensor,
            'face': face_tensor,
            'valence': label_tensor,
            'subject_id': subject_tensor
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
            outputs = model(eeg, face, lambda_val=0.0)  # No domain adaptation during evaluation
            logits = outputs['emotion_logits']
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
    output_dir = f'./outputs/advanced_simple_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    logger.info("Loading dataset...")
    data_dir = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed"
    full_dataset = AdvancedDataset(data_dir)

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
    model = AdvancedModel().to(device)

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
    criterion = CombinedLoss(
        alpha=0.3,  # Weight for contrastive loss
        beta=0.3,   # Weight for domain loss
        temperature=0.5,
        class_weights=class_weights
    )

    # Training loop
    logger.info("Starting training...")
    epochs = 100
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience = 15
    counter = 0

    # Lambda schedule for domain adaptation
    def get_lambda(epoch, max_epochs):
        return min(1.0, epoch / (max_epochs * 0.5))

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Calculate lambda for domain adaptation
        lambda_val = get_lambda(epoch, epochs)

        # Training phase
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)
            subject_ids = batch['subject_id'].to(device)

            # Forward pass
            outputs = model(eeg, face, lambda_val=lambda_val)

            # Calculate loss
            loss, loss_components = criterion(outputs, labels, subject_ids)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Update metrics
            train_loss += loss.item()

            # Calculate training accuracy
            _, predicted = torch.max(outputs['emotion_logits'].data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # Validation phase
        val_acc, val_f1, conf_matrix = evaluate(model, val_loader, device)

        logger.info(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Lambda: {lambda_val:.2f}")
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
