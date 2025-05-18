import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset, Subset
from pathlib import Path
import logging
import random
from eeg_augmentations import MultiModalAugmentationPipeline
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN
from collections import Counter

logger = logging.getLogger(__name__)

class EnhancedDEAPDataset(Dataset):
    """Enhanced DEAP dataset with advanced augmentations and subject-aware sampling"""

    def __init__(self,
                 processed_dir,
                 target='valence',
                 apply_augmentation=True,
                 augmentation_prob=0.7,
                 balance_classes=True,
                 balance_method='smote',
                 balance_subjects=True,
                 subject_batch_size=None,
                 cache_features=True):
        """
        Args:
            processed_dir: Directory with processed data
            target: Target emotion ('valence' or 'arousal')
            apply_augmentation: Whether to apply data augmentation
            augmentation_prob: Probability of applying augmentation
            balance_classes: Whether to balance class distribution
            balance_method: Method for balancing ('smote', 'adasyn', 'weighted')
            balance_subjects: Whether to balance subject distribution
            subject_batch_size: Number of samples per subject (for balanced subject sampling)
            cache_features: Whether to cache features in memory
        """
        self.processed_dir = Path(processed_dir)
        self.target = target
        self.apply_augmentation = apply_augmentation
        self.augmentation_prob = augmentation_prob
        self.balance_classes = balance_classes
        self.balance_method = balance_method
        self.balance_subjects = balance_subjects
        self.subject_batch_size = subject_batch_size
        self.cache_features = cache_features

        # Initialize feature cache
        self.feature_cache = {} if cache_features else None

        # Load all samples
        self.samples = self._load_all_samples()

        # Calculate and log class distribution
        self._analyze_class_distribution()

        # Balance classes if requested
        if balance_classes and balance_method != 'weighted':
            self._balance_classes()

        # Balance subjects if requested
        if balance_subjects:
            self._balance_subjects()

    def __len__(self):
        """Returns total number of samples in dataset"""
        return len(self.samples)

    def get_class_weights(self):
        """Get class weights for loss function"""
        return self.class_weight_tensor

    def get_subject_ids(self):
        """Get all unique subject IDs"""
        return list(set(s['subject_id'] for s in self.samples))

    def _load_all_samples(self):
        """Load all valid samples from the dataset"""
        all_samples = []

        # Search for all subject directories
        for subj_dir in sorted(self.processed_dir.glob("s*")):
            if not subj_dir.is_dir():
                continue

            # Get all trial directories for this subject
            for trial_dir in sorted(subj_dir.glob("trial*")):
                if not trial_dir.is_dir():
                    continue

                # Check for metadata file
                metadata_file = trial_dir / "metadata.json"
                if not metadata_file.exists():
                    continue

                # Load metadata
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)

                    # Skip if target label is not available
                    if metadata.get(self.target, -1) < 0:
                        continue

                    # Collect sample info
                    sample = {
                        'eeg_path': str(trial_dir / "eeg.npy"),
                        'face_path': str(trial_dir / "face.npy") if metadata.get('has_video', False) else None,
                        'label': metadata.get(self.target),
                        'subject_id': int(subj_dir.name[1:]),
                        'trial_id': int(trial_dir.name[5:]),
                        'metadata': metadata
                    }

                    # Ensure EEG file exists
                    if not os.path.exists(sample['eeg_path']):
                        continue

                    # For samples with video, ensure face.npy exists
                    if sample['face_path'] and not os.path.exists(sample['face_path']):
                        continue

                    all_samples.append(sample)

                except Exception as e:
                    logger.warning(f"Error loading {metadata_file}: {e}")

        logger.info(f"Loaded {len(all_samples)} valid samples from {self.processed_dir}")
        return all_samples

    def _analyze_class_distribution(self):
        """Analyze and log class distribution"""
        labels = [s['label'] for s in self.samples]
        unique_labels, counts = np.unique(labels, return_counts=True)

        self.class_counts = dict(zip(unique_labels, counts))
        self.class_weights = {
            label: len(self.samples) / (len(unique_labels) * count)
            for label, count in self.class_counts.items()
        }

        logger.info(f"Class distribution: {self.class_counts}")
        logger.info(f"Class weights: {self.class_weights}")

        # Convert to tensor for loss function
        if len(unique_labels) > 0:
            self.class_weight_tensor = torch.tensor(
                [self.class_weights[i] for i in range(len(unique_labels))],
                dtype=torch.float32
            )
        else:
            self.class_weight_tensor = None

        # Analyze subject distribution
        subject_ids = [s['subject_id'] for s in self.samples]
        unique_subjects, subject_counts = np.unique(subject_ids, return_counts=True)
        self.subject_counts = dict(zip(unique_subjects, subject_counts))
        logger.info(f"Subject distribution: {len(unique_subjects)} subjects, min={min(subject_counts)}, max={max(subject_counts)}")

    def _balance_classes(self):
        """Balance classes using SMOTE or ADASYN"""
        # Load and validate all features
        X = []
        valid_samples = []
        failed_samples = 0

        for sample in self.samples:
            try:
                # Load EEG features
                eeg = np.load(sample['eeg_path'])

                # Load face features or use zeros
                face = (np.load(sample['face_path'])
                        if sample['face_path'] and os.path.exists(sample['face_path'])
                        else np.zeros(768))

                # Combine modalities
                combined = np.concatenate([eeg, face])
                X.append(combined)
                valid_samples.append(sample)
            except Exception as e:
                failed_samples += 1
                logger.warning(f"Failed to load sample {sample['subject_id']}-{sample['trial_id']}: {str(e)}")

        if failed_samples > 0:
            logger.warning(f"Skipped {failed_samples} malformed samples during balancing")

        y = np.array([s['label'] for s in valid_samples])
        X = np.array(X)

        # Apply SMOTE or ADASYN if needed
        if len(np.unique(y)) > 1 and X.shape[0] > 0:
            if self.balance_method == 'smote':
                balancer = SMOTE(
                    sampling_strategy='auto',
                    k_neighbors=5,
                    random_state=42
                )
            else:  # adasyn
                balancer = ADASYN(
                    sampling_strategy='auto',
                    n_neighbors=5,
                    random_state=42
                )

            X_res, y_res = balancer.fit_resample(X, y)

            # Rebuild balanced samples
            balanced_samples = []

            # 1. Add original samples with in-memory features
            for idx, sample in enumerate(valid_samples):
                balanced_samples.append({
                    **sample,
                    'eeg_data': X[idx][:32],  # Store first 32 features as EEG
                    'face_data': X[idx][32:],  # Remaining as face
                    'is_synthetic': False
                })

            # 2. Add synthetic samples
            for i in range(len(valid_samples), len(X_res)):
                # Assign synthetic samples to random subjects
                random_subject = random.choice([s['subject_id'] for s in valid_samples])

                balanced_samples.append({
                    'eeg_data': X_res[i][:32],
                    'face_data': X_res[i][32:],
                    'label': int(y_res[i]),
                    'subject_id': random_subject,  # Assign to random subject
                    'trial_id': -1,
                    'is_synthetic': True,
                    'metadata': {
                        'has_video': np.linalg.norm(X_res[i][32:]) > 0.1  # Detect valid face
                    }
                })

            logger.info(f"{self.balance_method.upper()} balancing: {len(self.samples)} -> {len(balanced_samples)} samples")
            self.samples = balanced_samples
        else:
            logger.error("Class balancing failed - insufficient valid samples")

        # Update class distribution
        self._analyze_class_distribution()

    def _balance_subjects(self):
        """Balance subject distribution"""
        if not self.subject_batch_size:
            # Determine a reasonable batch size based on the minimum count
            min_count = min(self.subject_counts.values())
            self.subject_batch_size = max(min_count, 30)  # At least 30 samples per subject

        # Group samples by subject
        subject_samples = {}
        for sample in self.samples:
            subject_id = sample['subject_id']
            if subject_id not in subject_samples:
                subject_samples[subject_id] = []
            subject_samples[subject_id].append(sample)

        # Balance subjects by sampling
        balanced_samples = []
        for subject_id, samples in subject_samples.items():
            # If we have more samples than the batch size, sample randomly
            if len(samples) > self.subject_batch_size:
                sampled = random.sample(samples, self.subject_batch_size)
            else:
                # If we have fewer samples, use all and add synthetic if needed
                sampled = samples.copy()

                # Add synthetic samples if needed and if we have real samples to base them on
                if len(sampled) < self.subject_batch_size and len(sampled) > 0:
                    # Create synthetic samples by augmenting existing ones
                    num_synthetic = self.subject_batch_size - len(sampled)
                    for _ in range(num_synthetic):
                        # Choose a random sample to augment
                        base_sample = random.choice(samples)

                        # Create a synthetic sample
                        synthetic_sample = base_sample.copy()
                        synthetic_sample['is_synthetic'] = True

                        # Add to sampled list
                        sampled.append(synthetic_sample)

            balanced_samples.extend(sampled)

        logger.info(f"Subject balancing: {len(self.samples)} -> {len(balanced_samples)} samples")
        self.samples = balanced_samples

    def apply_augmentations(self, eeg_data, face_data):
        """Apply augmentations to EEG and face data"""
        if not self.apply_augmentation or random.random() > self.augmentation_prob:
            return eeg_data, face_data

        # Apply multimodal augmentations
        return MultiModalAugmentationPipeline.apply_augmentations(eeg_data, face_data, p=0.7)

    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample_info = self.samples[idx]

        # Check if features are cached
        if self.cache_features and 'eeg_data' in sample_info and 'face_data' in sample_info:
            eeg_data = sample_info['eeg_data'].copy()
            face_data = sample_info['face_data'].copy()
        else:
            # Load EEG data
            try:
                eeg_data = np.load(sample_info['eeg_path'])
                # Ensure consistent shape (32,) for loaded data
                if len(eeg_data.shape) != 1 or eeg_data.shape[0] != 32:
                    logger.warning(f"Reshaping EEG data from {eeg_data.shape} to (32,) in {sample_info['eeg_path']}")
                    eeg_data = eeg_data.flatten()[:32] if eeg_data.size >= 32 else np.zeros(32)
            except Exception as e:
                logger.error(f"EEG load failed {sample_info['eeg_path']}: {str(e)}")
                eeg_data = np.zeros(32)

            # Load face data
            face_data = np.zeros(768)
            if sample_info.get('face_path'):
                try:
                    face_data = np.load(sample_info['face_path'])
                    if face_data.shape != (768,):
                        logger.warning(f"Invalid face shape {face_data.shape} in {sample_info['face_path']}")
                        face_data = np.zeros(768)
                except Exception as e:
                    logger.error(f"Face load failed {sample_info['face_path']}: {str(e)}")

            # Cache features if enabled
            if self.cache_features:
                sample_info['eeg_data'] = eeg_data.copy()
                sample_info['face_data'] = face_data.copy()

        # Apply augmentations (skip for synthetic samples that are already augmented)
        if not sample_info.get('is_synthetic', False):
            eeg_data, face_data = self.apply_augmentations(eeg_data, face_data)

        # Numerical stability checks
        eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=1e6, neginf=-1e6)
        face_data = np.nan_to_num(face_data, nan=0.0, posinf=1e6, neginf=-1e6)

        # Convert to tensors
        eeg_tensor = torch.tensor(eeg_data, dtype=torch.float32)
        face_tensor = torch.tensor(face_data, dtype=torch.float32)
        label_tensor = torch.tensor(sample_info['label'], dtype=torch.long)
        subject_tensor = torch.tensor(sample_info['subject_id'], dtype=torch.long)

        # Final sanity checks
        if torch.isnan(eeg_tensor).any():
            logger.error(f"NaN in EEG tensor from {sample_info.get('eeg_path', 'synthetic')}")
            eeg_tensor = torch.zeros_like(eeg_tensor)

        if torch.isnan(face_tensor).any():
            logger.error(f"NaN in face tensor from {sample_info.get('face_path', 'synthetic')}")
            face_tensor = torch.zeros_like(face_tensor)

        return {
            'eeg': eeg_tensor,
            'face': face_tensor,
            'valence': label_tensor,
            'subject_id': subject_tensor,
            'trial_id': sample_info.get('trial_id', -1),
            'is_synthetic': sample_info.get('is_synthetic', False)
        }

def create_subject_aware_splits(dataset, test_size=0.15, val_size=0.15, random_state=42):
    """Create train/val/test splits while preserving subject distribution"""
    # Get all unique subject IDs
    subject_ids = dataset.get_subject_ids()

    # Split subjects into train+val and test
    train_val_subjects, test_subjects = train_test_split(
        subject_ids,
        test_size=test_size,
        random_state=random_state
    )

    # Split train+val subjects into train and val
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_size/(1-test_size),
        random_state=random_state
    )

    logger.info(f"Subject split: {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test")

    # Create indices for each split
    train_indices = []
    val_indices = []
    test_indices = []

    for i, sample in enumerate(dataset.samples):
        subject_id = sample['subject_id']
        if subject_id in train_subjects:
            train_indices.append(i)
        elif subject_id in val_subjects:
            val_indices.append(i)
        elif subject_id in test_subjects:
            test_indices.append(i)

    logger.info(f"Sample split: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    # Create subsets
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, val_set, test_set

def create_stratified_subject_splits(dataset, test_size=0.15, val_size=0.15, random_state=42):
    """Create stratified splits that preserve both subject and class distributions"""
    # Get all samples with their subject IDs and labels
    samples = [(i, s['subject_id'], s['label']) for i, s in enumerate(dataset.samples)]

    # Group by subject
    subject_groups = {}
    for idx, subject_id, label in samples:
        if subject_id not in subject_groups:
            subject_groups[subject_id] = []
        subject_groups[subject_id].append((idx, label))

    # Split subjects into train+val and test
    subject_ids = list(subject_groups.keys())
    train_val_subjects, test_subjects = train_test_split(
        subject_ids,
        test_size=test_size,
        random_state=random_state
    )

    # Split train+val subjects into train and val
    train_subjects, val_subjects = train_test_split(
        train_val_subjects,
        test_size=val_size/(1-test_size),
        random_state=random_state
    )

    # Create indices for each split, ensuring class balance
    train_indices = []
    val_indices = []
    test_indices = []

    # Add samples from each subject group to the appropriate split
    for subject_id, samples in subject_groups.items():
        indices = [idx for idx, _ in samples]
        labels = [label for _, label in samples]

        if subject_id in train_subjects:
            train_indices.extend(indices)
        elif subject_id in val_subjects:
            val_indices.extend(indices)
        elif subject_id in test_subjects:
            test_indices.extend(indices)

    # Log class distribution in each split
    train_labels = [dataset.samples[i]['label'] for i in train_indices]
    val_labels = [dataset.samples[i]['label'] for i in val_indices]
    test_labels = [dataset.samples[i]['label'] for i in test_indices]

    logger.info(f"Train class distribution: {Counter(train_labels)}")
    logger.info(f"Val class distribution: {Counter(val_labels)}")
    logger.info(f"Test class distribution: {Counter(test_labels)}")

    # Create subsets
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, val_set, test_set
