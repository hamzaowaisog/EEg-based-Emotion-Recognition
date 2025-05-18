import torch
import numpy as np
import json
import os
import cv2
from PIL import Image
import logging
from torch.utils.data import Dataset
from torchvision import transforms
from transformers import CLIPImageProcessor
from pathlib import Path
from collections import Counter
import random
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class FaceDataset(Dataset):
    """Dataset for loading facial features for emotion recognition"""
    
    def __init__(
        self,
        processed_dir,
        target='valence',
        image_size=224,
        clip_preprocess=True,
        balance_classes=True,
        augment=True,
        augment_prob=0.5,
        test_mode=False
    ):
        """
        Args:
            processed_dir: Directory with processed data
            target: Target emotion ('valence' or 'arousal')
            image_size: Size to resize images to
            clip_preprocess: Whether to use CLIP preprocessor
            balance_classes: Whether to balance classes
            augment: Whether to apply augmentations
            augment_prob: Probability of applying augmentation
            test_mode: If True, no augmentations will be applied
        """
        self.processed_dir = Path(processed_dir)
        self.target = target
        self.image_size = image_size
        self.clip_preprocess = clip_preprocess
        self.balance_classes = balance_classes
        self.augment = augment and not test_mode
        self.augment_prob = augment_prob
        self.test_mode = test_mode
        
        # Setup preprocessing
        if clip_preprocess:
            self.preprocessor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        else:
            # Standard preprocessing pipeline
            self.transforms = self._get_transforms(self.image_size, self.augment)
        
        # Load all samples with face data
        self.samples = self._load_all_samples()
        
        # Calculate and log class distribution
        self._analyze_class_distribution()
        
        # Balance classes if requested
        if balance_classes and not test_mode:
            self._balance_classes()
    
    def _get_transforms(self, image_size, augment):
        """Get image transforms with or without augmentations"""
        if augment:
            return {
                'train': transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            }
        else:
            # No augmentation for test/validation
            return {
                'train': transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ]),
                'val': transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            }
    
    def _load_all_samples(self):
        """Load all samples with video/face data"""
        all_samples = []
        
        # Search for all subject directories
        for subj_dir in sorted(self.processed_dir.glob("s*")):
            if not subj_dir.is_dir():
                continue
                
            # Only 1-22 have video
            if int(subj_dir.name[1:]) > 22:
                continue
                
            # Get all trial directories for this subject
            for trial_dir in sorted(subj_dir.glob("trial*")):
                if not trial_dir.is_dir():
                    continue
                    
                # Check for metadata file
                metadata_file = trial_dir / "metadata.json"
                if not metadata_file.exists():
                    continue
                    
                # Check for face.npy file
                face_path = trial_dir / "face.npy"
                if not face_path.exists():
                    continue
                    
                # Load metadata
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                        
                    # Skip if target label is not available
                    if metadata.get(self.target, -1) < 0:
                        continue
                        
                    # Skip if no video features
                    if not metadata.get('has_video', False):
                        continue
                        
                    # Collect sample info
                    sample = {
                        'face_path': str(face_path),
                        'label': metadata.get(self.target),
                        'subject_id': int(subj_dir.name[1:]),
                        'trial_id': int(trial_dir.name[5:]),
                        'metadata': metadata,
                        'raw_video_path': os.path.join(
                            r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\raw\videos",
                            f"s{int(subj_dir.name[1:]):02d}",
                            f"s{int(subj_dir.name[1:]):02d}_trial{int(trial_dir.name[5:]):02d}.avi"
                        )
                    }
                    
                    all_samples.append(sample)
                    
                except Exception as e:
                    logger.warning(f"Error loading {metadata_file}: {e}")
                    
        logger.info(f"Loaded {len(all_samples)} samples with face data")
        return all_samples
    
    def _analyze_class_distribution(self):
        """Analyze and log class distribution"""
        if not self.samples:
            logger.warning("No samples found for analysis")
            self.class_weights = {}
            return
            
        labels = [s['label'] for s in self.samples]
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        self.class_counts = dict(zip(unique_labels, counts))
        total_samples = len(self.samples)
        num_classes = len(unique_labels)
        
        # Calculate inverse frequency weights
        self.class_weights = {
            label: total_samples / (num_classes * count)
            for label, count in self.class_counts.items()
        }
        
        logger.info(f"Class distribution: {self.class_counts}")
        logger.info(f"Class weights: {self.class_weights}")
        
    def _balance_classes(self):
        """Balance classes by oversampling minority class"""
        if not self.samples:
            return
            
        # Group samples by class
        class_samples = {}
        for sample in self.samples:
            label = sample['label']
            if label not in class_samples:
                class_samples[label] = []
            class_samples[label].append(sample)
            
        # Find maximum class count
        max_count = max(len(samples) for samples in class_samples.values())
        
        # Oversample minority classes
        balanced_samples = []
        for label, samples in class_samples.items():
            # Add all original samples
            balanced_samples.extend(samples)
            
            # Oversample if needed
            if len(samples) < max_count:
                # Number of additional samples needed
                num_additional = max_count - len(samples)
                # Sample with replacement
                additional_samples = random.choices(samples, k=num_additional)
                balanced_samples.extend(additional_samples)
                
        logger.info(f"Balanced classes: {len(self.samples)} -> {len(balanced_samples)} samples")
        self.samples = balanced_samples
        
        # Update class distribution
        self._analyze_class_distribution()
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)
    
    def __getitem__(self, idx):
        """Get a sample from the dataset"""
        sample = self.samples[idx]
        
        # Load face features
        face_features = np.load(sample['face_path'])
        
        if self.test_mode:
            # Just return the precomputed features during testing
            return {
                'face_features': torch.tensor(face_features, dtype=torch.float32),
                'label': torch.tensor(sample['label'], dtype=torch.long),
                'subject_id': sample['subject_id'],
                'trial_id': sample['trial_id']
            }
        
        # Return features
        return {
            'face_features': torch.tensor(face_features, dtype=torch.float32),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'subject_id': sample['subject_id'],
            'trial_id': sample['trial_id']
        }


def create_subject_aware_splits(dataset, test_size=0.15, val_size=0.15, random_state=42):
    """Create train/val/test splits that are subject-aware"""
    # Get all unique subject IDs
    subject_ids = list(set(sample['subject_id'] for sample in dataset.samples))
    
    # Split subjects into train, val, test
    train_subjects, temp_subjects = train_test_split(
        subject_ids, 
        test_size=test_size+val_size,
        random_state=random_state
    )
    
    val_subjects, test_subjects = train_test_split(
        temp_subjects,
        test_size=test_size/(test_size+val_size),
        random_state=random_state
    )
    
    # Create masks for each split
    train_mask = [sample['subject_id'] in train_subjects for sample in dataset.samples]
    val_mask = [sample['subject_id'] in val_subjects for sample in dataset.samples]
    test_mask = [sample['subject_id'] in test_subjects for sample in dataset.samples]
    
    # Create sample indices for each split
    train_indices = [i for i, mask in enumerate(train_mask) if mask]
    val_indices = [i for i, mask in enumerate(val_mask) if mask]
    test_indices = [i for i, mask in enumerate(test_mask) if mask]
    
    # Log split sizes
    logger.info(f"Train split: {len(train_indices)} samples, {len(train_subjects)} subjects")
    logger.info(f"Val split: {len(val_indices)} samples, {len(val_subjects)} subjects")
    logger.info(f"Test split: {len(test_indices)} samples, {len(test_subjects)} subjects")
    
    # Log subject IDs in each split
    logger.info(f"Train subjects: {sorted(train_subjects)}")
    logger.info(f"Val subjects: {sorted(val_subjects)}")
    logger.info(f"Test subjects: {sorted(test_subjects)}")
    
    return train_indices, val_indices, test_indices 