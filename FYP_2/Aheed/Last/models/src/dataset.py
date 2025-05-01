import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class DEAPDataset(Dataset):
    def __init__(self, processed_dir=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed", subjects=None, target="valence"):
        """
        Args:
            processed_dir: Path to processed data directory
            subjects: List of subject IDs to include (e.g. ["s01", "s02"])
            target: "valence" or "arousal"
        """
        self.processed_dir = processed_dir
        self.target = target
        self.samples = []
        
        # Get all subjects if not specified
        if subjects is None:
            subjects = [d for d in os.listdir(processed_dir) 
                       if os.path.isdir(os.path.join(processed_dir, d))]
        
        # Collect all valid trials
        for subject in subjects:
            subj_dir = os.path.join(processed_dir, subject)
            for trial in os.listdir(subj_dir):
                trial_dir = os.path.join(subj_dir, trial)
                if os.path.isdir(trial_dir):
                    self.samples.append({
                        "eeg_path": os.path.join(trial_dir, "eeg.npy"),
                        "face_path": os.path.join(trial_dir, "face.npy"),
                        "meta_path": os.path.join(trial_dir, "metadata.json")
                    })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load EEG
        eeg = np.load(sample["eeg_path"])
        
        # Load Face (zero-pad if missing)
        try:
            face = np.load(sample["face_path"])
        except FileNotFoundError:
            face = np.zeros(768)  # Match ViT feature dimension
        
        # Load Metadata
        with open(sample["meta_path"], 'r') as f:
            meta = json.load(f)
        
        # Convert to tensors
        return {
            "eeg": torch.tensor(eeg, dtype=torch.float32),  # (32,)
            "face": torch.tensor(face, dtype=torch.float32),  # (768,)
            "valence": torch.tensor(meta["valence"], dtype=torch.long).squeeze(),  # Scalar ()
            "arousal": torch.tensor(meta["arousal"], dtype=torch.long).squeeze(),  # Scalar ()
            "has_video": torch.tensor(meta["has_video"], dtype=torch.bool)  # Scalar ()
        }

    def get_subject_ids(self):
        """Get list of unique subject IDs in dataset"""
        return list(set([os.path.basename(os.path.dirname(s["eeg_path"])) 
                         for s in self.samples]))