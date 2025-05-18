import numpy as np
import torch
from scipy import signal
import random

class EEGAugmentationPipeline:
    """Advanced EEG augmentation pipeline with multiple techniques"""
    
    @staticmethod
    def scaling(data, factor_range=(0.8, 1.2)):
        """Apply random scaling"""
        factor = np.random.uniform(factor_range[0], factor_range[1])
        return data * factor
    
    @staticmethod
    def gaussian_noise(data, std_range=(0.01, 0.05)):
        """Add random Gaussian noise"""
        std = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std, data.shape)
        return data + noise
    
    @staticmethod
    def band_stop_filter(data, fs=128, fstop=(55, 65), order=4):
        """Apply band-stop filter (e.g., to remove line noise)"""
        nyq = 0.5 * fs
        low = fstop[0] / nyq
        high = fstop[1] / nyq
        b, a = signal.butter(order, [low, high], btype='bandstop')
        return signal.filtfilt(b, a, data)
    
    @staticmethod
    def band_pass_filter(data, fs=128, fpass=(4, 45), order=4):
        """Apply band-pass filter to focus on relevant frequency bands"""
        nyq = 0.5 * fs
        low = fpass[0] / nyq
        high = fpass[1] / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        return signal.filtfilt(b, a, data)
    
    @staticmethod
    def channel_dropout(data, p=0.1, min_channels=16):
        """Randomly drop out channels"""
        mask = np.random.binomial(1, 1-p, size=len(data))
        # Ensure at least min_channels channels remain
        if np.sum(mask) < min_channels:
            # Randomly select channels to keep
            zero_indices = np.where(mask == 0)[0]
            num_to_restore = min_channels - np.sum(mask)
            to_restore = np.random.choice(zero_indices, size=int(num_to_restore), replace=False)
            mask[to_restore] = 1
        
        return data * mask
    
    @staticmethod
    def channel_shuffle(data, p=0.1):
        """Randomly shuffle a subset of channels"""
        if np.random.random() < p:
            # Select a random subset of channels to shuffle
            num_channels = len(data)
            num_to_shuffle = max(2, int(num_channels * np.random.uniform(0.1, 0.3)))
            channels_to_shuffle = np.random.choice(num_channels, num_to_shuffle, replace=False)
            
            # Shuffle the selected channels
            shuffled_indices = channels_to_shuffle.copy()
            np.random.shuffle(shuffled_indices)
            
            # Create a new array with shuffled channels
            result = data.copy()
            for i, j in zip(channels_to_shuffle, shuffled_indices):
                result[i] = data[j]
                
            return result
        return data
    
    @staticmethod
    def temporal_shift(data, max_shift=3):
        """Apply small temporal shifts to channels"""
        result = data.copy()
        for i in range(len(data)):
            if np.random.random() < 0.5:  # 50% chance to shift each channel
                shift = np.random.randint(-max_shift, max_shift+1)
                result[i] = np.roll(data[i], shift)
        return result
    
    @staticmethod
    def spectral_transform(data, fs=128):
        """Apply random spectral transformation"""
        # Convert to frequency domain
        fft_data = np.fft.rfft(data)
        
        # Apply random phase shift
        phases = np.exp(1j * np.random.uniform(0, 2*np.pi, fft_data.shape))
        fft_data = fft_data * phases
        
        # Convert back to time domain
        return np.fft.irfft(fft_data, n=len(data))
    
    @staticmethod
    def magnitude_warp(data, sigma=0.2, knot=4):
        """Apply magnitude warping"""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(len(data))
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=knot+2)
        warp_steps = (np.linspace(0, len(data)-1, num=knot+2))
        warper = CubicSpline(warp_steps, random_warps)(orig_steps)
        
        return data * warper
    
    @staticmethod
    def time_warp(data, sigma=0.2, knot=4):
        """Apply time warping"""
        from scipy.interpolate import CubicSpline
        
        orig_steps = np.arange(len(data))
        random_warps = np.random.normal(loc=1.0, scale=sigma, size=knot+2)
        warp_steps = (np.linspace(0, len(data)-1, num=knot+2))
        warper = CubicSpline(warp_steps, random_warps)(orig_steps)
        
        time_warp = np.cumsum(warper)
        time_warp = time_warp * (len(data) - 1) / time_warp[-1]
        
        ret = np.interp(time_warp, orig_steps, data)
        return ret
    
    @staticmethod
    def apply_augmentations(data, num_augmentations=2, p=0.5):
        """Apply multiple augmentations with probability p"""
        if np.random.random() > p:
            return data
        
        # List of augmentation functions
        augmentations = [
            EEGAugmentationPipeline.scaling,
            EEGAugmentationPipeline.gaussian_noise,
            EEGAugmentationPipeline.channel_dropout,
            EEGAugmentationPipeline.temporal_shift,
            EEGAugmentationPipeline.channel_shuffle
        ]
        
        # Advanced augmentations (apply with lower probability)
        if np.random.random() < 0.3:
            advanced_augmentations = [
                EEGAugmentationPipeline.band_pass_filter,
                EEGAugmentationPipeline.spectral_transform,
                EEGAugmentationPipeline.magnitude_warp
            ]
            augmentations.extend(advanced_augmentations)
        
        # Randomly select augmentations to apply
        selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
        
        # Apply selected augmentations
        result = data.copy()
        for aug_func in selected_augmentations:
            result = aug_func(result)
        
        return result

class FaceAugmentationPipeline:
    """Augmentation pipeline for facial features"""
    
    @staticmethod
    def feature_dropout(data, p=0.1):
        """Randomly drop out facial features"""
        mask = np.random.binomial(1, 1-p, size=len(data))
        return data * mask
    
    @staticmethod
    def gaussian_noise(data, std_range=(0.01, 0.03)):
        """Add random Gaussian noise"""
        std = np.random.uniform(std_range[0], std_range[1])
        noise = np.random.normal(0, std, data.shape)
        return data + noise
    
    @staticmethod
    def feature_scaling(data, factor_range=(0.9, 1.1)):
        """Apply random scaling to facial features"""
        factor = np.random.uniform(factor_range[0], factor_range[1])
        return data * factor
    
    @staticmethod
    def feature_jitter(data, sigma=0.05):
        """Add jitter to facial features"""
        return data + np.random.normal(0, sigma, data.shape)
    
    @staticmethod
    def apply_augmentations(data, num_augmentations=2, p=0.5):
        """Apply multiple augmentations with probability p"""
        if np.random.random() > p:
            return data
        
        # List of augmentation functions
        augmentations = [
            FaceAugmentationPipeline.feature_dropout,
            FaceAugmentationPipeline.gaussian_noise,
            FaceAugmentationPipeline.feature_scaling,
            FaceAugmentationPipeline.feature_jitter
        ]
        
        # Randomly select augmentations to apply
        selected_augmentations = random.sample(augmentations, min(num_augmentations, len(augmentations)))
        
        # Apply selected augmentations
        result = data.copy()
        for aug_func in selected_augmentations:
            result = aug_func(result)
        
        return result

class MultiModalAugmentationPipeline:
    """Combined augmentation pipeline for multimodal data"""
    
    @staticmethod
    def apply_augmentations(eeg_data, face_data, p=0.7):
        """Apply augmentations to both modalities"""
        # Apply EEG augmentations
        augmented_eeg = EEGAugmentationPipeline.apply_augmentations(eeg_data, num_augmentations=2, p=p)
        
        # Apply face augmentations
        augmented_face = FaceAugmentationPipeline.apply_augmentations(face_data, num_augmentations=2, p=p)
        
        return augmented_eeg, augmented_face
    
    @staticmethod
    def cross_modal_augmentation(eeg_data, face_data, p=0.3):
        """Apply correlated augmentations across modalities"""
        if np.random.random() > p:
            return eeg_data, face_data
        
        # Apply correlated scaling
        scale_factor = np.random.uniform(0.8, 1.2)
        eeg_scaled = eeg_data * scale_factor
        face_scaled = face_data * scale_factor
        
        # Apply correlated noise
        noise_level = np.random.uniform(0.01, 0.05)
        eeg_noise = np.random.normal(0, noise_level, eeg_scaled.shape)
        face_noise = np.random.normal(0, noise_level, face_scaled.shape)
        
        return eeg_scaled + eeg_noise, face_scaled + face_noise
