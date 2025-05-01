import os
import re
import pickle
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ViTModel, ViTImageProcessor
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from scipy import signal

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
PROCESSED_DIR = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed"
EEG_RAW_DIR = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\raw\eeg"
VIDEO_RAW_DIR = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\raw\videos"
RATINGS_FILE = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\metadata_csv\participant_ratings.csv"
FACE_DATA_JSON = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\output_json\face_data.json"

# EEG processing parameters
SAMPLING_RATE = 128  # Hz (adjust based on your actual sampling rate)
WINDOW_SIZE = 1.0    # seconds
OVERLAP = 0.5        # seconds

# Initialize ViT model globally
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Load model only once at the beginning
vit_model = None
vit_processor = None

def init_vit_model():
    """Initialize ViT model once and cache it"""
    global vit_model, vit_processor
    try:
        vit_model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k").to(device)
        vit_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        vit_model.eval()  # Set to evaluation mode
        logger.info("ViT model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ViT model: {str(e)}")
        raise

def load_eeg_data(subject_file):
    """Load and validate EEG data with DEAP specifications"""
    try:
        with open(os.path.join(EEG_RAW_DIR, subject_file), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # Extract raw data array
        raw_data = data.get('data')
        if raw_data is None or not isinstance(raw_data, np.ndarray):
            logger.warning(f"Invalid EEG data format in {subject_file}")
            return None

        # DEAP specification: (40 trials, 40 channels, 8064 samples)
        if raw_data.shape != (40, 40, 8064):
            logger.warning(f"Unexpected EEG data dimensions in {subject_file}: {raw_data.shape}")
            return None

        # Extract first 32 EEG channels (ignore 8 peripheral channels)
        eeg_data = raw_data[:, :32, :]  # Shape: (40, 32, 8064)
        
        return eeg_data
        
    except Exception as e:
        logger.error(f"Error loading {subject_file}: {str(e)}")
        return None

def apply_bandpass_filter(eeg_data, low_freq=1.0, high_freq=50.0, sampling_rate=SAMPLING_RATE):
    """Apply bandpass filter to EEG data"""
    nyquist_freq = 0.5 * sampling_rate
    low = low_freq / nyquist_freq
    high = high_freq / nyquist_freq
    b, a = signal.butter(4, [low, high], btype='band')
    return signal.filtfilt(b, a, eeg_data)

def process_eeg_features(eeg_trial):
    """Calculate differential entropy features for a single trial with windowing"""
    num_channels, num_samples = eeg_trial.shape
    
    # Apply filtering to remove noise
    filtered_data = np.array([apply_bandpass_filter(channel) for channel in eeg_trial])
    
    # Calculate window and step size in samples
    window_samples = int(WINDOW_SIZE * SAMPLING_RATE)
    step_samples = int((WINDOW_SIZE - OVERLAP) * SAMPLING_RATE)
    
    # Calculate number of windows
    num_windows = max(1, int((num_samples - window_samples) / step_samples) + 1)
    
    # Initialize features array
    all_features = np.zeros((num_channels, num_windows))
    
    # Process each window
    for w in range(num_windows):
        start = w * step_samples
        end = min(start + window_samples, num_samples)
        
        # Skip partial windows at the end
        if end - start < window_samples * 0.75:  # Ensure at least 75% of window is filled
            continue
            
        # Apply Hamming window
        window = signal.windows.hamming(end - start)
        
        # Calculate differential entropy for each channel
        for c in range(num_channels):
            # Apply window function
            windowed_data = filtered_data[c, start:end] * window
            
            # Calculate variance safely (add small epsilon to avoid log(0))
            var = np.var(windowed_data) + 1e-10
            
            # Differential entropy
            de = 0.5 * np.log(2 * np.pi * np.e * var)
            all_features[c, w] = de
    
    # Average across windows to get final features
    return np.mean(all_features, axis=1)

def load_face_data(batch=False):
    """Load face data with optional batch processing to save memory"""
    try:
        if batch:
            # For batch processing, return the file path instead of loading all data
            return FACE_DATA_JSON
        else:
            with open(FACE_DATA_JSON) as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error loading face data: {str(e)}")
        return {} if not batch else None

def get_video_face_data(video_file, all_face_data, batch=False):
    """Get face data for a specific video"""
    if batch:
        # For batch mode, load only the required video data
        try:
            with open(all_face_data) as f:  # all_face_data is the file path
                data = json.load(f)
                return data.get(video_file, [])
        except Exception as e:
            logger.error(f"Error loading face data for {video_file}: {str(e)}")
            return []
    else:
        # For non-batch mode, all_face_data is already loaded
        return all_face_data.get(video_file, [])

def preprocess_face_image(face_crop, target_size=(224, 224)):
    """Preprocess face image"""
    if face_crop.size == 0:
        return None
    
    try:
        # Convert to RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        
        # Resize to target size
        face_pil = Image.fromarray(face_rgb)
        return face_pil
    except Exception as e:
        logger.error(f"Error preprocessing face image: {str(e)}")
        return None

def process_video(video_path, face_data):
    """Process video to extract facial features using ViT"""
    # Extract video filename
    video_file = os.path.basename(video_path)
    
    if not os.path.exists(video_path):
        logger.warning(f"Video file not found: {video_path}")
        return np.zeros(768)
    
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.warning(f"Cannot open video: {video_path}")
            return np.zeros(768)
        
        # Get frame data
        frame_data = face_data
        if not frame_data:
            logger.warning(f"No face data available for {video_file}")
            cap.release()
            return np.zeros(768)
        
        # Process frames in batches to avoid memory issues
        total_frames = len(frame_data)
        batch_size = min(total_frames, 32)  # Process 32 frames at a time
        
        all_features = []
        
        for batch_start in range(0, total_frames, batch_size):
            batch_end = min(batch_start + batch_size, total_frames)
            batch_frames = []
            valid_indices = []
            
            # Extract faces from frames
            for i, frame_info in enumerate(frame_data[batch_start:batch_end]):
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_info["frame"])
                    ret, frame = cap.read()
                    if ret:
                        x1, y1, x2, y2 = map(int, frame_info["bbox"])
                        # Ensure valid bounding box
                        if x2 > x1 and y2 > y1 and (x2-x1)*(y2-y1) > 0:
                            face_crop = frame[y1:y2, x1:x2]
                            processed_face = preprocess_face_image(face_crop)
                            if processed_face is not None:
                                batch_frames.append(processed_face)
                                valid_indices.append(i)
                except Exception as e:
                    logger.warning(f"Error processing {video_file} frame {frame_info['frame']}: {str(e)}")
            
            if not batch_frames:
                continue
                
            # Process batch with ViT
            try:
                inputs = vit_processor(images=batch_frames, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = vit_model(**inputs)
                    # Use the [CLS] token as the image representation
                    batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                
                all_features.append(batch_features)
            except Exception as e:
                logger.error(f"Error in ViT processing for {video_file}: {str(e)}")
        
        cap.release()
        
        if not all_features:
            logger.warning(f"No valid features extracted for {video_file}")
            return np.zeros(768)
        
        # Concatenate all batches
        all_features = np.vstack(all_features)
        
        # Return average feature vector
        return np.mean(all_features, axis=0)
        
    except Exception as e:
        logger.error(f"Error processing video {video_path}: {str(e)}")
        return np.zeros(768)

def validate_subject_ratings(ratings_df, subject_id):
    """Validate ratings data for a subject"""
    subject_ratings = ratings_df[ratings_df["Participant_id"] == subject_id]
    
    if len(subject_ratings) == 0:
        logger.warning(f"No ratings found for subject {subject_id}")
        return None
    
    if len(subject_ratings) < 40:
        logger.warning(f"Incomplete ratings for subject {subject_id}: found {len(subject_ratings)}, expected 40")
    
    # Check for missing values
    missing_vals = subject_ratings[["Valence", "Arousal"]].isna().sum().sum()
    if missing_vals > 0:
        logger.warning(f"Subject {subject_id} has {missing_vals} missing rating values")
    
    return subject_ratings

def process_subject(subject_id, ratings_df, use_batch_processing=True):
    """Process all data for a single subject"""
    logger.info(f"Processing subject {subject_id}")
    
    # Create directories
    subj_dir = os.path.join(PROCESSED_DIR, f"s{subject_id:02d}")
    os.makedirs(subj_dir, exist_ok=True)

    # Load EEG data
    eeg_data = load_eeg_data(f"s{subject_id:02d}.dat")
    if eeg_data is None:
        logger.error(f"Failed to load EEG data for subject {subject_id}")
        return False

    # Load face data (path only if batch processing)
    face_data_source = load_face_data(batch=use_batch_processing)
    if face_data_source is None and subject_id <= 22:  # Only subjects 1-22 have video
        logger.error(f"Failed to load face data source for subject {subject_id}")
        return False

    # Validate ratings
    subject_ratings = validate_subject_ratings(ratings_df, subject_id)
    if subject_ratings is None and subject_id > 0:  # Allow processing without ratings for testing
        logger.error(f"Failed to validate ratings for subject {subject_id}")
        return False

    # Track successful trials
    successful_trials = 0
    
    # Process each trial
    for trial in tqdm(range(40), desc=f"Processing s{subject_id:02d}"):
        # Create trial directory
        trial_dir = os.path.join(subj_dir, f"trial{trial+1:02d}")
        os.makedirs(trial_dir, exist_ok=True)

        try:
            # Process EEG
            eeg_trial = eeg_data[trial, :, :]  # Shape: (32, 8064)
            de_features = process_eeg_features(eeg_trial)
            np.save(os.path.join(trial_dir, "eeg.npy"), de_features)

            # Process Video (if available)
            video_features = None
            has_video = subject_id <= 22
            
            if has_video:
                video_path = os.path.join(
                    VIDEO_RAW_DIR, 
                    f"s{subject_id:02d}",
                    f"s{subject_id:02d}_trial{trial+1:02d}.avi"
                )
                
                if use_batch_processing:
                    # Get face data for just this video
                    video_file = os.path.basename(video_path)
                    video_face_data = get_video_face_data(video_file, face_data_source, batch=True)
                else:
                    # Use already loaded face data
                    video_file = os.path.basename(video_path)
                    video_face_data = face_data_source.get(video_file, [])
                
                video_features = process_video(video_path, video_face_data)
                np.save(os.path.join(trial_dir, "face.npy"), video_features)

            # Save metadata
            try:
                if subject_ratings is not None:
                    trial_rating = subject_ratings[subject_ratings["Trial"] == trial+1]
                    if len(trial_rating) > 0:
                        trial_rating = trial_rating.iloc[0]
                        metadata = {
                            "valence": int(trial_rating["Valence"] >= 5),  # Binary classification
                            "arousal": int(trial_rating["Arousal"] >= 5),  # Binary classification
                            "valence_raw": float(trial_rating["Valence"]),  # Original score
                            "arousal_raw": float(trial_rating["Arousal"]),  # Original score
                            "has_video": has_video,
                            "video_features_shape": None if video_features is None else video_features.shape[0],
                            "eeg_features_shape": de_features.shape[0]
                        }
                    else:
                        logger.warning(f"Missing rating for s{subject_id:02d} trial{trial+1}")
                        metadata = {
                            "valence": -1, 
                            "arousal": -1, 
                            "valence_raw": -1,
                            "arousal_raw": -1,
                            "has_video": has_video,
                            "video_features_shape": None if video_features is None else video_features.shape[0],
                            "eeg_features_shape": de_features.shape[0]
                        }
                else:
                    metadata = {
                        "valence": -1, 
                        "arousal": -1,
                        "valence_raw": -1,
                        "arousal_raw": -1,
                        "has_video": has_video,
                        "video_features_shape": None if video_features is None else video_features.shape[0],
                        "eeg_features_shape": de_features.shape[0]
                    }
            except Exception as e:
                logger.error(f"Metadata error s{subject_id:02d} trial{trial+1}: {str(e)}")
                metadata = {
                    "valence": -1, 
                    "arousal": -1,
                    "has_video": has_video,
                    "error": str(e)
                }

            with open(os.path.join(trial_dir, "metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            successful_trials += 1
            
        except Exception as e:
            logger.error(f"Error processing s{subject_id:02d} trial{trial+1}: {str(e)}")
            # Create error metadata
            with open(os.path.join(trial_dir, "error.json"), 'w') as f:
                json.dump({"error": str(e)}, f)
    
    logger.info(f"Subject {subject_id}: {successful_trials}/40 trials processed successfully")
    return successful_trials > 0

def main():
    """Main function to process all subjects"""
    # Create output directory
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    # Initialize ViT model once
    init_vit_model()
    
    # Load ratings once
    try:
        ratings_df = pd.read_csv(RATINGS_FILE)
        logger.info(f"Loaded ratings for {ratings_df['Participant_id'].nunique()} participants")
    except Exception as e:
        logger.error(f"Failed to load ratings: {str(e)}")
        return
    
    # Check if column names match expected format
    expected_columns = ["Participant_id", "Trial", "Valence", "Arousal"]
    if not all(col in ratings_df.columns for col in expected_columns):
        logger.error(f"Ratings file missing expected columns. Found: {ratings_df.columns.tolist()}")
        logger.error(f"Expected: {expected_columns}")
        return
    
    # Process all subjects with parallel processing
    subjects = list(range(1, 33))  # 32 EEG subjects
    
    # Parameter for controlling batch processing
    use_batch_processing = True  # Set to False if memory is not a concern
    
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_subject, subject_id, ratings_df, use_batch_processing): subject_id 
            for subject_id in subjects
        }
        
        for future in tqdm(futures, desc="Overall Progress"):
            subject_id = futures[future]
            try:
                result = future.result()
                results.append((subject_id, result))
            except Exception as e:
                logger.error(f"Subject {subject_id} processing failed: {str(e)}")
                results.append((subject_id, False))
    
    # Report summary
    successful = sum(1 for _, res in results if res)
    logger.info(f"Processing complete. {successful}/{len(subjects)} subjects processed successfully.")
    
    # List failed subjects
    failed = [s for s, res in results if not res]
    if failed:
        logger.warning(f"Failed subjects: {failed}")

if __name__ == "__main__":
    main()