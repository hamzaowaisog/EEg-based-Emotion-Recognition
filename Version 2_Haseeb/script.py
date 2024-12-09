import mne
import numpy as np
import gc
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
from torcheeg.datasets import SEEDDataset
from scipy.signal import butter, lfilter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

# -------------------------------
# Load SEED Dataset
# -------------------------------
raw_dataset = SEEDDataset(
    root_path='D:/FAST/EEg-based-Emotion-Recognition/Preprocessed_EEG',
    io_path='D:/FAST/EEg-based-Emotion-Recognition/.torcheeg/datasets_1733174610032_5iJyS',
    online_transform=None,
    label_transform=None,
    num_worker=4
)

# Check the first sample
raw_sample = raw_dataset[0]
print(f"Raw EEG data shape: {raw_sample[0].shape}")

# -------------------------------
# Bandpass Filter
# -------------------------------
def bandpass_filter(data, lowcut=4, highcut=47, fs=200, order=4):
    """Applies a bandpass filter to the EEG data."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, data, axis=-1)

def filter_dataset(dataset, lowcut=4, highcut=47, fs=200):
    """Applies the bandpass filter to each sample in the dataset."""
    filtered_dataset = []
    for sample, metadata in dataset:
        filtered_sample = bandpass_filter(sample, lowcut, highcut, fs)
        filtered_dataset.append((filtered_sample, metadata))
    return filtered_dataset

# Apply bandpass filtering
filtered_dataset = filter_dataset(raw_dataset)
print(f"Filtered dataset size: {len(filtered_dataset)}")

# -------------------------------
# ICA Processing
# -------------------------------
def process_sample_with_ica(sample, metadata, fs=200, n_components=28, max_iter=800):
    """Processes a single EEG sample using ICA for artifact removal."""
    logging.info(f"Starting ICA processing for metadata: {metadata}")
    try:
        info = mne.create_info(
            ch_names=[f'ch_{i}' for i in range(sample.shape[0])],
            sfreq=fs,
            ch_types='eeg'
        )
        raw = mne.io.RawArray(sample, info)
        logging.info(f"Raw data created for metadata: {metadata}")

        # Fit ICA
        ica = mne.preprocessing.ICA(n_components=n_components, random_state=42, max_iter=max_iter, method='fastica')
        logging.info(f"Fitting ICA for sample with metadata: {metadata}")
        ica.fit(raw)

        # Apply ICA
        logging.info(f"Applying ICA for sample with metadata: {metadata}")
        raw_cleaned = ica.apply(raw)

        logging.info(f"ICA completed successfully for metadata: {metadata}")
        return raw_cleaned.get_data(), metadata

    except ValueError as e:
        logging.warning(f"ValueError: {e} for sample with metadata: {metadata}. Skipping...")
    except Exception as e:
        logging.error(f"Unexpected error: {e} for sample with metadata: {metadata}. Skipping...")
    return None

def parallel_process_ica(dataset, fs=200, n_components=28, max_iter=800, batch_size=5, num_workers=2):
    """Processes the dataset in batches using parallel ICA."""
    cleaned_dataset = []
    logging.info(f"Starting ICA processing with n_components={n_components}, max_iter={max_iter}, batch_size={batch_size}, num_workers={num_workers}...")

    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        batch = dataset[start:end]
        logging.info(f"Processing batch {start // batch_size + 1} with {len(batch)} samples...")

        # Execute tasks in parallel using ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(process_sample_with_ica, sample, metadata, fs, n_components, max_iter): (sample, metadata) for sample, metadata in batch}

            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    cleaned_dataset.append(result)

        # Clean up memory
        del batch, futures
        gc.collect()

    logging.info("ICA processing completed.")
    return cleaned_dataset

def save_results(cleaned_dataset, filename="cleaned_dataset.pkl"):
    """Saves the cleaned dataset to a file using pickle."""
    with open(filename, "wb") as f:
        pickle.dump(cleaned_dataset, f)
    logging.info(f"Cleaned dataset saved to {filename}")

# -------------------------------
# Execute the Script
# -------------------------------
if __name__ == "__main__":
    # Process a small test subset
    test_dataset = filtered_dataset[:10]
    cleaned_dataset = parallel_process_ica(test_dataset, fs=200, n_components=28, max_iter=800, batch_size=5, num_workers=2)
    save_results(cleaned_dataset, "test_cleaned_dataset.pkl")
