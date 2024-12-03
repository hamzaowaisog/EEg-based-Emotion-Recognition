import mne
import matplotlib.pyplot as plt

# Path to the .cnt file
file_path = r"E:\FYP\Egg-Based Emotion Recognition\EEg-based-Emotion-Recognition\SEED\SEED_EEG\SEED_RAW_EEG\1_1.cnt"

try:
    # Load the .cnt file without preloading data
    raw = mne.io.read_raw_cnt(file_path, preload=False, data_format='int16', date_format='dd/mm/yy')

    # Crop the raw data to the first 10 seconds
    raw.crop(tmin=0, tmax=10)

    # Load the cropped data into memory
    raw.load_data()

    # Plot the first 10 seconds of EEG signals
    raw.plot(duration=1000, n_channels=10, title="EEG Signals (First 10 Seconds)")

# Extract data and time for the first 10 channels
    data, times = raw[:10, :]  # First 10 channels, all times

# # Plot the signals
#     plt.figure(figsize=(15, 10))
#     for i, channel_data in enumerate(data):
#         plt.plot(times, channel_data + i * 50, label=f"Channel {i+1}")  # Offset each channel

#     plt.xlabel('Time (s)')
#     plt.ylabel('Amplitude')
#     plt.title('EEG Signals (First 10 Channels)')
#     plt.legend()
#     plt.grid()
    plt.show()

except Exception as e:
    print(f"An error occurred: {e}")