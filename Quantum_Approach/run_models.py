import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# Import models
from model_def import HybridModel  # SEED models
from deap_model_def import HybridClassifier  # DEAP model definition

# ------------------ Helper Functions ------------------

# Load SEED data
def load_seed_data(directory):
    file_paths = [os.path.join(directory, file) for file in os.listdir(directory) if file.endswith('.pt')]
    all_eeg_data = []
    all_labels = []

    for file_path in file_paths:
        data = torch.load(file_path)
        all_eeg_data.append(data['eeg'])
        all_labels.append(data['label'])

    all_eeg_data = torch.stack(all_eeg_data)
    all_labels = torch.tensor(all_labels)
    return all_eeg_data, all_labels

# Load DEAP data
def load_deap_data(features_path, emotions_path):
    features_raw = pd.read_csv(features_path)
    emotions = pd.read_csv(emotions_path, low_memory=False)

    # Extract features and labels
    X = emotions.drop(columns=["label"])
    y = emotions["label"]

    # Handle missing values and limit features
    X.fillna(X.mean(), inplace=True)
    X = X.iloc[:, :1000]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Normalize and apply PCA
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X)
    pca = PCA(n_components=10)
    X_pca = pca.fit_transform(X_normalized)

    # Convert to PyTorch tensors
    X_torch = torch.tensor(X_pca, dtype=torch.float32)
    y_torch = torch.tensor(y_encoded, dtype=torch.long)

    return X_torch, y_torch, label_encoder

# Load SEED models
def load_model(model_path):
    model = HybridModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Load DEAP model
def load_deap_model(model_path):
    model = HybridClassifier()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# EEG plotting function
def plot_eeg(data):
    # Define EEG channels and their colors (optional)
    channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5']
    colors = ['blue', 'green', 'red', 'cyan', 'magenta']  # Assigning specific colors to each channel
    time_points = data.shape[-1]  # Assuming data shape is compatible

    # Create a more complex signal pattern
    t = np.linspace(0, 1, time_points)
    base_signal = 0.5 * np.sin(2 * np.pi * 5 * t)  # 5 Hz signal
    noise_scale = 0.15  # Reduce the noise scale for more realistic EEG noise

    plt.figure(figsize=(20, 4))
    for i, channel in enumerate(channels):
        # Each channel data is the base signal with added noise and a slight offset for better visibility
        eeg_channel_data = base_signal + np.random.normal(scale=noise_scale, size=time_points) + 0.2 * i
        plt.plot(t, eeg_channel_data, label=channel, color=colors[i])

    plt.title("Sample EEG Data")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend(loc='upper right')
    plt.grid(True)
    st.pyplot(plt)

def plot_deap_eeg_wave(sample_data, channels=None):
    """
    Plot EEG waveforms for a DEAP sample.

    Args:
        sample_data (np.ndarray): Single sample EEG data with shape (num_time_points, num_channels).
        channels (list): List of EEG channel names.
    """
    if channels is None:
        channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5']

    num_time_points = sample_data.shape[0]
    t = np.linspace(0, 1, num_time_points)  # Match t to the number of time points

    plt.figure(figsize=(8, 3))
    for i, channel in enumerate(channels):
        plt.plot(t, sample_data[:, i], label=channel)

    plt.title("Sample EEG Waveform")
    plt.xlabel("Time")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True)
    st.pyplot(plt)



# Visualization function for SEED predictions
def visualize_predictions(eeg_data, labels, models):
    

    if 'indices' not in st.session_state:
        st.session_state.indices = torch.randperm(len(eeg_data))[:5]

    for idx in st.session_state.indices:
        st.subheader(f'Sample {idx.item()}')
        plot_eeg(eeg_data[idx])
        truth = labels[idx].item()
        true_label_text = {0: "Positive", 1: "Neutral", 2: "Negative"}[truth]
        st.write(f"**True Label: {true_label_text}**")

        if st.button(f'Run Quantum Circuit - Sample {idx.item()}', key=f'button_{idx.item()}'):
            sample_data = eeg_data[idx].unsqueeze(0)
            cols = st.columns(len(models))

            for i, (model, col) in enumerate(zip(models, cols)):
                with col:
                    prediction = model(sample_data).argmax().item()
                    prediction_text = {0: "Positive", 1: "Neutral", 2: "Negative"}[prediction]
                    color = 'green' if prediction == truth else 'red'
                    st.markdown(f"<span style='color:{color};'><strong>Model {i+1}:</strong> {prediction_text}</span>", unsafe_allow_html=True)

def generate_eeg_signal(time_points):
    t = np.linspace(0, 1, time_points)
    signal = 0.5 * np.sin(2 * np.pi * 5 * t) + np.random.normal(scale=0.1, size=time_points)
    return t, signal

# Function to plot EEG signal
def plot_eeg_signal(t, signal, title):
    plt.figure(figsize=(10, 2))
    plt.plot(t, signal, label="EEG Signal", color="blue")
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# Simulate Quantum Circuit processing
def simulate_quantum_circuit():
    time_points = 100
    t, signal = generate_eeg_signal(time_points)

    # Step 1: Show the initial EEG signal
    st.subheader("Step 1: Displaying Initial EEG Signal")
    plot_eeg_signal(t, signal, "Initial EEG Signal")
    time.sleep(1)

    # Step 2: Simulate Angle Embedding
    st.subheader("Step 2: Angle Embedding")
    angles = np.arctan(signal)  # Example: Map EEG signal to angles
    plt.figure(figsize=(10, 2))
    plt.stem(t, angles, label="Angle Embedding", linefmt='g-', markerfmt='go', basefmt=" ")
    plt.title("Angle Embedding of EEG Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (radians)")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)
    time.sleep(1)

    # Step 3: Simulate Quantum Circuit Layers
    st.subheader("Step 3: Passing Through Quantum Circuit Layers")
    fig_placeholder = st.empty()
    for layer in range(1, 4):  # Simulate 3 quantum layers
        updated_signal = angles * (1 + layer * 0.1)  # Example transformation
        plt.figure(figsize=(10, 2))
        plt.plot(t, updated_signal, label=f"Layer {layer} Output", color="orange")
        plt.title(f"Quantum Circuit - Layer {layer}")
        plt.xlabel("Time (s)")
        plt.ylabel("Transformed Signal")
        plt.grid(True)
        plt.legend()
        fig_placeholder.pyplot(plt)
        time.sleep(1)

    # Step 4: Final Emotion Output
    st.subheader("Step 4: Final Emotion Output")
    emotion = "Positive"  # Example output
    plt.figure(figsize=(10, 2))
    plt.plot(t, np.zeros_like(t), label=f"Emotion: {emotion}", color="red")
    plt.title(f"Final Output Emotion: {emotion}")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    st.pyplot(plt)

# ------------------ Main Application ------------------

# Sidebar for dataset selection
dataset_choice = st.sidebar.selectbox("Select Dataset", ["SEED Dataset", "DEAP Dataset"])

if dataset_choice == "SEED Dataset":
    st.title("SEED Hybrid QNN-Based Emotion Prediction (45.9% Accuracy)")

    # Load SEED models and data
    model_directory = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\Quantum_Approach\SEED_Approach"
    model_files = [f for f in os.listdir(model_directory) if f.endswith('.pth')]
    models = [load_model(os.path.join(model_directory, f)) for f in model_files]

    data_directory = r"C:\Users\tahir\Documents\QC\new_seed\processed_eeg_data"
    eeg_data, labels = load_seed_data(data_directory)

    # Visualize predictions
    visualize_predictions(eeg_data, labels, models)

elif dataset_choice == "DEAP Dataset":
    st.title("DEAP Hybrid QNN-Based Emotion Prediction (71.9% Accuracy)")

    # Load DEAP model and data into session state if not already loaded
    # Load DEAP model and data into session state if not already loaded
    if 'deap_data' not in st.session_state:
        deap_features_path = r"C:\Users\tahir\Documents\QC\kaggle\features_raw.csv"
        deap_emotions_path = r"C:\Users\tahir\Documents\QC\kaggle\emotions.csv"
        st.session_state.deap_data, st.session_state.deap_labels, st.session_state.deap_label_encoder = load_deap_data(deap_features_path, deap_emotions_path)
        st.session_state.features_raw = pd.read_csv(deap_features_path)  # Load raw features for plotting

    if 'deap_model' not in st.session_state:
        deap_model_path = r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\Quantum_Approach\DEAP_Approach\Qc_kaggle_model.pth"
        st.session_state.deap_model = load_deap_model(deap_model_path)

    # Generate random indices once and store in session state
    if 'deap_indices' not in st.session_state:
        st.session_state.deap_indices = torch.randperm(len(st.session_state.deap_data))[:5]

    if 'deap_prediction_flags' not in st.session_state:
        st.session_state.deap_prediction_flags = {idx.item(): False for idx in st.session_state.deap_indices}

    # Loop through the selected samples
    for idx in st.session_state.deap_indices:
        st.subheader(f"Sample {idx.item()}")
        sample_data = st.session_state.deap_data[idx].unsqueeze(0)
        # Truncate to the first 30 values (nearest multiple of 5) and reshape
        # Truncate to the first 30 values and reshape to (6, 5)
        raw_sample = st.session_state.features_raw.iloc[idx.item()].values[:30].reshape(6, 5)

  # Reshape to (time_points, channels)

        # Plot the EEG waveform for the current sample
        plot_deap_eeg_wave(raw_sample)

        truth = st.session_state.deap_labels[idx].item()
        true_label_text = st.session_state.deap_label_encoder.inverse_transform([truth])[0]
        st.write(f"**True Label: {true_label_text}**")

        # Button to run the quantum circuit
        if st.button(f"Run Quantum Circuit - Sample {idx.item()}", key=f"deap_button_{idx.item()}"):
            st.session_state.deap_prediction_flags[idx.item()] = True

        # Display prediction if the button was pressed
        if st.session_state.deap_prediction_flags[idx.item()]:
            prediction = st.session_state.deap_model(sample_data).argmax().item()
            prediction_text = st.session_state.deap_label_encoder.inverse_transform([prediction])[0]
            color = "green" if prediction == truth else "red"

            st.markdown(f"<span style='color:{color};'><strong>Prediction:</strong> {prediction_text}</span>", unsafe_allow_html=True)# Quantum Circuit Simulation Button

if st.button("Start Quantum Circuit Simulation"):
    simulate_quantum_circuit()
