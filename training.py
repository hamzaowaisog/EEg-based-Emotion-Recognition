import torch
import random
from model import CLISAEncoder, Projector, contrastive_loss
from data_preparation import enumerate_pairs

def train_clisa(eeg_data, epochs, batch_size, learning_rate, device):
    """Train the CLISA model using contrastive learning."""
    # Initialize encoder and projector
    encoder = CLISAEncoder(num_channels=32, num_spatial_filters=16, temporal_filter_size=64).to(device)
    projector = Projector(input_dim=16 * 20, latent_dim=128).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(projector.parameters()), lr=learning_rate)

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        epoch_loss = 0

        for subject_a, subject_b in enumerate_pairs(eeg_data):
            # Generate positive and negative pairs
            positive_pairs = []
            negative_pairs = []

            for label in eeg_data[subject_a]:
                if label in eeg_data[subject_b]:
                    for trial_a in eeg_data[subject_a][label]:
                        for trial_b in eeg_data[subject_b][label]:
                            positive_pairs.append((trial_a, trial_b))

                # Negative pairs
                for other_label in eeg_data[subject_b]:
                    if other_label != label:
                        for trial_a in eeg_data[subject_a][label]:
                            for trial_b in eeg_data[subject_b][other_label]:
                                negative_pairs.append((trial_a, trial_b))

            # Sample pairs for batch
            pairs = positive_pairs + negative_pairs
            random.shuffle(pairs)
            for i in range(0, len(pairs), batch_size):
                batch_pairs = pairs[i:i + batch_size]

                optimizer.zero_grad()
                batch_loss = 0

                for (x_i, x_j) in batch_pairs:
                    x_i = torch.tensor(x_i, dtype=torch.float32).unsqueeze(0).to(device)
                    x_j = torch.tensor(x_j, dtype=torch.float32).unsqueeze(0).to(device)

                    z_i = projector(encoder(x_i))
                    z_j = projector(encoder(x_j))

                    if (x_i, x_j) in positive_pairs:
                        loss = contrastive_loss(z_i, z_j)
                    else:
                        loss = contrastive_loss(-z_i, z_j)

                    batch_loss += loss

                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()

        print(f"Epoch Loss: {epoch_loss:.4f}")

    return encoder, projector
