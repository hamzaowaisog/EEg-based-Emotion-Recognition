import os
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import pandas as pd
from collections import defaultdict, Counter
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def validate_eeg_data(eeg_path, expected_shape=(32,)):
    """Validate EEG data file"""
    try:
        data = np.load(eeg_path)

        # Check shape
        if data.shape != expected_shape:
            return False, f"Invalid shape: {data.shape}, expected {expected_shape}"

        # Check for NaN or Inf values
        if np.isnan(data).any() or np.isinf(data).any():
            return False, f"Contains NaN or Inf values"

        # Check for zero variance
        if np.var(data) < 1e-10:
            return False, f"Near-zero variance: {np.var(data)}"

        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"

def validate_face_data(face_path, expected_shape=(768,)):
    """Validate facial feature data file"""
    try:
        data = np.load(face_path)

        # Check shape
        if data.shape != expected_shape:
            return False, f"Invalid shape: {data.shape}, expected {expected_shape}"

        # Check for NaN or Inf values
        if np.isnan(data).any() or np.isinf(data).any():
            return False, f"Contains NaN or Inf values"

        # Check for zero vectors (all zeros)
        if np.all(data == 0):
            return False, f"All zeros (no face detected)"

        # Check for very low variance
        if np.var(data) < 1e-10:
            return False, f"Near-zero variance: {np.var(data)}"

        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"

def validate_metadata(metadata_path):
    """Validate metadata file"""
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        # Check required fields
        required_fields = ["valence", "arousal", "has_video"]
        missing_fields = [field for field in required_fields if field not in metadata]

        if missing_fields:
            return False, f"Missing fields: {missing_fields}"

        # Check valence/arousal values
        if metadata["valence"] not in [0, 1, -1]:
            return False, f"Invalid valence value: {metadata['valence']}"

        if metadata["arousal"] not in [0, 1, -1]:
            return False, f"Invalid arousal value: {metadata['arousal']}"

        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"

def analyze_class_distribution(processed_dir, target='valence'):
    """Analyze class distribution in the dataset"""
    class_counts = Counter()
    subject_class_counts = defaultdict(Counter)

    # Iterate through all subject directories
    for subj_dir in sorted(Path(processed_dir).glob("s*")):
        if not subj_dir.is_dir():
            continue

        subject_id = subj_dir.name

        # Iterate through all trial directories
        for trial_dir in sorted(subj_dir.glob("trial*")):
            if not trial_dir.is_dir():
                continue

            # Check metadata
            metadata_path = trial_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                # Count classes
                if metadata[target] != -1:  # Skip invalid labels
                    class_counts[metadata[target]] += 1
                    subject_class_counts[subject_id][metadata[target]] += 1
            except Exception as e:
                logger.warning(f"Error reading metadata {metadata_path}: {str(e)}")

    return class_counts, subject_class_counts

def analyze_feature_statistics(processed_dir):
    """Analyze statistics of EEG and face features"""
    eeg_stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'has_zeros': [],
        'has_nan': [],
        'has_inf': []
    }

    face_stats = {
        'mean': [],
        'std': [],
        'min': [],
        'max': [],
        'has_zeros': [],
        'has_nan': [],
        'has_inf': [],
        'all_zeros': []
    }

    # Iterate through all subject directories
    for subj_dir in sorted(Path(processed_dir).glob("s*")):
        if not subj_dir.is_dir():
            continue

        # Iterate through all trial directories
        for trial_dir in sorted(subj_dir.glob("trial*")):
            if not trial_dir.is_dir():
                continue

            # Check EEG data
            eeg_path = trial_dir / "eeg.npy"
            if eeg_path.exists():
                try:
                    eeg_data = np.load(eeg_path)
                    eeg_stats['mean'].append(np.mean(eeg_data))
                    eeg_stats['std'].append(np.std(eeg_data))
                    eeg_stats['min'].append(np.min(eeg_data))
                    eeg_stats['max'].append(np.max(eeg_data))
                    eeg_stats['has_zeros'].append(np.any(eeg_data == 0))
                    eeg_stats['has_nan'].append(np.isnan(eeg_data).any())
                    eeg_stats['has_inf'].append(np.isinf(eeg_data).any())
                except Exception as e:
                    logger.warning(f"Error analyzing EEG data {eeg_path}: {str(e)}")

            # Check face data
            face_path = trial_dir / "face.npy"
            if face_path.exists():
                try:
                    face_data = np.load(face_path)
                    face_stats['mean'].append(np.mean(face_data))
                    face_stats['std'].append(np.std(face_data))
                    face_stats['min'].append(np.min(face_data))
                    face_stats['max'].append(np.max(face_data))
                    face_stats['has_zeros'].append(np.any(face_data == 0))
                    face_stats['has_nan'].append(np.isnan(face_data).any())
                    face_stats['has_inf'].append(np.isinf(face_data).any())
                    face_stats['all_zeros'].append(np.all(face_data == 0))
                except Exception as e:
                    logger.warning(f"Error analyzing face data {face_path}: {str(e)}")

    return eeg_stats, face_stats

def visualize_feature_distributions(eeg_stats, face_stats, output_dir):
    """Visualize distributions of feature statistics"""
    os.makedirs(output_dir, exist_ok=True)

    # EEG feature distributions
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot(eeg_stats['mean'], kde=True)
    plt.title('EEG Feature Mean Distribution')

    plt.subplot(2, 2, 2)
    sns.histplot(eeg_stats['std'], kde=True)
    plt.title('EEG Feature Std Distribution')

    plt.subplot(2, 2, 3)
    sns.histplot(eeg_stats['min'], kde=True)
    plt.title('EEG Feature Min Distribution')

    plt.subplot(2, 2, 4)
    sns.histplot(eeg_stats['max'], kde=True)
    plt.title('EEG Feature Max Distribution')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eeg_feature_distributions.png'))
    plt.close()

    # Face feature distributions
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    sns.histplot([x for x, all_zero in zip(face_stats['mean'], face_stats['all_zeros']) if not all_zero], kde=True)
    plt.title('Face Feature Mean Distribution (Non-zero)')

    plt.subplot(2, 2, 2)
    sns.histplot([x for x, all_zero in zip(face_stats['std'], face_stats['all_zeros']) if not all_zero], kde=True)
    plt.title('Face Feature Std Distribution (Non-zero)')

    plt.subplot(2, 2, 3)
    sns.histplot([x for x, all_zero in zip(face_stats['min'], face_stats['all_zeros']) if not all_zero], kde=True)
    plt.title('Face Feature Min Distribution (Non-zero)')

    plt.subplot(2, 2, 4)
    sns.histplot([x for x, all_zero in zip(face_stats['max'], face_stats['all_zeros']) if not all_zero], kde=True)
    plt.title('Face Feature Max Distribution (Non-zero)')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'face_feature_distributions.png'))
    plt.close()

    # Summary statistics
    plt.figure(figsize=(10, 6))
    plt.bar(['EEG NaN', 'EEG Inf', 'EEG Zeros', 'Face NaN', 'Face Inf', 'Face All Zeros'],
            [sum(eeg_stats['has_nan']), sum(eeg_stats['has_inf']), sum(eeg_stats['has_zeros']),
             sum(face_stats['has_nan']), sum(face_stats['has_inf']), sum(face_stats['all_zeros'])])
    plt.title('Data Quality Issues')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_quality_issues.png'))
    plt.close()

def visualize_class_distribution(class_counts, subject_class_counts, output_dir, target='valence'):
    """Visualize class distribution"""
    os.makedirs(output_dir, exist_ok=True)

    # Overall class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title(f'Overall {target.capitalize()} Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'overall_{target}_distribution.png'))
    plt.close()

    # Per-subject class distribution
    plt.figure(figsize=(15, 10))
    subjects = list(subject_class_counts.keys())
    neg_counts = [subject_class_counts[s][0] for s in subjects]
    pos_counts = [subject_class_counts[s][1] for s in subjects]

    x = np.arange(len(subjects))
    width = 0.35

    plt.bar(x - width/2, neg_counts, width, label='Negative')
    plt.bar(x + width/2, pos_counts, width, label='Positive')

    plt.xlabel('Subject')
    plt.ylabel('Count')
    plt.title(f'Per-Subject {target.capitalize()} Class Distribution')
    plt.xticks(x, subjects, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'per_subject_{target}_distribution.png'))
    plt.close()

    # Class imbalance per subject
    imbalance = []
    for subject in subjects:
        counts = subject_class_counts[subject]
        if 0 in counts and 1 in counts and counts[0] > 0 and counts[1] > 0:
            ratio = counts[1] / counts[0]
            imbalance.append((subject, ratio))

    if imbalance:
        imbalance.sort(key=lambda x: x[1])
        subjects, ratios = zip(*imbalance)

        plt.figure(figsize=(15, 8))
        plt.bar(subjects, ratios)
        plt.axhline(y=1.0, color='r', linestyle='-', alpha=0.3)
        plt.xlabel('Subject')
        plt.ylabel('Positive/Negative Ratio')
        plt.title(f'Class Imbalance Ratio per Subject ({target.capitalize()})')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'class_imbalance_{target}.png'))
        plt.close()

def collect_features_for_visualization(processed_dir, target='valence', max_samples=1000):
    """Collect features for visualization"""
    eeg_features = []
    face_features = []
    labels = []
    subject_ids = []

    # Iterate through all subject directories
    for subj_dir in sorted(Path(processed_dir).glob("s*")):
        if not subj_dir.is_dir():
            continue

        subject_id = subj_dir.name

        # Iterate through all trial directories
        for trial_dir in sorted(subj_dir.glob("trial*")):
            if not trial_dir.is_dir():
                continue

            # Check metadata
            metadata_path = trial_dir / "metadata.json"
            if not metadata_path.exists():
                continue

            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)

                if metadata[target] == -1:  # Skip invalid labels
                    continue

                # Load EEG data
                eeg_path = trial_dir / "eeg.npy"
                if not eeg_path.exists():
                    continue

                eeg_data = np.load(eeg_path)

                # Load face data if available
                face_path = trial_dir / "face.npy"
                if face_path.exists():
                    face_data = np.load(face_path)

                    # Skip if face data is all zeros
                    if np.all(face_data == 0):
                        continue
                else:
                    continue  # Skip if no face data

                # Collect features
                eeg_features.append(eeg_data)
                face_features.append(face_data)
                labels.append(metadata[target])
                subject_ids.append(subject_id)

                # Limit the number of samples
                if len(eeg_features) >= max_samples:
                    break
            except Exception as e:
                logger.warning(f"Error collecting features from {trial_dir}: {str(e)}")

        # Limit the number of samples
        if len(eeg_features) >= max_samples:
            break

    return np.array(eeg_features), np.array(face_features), np.array(labels), subject_ids

def visualize_feature_space(eeg_features, face_features, labels, subject_ids, output_dir):
    """Visualize feature space using PCA and t-SNE"""
    os.makedirs(output_dir, exist_ok=True)

    # Convert subject IDs to integers for coloring
    subject_int = [int(s[1:]) if s[0] == 's' and s[1:].isdigit() else 0 for s in subject_ids]

    # PCA for EEG features
    if len(eeg_features) > 0:
        pca = PCA(n_components=2)
        eeg_pca = pca.fit_transform(eeg_features)

        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(eeg_pca[:, 0], eeg_pca[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label='Class')
        plt.title('PCA of EEG Features (Colored by Class)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

        plt.subplot(2, 2, 2)
        scatter = plt.scatter(eeg_pca[:, 0], eeg_pca[:, 1], c=subject_int, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Subject')
        plt.title('PCA of EEG Features (Colored by Subject)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

        # t-SNE for EEG features (if enough samples)
        if len(eeg_features) >= 50:
            try:
                tsne = TSNE(n_components=2, random_state=42)
                eeg_tsne = tsne.fit_transform(eeg_features)

                plt.subplot(2, 2, 3)
                scatter = plt.scatter(eeg_tsne[:, 0], eeg_tsne[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
                plt.colorbar(scatter, label='Class')
                plt.title('t-SNE of EEG Features (Colored by Class)')

                plt.subplot(2, 2, 4)
                scatter = plt.scatter(eeg_tsne[:, 0], eeg_tsne[:, 1], c=subject_int, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Subject')
                plt.title('t-SNE of EEG Features (Colored by Subject)')
            except Exception as e:
                logger.warning(f"Error in t-SNE for EEG features: {str(e)}")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'eeg_feature_visualization.png'))
        plt.close()

    # PCA for face features
    if len(face_features) > 0:
        pca = PCA(n_components=2)
        face_pca = pca.fit_transform(face_features)

        plt.figure(figsize=(12, 10))
        plt.subplot(2, 2, 1)
        scatter = plt.scatter(face_pca[:, 0], face_pca[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
        plt.colorbar(scatter, label='Class')
        plt.title('PCA of Face Features (Colored by Class)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

        plt.subplot(2, 2, 2)
        scatter = plt.scatter(face_pca[:, 0], face_pca[:, 1], c=subject_int, cmap='viridis', alpha=0.7)
        plt.colorbar(scatter, label='Subject')
        plt.title('PCA of Face Features (Colored by Subject)')
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')

        # t-SNE for face features (if enough samples)
        if len(face_features) >= 50:
            try:
                tsne = TSNE(n_components=2, random_state=42)
                face_tsne = tsne.fit_transform(face_features)

                plt.subplot(2, 2, 3)
                scatter = plt.scatter(face_tsne[:, 0], face_tsne[:, 1], c=labels, cmap='coolwarm', alpha=0.7)
                plt.colorbar(scatter, label='Class')
                plt.title('t-SNE of Face Features (Colored by Class)')

                plt.subplot(2, 2, 4)
                scatter = plt.scatter(face_tsne[:, 0], face_tsne[:, 1], c=subject_int, cmap='viridis', alpha=0.7)
                plt.colorbar(scatter, label='Subject')
                plt.title('t-SNE of Face Features (Colored by Subject)')
            except Exception as e:
                logger.warning(f"Error in t-SNE for face features: {str(e)}")

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'face_feature_visualization.png'))
        plt.close()

def validate_processed_data(processed_dir, output_dir, target='valence'):
    """Validate all processed data and generate report"""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize counters and results
    total_trials = 0
    valid_eeg = 0
    valid_face = 0
    valid_metadata = 0

    eeg_issues = defaultdict(int)
    face_issues = defaultdict(int)
    metadata_issues = defaultdict(int)

    # Iterate through all subject directories
    for subj_dir in tqdm(sorted(Path(processed_dir).glob("s*")), desc="Validating subjects"):
        if not subj_dir.is_dir():
            continue

        # Iterate through all trial directories
        for trial_dir in sorted(subj_dir.glob("trial*")):
            if not trial_dir.is_dir():
                continue

            total_trials += 1

            # Validate EEG data
            eeg_path = trial_dir / "eeg.npy"
            if eeg_path.exists():
                is_valid, message = validate_eeg_data(eeg_path)
                if is_valid:
                    valid_eeg += 1
                else:
                    eeg_issues[message] += 1
            else:
                eeg_issues["File missing"] += 1

            # Validate face data
            face_path = trial_dir / "face.npy"
            if face_path.exists():
                is_valid, message = validate_face_data(face_path)
                if is_valid:
                    valid_face += 1
                else:
                    face_issues[message] += 1
            else:
                face_issues["File missing"] += 1

            # Validate metadata
            metadata_path = trial_dir / "metadata.json"
            if metadata_path.exists():
                is_valid, message = validate_metadata(metadata_path)
                if is_valid:
                    valid_metadata += 1
                else:
                    metadata_issues[message] += 1
            else:
                metadata_issues["File missing"] += 1

    # Analyze class distribution
    class_counts, subject_class_counts = analyze_class_distribution(processed_dir, target)

    # Analyze feature statistics
    eeg_stats, face_stats = analyze_feature_statistics(processed_dir)

    # Collect features for visualization
    eeg_features, face_features, labels, subject_ids = collect_features_for_visualization(processed_dir, target)

    # Generate visualizations
    visualize_class_distribution(class_counts, subject_class_counts, output_dir, target)
    visualize_feature_distributions(eeg_stats, face_stats, output_dir)
    visualize_feature_space(eeg_features, face_features, labels, subject_ids, output_dir)

    # Generate summary report
    report = {
        "total_trials": total_trials,
        "valid_eeg": valid_eeg,
        "valid_face": valid_face,
        "valid_metadata": valid_metadata,
        "eeg_issues": dict(eeg_issues),
        "face_issues": dict(face_issues),
        "metadata_issues": dict(metadata_issues),
        "class_distribution": dict(class_counts),
        "eeg_stats_summary": {
            "mean": np.mean(eeg_stats['mean']),
            "std": np.mean(eeg_stats['std']),
            "has_nan": sum(eeg_stats['has_nan']),
            "has_inf": sum(eeg_stats['has_inf']),
            "has_zeros": sum(eeg_stats['has_zeros'])
        },
        "face_stats_summary": {
            "mean": np.mean([x for x, all_zero in zip(face_stats['mean'], face_stats['all_zeros']) if not all_zero]),
            "std": np.mean([x for x, all_zero in zip(face_stats['std'], face_stats['all_zeros']) if not all_zero]),
            "has_nan": sum(face_stats['has_nan']),
            "has_inf": sum(face_stats['has_inf']),
            "all_zeros": sum(face_stats['all_zeros'])
        }
    }

    # Save report as JSON - convert numpy types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert the report to serializable format
    serializable_report = json.loads(json.dumps(report, default=convert_to_serializable))

    with open(os.path.join(output_dir, 'validation_report.json'), 'w') as f:
        json.dump(serializable_report, f, indent=2)

    # Print summary
    logger.info(f"Total trials: {total_trials}")
    logger.info(f"Valid EEG data: {valid_eeg}/{total_trials} ({valid_eeg/total_trials*100:.1f}%)")
    logger.info(f"Valid face data: {valid_face}/{total_trials} ({valid_face/total_trials*100:.1f}%)")
    logger.info(f"Valid metadata: {valid_metadata}/{total_trials} ({valid_metadata/total_trials*100:.1f}%)")
    logger.info(f"Class distribution ({target}): {dict(class_counts)}")

    return report

def main():
    parser = argparse.ArgumentParser(description="Validate processed DEAP dataset")
    parser.add_argument("--processed_dir", type=str,
                        default=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
                        help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="./validation_results",
                        help="Output directory for validation results")
    parser.add_argument("--target", type=str, default="valence", choices=["valence", "arousal"],
                        help="Target emotion to analyze")

    args = parser.parse_args()

    # Validate processed data
    report = validate_processed_data(args.processed_dir, args.output_dir, args.target)

    logger.info(f"Validation complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
