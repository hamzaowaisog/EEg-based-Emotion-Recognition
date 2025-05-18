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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_eeg_data(eeg_path):
    """Check EEG data file for validity"""
    try:
        data = np.load(eeg_path)

        # Check shape
        if len(data.shape) != 1 or data.shape[0] != 32:
            return False, f"Invalid shape: {data.shape}, expected (32,)"

        # Check for NaN or Inf values
        if np.isnan(data).any() or np.isinf(data).any():
            return False, f"Contains NaN or Inf values"

        # Check for zero variance
        if np.var(data) < 1e-10:
            return False, f"Near-zero variance: {np.var(data)}"

        # Check for reasonable range
        if np.max(np.abs(data)) > 100:
            return False, f"Contains extreme values: max abs = {np.max(np.abs(data))}"

        return True, "Valid"
    except Exception as e:
        return False, f"Error: {str(e)}"

def check_face_data(face_path):
    """Check facial feature data file for validity"""
    try:
        data = np.load(face_path)

        # Check shape
        if len(data.shape) != 1 or data.shape[0] != 768:
            return False, f"Invalid shape: {data.shape}, expected (768,)"

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

def check_metadata(metadata_path):
    """Check metadata file for validity"""
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

def check_processed_data(processed_dir, output_dir):
    """Check all processed data and generate report"""
    os.makedirs(output_dir, exist_ok=True)

    # Initialize counters and results
    total_trials = 0
    valid_eeg = 0
    valid_face = 0
    valid_metadata = 0

    eeg_issues = defaultdict(int)
    face_issues = defaultdict(int)
    metadata_issues = defaultdict(int)

    # Track class distribution
    valence_counts = Counter()
    arousal_counts = Counter()

    # Track per-subject statistics
    subject_stats = defaultdict(lambda: {
        'total': 0,
        'valid_eeg': 0,
        'valid_face': 0,
        'valid_metadata': 0,
        'valence_counts': Counter(),
        'arousal_counts': Counter()
    })

    # Iterate through all subject directories
    for subj_dir in tqdm(sorted(Path(processed_dir).glob("s*")), desc="Checking subjects"):
        if not subj_dir.is_dir():
            continue

        subject_id = subj_dir.name

        # Iterate through all trial directories
        for trial_dir in sorted(subj_dir.glob("trial*")):
            if not trial_dir.is_dir():
                continue

            total_trials += 1
            subject_stats[subject_id]['total'] += 1

            # Check EEG data
            eeg_path = trial_dir / "eeg.npy"
            if eeg_path.exists():
                is_valid, message = check_eeg_data(eeg_path)
                if is_valid:
                    valid_eeg += 1
                    subject_stats[subject_id]['valid_eeg'] += 1
                else:
                    eeg_issues[message] += 1
            else:
                eeg_issues["File missing"] += 1

            # Check face data
            face_path = trial_dir / "face.npy"
            if face_path.exists():
                is_valid, message = check_face_data(face_path)
                if is_valid:
                    valid_face += 1
                    subject_stats[subject_id]['valid_face'] += 1
                else:
                    face_issues[message] += 1
            else:
                face_issues["File missing"] += 1

            # Check metadata
            metadata_path = trial_dir / "metadata.json"
            if metadata_path.exists():
                is_valid, message = check_metadata(metadata_path)
                if is_valid:
                    valid_metadata += 1
                    subject_stats[subject_id]['valid_metadata'] += 1

                    # Read metadata for class distribution
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)

                    if metadata["valence"] != -1:
                        valence_counts[metadata["valence"]] += 1
                        subject_stats[subject_id]['valence_counts'][metadata["valence"]] += 1

                    if metadata["arousal"] != -1:
                        arousal_counts[metadata["arousal"]] += 1
                        subject_stats[subject_id]['arousal_counts'][metadata["arousal"]] += 1
                else:
                    metadata_issues[message] += 1
            else:
                metadata_issues["File missing"] += 1

    # Generate summary report
    report = {
        "total_trials": total_trials,
        "valid_eeg": valid_eeg,
        "valid_face": valid_face,
        "valid_metadata": valid_metadata,
        "eeg_issues": dict(eeg_issues),
        "face_issues": dict(face_issues),
        "metadata_issues": dict(metadata_issues),
        "valence_distribution": dict(valence_counts),
        "arousal_distribution": dict(arousal_counts),
        "subject_stats": {k: {
            'total': v['total'],
            'valid_eeg': v['valid_eeg'],
            'valid_face': v['valid_face'],
            'valid_metadata': v['valid_metadata'],
            'valence_counts': dict(v['valence_counts']),
            'arousal_counts': dict(v['arousal_counts'])
        } for k, v in subject_stats.items()}
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

    with open(os.path.join(output_dir, 'data_check_report.json'), 'w') as f:
        json.dump(serializable_report, f, indent=2)

    # Generate visualizations

    # 1. Overall data validity
    plt.figure(figsize=(10, 6))
    validity_data = [
        valid_eeg / total_trials * 100,
        valid_face / total_trials * 100,
        valid_metadata / total_trials * 100
    ]
    plt.bar(['EEG', 'Face', 'Metadata'], validity_data)
    plt.title('Data Validity Percentage')
    plt.ylabel('Valid Percentage')
    plt.ylim(0, 100)
    for i, v in enumerate(validity_data):
        plt.text(i, v + 1, f"{v:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'data_validity.png'))
    plt.close()

    # 2. Class distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.bar(['Negative (0)', 'Positive (1)'], [valence_counts.get(0, 0), valence_counts.get(1, 0)])
    plt.title('Valence Distribution')
    plt.ylabel('Count')

    plt.subplot(1, 2, 2)
    plt.bar(['Low (0)', 'High (1)'], [arousal_counts.get(0, 0), arousal_counts.get(1, 0)])
    plt.title('Arousal Distribution')
    plt.ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close()

    # 3. Per-subject validity
    subjects = sorted(subject_stats.keys())
    eeg_validity = [subject_stats[s]['valid_eeg'] / subject_stats[s]['total'] * 100 for s in subjects]
    face_validity = [subject_stats[s]['valid_face'] / subject_stats[s]['total'] * 100 for s in subjects]

    plt.figure(figsize=(15, 6))
    x = np.arange(len(subjects))
    width = 0.35

    plt.bar(x - width/2, eeg_validity, width, label='EEG')
    plt.bar(x + width/2, face_validity, width, label='Face')

    plt.xlabel('Subject')
    plt.ylabel('Valid Percentage')
    plt.title('Data Validity by Subject')
    plt.xticks(x, subjects, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_validity.png'))
    plt.close()

    # 4. Per-subject class distribution (valence)
    plt.figure(figsize=(15, 6))
    neg_counts = [subject_stats[s]['valence_counts'].get(0, 0) for s in subjects]
    pos_counts = [subject_stats[s]['valence_counts'].get(1, 0) for s in subjects]

    plt.bar(x - width/2, neg_counts, width, label='Negative')
    plt.bar(x + width/2, pos_counts, width, label='Positive')

    plt.xlabel('Subject')
    plt.ylabel('Count')
    plt.title('Valence Distribution by Subject')
    plt.xticks(x, subjects, rotation=90)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'subject_valence.png'))
    plt.close()

    # Generate text report
    with open(os.path.join(output_dir, 'data_check_summary.txt'), 'w') as f:
        f.write("# DEAP Processed Data Check Summary\n\n")

        f.write(f"Total trials: {total_trials}\n")
        f.write(f"Valid EEG data: {valid_eeg}/{total_trials} ({valid_eeg/total_trials*100:.1f}%)\n")
        f.write(f"Valid face data: {valid_face}/{total_trials} ({valid_face/total_trials*100:.1f}%)\n")
        f.write(f"Valid metadata: {valid_metadata}/{total_trials} ({valid_metadata/total_trials*100:.1f}%)\n\n")

        f.write("## Class Distribution\n\n")
        f.write(f"Valence - Negative (0): {valence_counts.get(0, 0)}, Positive (1): {valence_counts.get(1, 0)}\n")
        f.write(f"Arousal - Low (0): {arousal_counts.get(0, 0)}, High (1): {arousal_counts.get(1, 0)}\n\n")

        f.write("## Common Issues\n\n")

        f.write("### EEG Issues\n")
        for issue, count in sorted(eeg_issues.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {issue}: {count} trials\n")

        f.write("\n### Face Issues\n")
        for issue, count in sorted(face_issues.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {issue}: {count} trials\n")

        f.write("\n### Metadata Issues\n")
        for issue, count in sorted(metadata_issues.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {issue}: {count} trials\n")

    # Print summary
    logger.info(f"Total trials: {total_trials}")
    logger.info(f"Valid EEG data: {valid_eeg}/{total_trials} ({valid_eeg/total_trials*100:.1f}%)")
    logger.info(f"Valid face data: {valid_face}/{total_trials} ({valid_face/total_trials*100:.1f}%)")
    logger.info(f"Valid metadata: {valid_metadata}/{total_trials} ({valid_metadata/total_trials*100:.1f}%)")
    logger.info(f"Valence distribution: Negative={valence_counts.get(0, 0)}, Positive={valence_counts.get(1, 0)}")

    return report

def main():
    parser = argparse.ArgumentParser(description="Check processed DEAP dataset")
    parser.add_argument("--processed_dir", type=str,
                        default=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
                        help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="./data_check_results",
                        help="Output directory for check results")

    args = parser.parse_args()

    # Check processed data
    report = check_processed_data(args.processed_dir, args.output_dir)

    logger.info(f"Data check complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
