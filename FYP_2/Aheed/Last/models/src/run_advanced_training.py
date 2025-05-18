import os
import subprocess
import argparse
from datetime import datetime

def main(args):
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_base, f"advanced_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct command with optimal parameters
    cmd = [
        "python", "train_advanced.py",
        "--data_dir", args.data_dir,
        "--output_dir", output_dir,
        "--batch_size", "32",
        "--epochs", "150",
        "--lr", "5e-5",
        "--weight_decay", "1e-6",
        "--patience", "25",
        "--lambda_schedule", "linear",
        "--mixup_alpha", "0.2",
        "--label_smoothing", "0.1",
        "--seed", "42"
    ]
    
    # Add wandb flag if requested
    if args.use_wandb:
        cmd.append("--use_wandb")
    
    # Print command
    print("Running command:")
    print(" ".join(cmd))
    
    # Run command
    subprocess.run(cmd)
    
    # After training completes, run evaluation
    eval_cmd = [
        "python", "evaluate_advanced.py",
        "--model_path", os.path.join(output_dir, "best_model_acc.pth"),
        "--output_dir", os.path.join(output_dir, "evaluation"),
        "--split", "test",
        "--batch_size", "32"
    ]
    
    print("\nRunning evaluation:")
    print(" ".join(eval_cmd))
    
    subprocess.run(eval_cmd)
    
    print(f"\nTraining and evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run advanced EEG emotion recognition training with optimal parameters")
    
    parser.add_argument('--data_dir', type=str, 
                        default=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
                        help='Directory with processed data')
    parser.add_argument('--output_base', type=str, default="./outputs", 
                        help='Base directory for outputs')
    parser.add_argument('--use_wandb', action='store_true',
                        help='Use Weights & Biases for logging')
    
    args = parser.parse_args()
    
    main(args)
