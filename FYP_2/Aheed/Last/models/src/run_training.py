import os
import subprocess
import argparse
from datetime import datetime

def main(args):
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Choose which script to run
    if args.mode == 'balanced':
        script = "train_balanced.py"
    elif args.mode == 'advanced':
        script = "train_advanced_simple.py"
    else:
        print(f"Unknown mode: {args.mode}")
        return
    
    # Run the selected script
    cmd = ["python", script]
    
    print(f"Running {script}...")
    subprocess.run(cmd)
    
    print(f"\nTraining completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run EEG emotion recognition training")
    
    parser.add_argument('--mode', type=str, default='balanced', choices=['balanced', 'advanced'],
                        help='Training mode: balanced or advanced')
    
    args = parser.parse_args()
    
    main(args)
