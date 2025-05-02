# evaluate.py
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, f1_score, recall_score
from model import EmotionClassifier
from dataset import DEAPDataset
from train import train, evaluate  # Use the improved training/evaluation functions

def loso_evaluation(processed_dir=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed", 
                   target="valence", 
                   epochs=50, 
                   batch_size=64):
    """Enhanced LOSO cross-validation with proper resource management"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = []
    
    # Load dataset once
    full_dataset = DEAPDataset(processed_dir, target=target)
    subjects = full_dataset.get_subject_ids()
    groups = [int(s[1:]) for s in subjects]  # Convert "s01" to 1
    
    logo = LeaveOneGroupOut()
    
    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(full_dataset, groups=groups)):
        print(f"\n=== Processing Fold {fold_idx+1}/{len(subjects)} ===")
        
        # Initialize fresh model each fold
        model = EmotionClassifier().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        
        # Create splits
        train_set = Subset(full_dataset, train_idx)
        test_set = Subset(full_dataset, test_idx)
        
        # Train with early stopping support
        best_model = train(
            model=model,
            optimizer=optimizer,
            train_dataset=train_set,
            epochs=epochs,
            device=device,
            val_dataset=test_set  # Use test set for early stopping
        )
        
        # Final evaluation
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        acc, f1 = evaluate(best_model, test_loader, device)
        
        results.append({
            "subject": subjects[fold_idx],
            "accuracy": acc,
            "f1_score": f1
        })
        print(f"Subject {subjects[fold_idx]} - Acc: {acc:.4f}, F1: {f1:.4f}")
    
    # Aggregate results
    print("\n=== Final LOSO Results ===")
    print(f"Average Accuracy: {np.mean([r['accuracy'] for r in results]):.4f}")
    print(f"Average F1-Score: {np.mean([r['f1_score'] for r in results]):.4f}")
    print(f"UAR: {np.mean([r['accuracy'] for r in results]):.4f}")  # Unweighted Average Recall
    
    return results

if __name__ == "__main__":
    loso_evaluation()