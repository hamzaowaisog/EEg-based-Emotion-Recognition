import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import argparse
from tqdm import tqdm
import json
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score
from torch.utils.data import DataLoader
import sys

# Add the models directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'FYP_2', 'Aheed', 'Last', 'models', 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_model(model_path, device='cuda'):
    """Load a trained model"""
    try:
        # Try to load the model
        checkpoint = torch.load(model_path, map_location=device)

        # Check if it's a state dict or a full checkpoint
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            # Try to determine model type from the keys
            if any('eeg_domain_adapter' in k for k in state_dict.keys()):
                from advanced_model import MultiSourceContrastiveModel
                model = MultiSourceContrastiveModel(
                    eeg_input_dim=32,
                    face_input_dim=768,
                    hidden_dim=128,
                    output_dim=2,
                    num_subjects=32,
                    temperature=0.5,
                    dropout=0.5
                ).to(device)
            else:
                from model import EmotionClassifier
                model = EmotionClassifier(
                    eeg_input_dim=32,
                    face_input_dim=768,
                    hidden_dim=128,
                    output_dim=2,
                    dropout=0.5
                ).to(device)

            model.load_state_dict(state_dict)
        else:
            # Assume it's the model itself
            model = checkpoint

        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def analyze_model_architecture(model):
    """Analyze model architecture and parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Get parameter statistics by layer
    layer_stats = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            layer = name.split('.')[0]
            if layer not in layer_stats:
                layer_stats[layer] = {
                    'count': 0,
                    'mean': [],
                    'std': [],
                    'min': [],
                    'max': [],
                    'zero_fraction': []
                }

            layer_stats[layer]['count'] += param.numel()
            layer_stats[layer]['mean'].append(param.data.mean().item())
            layer_stats[layer]['std'].append(param.data.std().item())
            layer_stats[layer]['min'].append(param.data.min().item())
            layer_stats[layer]['max'].append(param.data.max().item())
            layer_stats[layer]['zero_fraction'].append((param.data == 0).float().mean().item())

    # Aggregate statistics
    for layer in layer_stats:
        for stat in ['mean', 'std', 'min', 'max', 'zero_fraction']:
            if layer_stats[layer][stat]:
                layer_stats[layer][stat] = np.mean(layer_stats[layer][stat])

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'layer_stats': layer_stats
    }

def visualize_model_parameters(model_stats, output_dir):
    """Visualize model parameter statistics"""
    os.makedirs(output_dir, exist_ok=True)

    # Layer parameter counts
    layers = list(model_stats['layer_stats'].keys())
    counts = [model_stats['layer_stats'][layer]['count'] for layer in layers]

    plt.figure(figsize=(12, 6))
    plt.bar(layers, counts)
    plt.title('Parameter Count by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Parameter Count')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_counts.png'))
    plt.close()

    # Parameter statistics by layer
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    means = [model_stats['layer_stats'][layer]['mean'] for layer in layers]
    plt.bar(layers, means)
    plt.title('Mean Parameter Value by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Mean Value')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    stds = [model_stats['layer_stats'][layer]['std'] for layer in layers]
    plt.bar(layers, stds)
    plt.title('Parameter Standard Deviation by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    mins = [model_stats['layer_stats'][layer]['min'] for layer in layers]
    maxs = [model_stats['layer_stats'][layer]['max'] for layer in layers]
    plt.bar(layers, maxs, label='Max')
    plt.bar(layers, mins, label='Min')
    plt.title('Parameter Min/Max Values by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    zero_fractions = [model_stats['layer_stats'][layer]['zero_fraction'] for layer in layers]
    plt.bar(layers, zero_fractions)
    plt.title('Fraction of Zero Parameters by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Zero Fraction')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'parameter_statistics.png'))
    plt.close()

def analyze_model_gradients(model, dataloader, criterion, device='cuda'):
    """Analyze model gradients during a forward/backward pass"""
    model.train()

    # Get a batch of data
    batch = next(iter(dataloader))
    eeg = batch['eeg'].to(device)
    face = batch['face'].to(device)
    labels = batch['valence'].to(device)

    # Zero gradients
    model.zero_grad()

    # Forward pass
    outputs = model(eeg, face)

    # Handle different output formats
    if isinstance(outputs, tuple):
        logits, eeg_proj, face_proj = outputs
        loss = criterion(logits, eeg_proj, face_proj, labels)
    elif isinstance(outputs, dict):
        logits = outputs['logits']
        loss, _ = criterion(outputs, labels, torch.zeros_like(labels))
    else:
        logits = outputs
        loss = F.cross_entropy(logits, labels)

    # Backward pass
    loss.backward()

    # Collect gradient statistics
    grad_stats = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            layer = name.split('.')[0]
            if layer not in grad_stats:
                grad_stats[layer] = {
                    'mean': [],
                    'std': [],
                    'min': [],
                    'max': [],
                    'zero_fraction': []
                }

            grad_stats[layer]['mean'].append(param.grad.abs().mean().item())
            grad_stats[layer]['std'].append(param.grad.std().item())
            grad_stats[layer]['min'].append(param.grad.min().item())
            grad_stats[layer]['max'].append(param.grad.max().item())
            grad_stats[layer]['zero_fraction'].append((param.grad == 0).float().mean().item())

    # Aggregate statistics
    for layer in grad_stats:
        for stat in ['mean', 'std', 'min', 'max', 'zero_fraction']:
            if grad_stats[layer][stat]:
                grad_stats[layer][stat] = np.mean(grad_stats[layer][stat])

    return grad_stats

def visualize_model_gradients(grad_stats, output_dir):
    """Visualize model gradient statistics"""
    os.makedirs(output_dir, exist_ok=True)

    layers = list(grad_stats.keys())

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    means = [grad_stats[layer]['mean'] for layer in layers]
    plt.bar(layers, means)
    plt.title('Mean Gradient Magnitude by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Mean Magnitude')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    stds = [grad_stats[layer]['std'] for layer in layers]
    plt.bar(layers, stds)
    plt.title('Gradient Standard Deviation by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    mins = [grad_stats[layer]['min'] for layer in layers]
    maxs = [grad_stats[layer]['max'] for layer in layers]
    plt.bar(layers, maxs, label='Max')
    plt.bar(layers, mins, label='Min')
    plt.title('Gradient Min/Max Values by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 4)
    zero_fractions = [grad_stats[layer]['zero_fraction'] for layer in layers]
    plt.bar(layers, zero_fractions)
    plt.title('Fraction of Zero Gradients by Layer')
    plt.xlabel('Layer')
    plt.ylabel('Zero Fraction')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gradient_statistics.png'))
    plt.close()

def analyze_model_activations(model, dataloader, device='cuda'):
    """Analyze model activations during forward pass"""
    model.eval()

    # Get a batch of data
    batch = next(iter(dataloader))
    eeg = batch['eeg'].to(device)
    face = batch['face'].to(device)

    # Register hooks to capture activations
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            # Handle different output types
            if isinstance(output, tuple):
                output = output[0]
            elif isinstance(output, dict):
                output = list(output.values())[0]

            activations[name] = output.detach()
        return hook

    # Register hooks for all modules
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(get_activation(name)))

    # Forward pass
    with torch.no_grad():
        _ = model(eeg, face)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Collect activation statistics
    activation_stats = {}
    for name, activation in activations.items():
        if len(activation.shape) > 0:  # Skip scalar activations
            activation_stats[name] = {
                'mean': activation.abs().mean().item(),
                'std': activation.std().item(),
                'min': activation.min().item(),
                'max': activation.max().item(),
                'zero_fraction': (activation == 0).float().mean().item(),
                'shape': list(activation.shape)
            }

    return activation_stats

def visualize_model_activations(activation_stats, output_dir):
    """Visualize model activation statistics"""
    os.makedirs(output_dir, exist_ok=True)

    # Sort layers by depth (assuming naming convention with dots)
    layers = sorted(activation_stats.keys(), key=lambda x: len(x.split('.')))

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    means = [activation_stats[layer]['mean'] for layer in layers]
    plt.bar(range(len(layers)), means)
    plt.title('Mean Activation Magnitude by Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Mean Magnitude')
    plt.xticks(range(len(layers)), [f"{i}" for i in range(len(layers))], rotation=45)

    plt.subplot(2, 2, 2)
    stds = [activation_stats[layer]['std'] for layer in layers]
    plt.bar(range(len(layers)), stds)
    plt.title('Activation Standard Deviation by Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Standard Deviation')
    plt.xticks(range(len(layers)), [f"{i}" for i in range(len(layers))], rotation=45)

    plt.subplot(2, 2, 3)
    mins = [activation_stats[layer]['min'] for layer in layers]
    maxs = [activation_stats[layer]['max'] for layer in layers]
    plt.bar(range(len(layers)), maxs, label='Max')
    plt.bar(range(len(layers)), mins, label='Min')
    plt.title('Activation Min/Max Values by Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Value')
    plt.legend()
    plt.xticks(range(len(layers)), [f"{i}" for i in range(len(layers))], rotation=45)

    plt.subplot(2, 2, 4)
    zero_fractions = [activation_stats[layer]['zero_fraction'] for layer in layers]
    plt.bar(range(len(layers)), zero_fractions)
    plt.title('Fraction of Zero Activations by Layer')
    plt.xlabel('Layer Index')
    plt.ylabel('Zero Fraction')
    plt.xticks(range(len(layers)), [f"{i}" for i in range(len(layers))], rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'activation_statistics.png'))
    plt.close()

    # Create a layer mapping file
    with open(os.path.join(output_dir, 'layer_mapping.json'), 'w') as f:
        json.dump({str(i): layer for i, layer in enumerate(layers)}, f, indent=2)

def analyze_model_performance(model, dataloader, device='cuda'):
    """Analyze model performance on a dataset"""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating model"):
            eeg = batch['eeg'].to(device)
            face = batch['face'].to(device)
            labels = batch['valence'].to(device)

            # Forward pass
            outputs = model(eeg, face)

            # Handle different output formats
            if isinstance(outputs, tuple):
                logits = outputs[0]
            elif isinstance(outputs, dict):
                logits = outputs['logits']
            else:
                logits = outputs

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    conf_mat = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'confusion_matrix': conf_mat.tolist(),
        'classification_report': class_report,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs
    }

def visualize_model_performance(performance, output_dir):
    """Visualize model performance metrics"""
    os.makedirs(output_dir, exist_ok=True)

    # Confusion matrix
    plt.figure(figsize=(8, 6))
    conf_mat = np.array(performance['confusion_matrix'])
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

    # Class-wise metrics
    plt.figure(figsize=(10, 6))
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {
        '0': [performance['classification_report']['0'][m] for m in metrics],
        '1': [performance['classification_report']['1'][m] for m in metrics]
    }

    x = np.arange(len(metrics))
    width = 0.35

    plt.bar(x - width/2, class_metrics['0'], width, label='Negative')
    plt.bar(x + width/2, class_metrics['1'], width, label='Positive')

    plt.title('Class-wise Performance Metrics')
    plt.xlabel('Metric')
    plt.ylabel('Score')
    plt.xticks(x, metrics)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_metrics.png'))
    plt.close()

    # Prediction probabilities
    plt.figure(figsize=(10, 6))
    probs = np.array(performance['probabilities'])
    labels = np.array(performance['labels'])

    # Probability distribution for each class
    for i in range(2):
        class_probs = probs[labels == i, i]
        plt.hist(class_probs, alpha=0.5, bins=20, label=f"Class {i}")

    plt.title('Prediction Probability Distribution')
    plt.xlabel('Probability')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'probability_distribution.png'))
    plt.close()

def analyze_model(model_path, data_dir, output_dir, target='valence'):
    """Analyze model architecture, parameters, and performance"""
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path, device)
    if model is None:
        logger.error("Failed to load model")
        return

    # Load dataset
    logger.info(f"Loading dataset from {data_dir}")
    try:
        from dataset import DEAPDataset
        dataset = DEAPDataset(
            processed_dir=data_dir,
            target=target,
            apply_augmentation=False,
            balance_classes=False
        )

        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
            num_workers=0
        )
    except Exception as e:
        logger.error(f"Error loading dataset: {str(e)}")
        return

    # Analyze model architecture
    logger.info("Analyzing model architecture")
    model_stats = analyze_model_architecture(model)
    visualize_model_parameters(model_stats, output_dir)

    # Analyze model activations
    logger.info("Analyzing model activations")
    activation_stats = analyze_model_activations(model, dataloader, device)
    visualize_model_activations(activation_stats, output_dir)

    # Analyze model gradients
    logger.info("Analyzing model gradients")
    try:
        from losses import HybridLoss
        criterion = HybridLoss(alpha=0.5, beta=0.2, temperature=0.5)
        grad_stats = analyze_model_gradients(model, dataloader, criterion, device)
        visualize_model_gradients(grad_stats, output_dir)
    except Exception as e:
        logger.warning(f"Error analyzing gradients: {str(e)}")

    # Analyze model performance
    logger.info("Analyzing model performance")
    performance = analyze_model_performance(model, dataloader, device)
    visualize_model_performance(performance, output_dir)

    # Save analysis results - convert numpy types to Python native types
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
    report_data = {
        'model_stats': {
            'total_params': model_stats['total_params'],
            'trainable_params': model_stats['trainable_params']
        },
        'performance': {
            'accuracy': performance['accuracy'],
            'f1_score': performance['f1_score']
        }
    }

    serializable_report = json.loads(json.dumps(report_data, default=convert_to_serializable))

    with open(os.path.join(output_dir, 'model_analysis.json'), 'w') as f:
        json.dump(serializable_report, f, indent=2)

    logger.info(f"Analysis complete. Results saved to {output_dir}")

    # Generate recommendations
    generate_recommendations(model_stats, activation_stats, performance, output_dir)

def generate_recommendations(model_stats, activation_stats, performance, output_dir):
    """Generate recommendations for improving model performance"""
    recommendations = []

    # Check model complexity
    if model_stats['total_params'] < 1_000_000:
        recommendations.append({
            'category': 'Model Complexity',
            'issue': 'Model may be too simple for the task',
            'recommendation': 'Consider increasing model capacity by adding more layers or increasing hidden dimensions'
        })
    elif model_stats['total_params'] > 10_000_000:
        recommendations.append({
            'category': 'Model Complexity',
            'issue': 'Model may be too complex and prone to overfitting',
            'recommendation': 'Consider reducing model size or adding more regularization'
        })

    # Check for vanishing/exploding activations
    activation_means = [stats['mean'] for stats in activation_stats.values()]
    if any(mean < 0.01 for mean in activation_means):
        recommendations.append({
            'category': 'Vanishing Activations',
            'issue': 'Some layers have very small activation values',
            'recommendation': 'Check initialization, consider using residual connections or layer normalization'
        })
    if any(mean > 10 for mean in activation_means):
        recommendations.append({
            'category': 'Exploding Activations',
            'issue': 'Some layers have very large activation values',
            'recommendation': 'Check for proper normalization, consider using batch normalization or weight normalization'
        })

    # Check for class imbalance in performance
    class_report = performance['classification_report']
    class_f1_scores = [class_report[str(i)]['f1-score'] for i in range(2)]
    if max(class_f1_scores) - min(class_f1_scores) > 0.2:
        recommendations.append({
            'category': 'Class Imbalance',
            'issue': 'Large difference in F1 scores between classes',
            'recommendation': 'Improve class balancing during training, adjust class weights, or use focal loss'
        })

    # Check overall performance
    if performance['accuracy'] < 0.6:
        recommendations.append({
            'category': 'Low Performance',
            'issue': 'Model accuracy is below 60%',
            'recommendation': 'Consider more advanced architectures, better feature extraction, or ensemble methods'
        })

    # Check for overfitting
    # (This would require training metrics, which we don't have here)

    # Save recommendations - convert numpy types to Python native types
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    # Convert the recommendations to serializable format
    serializable_recommendations = json.loads(json.dumps(recommendations, default=convert_to_serializable))

    with open(os.path.join(output_dir, 'recommendations.json'), 'w') as f:
        json.dump(serializable_recommendations, f, indent=2)

    # Generate recommendation report
    with open(os.path.join(output_dir, 'recommendations.txt'), 'w') as f:
        f.write("# Model Improvement Recommendations\n\n")

        for i, rec in enumerate(recommendations, 1):
            f.write(f"## {i}. {rec['category']}\n")
            f.write(f"**Issue**: {rec['issue']}\n\n")
            f.write(f"**Recommendation**: {rec['recommendation']}\n\n")

        # Add general recommendations
        f.write("## General Recommendations\n\n")
        f.write("1. **Data Augmentation**: Implement more advanced EEG-specific augmentation techniques\n")
        f.write("2. **Feature Engineering**: Extract more informative features from raw EEG data\n")
        f.write("3. **Cross-Subject Adaptation**: Improve domain adaptation techniques for better cross-subject generalization\n")
        f.write("4. **Ensemble Methods**: Combine multiple models for better performance\n")
        f.write("5. **Hyperparameter Tuning**: Systematically optimize hyperparameters\n")
        f.write("6. **Advanced Architectures**: Explore state-of-the-art architectures like Graph Neural Networks for EEG\n")

def main():
    parser = argparse.ArgumentParser(description="Analyze DEAP emotion recognition model")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model")
    parser.add_argument("--data_dir", type=str,
                        default=r"C:\Users\tahir\Documents\EEg-based-Emotion-Recognition\FYP_2\Aheed\Last\data\processed",
                        help="Directory with processed data")
    parser.add_argument("--output_dir", type=str, default="./model_analysis",
                        help="Output directory for analysis results")
    parser.add_argument("--target", type=str, default="valence", choices=["valence", "arousal"],
                        help="Target emotion to analyze")

    args = parser.parse_args()

    # Analyze model
    analyze_model(args.model_path, args.data_dir, args.output_dir, args.target)

if __name__ == "__main__":
    main()
