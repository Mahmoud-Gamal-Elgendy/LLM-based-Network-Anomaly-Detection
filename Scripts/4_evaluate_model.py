"""
================================================================================
MODULE 4: MODEL EVALUATION
================================================================================

Purpose:
    This module evaluates the fine-tuned phi-2 model on network anomaly detection.
    It provides comprehensive metrics and explainability analysis.

What this script does:
    1. Loads the fine-tuned phi-2 model with LoRA adapters
    2. Runs inference on the test dataset
    3. Computes classification metrics (Accuracy, Precision, Recall, F1)
    4. Generates confusion matrix and visualizations
    5. Tests explainability (can the model explain its decisions?)
    6. Provides sample predictions with confidence scores
    7. Compares performance across attack categories

Why we need this:
    - Validates that fine-tuning actually improved the model
    - Identifies which attack types are detected well vs. poorly
    - Demonstrates explainability for 6G Native AI requirements
    - Provides publication-ready metrics for research papers
    - Helps debug model behavior and identify failure modes

================================================================================
"""

import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from tqdm import tqdm

# HuggingFace libraries
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Sklearn for metrics
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    
    # Model paths
    base_model_id: str = "microsoft/phi-2"
    adapter_path: str = "./Models/phi2-kdd-lora/final"
    
    # Test data
    test_json: str = "./Data/test_textified_detailed_instruction.json"
    
    # Generation parameters
    max_new_tokens: int = 50  # Max tokens to generate
    temperature: float = 0.1  # Low temperature for deterministic outputs
    top_p: float = 0.9
    do_sample: bool = True  # Use sampling (with low temp)
    
    # Batch processing
    batch_size: int = 8  # Process multiple samples at once
    
    # Output
    output_dir: str = "./Results"
    save_predictions: bool = True  # Save all predictions to file
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Attack category mapping (must match training)
CATEGORY_LABELS = ['dos', 'normal', 'probe', 'r2l', 'u2r']

CATEGORY_DESCRIPTIONS = {
    'normal': 'Normal Network Activity',
    'dos': 'Denial of Service Attack',
    'probe': 'Network Reconnaissance/Probe',
    'r2l': 'Remote to Local Attack',
    'u2r': 'User to Root Privilege Escalation'
}


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model(config: EvalConfig):
    """
    Load the fine-tuned model with LoRA adapters.
    
    Args:
        config (EvalConfig): Evaluation configuration
    
    Returns:
        tuple: (model, tokenizer)
    """
    print(f"\n{'='*70}")
    print("LOADING FINE-TUNED MODEL")
    print(f"{'='*70}")
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_id,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    print(f"✓ Tokenizer loaded")
    
    # Load base model
    print("\n[2/3] Loading base phi-2 model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.base_model_id,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16
    )
    print(f"✓ Base model loaded")
    
    # Load LoRA adapters
    print("\n[3/3] Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, config.adapter_path)
    model.eval()  # Set to evaluation mode
    print(f"✓ LoRA adapters loaded from: {config.adapter_path}")
    
    print(f"\n✓ Model ready for inference on: {config.device}")
    
    return model, tokenizer


# ============================================================================
# INFERENCE
# ============================================================================

def format_prompt_for_inference(item: Dict) -> str:
    """
    Format a test item for inference (without the output).
    
    Args:
        item (Dict): Test data item
    
    Returns:
        str: Formatted prompt
    """
    prompt = f"""### Instruction:
{item['instruction']}

### Input:
{item['input']}

### Output:
"""
    return prompt


def extract_prediction(generated_text: str, prompt: str) -> str:
    """
    Extract the model's prediction from generated text.
    
    Args:
        generated_text (str): Full generated text
        prompt (str): Original prompt
    
    Returns:
        str: Extracted prediction (attack category)
    """
    # Remove the prompt to get only the generated part
    generated_only = generated_text[len(prompt):].strip()
    
    # Extract the category (look for keywords)
    generated_lower = generated_only.lower()
    
    # Try to find explicit "classification: <category>"
    if "classification:" in generated_lower:
        # Extract what comes after "classification:"
        classification_part = generated_lower.split("classification:")[-1].strip()
        
        # Check for each category
        for category in CATEGORY_LABELS:
            if category in classification_part[:20]:  # Look in first 20 chars
                return category
    
    # Fallback: look for category keywords anywhere in response
    for category in CATEGORY_LABELS:
        if category in generated_lower:
            return category
    
    # If no category found, return "unknown"
    return "unknown"


def run_inference(model, tokenizer, test_data: List[Dict], config: EvalConfig) -> List[Dict]:
    """
    Run inference on test dataset.
    
    Args:
        model: Fine-tuned model
        tokenizer: Tokenizer
        test_data (List[Dict]): Test dataset
        config (EvalConfig): Evaluation configuration
    
    Returns:
        List[Dict]: Predictions with metadata
    """
    print(f"\n{'='*70}")
    print("RUNNING INFERENCE ON TEST SET")
    print(f"{'='*70}")
    print(f"\nTest samples: {len(test_data):,}")
    print(f"Batch size: {config.batch_size}")
    
    predictions = []
    
    # Process in batches for efficiency
    for i in tqdm(range(0, len(test_data), config.batch_size), desc="Inference"):
        batch = test_data[i:i + config.batch_size]
        
        # Format prompts
        prompts = [format_prompt_for_inference(item) for item in batch]
        
        # Tokenize
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(config.device)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                top_p=config.top_p,
                do_sample=config.do_sample,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        # Extract predictions
        for j, (item, prompt, generated) in enumerate(zip(batch, prompts, generated_texts)):
            prediction = extract_prediction(generated, prompt)
            
            predictions.append({
                'true_label': item['attack_category'],
                'predicted_label': prediction,
                'original_attack': item['original_label'],
                'protocol': item['protocol'],
                'service': item['service'],
                'generated_text': generated[len(prompt):].strip(),
                'full_input': item['input']
            })
    
    print(f"\n✓ Inference complete: {len(predictions):,} predictions")
    
    return predictions


# ============================================================================
# METRICS CALCULATION
# ============================================================================

def calculate_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        predictions (List[Dict]): Model predictions
    
    Returns:
        Dict: Calculated metrics
    """
    print(f"\n{'='*70}")
    print("CALCULATING METRICS")
    print(f"{'='*70}")
    
    # Extract labels
    y_true = [p['true_label'] for p in predictions]
    y_pred = [p['predicted_label'] for p in predictions]
    
    # Overall accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        labels=CATEGORY_LABELS,
        average=None,
        zero_division=0
    )
    
    # Macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1)
    
    # Weighted averages (accounts for class imbalance)
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, 
        y_pred,
        labels=CATEGORY_LABELS,
        average='weighted',
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=CATEGORY_LABELS)
    
    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class': {
            label: {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1': float(f1[i]),
                'support': int(support[i])
            }
            for i, label in enumerate(CATEGORY_LABELS)
        },
        'confusion_matrix': cm.tolist()
    }
    
    return metrics


def print_metrics(metrics: Dict):
    """
    Pretty-print evaluation metrics.
    
    Args:
        metrics (Dict): Calculated metrics
    """
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS")
    print(f"{'='*70}")
    
    # Overall metrics
    print(f"\n{'─'*70}")
    print("Overall Performance:")
    print(f"{'─'*70}")
    print(f"  Accuracy:           {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"\n  Macro Averages:")
    print(f"    Precision:        {metrics['macro_precision']:.4f}")
    print(f"    Recall:           {metrics['macro_recall']:.4f}")
    print(f"    F1-Score:         {metrics['macro_f1']:.4f}")
    print(f"\n  Weighted Averages (accounts for class imbalance):")
    print(f"    Precision:        {metrics['weighted_precision']:.4f}")
    print(f"    Recall:           {metrics['weighted_recall']:.4f}")
    print(f"    F1-Score:         {metrics['weighted_f1']:.4f}")
    
    # Per-class metrics
    print(f"\n{'─'*70}")
    print("Per-Class Performance:")
    print(f"{'─'*70}")
    print(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
    print(f"{'─'*70}")
    
    for label in CATEGORY_LABELS:
        metrics_class = metrics['per_class'][label]
        print(f"{label:<15} "
              f"{metrics_class['precision']:>10.4f}  "
              f"{metrics_class['recall']:>10.4f}  "
              f"{metrics_class['f1']:>10.4f}  "
              f"{metrics_class['support']:>8,}")
    
    print(f"{'─'*70}")


def plot_confusion_matrix(metrics: Dict, output_dir: str):
    """
    Plot and save confusion matrix.
    
    Args:
        metrics (Dict): Metrics containing confusion matrix
        output_dir (str): Directory to save plot
    """
    print(f"\n{'='*70}")
    print("Generating Confusion Matrix Visualization...")
    print(f"{'='*70}")
    
    cm = np.array(metrics['confusion_matrix'])
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=[CATEGORY_DESCRIPTIONS[l] for l in CATEGORY_LABELS],
        yticklabels=[CATEGORY_DESCRIPTIONS[l] for l in CATEGORY_LABELS],
        cbar_kws={'label': 'Count'}
    )
    
    plt.title('Confusion Matrix - Network Anomaly Detection\n(phi-2 Fine-tuned with LoRA)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir) / "confusion_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved to: {output_path}")
    
    plt.close()


def plot_per_class_metrics(metrics: Dict, output_dir: str):
    """
    Plot per-class metrics (Precision, Recall, F1).
    
    Args:
        metrics (Dict): Calculated metrics
        output_dir (str): Directory to save plot
    """
    print(f"\nGenerating Per-Class Metrics Visualization...")
    
    categories = CATEGORY_LABELS
    precision = [metrics['per_class'][c]['precision'] for c in categories]
    recall = [metrics['per_class'][c]['recall'] for c in categories]
    f1 = [metrics['per_class'][c]['f1'] for c in categories]
    
    x = np.arange(len(categories))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, precision, width, label='Precision', color='skyblue')
    bars2 = ax.bar(x, recall, width, label='Recall', color='lightcoral')
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='lightgreen')
    
    ax.set_xlabel('Attack Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics\n(phi-2 Fine-tuned with LoRA)', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([CATEGORY_DESCRIPTIONS[c] for c in categories], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    autolabel(bars1)
    autolabel(bars2)
    autolabel(bars3)
    
    plt.tight_layout()
    
    output_path = Path(output_dir) / "per_class_metrics.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Per-class metrics plot saved to: {output_path}")
    
    plt.close()


# ============================================================================
# EXPLAINABILITY ANALYSIS
# ============================================================================

def analyze_explainability(predictions: List[Dict], n_samples: int = 5):
    """
    Analyze model's ability to explain its decisions.
    
    Args:
        predictions (List[Dict]): Model predictions
        n_samples (int): Number of samples to display per category
    """
    print(f"\n{'='*70}")
    print("EXPLAINABILITY ANALYSIS (XAI)")
    print(f"{'='*70}")
    print("\nSample predictions showing model reasoning:\n")
    
    # Get samples from each category
    for category in CATEGORY_LABELS:
        # Find correct predictions for this category
        category_preds = [
            p for p in predictions 
            if p['true_label'] == category and p['predicted_label'] == category
        ]
        
        if not category_preds:
            continue
        
        print(f"\n{'─'*70}")
        print(f"Category: {CATEGORY_DESCRIPTIONS[category].upper()}")
        print(f"{'─'*70}")
        
        # Show a few examples
        for i, pred in enumerate(category_preds[:min(n_samples, len(category_preds))]):
            print(f"\n[Example {i+1}]")
            print(f"True Label: {pred['true_label']}")
            print(f"Predicted: {pred['predicted_label']}")
            print(f"Original Attack Type: {pred['original_attack']}")
            print(f"\nInput (first 150 chars):")
            print(f"  {pred['full_input'][:150]}...")
            print(f"\nModel's Explanation:")
            print(f"  {pred['generated_text']}")
            
            if i < min(n_samples, len(category_preds)) - 1:
                print(f"\n{'-'*70}")


def analyze_failure_cases(predictions: List[Dict], n_samples: int = 3):
    """
    Analyze cases where the model failed.
    
    Args:
        predictions (List[Dict]): Model predictions
        n_samples (int): Number of failure cases to show
    """
    print(f"\n{'='*70}")
    print("FAILURE CASE ANALYSIS")
    print(f"{'='*70}")
    
    # Find misclassifications
    failures = [p for p in predictions if p['true_label'] != p['predicted_label']]
    
    print(f"\nTotal misclassifications: {len(failures)} out of {len(predictions)} "
          f"({len(failures)/len(predictions)*100:.2f}%)")
    
    if not failures:
        print("\n✓ Perfect classification! No failures to analyze.")
        return
    
    # Analyze failure patterns
    print(f"\nMost Common Misclassification Patterns:")
    failure_patterns = {}
    for f in failures:
        pattern = (f['true_label'], f['predicted_label'])
        failure_patterns[pattern] = failure_patterns.get(pattern, 0) + 1
    
    sorted_patterns = sorted(failure_patterns.items(), key=lambda x: x[1], reverse=True)
    
    for (true, pred), count in sorted_patterns[:5]:
        percentage = (count / len(failures)) * 100
        print(f"  {true} → {pred}: {count} cases ({percentage:.1f}% of failures)")
    
    # Show example failures
    print(f"\n{'─'*70}")
    print(f"Example Failure Cases:")
    print(f"{'─'*70}")
    
    for i, failure in enumerate(failures[:n_samples]):
        print(f"\n[Failure {i+1}]")
        print(f"True Label: {failure['true_label']}")
        print(f"Predicted: {failure['predicted_label']}")
        print(f"Original Attack: {failure['original_attack']}")
        print(f"\nInput (first 150 chars):")
        print(f"  {failure['full_input'][:150]}...")
        print(f"\nModel Output:")
        print(f"  {failure['generated_text']}")
        
        if i < n_samples - 1:
            print(f"\n{'-'*70}")


# ============================================================================
# SAVE RESULTS
# ============================================================================

def save_results(metrics: Dict, predictions: List[Dict], output_dir: str):
    """
    Save all results to files.
    
    Args:
        metrics (Dict): Calculated metrics
        predictions (List[Dict]): All predictions
        output_dir (str): Output directory
    """
    print(f"\n{'='*70}")
    print("Saving Results...")
    print(f"{'='*70}")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics as JSON
    metrics_file = output_path / "metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to: {metrics_file}")
    
    # Save predictions
    predictions_file = output_path / "predictions.json"
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    print(f"✓ Predictions saved to: {predictions_file}")
    
    # Save readable report
    report_file = output_path / "evaluation_report.txt"
    with open(report_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NETWORK ANOMALY DETECTION - EVALUATION REPORT\n")
        f.write("Model: phi-2 (2.7B) Fine-tuned with LoRA\n")
        f.write("="*70 + "\n\n")
        
        f.write("OVERALL METRICS:\n")
        f.write(f"  Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"  Macro F1: {metrics['macro_f1']:.4f}\n")
        f.write(f"  Weighted F1: {metrics['weighted_f1']:.4f}\n\n")
        
        f.write("PER-CLASS PERFORMANCE:\n")
        f.write(f"{'Category':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Support':<10}\n")
        f.write("-"*70 + "\n")
        
        for label in CATEGORY_LABELS:
            m = metrics['per_class'][label]
            f.write(f"{label:<15} {m['precision']:>10.4f}  {m['recall']:>10.4f}  "
                   f"{m['f1']:>10.4f}  {m['support']:>8,}\n")
    
    print(f"✓ Report saved to: {report_file}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Main evaluation pipeline.
    """
    print("\n" + "="*70)
    print(" "*15 + "MODEL EVALUATION PIPELINE")
    print(" "*10 + "Network Anomaly Detection (6G-AIOps)")
    print("="*70)
    
    # Configuration
    config = EvalConfig()
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load Model
    # ========================================================================
    model, tokenizer = load_model(config)
    
    # ========================================================================
    # STEP 2: Load Test Data
    # ========================================================================
    print(f"\n{'='*70}")
    print("Loading Test Data...")
    print(f"{'='*70}")
    
    with open(config.test_json, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    print(f"✓ Loaded {len(test_data):,} test samples")
    
    # ========================================================================
    # STEP 3: Run Inference
    # ========================================================================
    predictions = run_inference(model, tokenizer, test_data, config)
    
    # ========================================================================
    # STEP 4: Calculate Metrics
    # ========================================================================
    metrics = calculate_metrics(predictions)
    print_metrics(metrics)
    
    # ========================================================================
    # STEP 5: Generate Visualizations
    # ========================================================================
    plot_confusion_matrix(metrics, config.output_dir)
    plot_per_class_metrics(metrics, config.output_dir)
    
    # ========================================================================
    # STEP 6: Explainability Analysis
    # ========================================================================
    analyze_explainability(predictions, n_samples=2)
    
    # ========================================================================
    # STEP 7: Failure Analysis
    # ========================================================================
    analyze_failure_cases(predictions, n_samples=3)
    
    # ========================================================================
    # STEP 8: Save Results
    # ========================================================================
    save_results(metrics, predictions, config.output_dir)
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    print(f"\n{'='*70}")
    print(" "*20 + "EVALUATION COMPLETE!")
    print(f"{'='*70}")
    print(f"\nResults saved to: {config.output_dir}/")
    print(f"\nKey Findings:")
    print(f"  - Overall Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"  - Weighted F1-Score: {metrics['weighted_f1']:.4f}")
    print(f"  - Best performing category: {max(metrics['per_class'].items(), key=lambda x: x[1]['f1'])[0]}")
    print(f"\nNext Steps:")
    print(f"  1. Review visualizations in: {config.output_dir}/")
    print(f"  2. Analyze failure cases for model improvement")
    print(f"  3. Consider hyperparameter tuning if results are unsatisfactory")
    print(f"  4. Deploy model for real-time network monitoring")
    print(f"\n{'='*70}\n")


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run the evaluation pipeline.
    
    Usage:
        python 4_evaluate_model.py
    
    Requirements:
        - Trained model in Models/phi2-kdd-lora/final/
        - Test data in Data/test_textified_detailed_instruction.json
        - GPU recommended (but can run on CPU slowly)
    """
    main()
