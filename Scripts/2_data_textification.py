"""
================================================================================
MODULE 2: DATA TEXTIFICATION
================================================================================

Purpose:
    This module converts structured network traffic data into natural language
    descriptions suitable for training a Large Language Model (LLM).

What this script does:
    1. Loads processed CSV files from Module 1
    2. Converts numerical/categorical features into human-readable text
    3. Creates instruction-following format for phi-2 fine-tuning
    4. Generates three verbosity levels: basic, detailed, and rich
    5. Saves textified data as JSON files for model training

Why we need this:
    - LLMs are trained on natural language, not raw numerical features
    - Textification gives semantic meaning to network traffic patterns
    - Natural language enables the model to "reason" about network behavior
    - This is the bridge between traditional network data and AI understanding

How it works:
    We create templates that convert:
    [0, tcp, http, SF, 215, 45034, ...] 
    → "Network connection using tcp protocol on http service with SF flag. 
       Source sent 215 bytes, destination sent 45034 bytes. 
       This traffic is classified as: normal."

================================================================================
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# Feature groups for organized textification
FEATURE_GROUPS = {
    'basic': ['protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes'],
    
    'connection': ['duration', 'land', 'wrong_fragment', 'urgent'],
    
    'content': ['hot', 'num_failed_logins', 'logged_in', 'num_compromised',
                'root_shell', 'su_attempted', 'num_root'],
    
    'traffic': ['count', 'srv_count', 'serror_rate', 'srv_serror_rate',
                'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 
                'diff_srv_rate', 'srv_diff_host_rate'],
    
    'host': ['dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
             'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
             'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
             'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
             'dst_host_srv_rerror_rate']
}

# Attack category descriptions for context
ATTACK_DESCRIPTIONS = {
    'normal': 'normal network activity',
    'dos': 'denial of service attack',
    'probe': 'network reconnaissance or probing attack',
    'r2l': 'remote to local unauthorized access attempt',
    'u2r': 'user to root privilege escalation attempt'
}

# ============================================================================
# TEXTIFICATION TEMPLATES
# ============================================================================

def create_basic_description(row: pd.Series) -> str:
    """
    Create a basic text description with essential features only.
    
    This is suitable for:
    - Initial model training with limited compute
    - Quick inference
    - Models with smaller context windows
    
    Args:
        row (pd.Series): Single data record
    
    Returns:
        str: Basic text description
    """
    text = f"Network connection using {row['protocol_type']} protocol on {row['service']} service"
    
    # Add flag if meaningful
    if row['flag'] != 'SF':  # SF is the most common, normal flag
        text += f" with {row['flag']} flag"
    
    text += f". Source sent {int(row['src_bytes'])} bytes"
    
    if row['dst_bytes'] > 0:
        text += f", destination sent {int(row['dst_bytes'])} bytes"
    
    text += "."
    
    return text


def create_detailed_description(row: pd.Series) -> str:
    """
    Create a detailed text description with additional context.
    
    This includes:
    - Basic connection info
    - Connection characteristics (duration, errors)
    - Traffic patterns
    
    Args:
        row (pd.Series): Single data record
    
    Returns:
        str: Detailed text description
    """
    # Start with basic info
    parts = []
    
    # Connection basics
    parts.append(
        f"A {row['protocol_type']} connection to {row['service']} service "
        f"with connection status {row['flag']}"
    )
    
    # Duration
    if row['duration'] > 0:
        parts.append(f"lasted {int(row['duration'])} seconds")
    else:
        parts.append("was instantaneous")
    
    # Data transfer
    parts.append(
        f"transferred {int(row['src_bytes'])} bytes from source "
        f"and {int(row['dst_bytes'])} bytes from destination"
    )
    
    # Connection characteristics
    if row['wrong_fragment'] > 0:
        parts.append(f"had {int(row['wrong_fragment'])} wrong fragments")
    
    if row['urgent'] > 0:
        parts.append(f"had {int(row['urgent'])} urgent packets")
    
    # Traffic statistics
    if row['count'] > 1:
        parts.append(
            f"was part of {int(row['count'])} connections to the same host "
            f"in the past 2 seconds"
        )
    
    # Error rates
    if row['serror_rate'] > 0.5:
        parts.append(f"had high SYN error rate ({row['serror_rate']:.2f})")
    
    if row['rerror_rate'] > 0.5:
        parts.append(f"had high REJ error rate ({row['rerror_rate']:.2f})")
    
    # Join all parts
    text = ". ".join(parts) + "."
    
    return text


def create_rich_description(row: pd.Series) -> str:
    """
    Create a rich, comprehensive text description with all relevant features.
    
    This includes:
    - All connection details
    - Content features (login attempts, file operations)
    - Host-based traffic statistics
    - Semantic interpretation of patterns
    
    Args:
        row (pd.Series): Single data record
    
    Returns:
        str: Rich text description
    """
    parts = []
    
    # === CONNECTION HEADER ===
    parts.append(
        f"Network traffic analysis: {row['protocol_type'].upper()} protocol "
        f"connecting to {row['service']} service with {row['flag']} status"
    )
    
    # === TEMPORAL CHARACTERISTICS ===
    if row['duration'] > 0:
        parts.append(f"Connection duration: {int(row['duration'])} seconds")
    
    # === DATA TRANSFER ===
    total_bytes = int(row['src_bytes']) + int(row['dst_bytes'])
    parts.append(
        f"Data transfer: {int(row['src_bytes'])} bytes sent, "
        f"{int(row['dst_bytes'])} bytes received (total: {total_bytes} bytes)"
    )
    
    # === AUTHENTICATION & ACCESS ===
    if row['logged_in'] == 1:
        parts.append("User successfully logged in")
    
    if row['num_failed_logins'] > 0:
        parts.append(f"Failed login attempts: {int(row['num_failed_logins'])}")
    
    if row['num_compromised'] > 0:
        parts.append(
            f"WARNING: {int(row['num_compromised'])} compromised conditions detected"
        )
    
    # === PRIVILEGE & SYSTEM ACCESS ===
    if row['root_shell'] == 1:
        parts.append("Root shell access obtained")
    
    if row['su_attempted'] > 0:
        parts.append("'su root' command attempted")
    
    if row['num_root'] > 0:
        parts.append(f"{int(row['num_root'])} root accesses detected")
    
    if row['num_file_creations'] > 0:
        parts.append(f"{int(row['num_file_creations'])} file creation operations")
    
    if row['num_shells'] > 0:
        parts.append(f"{int(row['num_shells'])} shell prompts invoked")
    
    # === NETWORK BEHAVIOR PATTERNS ===
    if row['count'] > 1:
        parts.append(
            f"Network context: {int(row['count'])} connections to same host, "
            f"{int(row['srv_count'])} to same service in past 2 seconds"
        )
    
    # === ERROR ANALYSIS ===
    error_indicators = []
    if row['serror_rate'] > 0.3:
        error_indicators.append(f"SYN errors: {row['serror_rate']:.1%}")
    if row['rerror_rate'] > 0.3:
        error_indicators.append(f"REJ errors: {row['rerror_rate']:.1%}")
    
    if error_indicators:
        parts.append("Error rates: " + ", ".join(error_indicators))
    
    # === HOST-BASED STATISTICS ===
    if row['dst_host_count'] > 50:
        parts.append(
            f"Destination host statistics: {int(row['dst_host_count'])} "
            f"connections seen, with {row['dst_host_same_srv_rate']:.1%} "
            f"to same service"
        )
    
    # === SUSPICIOUS PATTERNS ===
    suspicious_patterns = []
    
    if row['land'] == 1:
        suspicious_patterns.append("same source/destination IP and port (LAND attack pattern)")
    
    if row['wrong_fragment'] > 0:
        suspicious_patterns.append(f"{int(row['wrong_fragment'])} fragmentation anomalies")
    
    if row['urgent'] > 0:
        suspicious_patterns.append(f"{int(row['urgent'])} urgent packets")
    
    # Very high connection rate
    if row['count'] > 100:
        suspicious_patterns.append(f"very high connection rate ({int(row['count'])} connections)")
    
    # Perfect error rates (potential scan)
    if row['serror_rate'] == 1.0 or row['rerror_rate'] == 1.0:
        suspicious_patterns.append("100% error rate (potential scanning)")
    
    if suspicious_patterns:
        parts.append("Suspicious indicators: " + "; ".join(suspicious_patterns))
    
    # Join all parts
    text = ". ".join(parts) + "."
    
    return text


# ============================================================================
# INSTRUCTION FORMATTING FOR LLM FINE-TUNING
# ============================================================================

def create_instruction_format(description: str, label: str, 
                              attack_category: str) -> Dict[str, str]:
    """
    Create instruction-following format for LLM training.
    
    This follows the format used by instruction-tuned models:
    - instruction: What the model should do
    - input: The data to analyze
    - output: The expected response
    
    Args:
        description (str): Textified network traffic description
        label (str): Original specific attack type
        attack_category (str): Broad attack category
    
    Returns:
        Dict: Formatted instruction-response pair
    """
    # Get attack description
    attack_desc = ATTACK_DESCRIPTIONS.get(attack_category, attack_category)
    
    # Create instruction
    instruction = (
        "Analyze the following network traffic log and classify it as one of: "
        "normal, dos (denial of service), probe (reconnaissance), "
        "r2l (remote to local), or u2r (user to root)."
    )
    
    # Input is the description
    input_text = description
    
    # Output is the classification with reasoning
    if attack_category == 'normal':
        output_text = (
            f"This network traffic represents {attack_desc}. "
            f"Classification: {attack_category}"
        )
    else:
        output_text = (
            f"This network traffic shows signs of {attack_desc} "
            f"(specifically: {label}). "
            f"Classification: {attack_category}"
        )
    
    return {
        'instruction': instruction,
        'input': input_text,
        'output': output_text
    }


def create_prompt_completion_format(description: str, label: str,
                                    attack_category: str) -> Dict[str, str]:
    """
    Create simple prompt-completion format for LLM training.
    
    This is an alternative format that's simpler and more direct.
    
    Args:
        description (str): Textified network traffic description
        label (str): Original specific attack type
        attack_category (str): Broad attack category
    
    Returns:
        Dict: Formatted prompt-completion pair
    """
    attack_desc = ATTACK_DESCRIPTIONS.get(attack_category, attack_category)
    
    # Create prompt with description
    prompt = f"Classify this network traffic:\n\n{description}\n\nClassification:"
    
    # Create completion
    completion = f" {attack_category} ({attack_desc})"
    
    return {
        'prompt': prompt,
        'completion': completion,
        'label': label,
        'category': attack_category
    }


# ============================================================================
# TEXTIFICATION PIPELINE
# ============================================================================

def textify_dataset(df: pd.DataFrame, verbosity: str = 'detailed',
                   format_type: str = 'instruction') -> List[Dict]:
    """
    Convert entire dataset to textified format.
    
    Args:
        df (pd.DataFrame): Processed dataframe
        verbosity (str): Level of detail ('basic', 'detailed', 'rich')
        format_type (str): Output format ('instruction' or 'prompt_completion')
    
    Returns:
        List[Dict]: List of textified records
    """
    print(f"\n{'='*70}")
    print(f"Textifying Dataset: {verbosity.upper()} verbosity")
    print(f"Format: {format_type}")
    print(f"{'='*70}\n")
    
    # Select textification function based on verbosity
    if verbosity == 'basic':
        text_func = create_basic_description
    elif verbosity == 'detailed':
        text_func = create_detailed_description
    elif verbosity == 'rich':
        text_func = create_rich_description
    else:
        raise ValueError(f"Unknown verbosity level: {verbosity}")
    
    # Select format function
    if format_type == 'instruction':
        format_func = create_instruction_format
    elif format_type == 'prompt_completion':
        format_func = create_prompt_completion_format
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    # Process each row
    textified_data = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Textifying"):
        # Create description
        description = text_func(row)
        
        # Format for LLM training
        formatted = format_func(
            description,
            row['label'],
            row['attack_category']
        )
        
        # Add metadata
        formatted['original_label'] = row['label']
        formatted['attack_category'] = row['attack_category']
        formatted['protocol'] = row['protocol_type']
        formatted['service'] = row['service']
        
        textified_data.append(formatted)
    
    return textified_data


def display_samples(textified_data: List[Dict], n_samples: int = 3):
    """
    Display sample textified records for inspection.
    
    Args:
        textified_data (List[Dict]): Textified dataset
        n_samples (int): Number of samples to display
    """
    print(f"\n{'='*70}")
    print(f"Sample Textified Records ({n_samples} examples)")
    print(f"{'='*70}\n")
    
    for i in range(min(n_samples, len(textified_data))):
        sample = textified_data[i]
        print(f"--- Sample {i+1} ---")
        
        if 'instruction' in sample:
            print(f"\nInstruction: {sample['instruction']}")
            print(f"\nInput: {sample['input'][:200]}...")
            print(f"\nOutput: {sample['output']}")
        else:
            print(f"\nPrompt: {sample['prompt'][:200]}...")
            print(f"\nCompletion: {sample['completion']}")
        
        print(f"\nMetadata:")
        print(f"  - Original Label: {sample['original_label']}")
        print(f"  - Category: {sample['attack_category']}")
        print(f"  - Protocol: {sample['protocol']}")
        print(f"  - Service: {sample['service']}")
        print("\n" + "-"*70 + "\n")


def save_textified_data(textified_data: List[Dict], output_path: str,
                       dataset_name: str = "Dataset"):
    """
    Save textified data to JSON file.
    
    Args:
        textified_data (List[Dict]): Textified dataset
        output_path (str): Path to save JSON file
        dataset_name (str): Name for logging
    """
    print(f"\n{'='*70}")
    print(f"Saving {dataset_name}...")
    print(f"{'='*70}")
    
    # Create directory if doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(textified_data, f, indent=2, ensure_ascii=False)
    
    # File info
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
    print(f"✓ Saved {len(textified_data):,} records to: {output_path}")
    print(f"✓ File size: {file_size:.2f} MB")


def get_label_distribution(textified_data: List[Dict]) -> Dict[str, int]:
    """
    Get distribution of labels in textified data.
    
    Args:
        textified_data (List[Dict]): Textified dataset
    
    Returns:
        Dict: Label distribution
    """
    distribution = {}
    for item in textified_data:
        category = item['attack_category']
        distribution[category] = distribution.get(category, 0) + 1
    
    return distribution


# ============================================================================
# MAIN TEXTIFICATION PIPELINE
# ============================================================================

def textification_pipeline(train_csv: str, test_csv: str, output_dir: str,
                          verbosity: str = 'detailed',
                          format_type: str = 'instruction'):
    """
    Main textification pipeline.
    
    Args:
        train_csv (str): Path to processed training CSV
        test_csv (str): Path to processed test CSV
        output_dir (str): Directory to save textified files
        verbosity (str): Level of detail
        format_type (str): Output format
    """
    print("\n" + "="*70)
    print(" "*15 + "DATA TEXTIFICATION PIPELINE")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load Processed Data
    # ========================================================================
    print(f"\n{'='*70}")
    print("Loading Processed Data...")
    print(f"{'='*70}")
    
    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)
    
    print(f"✓ Training data: {len(df_train):,} records")
    print(f"✓ Test data: {len(df_test):,} records")
    
    # ========================================================================
    # STEP 2: Textify Training Data
    # ========================================================================
    train_textified = textify_dataset(df_train, verbosity, format_type)
    
    # ========================================================================
    # STEP 3: Textify Test Data
    # ========================================================================
    test_textified = textify_dataset(df_test, verbosity, format_type)
    
    # ========================================================================
    # STEP 4: Display Samples
    # ========================================================================
    display_samples(train_textified, n_samples=2)
    
    # ========================================================================
    # STEP 5: Save Textified Data
    # ========================================================================
    train_output = os.path.join(
        output_dir, 
        f"train_textified_{verbosity}_{format_type}.json"
    )
    test_output = os.path.join(
        output_dir,
        f"test_textified_{verbosity}_{format_type}.json"
    )
    
    save_textified_data(train_textified, train_output, "Training Data")
    save_textified_data(test_textified, test_output, "Test Data")
    
    # ========================================================================
    # STEP 6: Summary Statistics
    # ========================================================================
    print(f"\n{'='*70}")
    print(" "*20 + "TEXTIFICATION COMPLETE")
    print(f"{'='*70}")
    
    train_dist = get_label_distribution(train_textified)
    test_dist = get_label_distribution(test_textified)
    
    print(f"\nTraining Set Label Distribution:")
    for label, count in sorted(train_dist.items()):
        percentage = (count / len(train_textified)) * 100
        print(f"  - {label:10s}: {count:6,} ({percentage:5.2f}%)")
    
    print(f"\nTest Set Label Distribution:")
    for label, count in sorted(test_dist.items()):
        percentage = (count / len(test_textified)) * 100
        print(f"  - {label:10s}: {count:6,} ({percentage:5.2f}%)")
    
    print(f"\n✓ Textified data ready for model training!")
    
    return train_textified, test_textified


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run the textification pipeline.
    
    Usage:
        python 2_data_textification.py
    
    You can modify the verbosity and format_type parameters:
        - verbosity: 'basic', 'detailed', 'rich'
        - format_type: 'instruction', 'prompt_completion'
    """
    
    # Define paths
    PROJECT_ROOT = Path(__file__).parent.parent
    TRAIN_CSV = PROJECT_ROOT / "Data" / "train_processed.csv"
    TEST_CSV = PROJECT_ROOT / "Data" / "test_processed.csv"
    OUTPUT_DIR = PROJECT_ROOT / "Data"
    
    # Configuration
    VERBOSITY = 'detailed'  # Options: 'basic', 'detailed', 'rich'
    FORMAT_TYPE = 'instruction'  # Options: 'instruction', 'prompt_completion'
    
    # Run pipeline
    train_data, test_data = textification_pipeline(
        str(TRAIN_CSV),
        str(TEST_CSV),
        str(OUTPUT_DIR),
        verbosity=VERBOSITY,
        format_type=FORMAT_TYPE
    )
    
    print(f"\n{'='*70}")
    print("Next Steps:")
    print(f"{'='*70}")
    print("\n1. Review the sample outputs above")
    print("2. Check the JSON files in the Data/ folder")
    print("3. Proceed to Module 3: Model Fine-Tuning")
    print(f"\nTextified files saved with this naming pattern:")
    print(f"  - train_textified_{VERBOSITY}_{FORMAT_TYPE}.json")
    print(f"  - test_textified_{VERBOSITY}_{FORMAT_TYPE}.json")
