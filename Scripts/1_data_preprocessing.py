"""
================================================================================
MODULE 1: DATA PREPROCESSING
================================================================================

Purpose:
    This module handles the initial data loading, cleaning, and preparation
    for the LLM-based Network Anomaly Detection system.

What this script does:
    1. Loads raw KDDTrain.txt and KDDTest.txt files
    2. Handles missing values and data quality issues
    3. Maps attack labels to broader categories (normal, dos, probe, r2l, u2r)
    4. Performs basic data validation
    5. Saves cleaned data for the textification module

Why we need this:
    - Raw NSL-KDD data has 42+ columns with various data types
    - Attack labels are specific (e.g., "neptune", "smurf") but we want to
      classify into broader categories for better generalization
    - Clean data improves model training quality
    - Separating preprocessing from textification keeps code modular

================================================================================
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Define all 41 features + label + difficulty_level
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate", "label", "difficulty_level"
]

# Attack type mapping: Specific attacks → Broad categories
# This is based on the NSL-KDD dataset documentation
ATTACK_MAPPING = {
    # Normal traffic
    'normal': 'normal',
    
    # Denial of Service (DoS) attacks
    'back': 'dos',
    'land': 'dos',
    'neptune': 'dos',
    'pod': 'dos',
    'smurf': 'dos',
    'teardrop': 'dos',
    'mailbomb': 'dos',
    'processtable': 'dos',
    'udpstorm': 'dos',
    'apache2': 'dos',
    'worm': 'dos',
    
    # Probe/Reconnaissance attacks
    'ipsweep': 'probe',
    'nmap': 'probe',
    'portsweep': 'probe',
    'satan': 'probe',
    'mscan': 'probe',
    'saint': 'probe',
    
    # Remote to Local (R2L) attacks
    'ftp_write': 'r2l',
    'guess_passwd': 'r2l',
    'imap': 'r2l',
    'multihop': 'r2l',
    'phf': 'r2l',
    'spy': 'r2l',
    'warezclient': 'r2l',
    'warezmaster': 'r2l',
    'sendmail': 'r2l',
    'named': 'r2l',
    'snmpgetattack': 'r2l',
    'snmpguess': 'r2l',
    'xlock': 'r2l',
    'xsnoop': 'r2l',
    'httptunnel': 'r2l',
    
    # User to Root (U2R) attacks
    'buffer_overflow': 'u2r',
    'loadmodule': 'u2r',
    'perl': 'u2r',
    'rootkit': 'u2r',
    'sqlattack': 'u2r',
    'xterm': 'u2r',
    'ps': 'u2r'
}

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_kdd_data(file_path, dataset_name="Dataset"):
    """
    Load NSL-KDD data from a text file.
    
    Args:
        file_path (str): Path to the KDD data file
        dataset_name (str): Name for logging purposes
    
    Returns:
        pd.DataFrame: Loaded dataframe with proper column names
    """
    print(f"\n{'='*70}")
    print(f"Loading {dataset_name}...")
    print(f"{'='*70}")
    
    try:
        # Read CSV without header (NSL-KDD files don't have headers)
        df = pd.read_csv(file_path, names=COLUMNS, header=None)
        
        print(f"✓ Successfully loaded {len(df):,} records")
        print(f"✓ Shape: {df.shape}")
        
        return df
    
    except FileNotFoundError:
        print(f"✗ Error: File not found at {file_path}")
        raise
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        raise


def display_data_info(df, dataset_name="Dataset"):
    """
    Display comprehensive information about the dataset.
    
    Args:
        df (pd.DataFrame): The dataset to analyze
        dataset_name (str): Name for display purposes
    """
    print(f"\n{'-'*70}")
    print(f"{dataset_name} Information")
    print(f"{'-'*70}")
    
    # Basic info
    print(f"\nDataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Check for missing values
    missing = df.isnull().sum().sum()
    print(f"Missing Values: {missing}")
    
    # Data types
    print(f"\nData Types:")
    print(f"  - Numerical: {len(df.select_dtypes(include=[np.number]).columns)}")
    print(f"  - Categorical: {len(df.select_dtypes(include=['object']).columns)}")
    
    # Label distribution
    if 'label' in df.columns:
        print(f"\nLabel Distribution (Top 10):")
        label_dist = df['label'].value_counts().head(10)
        for label, count in label_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  - {label:20s}: {count:7,} ({percentage:5.2f}%)")
    
    # Protocol types
    if 'protocol_type' in df.columns:
        print(f"\nProtocol Types:")
        for proto, count in df['protocol_type'].value_counts().items():
            percentage = (count / len(df)) * 100
            print(f"  - {proto:10s}: {count:7,} ({percentage:5.2f}%)")


# ============================================================================
# DATA CLEANING FUNCTIONS
# ============================================================================

def clean_label_column(df):
    """
    Clean the label column by removing trailing dots and whitespace.
    
    NSL-KDD labels sometimes have trailing dots (e.g., "normal." vs "normal")
    
    Args:
        df (pd.DataFrame): Dataset to clean
    
    Returns:
        pd.DataFrame: Dataset with cleaned labels
    """
    print(f"\n{'-'*70}")
    print("Cleaning Label Column...")
    print(f"{'-'*70}")
    
    # Remove trailing dots and whitespace
    df['label'] = df['label'].str.rstrip('.').str.strip().str.lower()
    
    # Show unique labels
    unique_labels = df['label'].nunique()
    print(f"✓ Found {unique_labels} unique attack types")
    
    return df


def map_attack_categories(df):
    """
    Map specific attack types to broader categories.
    
    Why we do this:
        - NSL-KDD has 39+ specific attack types
        - For practical IDS, we classify into 5 categories:
          1. normal
          2. dos (Denial of Service)
          3. probe (Reconnaissance)
          4. r2l (Remote to Local)
          5. u2r (User to Root)
        - This improves model generalization
    
    Args:
        df (pd.DataFrame): Dataset with original labels
    
    Returns:
        pd.DataFrame: Dataset with new 'attack_category' column
    """
    print(f"\n{'-'*70}")
    print("Mapping Attack Categories...")
    print(f"{'-'*70}")
    
    # Create new column
    df['attack_category'] = df['label'].map(ATTACK_MAPPING)
    
    # Check for unmapped labels (unknown attack types)
    unmapped = df['attack_category'].isna().sum()
    if unmapped > 0:
        print(f"\n⚠ Warning: {unmapped} records have unmapped labels:")
        unmapped_labels = df[df['attack_category'].isna()]['label'].unique()
        for label in unmapped_labels:
            count = (df['label'] == label).sum()
            print(f"  - {label}: {count} occurrences")
        
        # Fill unmapped with 'unknown'
        df['attack_category'] = df['attack_category'].fillna('unknown')
    
    # Display category distribution
    print(f"\nAttack Category Distribution:")
    category_dist = df['attack_category'].value_counts()
    for category, count in category_dist.items():
        percentage = (count / len(df)) * 100
        print(f"  - {category:10s}: {count:7,} ({percentage:5.2f}%)")
    
    return df


def handle_missing_values(df):
    """
    Handle missing or invalid values in the dataset.
    
    Args:
        df (pd.DataFrame): Dataset to clean
    
    Returns:
        pd.DataFrame: Dataset with handled missing values
    """
    print(f"\n{'-'*70}")
    print("Handling Missing Values...")
    print(f"{'-'*70}")
    
    # Check for missing values
    missing_counts = df.isnull().sum()
    total_missing = missing_counts.sum()
    
    if total_missing == 0:
        print("✓ No missing values found")
    else:
        print(f"Found {total_missing} missing values:")
        for col in missing_counts[missing_counts > 0].index:
            print(f"  - {col}: {missing_counts[col]}")
        
        # Strategy: Drop rows with missing values (NSL-KDD is clean, so this is rare)
        df = df.dropna()
        print(f"✓ Dropped rows with missing values. New shape: {df.shape}")
    
    return df


def validate_data_quality(df):
    """
    Perform data quality checks.
    
    Args:
        df (pd.DataFrame): Dataset to validate
    
    Returns:
        bool: True if data passes quality checks
    """
    print(f"\n{'-'*70}")
    print("Validating Data Quality...")
    print(f"{'-'*70}")
    
    issues = []
    
    # Check 1: Minimum number of records
    if len(df) < 100:
        issues.append(f"Too few records: {len(df)}")
    
    # Check 2: Required columns exist
    required_cols = ['label', 'attack_category', 'protocol_type', 'service', 'flag']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check 3: Label distribution (check for extreme imbalance)
    if 'attack_category' in df.columns:
        label_counts = df['attack_category'].value_counts()
        min_class_size = label_counts.min()
        if min_class_size < 10:
            issues.append(f"Class imbalance: smallest class has only {min_class_size} samples")
    
    # Report results
    if issues:
        print("✗ Data quality issues found:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✓ All data quality checks passed")
        return True


# ============================================================================
# EXPORT FUNCTIONS
# ============================================================================

def save_processed_data(df, output_path, dataset_name="Dataset"):
    """
    Save processed dataset to CSV.
    
    Args:
        df (pd.DataFrame): Processed dataset
        output_path (str): Path to save the file
        dataset_name (str): Name for logging
    """
    print(f"\n{'-'*70}")
    print(f"Saving {dataset_name}...")
    print(f"{'-'*70}")
    
    # Create directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
    print(f"✓ Saved to: {output_path}")
    print(f"✓ File size: {file_size:.2f} MB")


def save_label_mapping(output_path):
    """
    Save the attack category mapping as JSON for reference.
    
    Args:
        output_path (str): Path to save the JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(ATTACK_MAPPING, f, indent=2)
    print(f"✓ Label mapping saved to: {output_path}")


# ============================================================================
# MAIN PREPROCESSING PIPELINE
# ============================================================================

def preprocess_pipeline(train_path, test_path, output_dir):
    """
    Main preprocessing pipeline that orchestrates all steps.
    
    Args:
        train_path (str): Path to KDDTrain.txt
        test_path (str): Path to KDDTest.txt
        output_dir (str): Directory to save processed files
    """
    print("\n" + "="*70)
    print(" "*15 + "DATA PREPROCESSING PIPELINE")
    print("="*70)
    
    # ========================================================================
    # STEP 1: Load Training Data
    # ========================================================================
    df_train = load_kdd_data(train_path, "Training Data")
    display_data_info(df_train, "Training Data")
    
    # ========================================================================
    # STEP 2: Load Test Data
    # ========================================================================
    df_test = load_kdd_data(test_path, "Test Data")
    display_data_info(df_test, "Test Data")
    
    # ========================================================================
    # STEP 3: Clean Labels
    # ========================================================================
    df_train = clean_label_column(df_train)
    df_test = clean_label_column(df_test)
    
    # ========================================================================
    # STEP 4: Map Attack Categories
    # ========================================================================
    df_train = map_attack_categories(df_train)
    df_test = map_attack_categories(df_test)
    
    # ========================================================================
    # STEP 5: Handle Missing Values
    # ========================================================================
    df_train = handle_missing_values(df_train)
    df_test = handle_missing_values(df_test)
    
    # ========================================================================
    # STEP 6: Validate Data Quality
    # ========================================================================
    train_valid = validate_data_quality(df_train)
    test_valid = validate_data_quality(df_test)
    
    if not (train_valid and test_valid):
        print("\n✗ Data quality validation failed. Please review the issues.")
        return None, None
    
    # ========================================================================
    # STEP 7: Save Processed Data
    # ========================================================================
    train_output = os.path.join(output_dir, "train_processed.csv")
    test_output = os.path.join(output_dir, "test_processed.csv")
    mapping_output = os.path.join(output_dir, "attack_mapping.json")
    
    save_processed_data(df_train, train_output, "Processed Training Data")
    save_processed_data(df_test, test_output, "Processed Test Data")
    save_label_mapping(mapping_output)
    
    # ========================================================================
    # STEP 8: Summary Statistics
    # ========================================================================
    print(f"\n{'='*70}")
    print(" "*20 + "PREPROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"\nFinal Dataset Statistics:")
    print(f"  Training samples: {len(df_train):,}")
    print(f"  Test samples:     {len(df_test):,}")
    print(f"  Total features:   {len(df_train.columns)}")
    print(f"  Attack categories: {df_train['attack_category'].nunique()}")
    
    print(f"\n✓ Processed data ready for textification!")
    
    return df_train, df_test


# ============================================================================
# SCRIPT ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run the preprocessing pipeline.
    
    Usage:
        python 1_data_preprocessing.py
    """
    
    # Define paths (relative to project root)
    PROJECT_ROOT = Path(__file__).parent.parent
    TRAIN_PATH = PROJECT_ROOT / "Data" / "KDDTrain.txt"
    TEST_PATH = PROJECT_ROOT / "Data" / "KDDTest.txt"
    OUTPUT_DIR = PROJECT_ROOT / "Data"
    
    # Run pipeline
    df_train, df_test = preprocess_pipeline(
        str(TRAIN_PATH),
        str(TEST_PATH),
        str(OUTPUT_DIR)
    )
    
    # Optional: Display sample processed records
    if df_train is not None:
        print(f"\n{'='*70}")
        print("Sample Processed Records (First 3)")
        print(f"{'='*70}\n")
        
        # Select key columns for display
        display_cols = ['protocol_type', 'service', 'flag', 'src_bytes', 
                       'dst_bytes', 'label', 'attack_category']
        print(df_train[display_cols].head(3).to_string(index=False))
