"""
Data preprocessing module for spam detection project.
Handles text normalization, cleaning, and train/validation/test split.
"""

import os
import pandas as pd
import numpy as np
import re
import logging
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from utils.helpers import load_config, ensure_dir_exists, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def clean_text(text: str) -> str:
    """
    Clean and normalize text data.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace URLs with placeholder
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                  '<URL>', text)
    
    # Replace email addresses with placeholder
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                  '<EMAIL>', text)
    
    # Replace phone numbers with placeholder
    text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', 
                  '<PHONE>', text)
    
    # Replace numbers with placeholder
    text = re.sub(r'\b\d+\b', '<NUM>', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate messages from the dataset.
    
    Args:
        df: DataFrame with text and label columns
        
    Returns:
        DataFrame with duplicates removed
    """
    logger.info("Removing duplicates...")
    
    initial_count = len(df)
    df_clean = df.drop_duplicates(subset=['text'], keep='first')
    final_count = len(df_clean)
    
    removed_count = initial_count - final_count
    logger.info(f"Removed {removed_count} duplicate messages ({removed_count/initial_count*100:.1f}%)")
    
    return df_clean

def balance_dataset(df: pd.DataFrame, max_samples_per_class: int = None) -> pd.DataFrame:
    """
    Balance the dataset by limiting samples per class.
    
    Args:
        df: DataFrame with label column
        max_samples_per_class: Maximum number of samples per class
        
    Returns:
        Balanced DataFrame
    """
    if max_samples_per_class is None:
        return df
    
    logger.info(f"Balancing dataset with max {max_samples_per_class} samples per class...")
    
    balanced_dfs = []
    for label in df['label'].unique():
        class_df = df[df['label'] == label]
        if len(class_df) > max_samples_per_class:
            class_df = class_df.sample(n=max_samples_per_class, random_state=42)
        balanced_dfs.append(class_df)
    
    balanced_df = pd.concat(balanced_dfs, ignore_index=True)
    logger.info(f"Balanced dataset: {len(balanced_df)} messages")
    logger.info(f"Label distribution: {balanced_df['label'].value_counts().to_dict()}")
    
    return balanced_df

def split_data(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, 
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split data into train, validation, and test sets with stratification.
    
    Args:
        df: DataFrame to split
        test_size: Proportion of data for test set
        val_size: Proportion of data for validation set
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Splitting data into train/validation/test sets...")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=df['label']
    )
    
    # Second split: separate train and validation sets
    # Adjust val_size to account for the test set already removed
    adjusted_val_size = val_size / (1 - test_size)
    
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=adjusted_val_size, 
        random_state=random_state, 
        stratify=train_val_df['label']
    )
    
    logger.info(f"Data split completed:")
    logger.info(f"  Train: {len(train_df)} messages ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_df)} messages ({len(val_df)/len(df)*100:.1f}%)")
    logger.info(f"  Test: {len(test_df)} messages ({len(test_df)/len(df)*100:.1f}%)")
    
    # Log label distribution for each split
    for split_name, split_df in [("Train", train_df), ("Validation", val_df), ("Test", test_df)]:
        logger.info(f"  {split_name} label distribution: {split_df['label'].value_counts().to_dict()}")
    
    return train_df, val_df, test_df

def preprocess_data(input_file: str = None, df: pd.DataFrame = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main preprocessing function.
    
    Args:
        input_file: Path to input CSV file (optional)
        df: DataFrame to preprocess (optional)
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Starting data preprocessing...")
    
    # Load configuration
    config = load_config()
    
    # Load data
    if df is not None:
        data = df.copy()
    elif input_file and os.path.exists(input_file):
        data = pd.read_csv(input_file)
        logger.info(f"Loaded data from {input_file}: {len(data)} messages")
    else:
        # If no input provided, run data collection
        from data.collect import collect_data
        data = collect_data()
    
    # Step 1: Clean text
    logger.info("Step 1: Cleaning text...")
    data['text'] = data['text'].apply(clean_text)
    
    # Step 2: Remove empty messages
    initial_count = len(data)
    data = data[data['text'].str.len() > 0]
    logger.info(f"Removed {initial_count - len(data)} empty messages")
    
    # Step 3: Remove duplicates
    data = remove_duplicates(data)
    
    # Step 4: Balance dataset (optional)
    # data = balance_dataset(data, max_samples_per_class=5000)
    
    # Step 5: Split data
    train_df, val_df, test_df = split_data(
        data, 
        test_size=config['preprocessing']['test_size'],
        val_size=config['preprocessing']['val_size'],
        random_state=config['random_state']
    )
    
    # Step 6: Save processed data
    output_dir = config['data']['output_dir']
    ensure_dir_exists(output_dir)
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    logger.info(f"Processed data saved:")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Validation: {val_path}")
    logger.info(f"  Test: {test_path}")
    
    return train_df, val_df, test_df

def get_text_statistics(df: pd.DataFrame) -> dict:
    """
    Get basic statistics about the text data.
    
    Args:
        df: DataFrame with text column
        
    Returns:
        Dictionary with text statistics
    """
    stats = {
        'total_messages': len(df),
        'avg_length': df['text'].str.len().mean(),
        'median_length': df['text'].str.len().median(),
        'min_length': df['text'].str.len().min(),
        'max_length': df['text'].str.len().max(),
        'avg_words': df['text'].str.split().str.len().mean(),
        'median_words': df['text'].str.split().str.len().median(),
    }
    
    return stats

if __name__ == "__main__":
    # Run preprocessing
    train_data, val_data, test_data = preprocess_data()
    
    print("Preprocessing completed!")
    print(f"Train set: {len(train_data)} messages")
    print(f"Validation set: {len(val_data)} messages")
    print(f"Test set: {len(test_data)} messages")
    
    # Print statistics
    for split_name, split_data in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        stats = get_text_statistics(split_data)
        print(f"\n{split_name} set statistics:")
        print(f"  Average length: {stats['avg_length']:.1f} characters")
        print(f"  Average words: {stats['avg_words']:.1f} words")
        print(f"  Length range: {stats['min_length']} - {stats['max_length']} characters")
