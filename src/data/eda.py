"""
Exploratory Data Analysis module for spam detection project.
Creates visualizations and analyzes text patterns in the dataset.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from utils.helpers import load_config, ensure_dir_exists, setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def plot_class_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot class distribution (spam vs not_spam).
    
    Args:
        df: DataFrame with label column
        save_path: Path to save the plot
    """
    logger.info("Creating class distribution plot...")
    
    # Count labels
    label_counts = df['label'].value_counts()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    bars = ax1.bar(label_counts.index, label_counts.values, 
                   color=['#ff7f7f', '#7fbf7f'], alpha=0.8)
    ax1.set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Class', fontsize=12)
    ax1.set_ylabel('Number of Messages', fontsize=12)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    colors = ['#ff7f7f', '#7fbf7f']
    wedges, texts, autotexts = ax2.pie(label_counts.values, 
                                       labels=label_counts.index,
                                       autopct='%1.1f%%',
                                       colors=colors,
                                       startangle=90)
    ax2.set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {save_path}")
    
    plt.show()

def plot_message_length_distribution(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Plot message length distributions by class.
    
    Args:
        df: DataFrame with text and label columns
        save_path: Path to save the plot
    """
    logger.info("Creating message length distribution plot...")
    
    # Calculate message lengths
    df['message_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    # Create figure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Character length distribution
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        ax1.hist(subset['message_length'], bins=50, alpha=0.7, 
                label=label, density=True)
    
    ax1.set_title('Message Length Distribution (Characters)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Number of Characters', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Word count distribution
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        ax2.hist(subset['word_count'], bins=30, alpha=0.7, 
                label=label, density=True)
    
    ax2.set_title('Word Count Distribution', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Number of Words', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Box plot for character length
    df.boxplot(column='message_length', by='label', ax=ax3)
    ax3.set_title('Message Length by Class (Characters)', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Class', fontsize=12)
    ax3.set_ylabel('Number of Characters', fontsize=12)
    ax3.grid(True, alpha=0.3)
    
    # Box plot for word count
    df.boxplot(column='word_count', by='label', ax=ax4)
    ax4.set_title('Word Count by Class', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Class', fontsize=12)
    ax4.set_ylabel('Number of Words', fontsize=12)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Message length distribution plot saved to {save_path}")
    
    plt.show()

def get_top_ngrams(df: pd.DataFrame, ngram_range: Tuple[int, int] = (1, 1), 
                   max_features: int = 20, class_label: str = None) -> List[Tuple[str, float]]:
    """
    Get top n-grams for a specific class.
    
    Args:
        df: DataFrame with text and label columns
        ngram_range: Range of n-grams to extract
        max_features: Maximum number of features to return
        class_label: Specific class to analyze (None for all)
        
    Returns:
        List of (ngram, tfidf_score) tuples
    """
    # Filter by class if specified
    if class_label:
        texts = df[df['label'] == class_label]['text'].tolist()
    else:
        texts = df['text'].tolist()
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features * 2,  # Get more to filter out common words
        stop_words='english',
        lowercase=True
    )
    
    # Fit and transform
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get mean TF-IDF scores
    mean_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    
    # Create list of (ngram, score) tuples
    ngram_scores = list(zip(feature_names, mean_scores))
    
    # Sort by score and return top features
    ngram_scores.sort(key=lambda x: x[1], reverse=True)
    return ngram_scores[:max_features]

def plot_top_ngrams(df: pd.DataFrame, ngram_type: str = 'words', 
                   save_path: str = None) -> None:
    """
    Plot top n-grams for each class.
    
    Args:
        df: DataFrame with text and label columns
        ngram_type: 'words' or 'chars'
        save_path: Path to save the plot
    """
    logger.info(f"Creating top {ngram_type} n-grams plot...")
    
    # Set n-gram parameters
    if ngram_type == 'words':
        ngram_range = (1, 2)
        title_suffix = 'Words (1-2 grams)'
    else:  # chars
        ngram_range = (3, 5)
        title_suffix = 'Characters (3-5 grams)'
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    for idx, label in enumerate(df['label'].unique()):
        # Get top n-grams for this class
        top_ngrams = get_top_ngrams(df, ngram_range=ngram_range, 
                                   max_features=20, class_label=label)
        
        # Extract n-grams and scores
        ngrams = [item[0] for item in top_ngrams]
        scores = [item[1] for item in top_ngrams]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(ngrams))
        bars = axes[idx].barh(y_pos, scores, alpha=0.8)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(ngrams)
        axes[idx].set_xlabel('TF-IDF Score', fontsize=12)
        axes[idx].set_title(f'Top 20 {title_suffix} - {label.title()}', 
                           fontsize=14, fontweight='bold')
        axes[idx].grid(True, alpha=0.3, axis='x')
        
        # Invert y-axis to show highest scores at top
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Top {ngram_type} n-grams plot saved to {save_path}")
    
    plt.show()

def analyze_text_patterns(df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Analyze text patterns and create summary statistics.
    
    Args:
        df: DataFrame with text and label columns
        
    Returns:
        Dictionary with pattern analysis results
    """
    logger.info("Analyzing text patterns...")
    
    results = {}
    
    for label in df['label'].unique():
        subset = df[df['label'] == label]
        
        # Basic statistics
        char_lengths = subset['text'].str.len()
        word_counts = subset['text'].str.split().str.len()
        
        # Pattern analysis
        has_uppercase = subset['text'].str.contains(r'[A-Z]').sum()
        has_numbers = subset['text'].str.contains(r'\d').sum()
        has_special_chars = subset['text'].str.contains(r'[!@#$%^&*(),.?":{}|<>]').sum()
        has_urls = subset['text'].str.contains(r'http|www', case=False).sum()
        has_emails = subset['text'].str.contains(r'@').sum()
        
        results[label] = {
            'count': len(subset),
            'avg_char_length': char_lengths.mean(),
            'median_char_length': char_lengths.median(),
            'avg_word_count': word_counts.mean(),
            'median_word_count': word_counts.median(),
            'has_uppercase': has_uppercase,
            'has_numbers': has_numbers,
            'has_special_chars': has_special_chars,
            'has_urls': has_urls,
            'has_emails': has_emails,
            'uppercase_ratio': has_uppercase / len(subset),
            'numbers_ratio': has_numbers / len(subset),
            'special_chars_ratio': has_special_chars / len(subset),
            'urls_ratio': has_urls / len(subset),
            'emails_ratio': has_emails / len(subset)
        }
    
    return results

def create_eda_report(df: pd.DataFrame, output_dir: str) -> None:
    """
    Create comprehensive EDA report with all visualizations.
    
    Args:
        df: DataFrame to analyze
        output_dir: Directory to save outputs
    """
    logger.info("Creating comprehensive EDA report...")
    
    # Ensure output directory exists
    ensure_dir_exists(output_dir)
    
    # 1. Class distribution
    plot_class_distribution(df, 
                           save_path=os.path.join(output_dir, 'class_distribution.png'))
    
    # 2. Message length distributions
    plot_message_length_distribution(df, 
                                   save_path=os.path.join(output_dir, 'message_length_distribution.png'))
    
    # 3. Top words
    plot_top_ngrams(df, ngram_type='words', 
                   save_path=os.path.join(output_dir, 'top_words.png'))
    
    # 4. Top character n-grams
    plot_top_ngrams(df, ngram_type='chars', 
                   save_path=os.path.join(output_dir, 'top_char_ngrams.png'))
    
    # 5. Text pattern analysis
    pattern_analysis = analyze_text_patterns(df)
    
    # Save pattern analysis to CSV
    pattern_df = pd.DataFrame(pattern_analysis).T
    pattern_path = os.path.join(output_dir, 'text_patterns.csv')
    pattern_df.to_csv(pattern_path)
    logger.info(f"Text pattern analysis saved to {pattern_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EXPLORATORY DATA ANALYSIS SUMMARY")
    print("="*60)
    
    print(f"\nDataset Overview:")
    print(f"  Total messages: {len(df)}")
    print(f"  Classes: {df['label'].unique()}")
    print(f"  Class distribution: {df['label'].value_counts().to_dict()}")
    
    print(f"\nText Pattern Analysis:")
    for label, stats in pattern_analysis.items():
        print(f"\n  {label.upper()}:")
        print(f"    Average length: {stats['avg_char_length']:.1f} chars, {stats['avg_word_count']:.1f} words")
        print(f"    Uppercase ratio: {stats['uppercase_ratio']:.2%}")
        print(f"    Numbers ratio: {stats['numbers_ratio']:.2%}")
        print(f"    Special chars ratio: {stats['special_chars_ratio']:.2%}")
        print(f"    URLs ratio: {stats['urls_ratio']:.2%}")
        print(f"    Emails ratio: {stats['emails_ratio']:.2%}")

def run_eda(input_file: str = None, df: pd.DataFrame = None) -> None:
    """
    Main function to run exploratory data analysis.
    
    Args:
        input_file: Path to input CSV file (optional)
        df: DataFrame to analyze (optional)
    """
    logger.info("Starting exploratory data analysis...")
    
    # Load configuration
    config = load_config()
    
    # Load data
    if df is not None:
        data = df.copy()
    elif input_file and os.path.exists(input_file):
        data = pd.read_csv(input_file)
        logger.info(f"Loaded data from {input_file}: {len(data)} messages")
    else:
        # If no input provided, run preprocessing
        from data.preprocess import preprocess_data
        train_data, _, _ = preprocess_data()
        data = train_data
    
    # Create EDA report
    output_dir = os.path.join(config['data']['output_dir'], 'eda')
    create_eda_report(data, output_dir)
    
    logger.info("EDA completed successfully!")

if __name__ == "__main__":
    # Run EDA
    run_eda()
