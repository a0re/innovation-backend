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
    Get top n-grams for a specific class using frequency-based approach.
    
    Args:
        df: DataFrame with text and label columns
        ngram_range: Range of n-grams to extract
        max_features: Maximum number of features to return
        class_label: Specific class to analyze (None for all)
        
    Returns:
        List of (ngram, frequency) tuples
    """
    from collections import Counter
    import re
    
    # Filter by class if specified
    if class_label:
        texts = df[df['label'] == class_label]['text'].tolist()
    else:
        texts = df['text'].tolist()
    
    # Collect all n-grams
    all_ngrams = []
    
    # Extended stop words list
    stop_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'man', 'men', 'put', 'say', 'she', 'too', 'use', 'way', 'will', 'with', 'have', 'this', 'that', 'they', 'been', 'from', 'were', 'said', 'each', 'which', 'their', 'time', 'would', 'there', 'could', 'other', 'after', 'first', 'well', 'also', 'where', 'much', 'some', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'these', 'think', 'want', 'what', 'your', 'about', 'back', 'call', 'good', 'know', 'look', 'more', 'most', 'only', 'right', 'seem', 'tell', 'turn', 'work', 'year', 'find', 'give', 'hand', 'help', 'keep', 'kind', 'last', 'late', 'life', 'line', 'live', 'move', 'name', 'need', 'open', 'part', 'play', 'same', 'show', 'side', 'tell', 'turn', 'walk', 'want', 'week', 'went', 'were', 'what', 'when', 'will', 'with', 'word', 'work', 'year', 'yes', 'yet', 'you', 'your', 'if', 'in', 'is', 'it', 'of', 'on', 'or', 'to', 'be', 'do', 'go', 'no', 'so', 'up', 'we', 'he', 'me', 'my', 'as', 'at', 'by', 'an', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
    
    for text in texts:
        # Extract words (alphanumeric only)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        # Create n-grams
        for ngram_size in range(ngram_range[0], ngram_range[1] + 1):
            for i in range(len(words) - ngram_size + 1):
                ngram = ' '.join(words[i:i+ngram_size])
                
                # Filter out unwanted n-grams
                if (len(ngram) > 3 and  # Minimum length
                    not ngram.startswith('num') and  # Remove preprocessing artifacts
                    not ngram.startswith('number') and
                    not ngram.startswith('url') and
                    not ngram.startswith('email') and
                    not ngram.startswith('ect') and
                    not ngram.startswith('pdt') and
                    not ngram.startswith('pst') and
                    not ngram.startswith('hou') and
                    not re.match(r'^[a-z]{1,2}$', ngram) and  # Remove very short words
                    not re.match(r'^[a-z]{1,2}\s[a-z]{1,2}$', ngram) and  # Remove short bigrams
                    not re.search(r'[0-9]', ngram) and  # Remove n-grams with numbers
                    not re.search(r'[<>]', ngram) and  # Remove n-grams with angle brackets
                    not any(word in stop_words for word in ngram.split()) and  # Remove stop words
                    not re.search(r'hyperlink.*hyperlink', ngram)):  # Remove repetitive hyperlink patterns
                    all_ngrams.append(ngram)
    
    # Count frequencies
    ngram_counts = Counter(all_ngrams)
    
    # Get top n-grams by frequency
    top_ngrams = ngram_counts.most_common(max_features)
    
    return top_ngrams

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
        
        # Extract n-grams and frequencies
        ngrams = [item[0] for item in top_ngrams]
        frequencies = [item[1] for item in top_ngrams]
        
        # Create horizontal bar plot
        y_pos = np.arange(len(ngrams))
        bars = axes[idx].barh(y_pos, frequencies, alpha=0.8)
        axes[idx].set_yticks(y_pos)
        axes[idx].set_yticklabels(ngrams)
        axes[idx].set_xlabel('Frequency', fontsize=12)
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
    
    # 5. Advanced message characteristics
    plot_advanced_message_characteristics(df, 
                                        save_path=os.path.join(output_dir, 'message_characteristics.png'))
    
    # 6. Text pattern analysis
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

def plot_spam_trigger_words(df: pd.DataFrame, top_n: int = 20) -> None:
    """
    Identify and visualize words that strongly indicate spam.
    
    Args:
        df: DataFrame with text and label columns
        top_n: Number of top spam trigger words to display
    """
    logger.info("Creating spam trigger words analysis...")
    
    # Get all words from spam and ham messages
    spam_words = []
    ham_words = []
    
    for _, row in df.iterrows():
        words = row['text'].lower().split()
        if row['label'] == 'spam':
            spam_words.extend(words)
        else:
            ham_words.extend(words)
    
    # Count word frequencies
    spam_freq = Counter(spam_words)
    ham_freq = Counter(ham_words)
    
    # Calculate spam ratio for each word
    all_words = set(spam_freq.keys()) | set(ham_freq.keys())
    spam_ratios = {}
    
    for word in all_words:
        spam_count = spam_freq.get(word, 0)
        ham_count = ham_freq.get(word, 0)
        total = spam_count + ham_count
        
        # Only consider words that appear at least 5 times
        if total >= 5:
            spam_ratios[word] = spam_count / total
    
    # Get top spam trigger words
    top_spam_words = sorted(spam_ratios.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words, ratios = zip(*top_spam_words)
    
    # Create plot
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(words)), ratios, color='red', alpha=0.7)
    plt.yticks(range(len(words)), words)
    plt.xlabel('Spam Ratio (Spam Count / Total Count)')
    plt.title(f'Top {top_n} Spam Trigger Words')
    plt.gca().invert_yaxis()
    
    # Add value labels on bars
    for i, (bar, ratio) in enumerate(zip(bars, ratios)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{ratio:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed analysis
    print(f"\nTop {top_n} Spam Trigger Words:")
    print("-" * 50)
    for word, ratio in top_spam_words:
        spam_count = spam_freq.get(word, 0)
        ham_count = ham_freq.get(word, 0)
        print(f"{word:15} | Ratio: {ratio:.3f} | Spam: {spam_count:3d} | Ham: {ham_count:3d}")

def plot_advanced_message_characteristics(df: pd.DataFrame, save_path: str = None) -> None:
    """
    Create comprehensive message characteristic analysis.
    
    Args:
        df: DataFrame with text and label columns
    """
    import re
    logger.info("Creating advanced message characteristics analysis...")
    
    # Calculate additional features
    df_analysis = df.copy()
    df_analysis['message_length'] = df_analysis['text'].str.len()
    df_analysis['word_count'] = df_analysis['text'].str.split().str.len()
    
    # Fix uppercase ratio calculation (text is already lowercase, so this will be 0)
    df_analysis['uppercase_ratio'] = df_analysis['text'].apply(lambda x: len(re.findall(r'[A-Z]', x)) / len(x) if len(x) > 0 else 0)
    
    # Fix digit ratio calculation
    df_analysis['digit_ratio'] = df_analysis['text'].apply(lambda x: len(re.findall(r'\d', x)) / len(x) if len(x) > 0 else 0)
    
    # Fix special characters count
    df_analysis['special_chars'] = df_analysis['text'].apply(lambda x: len(re.findall(r'[!@#$%^&*(),.?":{}|<>]', x)))
    df_analysis['exclamations'] = df_analysis['text'].apply(lambda x: len(re.findall(r'!', x)))
    df_analysis['questions'] = df_analysis['text'].apply(lambda x: len(re.findall(r'\?', x)))
    df_analysis['caps_words'] = df_analysis['text'].apply(lambda x: len(re.findall(r'\b[A-Z]{2,}\b', x)))
    df_analysis['has_url'] = df_analysis['text'].str.contains(r'http', case=False).astype(int)
    df_analysis['has_phone'] = df_analysis['text'].str.contains(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b').astype(int)
    
    # Add more meaningful features since text is lowercase
    df_analysis['exclamation_ratio'] = df_analysis['exclamations'] / df_analysis['message_length']
    df_analysis['question_ratio'] = df_analysis['questions'] / df_analysis['message_length']
    df_analysis['special_char_ratio'] = df_analysis['special_chars'] / df_analysis['message_length']
    
    # Create subplots
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    # Define characteristics to plot
    characteristics = [
        ('message_length', 'Message Length (characters)'),
        ('word_count', 'Word Count'),
        ('digit_ratio', 'Digit Ratio'),
        ('special_char_ratio', 'Special Character Ratio'),
        ('exclamation_ratio', 'Exclamation Mark Ratio'),
        ('question_ratio', 'Question Mark Ratio'),
        ('exclamations', 'Exclamation Marks Count'),
        ('questions', 'Question Marks Count'),
        ('has_url', 'Contains URL (0/1)')
    ]
    
    for idx, (col, title) in enumerate(characteristics):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Use box plots for better comparison
        data_to_plot = []
        labels = []
        for label in df_analysis['label'].unique():
            subset = df_analysis[df_analysis['label'] == label]
            data_to_plot.append(subset[col].dropna())
            labels.append(label)
        
        ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        ax.set_title(title)
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(characteristics), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Advanced message characteristics plot saved to {save_path}")
    
    plt.show()
    
    # Print summary statistics
    print("\nMessage Characteristics Summary:")
    print("=" * 60)
    
    for col, title in characteristics:
        print(f"\n{title}:")
        for label in df_analysis['label'].unique():
            subset = df_analysis[df_analysis['label'] == label]
            mean_val = subset[col].mean()
            std_val = subset[col].std()
            print(f"  {label:8}: Mean={mean_val:.3f}, Std={std_val:.3f}")

def plot_tfidf_feature_analysis(df: pd.DataFrame, max_features: int = 1000) -> None:
    """
    Analyze TF-IDF features and their importance.
    
    Args:
        df: DataFrame with text and label columns
        max_features: Maximum number of features to analyze
    """
    logger.info("Creating TF-IDF feature analysis...")
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words='english',
        lowercase=True
    )
    
    # Fit and transform
    try:
        tfidf_matrix = vectorizer.fit_transform(df['text'])
        feature_names = vectorizer.get_feature_names_out()
    except Exception as e:
        logger.error(f"Error in TF-IDF vectorization: {e}")
        return
    
    # Calculate mean TF-IDF scores for each class
    spam_indices = df['label'] == 'spam'
    ham_indices = df['label'] == 'not_spam'
    
    # Handle case where there might be no spam or ham messages
    try:
        if spam_indices.sum() > 0:
            spam_tfidf = tfidf_matrix[spam_indices].mean(axis=0).A1
        else:
            spam_tfidf = np.zeros(tfidf_matrix.shape[1])
        
        if ham_indices.sum() > 0:
            ham_tfidf = tfidf_matrix[ham_indices].mean(axis=0).A1
        else:
            ham_tfidf = np.zeros(tfidf_matrix.shape[1])
    except Exception as e:
        logger.error(f"Error calculating TF-IDF means: {e}")
        return
    
    # Create DataFrame for analysis
    tfidf_df = pd.DataFrame({
        'feature': feature_names,
        'spam_tfidf': spam_tfidf,
        'ham_tfidf': ham_tfidf
    })
    
    # Calculate difference and ratio
    tfidf_df['difference'] = tfidf_df['spam_tfidf'] - tfidf_df['ham_tfidf']
    tfidf_df['ratio'] = tfidf_df['spam_tfidf'] / (tfidf_df['ham_tfidf'] + 1e-8)
    
    # Get top features for each class
    top_spam_features = tfidf_df.nlargest(20, 'spam_tfidf')
    top_ham_features = tfidf_df.nlargest(20, 'ham_tfidf')
    top_differentiating = tfidf_df.nlargest(20, 'difference')
    
    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Top spam features
    axes[0].barh(range(len(top_spam_features)), top_spam_features['spam_tfidf'])
    axes[0].set_yticks(range(len(top_spam_features)))
    axes[0].set_yticklabels(top_spam_features['feature'])
    axes[0].set_title('Top TF-IDF Features in Spam')
    axes[0].set_xlabel('Mean TF-IDF Score')
    
    # Top ham features
    axes[1].barh(range(len(top_ham_features)), top_ham_features['ham_tfidf'])
    axes[1].set_yticks(range(len(top_ham_features)))
    axes[1].set_yticklabels(top_ham_features['feature'])
    axes[1].set_title('Top TF-IDF Features in Ham')
    axes[1].set_xlabel('Mean TF-IDF Score')
    
    # Most differentiating features
    axes[2].barh(range(len(top_differentiating)), top_differentiating['difference'])
    axes[2].set_yticks(range(len(top_differentiating)))
    axes[2].set_yticklabels(top_differentiating['feature'])
    axes[2].set_title('Most Differentiating Features (Spam - Ham)')
    axes[2].set_xlabel('TF-IDF Difference')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print("\nTF-IDF Feature Analysis Summary:")
    print("=" * 50)
    print(f"Total features analyzed: {len(feature_names)}")
    print(f"Features with higher spam TF-IDF: {(tfidf_df['difference'] > 0).sum()}")
    print(f"Features with higher ham TF-IDF: {(tfidf_df['difference'] < 0).sum()}")
    
    print(f"\nTop 10 Most Differentiating Features:")
    for _, row in top_differentiating.head(10).iterrows():
        print(f"  {row['feature']:20} | Diff: {row['difference']:.4f} | Ratio: {row['ratio']:.2f}")

if __name__ == "__main__":
    # Run EDA
    run_eda()
