"""
Data collection module for spam detection project.
Downloads and combines real datasets from UCI and Kaggle.
"""

import os
import sys
import pandas as pd
import requests
import zipfile
import io
import logging

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, ensure_dir_exists, setup_logging

# Import kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logging.error("Kaggle API not available. Please install: pip install kaggle")

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def validate_kaggle_dataset(dataset_name: str) -> bool:
    """Validate if a Kaggle dataset exists and is accessible."""
    if not KAGGLE_AVAILABLE:
        return False
    
    try:
        api = KaggleApi()
        api.authenticate()
        response = api.dataset_list_files(dataset_name)
        # Check if response has files
        return hasattr(response, 'dataset_files') and len(response.dataset_files) > 0
    except Exception:
        return False

def download_sms_spam_data() -> pd.DataFrame:
    """Download SMS Spam Collection dataset from UCI."""
    logger.info("Downloading SMS Spam Collection dataset...")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Find the data file in the zip (SMSSpamCollection)
            data_files = [f for f in zip_file.namelist() if f.endswith('.csv') or f == 'SMSSpamCollection']
            if not data_files:
                raise ValueError("No data file found in the zip archive")
            
            # Read the data file
            data_content = zip_file.read(data_files[0])
            
            # Try different encodings
            for encoding in ['utf-8', 'latin-1']:
                try:
                    df = pd.read_csv(io.BytesIO(data_content), encoding=encoding, sep='\t', header=None, names=['label', 'text'])
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Could not decode CSV file with any supported encoding")
        
        logger.info(f"SMS dataset loaded: {len(df)} messages")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading SMS dataset: {e}")
        raise ValueError(f"Failed to download SMS dataset: {e}")


def download_kaggle_email_dataset() -> pd.DataFrame:
    """Download email spam dataset from Kaggle."""
    if not KAGGLE_AVAILABLE:
        raise ValueError("Kaggle API not available. Please install kaggle package and configure API key.")
    
    dataset_name = "ozlerhakan/spam-or-not-spam-dataset"
    
    if not validate_kaggle_dataset(dataset_name):
        raise ValueError(f"Dataset {dataset_name} is not accessible!")
    
    try:
        api = KaggleApi()
        api.authenticate()
        
        temp_dir = "temp_kaggle_download"
        os.makedirs(temp_dir, exist_ok=True)
        
        api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        
        if not csv_files:
            raise ValueError("No CSV file found in downloaded dataset")
        
        csv_path = os.path.join(temp_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        
        # Clean up
        import shutil
        shutil.rmtree(temp_dir)
        
        # Standardize columns
        if 'text' in df.columns and 'label' in df.columns:
            df = df[['text', 'label']].copy()
        elif 'message' in df.columns and 'label' in df.columns:
            df = df.rename(columns={'message': 'text'})
        elif 'Message' in df.columns and 'Category' in df.columns:
            df = df.rename(columns={'Message': 'text', 'Category': 'label'})
        else:
            text_cols = [col for col in df.columns if any(word in col.lower() for word in ['text', 'message', 'email', 'content'])]
            label_cols = [col for col in df.columns if any(word in col.lower() for word in ['label', 'class', 'spam', 'type', 'category'])]
            if text_cols and label_cols:
                df = df.rename(columns={text_cols[0]: 'text', label_cols[0]: 'label'})
            else:
                raise ValueError(f"Unexpected columns: {df.columns.tolist()}")
        
        # Normalize labels
        if df['label'].dtype == 'object':
            df['label'] = df['label'].str.lower()
        df['label'] = df['label'].astype(str)
        df['label'] = df['label'].replace({'spam': 'spam', 'ham': 'not_spam', 'not spam': 'not_spam', '0': 'not_spam', '1': 'spam'})
        df['label'] = df['label'].map({'spam': 'spam', 'not_spam': 'not_spam'}).fillna('not_spam')
        
        # Remove rows with missing text or labels
        df = df.dropna(subset=['text', 'label'])
        
        logger.info(f"Email dataset loaded: {len(df)} messages")
        logger.info(f"Spam messages: {(df['label'] == 'spam').sum()}")
        logger.info(f"Ham messages: {(df['label'] == 'not_spam').sum()}")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading email dataset: {e}")
        raise ValueError(f"Failed to download email dataset: {e}")


def download_additional_kaggle_datasets() -> pd.DataFrame:
    """Download additional spam datasets from Kaggle."""
    if not KAGGLE_AVAILABLE:
        return pd.DataFrame(columns=['text', 'label'])
    
    # Only use datasets we know work
    datasets_to_try = [
        "wcukierski/enron-email-dataset",
        "venky73/spam-mails-dataset",
        "shantanudhakadd/email-spam-detection-dataset-classification"
    ]
    
    additional_datasets = []
    
    for dataset_name in datasets_to_try:
        if not validate_kaggle_dataset(dataset_name):
            logger.warning(f"Skipping invalid dataset: {dataset_name}")
            continue
        
        try:
            api = KaggleApi()
            api.authenticate()
            
            temp_dir = f"temp_kaggle_{dataset_name.replace('/', '_')}"
            os.makedirs(temp_dir, exist_ok=True)
            
            api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
            
            if not csv_files:
                continue
            
            csv_path = os.path.join(temp_dir, csv_files[0])
            df = pd.read_csv(csv_path)
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir)
            
            # Standardize columns
            if 'message' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'message': 'text'})
            elif 'Message' in df.columns and 'Category' in df.columns:
                df = df.rename(columns={'Message': 'text', 'Category': 'label'})
            elif 'message' in df.columns and 'file' in df.columns:
                # Enron dataset - all legitimate emails
                df = df.rename(columns={'message': 'text'})
                df['label'] = 'not_spam'
                df = df[['text', 'label']].copy()
                # Sample to avoid extreme imbalance
                if len(df) > 10000:
                    df = df.sample(n=10000, random_state=42)
            elif 'text' in df.columns and 'label' in df.columns:
                # Already in correct format
                pass
            elif 'email' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'email': 'text'})
            elif 'content' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'content': 'text'})
            else:
                # Try to find text and label columns automatically
                text_cols = [col for col in df.columns if any(word in col.lower() for word in ['text', 'message', 'email', 'content', 'body'])]
                label_cols = [col for col in df.columns if any(word in col.lower() for word in ['label', 'class', 'spam', 'type', 'category', 'target'])]
                
                if text_cols and label_cols:
                    df = df.rename(columns={text_cols[0]: 'text', label_cols[0]: 'label'})
                else:
                    logger.warning(f"Could not map columns for {dataset_name}. Available columns: {df.columns.tolist()}")
                    continue
            
            # Set source label
            if "ozlerhakan" in dataset_name:
                df['source'] = 'kaggle_spam_classification'
            elif "wcukierski" in dataset_name:
                df['source'] = 'kaggle_enron_emails'
            elif "venky73" in dataset_name:
                df['source'] = 'kaggle_spam_mails'
            elif "shantanudhakadd" in dataset_name:
                df['source'] = 'kaggle_email_spam_detection'
            
            additional_datasets.append(df)
            logger.info(f"Successfully loaded {dataset_name}: {len(df)} messages")
            
        except Exception as e:
            logger.warning(f"Failed to download {dataset_name}: {e}")
            continue
    
    if additional_datasets:
        combined_df = pd.concat(additional_datasets, ignore_index=True)
        logger.info(f"Combined additional datasets: {len(combined_df)} total messages")
        return combined_df
    else:
        return pd.DataFrame(columns=['text', 'label'])

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize labels to 'spam' and 'not_spam'."""
    if df['label'].dtype == 'object':
        df['label'] = df['label'].str.lower()
    
    df['label'] = df['label'].replace({
        'spam': 'spam', 
        'ham': 'not_spam', 
        'not spam': 'not_spam',
        '0': 'not_spam', 
        '1': 'spam'
    })
    
    # Ensure all labels are properly normalized
    df['label'] = df['label'].map({'spam': 'spam', 'not_spam': 'not_spam'}).fillna('not_spam')
    
    return df

def combine_datasets(sms_df: pd.DataFrame, email_df: pd.DataFrame) -> pd.DataFrame:
    """Combine SMS and email datasets."""
    # Add source labels
    sms_df['source'] = 'uci_sms_spam'
    email_df['source'] = 'kaggle_spam_emails'
    
    # Combine datasets
    combined_df = pd.concat([sms_df, email_df], ignore_index=True)
    
    logger.info(f"Combined dataset: {len(combined_df)} total messages")
    return combined_df

def collect_data() -> pd.DataFrame:
    """Main function to collect and combine all datasets."""
    logger.info("Starting data collection process...")
    
    config = load_config()
    output_dir = config['data']['output_dir']
    ensure_dir_exists(output_dir)
    raw_data_path = os.path.join(output_dir, 'raw_data.csv')
    
    # Check if raw data already exists
    if os.path.exists(raw_data_path):
        logger.info(f"📁 Raw data already exists at {raw_data_path}")
        logger.info("🔄 Loading existing data instead of downloading...")
        
        try:
            combined_df = pd.read_csv(raw_data_path)
            logger.info(f"✅ Loaded existing dataset: {len(combined_df)} messages")
            
            # Print summary
            print(f"✅ Data loaded from existing file!")
            print(f"   Total messages: {len(combined_df)}")
            print(f"   Label distribution:\n{combined_df['label'].value_counts()}")
            if 'source' in combined_df.columns:
                print(f"   Source distribution:\n{combined_df['source'].value_counts()}")
            
            return combined_df
            
        except Exception as e:
            logger.warning(f"Error loading existing data: {e}")
            logger.info("🔄 Proceeding with fresh data download...")
    
    # Download datasets (only if raw data doesn't exist)
    logger.info("📱 Downloading SMS spam data...")
    sms_df = download_sms_spam_data()
    
    logger.info("📧 Downloading email spam data...")
    email_df = download_kaggle_email_dataset()
    
    # Check minimum data requirements
    min_required_messages = 1000
    if len(sms_df) + len(email_df) < min_required_messages:
        logger.error(f"Insufficient data collected: {len(sms_df) + len(email_df)} < {min_required_messages}")
        raise ValueError(f"Insufficient data collected. Need at least {min_required_messages} messages")
    
    logger.info("📊 Downloading additional datasets...")
    additional_df = download_additional_kaggle_datasets()
    
    # Normalize labels
    sms_df = normalize_labels(sms_df)
    email_df = normalize_labels(email_df)
    if not additional_df.empty:
        additional_df = normalize_labels(additional_df)
    
    # Combine datasets
    combined_df = combine_datasets(sms_df, email_df)
    
    # Add additional datasets if available
    if not additional_df.empty:
        combined_df = pd.concat([combined_df, additional_df], ignore_index=True)
        logger.info(f"Final combined dataset: {len(combined_df)} total messages")
    
    # Final validation
    label_counts = combined_df['label'].value_counts()
    if 'spam' not in label_counts or 'not_spam' not in label_counts:
        raise ValueError("Missing required labels. Both 'spam' and 'not_spam' labels are required.")
    
    if label_counts['spam'] < 100 or label_counts['not_spam'] < 100:
        raise ValueError("Insufficient class examples. Need at least 100 examples per class.")
    
    # Save raw data
    combined_df.to_csv(raw_data_path, index=False)
    logger.info(f"Raw data saved to {raw_data_path}")
    
    # Print summary
    print(f"✅ Data collection completed successfully!")
    print(f"   Total messages: {len(combined_df)}")
    print(f"   Label distribution:\n{combined_df['label'].value_counts()}")
    print(f"   Source distribution:\n{combined_df['source'].value_counts()}")
    
    return combined_df

if __name__ == "__main__":
    data = collect_data()
    print(f"Data collection completed. Total messages: {len(data)}")