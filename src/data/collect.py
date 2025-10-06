"""
Data collection module for spam detection project.
Downloads and combines multiple real datasets:
- SMS Spam Collection (UCI)
- Email Spam datasets (Kaggle)
- Additional spam datasets for comprehensive training
"""

import os
import pandas as pd
import requests
import zipfile
import io
import logging
from typing import Tuple
from utils.helpers import load_config, ensure_dir_exists, setup_logging

# Try to import kaggle, fallback if not available
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    logging.warning("Kaggle API not available. Will use sample datasets.")

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def download_sms_spam_data() -> pd.DataFrame:
    """
    Download and process SMS Spam Collection dataset from UCI.
    
    Returns:
        DataFrame with SMS data
    """
    logger.info("Downloading SMS Spam Collection dataset...")
    
    # URL for SMS Spam Collection dataset
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    try:
        # Download the zip file
        response = requests.get(url)
        response.raise_for_status()
        
        # Extract the zip file
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Find the CSV file in the zip
            csv_files = [f for f in zip_file.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise ValueError("No CSV file found in the zip archive")
            
            # Read the first CSV file with proper encoding
            with zip_file.open(csv_files[0]) as csv_file:
                # Try different encodings
                try:
                    df = pd.read_csv(csv_file, sep='\t', header=None, names=['label', 'text'], encoding='utf-8')
                except UnicodeDecodeError:
                    csv_file.seek(0)
                    df = pd.read_csv(csv_file, sep='\t', header=None, names=['label', 'text'], encoding='latin-1')
        
        # Clean the data - remove any extra columns and handle encoding issues
        df = df[['label', 'text']].copy()
        df = df.dropna()
        
        # Handle encoding issues in text
        df['text'] = df['text'].astype(str).apply(lambda x: x.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore'))
        
        logger.info(f"SMS dataset loaded: {len(df)} messages")
        logger.info(f"Spam messages: {(df['label'] == 'spam').sum()}")
        logger.info(f"Ham messages: {(df['label'] == 'ham').sum()}")
        return df
        
    except Exception as e:
        logger.error(f"Error downloading SMS dataset: {e}")
        # Fallback: create a sample dataset for demonstration
        logger.info("Creating sample SMS dataset...")
        return create_sample_sms_data()

def create_sample_sms_data() -> pd.DataFrame:
    """
    Create a sample SMS dataset for demonstration purposes.
    
    Returns:
        DataFrame with sample SMS data
    """
    sample_data = {
        'label': ['spam'] * 100 + ['ham'] * 400,
        'text': [
            # Spam messages
            'URGENT! You have won a £2000 Bonus Caller Prize!',
            'Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005.',
            'You have been selected to receive a £900 reward!',
            'Congratulations! You have won $1000 cash prize!',
            'URGENT! Your mobile number has won £2000.',
            'You have won a £2000 Bonus Caller Prize!',
            'Free entry in 2 a wkly comp to win FA Cup final tkts.',
            'You have been selected to receive a £900 reward!',
            'Congratulations! You have won $1000 cash prize!',
            'URGENT! Your mobile number has won £2000.',
        ] * 10 + [
            # Ham messages
            'Hey, how are you doing today?',
            'Can we meet for lunch tomorrow?',
            'Thanks for the help with the project.',
            'What time is the meeting?',
            'I will be there in 10 minutes.',
            'Have a great day!',
            'See you later.',
            'Thanks for the birthday wishes.',
            'How was your weekend?',
            'Looking forward to seeing you.',
        ] * 40
    }
    
    return pd.DataFrame(sample_data)

def download_kaggle_email_dataset() -> pd.DataFrame:
    """
    Download real email spam dataset from Kaggle.
    
    Returns:
        DataFrame with email data
    """
    if not KAGGLE_AVAILABLE:
        logger.warning("Kaggle API not available. Using sample email dataset.")
        return create_sample_email_data()
    
    logger.info("Downloading real email spam dataset from Kaggle...")
    
    try:
        # Initialize Kaggle API
        api = KaggleApi()
        api.authenticate()
        
        # Download the spam emails dataset
        dataset_name = "abdallahwagih/spam-emails"
        logger.info(f"Downloading dataset: {dataset_name}")
        
        # Create temporary directory for download
        temp_dir = "temp_kaggle_download"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Download dataset
        api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)
        
        # Find the CSV file
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        if not csv_files:
            raise ValueError("No CSV file found in downloaded dataset")
        
        # Read the first CSV file
        csv_path = os.path.join(temp_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir)
        
        # Standardize column names and labels
        if 'text' in df.columns and 'label' in df.columns:
            df = df[['text', 'label']].copy()
        elif 'message' in df.columns and 'label' in df.columns:
            df = df.rename(columns={'message': 'text'})
        elif 'email' in df.columns and 'label' in df.columns:
            df = df.rename(columns={'email': 'text'})
        else:
            # Try to infer columns
            text_cols = [col for col in df.columns if any(word in col.lower() for word in ['text', 'message', 'email', 'content'])]
            label_cols = [col for col in df.columns if any(word in col.lower() for word in ['label', 'class', 'spam', 'type'])]
            
            if text_cols and label_cols:
                df = df.rename(columns={text_cols[0]: 'text', label_cols[0]: 'label'})
            else:
                raise ValueError("Could not identify text and label columns")
        
        # Normalize labels
        df['label'] = df['label'].str.lower()
        df['label'] = df['label'].replace({'spam': 'spam', 'ham': 'not_spam', 'not spam': 'not_spam', '0': 'not_spam', '1': 'spam'})
        
        # Remove any rows with missing text or labels
        df = df.dropna(subset=['text', 'label'])
        
        logger.info(f"Real email dataset loaded: {len(df)} messages")
        logger.info(f"Spam messages: {(df['label'] == 'spam').sum()}")
        logger.info(f"Ham messages: {(df['label'] == 'not_spam').sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error downloading Kaggle email dataset: {e}")
        logger.info("Falling back to sample email dataset...")
        return create_sample_email_data()

def create_sample_email_data() -> pd.DataFrame:
    """
    Create a sample email dataset as fallback.
    
    Returns:
        DataFrame with sample email data
    """
    logger.info("Creating sample email dataset as fallback...")
    
    sample_emails = {
        'label': ['spam'] * 200 + ['not_spam'] * 300,
        'text': [
            # Spam emails
            'Get rich quick! Make money from home with our amazing program!',
            'URGENT: Your account has been compromised. Click here to verify.',
            'Congratulations! You have won $5000 in our lottery!',
            'Limited time offer! 50% off all products. Order now!',
            'You have been selected for a special promotion!',
            'Make money online with our proven system!',
            'URGENT: Verify your account immediately!',
            'You have won a free vacation! Click to claim.',
            'Special offer: Get 70% off today only!',
            'Your account will be closed unless you verify now!',
        ] * 20 + [
            # Ham emails
            'Hi, I wanted to follow up on our meeting yesterday.',
            'Thank you for your email. I will get back to you soon.',
            'The project deadline has been extended to next Friday.',
            'Please find attached the report you requested.',
            'I hope you are doing well. How is the new job?',
            'Can we schedule a call for tomorrow afternoon?',
            'Thanks for the great presentation today.',
            'I have reviewed the proposal and it looks good.',
            'The team meeting has been moved to 3 PM.',
            'Looking forward to working with you on this project.',
        ] * 30
    }
    
    return pd.DataFrame(sample_emails)

def download_additional_kaggle_datasets() -> pd.DataFrame:
    """
    Download additional spam datasets from Kaggle for more comprehensive training.
    
    Returns:
        DataFrame with additional email data
    """
    if not KAGGLE_AVAILABLE:
        logger.warning("Kaggle API not available. Skipping additional datasets.")
        return pd.DataFrame(columns=['text', 'label'])
    
    logger.info("Downloading additional Kaggle spam datasets...")
    
    additional_datasets = []
    
    # List of additional datasets to try
    datasets_to_try = [
        "ozlerhakan/spam-or-not-spam-dataset",
        # Add more datasets as needed
    ]
    
    for dataset_name in datasets_to_try:
        try:
            logger.info(f"Attempting to download: {dataset_name}")
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Create temporary directory for download
            temp_dir = f"temp_kaggle_{dataset_name.replace('/', '_')}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Download dataset
            api.dataset_download_files(dataset_name, path=temp_dir, unzip=True)
            
            # Find the CSV file
            csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
            if not csv_files:
                logger.warning(f"No CSV file found in {dataset_name}")
                continue
            
            # Read the first CSV file
            csv_path = os.path.join(temp_dir, csv_files[0])
            df = pd.read_csv(csv_path)
            
            # Clean up temporary directory
            import shutil
            shutil.rmtree(temp_dir)
            
            # Standardize column names
            if 'text' in df.columns and 'label' in df.columns:
                df = df[['text', 'label']].copy()
            elif 'message' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'message': 'text'})
            elif 'email' in df.columns and 'label' in df.columns:
                df = df.rename(columns={'email': 'text'})
            else:
                # Try to infer columns
                text_cols = [col for col in df.columns if any(word in col.lower() for word in ['text', 'message', 'email', 'content'])]
                label_cols = [col for col in df.columns if any(word in col.lower() for word in ['label', 'class', 'spam', 'type'])]
                
                if text_cols and label_cols:
                    df = df.rename(columns={text_cols[0]: 'text', label_cols[0]: 'label'})
                else:
                    logger.warning(f"Could not identify text and label columns in {dataset_name}")
                    continue
            
            # Normalize labels
            df['label'] = df['label'].str.lower()
            df['label'] = df['label'].replace({'spam': 'spam', 'ham': 'not_spam', 'not spam': 'not_spam', '0': 'not_spam', '1': 'spam'})
            
            # Remove any rows with missing text or labels
            df = df.dropna(subset=['text', 'label'])
            
            if len(df) > 0:
                additional_datasets.append(df)
                logger.info(f"Successfully loaded {dataset_name}: {len(df)} messages")
            else:
                logger.warning(f"No valid data found in {dataset_name}")
                
        except Exception as e:
            logger.warning(f"Failed to download {dataset_name}: {e}")
            continue
    
    if additional_datasets:
        combined_df = pd.concat(additional_datasets, ignore_index=True)
        logger.info(f"Combined additional datasets: {len(combined_df)} total messages")
        logger.info(f"Spam messages: {(combined_df['label'] == 'spam').sum()}")
        logger.info(f"Ham messages: {(combined_df['label'] == 'not_spam').sum()}")
        return combined_df
    else:
        logger.info("No additional datasets could be downloaded")
        return pd.DataFrame(columns=['text', 'label'])

def normalize_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize labels to 'spam' and 'not_spam'.
    
    Args:
        df: DataFrame with label column
        
    Returns:
        DataFrame with normalized labels
    """
    logger.info("Normalizing labels...")
    
    # Create a copy to avoid modifying original
    df_normalized = df.copy()
    
    # Map various spam indicators to 'spam'
    spam_indicators = ['spam', '1', 1, True, 'yes', 'y']
    ham_indicators = ['ham', '0', 0, False, 'no', 'n', 'not_spam']
    
    # Convert labels to lowercase if they are strings
    if df_normalized['label'].dtype == 'object':
        df_normalized['label'] = df_normalized['label'].str.lower()
    
    # Normalize labels
    df_normalized['label'] = df_normalized['label'].apply(
        lambda x: 'spam' if x in spam_indicators else 'not_spam'
    )
    
    logger.info(f"Label distribution: {df_normalized['label'].value_counts().to_dict()}")
    return df_normalized

def combine_datasets(sms_df: pd.DataFrame, email_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine SMS and email datasets.
    
    Args:
        sms_df: SMS dataset
        email_df: Email dataset
        
    Returns:
        Combined dataset
    """
    logger.info("Combining datasets...")
    
    # Ensure both datasets have the same column names
    if 'text' not in sms_df.columns or 'label' not in sms_df.columns:
        raise ValueError("SMS dataset must have 'text' and 'label' columns")
    if 'text' not in email_df.columns or 'label' not in email_df.columns:
        raise ValueError("Email dataset must have 'text' and 'label' columns")
    
    # Combine datasets
    combined_df = pd.concat([sms_df, email_df], ignore_index=True)
    
    # Add dataset source for tracking
    combined_df['source'] = ['sms'] * len(sms_df) + ['email'] * len(email_df)
    
    logger.info(f"Combined dataset: {len(combined_df)} total messages")
    logger.info(f"Source distribution: {combined_df['source'].value_counts().to_dict()}")
    
    return combined_df

def collect_data() -> pd.DataFrame:
    """
    Main function to collect and combine all datasets.
    
    Returns:
        Combined and normalized dataset
    """
    logger.info("Starting data collection process...")
    
    # Load configuration
    config = load_config()
    
    # Download/collect datasets
    sms_df = download_sms_spam_data()
    email_df = download_kaggle_email_dataset()
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
        logger.info("Adding additional datasets to combined dataset...")
        additional_df['source'] = 'kaggle_additional'
        combined_df = pd.concat([combined_df, additional_df], ignore_index=True)
        logger.info(f"Final combined dataset: {len(combined_df)} total messages")
        logger.info(f"Source distribution: {combined_df['source'].value_counts().to_dict()}")
    
    # Save raw data
    output_dir = config['data']['output_dir']
    ensure_dir_exists(output_dir)
    
    raw_data_path = os.path.join(output_dir, 'raw_data.csv')
    combined_df.to_csv(raw_data_path, index=False)
    logger.info(f"Raw data saved to {raw_data_path}")
    
    return combined_df

if __name__ == "__main__":
    # Run data collection
    data = collect_data()
    print(f"Data collection completed. Total messages: {len(data)}")
    print(f"Label distribution:\n{data['label'].value_counts()}")
    print(f"Source distribution:\n{data['source'].value_counts()}")
