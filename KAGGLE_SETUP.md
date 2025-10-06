# Kaggle API Setup Guide

This project now uses real datasets from Kaggle instead of synthetic data. Follow these steps to set up the Kaggle API:

## 1. Install Kaggle API

```bash
pip install kaggle
```

## 2. Get Your Kaggle API Credentials

1. Go to [Kaggle.com](https://www.kaggle.com) and sign in
2. Click on your profile picture (top right)
3. Go to "Account" tab
4. Scroll down to "API" section
5. Click "Create New API Token"
6. This downloads `kaggle.json` file

## 3. Set Up API Credentials

### Option A: Place kaggle.json in the right location

**Linux/Mac:**
```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Windows:**
```cmd
mkdir %USERPROFILE%\.kaggle
copy kaggle.json %USERPROFILE%\.kaggle\
```

### Option B: Set Environment Variables

```bash
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

## 4. Test the Setup

```bash
kaggle datasets list
```

## 5. Run the Project

Now when you run the spam detection pipeline, it will automatically download real datasets:

```bash
python src/run_pipeline.py
```

## Datasets Used

The project will attempt to download these real datasets:

1. **SMS Spam Collection** (UCI) - 5,572 SMS messages
2. **Spam Emails** (Kaggle: abdallahwagih/spam-emails) - Real email spam dataset
3. **Spam or Not Spam** (Kaggle: ozlerhakan/spam-or-not-spam-dataset) - Additional spam dataset

## Fallback Behavior

If Kaggle API is not available or datasets can't be downloaded:
- The system will automatically fall back to sample datasets
- You'll see warning messages in the logs
- The pipeline will still work with synthetic data

## Troubleshooting

**Error: "403 - Forbidden"**
- Check your API credentials
- Ensure your Kaggle account is verified
- Make sure you've accepted the dataset terms of use

**Error: "Dataset not found"**
- Some datasets may be private or removed
- The system will skip unavailable datasets and continue

**Error: "No module named 'kaggle'"**
- Install kaggle: `pip install kaggle`
- Or the system will use sample data automatically
