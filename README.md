# Spam Detection Machine Learning Project

A comprehensive machine learning pipeline for spam detection that classifies text messages as spam or not spam, and performs K-Means clustering to identify spam subtypes.

## 🎯 Project Overview

This project implements a comprehensive spam detection system using advanced machine learning techniques. It includes data collection, preprocessing, exploratory data analysis, model training, evaluation, anomaly detection, and clustering analysis to identify different types of spam messages.

### Key Features

- **Multi-Source Dataset Integration**: Downloads 23,742+ messages from 4 different datasets
- **Efficient Model Storage**: Metadata-based best model tracking (no duplication)
- **Advanced Text Preprocessing**: Enhanced text cleaning with 27+ engineered features
- **SMOTE Class Balancing**: Handles class imbalance with synthetic oversampling
- **Comprehensive EDA**: Advanced visualizations including spam trigger words analysis
- **Multiple ML Models**: 3 classifiers with hyperparameter tuning and automatic best model selection
- **Dynamic Model Management**: Automatic cleanup of old models and best model tracking
- **Model Evaluation**: Detailed performance metrics and visualizations
- **K-Means Clustering**: Identifies spam subtypes with silhouette analysis
- **Enhanced CLI**: Interactive and command-line prediction with confidence scores
- **Reproducible Research**: Fixed random seeds and comprehensive logging

## 📁 Project Structure

```
src/
├── data/
│   ├── collect.py          # Data collection from various sources
│   ├── preprocess.py       # Text preprocessing and data splitting
│   └── eda.py             # Exploratory data analysis and visualizations
├── models/
│   ├── train.py           # Model training with grid search
│   ├── evaluate.py        # Model evaluation and performance analysis
│   ├── cluster.py         # K-Means clustering analysis
│   ├── anomaly_detection.py # Advanced anomaly detection
│   └── predict.py         # CLI prediction tool
├── utils/
│   ├── config.yaml        # Configuration parameters
│   └── helpers.py         # Utility functions
└── requirements.txt       # Python dependencies

outputs/
├── eda/                   # EDA plots and analysis
├── models/                # Trained models
├── plots/                 # Model evaluation and clustering plots
└── reports/               # Performance reports and metrics
```

## 🚀 Quick Start

### 1. Environment Setup

#### Using Conda (Recommended)
```bash
# Create conda environment
conda create -n spam-detection python=3.9
conda activate spam-detection

# Install dependencies
cd src
pip install -r requirements.txt
```

#### Using Virtual Environment
```bash
# Create virtual environment
python -m venv spam-detection-env
source spam-detection-env/bin/activate  # On Windows: spam-detection-env\Scripts\activate

# Install dependencies
cd src
pip install -r requirements.txt

# Set up Kaggle API for real datasets (optional)
pip install kaggle
# See KAGGLE_SETUP.md for detailed API setup instructions
```

### 2. Run the Complete Pipeline

```bash
# Run the entire pipeline
python src/run_pipeline.py
```

### 3. Individual Components

```bash
# Data collection and preprocessing
python src/data/collect.py
python src/data/preprocess.py

# Exploratory data analysis
python src/data/eda.py

# Model training
python src/models/train.py

# Model evaluation
python src/models/evaluate.py

# Clustering analysis
python src/models/cluster.py
```

### 4. Make Predictions

```bash
# Predict a single message (uses best model automatically)
python src/models/predict.py "Congratulations! You have won $1000!"

# Use a specific model
python src/models/predict.py --model multinomial_nb "Free money! Click here now!"

# List available models
python src/models/predict.py --list-models

# Interactive mode
python src/models/predict.py
```

#### CLI Features
- **Automatic Best Model Selection**: Uses the highest-performing model by default
- **Model-Specific Predictions**: Choose specific models with `--model` flag
- **Confidence Scores**: Shows prediction confidence as percentages
- **Visual Indicators**: 🚨 for spam, ✅ for legitimate messages
- **Dynamic Thresholds**: Adjusts spam detection based on message characteristics

## 📊 Training Datasets

The system uses **4 comprehensive datasets** totaling **23,742 messages**:

### **Primary Datasets**
1. **UCI SMS Spam Collection** (5,572 messages)
   - Source: University of California Irvine
   - Content: SMS messages with spam/ham labels
   - Balance: ~87% legitimate, ~13% spam

2. **Kaggle Email Spam Classification** (2,999 messages)
   - Source: Kaggle (ozlerhakan/spam-or-not-spam-dataset)
   - Content: Email messages with classification labels
   - Balance: ~83% legitimate, ~17% spam

### **Additional Datasets**
3. **Enron Email Dataset** (10,000 messages)
   - Source: Kaggle (wcukierski/enron-email-dataset)
   - Content: Corporate email communications (all legitimate)
   - Purpose: Provides diverse legitimate message patterns

4. **Spam Mails Dataset** (5,171 messages)
   - Source: Kaggle (venky73/spam-mails-dataset)
   - Content: Email spam examples and legitimate messages
   - Purpose: Additional spam pattern diversity

### **Data Processing**
- **Automatic Download**: Scripts download and combine all datasets
- **Smart Column Mapping**: Handles different dataset formats automatically
- **Deduplication**: Removes duplicate messages (6.7% removed)
- **Balancing**: SMOTE oversampling for class balance
- **Final Training Set**: 5,177 messages (70% train, 15% validation, 15% test)

## 🆕 Enhanced Features

### **Efficient Model Storage**
- **Metadata-Based Tracking**: `best_model.json` tracks the best performing model
- **No Duplication**: Eliminates wasteful storage of duplicate models
- **Automatic Cleanup**: Removes old models before training new ones
- **Storage Savings**: ~2.1MB saved per training run

### **Advanced Data Processing**
- **Multi-Source Integration**: 23,742+ messages from 4 different datasets
- **SMOTE Balancing**: Handles class imbalance with synthetic minority oversampling
- **27+ Engineered Features**: Message length, character ratios, pattern detection, etc.

### **Enhanced Visualizations**
- **Spam Trigger Words**: Identifies words that strongly indicate spam
- **Advanced Message Characteristics**: Comprehensive feature analysis
- **TF-IDF Feature Analysis**: Feature importance and differentiation

### **Improved Performance**
- **Better Generalization**: Diverse datasets provide more robust training
- **Class Balance**: SMOTE ensures fair representation of both classes
- **Feature Richness**: 27+ engineered features improve classification accuracy
- **Reduced False Positives**: Significantly improved legitimate message detection

## 📊 Current Performance Results

### Model Performance
- **Best Classifier**: MultinomialNB with optimized hyperparameters
- **Validation Macro-F1**: 0.9444
- **Cross-Validation Score**: 0.9411
- **Best Parameters**: `{'classifier__alpha': 0.1}`

### Real-World Test Results
- **Legitimate Messages**: 91-97% confidence (excellent detection)
- **Spam Messages**: 94-99% confidence (high accuracy)
- **False Positives**: Significantly reduced with expanded training data
- **Dynamic Thresholds**: Adapts to message characteristics for better accuracy

### Clustering Results
- **Best k**: 8 clusters
- **Silhouette Score**: ~0.42
- **Spam Subtypes**: Identified through top TF-IDF terms per cluster

## 🔧 Configuration

Edit `src/utils/config.yaml` to modify:
- Data preprocessing parameters
- Model hyperparameters
- Clustering settings
- Evaluation metrics

## 📈 Outputs

The pipeline generates:

### Plots
- Class distribution analysis
- Message length distributions
- Top words and character n-grams
- Model performance comparisons
- ROC and Precision-Recall curves
- Clustering visualizations

### Reports
- Classification reports
- Confusion matrices
- Clustering analysis
- Performance metrics

### Models
- **Best Model**: Automatically selected and tracked via `outputs/models/best_model.json`
- **Individual Models**: `multinomial_nb.joblib`, `logistic_regression.joblib`, `linear_svc.joblib`
- **Metadata**: Model performance scores and timestamps stored in JSON format
- **Automatic Management**: Old models cleaned up automatically before retraining

## 🧪 Reproducibility

- All random seeds fixed to 42
- Version information logged
- Configuration parameters stored
- Complete pipeline automation

## 📚 Dependencies

- pandas >= 1.5.0
- numpy >= 1.21.0
- scikit-learn >= 1.1.0
- matplotlib >= 3.5.0
- seaborn >= 0.11.0
- joblib >= 1.2.0
- pyyaml >= 6.0

## 📝 Notes

- The project uses real datasets from UCI and Kaggle (with sample fallback)
- All preprocessing steps are documented and reproducible
- Model performance may vary based on data quality
- Clustering results depend on the spam message distribution

## 📄 License

This project is created for educational purposes.
