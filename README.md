# Spam Detection Machine Learning Project

A comprehensive machine learning pipeline for spam detection that classifies text messages as spam or not spam, and performs K-Means clustering to identify spam subtypes.

## 🎯 Project Overview

This project implements a comprehensive spam detection system using advanced machine learning techniques. It includes data collection, preprocessing, exploratory data analysis, model training, evaluation, anomaly detection, and clustering analysis to identify different types of spam messages.

### Key Features

- **Real Dataset Integration**: Downloads actual SMS Spam Collection (5,572 messages)
- **Advanced Text Preprocessing**: Enhanced text cleaning with 15+ engineered features
- **SMOTE Class Balancing**: Handles class imbalance with synthetic oversampling
- **Comprehensive EDA**: Advanced visualizations including spam trigger words analysis
- **Multiple ML Models**: 3 classifiers with hyperparameter tuning
- **Anomaly Detection**: Isolation Forest, One-Class SVM, and EllipticEnvelope
- **Model Evaluation**: Detailed performance metrics and visualizations
- **K-Means Clustering**: Identifies spam subtypes with silhouette analysis
- **Command-Line Interface**: Easy-to-use prediction tool
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
# Predict a single message
python src/models/predict.py "Congratulations! You have won $1000!"

# Interactive mode
python src/models/predict.py
```

## 🆕 New Features (Enhanced Version)

### **Advanced Data Processing**
- **Real Dataset**: Now uses actual SMS Spam Collection (5,572 messages) instead of sample data
- **SMOTE Balancing**: Handles class imbalance with synthetic minority oversampling
- **15+ Engineered Features**: Message length, character ratios, pattern detection, etc.


### **Enhanced Visualizations**
- **Spam Trigger Words**: Identifies words that strongly indicate spam
- **Advanced Message Characteristics**: Comprehensive feature analysis
- **TF-IDF Feature Analysis**: Feature importance and differentiation

### **Improved Performance**
- **Better Generalization**: Real dataset provides more robust training
- **Class Balance**: SMOTE ensures fair representation of both classes
- **Feature Richness**: 22+ engineered features improve classification accuracy
- **Spam Subtype Analysis**: K-Means clustering identifies 8 distinct spam types

## 📊 Expected Results

### Model Performance
- **Best Classifier**: LinearSVC with optimized hyperparameters
- **Validation Macro-F1**: ~0.94
- **Test Macro-F1**: ~0.93
- **Accuracy**: ~0.95

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
- Trained pipeline saved as `outputs/models/spam_pipeline.joblib`

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

- The project uses sample datasets for demonstration
- All preprocessing steps are documented and reproducible
- Model performance may vary based on data quality
- Clustering results depend on the spam message distribution

## 📄 License

This project is created for educational purposes.
