# Spam Detection Machine Learning Project

A comprehensive machine learning pipeline for spam detection using text classification and clustering analysis. This project implements multiple algorithms, feature engineering, and provides both training and prediction capabilities.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd innovation
   ```

2. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv spam-detection-env
   
   # Activate virtual environment
   # On Linux/Mac:
   source spam-detection-env/bin/activate
   
   # On Windows:
   spam-detection-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r src/requirements.txt
   ```

4. **Run the complete pipeline:**
   ```bash
   python src/run_pipeline.py
   ```

5. **Test prediction (optional):**
   ```bash
   # Test with a spam message
   python src/models/predict.py "Congratulations! You have won $1000!"
   
   # Test with a normal message
   python src/models/predict.py "Hey, how are you doing today?"
   ```

That's it! The pipeline will automatically load existing data (if available) or download it, then run the complete machine learning workflow.

## 📁 Project Structure

```
innovation/
├── src/                          # Source code
│   ├── data/                     # Data processing modules
│   │   ├── collect.py           # Data collection and download
│   │   ├── preprocess.py        # Text preprocessing and feature engineering
│   │   └── eda.py              # Exploratory data analysis
│   ├── models/                   # Machine learning models
│   │   ├── train.py            # Model training and hyperparameter tuning
│   │   ├── evaluate.py         # Model evaluation and metrics
│   │   ├── predict.py          # Prediction interface
│   │   └── cluster.py          # K-means clustering analysis
│   ├── utils/                    # Utility functions
│   │   ├── config.yaml         # Configuration parameters
│   │   └── helpers.py          # Helper functions
│   └── run_pipeline.py          # Main pipeline script
├── outputs/                      # Generated outputs
│   ├── eda/                     # EDA plots and analysis
│   ├── models/                  # Trained models
│   ├── plots/                   # Clustering visualizations
│   ├── reports/                 # Evaluation reports
│   └── raw_data.csv            # Raw dataset
└── README.md                    # This file
```

## 🔧 Individual Module Usage

### 1. Data Collection (`src/data/collect.py`)
Downloads and combines spam datasets from multiple sources.

```bash
# Make sure virtual environment is created and activated
python -m venv spam-detection-env
source spam-detection-env/bin/activate  # Linux/Mac
# or spam-detection-env\Scripts\activate  # Windows
pip install -r src/requirements.txt

python src/data/collect.py
```

**What it does:**
- Downloads SMS spam data from UCI ML Repository
- Downloads email spam data from Kaggle
- Combines and normalizes datasets
- Saves raw data to `outputs/raw_data.csv`
- **Note:** If `raw_data.csv` already exists, it will load the existing file instead of downloading

### 2. Data Preprocessing (`src/data/preprocess.py`)
Cleans text data and creates train/validation/test splits.

```bash
python src/data/preprocess.py
```

**What it does:**
- Cleans and normalizes text (removes headers, special characters)
- Creates 22 engineered features (length, word count, special characters, etc.)
- Applies SMOTE for class balancing
- Splits data into train/validation/test sets
- Saves processed data to `outputs/train.csv`, `outputs/val.csv`, `outputs/test.csv`

### 3. Exploratory Data Analysis (`src/data/eda.py`)
Creates visualizations and analyzes text patterns.

```bash
python src/data/eda.py
```

**What it does:**
- Generates class distribution plots
- Analyzes message characteristics
- Creates word frequency analysis
- Identifies spam trigger words
- Saves all plots to `outputs/eda/`

### 4. Model Training (`src/models/train.py`)
Trains multiple classification models with hyperparameter tuning.

```bash
python src/models/train.py
```

**What it does:**
- Trains 3 models: Multinomial Naive Bayes, Logistic Regression, Linear SVM
- Performs grid search for hyperparameter optimization
- Uses 5-fold cross-validation
- Saves best model and metadata to `outputs/models/`

### 5. Model Evaluation (`src/models/evaluate.py`)
Evaluates trained models and creates performance reports.

```bash
python src/models/evaluate.py
```

**What it does:**
- Generates confusion matrices
- Creates ROC and Precision-Recall curves
- Computes comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Saves evaluation plots to `outputs/reports/`

### 6. Clustering Analysis (`src/models/cluster.py`)
Performs K-means clustering on spam messages to identify subtypes.

```bash
python src/models/cluster.py
```

**What it does:**
- Clusters spam messages using K-means
- Tests different k values (5, 8, 12)
- Uses silhouette analysis to find optimal k
- Identifies spam subtypes through TF-IDF analysis
- Saves clustering visualizations to `outputs/plots/`

### 7. Prediction Interface (`src/models/predict.py`)
Command-line interface for making predictions on new messages.

```bash
# Make sure virtual environment is created and activated
python -m venv spam-detection-env
source spam-detection-env/bin/activate  # Linux/Mac
pip install -r src/requirements.txt

# Predict a single message
python src/models/predict.py "Your message here"

# Interactive mode
python src/models/predict.py

# Batch prediction from file
python src/models/predict.py --file messages.txt
```

**What it does:**
- Loads the best trained model
- Preprocesses input text
- Makes predictions with confidence scores
- Supports interactive and batch modes

## 📊 Output Files

After running the pipeline, you'll find the following outputs:

### Data Files
- `outputs/raw_data.csv` - Combined raw dataset
- `outputs/train.csv` - Training data
- `outputs/val.csv` - Validation data  
- `outputs/test.csv` - Test data

### Models
- `outputs/models/best_model.json` - Best model metadata
- `outputs/models/*.joblib` - Trained model files

### Visualizations
- `outputs/eda/` - EDA plots (class distribution, word analysis, etc.)
- `outputs/reports/` - Model evaluation plots (confusion matrices, ROC curves)
- `outputs/plots/` - Clustering analysis plots

### Reports
- `outputs/reports/evaluation_results.csv` - Detailed performance metrics
- `outputs/final_summary.txt` - Complete project summary

## ⚙️ Configuration

The project uses `src/utils/config.yaml` for configuration. Key parameters:

```yaml
# Data split ratios
preprocessing:
  test_size: 0.15
  val_size: 0.15

# Model hyperparameters
models:
  multinomial_nb:
    alpha: [0.1, 0.5, 1.0]
  logistic_regression:
    C: [0.5, 1, 2, 4]
  linear_svc:
    C: [0.5, 1, 2, 4]

# Clustering parameters
clustering:
  k_values: [5, 8, 12]
```

## 🎯 Key Features

- **Multiple Algorithms**: Implements Naive Bayes, Logistic Regression, and SVM
- **Advanced Feature Engineering**: 22 engineered features including text statistics
- **Class Balancing**: Uses SMOTE to handle imbalanced datasets
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Clustering Analysis**: K-means clustering to identify spam subtypes
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Easy Prediction**: Command-line interface for new predictions
- **Reproducible**: Fixed random seeds and complete automation

## 📈 Performance

The pipeline typically achieves:
- **Accuracy**: >95% on test set
- **F1-Score**: >0.90 for both classes
- **ROC-AUC**: >0.95
- **Best Model**: Usually Linear SVM or Logistic Regression

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the project root directory and virtual environment is activated
2. **Missing Dependencies**: Run `pip install -r src/requirements.txt`
3. **Data Download Issues**: The pipeline will use existing data if available
4. **Memory Issues**: For large datasets, consider reducing `max_features` in config
5. **Virtual Environment Issues**: Make sure to activate the virtual environment before running any scripts

### Quick Verification

To verify the setup is working, try running a simple prediction:

```bash
# Create and activate virtual environment
python -m venv spam-detection-env
source spam-detection-env/bin/activate  # Linux/Mac
pip install -r src/requirements.txt

# Test prediction
python src/models/predict.py "Congratulations! You have won $1000!"
```

This should return a spam prediction with high confidence.

### Logs
All operations are logged to `spam_detection.log` for debugging.

## 📚 Technical Details

### Algorithms Used
- **Multinomial Naive Bayes**: Fast baseline classifier
- **Logistic Regression**: Linear classifier with regularization
- **Linear SVM**: Support vector machine for text classification
- **K-Means Clustering**: Unsupervised learning for spam subtype analysis

### Feature Engineering
- Text length and word count statistics
- Special character and digit counts
- URL and email pattern detection
- TF-IDF vectorization with n-grams
- SMOTE oversampling for class balance

### Evaluation Metrics
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC and Precision-Recall AUC
- Confusion matrices and classification reports
- Cross-validation with 5 folds

## 🎓 Academic Use

This project is designed for educational and research purposes. It demonstrates:
- Complete ML pipeline implementation
- Multiple algorithm comparison
- Feature engineering techniques
- Model evaluation best practices
- Clustering for pattern discovery

## 📝 License

This project is for educational use. Please cite appropriately if used in academic work.

---

**Note**: The pipeline automatically handles data loading. If `outputs/raw_data.csv` exists, it will be used instead of downloading new data, making the setup process faster and more reliable.