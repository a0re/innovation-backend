# Spam Detection Machine Learning Project

A comprehensive spam detection system featuring a complete ML pipeline and production-ready REST API. This project implements multiple classification algorithms, clustering analysis, and provides both training and real-time prediction capabilities through FastAPI.

## 🚀 Quick Start

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone and navigate to the backend:**
   ```bash
   cd innovation-backend
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv spam-detection-env

   # On Linux/Mac:
   source spam-detection-env/bin/activate

   # On Windows:
   spam-detection-env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   # Install ML pipeline dependencies
   pip install -r src/requirements.txt

   # Install API dependencies
   pip install -r api/requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run the ML pipeline (train models):**
   ```bash
   python src/run_pipeline.py
   ```

6. **Start the API server:**
   ```bash
   cd api
   python main.py
   ```

The API will be available at `http://localhost:8000` with interactive docs at `/docs`.

---

## 📁 Project Structure

```
innovation-backend/
├── src/                          # ML Pipeline
│   ├── data/                     # Data processing
│   │   ├── collect.py           # Data collection/download
│   │   ├── preprocess.py        # Text preprocessing
│   │   └── eda.py              # Exploratory analysis
│   ├── models/                   # ML models
│   │   ├── train.py            # Model training
│   │   ├── evaluate.py         # Evaluation
│   │   ├── predict.py          # Prediction logic
│   │   └── cluster.py          # K-means clustering
│   ├── utils/                    # Utilities
│   │   ├── config.yaml         # Configuration
│   │   └── helpers.py          # Helper functions
│   └── run_pipeline.py          # Main pipeline
├── api/                          # REST API
│   ├── main.py                  # FastAPI application
│   ├── database.py              # Database operations
│   ├── seed_database.py         # Test data generator
│   └── requirements.txt         # API dependencies
├── outputs/                      # Generated files
│   ├── models/                  # Trained models
│   ├── eda/                     # Analysis plots
│   ├── reports/                 # Evaluation reports
│   └── plots/                   # Clustering viz
└── .env.example                 # Environment template
```

---

## 🌐 REST API

### Starting the Server

```bash
cd api
python main.py
```

Or with custom settings:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Interactive Documentation

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### API Endpoints

#### **Predictions**
- `POST /predict` - Single message prediction (best model)
- `POST /predict/batch` - Batch predictions (up to 100 messages)
- `POST /predict/multi-model` - Multi-model ensemble prediction
- `POST /predict/multi-model/batch` - Batch multi-model predictions

#### **Statistics & Info**
- `GET /stats` - Prediction statistics from database
- `DELETE /stats` - Reset statistics
- `GET /examples` - Example spam/not-spam messages
- `GET /cluster/info` - Clustering model information
- `GET /health` - API health check
- `GET /model/info` - Model metadata

### Example API Usage

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/predict/multi-model" \
  -H "Content-Type: application/json" \
  -d '{"message": "Congratulations! You won $1000!"}'
```

**Response:**
```json
{
  "message": "Congratulations! You won $1000!",
  "processed_message": "congratulations won",
  "multinomial_nb": {"prediction": "spam", "confidence": 0.9876, "is_spam": true},
  "logistic_regression": {"prediction": "spam", "confidence": 0.9654, "is_spam": true},
  "linear_svc": {"prediction": "spam", "confidence": 0.9823, "is_spam": true},
  "ensemble": {
    "prediction": "spam",
    "confidence": 0.9784,
    "is_spam": true,
    "spam_votes": 3,
    "total_votes": 3
  },
  "cluster": {
    "cluster_id": 0,
    "confidence": 0.85,
    "top_terms": [
      {"term": "prize", "score": 0.45},
      {"term": "winner", "score": 0.38}
    ],
    "total_clusters": 4
  },
  "timestamp": "2025-11-05T10:30:45.123456"
}
```

### Environment Configuration

Create a `.env` file:
```env
# Server
HOST=0.0.0.0
PORT=8000
RELOAD=true
LOG_LEVEL=info

# CORS (comma-separated origins)
CORS_ORIGINS=http://localhost:5174,http://localhost:5173

# Database
DATABASE_URL=sqlite:///./spam_detection.db
```

---

## 🤖 ML Pipeline

### Complete Pipeline Execution

```bash
python src/run_pipeline.py
```

This runs the full workflow:
1. Data collection
2. Preprocessing
3. EDA
4. Model training
5. Evaluation
6. Clustering analysis

### Individual Modules

#### 1. **Data Collection** (`src/data/collect.py`)
```bash
python src/data/collect.py
```
- Downloads SMS spam data (UCI ML Repository)
- Downloads email spam data (Kaggle)
- Combines and saves to `outputs/raw_data.csv`
- Skips download if file exists

#### 2. **Data Preprocessing** (`src/data/preprocess.py`)
```bash
python src/data/preprocess.py
```
- Text cleaning and normalization
- Feature engineering (22 features)
- SMOTE class balancing
- Train/val/test split
- Saves processed CSVs

#### 3. **Exploratory Data Analysis** (`src/data/eda.py`)
```bash
python src/data/eda.py
```
- Class distribution plots
- Message characteristics analysis
- Word frequency analysis
- Spam trigger word identification
- Saves plots to `outputs/eda/`

#### 4. **Model Training** (`src/models/train.py`)
```bash
python src/models/train.py
```
- Trains 3 models: Multinomial NB, Logistic Regression, Linear SVM
- Grid search hyperparameter tuning
- 5-fold cross-validation
- Saves best model to `outputs/models/`

#### 5. **Model Evaluation** (`src/models/evaluate.py`)
```bash
python src/models/evaluate.py
```
- Confusion matrices
- ROC and Precision-Recall curves
- Comprehensive metrics (accuracy, precision, recall, F1, AUC)
- Saves reports to `outputs/reports/`

#### 6. **Clustering Analysis** (`src/models/cluster.py`)
```bash
python src/models/cluster.py
```
- K-means clustering on spam messages
- Tests k values: 5, 8, 12
- Silhouette analysis for optimal k
- Identifies spam subtypes via TF-IDF
- Saves visualizations to `outputs/plots/`

#### 7. **CLI Prediction** (`src/models/predict.py`)
```bash
# Single prediction
python src/models/predict.py "Your message here"

# Interactive mode
python src/models/predict.py

# Batch from file
python src/models/predict.py --file messages.txt
```

---

## 🎯 Features

### ML Pipeline
- **Multiple Algorithms**: Naive Bayes, Logistic Regression, SVM
- **Advanced Features**: 22 engineered features (text stats, special chars, etc.)
- **Class Balancing**: SMOTE for imbalanced datasets
- **Hyperparameter Tuning**: Grid search with cross-validation
- **Clustering**: K-means to identify spam subtypes
- **Comprehensive Evaluation**: Multiple metrics and visualizations
- **Reproducible**: Fixed random seeds

### REST API
- **Multi-Model Ensemble**: Combines predictions from 3 models
- **Batch Processing**: Up to 100 messages per request
- **Clustering Integration**: Identifies spam subtypes in real-time
- **Database**: SQLite for prediction history and statistics
- **Type-Safe**: Pydantic models for request/response validation
- **Well-Documented**: OpenAPI/Swagger auto-generated docs
- **Production-Ready**: Proper error handling, logging, CORS

---

## 📊 Performance

Typical results on test set:
- **Accuracy**: >95%
- **F1-Score**: >0.90 for both classes
- **ROC-AUC**: >0.95
- **Best Model**: Usually Multinomial Naive Bayes

---

## ⚙️ Configuration

Edit `src/utils/config.yaml`:

```yaml
preprocessing:
  test_size: 0.15
  val_size: 0.15

models:
  multinomial_nb:
    alpha: [0.1, 0.5, 1.0]
  logistic_regression:
    C: [0.5, 1, 2, 4]
  linear_svc:
    C: [0.5, 1, 2, 4]

clustering:
  k_values: [5, 8, 12]
```

---

## 🗄️ Database

The API uses SQLite for development. The database stores:
- Prediction history
- Statistics and metrics
- User feedback (optional)

**Reset database:**
```bash
rm spam_detection.db
python -c "from api.database import init_database; init_database()"
```

**Seed test data:**
```bash
python api/seed_database.py
```

---

## 🔒 Security Considerations

**For Production:**
- Configure `CORS_ORIGINS` to specific domains only
- Implement authentication (API keys or JWT)
- Add rate limiting middleware
- Use PostgreSQL/MySQL instead of SQLite
- Enable HTTPS
- Add request logging to external service

---

## 🚀 Production Deployment

```bash
gunicorn api.main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --access-logfile - \
  --error-logfile -
```

**Required changes for production:**
1. Set `RELOAD=false` in `.env`
2. Configure database (PostgreSQL recommended)
3. Set specific CORS origins
4. Add authentication
5. Add rate limiting
6. Enable HTTPS

---

## 🔍 Troubleshooting

### Common Issues

1. **Import Errors**: Run from project root with activated venv
2. **Missing Dependencies**: `pip install -r src/requirements.txt api/requirements.txt`
3. **Data Download Issues**: Pipeline uses existing data if available
4. **CORS Errors**: Update `CORS_ORIGINS` in `.env`
5. **Database Errors**: Ensure database is initialized

### Logs

- All operations logged to `spam_detection.log`
- API request logging with processing time in headers

---

## 📈 Output Files

After running the pipeline:

**Data:**
- `outputs/raw_data.csv` - Combined raw dataset
- `outputs/{train,val,test}.csv` - Processed splits

**Models:**
- `outputs/models/best_model.json` - Model metadata
- `outputs/models/*.joblib` - Trained models

**Visualizations:**
- `outputs/eda/` - Exploratory analysis plots
- `outputs/reports/` - Evaluation plots
- `outputs/plots/` - Clustering visualizations

**Reports:**
- `outputs/reports/evaluation_results.csv` - Performance metrics
- `outputs/final_summary.txt` - Complete summary

---

## 📝 License

[Add your license here]

## 👥 Contributors

[Add contributors here]
