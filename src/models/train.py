"""
Model training module for spam detection project.
Implements TF-IDF vectorization and trains multiple classification models.
"""

import os
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from utils.helpers import load_config, ensure_dir_exists, setup_logging, save_model

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def clear_old_models(models_dir: str) -> None:
    """
    Clear all existing model files from the models directory.
    
    Args:
        models_dir: Path to the models directory
    """
    if not os.path.exists(models_dir):
        return
    
    # Get all .joblib files and best_model.json in the models directory
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib') or f == 'best_model.json']
    
    if model_files:
        logger.info(f"Clearing {len(model_files)} old model files...")
        for model_file in model_files:
            model_path = os.path.join(models_dir, model_file)
            try:
                os.remove(model_path)
                logger.info(f"Removed old model: {model_file}")
            except OSError as e:
                logger.warning(f"Could not remove {model_file}: {e}")
    else:
        logger.info("No old model files found to clear")

def save_best_model_metadata(models_dir: str, best_model_name: str, best_score: float) -> None:
    """
    Save metadata about the best model instead of duplicating the model file.
    
    Args:
        models_dir: Path to the models directory
        best_model_name: Name of the best performing model
        best_score: Validation score of the best model
    """
    import json
    from datetime import datetime
    
    metadata = {
        "best_model_name": best_model_name,
        "best_score": best_score,
        "timestamp": datetime.now().isoformat(),
        "model_file": f"{best_model_name}.joblib"
    }
    
    metadata_path = os.path.join(models_dir, 'best_model.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved best model metadata: {best_model_name} (score: {best_score:.4f})")
    logger.info(f"Best model reference saved to: {metadata_path}")

class SpamClassifier:
    """
    Spam classification pipeline with TF-IDF vectorization and multiple models.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the spam classifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.best_score = 0
        
    def create_vectorizers(self) -> Dict[str, TfidfVectorizer]:
        """
        Create TF-IDF vectorizers for word and character n-grams.
        
        Returns:
            Dictionary of vectorizers
        """
        logger.info("Creating TF-IDF vectorizers...")
        
        vectorizers = {}
        
        # Word n-gram vectorizer
        vectorizers['word_tfidf'] = TfidfVectorizer(
            ngram_range=tuple(self.config['preprocessing']['ngram_range_word']),
            min_df=self.config['preprocessing']['min_df'],
            max_features=self.config['preprocessing']['max_features'],
            norm='l2',
            lowercase=True,
            stop_words='english'
        )
        
        # Character n-gram vectorizer
        vectorizers['char_tfidf'] = TfidfVectorizer(
            ngram_range=tuple(self.config['preprocessing']['ngram_range_char']),
            min_df=self.config['preprocessing']['min_df'],
            max_features=self.config['preprocessing']['max_features'],
            norm='l2',
            lowercase=True,
            analyzer='char'
        )
        
        return vectorizers
    
    def create_models(self) -> Dict[str, Pipeline]:
        """
        Create model pipelines with TF-IDF vectorization.
        
        NOTE: Currently only uses TF-IDF features from text column.
        The 27 engineered features created in preprocessing are NOT used.
        To use them, this pipeline needs to be updated with ColumnTransformer.
        
        Returns:
            Dictionary of model pipelines
        """
        logger.info("Creating model pipelines...")
        logger.warning("⚠️  WARNING: Only using TF-IDF features. 27 engineered features are being ignored!")
        
        vectorizers = self.create_vectorizers()
        models = {}
        
        # Multinomial Naive Bayes (TF-IDF ONLY - NOT USING ENGINEERED FEATURES)
        models['multinomial_nb'] = Pipeline([
            ('tfidf', vectorizers['word_tfidf']),
            ('classifier', MultinomialNB())
        ])
        
        # Logistic Regression (TF-IDF ONLY - NOT USING ENGINEERED FEATURES)
        models['logistic_regression'] = Pipeline([
            ('tfidf', vectorizers['word_tfidf']),
            ('classifier', LogisticRegression(
                solver=self.config['models']['logistic_regression']['solver'],
                class_weight=self.config['models']['logistic_regression']['class_weight'],
                random_state=self.config['random_state']
            ))
        ])
        
        # Linear SVM (TF-IDF ONLY - NOT USING ENGINEERED FEATURES)
        models['linear_svc'] = Pipeline([
            ('tfidf', vectorizers['word_tfidf']),
            ('classifier', LinearSVC(
                class_weight=self.config['models']['linear_svc']['class_weight'],
                random_state=self.config['random_state']
            ))
        ])
        
        return models
    
    def create_param_grids(self) -> Dict[str, Dict]:
        """
        Create parameter grids for grid search.
        
        Returns:
            Dictionary of parameter grids
        """
        param_grids = {
            'multinomial_nb': {
                'classifier__alpha': self.config['models']['multinomial_nb']['alpha']
            },
            'logistic_regression': {
                'classifier__C': self.config['models']['logistic_regression']['C']
            },
            'linear_svc': {
                'classifier__C': self.config['models']['linear_svc']['C']
            }
        }
        
        return param_grids
    
    def train_models(self, X_train: pd.Series, y_train: pd.Series, 
                    X_val: pd.Series, y_val: pd.Series) -> Dict[str, Any]:
        """
        Train all models using grid search.
        
        Args:
            X_train: Training text data
            y_train: Training labels
            X_val: Validation text data
            y_val: Validation labels
            
        Returns:
            Dictionary with training results
        """
        logger.info("Starting model training...")
        
        models = self.create_models()
        param_grids = self.create_param_grids()
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Training {model_name}...")
            
            # Create grid search
            grid_search = GridSearchCV(
                model,
                param_grids[model_name],
                cv=self.config['evaluation']['cv_folds'],
                scoring=self.config['evaluation']['scoring'],
                n_jobs=self.config['evaluation']['n_jobs'],
                verbose=1
            )
            
            # Fit the model
            grid_search.fit(X_train, y_train)
            
            # Get validation score
            val_score = grid_search.score(X_val, y_val)
            
            # Store results
            results[model_name] = {
                'model': grid_search.best_estimator_,
                'best_params': grid_search.best_params_,
                'cv_score': grid_search.best_score_,
                'val_score': val_score,
                'grid_search': grid_search
            }
            
            # Update best model
            if val_score > self.best_score:
                self.best_score = val_score
                self.best_model = grid_search.best_estimator_
                self.best_model_name = model_name
            
            logger.info(f"{model_name} - CV Score: {grid_search.best_score_:.4f}, "
                       f"Val Score: {val_score:.4f}")
            logger.info(f"Best params: {grid_search.best_params_}")
        
        self.models = results
        return results
    
    def print_results_summary(self) -> None:
        """
        Print summary of training results.
        """
        print("\n" + "="*60)
        print("MODEL TRAINING RESULTS SUMMARY")
        print("="*60)
        
        # Create results table
        results_data = []
        for model_name, result in self.models.items():
            results_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'CV Score': f"{result['cv_score']:.4f}",
                'Val Score': f"{result['val_score']:.4f}",
                'Best Params': str(result['best_params'])
            })
        
        results_df = pd.DataFrame(results_data)
        print("\nValidation Macro-F1 Scores:")
        print(results_df.to_string(index=False))
        
        print(f"\nBest Model: {self.best_model_name.replace('_', ' ').title()}")
        print(f"Best Validation Score: {self.best_score:.4f}")
        print(f"Best Parameters: {self.models[self.best_model_name]['best_params']}")

def load_data(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    logger.info("Loading preprocessed data...")
    
    output_dir = config['data']['output_dir']
    
    train_path = os.path.join(output_dir, 'train.csv')
    val_path = os.path.join(output_dir, 'val.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    
    if not all(os.path.exists(path) for path in [train_path, val_path, test_path]):
        logger.warning("Preprocessed data not found. Running preprocessing...")
        from data.preprocess import preprocess_data
        return preprocess_data()
    
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)
    
    logger.info(f"Loaded data - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    return train_df, val_df, test_df

def train_spam_classifier() -> Tuple[SpamClassifier, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Main function to train spam classifier.
    
    Returns:
        Tuple of (trained_classifier, train_df, val_df, test_df)
    """
    logger.info("Starting spam classifier training...")
    
    # Load configuration
    config = load_config()
    
    # Load data
    train_df, val_df, test_df = load_data(config)
    
    # Initialize classifier
    classifier = SpamClassifier(config)
    
    # Train models
    results = classifier.train_models(
        train_df['text'], train_df['label'],
        val_df['text'], val_df['label']
    )
    
    # Print results summary
    classifier.print_results_summary()
    
    # Clear old models and save new ones
    models_dir = os.path.join(config['data']['output_dir'], 'models')
    clear_old_models(models_dir)
    ensure_dir_exists(models_dir)
    
    # Save all individual models
    for model_name, model_result in classifier.models.items():
        model_path = os.path.join(models_dir, f'{model_name}.joblib')
        save_model(model_result['model'], model_path)
        logger.info(f"Saved {model_name} model to {model_path}")
    
    # Save best model metadata (instead of duplicating the model)
    save_best_model_metadata(models_dir, classifier.best_model_name, classifier.best_score)
    
    logger.info("Training completed successfully!")
    
    return classifier, train_df, val_df, test_df

if __name__ == "__main__":
    # Run training
    classifier, train_data, val_data, test_data = train_spam_classifier()
    
    print(f"\nTraining completed!")
    print(f"Best model: {classifier.best_model_name}")
    print(f"Best validation score: {classifier.best_score:.4f}")
    print(f"Model saved to: outputs/models/spam_pipeline.joblib")
