"""
Main pipeline script for spam detection project.
Runs the complete machine learning pipeline from data collection to model deployment.
"""

import os
import sys
import logging
from typing import Dict, Any
from utils.helpers import load_config, setup_logging, log_version_info, ensure_dir_exists

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def run_data_pipeline() -> None:
    """
    Run data collection and preprocessing pipeline.
    """
    logger.info("="*60)
    logger.info("STEP 1: DATA COLLECTION AND PREPROCESSING")
    logger.info("="*60)
    
    # Import and run data collection
    from data.collect import collect_data
    logger.info("Running data collection...")
    raw_data = collect_data()
    
    # Import and run preprocessing
    from data.preprocess import preprocess_data
    logger.info("Running data preprocessing...")
    train_data, val_data, test_data = preprocess_data(df=raw_data)
    
    logger.info("Data pipeline completed successfully!")
    return train_data, val_data, test_data

def run_eda_pipeline(train_data) -> None:
    """
    Run exploratory data analysis pipeline.
    """
    logger.info("="*60)
    logger.info("STEP 2: EXPLORATORY DATA ANALYSIS")
    logger.info("="*60)
    
    # Import and run EDA
    from data.eda import run_eda, plot_spam_trigger_words, plot_advanced_message_characteristics, plot_tfidf_feature_analysis
    logger.info("Running exploratory data analysis...")
    run_eda(df=train_data)
    
    # Additional advanced EDA
    logger.info("Running advanced EDA analysis...")
    plot_spam_trigger_words(train_data)
    plot_advanced_message_characteristics(train_data)
    plot_tfidf_feature_analysis(train_data)
    
    logger.info("EDA pipeline completed successfully!")

def run_training_pipeline(train_data, val_data, test_data) -> Any:
    """
    Run model training pipeline.
    """
    logger.info("="*60)
    logger.info("STEP 3: MODEL TRAINING")
    logger.info("="*60)
    
    # Import and run training
    from models.train import train_spam_classifier
    logger.info("Running model training...")
    classifier, _, _, _ = train_spam_classifier()
    
    logger.info("Training pipeline completed successfully!")
    return classifier

def run_evaluation_pipeline(classifier, test_data) -> Dict[str, Any]:
    """
    Run model evaluation pipeline.
    """
    logger.info("="*60)
    logger.info("STEP 4: MODEL EVALUATION")
    logger.info("="*60)
    
    # Import and run evaluation
    from models.evaluate import evaluate_models
    logger.info("Running model evaluation...")
    results = evaluate_models(classifier, test_data)
    
    logger.info("Evaluation pipeline completed successfully!")
    return results


def run_clustering_pipeline(train_data, val_data, test_data) -> Any:
    """
    Run clustering analysis pipeline.
    """
    logger.info("="*60)
    logger.info("STEP 6: CLUSTERING ANALYSIS")
    logger.info("="*60)
    
    # Import and run clustering
    from models.cluster import run_clustering
    logger.info("Running clustering analysis...")
    
    # Combine all data for clustering
    import pandas as pd
    all_data = pd.concat([train_data, val_data, test_data], ignore_index=True)
    clusterer = run_clustering(all_data)
    
    logger.info("Clustering pipeline completed successfully!")
    return clusterer

def test_prediction_tool() -> None:
    """
    Test the prediction CLI tool.
    """
    logger.info("="*60)
    logger.info("STEP 6: TESTING PREDICTION TOOL")
    logger.info("="*60)
    
    # Test messages
    test_messages = [
        "Congratulations! You have won $1000 cash prize!",
        "Hey, how are you doing today?",
        "URGENT! Your account has been compromised. Click here to verify.",
        "Thanks for the great presentation today.",
        "Free entry in 2 a wkly comp to win FA Cup final tkts."
    ]
    
    logger.info("Testing prediction tool with sample messages...")
    
    try:
        from models.predict import load_trained_model, predict_message, format_prediction
        
        # Load model
        model = load_trained_model()
        
        # Test predictions
        for message in test_messages:
            prediction, confidence = predict_message(model, message)
            result = format_prediction(prediction, confidence)
            print(f"Message: {message}")
            print(f"Result: {result}")
            print("-" * 50)
        
        logger.info("Prediction tool test completed successfully!")
        
    except Exception as e:
        logger.error(f"Error testing prediction tool: {e}")

def create_final_summary(classifier, results, clusterer) -> None:
    """
    Create final summary report.
    """
    logger.info("="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    
    # Load configuration
    config = load_config()
    
    # Create summary
    summary = f"""
SPAM DETECTION PROJECT - FINAL SUMMARY
=====================================

PROJECT OVERVIEW:
- Complete machine learning pipeline for spam detection
- Multiple classifier comparison with grid search optimization
- K-Means clustering analysis for spam subtype identification
- Advanced feature engineering with 22 engineered features
- SMOTE class balancing for imbalanced datasets
- Command-line prediction tool

BEST MODEL RESULTS:
- Best Classifier: {classifier.best_model_name.replace('_', ' ').title()}
- Validation Macro-F1: {classifier.best_score:.4f}
- Best Parameters: {classifier.models[classifier.best_model_name]['best_params']}

TEST SET PERFORMANCE:
"""
    
    # Add test performance for best model
    best_model_name = classifier.best_model_name
    if best_model_name in results:
        test_result = results[best_model_name]
        summary += f"- Test Accuracy: {test_result['accuracy']:.4f}\n"
        summary += f"- Test Precision: {test_result['precision']:.4f}\n"
        summary += f"- Test Recall: {test_result['recall']:.4f}\n"
        summary += f"- Test F1-Score: {test_result['f1']:.4f}\n"
        summary += f"- Test F1-Macro: {test_result['f1_macro']:.4f}\n"
        summary += f"- Test ROC-AUC: {test_result['roc_auc']:.4f}\n"
        summary += f"- Test PR-AUC: {test_result['pr_auc']:.4f}\n"
    
    summary += f"""
CLUSTERING RESULTS:
- Best k: {clusterer.best_k}
- Best Silhouette Score: {clusterer.best_silhouette_score:.4f}
- Spam subtypes identified through TF-IDF term analysis

OUTPUTS GENERATED:
- EDA plots and analysis in outputs/eda/
- Model evaluation plots in outputs/plots/
- Clustering analysis in outputs/plots/
- Performance reports in outputs/reports/
- Trained model saved as outputs/models/spam_pipeline.joblib

USAGE:
- Predict single message: python src/models/predict.py "message text"
- Interactive mode: python src/models/predict.py

REPRODUCIBILITY:
- Random seed fixed to {config['random_state']}
- All parameters stored in config.yaml
- Complete pipeline automation
"""
    
    print(summary)
    
    # Save summary to file
    output_dir = config['data']['output_dir']
    ensure_dir_exists(output_dir)
    summary_path = os.path.join(output_dir, 'final_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    logger.info(f"Final summary saved to {summary_path}")

def main():
    """
    Main function to run the complete pipeline.
    """
    logger.info("Starting Spam Detection Machine Learning Pipeline")
    logger.info("="*60)
    
    # Log version information for reproducibility
    log_version_info()
    
    try:
        # Step 1: Data pipeline
        train_data, val_data, test_data = run_data_pipeline()
        
        # Step 2: EDA
        run_eda_pipeline(train_data)
        
        # Step 3: Training
        classifier = run_training_pipeline(train_data, val_data, test_data)
        
        # Step 4: Evaluation
        results = run_evaluation_pipeline(classifier, test_data)
        
        # Step 5: Clustering
        clusterer = run_clustering_pipeline(train_data, val_data, test_data)
        
        # Step 6: Test prediction tool
        test_prediction_tool()
        
        # Final summary
        create_final_summary(classifier, results, clusterer)
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        print("\n🎉 Spam Detection Pipeline Completed Successfully!")
        print(f"📊 Best Model: {classifier.best_model_name.replace('_', ' ').title()}")
        print(f"📈 Validation F1-Macro: {classifier.best_score:.4f}")
        print(f"🔍 Best Clustering k: {clusterer.best_k} (Silhouette: {clusterer.best_silhouette_score:.4f})")
        print(f"💾 Model saved to: outputs/models/spam_pipeline.joblib")
        print(f"📁 All outputs saved to: outputs/")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
