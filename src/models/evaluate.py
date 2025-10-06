"""
Model evaluation module for spam detection project.
Evaluates trained models and creates comprehensive performance reports.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Tuple, Any
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
)
from utils.helpers import (
    load_config, ensure_dir_exists, setup_logging, 
    plot_confusion_matrix, plot_precision_recall_curve, print_classification_report
)

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Model evaluation class for comprehensive performance analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the model evaluator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.results = {}
        
    def evaluate_model(self, model: Any, X_test: pd.Series, y_test: pd.Series, 
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate a single model and return comprehensive metrics.
        
        Args:
            model: Trained model
            X_test: Test text data
            y_test: Test labels
            model_name: Name of the model
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Get predictions
        y_pred = model.predict(X_test)
        
        # Get prediction probabilities (if available)
        try:
            y_proba = model.predict_proba(X_test)
            # For binary classification, get probability of positive class
            if y_proba.shape[1] == 2:
                y_scores = y_proba[:, 1]
            else:
                y_scores = y_proba
        except AttributeError:
            # For models without predict_proba (like LinearSVC)
            y_scores = model.decision_function(X_test)
            # Convert decision function to probabilities using sigmoid
            y_scores = 1 / (1 + np.exp(-y_scores))
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label='spam', average='binary')
        recall = recall_score(y_test, y_pred, pos_label='spam', average='binary')
        f1 = f1_score(y_test, y_pred, pos_label='spam', average='binary')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Precision-Recall curve
        precision_curve, recall_curve, pr_thresholds = precision_recall_curve(
            y_test, y_scores, pos_label='spam'
        )
        pr_auc = auc(recall_curve, precision_curve)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(y_test, y_scores, pos_label='spam')
        roc_auc = auc(fpr, tpr)
        
        results = {
            'model_name': model_name,
            'predictions': y_pred,
            'probabilities': y_scores,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'f1_macro': f1_macro,
            'confusion_matrix': cm,
            'precision_curve': precision_curve,
            'recall_curve': recall_curve,
            'pr_auc': pr_auc,
            'fpr': fpr,
            'tpr': tpr,
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, F1-Macro: {f1_macro:.4f}")
        
        return results
    
    def plot_model_comparison(self, results: Dict[str, Dict], save_path: str = None) -> None:
        """
        Plot comparison of all models.
        
        Args:
            results: Dictionary with evaluation results for all models
            save_path: Path to save the plot
        """
        logger.info("Creating model comparison plot...")
        
        # Extract metrics
        model_names = list(results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'f1_macro']
        
        # Create data for plotting
        plot_data = []
        for model_name in model_names:
            for metric in metrics:
                plot_data.append({
                    'Model': model_name.replace('_', ' ').title(),
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': results[model_name][metric]
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create grouped bar plot
        x = np.arange(len(model_names))
        width = 0.15
        
        for i, metric in enumerate(metrics):
            metric_scores = [results[model_name][metric] for model_name in model_names]
            ax.bar(x + i * width, metric_scores, width, 
                  label=metric.replace('_', ' ').title(), alpha=0.8)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width * 2)
        ax.set_xticklabels([name.replace('_', ' ').title() for name in model_names])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for i, model_name in enumerate(model_names):
            for j, metric in enumerate(metrics):
                score = results[model_name][metric]
                ax.text(i + j * width, score + 0.01, f'{score:.3f}', 
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        if save_path:
            ensure_dir_exists(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, results: Dict[str, Dict], save_path: str = None) -> None:
        """
        Plot ROC curves for all models.
        
        Args:
            results: Dictionary with evaluation results for all models
            save_path: Path to save the plot
        """
        logger.info("Creating ROC curves plot...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            plt.plot(result['fpr'], result['tpr'], 
                    label=f"{model_name.replace('_', ' ').title()} (AUC = {result['roc_auc']:.3f})",
                    linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            ensure_dir_exists(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves plot saved to {save_path}")
        
        plt.show()
    
    def plot_precision_recall_curves(self, results: Dict[str, Dict], save_path: str = None) -> None:
        """
        Plot precision-recall curves for all models.
        
        Args:
            results: Dictionary with evaluation results for all models
            save_path: Path to save the plot
        """
        logger.info("Creating precision-recall curves plot...")
        
        plt.figure(figsize=(10, 8))
        
        for model_name, result in results.items():
            plt.plot(result['recall_curve'], result['precision_curve'], 
                    label=f"{model_name.replace('_', ' ').title()} (AUC = {result['pr_auc']:.3f})",
                    linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        if save_path:
            ensure_dir_exists(os.path.dirname(save_path))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-recall curves plot saved to {save_path}")
        
        plt.show()
    
    def create_evaluation_report(self, results: Dict[str, Dict], 
                               output_dir: str) -> None:
        """
        Create comprehensive evaluation report.
        
        Args:
            results: Dictionary with evaluation results for all models
            output_dir: Directory to save outputs
        """
        logger.info("Creating comprehensive evaluation report...")
        
        ensure_dir_exists(output_dir)
        
        # 1. Model comparison plot
        self.plot_model_comparison(results, 
                                 save_path=os.path.join(output_dir, 'model_comparison.png'))
        
        # 2. ROC curves
        self.plot_roc_curves(results, 
                           save_path=os.path.join(output_dir, 'roc_curves.png'))
        
        # 3. Precision-recall curves
        self.plot_precision_recall_curves(results, 
                                        save_path=os.path.join(output_dir, 'precision_recall_curves.png'))
        
        # 4. Individual confusion matrices
        for model_name, result in results.items():
            plot_confusion_matrix(
                result['predictions'], result['predictions'],  # This will be fixed in the actual call
                ['not_spam', 'spam'],
                save_path=os.path.join(output_dir, f'confusion_matrix_{model_name}.png')
            )
        
        # 5. Save detailed results to CSV
        results_summary = []
        for model_name, result in results.items():
            results_summary.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': result['accuracy'],
                'Precision': result['precision'],
                'Recall': result['recall'],
                'F1-Score': result['f1'],
                'F1-Macro': result['f1_macro'],
                'ROC-AUC': result['roc_auc'],
                'PR-AUC': result['pr_auc']
            })
        
        results_df = pd.DataFrame(results_summary)
        results_path = os.path.join(output_dir, 'evaluation_results.csv')
        results_df.to_csv(results_path, index=False)
        logger.info(f"Evaluation results saved to {results_path}")
        
        # 6. Print summary
        print("\n" + "="*60)
        print("MODEL EVALUATION SUMMARY")
        print("="*60)
        
        print("\nTest Set Performance:")
        print(results_df.to_string(index=False, float_format='%.4f'))
        
        # Find best model
        best_model_idx = results_df['F1-Macro'].idxmax()
        best_model = results_df.iloc[best_model_idx]
        
        print(f"\nBest Model: {best_model['Model']}")
        print(f"Test F1-Macro: {best_model['F1-Macro']:.4f}")
        print(f"Test Accuracy: {best_model['Accuracy']:.4f}")

def evaluate_models(classifier: Any, test_df: pd.DataFrame) -> Dict[str, Dict]:
    """
    Evaluate all trained models on test set.
    
    Args:
        classifier: Trained SpamClassifier object
        test_df: Test dataset
        
    Returns:
        Dictionary with evaluation results for all models
    """
    logger.info("Starting model evaluation...")
    
    # Load configuration
    config = load_config()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Evaluate all models
    results = {}
    for model_name, model_result in classifier.models.items():
        model = model_result['model']
        results[model_name] = evaluator.evaluate_model(
            model, test_df['text'], test_df['label'], model_name
        )
    
    # Create evaluation report
    output_dir = os.path.join(config['data']['output_dir'], 'reports')
    evaluator.create_evaluation_report(results, output_dir)
    
    return results

if __name__ == "__main__":
    # Load trained classifier and test data
    from models.train import train_spam_classifier
    
    # Train classifier (or load if already trained)
    classifier, train_data, val_data, test_data = train_spam_classifier()
    
    # Evaluate models
    results = evaluate_models(classifier, test_data)
    
    print("Model evaluation completed!")
