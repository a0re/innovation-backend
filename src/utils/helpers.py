"""
Helper functions for the spam detection project.
"""

import os
import yaml
import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if config_path is None:
        # Try different possible paths
        possible_paths = [
            "src/utils/config.yaml",
            "utils/config.yaml", 
            os.path.join(os.path.dirname(__file__), "config.yaml")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
        else:
            raise FileNotFoundError("Could not find config.yaml in any of the expected locations")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Resolve output_dir relative to project root
    if 'data' in config and 'output_dir' in config['data']:
        output_dir = config['data']['output_dir']
        # If it's a relative path, make it relative to the project root
        if not os.path.isabs(output_dir):
            # Find project root (where .git or outputs directory exists)
            current = os.path.abspath(os.getcwd())
            while current != '/':
                if os.path.exists(os.path.join(current, '.git')) or os.path.exists(os.path.join(current, 'outputs')):
                    config['data']['output_dir'] = os.path.join(current, output_dir)
                    break
                current = os.path.dirname(current)

    return config

def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('spam_detection.log'),
            logging.StreamHandler()
        ]
    )

def ensure_dir_exists(directory: str) -> None:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        directory: Directory path to check/create
    """
    os.makedirs(directory, exist_ok=True)

def save_model(model: Any, filepath: str) -> None:
    """
    Save model using joblib.
    
    Args:
        model: Trained model to save
        filepath: Path where to save the model
    """
    ensure_dir_exists(os.path.dirname(filepath))
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath: str) -> Any:
    """
    Load model using joblib.
    
    Args:
        filepath: Path to the saved model
        
    Returns:
        Loaded model
    """
    return joblib.load(filepath)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: list, save_path: str = None) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, 
                               save_path: str = None) -> None:
    """
    Plot precision-recall curve.
    
    Args:
        y_true: True binary labels
        y_scores: Target scores (probability estimates)
        save_path: Path to save the plot
    """
    from sklearn.metrics import precision_recall_curve, auc
    
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    auc_score = auc(recall, precision)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, 
             label=f'PR curve (AUC = {auc_score:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-recall curve saved to {save_path}")
    
    plt.show()

def print_classification_report(y_true: np.ndarray, y_pred: np.ndarray, 
                               class_names: list, save_path: str = None) -> None:
    """
    Print and save classification report.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the report
    """
    report = classification_report(y_true, y_pred, target_names=class_names)
    print("\nClassification Report:")
    print(report)
    
    if save_path:
        ensure_dir_exists(os.path.dirname(save_path))
        with open(save_path, 'w') as f:
            f.write(report)
        print(f"Classification report saved to {save_path}")

def get_version_info() -> Dict[str, str]:
    """
    Get version information for reproducibility.
    
    Returns:
        Dictionary with version information
    """
    import sys
    import pandas as pd
    import numpy as np
    import sklearn
    
    return {
        'python': sys.version,
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scikit-learn': sklearn.__version__
    }

def log_version_info() -> None:
    """
    Log version information for reproducibility.
    """
    versions = get_version_info()
    print("\nVersion Information for Reproducibility:")
    print("=" * 50)
    for package, version in versions.items():
        print(f"{package}: {version}")
    print("=" * 50)
