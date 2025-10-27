"""
Prediction module for spam detection project.
Command-line interface for making predictions on new messages.
"""

import sys
import os
import joblib
import logging
import argparse
from typing import Union, List

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, setup_logging, load_model

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def get_available_models() -> List[str]:
    """
    Get list of available trained models.
    
    Returns:
        List of available model names
    """
    config = load_config()
    models_dir = os.path.join(config['data']['output_dir'], 'models')
    
    if not os.path.exists(models_dir):
        return []
    
    models = []
    for file in os.listdir(models_dir):
        if file.endswith('.joblib'):
            model_name = file.replace('.joblib', '')
            models.append(model_name)
    
    return sorted(models)

def get_best_model_info(models_dir: str) -> tuple:
    """
    Get information about the best model from metadata.
    
    Args:
        models_dir: Path to the models directory
        
    Returns:
        Tuple of (best_model_name, best_score, model_file)
    """
    import json
    
    metadata_path = os.path.join(models_dir, 'best_model.json')
    
    if not os.path.exists(metadata_path):
        return None, None, None
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return (
            metadata.get('best_model_name'),
            metadata.get('best_score'),
            metadata.get('model_file')
        )
    except (json.JSONDecodeError, KeyError) as e:
        logger.warning(f"Error reading best model metadata: {e}")
        return None, None, None

def load_trained_model(model_name: str = None) -> object:
    """
    Load the trained spam detection model.

    Args:
        model_name: Name of the model to load (e.g., 'multinomial_nb', 'logistic_regression', 'linear_svc')
                   If None, loads the best model from metadata

    Returns:
        Loaded model
    """
    config = load_config()
    models_dir = os.path.join(config['data']['output_dir'], 'models')

    if model_name is None:
        # Load best model from metadata
        best_model_name, best_score, model_file = get_best_model_info(models_dir)

        if best_model_name is None:
            available_models = get_available_models()
            if not available_models:
                raise FileNotFoundError(
                    f"No trained models found in {models_dir}.\n"
                    f"Please train the model first by running: python src/models/train.py"
                )
            else:
                raise FileNotFoundError(
                    f"No best model metadata found. Available models: {', '.join(available_models)}\n"
                    f"Please train the model first by running: python src/models/train.py"
                )

        model_path = os.path.join(models_dir, model_file)
        model_display_name = f"best model ({best_model_name}, score: {best_score:.4f})"
    else:
        # Load specific model
        model_path = os.path.join(models_dir, f'{model_name}.joblib')
        model_display_name = model_name

    if not os.path.exists(model_path):
        available_models = get_available_models()
        if not available_models:
            raise FileNotFoundError(
                f"No trained models found in {models_dir}.\n"
                f"Please train the model first by running: python src/models/train.py"
            )
        else:
            raise FileNotFoundError(
                f"Model '{model_name}' not found at {model_path}.\n"
                f"Available models: {', '.join(available_models)}\n"
                f"Please train the model first by running: python src/models/train.py"
            )

    logger.info(f"Loading {model_display_name} from {model_path}")
    model = load_model(model_path)

    return model

def load_all_models() -> dict:
    """
    Load all three trained models for ensemble predictions.

    Returns:
        Dictionary mapping model names to loaded models
    """
    config = load_config()
    models_dir = os.path.join(config['data']['output_dir'], 'models')

    model_names = ['multinomial_nb', 'logistic_regression', 'linear_svc']
    models = {}

    for model_name in model_names:
        model_path = os.path.join(models_dir, f'{model_name}.joblib')

        if os.path.exists(model_path):
            logger.info(f"Loading {model_name} from {model_path}")
            models[model_name] = load_model(model_path)
        else:
            logger.warning(f"Model '{model_name}' not found at {model_path}")

    if not models:
        raise FileNotFoundError(
            f"No trained models found in {models_dir}.\n"
            f"Please train the models first by running: python src/models/train.py"
        )

    return models

def preprocess_message(message: str) -> str:
    """
    Preprocess a single message for prediction.
    This should match the preprocessing used during training.
    
    Args:
        message: Raw message text
        
    Returns:
        Preprocessed message text
    """
    import re
    
    if not isinstance(message, str):
        return ""
    
    # Convert to lowercase
    message = message.lower()
    
    # Replace URLs with placeholder
    message = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', 
                     '<URL>', message)
    
    # Replace email addresses with placeholder
    message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                     '<EMAIL>', message)
    
    # Replace phone numbers with placeholder
    message = re.sub(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b', 
                     '<PHONE>', message)
    
    # Replace numbers with placeholder
    message = re.sub(r'\b\d+\b', '<NUM>', message)
    
    # Fix concatenated NUMBER patterns (common in preprocessed datasets)
    message = re.sub(r'NUMBER([a-zA-Z])', r'NUMBER \1', message, flags=re.IGNORECASE)
    message = re.sub(r'([a-zA-Z])NUMBER', r'\1 NUMBER', message, flags=re.IGNORECASE)
    message = re.sub(r'NUMBER\s*NUMBER', 'NUMBER', message, flags=re.IGNORECASE)
    
    # Remove extra whitespace
    message = re.sub(r'\s+', ' ', message)
    
    # Remove leading/trailing whitespace
    message = message.strip()
    
    return message

def get_dynamic_threshold(message: str) -> float:
    """
    Get dynamic threshold based on message characteristics.
    
    Args:
        message: Message text
        
    Returns:
        Dynamic threshold for spam classification
    """
    import re
    
    # Base threshold
    threshold = 0.5
    
    # Lower threshold (more likely to classify as spam) for suspicious patterns
    suspicious_patterns = [
        r'\b(?:verify|suspended|compromised|security|account|bank|paypal)\b',
        r'\b(?:prince|nigerian|inheritance|lottery|prize|million)\b',
        r'\b(?:urgent|immediately|click here|verify now)\b',
        r'\b(?:password|pin|ssn|bank account|credit card)\b',
        r'\$\d+|\£\d+|million|thousand|cash|money'
    ]
    
    for pattern in suspicious_patterns:
        if re.search(pattern, message, re.IGNORECASE):
            threshold -= 0.1  # Lower threshold for suspicious content
    
    # Ensure threshold is between 0.3 and 0.7
    return max(0.3, min(0.7, threshold))

def predict_message(model: object, message: str) -> tuple:
    """
    Predict whether a message is spam or not.

    Args:
        model: Trained model
        message: Message text to classify

    Returns:
        Tuple of (prediction, confidence_score)
    """
    # Preprocess the message
    processed_message = preprocess_message(message)

    if not processed_message:
        return "not_spam", 0.5

    # Get spam probability
    try:
        # For models with predict_proba
        probabilities = model.predict_proba([processed_message])[0]
        spam_prob = probabilities[1] if len(probabilities) > 1 else 0.5
    except AttributeError:
        # For models without predict_proba (like LinearSVC)
        decision_scores = model.decision_function([processed_message])
        # Convert decision function to probability using sigmoid
        spam_prob = 1 / (1 + abs(decision_scores[0]))

    # Use dynamic threshold
    threshold = get_dynamic_threshold(message)

    # Make prediction based on dynamic threshold
    if spam_prob >= threshold:
        prediction = "spam"
        confidence = spam_prob
    else:
        prediction = "not_spam"
        confidence = 1 - spam_prob

    return prediction, confidence

def load_clusterer() -> object:
    """
    Load the trained spam clusterer.

    Returns:
        Loaded clusterer object or None if not found
    """
    config = load_config()
    models_dir = os.path.join(config['data']['output_dir'], 'models')
    clusterer_path = os.path.join(models_dir, 'spam_clusterer.joblib')

    if not os.path.exists(clusterer_path):
        logger.warning(f"Clusterer not found at {clusterer_path}")
        return None

    logger.info(f"Loading clusterer from {clusterer_path}")
    clusterer = load_model(clusterer_path)

    return clusterer

def predict_cluster(clusterer: object, message: str) -> dict:
    """
    Predict which spam cluster a message belongs to.

    Args:
        clusterer: Trained SpamClusterer object
        message: Message text to classify

    Returns:
        Dictionary containing cluster prediction and top terms
    """
    if clusterer is None:
        return None

    # Preprocess the message
    processed_message = preprocess_message(message)

    if not processed_message or not clusterer.vectorizer:
        return None

    try:
        # Vectorize the message
        X = clusterer.vectorizer.transform([processed_message])

        # Get best k-means model
        if clusterer.best_k not in clusterer.cluster_results:
            return None

        kmeans = clusterer.cluster_results[clusterer.best_k]['kmeans']

        # Predict cluster
        cluster_id = int(kmeans.predict(X)[0])

        # Get distances to all cluster centers
        distances = kmeans.transform(X)[0]

        # Calculate confidence (inverse of distance to assigned cluster)
        assigned_distance = distances[cluster_id]
        # Normalize to 0-1 scale (closer = higher confidence)
        max_distance = max(distances)
        confidence = 1 - (assigned_distance / max_distance) if max_distance > 0 else 1.0

        # Get top terms for this cluster
        top_terms = clusterer.cluster_results[clusterer.best_k]['top_terms'].get(cluster_id, [])
        top_terms_list = [{'term': term, 'score': float(score)} for term, score in top_terms[:10]]

        return {
            'cluster_id': cluster_id,
            'confidence': confidence,
            'top_terms': top_terms_list,
            'total_clusters': clusterer.best_k
        }

    except Exception as e:
        logger.error(f"Error predicting cluster: {e}")
        return None

def predict_with_all_models(models: dict, message: str) -> dict:
    """
    Predict using all models and return individual predictions plus ensemble result.

    Args:
        models: Dictionary of model_name -> model object
        message: Message text to classify

    Returns:
        Dictionary containing predictions from each model and ensemble result
    """
    # Get predictions from each model
    model_predictions = {}

    for model_name, model in models.items():
        prediction, confidence = predict_message(model, message)
        model_predictions[model_name] = {
            'prediction': prediction,
            'confidence': confidence,
            'is_spam': prediction == 'spam'
        }

    # Calculate ensemble prediction (majority voting)
    spam_votes = sum(1 for pred in model_predictions.values() if pred['is_spam'])
    total_votes = len(model_predictions)

    # Ensemble decision
    ensemble_is_spam = spam_votes > (total_votes / 2)
    ensemble_prediction = 'spam' if ensemble_is_spam else 'not_spam'

    # Average confidence from models that agree with ensemble
    agreeing_confidences = [
        pred['confidence'] for pred in model_predictions.values()
        if pred['is_spam'] == ensemble_is_spam
    ]
    ensemble_confidence = sum(agreeing_confidences) / len(agreeing_confidences) if agreeing_confidences else 0.5

    return {
        'models': model_predictions,
        'ensemble': {
            'prediction': ensemble_prediction,
            'confidence': ensemble_confidence,
            'is_spam': ensemble_is_spam,
            'spam_votes': spam_votes,
            'total_votes': total_votes
        }
    }

def format_prediction(prediction: str, confidence: float) -> str:
    """
    Format prediction result for display.
    
    Args:
        prediction: Predicted class
        confidence: Confidence score
        
    Returns:
        Formatted prediction string
    """
    if prediction == 'spam':
        emoji = "🚨"
        status = "SPAM"
    else:
        emoji = "✅"
        status = "NOT SPAM"
    
    return f"{emoji} {status} (confidence: {confidence:.2%})"

def interactive_mode(model: object, model_name: str = "best model") -> None:
    """
    Run interactive prediction mode.
    
    Args:
        model: Trained model
        model_name: Name of the model being used
    """
    print("\n" + "="*50)
    print("SPAM DETECTION - INTERACTIVE MODE")
    print("="*50)
    print(f"Using model: {model_name}")
    print("Enter messages to classify (type 'quit' to exit)")
    print("-" * 50)
    
    while True:
        try:
            message = input("\nEnter message: ").strip()
            
            if message.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not message:
                print("Please enter a message.")
                continue
            
            # Make prediction
            prediction, confidence = predict_message(model, message)
            result = format_prediction(prediction, confidence)
            
            print(f"Result: {result}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

def main():
    """
    Main function for command-line interface.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Spam Detection CLI - Classify messages as spam or not spam",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/models/predict.py "Free money! Click here now!"
  python src/models/predict.py --model multinomial_nb "Verify your account"
  python src/models/predict.py --model logistic_regression
  python src/models/predict.py --list-models
        """
    )
    
    parser.add_argument(
        'message', 
        nargs='*', 
        help='Message to classify (if not provided, runs in interactive mode)'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        help='Model to use (multinomial_nb, logistic_regression, linear_svc). Default: best model'
    )
    
    parser.add_argument(
        '--list-models', '-l',
        action='store_true',
        help='List available models and exit'
    )
    
    args = parser.parse_args()
    
    try:
        # List models if requested
        if args.list_models:
            available_models = get_available_models()
            if available_models:
                # Get best model info
                config = load_config()
                models_dir = os.path.join(config['data']['output_dir'], 'models')
                best_model_name, best_score, _ = get_best_model_info(models_dir)
                
                print("Available models:")
                for model in available_models:
                    if model == best_model_name:
                        print(f"  {model} (best model - score: {best_score:.4f})")
                    else:
                        print(f"  {model}")
                
                if best_model_name:
                    print(f"\nDefault model: {best_model_name} (automatically selected)")
                else:
                    print("\nNo best model metadata found. Please retrain models.")
            else:
                print("No models found. Please train the model first:")
                print("  python src/models/train.py")
            return
        
        # Load the trained model
        model_name = args.model
        model = load_trained_model(model_name)
        
        # Determine display name for model
        if model_name is None:
            config = load_config()
            models_dir = os.path.join(config['data']['output_dir'], 'models')
            best_model_name, best_score, _ = get_best_model_info(models_dir)
            if best_model_name:
                display_name = f"best model ({best_model_name}, score: {best_score:.4f})"
            else:
                display_name = "best model"
        else:
            display_name = model_name
        
        # Check if message is provided as command line argument
        if args.message:
            # Get message from command line arguments
            message = ' '.join(args.message)
            
            # Make prediction
            prediction, confidence = predict_message(model, message)
            result = format_prediction(prediction, confidence)
            
            print(f"Model: {display_name}")
            print(f"Message: {message}")
            print(f"Prediction: {result}")
            
        else:
            # Run interactive mode
            interactive_mode(model, display_name)
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo train the model first, run:")
        print("  python src/models/train.py")
        sys.exit(1)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
