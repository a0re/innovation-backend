"""
Prediction module for spam detection project.
Command-line interface for making predictions on new messages.
"""

import sys
import os
import joblib
import logging
from typing import Union

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import load_config, setup_logging, load_model

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

def load_trained_model(model_path: str = None) -> object:
    """
    Load the trained spam detection model.
    
    Args:
        model_path: Path to the trained model file
        
    Returns:
        Loaded model
    """
    if model_path is None:
        # Load configuration to get default model path
        config = load_config()
        model_path = os.path.join(config['data']['output_dir'], 'models', 'spam_pipeline.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Please train the model first.")
    
    logger.info(f"Loading model from {model_path}")
    model = load_model(model_path)
    
    return model

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

def interactive_mode(model: object) -> None:
    """
    Run interactive prediction mode.
    
    Args:
        model: Trained model
    """
    print("\n" + "="*50)
    print("SPAM DETECTION - INTERACTIVE MODE")
    print("="*50)
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
    try:
        # Load the trained model
        model = load_trained_model()
        
        # Check if message is provided as command line argument
        if len(sys.argv) > 1:
            # Get message from command line arguments
            message = ' '.join(sys.argv[1:])
            
            # Make prediction
            prediction, confidence = predict_message(model, message)
            result = format_prediction(prediction, confidence)
            
            print(f"Message: {message}")
            print(f"Prediction: {result}")
            
        else:
            # Run interactive mode
            interactive_mode(model)
            
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
