#!/usr/bin/env python3
"""
Model Checker Script
Shows which model is currently being used for predictions.
"""

import sys
import os
import joblib

def check_model():
    """Check which model is currently in use."""
    
    print("=" * 60)
    print("SPAM DETECTION MODEL CHECKER")
    print("=" * 60)
    
    # Check if model file exists
    model_path = "outputs/models/spam_pipeline.joblib"
    
    if not os.path.exists(model_path):
        print("❌ Model file not found!")
        print(f"   Expected location: {model_path}")
        print("   Please run the training pipeline first:")
        print("   python src/run_pipeline.py")
        return
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        print("✅ Model loaded successfully!")
        print(f"📁 Model file: {model_path}")
        print()
        
        # Get the classifier
        classifier = model.named_steps['classifier']
        
        print("🎯 MODEL DETAILS:")
        print(f"   Algorithm: {type(classifier).__name__}")
        print(f"   Parameters: {classifier.get_params()}")
        print()
        
        # Get the vectorizer details
        vectorizer = model.named_steps['tfidf']
        print("🔤 VECTORIZER DETAILS:")
        print(f"   Type: {type(vectorizer).__name__}")
        print(f"   N-gram range: {vectorizer.ngram_range}")
        print(f"   Max features: {vectorizer.max_features}")
        print(f"   Min document frequency: {vectorizer.min_df}")
        print()
        
        # Test the model
        print("🧪 MODEL TEST:")
        test_messages = [
            "Congratulations! You have won $1000!",
            "Hey, how are you doing today?",
            "URGENT! Your account has been compromised."
        ]
        
        for msg in test_messages:
            prediction = model.predict([msg])[0]
            proba = model.predict_proba([msg])[0]
            confidence = max(proba)
            
            emoji = "🚨" if prediction == "spam" else "✅"
            print(f"   {emoji} \"{msg[:30]}...\" → {prediction} ({confidence:.1%})")
        
        print()
        print("📊 SUMMARY:")
        print(f"   The prediction tool uses: {type(classifier).__name__}")
        print(f"   With parameters: {classifier.get_params()}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")

if __name__ == "__main__":
    check_model()
