"""
Comprehensive Test Cases for Hybrid Fake News Detector

Test scenarios:
1. FAKE news with HIGH metadata (celebrity gossip)
2. REAL news with HIGH metadata (credible outlets)
3. FAKE news with LOW/ZERO metadata
4. REAL news with LOW/ZERO metadata
5. REAL news from verified sources
6. OBVIOUSLY FAKE news
7. OBVIOUSLY REAL news
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import joblib
import torch
import nltk
from transformers import RobertaTokenizer, RobertaModel

# Ensure NLTK data is downloaded
for resource in ['punkt', 'punkt_tab', 'stopwords']:
    try:
        nltk.data.find(f'tokenizers/{resource}')
    except LookupError:
        nltk.download(resource)

from preprocessing import clean_text_for_tfidf, clean_text_for_roberta
from tfidf_vectorizer import load_vectorizer
from feature_combiner import combine_features, extra_features
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# Load models with proper path handling
def load_models():
    """Load all required models with error handling"""
    try:
        model_path = os.path.join(PROJECT_ROOT, "fake_news_xgboost.pkl")
        if not os.path.exists(model_path):
            model_path = os.path.join(SCRIPT_DIR, "fake_news_xgboost.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found in {PROJECT_ROOT} or {SCRIPT_DIR}")
        clf = joblib.load(model_path)
        
        vectorizer_path = os.path.join(PROJECT_ROOT, "tfidf_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            vectorizer_path = os.path.join(SCRIPT_DIR, "tfidf_vectorizer.pkl")
        if not os.path.exists(vectorizer_path):
            raise FileNotFoundError(f"TF-IDF vectorizer not found in {PROJECT_ROOT} or {SCRIPT_DIR}")
        vectorizer = load_vectorizer(vectorizer_path)

        threshold_path = os.path.join(PROJECT_ROOT, "rumor_threshold.json")
        if not os.path.exists(threshold_path):
            threshold_path = os.path.join(SCRIPT_DIR, "rumor_threshold.json")
        rumor_threshold = 0.5
        best_iteration = None
        if os.path.exists(threshold_path):
            with open(threshold_path, "r", encoding="utf-8") as f:
                threshold_payload = json.load(f)
                rumor_threshold = float(threshold_payload.get("rumor_threshold", 0.5))
                best_iter_val = threshold_payload.get("best_iteration")
                if best_iter_val is not None:
                    best_iteration = int(best_iter_val)
        
        return clf, vectorizer, rumor_threshold, best_iteration
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        print(f"   Looking in directory: {SCRIPT_DIR}")
        sys.exit(1)

clf_hybrid, tfidf_vectorizer, RUMOR_THRESHOLD, BEST_ITERATION = load_models()

# Load RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")
roberta_model.to(device)
roberta_model.eval()


def get_roberta_embedding(text):
    """Get RoBERTa [CLS] embedding"""
    inputs = tokenizer(
        text,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = roberta_model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :]
    return embedding.cpu().numpy().flatten()


def predict_news(title, metadata=None):
    """Predict news label using model mapping: 0=TRUTH, 1=RUMOR."""
    
    clean_tfidf = clean_text_for_tfidf(title)
    clean_roberta = clean_text_for_roberta(title)
    
    tfidf_features = tfidf_vectorizer.transform([clean_tfidf]).toarray()
    roberta_features = np.array([get_roberta_embedding(clean_roberta)])
    content_df = pd.DataFrame({'content': [title]})
    engineered = extra_features(content_df)

    # Model is text-only; metadata argument is kept for compatibility and ignored.
    X = combine_features(tfidf_features, roberta_features, engineered_features=engineered)
    if BEST_ITERATION is not None and BEST_ITERATION >= 0:
        proba = clf_hybrid.predict_proba(X, iteration_range=(0, BEST_ITERATION + 1))[0]
    else:
        proba = clf_hybrid.predict_proba(X)[0]
    pred = int(proba[1] >= RUMOR_THRESHOLD)
    
    return {
        'prediction': pred,
        'truth_prob': float(proba[0]),
        'rumor_prob': float(proba[1]),
        'confidence': (proba[1] if pred == 1 else proba[0]) * 100
    }


# ==================== TEST CASE GROUPS ====================

TEST_CASES = {
    
    # Group 1: FAKE NEWS với metadata CAO (celebrity gossip)
    "FAKE_HIGH_METADATA": [
        {
            "title": "Brad Pitt and Angelina Jolie reveal secret child - this will shock you!",
            "metadata": {
                "tweet_count": 50000,
                "retweet_count": 15000,
                "like_count": 85000,
                "reply_count": 9000,
                "user_verified": 1
            },
            "expected": 0,
            "description": "Celebrity gossip FAKE - high engagement"
        },
        {
            "title": "Justin Bieber announces retirement - shocking news just broke!",
            "metadata": {
                "tweet_count": 40000,
                "retweet_count": 12000,
                "like_count": 70000,
                "reply_count": 8500,
                "user_verified": 1
            },
            "expected": 0,
            "description": "Celebrity fake news - verified account with high engagement"
        },
    ],
    
    # Group 2: REAL NEWS với metadata CAO (official sources)
    "REAL_HIGH_METADATA": [
        {
            "title": "WHO announces new global health initiative to combat malnutrition",
            "metadata": {
                "tweet_count": 100000,
                "retweet_count": 25000,
                "like_count": 120000,
                "reply_count": 15000,
                "user_verified": 1
            },
            "expected": 1,
            "description": "Real news from WHO - verified, high engagement"
        },
        {
            "title": "NASA Successfully Launches New Climate Monitoring Satellite",
            "metadata": {
                "tweet_count": 80000,
                "retweet_count": 20000,
                "like_count": 95000,
                "reply_count": 12000,
                "user_verified": 1
            },
            "expected": 1,
            "description": "Real news from NASA - verified, high engagement"
        },
    ],
    
    # Group 3: FAKE NEWS với metadata THẤP/ZERO (conspiracy theories)
    "FAKE_LOW_METADATA": [
        {
            "title": "Secret government experiment created giant sea monsters in the ocean",
            "metadata": {
                "tweet_count": 5,
                "retweet_count": 2,
                "like_count": 0,
                "reply_count": 1,
                "user_verified": 0
            },
            "expected": 0,
            "description": "Conspiracy theory FAKE - low engagement"
        },
        {
            "title": "Scientists confirm drinking bleach can cure cancer",
            "metadata": {
                "tweet_count": 0,
                "retweet_count": 0,
                "like_count": 0,
                "reply_count": 0,
                "user_verified": 0
            },
            "expected": 0,
            "description": "Dangerous misinformation - NO metadata (metadata=0)"
        },
        {
            "title": "Aliens are secretly running the government according to leaked files",
            "metadata": {
                "tweet_count": 10,
                "retweet_count": 5,
                "like_count": 15,
                "reply_count": 3,
                "user_verified": 0
            },
            "expected": 0,
            "description": "Conspiracy theory - minimal engagement, unverified"
        },
    ],
    
    # Group 4: REAL NEWS với metadata THẤP/ZERO (local news, small outlets)
    "REAL_LOW_METADATA": [
        {
            "title": "Local community center opens new after-school program for youth",
            "metadata": {
                "tweet_count": 50,
                "retweet_count": 8,
                "like_count": 25,
                "reply_count": 5,
                "user_verified": 0
            },
            "expected": 1,
            "description": "Real local news - low engagement"
        },
        {
            "title": "University researchers publish breakthrough in renewable energy",
            "metadata": {
                "tweet_count": 0,
                "retweet_count": 0,
                "like_count": 0,
                "reply_count": 0,
                "user_verified": 0
            },
            "expected": 1,
            "description": "Real academic news - NO metadata (metadata=0)"
        },
        {
            "title": "City council approves new environmental protection ordinance",
            "metadata": {
                "tweet_count": 20,
                "retweet_count": 3,
                "like_count": 10,
                "reply_count": 2,
                "user_verified": 0
            },
            "expected": 1,
            "description": "Real government news - minimal engagement"
        },
    ],
    
    # Group 5: EDGE CASES - test model robustness
    "EDGE_CASES": [
        {
            "title": "Breaking news: major earthquake reported",
            "metadata": {
                "tweet_count": 200000,  # Extremely high
                "retweet_count": 50000,
                "like_count": 300000,
                "reply_count": 40000,
                "user_verified": 1
            },
            "expected": 1,
            "description": "Breaking real news - EXTREME engagement"
        },
        {
            "title": "This one weird trick doctors hate - click to learn more",
            "metadata": {
                "tweet_count": 1,
                "retweet_count": 0,
                "like_count": 0,
                "reply_count": 0,
                "user_verified": 0
            },
            "expected": 0,
            "description": "Clickbait FAKE - minimal metadata"
        },
        {
            "title": "Everything is a lie believe nothing",
            "metadata": {
                "tweet_count": 0,
                "retweet_count": 0,
                "like_count": 0,
                "reply_count": 0,
                "user_verified": 0
            },
            "expected": 0,
            "description": "Conspiracy FAKE - NO metadata"}
        },
        {
            "title": "Official statement from government agency",
            "metadata": {
                "tweet_count": 500,
                "retweet_count": 100,
                "like_count": 200,
                "reply_count": 50,
                "user_verified": 1
            },
            "expected": 1,
            "description": "Official REAL news - low-moderate engagement"
        },
    ],
    
    # Group 6: MIXED DIFFICULTY
    "MIXED_DIFFICULTY": [
        {
            "title": "New study finds coffee consumption linked to longevity",
            "metadata": {
                "tweet_count": 30000,
                "retweet_count": 8000,
                "like_count": 50000,
                "reply_count": 5000,
                "user_verified": 1
            },
            "expected": 1,
            "description": "Real research news - high engagement"
        },
        {
            "title": "Miracle cure discovered that pharmaceutical companies dont want you to know about",
            "metadata": {
                "tweet_count": 20000,
                "retweet_count": 5000,
                "like_count": 40000,
                "reply_count": 3000,
                "user_verified": 0
            },
            "expected": 0,
            "description": "Medical misinformation FAKE - high engagement"
        },
    ],
}


def run_test_group(group_name, test_cases_list):
    """Run and evaluate a group of test cases"""
    
    print(f"\n{'='*80}")
    print(f"TEST GROUP: {group_name}")
    print(f"{'='*80}")
    
    y_true = []
    y_pred = []
    y_proba = []
    
    for i, case in enumerate(test_cases_list, 1):
        title = case['title']
        metadata = case['metadata']
        expected = case['expected']
        description = case['description']
        
        result = predict_news(title, metadata)
        pred = result['prediction']
        # Existing test cases use FAKE=0, REAL=1.
        # Model uses TRUTH=0, RUMOR=1, so map expected labels accordingly.
        expected_model = 1 if expected == 0 else 0
        
        y_true.append(expected_model)
        y_pred.append(pred)
        y_proba.append(result['rumor_prob'])
        
        status = "✓ CORRECT" if pred == expected_model else "✗ WRONG"
        label_pred = "FAKE" if pred == 1 else "REAL"
        label_true = "FAKE" if expected == 0 else "REAL"
        
        print(f"\n[Test {i}] {status}")
        print(f"  Title: {title[:70]}...")
        print(f"  Description: {description}")
        print(f"  Expected: {label_true} | Predicted: {label_pred}")
        print(f"  Confidence: {result['confidence']:.1f}% | Fake: {result['rumor_prob']:.4f}, Real: {result['truth_prob']:.4f}")
    
    # Statistics
    accuracy = accuracy_score(y_true, y_pred)
    correct = sum(np.array(y_true) == np.array(y_pred))
    total = len(y_true)
    
    print(f"\n{'-'*80}")
    print(f"GROUP STATISTICS:")
    print(f"  Total tests: {total}")
    print(f"  Correct: {correct}")
    print(f"  Wrong: {total - correct}")
    print(f"  Accuracy: {accuracy:.2%}")
    
    try:
        roc_auc = roc_auc_score(y_true, y_proba)
        print(f"  ROC-AUC: {roc_auc:.4f}")
    except:
        pass
    
    return accuracy, correct, total


if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("COMPREHENSIVE TEST SUITE FOR HYBRID FAKE NEWS DETECTOR")
    print("="*80)
    print(f"Device: {device}")
    print(f"Model loaded from: fake_news_xgboost.pkl")
    print(f"TF-IDF vectorizer loaded")
    print(f"RoBERTa model loaded")
    
    # Run all test groups
    total_accuracy = 0
    total_correct = 0
    total_tests = 0
    results = {}
    
    for group_name, test_cases_list in TEST_CASES.items():
        acc, correct, total = run_test_group(group_name, test_cases_list)
        results[group_name] = {'accuracy': acc, 'correct': correct, 'total': total}
        total_accuracy += acc * total
        total_correct += correct
        total_tests += total
    
    # Overall statistics
    print(f"\n\n{'='*80}")
    print("OVERALL TEST RESULTS")
    print(f"{'='*80}")
    
    for group_name, stats in results.items():
        print(f"{group_name:30} | Accuracy: {stats['accuracy']:.2%} ({stats['correct']}/{stats['total']})")
    
    overall_accuracy = total_correct / total_tests
    print(f"\n{'Total':30} | Accuracy: {overall_accuracy:.2%} ({total_correct}/{total_tests})")
    print(f"{'='*80}\n")
