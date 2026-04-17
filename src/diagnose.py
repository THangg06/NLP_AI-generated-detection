import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import clean_text_for_tfidf, clean_text_for_roberta
from tfidf_vectorizer import compute_tfidf, load_vectorizer
from roberta_embedder import extract_embeddings
from feature_combiner import combine_features, extra_features


def _encode_labels(series):
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {
        'truth': 0,
        'rumor': 1,
        '0': 0,
        '1': 1,
    }
    return normalized.map(mapping)


def _build_features(df, tfidf_vectorizer):
    df = df.copy()
    df['clean_text_tfidf'] = df['content'].apply(clean_text_for_tfidf)
    df['clean_text_roberta'] = df['content'].apply(clean_text_for_roberta)

    labels = _encode_labels(df['label'])
    valid = ~labels.isna()
    df = df.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].astype(int)

    tfidf_features, _ = compute_tfidf(df['clean_text_tfidf'], fitted_vectorizer=tfidf_vectorizer)
    roberta_features = extract_embeddings(df['clean_text_roberta'])
    engineered = extra_features(df)
    X = combine_features(tfidf_features, roberta_features, engineered_features=engineered)

    return X, labels

def diagnose_model():
    """Diagnose model performance on training vs test data."""
    
    print("="*70)
    print("MODEL DIAGNOSTIC REPORT")
    print("="*70)
    
    # Load model
    clf = joblib.load("fake_news_xgboost.pkl")
    tfidf_vectorizer = load_vectorizer("tfidf_vectorizer.pkl")
    
    # ===== CHECK TRAINING DATA =====
    print("\n[1] CHECKING TRAINING DATA")
    print("-" * 70)
    
    try:
        df_train = pd.read_csv("data/train.csv")
        print(f"Training data shape: {df_train.shape}")
        print(f"Label distribution:\n{df_train['label'].value_counts()}\n")
        X_train, y_train = _build_features(df_train, tfidf_vectorizer)
        
        y_train_pred = clf.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"\nTraining Classification Report:")
        print(classification_report(y_train, y_train_pred, target_names=["Truth", "Rumor"]))
        
    except Exception as e:
        print(f"⚠️  Error processing training data: {e}")
    
    # ===== CHECK TEST DATA =====
    print("\n[2] CHECKING TEST DATA (data/test.csv)")
    print("-" * 70)
    
    try:
        df_test = pd.read_csv("data/test.csv")
        print(f"Test data shape: {df_test.shape}")

        print(f"Label distribution:\n{df_test['label'].value_counts()}\n")
        X_test, y_test = _build_features(df_test, tfidf_vectorizer)

        y_test_pred = clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        print(f"Test Accuracy: {test_accuracy:.4f}")
        print(f"\nTest Classification Report:")
        print(classification_report(y_test, y_test_pred, target_names=["Truth", "Rumor"]))
        
    except Exception as e:
        print(f"⚠️  Error processing test data: {e}")
        test_accuracy = None
    
    # ===== COMPARISON =====
    print("\n[3] MODEL DIAGNOSIS")
    print("-" * 70)
    
    try:
        if test_accuracy is None:
            print("Không thể so sánh train/test do lỗi ở bước test.")
            print("\n" + "="*70)
            return

        if train_accuracy > test_accuracy and (train_accuracy - test_accuracy) > 0.15:
            print("⚠️  OVERFITTING DETECTED:")
            print(f"   - Training: {train_accuracy:.4f}, Test: {test_accuracy:.4f}")
            print(f"   - Gap: {(train_accuracy - test_accuracy):.4f}")
        
        if train_accuracy < 0.6 and test_accuracy < 0.6:
            print("⚠️  POOR MODEL PERFORMANCE:")
            print(f"   - Both accuracies are low (< 60%)")
            print(f"   - Check:");
            print(f"     1. Data quality and preprocessing")
            print(f"     2. Feature extraction (TF-IDF, RoBERTa)")
            print(f"     3. Model hyperparameters")
            print(f"     4. Training data volume")
        
        if abs(train_accuracy - test_accuracy) < 0.05:
            print("✓ Balanced performance (no clear overfitting)")
    
    except Exception as e:
        print(f"Error in comparison: {e}")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    diagnose_model()
