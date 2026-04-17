import pandas as pd
import joblib
import numpy as np
import json
import os
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from preprocessing import clean_text_for_tfidf, clean_text_for_roberta
from tfidf_vectorizer import load_vectorizer
from feature_combiner import combine_features, extra_features

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading models...")

# Text-only model (TF-IDF + RoBERTa)
clf_model = joblib.load("fake_news_xgboost.pkl")

# TFIDF
tfidf_vectorizer = load_vectorizer("tfidf_vectorizer.pkl")

# RoBERTa
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")

roberta_model.to(device)
roberta_model.eval()

RUMOR_THRESHOLD = 0.5
BEST_ITERATION = None
if os.path.exists("rumor_threshold.json"):
    try:
        with open("rumor_threshold.json", "r", encoding="utf-8") as f:
            threshold_payload = json.load(f)
            RUMOR_THRESHOLD = float(threshold_payload.get("rumor_threshold", 0.5))
            best_iter_val = threshold_payload.get("best_iteration")
            if best_iter_val is not None:
                BEST_ITERATION = int(best_iter_val)
    except Exception:
        RUMOR_THRESHOLD = 0.5
        BEST_ITERATION = None


def get_roberta_embedding(text):
    """Get RoBERTa [CLS] embedding for text (better for classification)."""
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

    # Use [CLS] token (index 0) instead of mean pooling
    # [CLS] is designed for classification tasks
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding.cpu().numpy().flatten()


def predict_news(title, metadata=None):
    """
    Predict if news is rumor or truth using text-only model.
    
    Args:
        title: Article title/text to predict
    
    Returns:
        Dict with prediction and confidence
    """
    
    # -------- CLEAN TEXT --------
    clean_tfidf = clean_text_for_tfidf(title)      # Aggressive cleaning
    clean_roberta = clean_text_for_roberta(title)  # Minimal cleaning

    # -------- TFIDF (aggressive cleaning) --------
    tfidf_features = tfidf_vectorizer.transform([clean_tfidf]).toarray()

    # -------- ROBERTA (minimal cleaning + [CLS] token) --------
    roberta_features = np.array([get_roberta_embedding(clean_roberta)])

    # -------- ENGINEERED FEATURES --------
    content_df = pd.DataFrame({'content': [title]})
    engineered = extra_features(content_df)

    # Combine text features
    X = combine_features(tfidf_features, roberta_features, engineered_features=engineered)

    # -------- PREDICT using text-only model --------
    if BEST_ITERATION is not None and BEST_ITERATION >= 0:
        pred_proba = clf_model.predict_proba(X, iteration_range=(0, BEST_ITERATION + 1))[0]
    else:
        pred_proba = clf_model.predict_proba(X)[0]
    pred = int(pred_proba[1] >= RUMOR_THRESHOLD)

    label = "🔴 RUMOR" if pred == 1 else "✅ TRUTH"
    confidence = (pred_proba[1] if pred == 1 else pred_proba[0]) * 100

    print(f"\n{'='*60}")
    print(f"Title: {title}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.1f}%")
    print(f"Truth Probability (class 0): {pred_proba[0]:.4f}")
    print(f"Rumor Probability (class 1): {pred_proba[1]:.4f}")
    print(f"{'='*60}\n")

    return {
        'pred_class': int(pred),
        'prediction': label,
        'confidence': confidence,
        'truth_prob': float(pred_proba[0]),
        'rumor_prob': float(pred_proba[1])
    }


# ---------------- TEST CASES ----------------

test_cases = [
    "New research study published about global health trends",
    "NASA launches new satellite to monitor global climate change",
    "Scientists confirm drinking bleach can cure all diseases",
    "Aliens officially contacted Earth according to leaked government files",
    "Secret government experiment created giant sea monsters",
    "World Health Organization releases new vaccination guidelines"
]


# if __name__ == "__main__":

#     y_true = []
#     y_pred = []

#     for case in test_cases:

#         pred = predict_news(case["title"], case["metadata"])

#         y_pred.append(pred)
#         y_true.append(case["label"])

#     print("\n==============================")
#     print("EVALUATION RESULTS")
#     print("==============================")

#     correct = sum(np.array(y_true) == np.array(y_pred))
#     wrong = len(y_true) - correct

#     print("Total test cases:", len(y_true))
#     print("Correct predictions:", correct)
#     print("Wrong predictions:", wrong)

#     acc = accuracy_score(y_true, y_pred)
#     print("Accuracy:", round(acc,3))

#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=["Fake","Real"]))

#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_true, y_pred))

# # ---------------- LOAD TEST DATA FROM EXCEL ----------------

df = pd.read_csv("src/fake_news_test_data_200_rows.csv")

if __name__ == "__main__":

    y_true = []
    y_pred = []

    for _, row in df.iterrows():

        title = row["title"]

        metadata = [
            row["tweet_count"],
            row["retweet_count"],
            row["like_count"],
            row["reply_count"],
            row["user_verified"]
        ]

        true_label = row["label"]

        result = predict_news(title, metadata)
        pred = result['pred_class']

        y_pred.append(pred)
        y_true.append(true_label)

    print("\n==============================")
    print("EVALUATION RESULTS")
    print("==============================")

    correct = sum(np.array(y_true) == np.array(y_pred))
    wrong = len(y_true) - correct

    print("Total test cases:", len(y_true))
    print("Correct predictions:", correct)
    print("Wrong predictions:", wrong)

    acc = accuracy_score(y_true, y_pred)
    print("Accuracy:", round(acc,3))

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Fake","Real"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

# if __name__ == "__main__":

#     y_true = []
#     y_pred = []

#     for case in test_cases:

#         pred = predict_news(case["title"], case["metadata"])

#         y_pred.append(pred)
#         y_true.append(case["label"])

#     print("\n==============================")
#     print("EVALUATION RESULTS")
#     print("==============================")

#     correct = sum(np.array(y_true) == np.array(y_pred))
#     wrong = len(y_true) - correct

#     print("Total test cases:", len(y_true))
#     print("Correct predictions:", correct)
#     print("Wrong predictions:", wrong)

#     acc = accuracy_score(y_true, y_pred)
#     print("Accuracy:", round(acc,3))

#     print("\nClassification Report:")
#     print(classification_report(y_true, y_pred, target_names=["Fake","Real"]))

#     print("\nConfusion Matrix:")
#     print(confusion_matrix(y_true, y_pred))

