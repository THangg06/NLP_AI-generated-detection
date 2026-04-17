import os
from pathlib import Path

import pandas as pd

from classifier import train_xgboost
from feature_combiner import combine_features, extra_features
from preprocessing import clean_text_for_roberta, clean_text_for_tfidf
from roberta_embedder import extract_embeddings
from tfidf_vectorizer import compute_tfidf, save_vectorizer

# Set working directory to project root
project_root = Path(__file__).parent.parent
os.chdir(project_root)
print(f"Working directory: {os.getcwd()}")

def _validate_schema(df: pd.DataFrame, file_name: str) -> None:
    required = {"id", "content", "label", "year", "month", "day", "year_month", "year_season", "weight"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{file_name} thiếu cột bắt buộc: {missing}")


def _encode_labels(label_series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(label_series):
        numeric = pd.to_numeric(label_series, errors="coerce")
        if set(numeric.dropna().unique()).issubset({0, 1}):
            return numeric

    normalized = label_series.astype(str).str.strip().str.lower()
    mapping = {
        "truth": 0,
        "rumor": 1,
        "0": 0,
        "1": 1,
    }
    encoded = normalized.map(mapping)

    invalid_mask = encoded.isna()
    if invalid_mask.any():
        invalid_values = sorted(normalized[invalid_mask].unique().tolist())
        raise ValueError(
            "Chi nhan 2 nhan duoc phep: 'truth'->0 va 'rumor'->1. "
            f"Gia tri khong hop le: {invalid_values}"
        )

    return encoded


def _prepare_split(df: pd.DataFrame, split_name: str):
    print(f"\n[{split_name}] cleaning text...")
    df = df.copy()
    df["clean_text_tfidf"] = df["content"].apply(clean_text_for_tfidf)
    df["clean_text_roberta"] = df["content"].apply(clean_text_for_roberta)
    labels = _encode_labels(df["label"])
    weights = pd.to_numeric(df["weight"], errors="coerce").fillna(1.0)
    valid = (~labels.isna()) & (~df["clean_text_roberta"].isna())

    dropped = int((~valid).sum())
    if dropped > 0:
        print(f"[{split_name}] bỏ {dropped} dòng do label/text không hợp lệ")

    df = df.loc[valid].reset_index(drop=True)
    labels = labels.loc[valid].astype(int).reset_index(drop=True)
    weights = weights.loc[valid].astype(float).reset_index(drop=True)

    print(f"[{split_name}] samples hợp lệ: {len(df)}")
    print(f"[{split_name}] label distribution:\n{labels.value_counts().sort_index()}")

    return df, labels.values, weights.values


def main():
    print("Loading train/val/test data...")
    train_df = pd.read_csv("data/train.csv")
    val_df = pd.read_csv("data/val.csv")
    test_df = pd.read_csv("data/test.csv")

    _validate_schema(train_df, "train.csv")
    _validate_schema(val_df, "val.csv")
    _validate_schema(test_df, "test.csv")

    print(f"Train shape: {train_df.shape}")
    print(f"Val shape:   {val_df.shape}")
    print(f"Test shape:  {test_df.shape}")

    train_df, y_train, train_weights = _prepare_split(train_df, "TRAIN")
    val_df, y_val, _ = _prepare_split(val_df, "VAL")
    test_df, y_test, _ = _prepare_split(test_df, "TEST")

    print("\nGenerating TF-IDF features (fit on TRAIN)...")
    X_train_tfidf, tfidf_vectorizer = compute_tfidf(train_df["clean_text_tfidf"])
    X_val_tfidf, _ = compute_tfidf(val_df["clean_text_tfidf"], fitted_vectorizer=tfidf_vectorizer)
    X_test_tfidf, _ = compute_tfidf(test_df["clean_text_tfidf"], fitted_vectorizer=tfidf_vectorizer)

    print("Generating RoBERTa embeddings...")
    X_train_roberta = extract_embeddings(train_df["clean_text_roberta"])
    X_val_roberta = extract_embeddings(val_df["clean_text_roberta"])
    X_test_roberta = extract_embeddings(test_df["clean_text_roberta"])

    print("Generating engineered text features...")
    X_train_extra = extra_features(train_df)
    X_val_extra = extra_features(val_df)
    X_test_extra = extra_features(test_df)

    print("Combining text features...")
    X_train = combine_features(X_train_tfidf, X_train_roberta, engineered_features=X_train_extra)
    X_val = combine_features(X_val_tfidf, X_val_roberta, engineered_features=X_val_extra)
    X_test = combine_features(X_test_tfidf, X_test_roberta, engineered_features=X_test_extra)

    print("Training XGBoost classifier...")
    train_xgboost(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_weights=train_weights,
    )

    print("\nSaving vectorizer...")
    save_vectorizer(tfidf_vectorizer, "tfidf_vectorizer.pkl")
    print("Done.")

if __name__ == "__main__":
    main()
