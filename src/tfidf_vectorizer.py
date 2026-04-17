from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

def compute_tfidf(corpus, max_features=4000, fitted_vectorizer=None):
    """Compute TF-IDF features from cleaned text.
    
    Args:
        corpus: Text data to vectorize
        max_features: Max features (only used if fitted_vectorizer is None)
        fitted_vectorizer: Pre-fitted vectorizer (for prediction)
    
    Returns:
        TF-IDF array and vectorizer object
    """
    if fitted_vectorizer is None:
        # Training: create new vectorizer with better config
        # - min_df: ignore terms with frequency < 5 (stronger noise reduction)
        # - max_df: ignore terms with frequency > 85% (common stopwords)
        # - ngram_range: (1,2) = unigram + bigram
        # - sublinear_tf: log scaling for TF component
        tfidf = TfidfVectorizer(
            max_features=max_features,
            min_df=5,
            max_df=0.85,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        features = tfidf.fit_transform(corpus).toarray()
    else:
        # Prediction: use fitted vectorizer
        tfidf = fitted_vectorizer
        features = tfidf.transform(corpus).toarray()
    
    return features, tfidf

def save_vectorizer(vectorizer, path="tfidf_vectorizer.pkl"):
    """Save fitted vectorizer."""
    joblib.dump(vectorizer, path)
    print(f"Vectorizer saved to {path}")

def load_vectorizer(path="tfidf_vectorizer.pkl"):
    """Load fitted vectorizer."""
    return joblib.load(path)
