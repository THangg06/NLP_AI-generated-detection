import re
import nltk
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
stop_words = set(nltk.corpus.stopwords.words('english'))

def clean_text(text: str) -> str:
    """Clean input text by removing URLs, special characters, numbers, and stopwords."""
    # Handle NaN and non-string values
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r"\burl\b", "", text)
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    text = " ".join([word for word in nltk.word_tokenize(text) if word not in stop_words])
    return text

def clean_text_for_tfidf(text: str) -> str:
    """
    Aggressive cleaning for TF-IDF (remove stopwords, special chars).
    ✅ Optimized for bag-of-words model.
    """
    # Handle NaN and non-string values
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove mentions and hashtags
    text = re.sub(r'\@\w+|\#\w+', '', text)
    text = re.sub(r"\burl\b", "", text)
    # Remove ALL special characters and numbers
    text = re.sub(r'[^A-Za-z\s]', '', text)
    text = text.lower()
    
    # Remove stopwords
    text = " ".join([word for word in nltk.word_tokenize(text) if word not in stop_words])
    return text

def clean_text_for_roberta(text: str) -> str:
    """
    Minimal cleaning for RoBERTa (keep numbers, punctuation, negations).
    ✅ RoBERTa is a language model - needs full context + punctuation.
    ❌ DO NOT remove: numbers, "not/no/never", punctuation, structure.
    
    Example:
        Input:  "This is NOT true!!! Check COVID-19 cure 100%"
        TF-IDF: "cure" (lost meaning)
        RoBERTa: "This is NOT true!!! Check COVID-19 cure 100%" (keeps negation + urgency)
    """
    # Handle NaN and non-string values
    if pd.isna(text) or not isinstance(text, str):
        return ""
    
    # Only remove URLs (completely unnecessary for fake news detection)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    
    # Convert to lowercase but KEEP everything else
    text = text.lower()
    
    # ✅ Keep:
    # - Numbers (100%, 5G, COVID-19)
    # - Punctuation (!, ?, ...)
    # - Negations (not, no, never)
    # - Special chars (hyphen, apostrophe)
    
    return text.strip()

def apply_text_cleaning(df: pd.DataFrame, text_column: str = 'title') -> pd.DataFrame:
    """Apply cleaning to an entire column of text data."""
    df['clean_text'] = df[text_column].apply(clean_text)
    return df

def normalize_metadata(df: pd.DataFrame, metadata_cols: list) -> pd.DataFrame:
    """
    Normalize metadata features using MinMaxScaler
    and save the scaler for later prediction.
    """

    scaler = MinMaxScaler()

    df[metadata_cols] = scaler.fit_transform(df[metadata_cols])
    joblib.dump(scaler, "metadata_scaler.pkl")

    return df
