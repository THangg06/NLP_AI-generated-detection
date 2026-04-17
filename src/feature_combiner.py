import numpy as np

def extra_features(df):
    """Build simple handcrafted content features for rumor detection."""
    content = df['content'].fillna('').astype(str)
    return np.vstack([
        content.apply(len),
        content.apply(lambda x: len(x.split())),
        content.apply(lambda x: x.count('!')),
        content.apply(lambda x: x.count('?')),
        content.apply(lambda x: int(x.isupper())),
    ]).T.astype(np.float32)


def combine_features(tfidf_matrix, roberta_embeddings, engineered_features=None, roberta_weight=2.0):
    """
    Concatenate text features (TF-IDF + RoBERTa + optional engineered stats).
    
    Args:
        tfidf_matrix: TF-IDF features (already normalized 0-1)
        roberta_embeddings: RoBERTa embeddings (scaled to emphasize)
        engineered_features: Optional handcrafted features from raw content
        roberta_weight: Multiplier for RoBERTa embeddings (default 2.0)
    
    Returns:
        Combined feature matrix
    """
    # Scale RoBERTa features to balance importance with TF-IDF
    roberta_scaled = roberta_embeddings * roberta_weight

    feature_blocks = [
        tfidf_matrix,           # TF-IDF features from text
        roberta_scaled,         # RoBERTa embeddings from text
    ]
    if engineered_features is not None:
        feature_blocks.append(engineered_features)

    return np.hstack(tuple(feature_blocks))
    
# Total: TF-IDF dims + 768 RoBERTa dims + optional engineered dims
