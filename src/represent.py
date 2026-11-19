"""Leakage-safe representations: fit vocabulary/embeddings on training data only."""
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from kmer_utils import seq_to_kmer_string


def kmer_documents(df: pd.DataFrame, k: int) -> list[str]:
    return [seq_to_kmer_string(sequence, k) for sequence in df["sequence"]]


def make_vectorizer(kind: str, max_features: int = 2000):
    if kind == "bag_of_words": return CountVectorizer(lowercase=False, max_features=max_features)
    if kind == "tfidf": return TfidfVectorizer(lowercase=False, max_features=max_features)
    raise ValueError(f"Unknown representation: {kind}")


def fit_transform_text(train_df, test_df, k, kind, max_features=2000):
    """Fit only on train_df, then transform both partitions."""
    vectorizer = make_vectorizer(kind, max_features)
    return (vectorizer.fit_transform(kmer_documents(train_df, k)),
            vectorizer.transform(kmer_documents(test_df, k)), vectorizer)


def fit_transform_svd_embedding(train_df, test_df, k, vector_size=50):
    """Deterministic, lightweight distributional sequence embeddings."""
    X_train, X_test, vectorizer = fit_transform_text(train_df, test_df, k, "tfidf")
    components = min(vector_size, X_train.shape[0] - 1, X_train.shape[1] - 1)
    if components < 1: return X_train.toarray(), X_test.toarray(), None, vectorizer
    model = TruncatedSVD(n_components=components, random_state=42)
    return model.fit_transform(X_train), model.transform(X_test), model, vectorizer


# Legacy convenience functions. New pipeline code must split before fitting.
def bag_of_words(df, k, max_features=2000):
    vec = make_vectorizer("bag_of_words", max_features); return vec.fit_transform(kmer_documents(df, k)), vec


def tfidf(df, k, max_features=2000):
    vec = make_vectorizer("tfidf", max_features); return vec.fit_transform(kmer_documents(df, k)), vec
