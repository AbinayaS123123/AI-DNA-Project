"""
represent.py
------------
NLP-style representations of DNA/RNA "sentences" (sequences) made of
"words" (k-mers): Bag-of-Words, TF-IDF, and word2vec-style embeddings
averaged into a sequence-level vector (like a document embedding).
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec

from kmer_utils import seq_to_kmer_string, seq_to_kmers


def bag_of_words(df: pd.DataFrame, k: int, max_features=2000):
    kmer_strings = df["sequence"].apply(lambda s: seq_to_kmer_string(s, k))
    vec = CountVectorizer(lowercase=False, max_features=max_features)
    X = vec.fit_transform(kmer_strings)
    return X, vec


def tfidf(df: pd.DataFrame, k: int, max_features=2000):
    kmer_strings = df["sequence"].apply(lambda s: seq_to_kmer_string(s, k))
    vec = TfidfVectorizer(lowercase=False, max_features=max_features)
    X = vec.fit_transform(kmer_strings)
    return X, vec


def train_word2vec(df: pd.DataFrame, k: int, vector_size=50, window=5, min_count=2):
    """Train word2vec where each 'sentence' is the list of k-mers in one sequence."""
    sentences = [seq_to_kmers(s, k) for s in df["sequence"]]
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,          # skip-gram, analogous to predicting context k-mers
        workers=2,
        seed=42,
        epochs=15,
    )
    return model


def sequence_embedding(seq: str, k: int, w2v_model, vector_size=50):
    """Average word2vec k-mer vectors -> one vector per sequence (doc embedding)."""
    kmers = seq_to_kmers(seq, k)
    vecs = [w2v_model.wv[km] for km in kmers if km in w2v_model.wv]
    if not vecs:
        return np.zeros(vector_size)
    return np.mean(vecs, axis=0)


def embed_dataset(df: pd.DataFrame, k: int, w2v_model, vector_size=50):
    X = np.vstack([
        sequence_embedding(s, k, w2v_model, vector_size) for s in df["sequence"]
    ])
    return X


def most_similar_kmers(w2v_model, kmer: str, topn=10):
    """Sanity check: do k-mers in similar contexts get similar embeddings?"""
    if kmer not in w2v_model.wv:
        return f"'{kmer}' not in vocabulary (try a more frequent k-mer)"
    return w2v_model.wv.most_similar(kmer, topn=topn)


if __name__ == "__main__":
    df = pd.read_csv("data/raw/sequences.csv")
    k = 4
    w2v = train_word2vec(df, k=k)
    example_kmer = df["sequence"].iloc[0][:k]
    print(f"Most similar k-mers to '{example_kmer}':")
    print(most_similar_kmers(w2v, example_kmer))
