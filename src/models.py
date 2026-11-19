"""
models.py
---------
Train and compare classifiers across different sequence representations.
Uses GROUP-AWARE splitting (by source_group, i.e. simulated species/strain)
so the test set contains sequence "families" unseen during training --
this avoids the near-duplicate-leakage trap described in the project plan.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report


def group_train_test_split(df: pd.DataFrame, test_size=0.25, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(df, groups=df["source_group"]))
    return train_idx, test_idx


def train_and_evaluate(X, y, train_idx, test_idx, model_name="logreg"):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    if model_name == "logreg":
        clf = LogisticRegression(max_iter=2000)
    elif model_name == "tree":
        clf = DecisionTreeClassifier(max_depth=8, random_state=42)
    elif model_name == "forest":
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    else:
        raise ValueError(model_name)

    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "f1": f1_score(y_test, preds, pos_label="viral"),
    }
    report = classification_report(y_test, preds)
    return clf, metrics, report


def run_all_representations(df: pd.DataFrame, k=4):
    """Compare BoW, TF-IDF, and word2vec-avg embeddings across 3 model types."""
    import represent

    train_idx, test_idx = group_train_test_split(df)
    y = df["label"].values

    results = []
    trained = {}

    # --- Bag of Words ---
    X_bow, _ = represent.bag_of_words(df, k=k)
    X_bow = X_bow.toarray()
    for model_name in ["logreg", "tree", "forest"]:
        clf, metrics, report = train_and_evaluate(X_bow, y, train_idx, test_idx, model_name)
        metrics["representation"] = "bag_of_words"
        results.append(metrics)
        trained[f"bow_{model_name}"] = clf

    # --- TF-IDF ---
    X_tfidf, _ = represent.tfidf(df, k=k)
    X_tfidf = X_tfidf.toarray()
    for model_name in ["logreg", "tree", "forest"]:
        clf, metrics, report = train_and_evaluate(X_tfidf, y, train_idx, test_idx, model_name)
        metrics["representation"] = "tfidf"
        results.append(metrics)
        trained[f"tfidf_{model_name}"] = clf

    # --- word2vec-style averaged embedding ---
    w2v = represent.train_word2vec(df, k=k)
    X_emb = represent.embed_dataset(df, k=k, w2v_model=w2v)
    for model_name in ["logreg", "tree", "forest"]:
        clf, metrics, report = train_and_evaluate(X_emb, y, train_idx, test_idx, model_name)
        metrics["representation"] = "word2vec_avg"
        results.append(metrics)
        trained[f"emb_{model_name}"] = clf

    results_df = pd.DataFrame(results)[["representation", "model", "accuracy", "f1"]]
    return results_df, trained, (train_idx, test_idx)


if __name__ == "__main__":
    df = pd.read_csv("data/raw/sequences.csv")
    results_df, trained, splits = run_all_representations(df, k=4)
    print(results_df.sort_values("accuracy", ascending=False))
