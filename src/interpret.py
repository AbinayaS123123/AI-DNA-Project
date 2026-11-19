"""
interpret.py
------------
Asks "what did the model actually learn?"
  1. Which k-mers/features drove the logistic regression decision.
  2. A crucial ARTIFACT CHECK: is the model just learning GC content /
     length instead of real sequence patterns? We test this by training
     a trivial classifier on ONLY length + GC content and comparing its
     accuracy to the full k-mer model. If the trivial model gets
     nearly the same accuracy, the "real" model may not be learning
     much beyond a compositional shortcut.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score

from models import group_train_test_split


def top_logreg_features(clf, vectorizer, top_n=15):
    """Only valid for CountVectorizer/TfidfVectorizer-based logistic regression."""
    coefs = clf.coef_[0]
    feature_names = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(coefs)
    viral_side = pd.Series(coefs[order[:top_n]], index=feature_names[order[:top_n]])
    human_side = pd.Series(coefs[order[-top_n:]], index=feature_names[order[-top_n:]])
    return {
        "pushes_toward_viral": viral_side.sort_values(),
        "pushes_toward_human": human_side.sort_values(ascending=False),
    }


def artifact_check(df: pd.DataFrame, kmer_accuracy: float):
    """Train a classifier using ONLY length + GC content, compare to the
    k-mer model's accuracy. Large gap = k-mer model is learning something
    beyond trivial composition. Small/no gap = likely a shortcut."""
    train_idx, test_idx = group_train_test_split(df)
    X = df[["length", "gc_content"]].values
    y = df["label"].values

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    trivial_acc = accuracy_score(y[test_idx], preds)

    gap = kmer_accuracy - trivial_acc
    verdict = (
        "K-mer model substantially outperforms the trivial length/GC "
        "baseline -- evidence it captured more than superficial composition."
        if gap > 0.05 else
        "K-mer model performs close to the trivial length/GC baseline -- "
        "the classes may be separable mostly by superficial composition, "
        "not by specific sequence patterns. Worth investigating further."
    )

    return {
        "trivial_length_gc_accuracy": trivial_acc,
        "kmer_model_accuracy": kmer_accuracy,
        "gap": gap,
        "verdict": verdict,
    }


if __name__ == "__main__":
    import represent
    from models import group_train_test_split, train_and_evaluate

    df = pd.read_csv("data/raw/sequences.csv")
    k = 4
    X_tfidf, vec = represent.tfidf(df, k=k)
    train_idx, test_idx = group_train_test_split(df)
    y = df["label"].values
    clf, metrics, report = train_and_evaluate(X_tfidf.toarray(), y, train_idx, test_idx, "logreg")

    print("=== Top discriminative k-mer features (logistic regression) ===")
    top = top_logreg_features(clf, vec)
    print(top["pushes_toward_human"])
    print(top["pushes_toward_viral"])

    print("\n=== Artifact check ===")
    check = artifact_check(df, kmer_accuracy=metrics["accuracy"])
    for kk, v in check.items():
        print(f"{kk}: {v}")
