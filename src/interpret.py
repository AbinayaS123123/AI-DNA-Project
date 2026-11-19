"""Feature importance and a simple superficial-feature baseline."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def top_logreg_features(clf, vectorizer, labels, top_n=15):
    if len(labels) != 2: raise ValueError("feature table currently supports binary classification")
    features, coefficients = np.array(vectorizer.get_feature_names_out()), clf.coef_[0]
    rows = []
    for idx in np.argsort(coefficients)[:top_n]: rows.append({"kmer": features[idx], "associated_class": clf.classes_[0], "coefficient": coefficients[idx]})
    for idx in np.argsort(coefficients)[-top_n:][::-1]: rows.append({"kmer": features[idx], "associated_class": clf.classes_[1], "coefficient": coefficients[idx]})
    return pd.DataFrame(rows)

def artifact_check(df, kmer_accuracy, train_idx, test_idx):
    X, y = df[["length", "gc_content"]].values, df.label.values
    clf = LogisticRegression(max_iter=1000, random_state=42).fit(X[train_idx], y[train_idx])
    trivial_accuracy = accuracy_score(y[test_idx], clf.predict(X[test_idx])); gap = kmer_accuracy - trivial_accuracy
    return {"trivial_length_gc_accuracy": trivial_accuracy, "kmer_model_accuracy": kmer_accuracy, "gap": gap, "verdict": "K-mer model exceeds the simple composition baseline." if gap > .05 else "Investigate possible length/GC shortcut; performance is close to the baseline."}
