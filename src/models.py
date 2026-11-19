"""Leakage-safe training, evaluation and model comparison."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit
from sklearn.tree import DecisionTreeClassifier
import represent

SEED = 42


def group_train_test_split(df: pd.DataFrame, test_size=0.25, seed=SEED):
    missing = {"label", "source_group"} - set(df.columns)
    if missing: raise ValueError(f"dataset is missing required columns: {sorted(missing)}")
    if df.label.nunique() < 2 or df.source_group.nunique() < 2: raise ValueError("need at least two labels and source groups")
    for attempt in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + attempt)
        train_idx, test_idx = next(splitter.split(df, groups=df.source_group))
        if set(df.iloc[train_idx].label) == set(df.label) and set(df.iloc[test_idx].label) == set(df.label): return train_idx, test_idx
    raise ValueError("could not create group-separated split containing every class")


def build_model(name):
    if name == "logreg": return LogisticRegression(max_iter=2000, random_state=SEED)
    if name == "tree": return DecisionTreeClassifier(max_depth=8, random_state=SEED)
    if name == "forest": return RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="logreg"):
    clf = build_model(model_name).fit(X_train, y_train); predictions = clf.predict(X_test); labels = sorted(set(y_train))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="weighted", zero_division=0)
    return clf, {"model": model_name, "accuracy": accuracy_score(y_test, predictions), "precision_weighted": precision, "recall_weighted": recall, "f1_weighted": f1}, confusion_matrix(y_test, predictions, labels=labels), predictions


def run_all_representations(df: pd.DataFrame, k=4):
    train_idx, test_idx = group_train_test_split(df); train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]
    results, trained, artifacts = [], {}, {"train_idx": train_idx, "test_idx": test_idx}
    for representation in ("bag_of_words", "tfidf"):
        X_train, X_test, vectorizer = represent.fit_transform_text(train_df, test_df, k, representation)
        artifacts[representation] = {"vectorizer": vectorizer, "X_train": X_train, "X_test": X_test}
        for name in ("logreg", "tree", "forest"):
            clf, metrics, matrix, _ = train_and_evaluate(X_train, X_test, train_df.label, test_df.label, name)
            metrics["representation"] = representation; results.append(metrics); trained[f"{representation}_{name}"] = clf; artifacts[f"{representation}_{name}_cm"] = matrix
    X_train, X_test, embedder, vectorizer = represent.fit_transform_svd_embedding(train_df, test_df, k)
    artifacts["svd_embedding"] = {"embedder": embedder, "vectorizer": vectorizer}
    for name in ("logreg", "tree", "forest"):
        clf, metrics, matrix, _ = train_and_evaluate(X_train, X_test, train_df.label, test_df.label, name)
        metrics["representation"] = "svd_embedding"; results.append(metrics); trained[f"svd_embedding_{name}"] = clf; artifacts[f"svd_embedding_{name}_cm"] = matrix
    return pd.DataFrame(results), trained, artifacts
