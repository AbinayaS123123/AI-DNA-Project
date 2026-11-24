"""
explore.py
----------
Exploratory analysis: k-mer frequency stats + dimensionality reduction
to check (before modeling) whether classes separate in sequence-word space.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

from kmer_utils import seq_to_kmer_string, dataset_vocab_stats


def plot_length_gc_distributions(df: pd.DataFrame, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for label, group in df.groupby("label"):
        axes[0].hist(group["length"], bins=25, alpha=0.6, label=label)
        axes[1].hist(group["gc_content"], bins=25, alpha=0.6, label=label)
    axes[0].set_title("Sequence length by class")
    axes[0].set_xlabel("length (bp)")
    axes[0].legend()
    axes[1].set_title("GC content by class")
    axes[1].set_xlabel("GC fraction")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/length_gc_distributions.png", dpi=150)
    plt.close()


def vocab_report(df: pd.DataFrame, ks=(3, 4, 6)) -> pd.DataFrame:
    rows = []
    for k in ks:
        stats = dataset_vocab_stats(df["sequence"].tolist(), k)
        rows.append(stats)
    return pd.DataFrame(rows)[["k", "observed_vocab", "theoretical_vocab", "coverage_pct"]]


def kmer_count_matrix(df: pd.DataFrame, k: int):
    """Bag-of-k-mers count matrix via CountVectorizer on space-joined k-mers."""
    kmer_strings = df["sequence"].apply(lambda s: seq_to_kmer_string(s, k))
    vec = CountVectorizer(lowercase=False)
    X = vec.fit_transform(kmer_strings)
    return X, vec


def plot_cluster_projection(df: pd.DataFrame, k: int, out_dir: str, method="pca"):
    X, _ = kmer_count_matrix(df, k)
    X = X.toarray()

    if method == "pca":
        proj = PCA(n_components=2, random_state=42).fit_transform(X)
        title = f"PCA of {k}-mer counts"
    else:
        perplexity = min(30, max(2, len(df) - 1))
        proj = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                    init="pca").fit_transform(X)
        title = f"t-SNE of {k}-mer counts"

    fig, ax = plt.subplots(figsize=(6, 5))
    for label, color in zip(["human", "viral"], ["#4C72B0", "#DD8452"]):
        mask = df["label"] == label
        ax.scatter(proj[mask, 0], proj[mask, 1], alpha=0.6, s=18,
                   label=label, color=color)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fname = f"{out_dir}/{method}_k{k}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def top_discriminative_kmers(df: pd.DataFrame, k: int, top_n=15):
    """Simple frequency-ratio view: which k-mers differ most between classes."""
    from kmer_utils import kmer_frequency
    human_counts = pd.Series(dtype=int)
    viral_counts = pd.Series(dtype=int)

    human_ctr, viral_ctr = {}, {}
    for _, row in df.iterrows():
        ctr = kmer_frequency(row["sequence"], k)
        target = human_ctr if row["label"] == "human" else viral_ctr
        for kmer, c in ctr.items():
            target[kmer] = target.get(kmer, 0) + c

    human_s = pd.Series(human_ctr, dtype=float)
    viral_s = pd.Series(viral_ctr, dtype=float)
    human_s = human_s / human_s.sum()
    viral_s = viral_s / viral_s.sum()

    all_kmers = set(human_s.index) | set(viral_s.index)
    ratio = {}
    for km in all_kmers:
        h = human_s.get(km, 1e-6)
        v = viral_s.get(km, 1e-6)
        ratio[km] = np.log2(h / v)

    ratio_s = pd.Series(ratio).sort_values()
    return pd.concat([
        ratio_s.head(top_n).rename("log2(human/viral) -- viral-enriched"),
        ratio_s.tail(top_n).rename("log2(human/viral) -- human-enriched"),
    ])


if __name__ == "__main__":
    from data_io import prepare_dataset
    df, rejected = prepare_dataset(pd.read_csv("data/real/nextvir_sequences.csv"))
    if rejected:
        print(f"Skipped {rejected} invalid sequences")
    plot_length_gc_distributions(df, "outputs/figures")
    print(vocab_report(df))
    plot_cluster_projection(df, k=4, out_dir="outputs/figures", method="pca")
    print(top_discriminative_kmers(df, k=4))
