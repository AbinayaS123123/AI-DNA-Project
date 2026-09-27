"""
kmer_utils.py
-------------
Turns DNA/RNA sequences into "words" (k-mers), the core step in
treating biological sequences as a language.
"""

from collections import Counter
import re

VALID_BASES = set("ACGT")


def clean_sequence(seq: str) -> str:
    """Uppercase, convert RNA->DNA alphabet, strip anything not ACGT."""
    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGT]", "", seq)
    return seq


def seq_to_kmers(seq: str, k: int) -> list[str]:
    """Slide a window of size k across the sequence (overlapping k-mers)."""
    seq = clean_sequence(seq)
    if len(seq) < k:
        return []
    return [seq[i:i + k] for i in range(len(seq) - k + 1)]


def seq_to_kmer_string(seq: str, k: int) -> str:
    """Space-joined k-mers, so sklearn's text vectorizers work directly."""
    return " ".join(seq_to_kmers(seq, k))


def kmer_frequency(seq: str, k: int) -> Counter:
    return Counter(seq_to_kmers(seq, k))


def vocabulary_size(k: int) -> int:
    """Theoretical max vocabulary size for alphabet size 4 (A,C,G,T)."""
    return 4 ** k


def dataset_vocab_stats(sequences: list[str], k: int) -> dict:
    """Observed vocabulary size and sparsity for a given k on real data."""
    all_kmers = Counter()
    for s in sequences:
        all_kmers.update(seq_to_kmers(s, k))
    observed = len(all_kmers)
    theoretical = vocabulary_size(k)
    return {
        "k": k,
        "observed_vocab": observed,
        "theoretical_vocab": theoretical,
        "coverage_pct": round(100 * observed / theoretical, 2),
        "top_10": all_kmers.most_common(10),
    }


if __name__ == "__main__":
    demo = "ATGCGTATGCGATCGATCGGATCG"
    for k in [3, 4, 6]:
        print(f"k={k}: {seq_to_kmers(demo, k)[:5]}... "
              f"({len(seq_to_kmers(demo, k))} k-mers)")
