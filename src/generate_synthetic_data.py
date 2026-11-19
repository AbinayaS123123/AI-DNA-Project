"""
generate_synthetic_data.py
---------------------------
Generates a STAND-IN dataset of human-like vs viral-like DNA sequences.

WHY THIS EXISTS:
This sandbox cannot reach NCBI or other genomic databases (network is
restricted to package registries only). Rather than use pure random
noise, sequences here are generated with a first-order Markov chain
whose transition probabilities are set from REAL, documented
biological statistics:

  - Human genomic DNA: ~41% GC content, and strong CpG dinucleotide
    SUPPRESSION (observed/expected CpG ratio ~0.2-0.25, due to
    methylation-driven C->T mutation at CpG sites over evolutionary time).
  - Many viral genomes (esp. those replicating via host machinery under
    different pressures): higher AT content on average, weaker or absent
    CpG suppression, and different codon-position biases from dense
    coding sequence.

This produces a dataset that is NOT trivially separable by a single
scalar (like sequence length) but DOES contain the kind of compositional
"words" (k-mer) signal a k-mer/NLP pipeline is meant to detect --
which is the whole point of a rehearsal dataset like this.

==> TO USE REAL DATA: replace the two blocks in `build_dataset()` that
call `generate_class_sequences(...)` with code that reads real FASTA
files you've downloaded (e.g. from NCBI Virus / GRCh38) into a list of
sequence strings. Everything downstream (kmer_utils, represent, models)
is agnostic to where the strings came from.
"""

import random
import numpy as np
import pandas as pd
from pathlib import Path

random.seed(42)
np.random.seed(42)

BASES = ["A", "C", "G", "T"]

# --- Transition matrices: rows/cols ordered A, C, G, T -----------------
# Human-like: CpG suppressed (low prob of C->G), moderate GC (~41%)
HUMAN_TRANS = np.array([
    # to:  A      C      G      T
    [0.32, 0.18, 0.20, 0.30],  # from A
    [0.35, 0.20, 0.05, 0.40],  # from C  (C->G suppressed: CpG rarity)
    [0.28, 0.24, 0.22, 0.26],  # from G
    [0.22, 0.22, 0.24, 0.32],  # from T
])

# Viral-like: higher AT, no CpG suppression, more uniform-ish with a
# codon-position-esque skew (approximates dense coding sequence bias)
VIRAL_TRANS = np.array([
    # to:  A      C      G      T
    [0.34, 0.16, 0.16, 0.34],  # from A
    [0.30, 0.20, 0.20, 0.30],  # from C  (no CpG suppression)
    [0.30, 0.20, 0.20, 0.30],  # from G
    [0.34, 0.16, 0.16, 0.34],  # from T
])


def _sample_sequence(trans_matrix, length, start_probs=(0.3, 0.2, 0.2, 0.3)):
    """Sample one sequence from a first-order Markov chain over ACGT."""
    seq = [np.random.choice(BASES, p=start_probs)]
    idx = {b: i for i, b in enumerate(BASES)}
    for _ in range(length - 1):
        prev = idx[seq[-1]]
        probs = trans_matrix[prev]
        probs = probs / probs.sum()
        seq.append(np.random.choice(BASES, p=probs))
    return "".join(seq)


def generate_class_sequences(n, trans_matrix, length_range=(300, 500), n_variants=8):
    """
    Generate n sequences for one class. To mimic "many related genomes"
    rather than one repeated pattern, we jitter the transition matrix
    slightly per variant group (like different species/strains would
    each have their own dialect of the same broad statistics).
    """
    seqs = []
    variant_matrices = []
    for _ in range(n_variants):
        jitter = np.random.normal(0, 0.02, trans_matrix.shape)
        m = np.clip(trans_matrix + jitter, 0.01, None)
        m = m / m.sum(axis=1, keepdims=True)
        variant_matrices.append(m)

    for i in range(n):
        m = variant_matrices[i % n_variants]
        length = np.random.randint(length_range[0], length_range[1] + 1)
        seqs.append(_sample_sequence(m, length))
    return seqs


def build_dataset(n_per_class=600, out_path="data/raw/sequences.csv"):
    # ==> SWAP-IN POINT for real data: replace these two calls with
    # loaders that read real FASTA files instead of sampling.
    human_seqs = generate_class_sequences(n_per_class, HUMAN_TRANS, n_variants=10)
    viral_seqs = generate_class_sequences(n_per_class, VIRAL_TRANS, n_variants=25)

    rows = []
    for i, s in enumerate(human_seqs):
        rows.append({"seq_id": f"human_{i:04d}", "sequence": s,
                      "label": "human", "source_group": f"human_grp_{i % 10}"})
    for i, s in enumerate(viral_seqs):
        rows.append({"seq_id": f"viral_{i:04d}", "sequence": s,
                      "label": "viral", "source_group": f"viral_grp_{i % 25}"})

    df = pd.DataFrame(rows)
    df["length"] = df["sequence"].str.len()
    df["gc_content"] = df["sequence"].apply(
        lambda s: (s.count("G") + s.count("C")) / len(s)
    )
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} sequences to {out_path}")
    print(df.groupby("label")[["length", "gc_content"]].mean())
    return df


if __name__ == "__main__":
    build_dataset()
