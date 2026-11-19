"""Safe sequence cleaning and k-mer utilities shared by the pipeline."""
from collections import Counter

DNA_BASES = frozenset("ACGT")


def clean_sequence(seq: str) -> str:
    """Normalize DNA/RNA without silently joining across ambiguous bases."""
    if not isinstance(seq, str):
        raise TypeError("sequence must be a string")
    return "".join(base if base in DNA_BASES else "N"
                   for base in seq.upper().replace("U", "T").replace(" ", ""))


def validate_sequence(seq: str, min_length: int = 1) -> tuple[bool, str]:
    cleaned = clean_sequence(seq)
    if len(cleaned) < min_length:
        return False, f"sequence is shorter than {min_length} bases"
    if "N" in cleaned:
        return False, "sequence contains ambiguous or invalid bases"
    return True, ""


def seq_to_kmers(seq: str, k: int) -> list[str]:
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    seq = clean_sequence(seq)
    return [window for i in range(max(0, len(seq) - k + 1))
            if set(window := seq[i:i + k]) <= DNA_BASES]


def seq_to_kmer_string(seq: str, k: int) -> str:
    return " ".join(seq_to_kmers(seq, k))


def kmer_frequency(seq: str, k: int) -> Counter:
    return Counter(seq_to_kmers(seq, k))


def vocabulary_size(k: int) -> int:
    if not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")
    return 4 ** k


def dataset_vocab_stats(sequences: list[str], k: int) -> dict:
    counts = Counter()
    for sequence in sequences:
        counts.update(seq_to_kmers(sequence, k))
    theoretical = vocabulary_size(k)
    return {"k": k, "observed_vocab": len(counts), "theoretical_vocab": theoretical,
            "coverage_pct": round(100 * len(counts) / theoretical, 2),
            "top_10": counts.most_common(10)}
