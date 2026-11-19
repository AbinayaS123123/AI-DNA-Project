"""Dataset validation and small FASTA loading helpers."""
from pathlib import Path
import pandas as pd
from kmer_utils import clean_sequence, validate_sequence

def load_fasta(path, label, source_group=None):
    records, identifier, chunks = [], None, []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line: continue
        if line.startswith(">"):
            if identifier is not None: records.append((identifier, "".join(chunks)))
            identifier, chunks = line[1:].split()[0] or f"record_{len(records)}", []
        else: chunks.append(line)
    if identifier is not None: records.append((identifier, "".join(chunks)))
    if not records: raise ValueError(f"No FASTA records found in {path}")
    return pd.DataFrame({"seq_id": [r[0] for r in records], "sequence": [clean_sequence(r[1]) for r in records], "label": label, "source_group": source_group or Path(path).stem})

def prepare_dataset(df, min_length=4):
    if not {"sequence", "label"}.issubset(df.columns): raise ValueError("dataset requires sequence and label columns")
    data = df.copy(); data["sequence"] = data.sequence.map(clean_sequence)
    valid = data.sequence.map(lambda s: validate_sequence(s, min_length)[0]); rejected = int((~valid).sum()); data = data.loc[valid].copy()
    if data.empty or data.label.nunique() < 2: raise ValueError("need valid sequences from at least two labels")
    if "seq_id" not in data: data["seq_id"] = [f"sequence_{i}" for i in data.index]
    if "source_group" not in data: data["source_group"] = data.seq_id.astype(str)  # Legacy data: record-level separation only.
    data["length"] = data.sequence.str.len(); data["gc_content"] = data.sequence.map(lambda s: (s.count("G") + s.count("C")) / len(s))
    return data.reset_index(drop=True), rejected
