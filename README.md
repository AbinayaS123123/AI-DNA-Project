# Can AI Understand the Language of DNA/RNA?
### Human vs. Viral sequence classification — working pipeline

This is a full, runnable pipeline for the course project: treating DNA as a
four-letter language, tokenizing it into k-mer "words," applying NLP-style
representations, training classifiers, and checking whether the model
learned something real or just a shortcut.

## ⚠️ Important: this ships with SYNTHETIC data, not real NCBI sequences

The environment this was built in cannot reach NCBI or any genomic database
(only software package registries are network-accessible). So
`src/generate_synthetic_data.py` generates stand-in sequences using a
first-order Markov chain parameterized with **real, documented** human vs.
viral genome statistics:

- Human: ~41% GC content, strong **CpG suppression** (a real, well-known
  feature of vertebrate genomes caused by methylation-driven mutation at
  CpG sites)
- Viral: higher AT content, no CpG suppression, different local biases

This is good enough to prove every stage of the pipeline works and produces
sensible results (see `outputs/results/` and `outputs/figures/`), but it is
**not real biological data** and should not be used for actual conclusions
about DNA/RNA sequence patterns. Swap in real data before you write up
results — see below.

## How to swap in real data

1. Download real sequences yourself (this typically works outside this
   sandbox):
   - Human: [NCBI GRCh38 reference genome](https://www.ncbi.nlm.nih.gov/data-hub/genome/GCF_000001405.40/) or [UCSC Genome Browser](https://genome.ucsc.edu/) — chop into random windows.
   - Viral: [NCBI Virus](https://www.ncbi.nlm.nih.gov/labs/virus/vssi/) — download complete genomes across many viral families.
2. Write a loader that produces a DataFrame with columns:
   `seq_id, sequence, label, source_group` (source_group = species/strain/
   chromosome, used for leakage-safe splitting — see `models.py`).
3. Replace the two `generate_class_sequences(...)` calls in
   `build_dataset()` inside `src/generate_synthetic_data.py` with your
   loader, or just write a new script that outputs the same CSV shape to
   `data/raw/sequences.csv`. Nothing else in the pipeline needs to change.

Tip: keep windows a consistent length across classes, and split train/test
by `source_group` (already implemented in `models.py`) so the model can't
just memorize near-duplicate windows from the same genome.

## Project structure

```
dna_ai_project/
├── run_pipeline.py           # runs everything end-to-end, start here
├── src/
│   ├── generate_synthetic_data.py   # <-- swap-in point for real data
│   ├── kmer_utils.py                # DNA -> k-mer "words"
│   ├── explore.py                   # stats, PCA/t-SNE clustering plots
│   ├── represent.py                 # BoW, TF-IDF, word2vec-style embeddings
│   ├── models.py                    # train/compare classifiers (group-safe split)
│   └── interpret.py                 # feature importance + artifact check
├── data/raw/sequences.csv    # generated dataset (regenerate anytime)
└── outputs/
    ├── figures/               # PNG plots
    └── results/               # CSV tables
```

## How to run

```bash
pip install scikit-learn pandas numpy matplotlib gensim scipy
python run_pipeline.py
```

Each module also runs standalone (`python src/explore.py`, etc.) once
`data/raw/sequences.csv` exists.

## What each project task maps to

| Proposal task | Where it happens |
|---|---|
| 1. Represent DNA as language (k-mers) | `kmer_utils.py` |
| 2. Explore the sequence language | `explore.py` (vocab stats, PCA/t-SNE) |
| 3. NLP representations (BoW/TF-IDF/word2vec) | `represent.py` |
| 4. Build AI/ML models | `models.py` (logreg, decision tree, random forest × 3 representations) |
| 5. What did the AI learn? | `interpret.py` (top features + artifact check) |
| 6. AI-generated explanations | not automated — feed `outputs/results/top_kmers_frequency_ratio.csv` to an LLM yourself and compare/verify (see project write-up) |
| 7. Final challenge | `outputs/results/model_comparison.csv` + your own write-up |

## Current results (on synthetic data — re-run after swapping in real data)

- Best models hit ~99-100% accuracy on held-out sequence groups using
  4-mer TF-IDF / Bag-of-Words + logistic regression or random forest.
- **Artifact check**: a trivial classifier using ONLY sequence length +
  GC content gets ~53% accuracy (near chance), while the k-mer model gets
  ~100% — meaning the model is using real k-mer pattern information, not
  just exploiting length or overall composition. This is exactly the kind
  of check task 5 in the proposal asks for, and it's worth re-running on
  real data since real biological sequences are messier than this
  synthetic set.
- Top human-associated 4-mers include `TATA`-like AT-rich motifs; top
  viral-associated 4-mers are more GC/pyrimidine-rich (`CTCT`, `TCTC`,
  `CTGC`) — plausible given the synthetic generative model, but **these
  specific results are an artifact of the synthetic data and will change
  completely with real sequences.**

## Known limitations / things to fix with real data

- Synthetic sequences use only a first-order Markov model — real genomes
  have much richer higher-order structure (codons, motifs, repeats).
- No true "coding vs. non-coding" distinction is modeled — flagged in the
  main conversation as a likely real confound for human vs. viral data;
  check for it once you have real sequences.
- word2vec was trained on this dataset only; with real data consider a
  larger training corpus for k-mer embeddings.
