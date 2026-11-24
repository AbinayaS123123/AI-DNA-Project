# Can AI Understand the Language of DNA/RNA?

### Human vs. Viral DNA Sequence Classification Using NLP and Machine Learning

## Project Overview

This project explores whether DNA sequences can be treated as a language and analysed using Natural Language Processing (NLP) techniques.

DNA consists of four nucleotide bases: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C). By breaking DNA sequences into smaller overlapping subsequences called **k-mers**, we can represent DNA using language-inspired features and investigate whether machine learning models can distinguish human and viral sequences.

The project applies classical machine learning techniques to compare different sequence representations, identify patterns, and evaluate their usefulness for DNA sequence classification.

## Dataset

The project primarily uses a real DNA sequence dataset containing human and viral sequences.

**Location:** `data/real/`

The combined dataset, `nextvir_sequences.csv`, contains the following columns:

| Column | Description |
|---|---|
| `seq_id` | Unique sequence identifier |
| `sequence` | DNA sequence |
| `label` | Human or viral class |
| `source_group` | Biological source used for group-aware splitting |

The `source_group` column is important because sequences originating from the same chromosome, species, or viral accession should not be distributed across both training and testing partitions.

A small synthetic dataset is also retained in `data/raw/` for preliminary pipeline testing and baseline experimentation. It is separate from the real-data experiments and is not used to draw biological conclusions.

## Methodology

### 1. K-mer Tokenisation

DNA sequences are divided into overlapping subsequences of different lengths.

For example, given the sequence `ATGCAT` and k = 3:

`ATG`, `TGC`, `GCA`, `CAT`

The project explores k-mer sizes of 3, 4, and 6.

### 2. Sequence Representations

Three representations are explored:

- **Bag-of-Words (BoW):** Represents sequences using k-mer occurrence counts.
- **TF-IDF:** Weights k-mers according to their importance across sequences.
- **Dense SVD Embeddings:** Produces lower-dimensional sequence representations using truncated Singular Value Decomposition.

### 3. Machine Learning Models

The following classifiers are compared across the different sequence representations:

- Logistic Regression
- Decision Tree
- Random Forest

### 4. Data Splitting and Leakage Prevention

The pipeline uses group-separated, fixed-seed train/test splits.

K-mer vocabularies, TF-IDF weights, and SVD transformations are fitted using training data only.

This helps prevent information leakage between training and testing partitions.

For real data, the biological `source_group` must be meaningful and sufficiently diverse in both classes. A group-aware split is not reliable if one class has too few independent source groups.

### 5. Model Evaluation

The pipeline evaluates model performance using:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrices

It also examines feature importance and compares the full k-mer model against a baseline using only sequence length and GC content.

### 6. Visualisation and Interpretation

The project explores:

- K-mer frequency distributions.
- PCA and t-SNE visualisations.
- Model performance comparisons.
- Important k-mer features.
- Potential classification shortcuts based on sequence length and GC content.

## Project Structure

```text
AI-DNA-Project/
│
├── data/
│   ├── raw/                  # Synthetic baseline data
│   └── real/                 # Real human and viral DNA data
│       └── nextvir_sequences.csv
│
├── notebooks/                # Experimentation notebooks
│
├── src/
│   ├── generate_synthetic_data.py
│   ├── fasta_to_csv.py
│   ├── kmer_utils.py
│   ├── explore.py
│   ├── represent.py
│   ├── models.py
│   └── interpret.py
│
├── tests/                    # Unit tests
│
├── outputs/
│   ├── figures/              # Generated visualisations
│   └── results/              # Evaluation tables and metrics
│
├── run_pipeline.py
├── requirements.txt
└── README.md
```

## Installation

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Pipeline

### Real Dataset — Human vs. Viral DNA

To run the pipeline on the real dataset:

```bash
python run_pipeline.py --data data/real/nextvir_sequences.csv
```

Before running the experiment, check the number of independent source groups in each class:

```python
import pandas as pd

df = pd.read_csv("data/real/nextvir_sequences.csv")

print(df.groupby("label")["source_group"].nunique())
```

Ensure that both classes have enough independent biological source groups for a meaningful group-separated evaluation.

### Synthetic Baseline

The synthetic dataset can be used for preliminary testing:

```bash
python run_pipeline.py --data data/raw/sequences.csv
```

To regenerate it:

```bash
python run_pipeline.py --regenerate-synthetic
```

### Running Tests

```bash
python -m unittest discover -s tests -v
```

## Outputs

The pipeline generates result tables and visualisations under:

- `outputs/results/`
- `outputs/figures/`

These include:

- Model comparison tables.
- Confusion matrices.
- Feature-importance k-mer tables.
- Length and GC-content baseline comparisons.
- Exploratory visualisations.

Results should always be associated with the dataset used to generate them. Synthetic and real-data experiments must be reported separately.

## Project Objectives

| Objective | Implementation |
|---|---|
| Represent DNA as a language | K-mer tokenisation |
| Explore sequence patterns | Frequency analysis and visualisation |
| Apply NLP representations | BoW, TF-IDF and SVD |
| Build classification models | Logistic Regression, Decision Tree and Random Forest |
| Evaluate model performance | Accuracy, precision, recall, F1 and confusion matrices |
| Interpret learned patterns | Important k-mers and artifact checks |
| Investigate human vs. viral DNA | Classification using real sequence data |

## Limitations

- Human and viral sequences may differ in length, composition, and biological structure.
- Group-aware evaluation depends on having sufficient independent source groups.
- High classification performance alone does not establish that a model has learned biologically meaningful motifs.
- Important k-mers must be interpreted cautiously and should not automatically be treated as biological discoveries.
- The SVD representation is a lightweight dimensionality-reduction baseline, not a biological foundation model.

## Conclusion

This project investigates whether NLP-inspired representations can capture useful patterns in DNA sequences and distinguish human from viral DNA.

By combining k-mer tokenisation, multiple sequence representations, classical machine learning models, and feature interpretation, the project explores the potential of language-inspired approaches for biological sequence analysis.
