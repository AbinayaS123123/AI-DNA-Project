# Can AI Understand the Language of DNA/RNA?

### Human vs. Viral DNA Sequence Classification Using NLP and Machine Learning

## Project Overview

This project explores whether DNA sequences can be treated as a language and analysed using Natural Language Processing (NLP) techniques.

DNA consists of four nucleotide bases: Adenine (A), Thymine (T), Guanine (G), and Cytosine (C). By breaking DNA sequences into smaller overlapping subsequences called **k-mers**, we can represent DNA using language-inspired features and investigate whether machine learning models can distinguish human and viral sequences.

The project uses classical machine learning techniques and compares different sequence representations to understand which patterns are useful for classification.

## Datasets

The project maintains two separate datasets for experimentation.

### 1. Synthetic Dataset — Midsem Baseline

Location: `data/raw/`

The synthetic dataset was generated using a first-order Markov model parameterised using documented human and viral genome statistics.

It was used during the midsem stage to:
- Develop and test the DNA-to-k-mer processing pipeline.
- Experiment with different NLP representations.
- Implement and compare machine learning classifiers.
- Test the evaluation and visualisation workflow.

**Important:** Synthetic sequences are generated data. Their results demonstrate the behaviour of the pipeline and must not be interpreted as actual biological findings.

### 2. Real Dataset — Human vs. Viral DNA

Location: `data/real/`

The real dataset contains human DNA sequences and viral DNA sequences.

The combined dataset is:

`data/real/nextvir_sequences.csv`

Expected columns:

| Column | Description |
|---|---|
| `seq_id` | Unique sequence identifier |
| `sequence` | DNA sequence |
| `label` | Human or viral class |
| `source_group` | Biological source used for group-aware splitting |

The `source_group` column is important because sequences originating from the same chromosome, species, or viral accession should not be distributed across both training and testing partitions.

Real-data results are evaluated separately from the synthetic baseline.

## Methodology

### 1. K-mer Tokenisation

DNA sequences are divided into overlapping subsequences of different lengths.

For example, with a sequence `ATGCAT` and k = 3:

`ATG`, `TGC`, `GCA`, `CAT`

The project explores k-mer sizes of 3, 4, and 6.

### 2. Sequence Representations

Three representations are explored:

- **Bag-of-Words (BoW):** Represents sequences using k-mer occurrence counts.
- **TF-IDF:** Weights k-mers according to their importance across sequences.
- **Dense SVD Embeddings:** Produces lower-dimensional sequence representations using truncated Singular Value Decomposition.

### 3. Machine Learning Models

The following classifiers are compared:

- Logistic Regression
- Decision Tree
- Random Forest

Models are evaluated across the different sequence representations.

### 4. Data Splitting and Leakage Prevention

The pipeline uses group-separated, fixed-seed train/test splits.

Feature vocabularies, TF-IDF weights, and SVD transformations are fitted using training data only.

This helps prevent information leakage between training and testing data.

For real data, the biological `source_group` must be meaningful and sufficiently diverse in both classes. A group-aware split is not reliable if one class has too few independent source groups.

### 5. Evaluation

The pipeline calculates:

- Accuracy
- Weighted precision
- Weighted recall
- Weighted F1-score
- Confusion matrices
- Important k-mers
- Length and GC-content baseline comparisons

### 6. Visualisation and Interpretation

The project also explores:
- K-mer frequency distributions.
- PCA and t-SNE visualisations.
- Model performance comparisons.
- Important k-mer features.
- Potential shortcuts based on sequence length and GC content.

## Project Structure

```text
AI-DNA-Project/
│
├── data/
│   ├── raw/                  # Synthetic midsem dataset
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

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Pipeline

### Synthetic Baseline

To evaluate the existing synthetic dataset:

```bash
python run_pipeline.py --data data/raw/sequences.csv
```

To regenerate the synthetic dataset and run the pipeline:

```bash
python run_pipeline.py --regenerate-synthetic
```

### Real Dataset

To evaluate the real human-versus-viral dataset:

```bash
python run_pipeline.py --data data/real/nextvir_sequences.csv
```

Before running the real-data experiment, check the number of independent source groups in each class:

```python
import pandas as pd

df = pd.read_csv("data/real/nextvir_sequences.csv")

print(df.groupby("label")["source_group"].nunique())
```

Ensure that both classes have enough independent biological source groups for a meaningful group-separated evaluation.

### Running Tests

```bash
python -m unittest discover -s tests -v
```

## Outputs

The pipeline generates result tables and visualisations under:

- `outputs/results/`
- `outputs/figures/`

These include model comparisons, confusion matrices, feature-importance tables, artifact checks, and exploratory plots.

**Important:** Always identify whether an output was generated using synthetic or real data. Do not combine the two experiments when reporting performance.

## Project Objectives

| Objective | Implementation |
|---|---|
| Represent DNA as a language | K-mer tokenisation |
| Explore sequence patterns | Frequency analysis and visualisation |
| Apply NLP representations | BoW, TF-IDF and SVD |
| Build classification models | Logistic Regression, Decision Tree and Random Forest |
| Evaluate model performance | Accuracy, precision, recall, F1 and confusion matrices |
| Interpret learned patterns | Important k-mers and artifact checks |
| Compare synthetic and real experiments | Separate dataset evaluation and reporting |

## Limitations

- Synthetic sequences do not capture the full complexity of real biological genomes.
- Human and viral sequences may differ in length, composition, and biological structure.
- Group-aware evaluation depends on having sufficient independent source groups.
- High classification performance alone does not establish that a model has learned biologically meaningful motifs.
- Important k-mers must be interpreted cautiously and should not automatically be treated as biological discoveries.
- The SVD representation is a lightweight dimensionality-reduction baseline, not a biological foundation model.

## Final Goal

The project investigates whether NLP-inspired representations can capture useful patterns in DNA sequences and distinguish human from viral DNA.

By comparing synthetic and real datasets, different sequence representations, and multiple machine learning models, the project aims to understand both the capabilities and limitations of treating DNA as a language.
