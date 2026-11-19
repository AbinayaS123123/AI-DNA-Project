"""Run the lightweight, reproducible DNA/RNA classification baseline."""
import argparse, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
sys.path.insert(0, str(Path(__file__).parent / "src"))
from data_io import prepare_dataset
from generate_synthetic_data import build_dataset
from interpret import artifact_check, top_logreg_features
from models import run_all_representations

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--data", default="data/raw/sequences.csv"); parser.add_argument("--k", type=int, default=4); parser.add_argument("--regenerate-synthetic", action="store_true")
    args = parser.parse_args()
    if args.regenerate_synthetic: build_dataset(out_path=args.data)
    raw = pd.read_csv(args.data); df, rejected = prepare_dataset(raw, min_length=args.k)
    results, trained, artifacts = run_all_representations(df, k=args.k); results = results.sort_values("f1_weighted", ascending=False)
    result_dir, figure_dir = Path("outputs/results"), Path("outputs/figures"); result_dir.mkdir(parents=True, exist_ok=True); figure_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(result_dir / "model_comparison.csv", index=False); labels = sorted(df.label.unique())
    for key, matrix in artifacts.items():
        if key.endswith("_cm"): pd.DataFrame(matrix, index=labels, columns=labels).to_csv(result_dir / f"confusion_matrix_{key[:-3]}.csv")
    top_logreg_features(trained["tfidf_logreg"], artifacts["tfidf"]["vectorizer"], labels).to_csv(result_dir / "top_kmers_logreg.csv", index=False)
    best = results.iloc[0]; check = artifact_check(df, best.accuracy, artifacts["train_idx"], artifacts["test_idx"]); pd.DataFrame([check]).to_csv(result_dir / "artifact_check.csv", index=False)
    cm = artifacts[f"{best.representation}_{best.model}_cm"]; fig, ax = plt.subplots(figsize=(4, 3)); image = ax.imshow(cm, cmap="Blues")
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="True", title=f"Best: {best.representation}/{best.model}")
    for i in range(len(labels)):
        for j in range(len(labels)): ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image); fig.tight_layout(); fig.savefig(figure_dir / "best_confusion_matrix.png", dpi=150)
    print(f"Processed {len(df)} sequences ({rejected} rejected). {'Synthetic baseline' if args.regenerate_synthetic else 'Supplied dataset'} results:"); print(results.to_string(index=False))
if __name__ == "__main__": main()
