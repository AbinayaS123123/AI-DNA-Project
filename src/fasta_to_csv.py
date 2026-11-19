"""Convert one labelled FASTA file to the pipeline CSV schema."""
import argparse
from pathlib import Path
from data_io import load_fasta, prepare_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fasta"); parser.add_argument("--label", required=True)
    parser.add_argument("--source-group", required=True, help="species, strain, chromosome, or other provenance group")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data, rejected = prepare_dataset(load_fasta(args.fasta, args.label, args.source_group))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); data.to_csv(args.output, index=False)
    print(f"Wrote {len(data)} validated records; rejected {rejected} records.")
if __name__ == "__main__": main()
