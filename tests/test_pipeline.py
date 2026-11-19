import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from data_io import prepare_dataset
from kmer_utils import clean_sequence, seq_to_kmers, validate_sequence
from models import group_train_test_split
from represent import fit_transform_text


class SequenceTests(unittest.TestCase):
    def test_rna_and_ambiguous_bases_are_safe(self):
        self.assertEqual(clean_sequence("augr-c"), "ATGNNC")
        self.assertEqual(seq_to_kmers("ACNGT", 2), ["AC", "GT"])
        self.assertFalse(validate_sequence("ACNG", 2)[0])

    def test_kmer_input_validation(self):
        with self.assertRaises(ValueError): seq_to_kmers("ACGT", 0)


class LeakageTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"seq_id": list("abcdefgh"), "sequence": ["AAAACCCC", "AAAAGCCC", "AAAATCCC", "AAAACCCA", "GGGGTTTT", "GGGGATTT", "GGGGCTTT", "GGGGTTTA"], "label": ["human"] * 4 + ["viral"] * 4, "source_group": ["h1", "h1", "h2", "h2", "v1", "v1", "v2", "v2"]})

    def test_groups_never_cross_split(self):
        train, test = group_train_test_split(self.df, test_size=.5)
        self.assertTrue(set(self.df.iloc[train].source_group).isdisjoint(self.df.iloc[test].source_group))

    def test_vectorizer_does_not_learn_test_only_kmer(self):
        train, test = self.df.iloc[:4], self.df.iloc[4:]
        _, _, vectorizer = fit_transform_text(train, test, 4, "tfidf")
        self.assertNotIn("GGGG", vectorizer.vocabulary_)

    def test_prepare_legacy_data_adds_required_columns(self):
        data, rejected = prepare_dataset(pd.DataFrame({"sequence": ["ACGT", "TGCA"], "label": [0, 1]}))
        self.assertEqual(rejected, 0); self.assertTrue({"seq_id", "source_group", "length", "gc_content"}.issubset(data.columns))

if __name__ == "__main__": unittest.main()
